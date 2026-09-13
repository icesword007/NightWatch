import hashlib
import json
import time
from dataclasses import dataclass
from typing import Callable

from .actions import ActionProposal, PlannedAction
from .grid import next_step
from .protocol import (
    PIONEER,
    PlayerTask,
    Pos,
    Turn,
    Unit,
    accept_task_command,
    distance,
    move_command,
    submit_answer_command,
)
from .state import SessionState, TaskMemory

MAX_LLM_RESPONSE_CHARS = 20_000
MAX_COMMAND_CHARS = 4_096
MAX_ANSWER_CHARS = 16_384
MAX_TASK_PROMPT_CHARS = 32_768
MAX_TOOL_CONTEXT_CHARS = 8_192
MAX_SOLVER_EVENT_CHARS = 4_096
MAX_SOLVER_EVENTS = 8
MAX_SOLVER_HISTORY_CHARS = 16_384


@dataclass(frozen=True, slots=True)
class LlmEnvelope:
    kind: str
    content: str
    complete: bool | None = None


@dataclass(frozen=True, slots=True)
class TaskTurnProposal:
    actions: tuple[PlannedAction, ...] = ()
    prompt: str = ""
    execute_cmd: str = ""


def parse_llm_envelope(raw: str) -> LlmEnvelope | None:
    if not isinstance(raw, str) or not raw or len(raw) > MAX_LLM_RESPONSE_CHARS:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    kind = value.get("kind")
    content = value.get("content")
    if not isinstance(content, str) or not content:
        return None
    if kind == "command":
        if set(value) != {"kind", "content"} or len(content) > MAX_COMMAND_CHARS:
            return None
        return LlmEnvelope(kind, content)
    if kind == "answer":
        complete = value.get("complete")
        if (
            set(value) != {"kind", "content", "complete"}
            or not isinstance(complete, bool)
            or len(content) > MAX_ANSWER_CHARS
        ):
            return None
        return LlmEnvelope(kind, content, complete)
    return None


def command_result_complete(result: str) -> bool:
    if not result or len(result) > 65_536:
        return False
    if "[TRUNCATED]" in result:
        return False
    first_line = result.splitlines()[0] if result.splitlines() else ""
    return first_line == "[exitCode:0]"


def propose_tasks(
    turn: Turn,
    state: SessionState,
    *,
    clock: Callable[[], float] = time.monotonic,
    deadline: float,
    max_expansions: int,
) -> TaskTurnProposal:
    task = state.active_task
    if task is not None and turn.phase_task:
        return _continue_active_task(turn, state, task)
    if turn.phase_task:
        return TaskTurnProposal()
    pioneers = turn.pioneers()
    if not pioneers:
        return TaskTurnProposal()
    pioneer = pioneers[0]
    selected = _select_task(
        turn, pioneer, clock, deadline, max_expansions,
    )
    if selected is None:
        return TaskTurnProposal()
    task_spec, _, step = selected
    reason = f"task:{task_spec.task_type}:{task_spec.pos.x}:{task_spec.pos.y}"
    if step is None:
        return TaskTurnProposal(actions=(PlannedAction(
            ActionProposal(
                pioneer.unit_id,
                pioneer.unit_id,
                accept_task_command(),
            ),
            task_spec.pos,
            reason,
        ),))
    return TaskTurnProposal(actions=(PlannedAction(
        ActionProposal(
            pioneer.unit_id,
            pioneer.unit_id,
            move_command(step),
            destination=step,
        ),
        task_spec.pos,
        reason,
    ),))


def _continue_active_task(
    turn: Turn,
    state: SessionState,
    task: TaskMemory,
) -> TaskTurnProposal:
    owner = turn.unit(task.owner_id) if task.owner_id is not None else None
    if owner is None or owner.kind != PIONEER:
        return TaskTurnProposal()
    if task.solver_stopped_reason is not None:
        return TaskTurnProposal()
    if task.pending_llm_round is not None or task.pending_cmd_round is not None:
        return TaskTurnProposal()
    if task.deferred_answer is not None:
        return TaskTurnProposal(actions=(PlannedAction(ActionProposal(
            owner.unit_id,
            owner.unit_id,
            submit_answer_command(task.deferred_answer),
        )),))

    if task.consumed_tool_results < len(task.tool_results):
        kind, result = task.tool_results[task.consumed_tool_results]
        task.consumed_tool_results += 1
        result_fingerprint = hashlib.sha256(
            f"{kind}\0{result}".encode("utf-8")
        ).hexdigest()
        if result_fingerprint == task.last_tool_result_fingerprint:
            task.repeated_tool_result_count += 1
        else:
            task.last_tool_result_fingerprint = result_fingerprint
            task.repeated_tool_result_count = 1
        if task.repeated_tool_result_count > 2:
            task.solver_stopped_reason = "repeated_identical_tool_result"
            return TaskTurnProposal()
        if kind == "llm":
            envelope = parse_llm_envelope(result)
            if envelope is None:
                _remember(task, "Rejected LLM response", result)
                return TaskTurnProposal(prompt=_solver_prompt(
                    turn, task,
                    "The previous LLM response violated the JSON envelope.",
                ))
            if envelope.kind == "command":
                _remember(task, "Platform command requested", envelope.content)
                return TaskTurnProposal(execute_cmd=envelope.content)
            if envelope.content == task.last_submitted_answer:
                return TaskTurnProposal(prompt=_solver_prompt(
                    turn, task,
                    "Do not repeat the unchanged submitted answer; improve it.",
                ))
            _remember(task, "Answer proposed for submission", envelope.content)
            task.deferred_answer = envelope.content
            return TaskTurnProposal(actions=(PlannedAction(ActionProposal(
                owner.unit_id,
                owner.unit_id,
                submit_answer_command(envelope.content),
            )),))
        status = (
            "The platform sandbox command completed successfully."
            if command_result_complete(result)
            else "The platform sandbox result was empty, failed, timed out, or truncated."
        )
        _remember(task, "Platform command result", result)
        return TaskTurnProposal(prompt=_solver_prompt(
            turn, task,
            f"{status}\nPlatform result:\n{_bounded(result, MAX_TOOL_CONTEXT_CHARS)}",
        ))

    previous_submit = next((
        completed
        for completed in reversed(state.action_history)
        if completed.pending.action == "submitAnswer"
        and completed.pending.task_instance_id == task.instance_id
    ), None)
    if previous_submit is not None:
        error_text = "; ".join(
            error.description or f"errorCode={error.code}"
            for error in turn.errors
            if error.code == 2
        )
        context = "The previous answer was not confirmed complete."
        if previous_submit.success is False or error_text:
            context = f"The previous submission failed or was incomplete. {error_text}"
        if task.last_submission_feedback_round != previous_submit.pending.round_no:
            _remember(task, "Submission feedback", context)
            task.last_submission_feedback_round = previous_submit.pending.round_no
        return TaskTurnProposal(prompt=_solver_prompt(turn, task, context))

    urgency = ""
    if task.timeout_round is not None and task.timeout_round - turn.round_no <= 1:
        urgency = "The task deadline is imminent; return the best submit-ready answer now."
    return TaskTurnProposal(prompt=_solver_prompt(turn, task, urgency))


def _solver_prompt(turn: Turn, task: TaskMemory, context: str) -> str:
    task_text = _bounded(turn.phase_task, MAX_TASK_PROMPT_CHARS)
    context_text = _bounded(context, MAX_TOOL_CONTEXT_CHARS)
    history_text = _solver_history_text(task)
    return (
        "Solve the following competition task using only the stated task and "
        "platform sandbox evidence. Return exactly one JSON object and no markdown. "
        'Use {"kind":"command","content":"<sandbox command>"} when another '
        "platform sandbox step is necessary. Use "
        '{"kind":"answer","content":"<submit-ready answer>","complete":true} '
        "for a complete answer, or complete:false for a reliable partial answer. "
        "Never claim success from an empty, failed, timed-out, or truncated result.\n"
        f"{context_text}\nSolver history:\n{history_text}\nTask:\n{task_text}"
    )


def _remember(task: TaskMemory, label: str, content: str) -> None:
    task.solver_history.append(_bounded(
        f"{label}:\n{content}", MAX_SOLVER_EVENT_CHARS,
    ))
    if len(task.solver_history) > MAX_SOLVER_EVENTS:
        del task.solver_history[:-MAX_SOLVER_EVENTS]
        task.solver_history_truncated = True


def _solver_history_text(task: TaskMemory) -> str:
    joined = "\n\n".join(task.solver_history)
    if (
        not task.solver_history_truncated
        and len(joined) <= MAX_SOLVER_HISTORY_CHARS
    ):
        return joined or "(none)"
    prefix = "[TRUNCATED OLDER HISTORY]\n"
    budget = MAX_SOLVER_HISTORY_CHARS - len(prefix)
    selected: list[str] = []
    used = 0
    for event in reversed(task.solver_history):
        separator = 2 if selected else 0
        if used + separator + len(event) > budget:
            break
        selected.append(event)
        used += separator + len(event)
    selected.reverse()
    body = "\n\n".join(selected)
    return f"{prefix}{body}"


def _select_task(
    turn: Turn,
    pioneer: Unit,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[PlayerTask, Pos, Pos | None] | None:
    options = []
    for task in turn.player_tasks:
        if not task.is_valid or task.cooldown_rounds != 0:
            continue
        for cell in turn.task_cells(task):
            for stand in _adjacent_stands(turn, pioneer, cell):
                path = next_step(
                    turn,
                    pioneer,
                    stand,
                    clock=clock,
                    deadline=deadline,
                    max_expansions=max_expansions,
                )
                if path.status == "already_there":
                    options.append((0, task, cell, None))
                elif path.status == "found" and path.step is not None:
                    options.append((path.cost or 0, task, cell, path.step))
                if clock() >= deadline:
                    break
            if clock() >= deadline:
                break
        if clock() >= deadline:
            break
    if not options:
        return None
    _, task, cell, step = min(options, key=lambda option: (
        option[0],
        -(option[1].score_reward + option[1].gold_reward),
        -(option[1].timeout_rounds or 0),
        option[1].pos.x,
        option[1].pos.y,
    ))
    return task, cell, step


def _adjacent_stands(turn: Turn, pioneer: Unit, target: Pos) -> tuple[Pos, ...]:
    blocked = turn.blocked(pioneer)
    return tuple(sorted(
        (
            Pos(target.x + dx, target.y + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx or dy)
            and turn.land(Pos(target.x + dx, target.y + dy))
            and Pos(target.x + dx, target.y + dy) not in blocked
        ),
        key=lambda pos: (distance(pioneer.pos, pos), pos.x, pos.y),
    ))


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n[TRUNCATED]"
    if limit <= len(marker):
        return marker[-limit:]
    return f"{value[:limit - len(marker)]}{marker}"
