import hashlib
import json
import re
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
MAX_SOLVER_EVIDENCE_EVENTS = 3
MAX_SOLVER_EVIDENCE_CHARS = 12_288
MAX_UNKNOWN_TASK_COMMANDS = 4
MAX_ENVIRONMENT_PATHS = 8
MAX_ENVIRONMENT_PATH_CHARS = 512
_STANDALONE_PATH = re.compile(r"/[^\s\"'<>|：]{1,511}")
_PATH_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_.-])(/[^\s\"'<>|：]{1,511})"
)


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
    start_skip_reason: str | None = None


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
    if kind == "abandon":
        reason = value.get("reason")
        if (
            set(value) != {"kind", "reason"}
            or not isinstance(reason, str)
            or not reason
            or len(reason) > MAX_COMMAND_CHARS
        ):
            return None
        return LlmEnvelope(kind, reason)
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
    start_guard: Callable[[PlayerTask, Pos, int], str | None] | None = None,
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
    selected, skip_reason = _select_task(
        turn, pioneer, clock, deadline, max_expansions, start_guard,
    )
    if selected is None:
        return TaskTurnProposal(start_skip_reason=skip_reason)
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
    if task.pending_llm_round is not None or task.pending_cmd_round is not None:
        return TaskTurnProposal()
    if task.deferred_answer is not None:
        return TaskTurnProposal(actions=(PlannedAction(ActionProposal(
            owner.unit_id,
            owner.unit_id,
            submit_answer_command(task.deferred_answer),
        )),))
    if task.solver_stopped_reason is not None:
        return _leave_task(turn, task, owner)

    remaining = _remaining_rounds(turn, task)

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
            return _leave_task(turn, task, owner)
        if kind == "llm":
            envelope = parse_llm_envelope(result)
            if envelope is None:
                _remember(task, "Rejected LLM response", result)
                if remaining == 0:
                    task.solver_stopped_reason = "deadline_without_answer"
                    return _leave_task(turn, task, owner)
                return TaskTurnProposal(prompt=_solver_prompt(
                    turn, task,
                    "The previous LLM response violated the JSON envelope.",
                ))
            if envelope.kind == "command":
                if _final_answer_required(task) or (
                    remaining is not None and remaining < 2
                ):
                    task.solver_stopped_reason = "command_after_final_request"
                    return _leave_task(turn, task, owner)
                _remember(task, "Platform command requested", envelope.content)
                task.last_command = envelope.content
                task.command_count += 1
                return TaskTurnProposal(execute_cmd=envelope.content)
            if envelope.kind == "abandon":
                _remember(task, "Solver abandoned task", envelope.content)
                task.solver_stopped_reason = "solver_abandoned"
                return _leave_task(turn, task, owner)
            if envelope.content == task.last_submitted_answer:
                if remaining == 0:
                    task.solver_stopped_reason = "deadline_repeated_answer"
                    return _leave_task(turn, task, owner)
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
        if command_result_complete(result):
            _remember_evidence(task, "Verified sandbox result", result)
            _remember_environment_paths(state, task, result)
        if remaining == 0:
            task.solver_stopped_reason = "command_result_at_deadline"
            return _leave_task(turn, task, owner)
        cycle_fingerprint = hashlib.sha256(
            f"{task.last_command or ''}\0{result}".encode("utf-8")
        ).hexdigest()
        if cycle_fingerprint == task.last_cycle_fingerprint:
            task.repeated_cycle_count += 1
        else:
            task.last_cycle_fingerprint = cycle_fingerprint
            task.repeated_cycle_count = 1
        if task.repeated_cycle_count >= 3:
            if remaining == 0:
                task.solver_stopped_reason = "repeated_cycle_at_deadline"
                return _leave_task(turn, task, owner)
            task.final_answer_requested = True
            return TaskTurnProposal(prompt=_solver_prompt(
                turn,
                task,
                "No more command exploration. Return an evidence-based complete "
                "or partial answer now, or abandon the task.",
            ))
        if (
            _effective_deadline(task) is None
            and task.command_count >= MAX_UNKNOWN_TASK_COMMANDS
        ):
            task.final_answer_requested = True
            return TaskTurnProposal(prompt=_solver_prompt(
                turn,
                task,
                "No more command exploration. Return an evidence-based complete "
                "or partial answer now, or abandon the task.",
            ))
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
        if remaining == 0:
            task.solver_stopped_reason = "deadline_after_failed_submission"
            return _leave_task(turn, task, owner)
        return TaskTurnProposal(prompt=_solver_prompt(turn, task, context))

    if remaining == 0:
        task.solver_stopped_reason = "deadline_without_answer"
        return _leave_task(turn, task, owner)
    urgency = ""
    if remaining is not None and remaining <= 1:
        task.final_answer_requested = True
        urgency = (
            "The task deadline is imminent; return the best evidence-based complete "
            "or partial submit-ready answer now, or abandon the task."
        )
    return TaskTurnProposal(prompt=_solver_prompt(turn, task, urgency))


def _solver_prompt(turn: Turn, task: TaskMemory, context: str) -> str:
    task_text = _bounded(turn.phase_task, MAX_TASK_PROMPT_CHARS)
    context_text = _bounded(context, MAX_TOOL_CONTEXT_CHARS)
    history_text = _solver_history_text(task)
    evidence_text = _solver_evidence_text(task)
    environment_text = _environment_path_text(task, task_text)
    remaining = _remaining_rounds(turn, task)
    if remaining is not None and remaining <= 3:
        if (
            task.coordination_deadline_round == _effective_deadline(task)
            and (
                task.timeout_round is None
                or task.coordination_deadline_round < task.timeout_round
            )
        ):
            task.coordination_final_requested = True
        else:
            task.final_answer_requested = True
    effective_deadline = _effective_deadline(task)
    if remaining is None:
        budget_text = "Known remaining task rounds: unknown."
    else:
        budget_text = (
            f"Known remaining task rounds: {remaining} "
            f"(current round {turn.round_no}, deadline round {effective_deadline})."
        )
    command_allowed = (
        not _final_answer_required(task)
        and (remaining is None or remaining >= 3)
    )
    command_text = (
        "Command exploration is allowed."
        if command_allowed
        else "Command exploration is not allowed; answer from existing evidence or abandon."
    )
    return (
        "Solve the following competition task using only the stated task and "
        "platform sandbox evidence. Return exactly one JSON object and no markdown. "
        f"{budget_text} {command_text} "
        'Use {"kind":"command","content":"<sandbox command>"} only when another '
        "platform sandbox step is allowed and necessary. Use "
        '{"kind":"answer","content":"<submit-ready answer>","complete":true} '
        "for a complete answer, or complete:false for a reliable partial answer. "
        'Use {"kind":"abandon","reason":"<why evidence is insufficient>"} '
        "instead of fabricating an answer. "
        "If the task gives an explicit file path, inspect that exact path directly. "
        "If it gives only a filename, use a bounded filename search. "
        "One command request consumes two game-round transitions before its result "
        "can inform the next answer. When safe, combine bounded discovery and the "
        "necessary read in one command rather than issuing blind cat, ls, and find "
        "steps separately. "
        "If it names no file, use a single input only when exactly one task-relevant input "
        "is evident. Do not assume the entire sandbox contains only one file. "
        "Handle line endings only when sandbox evidence specifically proves an "
        "interpreter or file-format problem. "
        "Never claim success from an empty, failed, timed-out, or truncated result.\n"
        f"{context_text}\nObserved environment path clues from successful "
        "sandbox output; re-check for this task:\n"
        f"{environment_text}\nCritical verified evidence:\n{evidence_text}\n"
        f"Solver history:\n{history_text}\nTask:\n{task_text}"
    )


def _remaining_rounds(turn: Turn, task: TaskMemory) -> int | None:
    deadline = _effective_deadline(task)
    if deadline is None:
        return None
    return max(0, deadline - turn.round_no)


def _effective_deadline(task: TaskMemory) -> int | None:
    deadlines = tuple(
        deadline for deadline in (
            task.timeout_round, task.coordination_deadline_round,
        )
        if deadline is not None
    )
    return min(deadlines) if deadlines else None


def _final_answer_required(task: TaskMemory) -> bool:
    return task.final_answer_requested or task.coordination_final_requested


def _leave_task(
    turn: Turn,
    task: TaskMemory,
    owner: Unit,
) -> TaskTurnProposal:
    if task.abandon_move_attempted:
        return TaskTurnProposal()
    if not task.task_cells:
        return TaskTurnProposal()
    blocked = turn.blocked(owner)
    candidates = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if not (dx or dy):
                continue
            target = Pos(owner.pos.x + dx, owner.pos.y + dy)
            if not turn.land(target) or target in blocked:
                continue
            separation = min(distance(target, cell) for cell in task.task_cells)
            if separation > 1:
                candidates.append((separation, target))
    if not candidates:
        return TaskTurnProposal()
    _, target = max(candidates, key=lambda item: (item[0], -item[1].x, -item[1].y))
    return TaskTurnProposal(actions=(PlannedAction(ActionProposal(
        owner.unit_id,
        owner.unit_id,
        move_command(target),
        destination=target,
    ), target, "task-abandon"),))


def _remember(task: TaskMemory, label: str, content: str) -> None:
    task.solver_history.append(_bounded(
        f"{label}:\n{content}", MAX_SOLVER_EVENT_CHARS,
    ))
    if len(task.solver_history) > MAX_SOLVER_EVENTS:
        del task.solver_history[:-MAX_SOLVER_EVENTS]
        task.solver_history_truncated = True


def _remember_evidence(task: TaskMemory, label: str, content: str) -> None:
    event = _bounded(f"{label}:\n{content}", MAX_SOLVER_EVENT_CHARS)
    if len(task.solver_evidence) < MAX_SOLVER_EVIDENCE_EVENTS:
        task.solver_evidence.append(event)
    else:
        # Preserve the first two verified context anchors (typically bounded
        # discovery and its read) while refreshing the current evidence slot.
        task.solver_evidence[-1] = event


def _solver_evidence_text(task: TaskMemory) -> str:
    joined = "\n\n".join(task.solver_evidence)
    return _bounded(joined, MAX_SOLVER_EVIDENCE_CHARS) if joined else "(none)"


def _remember_environment_paths(
    state: SessionState,
    task: TaskMemory,
    result: str,
) -> None:
    lines = result.splitlines()[1:]
    for line in lines:
        path = line.strip()
        if (
            not _STANDALONE_PATH.fullmatch(path)
            or len(path) > MAX_ENVIRONMENT_PATH_CHARS
            or path == "/"
            or path.startswith("//")
            or path[-1] in ".,:;)]}，。：；）】"
        ):
            continue
        if path in state.task_environment_paths:
            state.task_environment_paths.remove(path)
        state.task_environment_paths.append(path)
        if path in task.current_environment_paths:
            task.current_environment_paths.remove(path)
        task.current_environment_paths.append(path)
    del state.task_environment_paths[:-MAX_ENVIRONMENT_PATHS]
    task.environment_paths = tuple(state.task_environment_paths)
    task.current_environment_paths[:] = [
        path for path in task.current_environment_paths
        if path in state.task_environment_paths
    ]


def _environment_path_text(task: TaskMemory, task_text: str) -> str:
    if not task.environment_paths:
        return "(none)"
    current = set(task.current_environment_paths)
    historical = [
        path for path in task.environment_paths if path not in current
    ]
    explicit = _task_path_references(task_text)
    related = [path for path in historical if path in explicit]
    if not explicit:
        by_name: dict[str, list[str]] = {}
        for path in historical:
            by_name.setdefault(path.rsplit("/", 1)[-1], []).append(path)
        related = [
            paths[0]
            for name, paths in by_name.items()
            if len(paths) == 1 and _mentions_filename(task_text, name)
        ]
    paths = [
        path for path in task.environment_paths
        if path in current or path in related
    ]
    if not paths:
        return "(none)"
    return (
        "These paths appeared in successful sandbox output from this match; "
        "verify each path for the current task before relying on it:\n"
        + "\n".join(paths)
    )


def _task_path_references(task_text: str) -> frozenset[str]:
    paths = []
    for match in _PATH_REFERENCE.finditer(task_text):
        path = match.group(1).rstrip(".,:;)]}，。：；）】")
        if path and path != "/" and not path.startswith("//"):
            paths.append(path)
    return frozenset(paths)


def _mentions_filename(task_text: str, name: str) -> bool:
    if not name:
        return False
    return re.search(
        rf"(?<![A-Za-z0-9_.-]){re.escape(name)}(?![A-Za-z0-9_.-])",
        task_text,
    ) is not None


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
    start_guard: Callable[[PlayerTask, Pos, int], str | None] | None,
) -> tuple[tuple[PlayerTask, Pos, Pos | None] | None, str | None]:
    options = []
    skip_reason = None
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
                    reason = start_guard(task, stand, 0) if start_guard else None
                    if reason is None:
                        options.append((0, task, cell, None))
                    elif skip_reason is None:
                        skip_reason = reason
                elif path.status == "found" and path.step is not None:
                    cost = path.cost or 0
                    reason = (
                        start_guard(task, stand, cost) if start_guard else None
                    )
                    if reason is None:
                        options.append((cost, task, cell, path.step))
                    elif skip_reason is None:
                        skip_reason = reason
                if clock() >= deadline:
                    break
            if clock() >= deadline:
                break
        if clock() >= deadline:
            break
    if not options:
        return None, skip_reason
    _, task, cell, step = min(options, key=lambda option: (
        option[0],
        -(option[1].score_reward + option[1].gold_reward),
        -(option[1].timeout_rounds or 0),
        option[1].pos.x,
        option[1].pos.y,
    ))
    return (task, cell, step), None


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
