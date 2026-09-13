import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .protocol import Pos, Turn, distance

MAX_ACTION_HISTORY = 64
MAX_ENDED_TASKS = 16
MAX_HISTORY_FACTS = 256


def request_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PendingAction:
    round_no: int
    command_owner_id: int
    actor_id: int
    action: str
    target: Pos | None
    task_instance_id: str | None = None
    source_session: int = 1
    name: str | None = None


@dataclass(frozen=True, slots=True)
class CompletedAction:
    pending: PendingAction
    success: bool | None


@dataclass(frozen=True, slots=True)
class PlanState:
    role_id: int
    target: Pos
    reason: str
    deadline_round: int | None
    source_session: int


@dataclass(slots=True)
class TaskMemory:
    instance_id: str
    prompt_fingerprint: str
    phase: str = "solving"
    pending_llm_round: int | None = None
    pending_cmd_round: int | None = None
    tool_results: list[tuple[str, str]] = field(default_factory=list)
    consumed_tool_results: int = 0
    owner_id: int | None = None
    task_cells: tuple[Pos, ...] = ()
    accepted_round: int | None = None
    timeout_round: int | None = None
    last_submitted_answer: str | None = None
    deferred_answer: str | None = None
    last_submission_feedback_round: int | None = None
    solver_history: list[str] = field(default_factory=list)
    solver_history_truncated: bool = False
    solver_stopped_reason: str | None = None
    last_tool_result_fingerprint: str | None = None
    repeated_tool_result_count: int = 0
    end_reason: str | None = None


@dataclass(frozen=True, slots=True)
class HistoricalFact:
    category: str
    value: str
    source_session: int
    source_round: int


@dataclass(slots=True)
class SessionState:
    team_id: str
    team_type: str
    session_index: int = 1
    last_round_no: int | None = None
    last_fingerprint: str | None = None
    last_response: dict[str, Any] | None = None
    observation_count: int = 0
    task_sequence: int = 0
    plans: dict[int, PlanState] = field(default_factory=dict)
    pending_actions: dict[int, PendingAction] = field(default_factory=dict)
    action_history: list[CompletedAction] = field(default_factory=list)
    active_task: TaskMemory | None = None
    ended_tasks: list[TaskMemory] = field(default_factory=list)
    history: list[HistoricalFact] = field(default_factory=list)
    late_tool_results: int = 0


@dataclass(frozen=True, slots=True)
class ObservationResult:
    cached_response: dict[str, Any] | None
    boundary: str | None = None


class StateStore:
    def __init__(self) -> None:
        self.state: SessionState | None = None

    def observe(
        self,
        turn: Turn,
        payload: dict[str, Any],
        fingerprint: str,
    ) -> ObservationResult:
        team = payload["teamOur"]
        identity = (str(team["teamId"]), str(team["type"]))
        boundary = None
        if self.state is None:
            self.state = SessionState(*identity)
        elif identity != (self.state.team_id, self.state.team_type):
            boundary = "team_identity_changed"
            self._start_session(identity)
        elif (
            self.state.last_round_no is not None
            and turn.round_no < self.state.last_round_no
        ):
            boundary = "suspected_round_rewind"
            self._start_session(identity)

        state = self.state
        if (
            boundary is None
            and state.last_round_no == turn.round_no
            and state.last_fingerprint == fingerprint
            and state.last_response is not None
        ):
            return ObservationResult(copy.deepcopy(state.last_response))

        state.observation_count += 1
        self._apply_feedback(turn, payload)
        self._release_dead_roles(turn)
        self._release_invalid_plans(turn)
        self._update_task(turn, payload, accept_results=boundary is None)
        self._record_news(turn.round_no, payload)
        state.last_round_no = turn.round_no
        state.last_fingerprint = fingerprint
        state.last_response = None
        return ObservationResult(None, boundary)

    def record_response(
        self,
        turn: Turn,
        fingerprint: str,
        response: dict[str, Any],
    ) -> None:
        state = self._require_state()
        self._withdraw_round(turn.round_no)
        state.last_round_no = turn.round_no
        state.last_fingerprint = fingerprint
        state.last_response = copy.deepcopy(response)
        active_task_id = (
            state.active_task.instance_id if state.active_task is not None else None
        )
        for owner, command in response.get("roleCommandMap", {}).items():
            owner_id = int(owner)
            controller_id = command.get("controllerId")
            actor_id = int(controller_id) if controller_id is not None else owner_id
            target = None
            targets = command.get("targetPos")
            if isinstance(targets, list) and targets:
                target = Pos.load(targets[0])
            state.pending_actions[owner_id] = PendingAction(
                round_no=turn.round_no,
                command_owner_id=owner_id,
                actor_id=actor_id,
                action=str(command["action"]),
                target=target,
                task_instance_id=active_task_id,
                source_session=state.session_index,
                name=(
                    str(command["name"])
                    if isinstance(command.get("name"), str)
                    else None
                ),
            )
            if (
                state.active_task is not None
                and command.get("action") == "submitAnswer"
            ):
                state.active_task.phase = "submit_pending"
                state.active_task.last_submitted_answer = str(
                    command.get("taskAnswer") or ""
                )
                state.active_task.deferred_answer = None
            if (
                command.get("action") == "move"
                and target is not None
                and actor_id not in state.plans
            ):
                state.plans[actor_id] = PlanState(
                    role_id=actor_id,
                    target=target,
                    reason="s0_probe",
                    deadline_round=turn.round_no + 1,
                    source_session=state.session_index,
                )
        if state.active_task is not None:
            if response.get("prompt"):
                state.active_task.pending_llm_round = turn.round_no
            if response.get("executeCmd"):
                state.active_task.pending_cmd_round = turn.round_no

    def set_plan(
        self,
        role_id: int,
        target: Pos,
        reason: str,
        deadline_round: int | None,
    ) -> None:
        state = self._require_state()
        state.plans[role_id] = PlanState(
            role_id=role_id,
            target=target,
            reason=reason,
            deadline_round=deadline_round,
            source_session=state.session_index,
        )

    def clear_plan(self, role_id: int) -> None:
        self._require_state().plans.pop(role_id, None)

    def _start_session(self, identity: tuple[str, str]) -> None:
        previous = self._require_state()
        self.state = SessionState(
            team_id=identity[0],
            team_type=identity[1],
            session_index=previous.session_index + 1,
            history=previous.history,
            action_history=previous.action_history,
            ended_tasks=previous.ended_tasks,
            late_tool_results=previous.late_tool_results,
        )

    def _apply_feedback(self, turn: Turn, payload: dict[str, Any]) -> None:
        state = self._require_state()
        raw_feedback = payload.get("lastRoundRoleActionResults")
        feedback: dict[int, bool] = {}
        if isinstance(raw_feedback, dict):
            for raw_id, success in raw_feedback.items():
                if not isinstance(success, bool):
                    continue
                try:
                    feedback[int(raw_id)] = success
                except (TypeError, ValueError):
                    continue
        for owner_id, pending in tuple(state.pending_actions.items()):
            if pending.round_no >= turn.round_no:
                continue
            success = None
            if pending.round_no == turn.round_no - 1:
                success = feedback.get(owner_id)
            state.pending_actions.pop(owner_id)
            state.action_history.append(CompletedAction(pending, success))
            plan = state.plans.get(pending.actor_id)
            if (
                pending.action != "move"
                or (
                    success is True
                    and plan is not None
                    and plan.reason == "s0_probe"
                )
            ):
                state.plans.pop(pending.actor_id, None)
        del state.action_history[:-MAX_ACTION_HISTORY]

    def _withdraw_round(self, round_no: int) -> None:
        state = self._require_state()
        for owner_id, pending in tuple(state.pending_actions.items()):
            if pending.round_no != round_no:
                continue
            state.pending_actions.pop(owner_id)
            state.plans.pop(pending.actor_id, None)
        task = state.active_task
        if task is None:
            return
        if task.pending_llm_round == round_no:
            task.pending_llm_round = None
        if task.pending_cmd_round == round_no:
            task.pending_cmd_round = None

    def _release_dead_roles(self, turn: Turn) -> None:
        state = self._require_state()
        alive = {unit.unit_id for unit in turn.controllable()}
        for role_id in tuple(state.plans):
            if role_id not in alive:
                state.plans.pop(role_id)
        for owner_id, pending in tuple(state.pending_actions.items()):
            if pending.actor_id not in alive:
                state.pending_actions.pop(owner_id)
                state.action_history.append(CompletedAction(pending, None))
        del state.action_history[:-MAX_ACTION_HISTORY]

    def _release_invalid_plans(self, turn: Turn) -> None:
        state = self._require_state()
        for role_id, plan in tuple(state.plans.items()):
            reason = plan.reason
            role = turn.unit(role_id)
            if plan.deadline_round is not None and turn.round_no > plan.deadline_round:
                state.plans.pop(role_id)
                continue
            if reason.startswith("mine:") and (
                role is None
                or role.backpack_full
                or turn.zones.get(plan.target) not in ("stone", "iron", "copper")
            ):
                state.plans.pop(role_id)
            elif reason == "vendor" and (
                role is None
                or turn.zones.get(plan.target) != "vendor"
                or not any(
                    item in ("stone", "iron", "copper")
                    for item in role.backpack
                )
            ):
                state.plans.pop(role_id)
            elif reason.startswith("shop:") and (
                role is None
                or turn.zones.get(plan.target) != "weaponShop"
                or reason.split(":", 1)[1] not in turn.weapon_prices
                or role.capacity is None
                or role.backpack_full
                or turn.weapon_prices.get(reason.split(":", 1)[1], turn.gold + 1)
                > turn.gold
            ):
                state.plans.pop(role_id)
            elif reason.startswith("build:"):
                kind = reason.split(":", 1)[1]
                invalid = (
                    role is None
                    or not turn.is_day
                    or not turn.land(plan.target)
                    or plan.target in turn.occupied_cells()
                )
                if kind in ("gatling", "railgun", "rocket"):
                    invalid = invalid or (
                        turn.station() is None
                        or len(turn.weapons()) >= 3
                        or turn.gold < 25
                    )
                elif kind == "wall":
                    invalid = invalid or role is None or "stone" not in role.backpack
                else:
                    invalid = True
                if invalid:
                    state.plans.pop(role_id)
            elif reason.startswith("gunner:"):
                try:
                    weapon_id = int(reason.split(":", 1)[1])
                except ValueError:
                    state.plans.pop(role_id)
                    continue
                weapon = turn.unit(weapon_id)
                if weapon is None or weapon.kind not in ("gatling", "railgun", "rocket"):
                    state.plans.pop(role_id)
            elif reason.startswith("use:"):
                parts = reason.split(":")
                role = turn.unit(role_id)
                try:
                    target = turn.unit(int(parts[2])) if len(parts) == 3 else None
                except ValueError:
                    target = None
                if (
                    len(parts) != 3
                    or role is None
                    or parts[1] not in role.backpack
                    or target is None
                ):
                    state.plans.pop(role_id)

    def _update_task(
        self,
        turn: Turn,
        payload: dict[str, Any],
        *,
        accept_results: bool,
    ) -> None:
        state = self._require_state()
        round_no = turn.round_no
        phase_task = payload.get("phaseTask")
        phase_task = phase_task if isinstance(phase_task, str) else ""
        llm_result = payload.get("llmResp")
        llm_result = llm_result if isinstance(llm_result, str) else ""
        cmd_result = payload.get("lastCmdResult")
        cmd_result = cmd_result if isinstance(cmd_result, str) else ""

        if not phase_task:
            if state.active_task is not None:
                state.active_task.phase = "ended"
                state.active_task.end_reason = self._task_end_reason(
                    turn, state.active_task,
                )
                state.active_task.pending_llm_round = None
                state.active_task.pending_cmd_round = None
                state.ended_tasks.append(state.active_task)
                del state.ended_tasks[:-MAX_ENDED_TASKS]
                state.active_task = None
            state.late_tool_results += bool(llm_result) + bool(cmd_result)
            return

        task_fingerprint = hashlib.sha256(
            phase_task.encode("utf-8")
        ).hexdigest()
        if (
            state.active_task is None
            or state.active_task.prompt_fingerprint != task_fingerprint
        ):
            if state.active_task is not None:
                state.active_task.phase = "ended"
                state.active_task.end_reason = "replaced"
                state.active_task.pending_llm_round = None
                state.active_task.pending_cmd_round = None
                state.ended_tasks.append(state.active_task)
                del state.ended_tasks[:-MAX_ENDED_TASKS]
            state.task_sequence += 1
            owner_id, task_cells, accepted_round, timeout_round = (
                self._active_task_context(turn)
            )
            state.active_task = TaskMemory(
                instance_id=f"{state.session_index}:{state.task_sequence}",
                prompt_fingerprint=task_fingerprint,
                owner_id=owner_id,
                task_cells=task_cells,
                accepted_round=accepted_round,
                timeout_round=timeout_round,
            )

        task = state.active_task
        self._apply_tool_result(
            task,
            kind="llm",
            result=llm_result,
            pending_attr="pending_llm_round",
            round_no=round_no,
            accept_result=accept_results,
        )
        self._apply_tool_result(
            task,
            kind="cmd",
            result=cmd_result,
            pending_attr="pending_cmd_round",
            round_no=round_no,
            accept_result=accept_results,
        )

    def _apply_tool_result(
        self,
        task: TaskMemory,
        *,
        kind: str,
        result: str,
        pending_attr: str,
        round_no: int,
        accept_result: bool,
    ) -> None:
        state = self._require_state()
        pending_round = getattr(task, pending_attr)
        if accept_result and pending_round == round_no - 1:
            task.tool_results.append((kind, result))
            task.phase = "solving"
        elif result:
            state.late_tool_results += 1
        if pending_round is not None and pending_round < round_no:
            setattr(task, pending_attr, None)

    def _active_task_context(
        self,
        turn: Turn,
    ) -> tuple[int | None, tuple[Pos, ...], int | None, int | None]:
        state = self._require_state()
        accepted = next((
            completed.pending
            for completed in reversed(state.action_history)
            if completed.pending.action == "acceptTask"
            and completed.pending.source_session == state.session_index
            and completed.pending.round_no == turn.round_no - 1
            and completed.success is not False
        ), None)
        owner_id = accepted.actor_id if accepted is not None else None
        if owner_id is None and turn.pioneers():
            owner_id = turn.pioneers()[0].unit_id
        owner = turn.unit(owner_id) if owner_id is not None else None
        matched = next((
            task for task in turn.player_tasks
            if owner is not None
            and any(distance(owner.pos, cell) == 1 for cell in turn.task_cells(task))
        ), None)
        task_cells = turn.task_cells(matched) if matched is not None else ()
        accepted_round = accepted.round_no if accepted is not None else None
        timeout_round = None
        if (
            accepted_round is not None
            and matched is not None
            and matched.timeout_rounds is not None
        ):
            timeout_round = accepted_round + matched.timeout_rounds
        return owner_id, task_cells, accepted_round, timeout_round

    @staticmethod
    def _task_end_reason(turn: Turn, task: TaskMemory) -> str:
        if any(error.code == 1 for error in turn.errors):
            return "timeout"
        if task.timeout_round is not None and turn.round_no > task.timeout_round:
            return "timeout"
        owner = turn.unit(task.owner_id) if task.owner_id is not None else None
        if task.owner_id is not None and owner is None:
            return "death"
        if (
            owner is not None
            and task.task_cells
            and all(distance(owner.pos, cell) != 1 for cell in task.task_cells)
        ):
            return "left_point"
        return "unknown"

    def _record_news(self, round_no: int, payload: dict[str, Any]) -> None:
        state = self._require_state()
        news = payload.get("worldNews")
        if not isinstance(news, dict):
            return
        known = {(fact.category, fact.value) for fact in state.history}
        for category in ("officialNews", "folkLegends"):
            value = news.get(category)
            if not isinstance(value, str) or not value or (category, value) in known:
                continue
            state.history.append(HistoricalFact(
                category=category,
                value=value,
                source_session=state.session_index,
                source_round=round_no,
            ))
            known.add((category, value))
        del state.history[:-MAX_HISTORY_FACTS]

    def _require_state(self) -> SessionState:
        if self.state is None:
            raise RuntimeError("state has not observed a turn")
        return self.state
