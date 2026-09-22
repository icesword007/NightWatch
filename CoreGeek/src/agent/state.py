import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .defense_pressure import WaveForecast, WaveNight, observe_pressure
from .intelligence import (
    MAX_NEWS_CALLS_PER_DAY,
    MAX_NEWS_CANDIDATES,
    NewsCandidate,
    NewsRequest,
    build_news_request,
    parse_news_response,
    source_id,
)
from .protocol import ROUNDS_PER_DAY, Pos, Turn, distance
from .pressure_shadow import ShadowState

MAX_ACTION_HISTORY = 64
MAX_ENDED_TASKS = 16
MAX_HISTORY_FACTS = 256
MAX_NEWS_TEXT_CHARS = 2_048
MAX_NEWS_ATTEMPTS = 256
MAX_NEWS_DAY_RECORDS = 2
MAX_NEWS_EVENT_TEXT_CHARS = 4_096


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
    task_type: str = ""
    phase: str = "solving"
    pending_llm_round: int | None = None
    pending_cmd_round: int | None = None
    tool_results: list[tuple[str, str]] = field(default_factory=list)
    consumed_tool_results: int = 0
    tool_inputs_this_round: list[dict[str, int | str]] = field(default_factory=list)
    owner_id: int | None = None
    task_cells: tuple[Pos, ...] = ()
    accepted_round: int | None = None
    timeout_round: int | None = None
    last_submitted_answer: str | None = None
    deferred_answer: str | None = None
    last_submission_feedback_round: int | None = None
    solver_history: list[str] = field(default_factory=list)
    solver_history_truncated: bool = False
    solver_evidence: list[str] = field(default_factory=list)
    failed_tool_observations: list[tuple[str, str]] = field(
        default_factory=list
    )
    environment_paths: tuple[str, ...] = ()
    current_environment_paths: list[str] = field(default_factory=list)
    solver_stopped_reason: str | None = None
    last_tool_result_fingerprint: str | None = None
    last_accepted_cmd_result_round: int | None = None
    repeated_tool_result_count: int = 0
    last_command: str | None = None
    consecutive_nonzero_command: str | None = None
    consecutive_nonzero_count: int = 0
    repeated_command_correction_requested: bool = False
    last_cycle_fingerprint: str | None = None
    repeated_cycle_count: int = 0
    command_count: int = 0
    entry_read_attempted: bool = False
    final_answer_requested: bool = False
    final_only_correction_requested: bool = False
    vacuous_partial_correction_requested: bool = False
    crlf_hint_requested: bool = False
    envelope_correction_requested: bool = False
    envelope_correction_pending: bool = False
    last_envelope_rejection: str | None = None
    coordination_final_requested: bool = False
    coordination_deadline_round: int | None = None
    abandon_move_attempted: bool = False
    end_reason: str | None = None
    submission_count: int = 0
    start_total_score: int | None = None
    start_gold: int | None = None
    associated_error_codes: list[int] = field(default_factory=list)
    pagination_checked: bool = False
    pagination_pages_seen: list[tuple[int, int, int]] = field(default_factory=list)
    pagination_hint_count: int = 0
    sop_hint: str = ""


@dataclass(frozen=True, slots=True)
class HistoricalFact:
    category: str
    value: str
    value_fingerprint: str
    original_length: int
    value_truncated: bool
    source_session: int
    source_round: int
    source_day: int


@dataclass(frozen=True, slots=True)
class NewsObservation:
    fact: HistoricalFact
    is_new: bool
    observed_round: int
    observed_day: int


@dataclass(frozen=True, slots=True)
class BaseReserve:
    holder_id: int
    station_id: int
    station_level: int
    item: str


@dataclass(slots=True)
class SessionState:
    team_id: str
    team_type: str
    session_index: int = 1
    pressure_shadow: ShadowState = field(default_factory=ShadowState)
    base_reserve: BaseReserve | None = None
    base_last_sample: tuple[int, int, int, int] | None = None
    base_recent_drops: tuple[int, ...] = ()
    base_reserve_event: str | None = None
    base_blocked_day: int | None = None
    base_blocked_holder_id: int | None = None
    base_reserve_reason: str | None = None
    last_round_no: int | None = None
    last_fingerprint: str | None = None
    last_response: dict[str, Any] | None = None
    observation_count: int = 0
    task_sequence: int = 0
    plans: dict[int, PlanState] = field(default_factory=dict)
    procurement_blocked_days: dict[int, int] = field(default_factory=dict)
    emergency_purchase_night: int | None = None
    pending_actions: dict[int, PendingAction] = field(default_factory=dict)
    action_history: list[CompletedAction] = field(default_factory=list)
    active_task: TaskMemory | None = None
    ended_tasks: list[TaskMemory] = field(default_factory=list)
    task_end_this_round: dict[str, Any] | None = None
    history: list[HistoricalFact] = field(default_factory=list)
    news_observations: tuple[NewsObservation, ...] = ()
    pending_news_request: NewsRequest | None = None
    news_calls_by_day: dict[int, int] = field(default_factory=dict)
    news_blocked_days: set[int] = field(default_factory=set)
    news_attempted_source_ids: list[str] = field(default_factory=list)
    news_candidates: list[NewsCandidate] = field(default_factory=list)
    news_events: list[dict[str, Any]] = field(default_factory=list)
    news_skip_reason: str | None = None
    late_tool_results: int = 0
    task_environment_paths: list[str] = field(default_factory=list)
    layout_initialized: bool = False
    layout_direction: tuple[int, int] = (1, 0)
    layout_direction_source: str = "uninitialized"
    layout_tower_targets: tuple[Pos, ...] = ()
    layout_gunner_stands: tuple[Pos, ...] = ()
    layout_wall_targets: tuple[Pos, ...] = ()
    layout_exit_cells: tuple[Pos, ...] = ()
    layout_complete: bool = False
    layout_degraded_reason: str | None = None
    layout_observed_deviation: str | None = None
    day_return_day: int | None = None
    day_return_weapons: dict[int, int] = field(default_factory=dict)
    day_return_stands: dict[int, Pos] = field(default_factory=dict)
    fortification_initialized: bool = False
    fortification_builder_id: int | None = None
    fortification_targets: tuple[Pos, ...] = ()
    fortification_batch_targets: tuple[Pos, ...] = ()
    fortification_batch_signature: str | None = None
    fortification_return_stand: Pos | None = None
    fortification_builder_snapshot: tuple[Pos, tuple[str, ...]] | None = None
    fortification_completed: set[Pos] = field(default_factory=set)
    fortification_observed_days: dict[Pos, int] = field(default_factory=dict)
    fortification_recovery_targets: set[Pos] = field(default_factory=set)
    fortification_attempt_days: dict[Pos, int] = field(
        default_factory=dict
    )
    fortification_planning_day: int | None = None
    fortification_failed: set[Pos] = field(default_factory=set)
    fortification_deferred: dict[Pos, tuple[str, str, int]] = field(
        default_factory=dict
    )
    fortification_phase: str = "idle"
    fortification_skip_reason: str | None = None
    wave_history: list[WaveNight] = field(default_factory=list)
    wave_history_truncated: bool = False
    wave_forecasts: list[WaveForecast] = field(default_factory=list)
    last_trace: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ObservationResult:
    cached_response: dict[str, Any] | None
    boundary: str | None = None
    cached_trace: dict[str, Any] | None = None


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
            return ObservationResult(
                copy.deepcopy(state.last_response),
                cached_trace=copy.deepcopy(state.last_trace),
            )

        state.observation_count += 1
        state.task_end_this_round = None
        state.news_events = []
        state.news_skip_reason = None
        news_response_claimed = self._consume_news_response(
            turn, payload, accept_result=boundary is None,
        )
        self._apply_feedback(turn, payload)
        self._release_dead_roles(turn)
        self._release_invalid_plans(turn)
        self._observe_base_upgrade(turn)
        self._observe_fortification(turn)
        observe_pressure(turn, state)
        self._update_task(
            turn,
            payload,
            accept_results=boundary is None,
            ignore_llm_result=news_response_claimed,
        )
        self._record_news(turn, payload)
        state.last_round_no = turn.round_no
        state.last_fingerprint = fingerprint
        state.last_response = None
        return ObservationResult(None, boundary)

    def record_response(
        self,
        turn: Turn,
        fingerprint: str,
        response: dict[str, Any],
        trace: dict[str, Any] | None = None,
    ) -> None:
        state = self._require_state()
        self._withdraw_round(turn.round_no)
        state.last_round_no = turn.round_no
        state.last_fingerprint = fingerprint
        state.last_response = copy.deepcopy(response)
        state.last_trace = copy.deepcopy(trace)
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
                command.get("action") == "build"
                and command.get("name") == "wall"
                and target in state.fortification_targets
            ):
                state.fortification_attempt_days[target] = self._day(
                    turn.round_no
                )
            task = state.active_task
            if (
                task is not None
                and task.solver_stopped_reason is not None
                and owner_id == task.owner_id
                and command.get("action") == "move"
                and target is not None
                and task.task_cells
                and all(distance(target, cell) != 1 for cell in task.task_cells)
            ):
                task.abandon_move_attempted = True
            if (
                state.active_task is not None
                and command.get("action") == "submitAnswer"
            ):
                state.active_task.submission_count += 1
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

    def prepare_news_request(self, turn: Turn) -> NewsRequest | None:
        state = self._require_state()
        day = self._day(turn.round_no)
        self._trim_news_days(day)
        if state.pending_news_request is not None:
            state.news_skip_reason = "request_pending"
            return None
        if day in state.news_blocked_days:
            state.news_skip_reason = "quota_error_for_day"
            return None
        if state.news_calls_by_day.get(day, 0) >= MAX_NEWS_CALLS_PER_DAY:
            state.news_skip_reason = "daily_policy_limit"
            return None
        attempted = set(state.news_attempted_source_ids)
        current_session = [
            fact for fact in state.history
            if fact.source_session == state.session_index
        ]
        eligible = sorted(
            (
                fact for fact in current_session
                if source_id(
                    state.session_index,
                    fact.category,
                    fact.value_fingerprint,
                ) not in attempted
            ),
            key=lambda fact: (-fact.source_round, fact.category),
        )
        context = sorted(
            (
                fact for fact in current_session
                if source_id(
                    state.session_index,
                    fact.category,
                    fact.value_fingerprint,
                ) in attempted
            ),
            key=lambda fact: (-fact.source_round, fact.category),
        )
        request = build_news_request(
            eligible,
            context_sources=context,
            session=state.session_index,
            round_no=turn.round_no,
            day=day,
        )
        state.news_skip_reason = None if request is not None else "no_new_evidence"
        return request

    def record_news_request(self, request: NewsRequest) -> None:
        state = self._require_state()
        if state.pending_news_request is not None:
            raise RuntimeError("news request already pending")
        state.pending_news_request = request
        state.news_calls_by_day[request.issued_day] = (
            state.news_calls_by_day.get(request.issued_day, 0) + 1
        )
        state.news_attempted_source_ids.extend(
            source.source_id for source in request.sources
            if source.evidence_role == "new_evidence"
        )
        del state.news_attempted_source_ids[:-MAX_NEWS_ATTEMPTS]
        state.news_events.append({
            "kind": "request_issued",
            "requestId": request.request_id,
            "session": request.source_session,
            "issuedRound": request.issued_round,
            "issuedDay": request.issued_day,
            "sourceIds": [source.source_id for source in request.sources],
            "newSourceIds": [
                source.source_id for source in request.sources
                if source.evidence_role == "new_evidence"
            ],
            "contextSourceIds": [
                source.source_id for source in request.sources
                if source.evidence_role == "context"
            ],
            "prompt": self._news_text_detail(request.prompt),
        })

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
            reserve = state.base_reserve
            if (
                reserve is not None
                and pending.actor_id == reserve.holder_id
                and pending.name == reserve.item
                and pending.action in ("buy", "use")
                and success is False
            ):
                state.base_reserve = None
                state.base_reserve_event = f"{pending.action}_failed"
                state.base_blocked_day = self._day(turn.round_no)
                state.base_blocked_holder_id = pending.actor_id
            elif (
                reserve is not None
                and pending.actor_id == reserve.holder_id
                and pending.name == reserve.item
                and pending.action in ("buy", "use")
                and success is True
            ):
                state.base_reserve_event = f"{pending.action}_reported"
            if (
                pending.action == "build"
                and pending.name == "wall"
                and pending.target in state.fortification_targets
            ):
                if success is True:
                    state.fortification_completed.add(pending.target)
                else:
                    state.fortification_failed.add(pending.target)
            plan = state.plans.get(pending.actor_id)
            if (
                plan is not None
                and plan.reason.startswith("batch:")
                and success is False
            ):
                state.procurement_blocked_days[pending.actor_id] = self._day(
                    turn.round_no,
                )
            keep_funding = (
                plan is not None
                and plan.reason.startswith(("fund:", "batch:"))
                and success is not False
            )
            if not keep_funding and (
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

    def _observe_base_upgrade(self, turn: Turn) -> None:
        state = self._require_state()
        station = turn.station()
        previous = state.base_last_sample
        if station is None:
            state.base_last_sample = None
            state.base_recent_drops = ()
            state.base_reserve = None
            return
        if (
            previous is not None
            and previous[:2] == (station.unit_id, station.level)
            and previous[2] == turn.round_no - 1
        ):
            state.base_recent_drops = (
                *state.base_recent_drops[-2:],
                max(0, previous[3] - station.health),
            )
        else:
            state.base_recent_drops = ()
        state.base_last_sample = (
            station.unit_id, station.level, turn.round_no, station.health,
        )
        reserve = state.base_reserve
        if reserve is None:
            return
        holder = turn.unit(reserve.holder_id)
        if station.unit_id != reserve.station_id or station.level != reserve.station_level:
            state.base_reserve = None
            state.base_reserve_event = "upgrade_observed"
        elif holder is None:
            state.base_reserve = None
            state.base_reserve_event = "holder_lost"
        elif state.base_reserve_event == "use_reported":
            state.base_reserve = None
            state.base_reserve_event = "use_unconfirmed"
            state.base_blocked_day = self._day(turn.round_no)
            state.base_blocked_holder_id = reserve.holder_id
        elif state.base_reserve_event == "buy_reported" and reserve.item not in holder.backpack:
            state.base_reserve = None
            state.base_reserve_event = "buy_unconfirmed"
            state.base_blocked_day = self._day(turn.round_no)
            state.base_blocked_holder_id = reserve.holder_id
        elif reserve.item not in holder.backpack and not any(
            plan.role_id == reserve.holder_id
            and plan.reason.startswith(f"fund:{reserve.item}:")
            for plan in state.plans.values()
        ):
            state.base_reserve = None
            state.base_reserve_event = "voucher_missing"
        elif reserve.item in holder.backpack and state.base_reserve_event == "buy_reported":
            state.base_reserve_event = "held"

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
            elif reason.startswith("fund:") and (
                role is None
                or plan.deadline_round is None
                or turn.round_no > plan.deadline_round
            ):
                state.plans.pop(role_id)
            elif reason.startswith("batch:") and (
                role is None
                or plan.deadline_round is None
                or turn.round_no > plan.deadline_round
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

    def _observe_fortification(self, turn: Turn) -> None:
        state = self._require_state()
        if not state.fortification_initialized:
            return
        day = self._day(turn.round_no)
        if state.fortification_planning_day is None:
            state.fortification_planning_day = day
        elif state.fortification_planning_day != day:
            state.fortification_planning_day = day
            state.fortification_batch_targets = ()
            state.fortification_deferred.clear()
        occupied_walls = {wall.pos for wall in turn.walls()}
        observed = occupied_walls.intersection(state.fortification_targets)
        for target in observed:
            state.fortification_completed.add(target)
            state.fortification_observed_days[target] = day
            state.fortification_recovery_targets.discard(target)
            state.fortification_attempt_days.pop(target, None)
        if not turn.is_day:
            return
        reopened = {
            target for target in state.fortification_completed
            if target not in occupied_walls
            and target not in state.fortification_failed
            and state.fortification_observed_days.get(target, day) < day
        }
        if not reopened:
            return
        state.fortification_completed.difference_update(reopened)
        state.fortification_recovery_targets.update(reopened)
        state.fortification_batch_targets = ()

    def _update_task(
        self,
        turn: Turn,
        payload: dict[str, Any],
        *,
        accept_results: bool,
        ignore_llm_result: bool = False,
    ) -> None:
        state = self._require_state()
        round_no = turn.round_no
        phase_task = payload.get("phaseTask")
        phase_task = phase_task if isinstance(phase_task, str) else ""
        llm_result = "" if ignore_llm_result else payload.get("llmResp")
        llm_result = llm_result if isinstance(llm_result, str) else ""
        cmd_result = payload.get("lastCmdResult")
        cmd_result = cmd_result if isinstance(cmd_result, str) else ""

        if not phase_task:
            if state.active_task is not None:
                self._capture_task_end(
                    turn, payload,
                    self._task_end_reason(turn, state.active_task),
                )
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
                self._capture_task_end(turn, payload, "replaced")
            state.task_sequence += 1
            owner_id, task_cells, accepted_round, timeout_round, task_type = (
                self._active_task_context(turn)
            )
            state.active_task = TaskMemory(
                instance_id=f"{state.session_index}:{state.task_sequence}",
                prompt_fingerprint=task_fingerprint,
                task_type=task_type,
                owner_id=owner_id,
                task_cells=task_cells,
                accepted_round=accepted_round,
                timeout_round=timeout_round,
                environment_paths=tuple(state.task_environment_paths),
                start_total_score=(
                    payload.get("teamOur", {}).get("totalScore")
                    if type(payload.get("teamOur", {}).get("totalScore")) is int
                    else None
                ),
                start_gold=(
                    payload.get("teamOur", {}).get("goldNum")
                    if type(payload.get("teamOur", {}).get("goldNum")) is int
                    else None
                ),
            )

        task = state.active_task
        prior_submit = any(
            completed.pending.action == "submitAnswer"
            and completed.pending.task_instance_id == task.instance_id
            and completed.pending.round_no == round_no - 1
            for completed in state.action_history
        )
        if prior_submit and any(error.code == 2 for error in turn.errors):
            if 2 not in task.associated_error_codes:
                task.associated_error_codes.append(2)
        task.tool_inputs_this_round = []
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

    def _capture_task_end(
        self, turn: Turn, payload: dict[str, Any], reason: str,
    ) -> None:
        state = self._require_state()
        task = state.active_task
        if task is None:
            return
        task.phase = "ended"
        task.end_reason = reason
        task.pending_llm_round = None
        task.pending_cmd_round = None
        prior_submit = any(
            completed.pending.action == "submitAnswer"
            and completed.pending.task_instance_id == task.instance_id
            and completed.pending.round_no == turn.round_no - 1
            for completed in state.action_history
        )
        if prior_submit and any(error.code == 2 for error in turn.errors):
            if 2 not in task.associated_error_codes:
                task.associated_error_codes.append(2)
        if reason == "timeout" and any(error.code == 1 for error in turn.errors):
            if 1 not in task.associated_error_codes:
                task.associated_error_codes.append(1)
        team = payload.get("teamOur")
        team = team if isinstance(team, dict) else {}
        score = team.get("totalScore")
        gold = team.get("goldNum")
        state.task_end_this_round = {
            "instanceId": task.instance_id,
            "acceptedRound": task.accepted_round,
            "timeoutRound": task.timeout_round,
            "endRound": turn.round_no,
            "endReason": reason,
            "submissionCount": task.submission_count,
            "associatedErrorCodes": sorted(task.associated_error_codes),
            "solverStoppedReason": task.solver_stopped_reason,
            "successStatus": "unknown",
            "observedTeamScoreDelta": (
                score - task.start_total_score
                if type(score) is int and task.start_total_score is not None
                else None
            ),
            "observedGoldDelta": (
                gold - task.start_gold
                if type(gold) is int and task.start_gold is not None
                else None
            ),
            "attribution": "unattributed",
        }
        state.ended_tasks.append(task)
        del state.ended_tasks[:-MAX_ENDED_TASKS]
        state.active_task = None

    def _consume_news_response(
        self,
        turn: Turn,
        payload: dict[str, Any],
        *,
        accept_result: bool,
    ) -> bool:
        state = self._require_state()
        pending = state.pending_news_request
        if pending is None or not accept_result:
            return False
        if turn.round_no <= pending.issued_round:
            return False
        raw = payload.get("llmResp")
        raw = raw if isinstance(raw, str) else ""
        if turn.round_no != pending.issued_round + 1:
            state.pending_news_request = None
            state.news_events.append({
                "kind": "response_rejected",
                "requestId": pending.request_id,
                "reason": "late_response",
                "response": self._news_text_detail(raw),
            })
            return bool(raw)
        if any(error.code == 5 for error in turn.errors):
            state.news_blocked_days.add(pending.issued_day)
            state.pending_news_request = None
            state.news_events.append({
                "kind": "response_rejected",
                "requestId": pending.request_id,
                "reason": "platform_quota_error",
                "requestDay": pending.issued_day,
                "response": self._news_text_detail(raw),
            })
            return bool(raw)
        parsed = parse_news_response(pending, raw)
        state.pending_news_request = None
        if parsed.rejection_reason is None:
            state.news_candidates.extend(parsed.candidates)
            del state.news_candidates[:-MAX_NEWS_CANDIDATES]
            state.news_events.append({
                "kind": "response_accepted",
                "requestId": pending.request_id,
                "candidateCount": len(parsed.candidates),
                "candidates": [self._candidate_detail(candidate)
                               for candidate in parsed.candidates],
                "response": self._news_text_detail(raw),
            })
        else:
            event = {
                "kind": "response_rejected",
                "requestId": pending.request_id,
                "reason": parsed.rejection_reason,
                "response": self._news_text_detail(raw),
            }
            if parsed.rejection_detail is not None:
                event["rejectionDetail"] = parsed.rejection_detail
            state.news_events.append(event)
        return bool(raw)

    def _trim_news_days(self, current_day: int) -> None:
        state = self._require_state()
        previous = sorted(
            (
                day
                for day in set(state.news_calls_by_day) | state.news_blocked_days
                if day != current_day
            ),
            reverse=True,
        )
        keep_set = {current_day, *previous[:MAX_NEWS_DAY_RECORDS - 1]}
        state.news_calls_by_day = {
            day: count for day, count in state.news_calls_by_day.items()
            if day in keep_set
        }
        state.news_blocked_days.intersection_update(keep_set)

    @staticmethod
    def _day(round_no: int) -> int:
        return (round_no - 1) // ROUNDS_PER_DAY + 1

    @staticmethod
    def _news_text_detail(value: str) -> dict[str, Any]:
        return {
            "value": value[:MAX_NEWS_EVENT_TEXT_CHARS],
            "originalLength": len(value),
            "truncated": len(value) > MAX_NEWS_EVENT_TEXT_CHARS,
            "fingerprint": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def _candidate_detail(candidate: NewsCandidate) -> dict[str, Any]:
        detail = {
            "type": candidate.kind,
            "interpretation": candidate.interpretation,
            "citations": [{
                "sourceId": citation.source_id,
                "excerpt": citation.excerpt,
            } for citation in candidate.citations],
            "missingConditions": list(candidate.missing_conditions),
            "conflicts": list(candidate.conflicts),
            "status": candidate.status,
            "citationSourceSessions": list(
                candidate.citation_source_sessions,
            ),
            "citationSourceTruncated": candidate.citation_source_truncated,
        }
        if candidate.treasure_conditions is not None:
            conditions = candidate.treasure_conditions
            detail["treasureConditions"] = {
                "location": StateStore._treasure_field_detail(
                    conditions.location,
                ),
                "window": StateStore._treasure_field_detail(conditions.window),
                "items": StateStore._treasure_field_detail(conditions.items),
            }
        return detail

    @staticmethod
    def _treasure_field_detail(field: Any) -> dict[str, Any] | None:
        if field is None:
            return None
        value = field.value
        if isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, dict):
            value = dict(value)
        detail = {
            "value": value,
            "citations": [{
                "sourceId": citation.source_id,
                "excerpt": citation.excerpt,
            } for citation in field.citations],
            "sourceSessions": list(field.source_sessions),
            "sourceTruncated": field.source_truncated,
        }
        if field.derivation is not None:
            detail["derivation"] = {
                "kind": field.derivation.kind,
                "explanation": field.derivation.explanation,
                "unresolved": list(field.derivation.unresolved),
                "timeBasis": field.derivation.time_basis,
            }
        return detail

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
            task.tool_inputs_this_round.append({
                "kind": kind,
                "issuedRound": pending_round,
                "receivedRound": round_no,
                "originalChars": len(result),
            })
            task.phase = "solving"
            if kind == "cmd":
                task.last_accepted_cmd_result_round = round_no
        elif result:
            state.late_tool_results += 1
        if pending_round is not None and pending_round < round_no:
            setattr(task, pending_attr, None)

    def _active_task_context(
        self,
        turn: Turn,
    ) -> tuple[int | None, tuple[Pos, ...], int | None, int | None, str]:
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
        return (
            owner_id, task_cells, accepted_round, timeout_round,
            matched.task_type if matched is not None else "",
        )

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

    def _record_news(self, turn: Turn, payload: dict[str, Any]) -> None:
        state = self._require_state()
        state.news_observations = ()
        news = payload.get("worldNews")
        if not isinstance(news, dict):
            return
        known = {
            (fact.category, fact.value_fingerprint): fact
            for fact in state.history
            if fact.source_session == state.session_index
        }
        observations = []
        observed_day = (turn.round_no - 1) // ROUNDS_PER_DAY + 1
        for category in ("officialNews", "folkLegends"):
            value = news.get(category)
            if not isinstance(value, str) or not value:
                continue
            fingerprint = hashlib.sha256(value.encode("utf-8")).hexdigest()
            fact = known.get((category, fingerprint))
            is_new = fact is None
            if fact is None:
                fact = HistoricalFact(
                    category=category,
                    value=value[:MAX_NEWS_TEXT_CHARS],
                    value_fingerprint=fingerprint,
                    original_length=len(value),
                    value_truncated=len(value) > MAX_NEWS_TEXT_CHARS,
                    source_session=state.session_index,
                    source_round=turn.round_no,
                    source_day=observed_day,
                )
                state.history.append(fact)
                known[(category, fingerprint)] = fact
            observations.append(NewsObservation(
                fact=fact,
                is_new=is_new,
                observed_round=turn.round_no,
                observed_day=observed_day,
            ))
        state.news_observations = tuple(observations)
        del state.history[:-MAX_HISTORY_FACTS]

    def _require_state(self) -> SessionState:
        if self.state is None:
            raise RuntimeError("state has not observed a turn")
        return self.state
