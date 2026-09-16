import copy
import hashlib
import threading
import time
from typing import Any, Callable

from .actions import ActionAllocator, ActionProposal
from .defense import (
    DUSK_POSITIONING_ROUNDS,
    propose_defense,
    protected_gunners,
    task_start_skip_reason,
    task_pioneer_day_return_action,
    task_pioneer_recall_action,
)
from .emergency import propose_held_emergency
from .defense_pressure import pressure_diagnostic
from .economy import propose_economy, wall_build_positions
from .fortification import fortification_diagnostic, prepare_fortification
from .intelligence import MAX_NEWS_CALLS_PER_DAY, MAX_NEWS_CANDIDATES
from .layout import ensure_defense_layout
from .protocol import (
    ROUNDS_PER_DAY,
    TOWER_TYPES,
    Pos,
    Turn,
    Unit,
    distance,
    move_command,
)
from .state import MAX_HISTORY_FACTS, StateStore, request_fingerprint
from .tasks import TaskTurnProposal, propose_tasks
from .treasure import MAX_TREASURE_CANDIDATES, evaluate_treasure_candidates

_NEIGHBOUR_STEPS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
ECONOMY_BUDGET_FRACTION = 0.75


class DecisionEngine:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        budget_seconds: float = 4.0,
        max_search_expansions: int = 256,
    ) -> None:
        self.clock = clock
        self.budget_seconds = budget_seconds
        self.max_search_expansions = max_search_expansions
        self.state = StateStore()
        self._lock = threading.Lock()

    def decide(
        self,
        payload: dict[str, Any],
        *,
        trace_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Build the S0 probe through the B-batch state and constraint path."""
        started = self.clock()
        deadline = started + self.budget_seconds
        turn = Turn.load(payload)
        fingerprint = request_fingerprint(payload)
        allocator = ActionAllocator(turn)
        self._basic_probe(turn, allocator)
        fallback = allocator.complete_response(
            task_active=bool(payload.get("phaseTask")),
        )

        lock_budget = deadline - self.clock()
        if lock_budget <= 0 or not self._lock.acquire(timeout=lock_budget):
            self._emit_trace(
                trace_sink,
                self._decision_trace(
                    turn, payload, fallback, (), None, "budget_fallback",
                ),
            )
            return fallback
        try:
            observation = self.state.observe(turn, payload, fingerprint)
            if observation.cached_response is not None:
                self._emit_trace(trace_sink, observation.cached_trace)
                return observation.cached_response

            allocator = ActionAllocator(turn)
            state = self.state.state
            ensure_defense_layout(turn, state)
            task_role_ids = self._active_task_role_ids(turn, state)
            task_pioneer = next((
                turn.unit(role_id) for role_id in task_role_ids
            ), None)
            economy_diagnostics = []
            economy_deadline = min(
                deadline,
                started + self.budget_seconds * ECONOMY_BUDGET_FRACTION,
            )
            if turn.is_day and turn.rounds_until_night <= DUSK_POSITIONING_ROUNDS:
                fortification_builder_id = None
                state.fortification_phase = "waiting"
                state.fortification_skip_reason = "dusk_positioning"
            else:
                fortification_builder_id = prepare_fortification(
                    turn,
                    state,
                    wall_build_positions(turn),
                    reserved_role_ids=task_role_ids,
                    clock=self.clock,
                    deadline=economy_deadline,
                    max_expansions=self.max_search_expansions,
                )
            economy_candidates = propose_economy(
                turn,
                state,
                clock=self.clock,
                deadline=economy_deadline,
                max_expansions=self.max_search_expansions,
                need_wall=self._needs_fortification(turn, state),
                fortification_builder_id=fortification_builder_id,
                reserved_role_ids=task_role_ids,
                diagnostic_sink=economy_diagnostics.append,
            )
            if economy_diagnostics:
                economy_diagnostics[0]["fortification"] = (
                    fortification_diagnostic(turn, state)
                )
            critical_economy = tuple(
                candidate for candidate in economy_candidates
                if self._is_critical_economy(candidate)
            )
            ordinary_economy = tuple(
                candidate for candidate in economy_candidates
                if candidate not in critical_economy
            )
            funding_roles, funding_posts = self._funding_reservations(
                critical_economy if turn.is_day else (),
                state,
                turn,
                include_existing=turn.is_day,
            )
            day_return = (
                task_pioneer_day_return_action(
                    turn,
                    state,
                    task_pioneer,
                    clock=self.clock,
                    deadline=deadline,
                    max_expansions=self.max_search_expansions,
                    reserved_role_ids=funding_roles,
                    reserved_weapon_ids=funding_posts,
                )
                if task_pioneer is not None
                else None
            )
            if state.active_task is not None:
                state.active_task.coordination_deadline_round = (
                    turn.round_no + max(day_return[1], 0)
                    if day_return is not None and day_return[1] > 1
                    else None
                )
                state.active_task.coordination_final_requested = (
                    day_return is not None and day_return[1] <= 2
                )
            urgent_recall = (
                task_pioneer_recall_action(
                    turn,
                    task_pioneer,
                    clock=self.clock,
                    deadline=deadline,
                    max_expansions=self.max_search_expansions,
                    state=state,
                )
                if task_pioneer is not None
                else None
            )
            unavailable_for_defense = self._task_reserved_roles(
                turn, state,
            )
            defense_candidates = propose_defense(
                turn,
                state,
                clock=self.clock,
                deadline=deadline,
                max_expansions=self.max_search_expansions,
                unavailable_role_ids=(
                    task_role_ids | funding_roles
                    if urgent_recall is not None
                    else unavailable_for_defense | funding_roles
                ),
                reserved_weapon_ids=funding_posts,
            )
            emergency = propose_held_emergency(
                turn,
                state,
                defense_actions=defense_candidates,
                unavailable_role_ids=(
                    task_role_ids
                    | funding_roles
                    | frozenset(
                        candidate.proposal.actor_id
                        for candidate in critical_economy
                    )
                ),
                reserved_weapon_ids=funding_posts,
                clock=self.clock,
                deadline=deadline,
            )
            if emergency is not None:
                defense_candidates = (emergency, *defense_candidates)
            task_turn = propose_tasks(
                turn,
                state,
                clock=self.clock,
                deadline=deadline,
                max_expansions=self.max_search_expansions,
                start_guard=(
                    lambda task, stand, arrival: task_start_skip_reason(
                        turn,
                        state,
                        turn.pioneers()[0],
                        task,
                        stand,
                        arrival,
                        clock=self.clock,
                        deadline=deadline,
                        max_expansions=self.max_search_expansions,
                        reserved_role_ids=funding_roles,
                        reserved_weapon_ids=funding_posts,
                    )
                    if state.active_task is None and turn.pioneers()
                    else None
                ),
            )
            if day_return is not None and day_return[1] <= 1:
                submits_now = any(
                    candidate.proposal.command.get("action") == "submitAnswer"
                    for candidate in task_turn.actions
                )
                if not (day_return[1] == 1 and submits_now):
                    urgent_recall = day_return[0]
                    task_turn = TaskTurnProposal()
                if state.active_task is not None:
                    state.active_task.coordination_deadline_round = (
                        turn.round_no + max(day_return[1], 0)
                    )
            if urgent_recall is not None:
                defense_candidates = (urgent_recall, *defense_candidates)
            defense_first = (
                not turn.is_day
                or turn.rounds_until_night <= DUSK_POSITIONING_ROUNDS
                or bool(defense_candidates)
            )
            domains = (
                ("economy", critical_economy),
                ("defense", defense_candidates),
                ("tasks", task_turn.actions),
                ("economy", ordinary_economy),
            )
            if not defense_first:
                domains = (
                    ("tasks", task_turn.actions),
                    ("economy", economy_candidates),
                    ("defense", defense_candidates),
                )
            protected = protected_gunners(turn) if defense_first else frozenset()
            accepted = []
            rejected_economy: set[int] = set()
            blocked_new_task_by_gunner = False
            for domain, prepared_candidates in domains:
                if self.clock() >= deadline:
                    break
                candidates = prepared_candidates or ()
                for candidate in candidates:
                    if self.clock() >= deadline:
                        break
                    if (
                        domain == "tasks"
                        and not turn.phase_task
                        and candidate.proposal.actor_id in protected
                    ):
                        blocked_new_task_by_gunner = True
                        continue
                    if (
                        domain == "economy"
                        and candidate.proposal.actor_id in protected
                        and not (
                            candidate.proposal.command.get("action") == "use"
                            and candidate.proposal.command.get("name") == "Medicine"
                        )
                        and not (
                            turn.is_day
                            and candidate.plan_reason is not None
                            and candidate.plan_reason.startswith("fund:")
                            and candidate.estimated_rounds is not None
                            and candidate.estimated_rounds
                            <= turn.rounds_until_night
                        )
                        and not self._is_immediate_held_investment(candidate)
                    ):
                        rejected_economy.add(id(candidate))
                        continue
                    if allocator.try_add(candidate.proposal):
                        accepted.append((domain, candidate))
                    elif domain == "economy":
                        rejected_economy.add(id(candidate))

            if (
                task_turn.start_skip_reason is not None
                and turn.pioneers()
                and turn.pioneers()[0].unit_id in protected
            ):
                blocked_new_task_by_gunner = True

            last_valid = allocator.complete_response(
                prompt=task_turn.prompt,
                execute_cmd=task_turn.execute_cmd,
                task_active=bool(turn.phase_task),
            )
            if (
                not accepted
                and not turn.phase_task
                and not (defense_first and turn.weapons())
            ):
                last_valid = fallback

            has_task_accept = any(
                isinstance(command, dict)
                and command.get("action") == "acceptTask"
                for command in last_valid["roleCommandMap"].values()
            )
            if turn.phase_task:
                state.news_skip_reason = "task_active"
            elif last_valid["prompt"] or last_valid["executeCmd"]:
                state.news_skip_reason = "task_tool_priority"
            elif has_task_accept:
                state.news_skip_reason = "task_accept_priority"
            else:
                news_request = self.state.prepare_news_request(turn)
                if news_request is not None:
                    last_valid = copy.deepcopy(last_valid)
                    last_valid["prompt"] = news_request.prompt
                    self.state.record_news_request(news_request)

            coordination_reason = self._coordination_reason(
                accepted,
                urgent_recall,
                task_turn,
                blocked_new_task_by_gunner,
            )
            economy_planning = (
                copy.deepcopy(economy_diagnostics[0])
                if economy_diagnostics else None
            )
            if economy_planning is not None:
                economy_planning["rejectedActions"] = len(rejected_economy)
            trace = self._decision_trace(
                turn,
                payload,
                last_valid,
                accepted,
                state,
                coordination_reason,
                economy_planning,
                task_start_skip_reason=task_turn.start_skip_reason,
                session_boundary=observation.boundary,
            )
            self.state.record_response(turn, fingerprint, last_valid, trace)
            for _, candidate in accepted:
                actor_id = candidate.proposal.actor_id
                if (
                    candidate.plan_target is not None
                    and candidate.plan_reason is not None
                ):
                    self.state.set_plan(
                        actor_id,
                        candidate.plan_target,
                        candidate.plan_reason,
                        candidate.deadline_round,
                    )
                else:
                    self.state.clear_plan(actor_id)
            self._emit_trace(trace_sink, trace)
            return copy.deepcopy(last_valid)
        finally:
            self._lock.release()

    @staticmethod
    def _is_immediate_held_investment(candidate: Any) -> bool:
        return (
            candidate.proposal.command.get("action") == "use"
            and isinstance(candidate.diagnostic, dict)
            and candidate.diagnostic.get("kind") == "heldInvestment"
        )

    @staticmethod
    def _is_critical_economy(candidate: Any) -> bool:
        return (
            bool(candidate.plan_reason)
            and candidate.plan_reason.startswith("fund:")
        ) or DecisionEngine._is_immediate_held_investment(candidate)

    @staticmethod
    def _emit_trace(
        trace_sink: Callable[[dict[str, Any]], None] | None,
        trace: dict[str, Any] | None,
    ) -> None:
        if trace_sink is not None and trace is not None:
            trace_sink(copy.deepcopy(trace))

    @staticmethod
    def _coordination_reason(
        accepted: list[Any],
        urgent_recall: Any,
        task_turn: TaskTurnProposal,
        blocked_new_task_by_gunner: bool = False,
    ) -> str:
        if urgent_recall is not None and any(
            candidate == urgent_recall for _, candidate in accepted
        ):
            return "task_defense_return"
        if any(
            candidate.plan_reason
            and candidate.plan_reason.startswith("fund:")
            for _, candidate in accepted
        ):
            return "critical_funding"
        if blocked_new_task_by_gunner:
            return "gunner_hold"
        if task_turn.actions or task_turn.prompt or task_turn.execute_cmd:
            return "task_active"
        return "normal"

    @staticmethod
    def _funding_reservations(
        candidates: tuple[Any, ...],
        state: Any,
        turn: Turn,
        *,
        include_existing: bool = True,
    ) -> tuple[frozenset[int], frozenset[int]]:
        roles: set[int] = set()
        weapons: set[int] = set()
        for candidate in candidates:
            reason = candidate.plan_reason
            if not isinstance(reason, str) or reason.startswith("fund:build:"):
                continue
            parts = reason.split(":")
            if len(parts) < 3:
                continue
            try:
                weapon_id = int(parts[2])
            except ValueError:
                continue
            if weapon_id <= 0:
                continue
            roles.add(candidate.proposal.actor_id)
            weapons.add(weapon_id)
        for role_id, plan in state.plans.items() if include_existing else ():
            if (
                not plan.reason.startswith("fund:")
                or plan.reason.startswith("fund:build:")
            ):
                continue
            parts = plan.reason.split(":")
            if len(parts) < 3:
                continue
            try:
                weapon_id = int(parts[2])
            except ValueError:
                continue
            if weapon_id <= 0:
                continue
            role = turn.unit(role_id)
            weapon = turn.unit(weapon_id)
            if (
                role is None
                or weapon is None
                or weapon.kind not in TOWER_TYPES
            ):
                continue
            roles.add(role_id)
            weapons.add(weapon_id)
        return frozenset(roles), frozenset(weapons)

    @staticmethod
    def _decision_trace(
        turn: Turn,
        payload: dict[str, Any],
        response: dict[str, Any],
        accepted: tuple[Any, ...] | list[Any],
        state: Any,
        coordination_reason: str,
        economy_planning: dict[str, Any] | None = None,
        task_start_skip_reason: str | None = None,
        session_boundary: str | None = None,
    ) -> dict[str, Any]:
        task = state.active_task if state is not None else None
        remaining = None
        solver_state = "idle"
        solver_reason = None
        leave_reason = None
        task_instance_id = None
        cycle_fingerprint = None
        entry_read_attempted = False
        final_only_correction_requested = False
        if task is not None:
            task_instance_id = task.instance_id
            solver_state = task.phase
            task_deadlines = tuple(
                deadline for deadline in (
                    task.timeout_round, task.coordination_deadline_round,
                )
                if deadline is not None
            )
            if task_deadlines:
                remaining = max(0, min(task_deadlines) - turn.round_no)
            if task.solver_stopped_reason is not None:
                solver_reason = task.solver_stopped_reason
            elif task.pending_cmd_round is not None:
                solver_reason = "command_pending"
            elif task.pending_llm_round is not None:
                solver_reason = "llm_pending"
            elif task.deferred_answer is not None:
                solver_reason = "answer_ready"
            else:
                solver_reason = "solving"
            if response.get("executeCmd"):
                solver_reason = "command_requested"
            elif response.get("prompt"):
                solver_reason = "llm_requested"
            elif any(
                command.get("action") == "submitAnswer"
                for command in response.get("roleCommandMap", {}).values()
                if isinstance(command, dict)
            ):
                solver_reason = "submit_requested"
            leave_reason = task.end_reason
            cycle_fingerprint = DecisionEngine._short_fingerprint(
                task.last_cycle_fingerprint,
            )
            entry_read_attempted = task.entry_read_attempted
            final_only_correction_requested = (
                task.final_only_correction_requested
            )
        actions = []
        for domain, candidate in accepted:
            action = {
                "roleId": str(candidate.proposal.actor_id),
                "domain": domain,
                "reason": DecisionEngine._safe_action_reason(candidate),
                "estimatedRounds": candidate.estimated_rounds or 1,
                "deadlineRound": candidate.deadline_round,
            }
            if candidate.diagnostic is not None:
                if candidate.diagnostic.get("kind") == "heldEmergency":
                    action["emergency"] = copy.deepcopy(candidate.diagnostic)
                else:
                    action["economy"] = copy.deepcopy(candidate.diagnostic)
            actions.append(action)
        trace = {
            "roundNo": turn.round_no,
            "taskInstanceId": task_instance_id,
            "taskRemainingRounds": remaining,
            "solverState": solver_state,
            "solverReason": solver_reason,
            "leaveReason": leave_reason,
            "commandFingerprint": DecisionEngine._short_fingerprint(
                response.get("executeCmd"),
            ),
            "resultFingerprint": DecisionEngine._short_fingerprint(
                payload.get("lastCmdResult")
                if task is not None
                and task.last_accepted_cmd_result_round == turn.round_no
                else None,
            ),
            "cycleFingerprint": cycle_fingerprint,
            "taskEntryReadAttempted": entry_read_attempted,
            "taskFinalOnlyCorrectionRequested": (
                final_only_correction_requested
            ),
            "coordinationReason": coordination_reason,
            "taskStartSkipReason": task_start_skip_reason,
            "actions": actions,
            "newsEvidence": {
                "currentSession": state.session_index if state is not None else None,
                "sessionBoundary": session_boundary,
                "retainedFacts": len(state.history) if state is not None else 0,
                "retainedLimit": MAX_HISTORY_FACTS,
                "observations": [
                    {
                        "source": observation.fact.category,
                        "status": (
                            "new_current_session" if observation.is_new
                            else "seen_current_session"
                        ),
                        "firstObserved": {
                            "session": observation.fact.source_session,
                            "round": observation.fact.source_round,
                            "day": observation.fact.source_day,
                        },
                        "observed": {
                            "round": observation.observed_round,
                            "day": observation.observed_day,
                        },
                        "publicationTimeKnown": False,
                        "text": {
                            "value": observation.fact.value,
                            "originalLength": observation.fact.original_length,
                            "truncated": observation.fact.value_truncated,
                            "fingerprint": observation.fact.value_fingerprint,
                        },
                    }
                    for observation in (
                        state.news_observations if state is not None else ()
                    )
                ],
            },
            "newsInterpretation": {
                "dailyPolicyLimit": MAX_NEWS_CALLS_PER_DAY,
                "callsToday": (
                    state.news_calls_by_day.get(
                        (turn.round_no - 1) // ROUNDS_PER_DAY + 1, 0,
                    ) if state is not None else 0
                ),
                "remainingToday": max(
                    0,
                    MAX_NEWS_CALLS_PER_DAY - (
                        state.news_calls_by_day.get(
                            (turn.round_no - 1) // ROUNDS_PER_DAY + 1, 0,
                        ) if state is not None else 0
                    ),
                ),
                "pendingRequestId": (
                    state.pending_news_request.request_id
                    if state is not None
                    and state.pending_news_request is not None
                    else None
                ),
                "skipReason": (
                    state.news_skip_reason if state is not None else "no_state"
                ),
                "candidateCount": (
                    len(state.news_candidates) if state is not None else 0
                ),
                "candidateLimit": MAX_NEWS_CANDIDATES,
                "events": (
                    copy.deepcopy(state.news_events) if state is not None else []
                ),
            },
            "defensePressure": (
                pressure_diagnostic(turn, state)
                if state is not None else None
            ),
        }
        if economy_planning is not None:
            trace["economyPlanning"] = copy.deepcopy(economy_planning)
        treasure_candidates = (
            evaluate_treasure_candidates(
                turn,
                tuple(state.news_candidates),
                session_index=state.session_index,
            )
            if state is not None else ()
        )
        if treasure_candidates:
            trace["treasureConditions"] = {
                "candidateCount": len(treasure_candidates),
                "candidateLimit": MAX_TREASURE_CANDIDATES,
                "candidates": [
                    DecisionEngine._treasure_trace_summary(candidate)
                    for candidate in treasure_candidates
                ],
            }
        return trace

    @staticmethod
    def _treasure_trace_summary(candidate: dict[str, Any]) -> dict[str, Any]:
        summary = copy.deepcopy(candidate)
        missing = summary.pop("missingItems", {})
        summary["missingItemCount"] = sum(missing.values())
        summary["missingItemKinds"] = len(missing)
        return summary

    @staticmethod
    def _short_fingerprint(value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        if len(value) == 64 and all(char in "0123456789abcdef" for char in value):
            return value[:16]
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _safe_action_reason(candidate: Any) -> str:
        reason = candidate.plan_reason
        if isinstance(reason, str):
            if reason.startswith("task:"):
                return "task_route"
            prefix = reason.split(":", 1)[0]
            if prefix in {
                "build", "emergency", "fund", "gunner", "mine", "shop", "use",
            }:
                return prefix
            if reason in {"vendor", "s0_probe"}:
                return reason
        action = candidate.proposal.command.get("action")
        if action in {
            "acceptTask", "attack", "build", "buy", "collect", "move",
            "sell", "submitAnswer", "use",
        }:
            return str(action)
        return "unknown"

    def _task_reserved_roles(
        self,
        turn: Turn,
        state: Any,
    ) -> frozenset[int]:
        task = state.active_task
        if task is None or not turn.phase_task or task.owner_id is None:
            return frozenset()
        pioneer = turn.unit(task.owner_id)
        if pioneer is None or self._pioneer_support_is_immediate(
            turn, pioneer,
        ):
            return frozenset()
        return frozenset((pioneer.unit_id,))

    @staticmethod
    def _active_task_role_ids(turn: Turn, state: Any) -> frozenset[int]:
        task = state.active_task
        if task is None or not turn.phase_task or task.owner_id is None:
            return frozenset()
        pioneer = turn.unit(task.owner_id)
        if pioneer is None:
            return frozenset()
        return frozenset((pioneer.unit_id,))

    @staticmethod
    def _pioneer_support_is_immediate(turn: Turn, pioneer: Unit) -> bool:
        if turn.is_day:
            return False
        for weapon in turn.weapons():
            if weapon.cooldown > 0 or distance(pioneer.pos, weapon.pos) != 1:
                continue
            staffed_by_other = any(
                role.unit_id != pioneer.unit_id
                and distance(role.pos, weapon.pos) == 1
                for role in turn.controllable()
            )
            if staffed_by_other:
                continue
            if any(
                robot.health > 0
                and distance(weapon.pos, robot.pos) <= weapon.range_of_attack()
                for robot in turn.robots
            ):
                return True
        return False

    @staticmethod
    def _needs_fortification(turn: Turn, state: Any) -> bool:
        if (
            not turn.is_day
            or turn.rounds_until_night <= DUSK_POSITIONING_ROUNDS
            or len(turn.weapons()) < 2
        ):
            return False
        return bool(state.fortification_targets)

    def _basic_probe(
        self,
        turn: Turn,
        allocator: ActionAllocator,
    ) -> tuple[Unit, Pos] | None:
        for role in turn.controllable():
            blocked = turn.blocked(role)
            for dx, dy in _NEIGHBOUR_STEPS:
                target = Pos(role.pos.x + dx, role.pos.y + dy)
                if not turn.land(target) or target in blocked:
                    continue
                proposal = ActionProposal(
                    command_owner_id=role.unit_id,
                    actor_id=role.unit_id,
                    command=move_command(target),
                    destination=target,
                )
                if allocator.try_add(proposal):
                    return role, target
        return None


ENGINE = DecisionEngine()


def decide(
    payload: dict[str, Any],
    *,
    trace_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    return ENGINE.decide(payload, trace_sink=trace_sink)
