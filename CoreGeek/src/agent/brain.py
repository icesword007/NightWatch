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
    task_pioneer_day_return_action,
    task_pioneer_recall_action,
)
from .economy import propose_economy
from .protocol import Pos, Turn, Unit, distance, move_command
from .state import StateStore, request_fingerprint
from .tasks import TaskTurnProposal, propose_tasks

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
            task_role_ids = self._active_task_role_ids(turn, state)
            task_pioneer = next((
                turn.unit(role_id) for role_id in task_role_ids
            ), None)
            economy_candidates = propose_economy(
                turn,
                state,
                clock=self.clock,
                deadline=deadline,
                max_expansions=self.max_search_expansions,
                need_wall=self._needs_wall_trial(turn, state),
                reserved_role_ids=task_role_ids,
            )
            critical_economy = tuple(
                candidate for candidate in economy_candidates
                if candidate.plan_reason
                and candidate.plan_reason.startswith("fund:")
            )
            ordinary_economy = tuple(
                candidate for candidate in economy_candidates
                if candidate not in critical_economy
            )
            funding_roles, funding_posts = self._funding_reservations(
                critical_economy if turn.is_day else (),
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
            task_turn = propose_tasks(
                turn,
                state,
                clock=self.clock,
                deadline=deadline,
                max_expansions=self.max_search_expansions,
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
            for domain, prepared_candidates in domains:
                if self.clock() >= deadline:
                    break
                candidates = prepared_candidates or ()
                for candidate in candidates:
                    if self.clock() >= deadline:
                        break
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
                    ):
                        continue
                    if allocator.try_add(candidate.proposal):
                        accepted.append((domain, candidate))

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

            coordination_reason = self._coordination_reason(
                accepted, urgent_recall, task_turn,
            )
            trace = self._decision_trace(
                turn,
                payload,
                last_valid,
                accepted,
                state,
                coordination_reason,
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
        if task_turn.actions or task_turn.prompt or task_turn.execute_cmd:
            return "task_active"
        return "normal"

    @staticmethod
    def _funding_reservations(
        candidates: tuple[Any, ...],
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
        return frozenset(roles), frozenset(weapons)

    @staticmethod
    def _decision_trace(
        turn: Turn,
        payload: dict[str, Any],
        response: dict[str, Any],
        accepted: tuple[Any, ...] | list[Any],
        state: Any,
        coordination_reason: str,
    ) -> dict[str, Any]:
        task = state.active_task if state is not None else None
        remaining = None
        solver_state = "idle"
        solver_reason = None
        leave_reason = None
        task_instance_id = None
        cycle_fingerprint = None
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
        actions = []
        for domain, candidate in accepted:
            actions.append({
                "roleId": str(candidate.proposal.actor_id),
                "domain": domain,
                "reason": DecisionEngine._safe_action_reason(candidate),
                "estimatedRounds": candidate.estimated_rounds or 1,
                "deadlineRound": candidate.deadline_round,
            })
        return {
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
            "coordinationReason": coordination_reason,
            "actions": actions,
        }

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
            if prefix in {"build", "fund", "gunner", "mine", "shop", "use"}:
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
    def _needs_wall_trial(turn: Turn, state: Any) -> bool:
        if (
            not turn.is_day
            or turn.rounds_until_night <= DUSK_POSITIONING_ROUNDS
            or len(turn.weapons()) < 3
            or turn.walls()
            or state.wall_trial_started
            or any(plan.reason == "build:wall" for plan in state.plans.values())
        ):
            return False
        if any(
            completed.pending.action == "build"
            and completed.pending.name == "wall"
            and completed.pending.source_session == state.session_index
            for completed in state.action_history
        ):
            return False
        return any("stone" in worker.backpack for worker in turn.workers())

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
