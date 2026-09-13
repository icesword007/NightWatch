import copy
import threading
import time
from typing import Any, Callable

from .actions import ActionAllocator, ActionProposal
from .defense import (
    DUSK_POSITIONING_ROUNDS,
    propose_defense,
    protected_gunners,
    task_pioneer_recall_action,
)
from .economy import propose_economy
from .protocol import Pos, Turn, Unit, distance, move_command
from .state import StateStore, request_fingerprint
from .tasks import propose_tasks

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

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
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
            return fallback
        try:
            observation = self.state.observe(turn, payload, fingerprint)
            if observation.cached_response is not None:
                return observation.cached_response

            allocator = ActionAllocator(turn)
            state = self.state.state
            task_role_ids = self._active_task_role_ids(turn, state)
            task_pioneer = next((
                turn.unit(role_id) for role_id in task_role_ids
            ), None)
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
                    task_role_ids
                    if urgent_recall is not None
                    else unavailable_for_defense
                ),
            )
            if urgent_recall is not None:
                defense_candidates = (urgent_recall, *defense_candidates)
            task_turn = propose_tasks(
                turn,
                state,
                clock=self.clock,
                deadline=deadline,
                max_expansions=self.max_search_expansions,
            )
            defense_first = (
                not turn.is_day
                or turn.rounds_until_night <= DUSK_POSITIONING_ROUNDS
                or bool(defense_candidates)
            )
            domains = (
                ("defense", defense_candidates),
                ("tasks", task_turn.actions),
                ("economy", None),
            )
            if not defense_first:
                domains = (
                    ("tasks", task_turn.actions),
                    ("economy", None),
                    ("defense", defense_candidates),
                )
            protected = protected_gunners(turn) if defense_first else frozenset()
            accepted = []
            for domain, prepared_candidates in domains:
                if self.clock() >= deadline:
                    break
                if domain == "economy":
                    candidates = propose_economy(
                        turn,
                        state,
                        clock=self.clock,
                        deadline=deadline,
                        max_expansions=self.max_search_expansions,
                        need_wall=self._needs_wall_trial(turn, state),
                        reserved_role_ids=task_role_ids,
                    )
                else:
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
                    ):
                        continue
                    if allocator.try_add(candidate.proposal):
                        accepted.append(candidate)

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

            self.state.record_response(turn, fingerprint, last_valid)
            for candidate in accepted:
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
            return copy.deepcopy(last_valid)
        finally:
            self._lock.release()

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


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    return ENGINE.decide(payload)
