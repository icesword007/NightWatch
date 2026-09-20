import time
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Callable

from .actions import ActionProposal, PlannedAction
from .defense import DAY_WORK_RETURN_MARGIN
from .fortification import (
    MAX_WALL_TARGETS,
    fortification_stone_target,
    ordered_wall_targets,
    remaining_wall_targets,
)
from .grid import PathResult, next_step
from .layout import ensure_defense_layout
from .protocol import (
    DUSK_POSITIONING_ROUNDS,
    PIONEER,
    ROUNDS_PER_DAY,
    STATION,
    TOWER_TYPES,
    WALL,
    Pos,
    Turn,
    Unit,
    build_command,
    buy_command,
    collect_command,
    distance,
    move_command,
    sell_command,
    use_command,
)
from .state import BaseReserve, SessionState

MINERALS = ("stone", "iron", "copper")
MAX_WEAPONS = 3
MAX_MINE_CANDIDATES = 16
MINE_SWITCH_MARGIN_PERCENT = 10
MAX_ECONOMY_PATH_SEARCHES = 1600
INVESTMENT_ITEM_PREFIXES = (
    "StationUpgradeVoucher",
    "WeaponUpgradeVoucher",
    "WallUpgradeVoucher",
)
EMERGENCY_MEDICINE_HEALTH = 40
ROBOT_ATTACK_RANGE = 3


@dataclass(frozen=True, slots=True)
class FundingRoute:
    rounds: int
    vendor: Pos | None
    vendor_stand: Pos | None
    shop: Pos | None
    shop_stand: Pos | None
    use_target_id: int | None
    use_stand: Pos
    post_weapon_id: int
    post_stand: Pos


@dataclass(frozen=True, slots=True)
class ProcurementBatch:
    item: str
    post_weapon_id: int
    primary_target_id: int
    mineral: str
    mine: Pos
    goal_count: int
    quote: int
    item_price: int
    reserved_target_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BuildingFundingRoute:
    rounds: int
    vendor: Pos
    vendor_stand: Pos
    build_stand: Pos


@dataclass(frozen=True, slots=True)
class MiningOpportunity:
    target: Pos
    stand: Pos
    value: int
    rounds: int


@dataclass(frozen=True, slots=True)
class JointFundingRoute:
    buyer_id: int
    contributor_id: int
    item: str
    use_target_id: int | None
    buyer_post_id: int
    contributor_post_id: int
    buyer_sales: tuple[tuple[str, int], ...]
    contributor_sales: tuple[tuple[str, int], ...]
    buyer_vendor: Pos | None
    buyer_vendor_stand: Pos | None
    contributor_vendor: Pos | None
    contributor_vendor_stand: Pos | None
    purchase: FundingRoute
    contributor_post_stand: Pos
    buyer_rounds: int
    contributor_rounds: int


@dataclass(slots=True)
class RouteSearchContext:
    routes: dict[
        tuple[int, int, Pos, Pos, int],
        tuple[tuple[Pos, int], ...],
    ]
    paths: dict[tuple[int, int, Pos, Pos, int], PathResult] = field(
        default_factory=dict,
    )
    turns: dict[int, Turn] = field(default_factory=dict)
    path_searches: int = 0
    cache_hits: int = 0
    path_computations: int = 0
    path_cache_hits: int = 0
    path_expansions: int = 0
    truncated_reason: str | None = None
    joint_status: str | None = None
    joint_blocker: str | None = None
    tower_status: str | None = None
    investment_targets: set[int] = field(default_factory=set)
    investment_owners: dict[int, int] = field(default_factory=dict)
    wall_upgrade_targets: tuple[Pos, ...] = ()
    procurement_blocked_role_ids: set[int] = field(default_factory=set)


_ROUTE_SEARCH_CONTEXT: ContextVar[RouteSearchContext | None] = ContextVar(
    "economy_route_search_context",
    default=None,
)


def propose_economy(
    turn: Turn,
    state: SessionState,
    *,
    clock: Callable[[], float] = time.monotonic,
    deadline: float,
    max_expansions: int,
    need_wall: bool = False,
    fortification_builder_id: int | None = None,
    reserved_role_ids: frozenset[int] = frozenset(),
    night_cleared: bool = False,
    diagnostic_sink: Callable[[dict], None] | None = None,
) -> tuple[PlannedAction, ...]:
    context = RouteSearchContext({})
    token = _ROUTE_SEARCH_CONTEXT.set(context)
    try:
        result = _propose_economy(
            turn,
            state,
            clock=clock,
            deadline=deadline,
            max_expansions=max_expansions,
            need_wall=need_wall,
            fortification_builder_id=fortification_builder_id,
            reserved_role_ids=reserved_role_ids,
            night_cleared=night_cleared,
        )
        if diagnostic_sink is not None:
            held = next((
                candidate for candidate in result
                if isinstance(candidate.diagnostic, dict)
                and candidate.diagnostic.get("kind") == "heldInvestment"
            ), None)
            held_in_backpack = any(
                _is_investment_item(item)
                and _item_target(turn, item) is not None
                for role in turn.controllable()
                for item in role.backpack
            )
            investment = next((
                candidate for candidate in result
                if candidate.plan_reason is not None
                and candidate.plan_reason.startswith(("fund:", "use:", "batch:"))
                and _is_investment_item(
                    candidate.plan_reason.split(":", 2)[1]
                )
                and _plan_use_target_id(candidate) is not None
            ), None)
            investment_target = None
            if investment is not None:
                target_id = _plan_use_target_id(investment)
                previous = state.plans.get(investment.proposal.actor_id)
                investment_target = {
                    "item": investment.plan_reason.split(":", 2)[1],
                    "targetId": str(target_id),
                    "selection": (
                        "continued"
                        if _plan_use_target_id(previous) == target_id
                        else "selected"
                    ),
                }
            diagnostic_sink({
                "pathSearches": context.path_searches,
                "cacheHits": context.cache_hits,
                "pathComputations": context.path_computations,
                "pathCacheHits": context.path_cache_hits,
                "pathExpansions": context.path_expansions,
                "truncatedReason": context.truncated_reason,
                "jointStatus": context.joint_status,
                "towerPriority": context.tower_status,
                "heldInvestment": (
                    held.proposal.command.get("action")
                    if held is not None
                    else "pending" if held_in_backpack else None
                ),
                "investmentTarget": investment_target,
            })
        return result
    finally:
        _ROUTE_SEARCH_CONTEXT.reset(token)


def _propose_economy(
    turn: Turn,
    state: SessionState,
    *,
    clock: Callable[[], float] = time.monotonic,
    deadline: float,
    max_expansions: int,
    need_wall: bool = False,
    fortification_builder_id: int | None = None,
    reserved_role_ids: frozenset[int] = frozenset(),
    night_cleared: bool = False,
) -> tuple[PlannedAction, ...]:
    context = _ROUTE_SEARCH_CONTEXT.get()
    current_day = (turn.round_no - 1) // ROUNDS_PER_DAY + 1
    state.procurement_blocked_days = {
        role_id: day for role_id, day in state.procurement_blocked_days.items()
        if day == current_day
    }
    if context is not None:
        context.procurement_blocked_role_ids.update(
            state.procurement_blocked_days,
        )
    for role_id in reserved_role_ids:
        plan = state.plans.get(role_id)
        if plan is not None and plan.reason.startswith("batch:"):
            state.plans.pop(role_id, None)
            state.procurement_blocked_days[role_id] = current_day
            if context is not None:
                context.procurement_blocked_role_ids.add(role_id)
    if context is not None:
        contenders: dict[int, list[tuple[bool, int]]] = {}
        live = {role.unit_id: role for role in turn.controllable()}
        for role_id, plan in state.plans.items():
            if (
                plan.source_session != state.session_index
                or role_id in reserved_role_ids
                or plan.deadline_round is not None
                and turn.round_no > plan.deadline_round
            ):
                continue
            parts = plan.reason.split(":")
            joint = _joint_plan(plan.reason)
            owner_id = joint[3] if joint is not None else role_id
            if joint is not None and role_id != owner_id:
                continue
            role = live.get(owner_id)
            target_ids = _plan_use_target_ids(plan)
            if not target_ids or role is None or len(parts) < 2:
                continue
            item = parts[1]
            if plan.reason.startswith("use:") and item not in role.backpack:
                continue
            for target_id in target_ids:
                target = turn.unit(target_id)
                if target is None or not _item_matches_target(item, target):
                    continue
                contenders.setdefault(target_id, []).append(
                    (item not in role.backpack, owner_id)
                )
        context.investment_owners = {
            target_id: min(options)[1]
            for target_id, options in contenders.items()
        }
        context.investment_targets.update(context.investment_owners)
    ensure_defense_layout(turn, state)
    if context is not None:
        context.wall_upgrade_targets = state.layout_wall_targets
    candidates: list[PlannedAction] = []
    maintained_roles: set[int] = set()
    claimed_build_targets: set[Pos] = set()
    claimed_tower_types: Counter[str] = Counter()
    claimed_gold = 0
    failed_builds = {
        completed.pending.target
        for completed in state.action_history
        if completed.pending.action == "build"
        and completed.success is False
        and completed.pending.target is not None
        and completed.pending.source_session == state.session_index
    }
    unconfirmed_builds = {
        completed.pending.target
        for completed in state.action_history
        if completed.pending.action == "build"
        and completed.success is None
        and completed.pending.target is not None
        and completed.pending.source_session == state.session_index
    }
    failed_mines = {
        completed.pending.target
        for completed in state.action_history
        if completed.pending.action == "collect"
        and completed.success is False
        and completed.pending.target is not None
        and completed.pending.source_session == state.session_index
    }
    if context is not None:
        context.procurement_blocked_role_ids.update(
            completed.pending.actor_id
            for completed in state.action_history
            if completed.pending.action == "collect"
            and completed.success is False
            and completed.pending.source_session == state.session_index
        )
    direct_wall_request = (
        need_wall
        and not state.fortification_targets
        and state.fortification_builder_id is None
    )
    if need_wall and not state.fortification_targets:
        state.fortification_targets = (
            state.layout_wall_targets[:MAX_WALL_TARGETS]
            or ordered_wall_targets(turn, wall_build_positions(turn))
        )
    if (
        direct_wall_request
        and fortification_builder_id is None
    ):
        fortification_builder_id = next(
            (worker.unit_id for worker in turn.workers()), None,
        )

    urgent_purchase: PlannedAction | None = None
    urgent_walls = (
        tuple(wall for wall in turn.walls() if _wall_repair_urgent(turn, wall))
        if turn.is_day and len(turn.weapons()) < MAX_WEAPONS
        else ()
    )
    urgent_held: dict[int, PlannedAction] = {}
    covered_urgent_ids: set[int] = set()
    for wall in urgent_walls:
        previous_owner = (
            context.investment_owners.get(wall.unit_id)
            if context is not None else None
        )
        if context is not None:
            context.investment_owners.pop(wall.unit_id, None)
            context.investment_targets.discard(wall.unit_id)
        held_options: list[tuple[tuple[int, int, int, int], PlannedAction]] = []
        for worker in sorted(turn.workers(), key=lambda entry: entry.unit_id):
            if (
                worker.unit_id in reserved_role_ids
                or worker.unit_id in urgent_held
                or "WallFixer" not in worker.backpack
                or "Medicine" in worker.backpack
                and worker.health < _max_role_health(worker)
            ):
                continue
            held = _maintenance_action(
                turn, worker, clock, deadline, max_expansions,
                preferred_use_target_id=wall.unit_id,
            )
            if (
                _is_held_investment_action(held)
                and _plan_use_target_id(held) == wall.unit_id
            ):
                held_options.append((
                    (
                        int(held.proposal.command.get("action") != "use"),
                        int(worker.unit_id != previous_owner),
                        held.estimated_rounds or 0,
                        worker.unit_id,
                    ),
                    held,
                ))
        if held_options:
            held = min(held_options, key=lambda option: option[0])[1]
            role_id = held.proposal.actor_id
            urgent_held[role_id] = held
            covered_urgent_ids.add(wall.unit_id)
            if context is not None:
                context.investment_owners[wall.unit_id] = role_id
                context.investment_targets.add(wall.unit_id)
    if (
        turn.is_day
        and len(turn.weapons()) < MAX_WEAPONS
        and any(wall.unit_id not in covered_urgent_ids for wall in urgent_walls)
        and "WallFixer" in turn.weapon_prices
    ):
        for worker in sorted(turn.workers(), key=lambda entry: entry.unit_id):
            if (
                worker.unit_id in reserved_role_ids
                or "WallFixer" in worker.backpack
                or "Medicine" in worker.backpack
                and worker.health < _max_role_health(worker)
            ):
                continue
            uncovered = next((
                wall for wall in urgent_walls
                if wall.unit_id not in covered_urgent_ids
            ), None)
            if uncovered is None:
                break
            urgent_purchase = _funding_action(
                turn, worker, "WallFixer",
                turn.round_no + turn.rounds_until_night - 1,
                clock, deadline, max_expansions,
                preferred_use_target_id=uncovered.unit_id,
            )
            if urgent_purchase is not None:
                claimed_gold += min(turn.gold, turn.weapon_prices["WallFixer"])
                break
    tower_candidates: dict[int, PlannedAction] = {}
    urgent_tower_workers: set[int] = set()
    if turn.is_day and len(turn.weapons()) < MAX_WEAPONS:
        tower_positions = (
            state.layout_tower_targets or weapon_build_positions(turn)
        )
        tower_workers = sorted(turn.workers(), key=lambda worker: (
            not (
                (plan := state.plans.get(worker.unit_id)) is not None
                and plan.reason.startswith(("build:rocket", "fund:build:rocket"))
            ),
            need_wall and worker.unit_id == fortification_builder_id,
            min(
                (distance(worker.pos, pos) for pos in tower_positions),
                default=10**9,
            ),
            worker.unit_id,
        ))
        for worker in tower_workers:
            if (
                clock() >= deadline
                or worker.unit_id in reserved_role_ids
                or worker.unit_id in urgent_held
                or urgent_purchase is not None
                and worker.unit_id == urgent_purchase.proposal.actor_id
            ):
                continue
            plan = state.plans.get(worker.unit_id)
            maintenance = _maintenance_action(
                turn, worker, clock, deadline, max_expansions,
                preferred_use_target_id=_plan_use_target_id(plan),
            )
            if _urgent_maintenance_candidate(turn, maintenance):
                urgent_tower_workers.add(worker.unit_id)
                continue
            candidate = None
            if plan is not None and plan.reason.startswith((
                "build:rocket", "fund:build:rocket",
            )):
                candidate = _continue_plan(
                    turn, state, worker, failed_builds, failed_mines,
                    clock, deadline, max_expansions,
                )
            if candidate is None:
                candidate = _tower_action(
                    turn, state, worker,
                    failed_builds | unconfirmed_builds | claimed_build_targets,
                    claimed_tower_types, turn.gold - claimed_gold,
                    clock, deadline, max_expansions,
                )
            if candidate is None:
                continue
            tower_candidates[worker.unit_id] = candidate
            if candidate.plan_target is not None:
                claimed_build_targets.add(candidate.plan_target)
            claimed_tower_types["rocket"] += 1
            claimed_gold += 25
    if context is not None and tower_candidates:
        paused_owners = set(tower_candidates) | {
            role_id for role_id, plan in state.plans.items()
            if plan.reason.startswith(("fund:", "batch:"))
            and not plan.reason.startswith("fund:build:")
        }
        context.investment_owners = {}
        for target_id, options in contenders.items():
            available = tuple(
                option for option in options
                if option[1] not in paused_owners
            )
            if available:
                context.investment_owners[target_id] = min(available)[1]
        context.investment_targets = set(context.investment_owners)
        for role_id, held in urgent_held.items():
            target_id = _plan_use_target_id(held)
            if target_id is not None:
                context.investment_owners[target_id] = role_id
                context.investment_targets.add(target_id)
    if context is not None:
        if tower_candidates:
            context.tower_status = "committed"
        elif len(turn.weapons()) >= MAX_WEAPONS:
            context.tower_status = "complete"
        elif not turn.is_day:
            context.tower_status = "night"
        elif turn.station() is None:
            context.tower_status = "station_missing"
        elif urgent_purchase is not None:
            context.tower_status = "emergency_maintenance"
        elif not any(
            worker.unit_id not in reserved_role_ids for worker in turn.workers()
        ):
            context.tower_status = "worker_unavailable"
        elif all(
            worker.unit_id in reserved_role_ids
            or worker.unit_id in urgent_tower_workers
            for worker in turn.workers()
        ):
            context.tower_status = "emergency_maintenance"
        elif context.truncated_reason is not None or clock() >= deadline:
            context.tower_status = "planning_budget_unknown"
        elif not any(
            turn.gold + sum(
                turn.vendor_prices.get(item, 0)
                for item in worker.backpack if item in MINERALS
            ) >= 25
            for worker in turn.workers()
            if worker.unit_id not in reserved_role_ids
        ):
            context.tower_status = "funds_insufficient"
        else:
            context.tower_status = "no_complete_route_or_window"

    reserve = state.base_reserve
    if context is not None and reserve is not None:
        context.investment_owners[reserve.station_id] = reserve.holder_id
        context.investment_targets.add(reserve.station_id)
    for role in sorted(turn.controllable(), key=lambda entry: entry.unit_id):
        if clock() >= deadline:
            break
        plan = state.plans.get(role.unit_id)
        joint_plan = _joint_plan(plan.reason) if plan is not None else None
        candidate = urgent_held.get(role.unit_id)
        if (
            candidate is None
            and urgent_purchase is not None
            and role.unit_id == urgent_purchase.proposal.actor_id
        ):
            candidate = urgent_purchase
        if candidate is None:
            candidate = tower_candidates.get(role.unit_id)
        reserve = state.base_reserve
        reserved_item = (
            reserve.item if reserve is not None
            and role.unit_id == reserve.holder_id
            and reserve.item in role.backpack else None
        )
        if candidate is None:
            candidate = _maintenance_action(
                turn, role, clock, deadline, max_expansions,
                preferred_use_target_id=_plan_use_target_id(plan),
                skip_item=reserved_item,
            )
            if plan is not None and plan.reason.startswith(("fund:", "batch:")):
                self_care = (
                    candidate is not None
                    and candidate.proposal.command.get("action") == "use"
                    and candidate.proposal.command.get("name") == "Medicine"
                )
                immediate_investment = (
                    joint_plan is None
                    and _is_immediate_held_investment_action(candidate)
                )
                if not self_care and not immediate_investment:
                    candidate = None
            if (
                role.unit_id in reserved_role_ids
                and candidate is not None
                and not (
                    candidate.proposal.command.get("action") == "use"
                    and candidate.proposal.command.get("name") == "Medicine"
                )
            ):
                candidate = None
        if candidate is not None:
            candidates.append(candidate)
            _reserve_investment_target(candidate, _ROUTE_SEARCH_CONTEXT.get())
            maintained_roles.add(role.unit_id)

    joint_candidates, joint_roles, joint_cancellations = _joint_funding_actions(
        turn,
        state,
        excluded_role_ids=(
            frozenset(maintained_roles)
            | reserved_role_ids
            | (
                frozenset((fortification_builder_id,))
                if fortification_builder_id is not None
                else frozenset()
            )
        ),
        clock=clock,
        deadline=deadline,
        max_expansions=max_expansions,
    )
    candidates.extend(joint_candidates)
    for candidate in joint_candidates:
        _reserve_investment_target(candidate, context)
    maintained_roles.update(joint_roles)
    for worker in sorted(turn.workers(), key=lambda entry: entry.unit_id):
        if (
            worker.unit_id in maintained_roles
            or worker.unit_id not in joint_cancellations
        ):
            continue
        plan = state.plans.get(worker.unit_id)
        if plan is None or _joint_plan(plan.reason) is None:
            continue
        held = _maintenance_action(
            turn, worker, clock, deadline, max_expansions,
            preferred_use_target_id=_plan_use_target_id(plan),
        )
        if not _is_held_investment_action(held):
            continue
        diagnostic = dict(held.diagnostic or {})
        diagnostic["jointCancelReason"] = joint_cancellations[worker.unit_id]
        fallback = replace(
            held,
            estimated_rounds=(
                1
                if held.proposal.command.get("action") == "use"
                else held.estimated_rounds
            ),
            diagnostic=diagnostic,
        )
        candidates.append(fallback)
        _reserve_investment_target(fallback, context)
        maintained_roles.add(worker.unit_id)
        joint_cancellations.pop(worker.unit_id)
    for role_id, plan in tuple(state.plans.items()):
        if _joint_plan(plan.reason) is not None and role_id not in joint_roles:
            state.plans.pop(role_id)

    for worker in sorted(turn.workers(), key=lambda entry: entry.unit_id):
        if clock() >= deadline:
            break
        if worker.unit_id in maintained_roles:
            continue
        if worker.unit_id in reserved_role_ids:
            continue
        candidate = None
        if (
            need_wall
            and worker.unit_id == fortification_builder_id
        ):
            candidate = _wall_action(
                turn,
                state,
                worker,
                failed_builds | claimed_build_targets | {
                    plan.target for role_id, plan in state.plans.items()
                    if role_id != worker.unit_id
                },
                failed_mines,
                clock, deadline, max_expansions,
            )
        current_plan = state.plans.get(worker.unit_id)
        batch_invalidated = False
        funding_rechecked = False
        if (
            candidate is None
            and current_plan is not None
            and current_plan.reason.startswith("mine:")
        ):
            funding_rechecked = True
            candidate = _single_funding_action(
                turn,
                worker,
                turn.gold - claimed_gold,
                clock,
                deadline,
                max_expansions,
            )
        managed_wall_plan = (
            current_plan is not None
            and current_plan.reason == "build:wall"
            and state.fortification_initialized
        )
        if managed_wall_plan:
            if candidate is None:
                state.plans.pop(worker.unit_id, None)
        elif candidate is None:
            if not (
                tower_candidates
                and current_plan is not None
                and current_plan.reason.startswith(("fund:", "batch:"))
                and not current_plan.reason.startswith("fund:build:")
            ):
                candidate = _continue_plan(
                    turn, state, worker, failed_builds, failed_mines,
                    clock, deadline, max_expansions,
                )
            if (
                candidate is None
                and current_plan is not None
                and current_plan.reason.startswith("batch:")
            ):
                state.plans.pop(worker.unit_id, None)
                state.procurement_blocked_days[worker.unit_id] = current_day
                if context is not None:
                    context.procurement_blocked_role_ids.add(worker.unit_id)
                batch_invalidated = True
        if candidate is None:
            candidate = _tower_action(
                turn,
                state,
                worker,
                failed_builds | unconfirmed_builds | claimed_build_targets,
                claimed_tower_types,
                turn.gold - claimed_gold,
                clock,
                deadline,
                max_expansions,
            )
        if candidate is None:
            candidate = _trade_or_mine(
                turn, worker, turn.gold - claimed_gold,
                clock, deadline, max_expansions, failed_mines,
                check_funding=not funding_rechecked,
                night_cleared=night_cleared,
                allow_procurement_batch=not batch_invalidated,
            )
        if candidate is not None and worker.unit_id in joint_cancellations:
            diagnostic = dict(candidate.diagnostic or {})
            diagnostic["jointCancelReason"] = joint_cancellations[worker.unit_id]
            candidate = replace(candidate, diagnostic=diagnostic)
        if candidate is not None:
            candidates.append(candidate)
            _reserve_investment_target(candidate, _ROUTE_SEARCH_CONTEXT.get())
            if (
                candidate.plan_reason
                and (
                    candidate.plan_reason.startswith("build:")
                    or candidate.plan_reason.startswith("fund:build:")
                )
                and candidate.plan_target is not None
            ):
                claimed_build_targets.add(candidate.plan_target)
                claimed_tower_types[
                    candidate.plan_reason.rsplit(":", 1)[1]
                ] += 1
            action = candidate.proposal.command["action"]
            if action == "build" and candidate.proposal.command.get("name") in TOWER_TYPES:
                claimed_gold += 25
            elif action == "buy":
                name = candidate.proposal.command["name"]
                claimed_gold += turn.weapon_prices[name]
    if state.base_blocked_day == current_day:
        station = turn.station()
        if station is not None and station.level in (1, 2):
            item = f"StationUpgradeVoucher{station.level}"
            candidates = [
                candidate for candidate in candidates
                if candidate.proposal.actor_id != state.base_blocked_holder_id
                or not (
                    candidate.plan_reason is not None
                    and candidate.plan_reason.startswith((
                        f"fund:{item}:", f"use:{item}:",
                    ))
                    or candidate.proposal.command.get("name") == item
                )
            ]
    return tuple(candidates)


def prepare_base_reserve(
    turn: Turn, state: SessionState,
    unavailable_role_ids: frozenset[int],
    clock: Callable[[], float], deadline: float, max_expansions: int,
) -> None:
    station = turn.station()
    reserve = state.base_reserve
    if reserve is not None:
        holder = turn.unit(reserve.holder_id)
        if reserve.holder_id in unavailable_role_ids:
            state.base_reserve = None
            state.base_reserve_event = "holder_task_reserved"
            return
        plan = state.plans.get(reserve.holder_id)
        if (
            holder is not None and reserve.item in holder.backpack
            and plan is not None
            and plan.reason.startswith(f"fund:{reserve.item}:")
        ):
            state.plans.pop(reserve.holder_id)
        if (
            turn.is_day and station is not None
            and station.health < _max_building_health(station)
        ):
            state.base_reserve = None
            state.base_reserve_event = "day_damage_released"
        return
    if (
        station is None or station.level not in (1, 2)
        or len(turn.weapons()) < MAX_WEAPONS or not turn.is_day
        or turn.rounds_until_night <= DUSK_POSITIONING_ROUNDS
    ):
        return
    item = f"StationUpgradeVoucher{station.level}"
    holders = tuple(
        worker for worker in turn.workers()
        if worker.unit_id not in unavailable_role_ids
        and not (
            state.base_blocked_day == (turn.round_no - 1) // ROUNDS_PER_DAY + 1
            and worker.unit_id == state.base_blocked_holder_id
        )
        and worker.unit_id not in state.plans
        and item in worker.backpack
        and (
            station.health == _max_building_health(station)
                and not any(
                    robot.target_team == turn.team_type
                    and min(distance(robot.pos, cell)
                            for cell in turn.footprint(station)) <= ROBOT_ATTACK_RANGE
                    for robot in turn.robots
                )
                and (route := _held_item_route(
                    turn, worker, item, clock, deadline, max_expansions,
                    preferred_use_target_id=station.unit_id,
                )) is not None
                and route.rounds <= turn.rounds_until_night
        )
    )
    if holders:
        holder = min(holders, key=lambda role: (
            distance(role.pos, station.pos), role.unit_id,
        ))
        state.base_reserve = BaseReserve(
            holder.unit_id, station.unit_id, station.level, item,
        )
        state.base_reserve_event = "held"
        state.base_reserve_reason = "existing_voucher"


def reserve_purchase_candidate(turn: Turn, state: SessionState,
                               candidates: tuple[PlannedAction, ...],
                               unavailable_role_ids: frozenset[int]) -> tuple[PlannedAction, ...]:
    station = turn.station()
    reason = None
    if not turn.is_day:
        reason = "night"
    elif turn.rounds_until_night <= DUSK_POSITIONING_ROUNDS:
        reason = "dusk"
    elif station is None:
        reason = "station_missing"
    elif station.level not in (1, 2):
        reason = "station_level_ineligible"
    elif len(turn.weapons()) < MAX_WEAPONS:
        reason = "towers_incomplete"
    elif station.health != _max_building_health(station):
        reason = (
            "station_damaged" if station.health < _max_building_health(station)
            else "station_health_ineligible"
        )
    elif state.base_reserve is not None:
        reason = "held_for_night"
    elif any(min(distance(robot.pos, cell)
                 for cell in turn.footprint(station)) <= ROBOT_ATTACK_RANGE
             and robot.target_team == turn.team_type for robot in turn.robots):
        reason = "threat_near_station"
    elif any(
        candidate.plan_reason and candidate.plan_reason.startswith("fund:")
        and not candidate.plan_reason.startswith(
            f"fund:StationUpgradeVoucher{station.level}:"
        ) for candidate in candidates
    ):
        reason = "other_funding_priority"
    if reason is not None:
        state.base_reserve_reason = reason
        return candidates
    item = f"StationUpgradeVoucher{station.level}"
    price = turn.weapon_prices.get(item)
    if price is None or turn.gold < price:
        state.base_reserve_reason = "cash_insufficient"
        return candidates
    result = list(candidates)
    for index, candidate in enumerate(result):
        if (
            candidate.proposal.actor_id not in unavailable_role_ids
            and not (
                state.base_blocked_day == (turn.round_no - 1) // ROUNDS_PER_DAY + 1
                and candidate.proposal.actor_id == state.base_blocked_holder_id
            )
            and candidate.plan_reason is not None
            and candidate.plan_reason.startswith(f"fund:{item}:")
            and candidate.estimated_rounds is not None
            and candidate.estimated_rounds <= turn.rounds_until_night
        ):
            diagnostic = dict(candidate.diagnostic or {})
            diagnostic.update({
                "kind": "baseReserve", "phase": "preparing",
                "stationId": str(station.unit_id), "level": station.level,
            })
            result[index] = replace(candidate, diagnostic=diagnostic)
            state.base_reserve_reason = "preparing"
            break
    else:
        state.base_reserve_reason = "route_or_priority_unavailable"
    return tuple(result)


def propose_base_upgrade(
    turn: Turn, state: SessionState, *,
    unavailable_role_ids: frozenset[int],
    protected_role_ids: frozenset[int],
    clock: Callable[[], float], deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    reserve = state.base_reserve
    station = turn.station()
    if turn.is_day:
        return None
    state.base_reserve_reason = "not_reserved"
    if reserve is None or station is None:
        return None
    if clock() >= deadline:
        state.base_reserve_reason = "planning_deadline"
        return None
    holder = turn.unit(reserve.holder_id)
    if station.unit_id != reserve.station_id or station.level != reserve.station_level:
        state.base_reserve_reason = "station_changed"
        return None
    if holder is None or holder.unit_id in unavailable_role_ids:
        state.base_reserve_reason = "holder_unavailable"
        return None
    if reserve.item not in holder.backpack:
        state.base_reserve_reason = "voucher_not_held"
        return None
    if "Medicine" in holder.backpack and holder.health < _max_role_health(holder):
        state.base_reserve_reason = "medicine_first"
        return None
    adjacent = distance(holder.pos, station.pos) == 1
    route = (holder.pos, 0) if adjacent else _best_adjacent_route(
        turn, holder, station.pos, clock, deadline, max_expansions,
    )
    if route is None:
        state.base_reserve_reason = "no_legal_route"
        return None
    stand, travel = route
    rounds = travel + 1
    visible_rate = sum(
        robot.attack_power for robot in turn.robots
        if robot.target_team == turn.team_type
        and min(distance(robot.pos, cell)
                for cell in turn.footprint(station)) <= ROBOT_ATTACK_RANGE
    )
    observed_rate = max(state.base_recent_drops, default=0)
    rate = max(visible_rate, observed_rate)
    uncertain_damage = (
        rate <= 0 and adjacent
        and station.health < _max_building_health(station)
    )
    if rate <= 0:
        if not uncertain_damage:
            state.base_reserve_reason = "no_damage_estimate"
            return None
    elif station.health > rate * (rounds + 2):
        state.base_reserve_reason = "margin_sufficient"
        return None
    if not adjacent and holder.unit_id in protected_role_ids:
        state.base_reserve_reason = "gunner_hold"
        return None
    if not adjacent and station.health <= rate * rounds:
        state.base_reserve_reason = "too_late_to_approach"
        return None
    proposal = (
        ActionProposal(
            holder.unit_id, holder.unit_id,
            use_command(reserve.item, station.pos),
            item_costs=(reserve.item,),
        ) if adjacent else _move_to_stand(
            turn, holder, stand, clock, deadline, max_expansions,
        )
    )
    if proposal is None:
        state.base_reserve_reason = "move_unavailable"
        return None
    state.base_reserve_reason = (
        "use_under_uncertainty" if uncertain_damage
        else "use_now" if adjacent else "approach_now"
    )
    return PlannedAction(
        proposal, station.pos,
        f"use:{reserve.item}:{station.unit_id}", None, rounds,
        {"kind": "baseUpgrade", "holderId": str(holder.unit_id),
         "observedDrop": observed_rate, "visibleThreat": visible_rate,
         "estimatedRounds": rounds, "safetyRounds": 2,
         "phase": "use" if adjacent else "approach"},
    )


def weapon_build_positions(turn: Turn) -> tuple[Pos, ...]:
    station = turn.station()
    if station is None:
        return ()
    footprint = frozenset(turn.footprint(station))
    return _ring_positions(turn, footprint, 1)


def wall_build_positions(turn: Turn) -> tuple[Pos, ...]:
    station = turn.station()
    if station is None:
        return ()
    footprint = frozenset(turn.footprint(station))
    return _ring_positions(turn, footprint, 2)


def _ring_positions(
    turn: Turn,
    footprint: frozenset[Pos],
    ring: int,
) -> tuple[Pos, ...]:
    occupied = turn.occupied_cells()
    positions = []
    for x in range(turn.width):
        for y in range(turn.height):
            pos = Pos(x, y)
            if (
                pos not in occupied
                and turn.land(pos)
                and min(distance(pos, cell) for cell in footprint) == ring
            ):
                positions.append(pos)
    return tuple(sorted(positions, key=lambda pos: (pos.x, pos.y)))


def _maintenance_action(
    turn: Turn,
    worker: Unit,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    preferred_use_target_id: int | None = None,
    skip_item: str | None = None,
) -> PlannedAction | None:
    items = Counter(worker.backpack)
    if items["Medicine"] and worker.health < _max_role_health(worker):
        return PlannedAction(ActionProposal(
            worker.unit_id, worker.unit_id, use_command("Medicine"),
        ))

    for item in worker.backpack:
        if item == skip_item:
            continue
        targets = _item_targets(turn, item)
        context = _ROUTE_SEARCH_CONTEXT.get()
        if context is not None:
            targets = tuple(
                target for target in targets
                if context.investment_owners.get(
                    target.unit_id, worker.unit_id,
                ) == worker.unit_id
            )
        preferred = ()
        if preferred_use_target_id is not None:
            preferred = tuple(
                target for target in targets
                if target.unit_id == preferred_use_target_id
            )
            if preferred:
                targets = preferred
        if not targets:
            continue
        held_investment = _is_investment_item(item)
        if held_investment and turn.is_day:
            context = _ROUTE_SEARCH_CONTEXT.get()
            route = _held_item_route(
                turn, worker, item, clock, deadline, max_expansions,
                preferred_use_target_id=(
                    preferred_use_target_id if preferred else None
                ),
            )
            if (
                (route is None or route.rounds > turn.rounds_until_night)
                and preferred
                and (context is None or context.truncated_reason is None)
            ):
                route = _held_item_route(
                    turn, worker, item, clock, deadline, max_expansions,
                )
            if route is None or route.rounds > turn.rounds_until_night:
                return None
            target = turn.unit(route.use_target_id)
            if target is None:
                return None
            proposal = (
                ActionProposal(
                    worker.unit_id, worker.unit_id,
                    use_command(item, target.pos), item_costs=(item,),
                )
                if worker.pos == route.use_stand
                else _move_to_stand(
                    turn, worker, route.use_stand,
                    clock, deadline, max_expansions,
                )
            )
            if proposal is None:
                return None
            candidate = PlannedAction(
                proposal, target.pos, f"use:{item}:{target.unit_id}",
                turn.round_no + turn.rounds_until_night - 1,
                route.rounds,
            )
            return replace(candidate, diagnostic={
                "kind": "heldInvestment", "item": item,
                "useTargetId": str(route.use_target_id),
            })
        for target in targets:
            if distance(worker.pos, target.pos) == 1:
                candidate = PlannedAction(
                    ActionProposal(
                        worker.unit_id,
                        worker.unit_id,
                        use_command(item, target.pos),
                        item_costs=(item,),
                    ),
                    target.pos,
                    f"use:{item}:{target.unit_id}",
                )
                return replace(
                    candidate,
                    diagnostic={
                        "kind": "heldInvestment", "item": item,
                        "useTargetId": str(target.unit_id),
                    },
                ) if held_investment else candidate
            if held_investment and not turn.is_day:
                continue
            candidate = _move_adjacent(
                turn,
                worker,
                target.pos,
                reason=f"use:{item}:{target.unit_id}",
                deadline_round=None,
                clock=clock,
                deadline=deadline,
                max_expansions=max_expansions,
            )
            if candidate is None:
                continue
            if not held_investment:
                return candidate
            return replace(
                candidate, diagnostic={
                    "kind": "heldInvestment", "item": item,
                    "useTargetId": str(target.unit_id),
                },
            )
    return None


def _is_investment_item(item: str) -> bool:
    return item == "WallFixer" or item.startswith(INVESTMENT_ITEM_PREFIXES)


def _is_held_investment_action(candidate: PlannedAction | None) -> bool:
    return (
        candidate is not None
        and isinstance(candidate.diagnostic, dict)
        and candidate.diagnostic.get("kind") == "heldInvestment"
    )


def _is_immediate_held_investment_action(
    candidate: PlannedAction | None,
) -> bool:
    return (
        _is_held_investment_action(candidate)
        and candidate.proposal.command.get("action") == "use"
    )


def _urgent_maintenance_candidate(
    turn: Turn,
    candidate: PlannedAction | None,
) -> bool:
    if candidate is None:
        return False
    command = candidate.proposal.command
    if command.get("action") == "use" and command.get("name") == "Medicine":
        return True
    if not _is_held_investment_action(candidate):
        return False
    if command.get("name") != "WallFixer" and not (
        candidate.plan_reason or ""
    ).startswith("use:WallFixer:"):
        return False
    target_id = _plan_use_target_id(candidate)
    target = turn.unit(target_id) if target_id is not None else None
    return target is not None and _wall_repair_urgent(turn, target)


def _plan_use_target_id(plan) -> int | None:
    if plan is None:
        return None
    reason = getattr(plan, "reason", None) or getattr(plan, "plan_reason", None)
    if reason is None:
        return None
    parts = reason.split(":")
    try:
        if len(parts) == 3 and parts[0] == "use":
            return int(parts[2])
        if len(parts) >= 4 and parts[0] == "fund":
            return int(parts[3]) or None
        batch = _procurement_batch(reason)
        if batch is not None:
            return batch.primary_target_id
    except ValueError:
        return None
    return None


def _procurement_reason(batch: ProcurementBatch) -> str:
    reserved = ",".join(str(target_id) for target_id in batch.reserved_target_ids)
    return ":".join((
        "batch", batch.item, str(batch.post_weapon_id),
        str(batch.primary_target_id), batch.mineral,
        str(batch.mine.x), str(batch.mine.y), str(batch.goal_count),
        str(batch.quote), str(batch.item_price), reserved,
    ))


def _procurement_batch(reason: str | None) -> ProcurementBatch | None:
    if not isinstance(reason, str) or not reason.startswith("batch:"):
        return None
    parts = reason.split(":")
    if len(parts) != 11 or parts[4] not in MINERALS:
        return None
    try:
        reserved = tuple(
            int(target_id) for target_id in parts[10].split(",") if target_id
        )
        batch = ProcurementBatch(
            parts[1], int(parts[2]), int(parts[3]), parts[4],
            Pos(int(parts[5]), int(parts[6])), int(parts[7]),
            int(parts[8]), int(parts[9]), reserved,
        )
    except ValueError:
        return None
    if (
        batch.post_weapon_id <= 0
        or batch.primary_target_id <= 0
        or batch.goal_count <= 0
        or batch.quote <= 0
        or batch.item_price <= 0
        or batch.primary_target_id not in batch.reserved_target_ids
    ):
        return None
    return batch


def _plan_use_target_ids(plan) -> tuple[int, ...]:
    reason = getattr(plan, "reason", None) or getattr(plan, "plan_reason", None)
    batch = _procurement_batch(reason)
    if batch is not None:
        return batch.reserved_target_ids
    target_id = _plan_use_target_id(plan)
    return (target_id,) if target_id is not None else ()


def _reserve_investment_target(
    candidate: PlannedAction,
    context: RouteSearchContext | None,
) -> None:
    if context is None:
        return
    for target_id in _plan_use_target_ids(candidate):
        joint = _joint_plan(candidate.plan_reason or "")
        owner_id = joint[3] if joint is not None else candidate.proposal.actor_id
        context.investment_targets.add(target_id)
        context.investment_owners.setdefault(
            target_id, owner_id,
        )


def _continue_plan(
    turn: Turn,
    state: SessionState,
    worker: Unit,
    failed_builds: set[Pos],
    failed_mines: set[Pos],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    plan = state.plans.get(worker.unit_id)
    if plan is None or plan.reason == "s0_probe":
        return None
    if plan.reason in ("build:rocket", "fund:build:rocket"):
        completed_tower = next((
            weapon for weapon in turn.weapons()
            if weapon.pos == plan.target
        ), None)
        if completed_tower is not None:
            pending_walls = set(state.fortification_targets) - {
                wall.pos for wall in turn.walls()
            }
            if worker.pos not in pending_walls:
                return None
            safe_stands = tuple(
                route for route in _routes_to_adjacent(
                    turn, worker, completed_tower.pos,
                    clock, deadline, max_expansions,
                )
                if route[0] not in pending_walls
            )
            if not safe_stands:
                return None
            stand, cost = min(safe_stands, key=lambda route: (
                route[1], route[0].x, route[0].y,
            ))
            proposal = _move_to_stand(
                turn, worker, stand, clock, deadline, max_expansions,
            )
            if proposal is None:
                return None
            return PlannedAction(
                proposal, completed_tower.pos, "fund:build:rocket",
                plan.deadline_round, cost,
            )
    if plan.deadline_round is not None and turn.round_no > plan.deadline_round:
        return None
    if plan.reason.startswith("batch:"):
        return _continue_procurement_batch(
            turn, worker, plan.reason, plan.deadline_round,
            clock, deadline, max_expansions,
        )
    if plan.reason.startswith("build:"):
        kind = plan.reason.split(":", 1)[1]
        if (
            plan.target in failed_builds
            or not _build_plan_valid(turn, worker, plan.target, kind)
        ):
            return None
        if kind == "rocket":
            route = _best_adjacent_route(
                turn, worker, plan.target,
                clock, deadline, max_expansions,
            )
            if route is None or not _tower_route_can_return(
                turn, state, worker, plan.target, route[0], route[1] + 1,
                clock, deadline, max_expansions,
            ):
                return None
        return _build_or_move(
            turn, worker, plan.target, kind, clock, deadline, max_expansions,
        )
    if plan.reason.startswith("use:"):
        parts = plan.reason.split(":")
        if len(parts) != 3 or parts[1] not in worker.backpack:
            return None
        target = turn.unit(int(parts[2]))
        if target is None or target.pos != plan.target:
            return None
        if not _item_matches_target(parts[1], target):
            return None
        context = _ROUTE_SEARCH_CONTEXT.get()
        if (
            context is not None
            and context.investment_owners.get(
                target.unit_id, worker.unit_id,
            ) != worker.unit_id
        ):
            return None
        route = _held_item_route(
            turn, worker, parts[1], clock, deadline, max_expansions,
            preferred_use_target_id=target.unit_id,
        )
        context = _ROUTE_SEARCH_CONTEXT.get()
        if (
            route is not None
            and route.rounds > turn.rounds_until_night
            and (context is None or context.truncated_reason is None)
        ):
            return None
        if distance(worker.pos, target.pos) == 1:
            return PlannedAction(ActionProposal(
                worker.unit_id,
                worker.unit_id,
                use_command(parts[1], target.pos),
                item_costs=(parts[1],),
            ))
        return _move_adjacent(
            turn, worker, target.pos, plan.reason, plan.deadline_round,
            clock, deadline, max_expansions,
        )
    if plan.reason.startswith("fund:build:"):
        kind = plan.reason.rsplit(":", 1)[1]
        if plan.target in failed_builds:
            return None
        if kind == "rocket":
            if turn.gold >= 25:
                route = _best_adjacent_route(
                    turn, worker, plan.target,
                    clock, deadline, max_expansions,
                )
                stand = route[0] if route is not None else None
                rounds = route[1] + 1 if route is not None else None
            else:
                funded = _building_funding_chain(
                    turn, worker, plan.target, 25 - turn.gold,
                    clock, deadline, max_expansions,
                )
                stand = funded.build_stand if funded is not None else None
                rounds = funded.rounds if funded is not None else None
            if stand is None or rounds is None or not _tower_route_can_return(
                turn, state, worker, plan.target, stand, rounds,
                clock, deadline, max_expansions,
            ):
                return None
        return _funding_build_action(
            turn,
            worker,
            plan.target,
            kind,
            plan.deadline_round,
            clock,
            deadline,
            max_expansions,
        )
    if plan.reason.startswith("fund:"):
        if _joint_plan(plan.reason) is not None:
            return None
        parts = plan.reason.split(":")
        item = parts[1]
        try:
            post_weapon_id = int(parts[2]) if len(parts) == 3 else None
            if len(parts) >= 4:
                post_weapon_id = int(parts[2])
                use_target_id = int(parts[3]) or None
            else:
                use_target_id = None
        except ValueError:
            return None
        return _funding_action(
            turn,
            worker,
            item,
            plan.deadline_round,
            clock,
            deadline,
            max_expansions,
            preferred_post_id=post_weapon_id,
            preferred_use_target_id=use_target_id,
            gate_new_maintenance=False,
        )
    if plan.reason.startswith("mine:"):
        if worker.backpack_full or turn.zones.get(plan.target) not in MINERALS:
            return None
        return _mine_action(
            turn,
            worker,
            clock,
            deadline,
            max_expansions,
            preferred_target=plan.target,
            excluded_targets=failed_mines,
        )
    if (
        plan.reason == "vendor"
        and turn.zones.get(plan.target) == "vendor"
        and any(item in MINERALS for item in worker.backpack)
    ):
        return _sell_or_move(
            turn, worker, plan.target, clock, deadline, max_expansions,
        )
    if plan.reason.startswith("shop:") and turn.zones.get(plan.target) == "weaponShop":
        item = plan.reason.split(":", 1)[1]
        if not _planned_purchase_valid(turn, worker, item):
            return None
        return _buy_or_move(
            turn, worker, plan.target, item, clock, deadline, max_expansions,
        )
    return None


def _continue_procurement_batch(
    turn: Turn,
    worker: Unit,
    reason: str,
    hard_deadline: int | None,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    batch = _procurement_batch(reason)
    if (
        batch is None
        or not turn.is_day
        or hard_deadline is None
        or turn.round_no > hard_deadline
        or turn.vendor_prices.get(batch.mineral) != batch.quote
        or turn.weapon_prices.get(batch.item) != batch.item_price
    ):
        return None
    primary = turn.unit(batch.primary_target_id)
    if primary is None or not _item_matches_target(batch.item, primary):
        return None
    if turn.station() is None or len(turn.weapons()) < MAX_WEAPONS:
        return None
    if _critical_purchase_available(
        turn, worker, clock, deadline, max_expansions,
    ):
        return None
    if (
        worker.health <= EMERGENCY_MEDICINE_HEALTH
        and worker.health < _max_role_health(worker)
        and "Medicine" in turn.weapon_prices
    ):
        return None
    if batch.item == "WallFixer" and any(
        _wall_repair_urgent(turn, target)
        for target in _item_targets(turn, batch.item)
    ):
        return None
    remaining_targets = _continuing_procurement_target_ids(
        turn,
        worker,
        batch,
        clock,
        deadline,
        max_expansions,
    )
    if batch.primary_target_id not in remaining_targets:
        return None
    current_value = sum(
        turn.vendor_prices.get(entry, 0)
        for entry in worker.backpack if entry in MINERALS
    )
    current_count = worker.backpack.count(batch.mineral)
    other_value = sum(
        turn.vendor_prices.get(entry, 0)
        for entry in worker.backpack
        if entry in MINERALS and entry != batch.mineral
    )
    required_mineral_value = max(
        0,
        batch.item_price * len(remaining_targets) - turn.gold - other_value,
    )
    demand_goal = (
        required_mineral_value + batch.quote - 1
    ) // batch.quote
    effective_goal = min(batch.goal_count, demand_goal)
    can_finish_primary = turn.gold + current_value >= batch.item_price
    keep_collecting = (
        len(remaining_targets) >= 2
        and turn.gold + current_value
        < batch.item_price * len(remaining_targets)
        and current_count < effective_goal
        and not worker.backpack_full
        and turn.zones.get(batch.mine) == batch.mineral
        and distance(worker.pos, batch.mine) == 1
    )
    if keep_collecting:
        remaining = effective_goal - current_count
        projected = replace(
            worker, backpack=worker.backpack + (batch.mineral,) * remaining,
        )
        route = _batch_funding_chain(
            turn,
            projected,
            batch.item,
            clock,
            deadline,
            max_expansions,
            preferred_post_id=batch.post_weapon_id,
            preferred_use_target_id=batch.primary_target_id,
        )
        if route is None or remaining + route.rounds > turn.rounds_until_night:
            return None
        return PlannedAction(
            ActionProposal(
                worker.unit_id,
                worker.unit_id,
                collect_command(batch.mine),
                destination=batch.mine,
            ),
            batch.mine,
            reason,
            hard_deadline,
            remaining + route.rounds,
            diagnostic={"kind": "procurementBatch", "committed": True},
        )
    if not can_finish_primary:
        return None
    minerals = Counter(entry for entry in worker.backpack if entry in MINERALS)
    if minerals:
        route = _batch_funding_chain(
            turn,
            worker,
            batch.item,
            clock,
            deadline,
            max_expansions,
            preferred_post_id=batch.post_weapon_id,
            preferred_use_target_id=batch.primary_target_id,
        )
        if route is None or route.rounds > turn.rounds_until_night:
            return None
        sale = next((
            (name, count) for name, count in sorted(
                minerals.items(),
                key=lambda entry: (
                    -turn.vendor_prices.get(entry[0], 0), entry[0],
                ),
            )
            if count > 0 and turn.vendor_prices.get(name, 0) > 0
        ), None)
        if sale is None or route.vendor is None or route.vendor_stand is None:
            return None
        if worker.pos == route.vendor_stand:
            proposal = ActionProposal(
                worker.unit_id,
                worker.unit_id,
                sell_command(sale[0], sale[1]),
            )
        else:
            proposal = _move_to_stand(
                turn, worker, route.vendor_stand,
                clock, deadline, max_expansions,
            )
        if proposal is None:
            return None
        return PlannedAction(
            proposal,
            route.vendor,
            reason,
            hard_deadline,
            route.rounds,
            diagnostic={"kind": "procurementBatch", "committed": True},
        )
    return _funding_action(
        turn,
        worker,
        batch.item,
        hard_deadline,
        clock,
        deadline,
        max_expansions,
        preferred_post_id=batch.post_weapon_id,
        preferred_use_target_id=batch.primary_target_id,
        gate_new_maintenance=False,
    )


def _continuing_procurement_target_ids(
    turn: Turn,
    worker: Unit,
    batch: ProcurementBatch,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[int, ...]:
    result = []
    for target_id in batch.reserved_target_ids:
        if clock() >= deadline:
            break
        target = turn.unit(target_id)
        if target is None or not _item_matches_target(batch.item, target):
            continue
        route = _batch_funding_chain(
            turn,
            worker,
            batch.item,
            clock,
            deadline,
            max_expansions,
            preferred_post_id=batch.post_weapon_id,
            preferred_use_target_id=target_id,
        )
        if (
            route is None
            or route.rounds > turn.rounds_until_night
            or not _maintenance_purchase_worthwhile(
                turn, worker, batch.item, route,
            )
        ):
            continue
        result.append(target_id)
    return tuple(result)


def _critical_purchase_available(
    turn: Turn,
    worker: Unit,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> bool:
    inventory_value = sum(
        turn.vendor_prices.get(entry, 0)
        for entry in worker.backpack if entry in MINERALS
    )
    for item in _purchase_candidates(turn, worker):
        if not item.startswith((
            "WeaponUpgradeVoucher", "StationUpgradeVoucher",
        )):
            continue
        price = turn.weapon_prices.get(item)
        if price is None or price > turn.gold + inventory_value:
            continue
        route = (
            _purchase_route(
                turn, worker, item, clock, deadline, max_expansions,
            )
            if price <= turn.gold
            else _funding_chain(
                turn,
                worker,
                item,
                price - turn.gold,
                clock,
                deadline,
                max_expansions,
            )
        )
        if route is not None and route.rounds <= turn.rounds_until_night:
            return True
    return False


def _wall_action(
    turn: Turn,
    state: SessionState,
    worker: Unit,
    excluded: set[Pos],
    failed_mines: set[Pos],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    if not turn.is_day:
        return None
    batch_targets = tuple(
        target for target in state.fortification_batch_targets
        if target not in state.fortification_completed
        and target not in state.fortification_failed
    )
    stone_goal = max(len(batch_targets), 1)
    if worker.backpack.count("stone") < stone_goal:
        stone = fortification_stone_target(turn, worker, failed_mines)
        if stone is None:
            return None
        return _collect_or_move(
            turn, worker, stone, clock, deadline, max_expansions,
            diagnostic={
                "kind": "fortification",
                "phase": "mining",
                "target": stone.dump(),
            },
        )
    target = next((
        pos for pos in (batch_targets or remaining_wall_targets(state))
        if pos not in excluded
    ), None)
    if target is None:
        return None
    return _build_or_move(
        turn, worker, target, WALL, clock, deadline, max_expansions,
    )


def _tower_action(
    turn: Turn,
    state: SessionState,
    worker: Unit,
    excluded: set[Pos],
    claimed_types: Counter[str],
    available_gold: int,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    weapons = turn.weapons()
    if (
        not turn.is_day
        or turn.station() is None
        or len(weapons) + sum(claimed_types.values()) >= MAX_WEAPONS
    ):
        return None
    kind = "rocket"
    preferred = tuple(
        pos for pos in state.layout_tower_targets
        if pos not in excluded
        and pos not in turn.occupied_cells()
        and turn.land(pos)
    )
    fallback = tuple(
        pos for pos in weapon_build_positions(turn) if pos not in excluded
    )
    target = _nearest(
        worker.pos,
        preferred if state.layout_tower_targets else fallback,
    )
    if target is None:
        return None
    if turn.rounds_until_night < distance(worker.pos, target):
        return None
    if available_gold < 25:
        inventory_value = sum(
            turn.vendor_prices.get(item, 0)
            for item in worker.backpack
            if item in MINERALS
        )
        if available_gold + inventory_value < 25:
            return None
        route = _building_funding_chain(
            turn,
            worker,
            target,
            25 - available_gold,
            clock,
            deadline,
            max_expansions,
        )
        if route is None or route.rounds > turn.rounds_until_night:
            return None
        if not _tower_route_can_return(
            turn, state, worker, target, route.build_stand,
            route.rounds, clock, deadline, max_expansions,
        ):
            return None
        return _funding_build_action(
            turn,
            worker,
            target,
            kind,
            turn.round_no + turn.rounds_until_night - 1,
            clock,
            deadline,
            max_expansions,
            verified_route=route,
            available_gold=available_gold,
        )
    build_route = _best_adjacent_route(
        turn, worker, target, clock, deadline, max_expansions,
    )
    if build_route is None or not _tower_route_can_return(
        turn, state, worker, target, build_route[0], build_route[1] + 1,
        clock, deadline, max_expansions,
    ):
        return None
    return _build_or_move(
        turn, worker, target, kind, clock, deadline, max_expansions,
    )


def _tower_route_can_return(
    turn: Turn,
    state: SessionState,
    worker: Unit,
    target: Pos,
    build_stand: Pos,
    build_rounds: int,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> bool:
    if not turn.weapons():
        return build_rounds + DAY_WORK_RETURN_MARGIN <= turn.rounds_until_night
    projected = replace(
        turn,
        zones={**turn.zones, target: "rocket"},
        ours=tuple(
            replace(role, pos=build_stand)
            if role.unit_id == worker.unit_id else role
            for role in turn.ours
        ),
    )
    from_build = replace(worker, pos=build_stand)
    post = state.day_return_stands.get(worker.unit_id)
    if post is not None:
        result = _search_path(
            projected, from_build, post,
            clock=clock, deadline=deadline,
            max_expansions=max_expansions,
        )
        return_cost = (
            0 if result.status == "already_there" else result.cost
            if result.status == "found" else None
        )
    else:
        routes = (
            route for weapon in turn.weapons()
            for route in _routes_to_adjacent(
                projected, from_build, weapon.pos,
                clock, deadline, max_expansions,
            )
        )
        return_cost = min((route[1] for route in routes), default=None)
    return (
        return_cost is not None
        and build_rounds + return_cost + DAY_WORK_RETURN_MARGIN
        <= turn.rounds_until_night
    )


def _build_or_move(
    turn: Turn,
    worker: Unit,
    target: Pos,
    kind: str,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    reason = f"build:{kind}"
    if distance(worker.pos, target) == 1:
        return PlannedAction(
            ActionProposal(
                worker.unit_id,
                worker.unit_id,
                build_command(target, kind),
                destination=target,
            ),
            target,
            reason,
        )
    return _move_adjacent(
        turn, worker, target, reason, None,
        clock, deadline, max_expansions,
    )


def _joint_funding_actions(
    turn: Turn,
    state: SessionState,
    *,
    excluded_role_ids: frozenset[int],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[
    tuple[PlannedAction, ...],
    frozenset[int],
    dict[int, str],
]:
    existing_plans = {
        role_id: parsed
        for role_id, plan in state.plans.items()
        for parsed in (_joint_plan(plan.reason),)
        if parsed is not None
    }
    context = _ROUTE_SEARCH_CONTEXT.get()
    if not turn.is_day or len(turn.weapons()) < MAX_WEAPONS:
        if context is not None and existing_plans:
            context.joint_status = (
                "deadline_reached" if not turn.is_day
                else "tower_line_incomplete"
            )
        return (), frozenset(), {
            role_id: (
                "deadline_reached"
                if not turn.is_day
                else "tower_line_incomplete"
            )
            for role_id in existing_plans
        }
    workers = tuple(
        worker for worker in turn.workers()
        if worker.unit_id not in excluded_role_ids
    )
    if len(workers) != 2:
        if context is not None and existing_plans:
            context.joint_status = "participant_unavailable"
        return (), frozenset(), {
            role_id: "participant_unavailable" for role_id in existing_plans
        }

    commitment = next((
        parsed
        for worker in workers
        for plan in (state.plans.get(worker.unit_id),)
        if plan is not None
        for parsed in (_joint_plan(plan.reason),)
        if parsed is not None
    ), None)
    existing = None
    if commitment is not None:
        item, _, target_id, buyer_id, _ = commitment
        buyer_plan = existing_plans.get(buyer_id)
        buyer_post_id = (
            buyer_plan[1]
            if buyer_plan is not None
            and (buyer_plan[0], buyer_plan[2], buyer_plan[3])
            == (item, target_id, buyer_id)
            else None
        )
        existing = (item, buyer_post_id, target_id, buyer_id)
        completed = _joint_completion_actions(
            turn,
            state,
            workers,
            existing_plans,
            existing,
            clock,
            deadline,
            max_expansions,
        )
        if completed is not None:
            if context is not None:
                context.joint_status = "returning"
            return completed
    buyer_order = sorted(
        workers,
        key=lambda worker: (
            existing is None or worker.unit_id != existing[3],
            worker.unit_id,
        ),
    )
    options = []
    for buyer in buyer_order:
        contributor = next(
            worker for worker in workers if worker.unit_id != buyer.unit_id
        )
        items = _purchase_candidates(turn, buyer)
        if existing is not None and buyer.unit_id == existing[3]:
            items = tuple(sorted(
                items,
                key=lambda item: (item != existing[0], items.index(item)),
            ))
        for item in items:
            if clock() >= deadline:
                break
            price = turn.weapon_prices.get(item)
            if price is None:
                continue
            buyer_value = _inventory_value(turn, buyer)
            contributor_value = _inventory_value(turn, contributor)
            continuing = (
                existing is not None
                and existing[0] == item
                and existing[3] == buyer.unit_id
            )
            if not continuing and (
                item in buyer.backpack
                or price <= turn.gold
                or buyer_value <= 0
                or contributor_value <= 0
            ):
                continue
            if item not in buyer.backpack and price > turn.gold:
                if turn.gold + buyer_value + contributor_value < price:
                    continue
                if not continuing and (
                    turn.gold + buyer_value >= price
                    or turn.gold + contributor_value >= price
                ):
                    continue
            option = _joint_funding_route(
                turn,
                buyer,
                contributor,
                item,
                existing if continuing else None,
                clock,
                deadline,
                max_expansions,
            )
            if option is not None:
                options.append(option)
    options.sort(key=lambda option: (
        existing is None or option.buyer_id != existing[3],
        max(option.buyer_rounds, option.contributor_rounds),
        option.buyer_rounds + option.contributor_rounds,
        option.buyer_id,
        option.item,
        option.buyer_post_id,
        option.contributor_post_id,
    ))
    for option in options:
        actions = _joint_route_actions(
            turn, option, clock, deadline, max_expansions,
        )
        destinations = [
            action.proposal.destination
            for action in actions
            if action.proposal.destination is not None
        ]
        if len(destinations) != len(set(destinations)):
            _note_joint_blocker(context, "action_conflict")
            continue
        if context is not None:
            context.joint_status = "accepted"
        return (
            actions,
            frozenset((option.buyer_id, option.contributor_id)),
            {},
        )
    reason = _joint_cancellation_reason(turn, workers, existing)
    if context is not None:
        if context.truncated_reason is not None:
            context.joint_status = "search_truncated"
        elif existing is not None:
            context.joint_status = reason
        elif _joint_actual_funds_insufficient(turn, workers):
            context.joint_status = "actual_funds_insufficient"
        elif context.joint_blocker is not None:
            context.joint_status = context.joint_blocker
        else:
            context.joint_status = reason
    return (), frozenset(), {
        role_id: reason for role_id in existing_plans
    }


def _joint_actual_funds_insufficient(
    turn: Turn,
    workers: tuple[Unit, ...],
) -> bool:
    actual_funds = turn.gold + sum(
        _inventory_value(turn, worker) for worker in workers
    )
    prices = {
        turn.weapon_prices[item]
        for worker in workers
        for item in _purchase_candidates(turn, worker)
        if item not in worker.backpack and item in turn.weapon_prices
    }
    return bool(prices) and actual_funds < min(prices)


def _note_joint_blocker(
    context: RouteSearchContext | None,
    blocker: str,
) -> None:
    if context is None:
        return
    priority = {
        "target_unavailable": 1,
        "sale_route_unavailable": 2,
        "purchase_route_unavailable": 3,
        "distinct_post_unavailable": 4,
        "return_deadline": 5,
        "action_conflict": 6,
    }
    current = priority.get(context.joint_blocker or "", 0)
    if priority.get(blocker, 0) > current:
        context.joint_blocker = blocker


def _joint_funding_route(
    turn: Turn,
    buyer: Unit,
    contributor: Unit,
    item: str,
    existing: tuple[str, int | None, int | None, int] | None,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    candidate_use_target_id: int | None = None,
) -> JointFundingRoute | None:
    context = _ROUTE_SEARCH_CONTEXT.get()
    price = turn.weapon_prices.get(item)
    if price is None:
        return None
    if (
        existing is None
        and item == "WallFixer"
        and candidate_use_target_id is None
    ):
        options = tuple(
            option
            for target in _item_targets(turn, item)
            for option in (
                _joint_funding_route(
                    turn,
                    buyer,
                    contributor,
                    item,
                    None,
                    clock,
                    deadline,
                    max_expansions,
                    candidate_use_target_id=target.unit_id,
                ),
            )
            if option is not None
        )
        return min(
            options,
            key=lambda option: (
                max(option.buyer_rounds, option.contributor_rounds),
                option.buyer_rounds + option.contributor_rounds,
                option.use_target_id or 0,
            ),
            default=None,
        )
    preferred_use_target_id = (
        existing[2] if existing is not None else candidate_use_target_id
    )
    preferred_target = (
        turn.unit(preferred_use_target_id)
        if preferred_use_target_id is not None else None
    )
    if item != "Medicine":
        target_available = (
            bool(_item_targets(turn, item))
            if preferred_use_target_id is None
            else preferred_target is not None
            and _item_matches_target(item, preferred_target)
        )
        if not target_available:
            _note_joint_blocker(context, "target_unavailable")
            return None

    deficit = 0 if item in buyer.backpack else max(0, price - turn.gold)
    buyer_sales = _sales_for_deficit(turn, buyer, deficit)
    buyer_contribution = _sales_value(turn, buyer_sales)
    contributor_sales = _sales_for_deficit(
        turn, contributor, max(0, deficit - buyer_contribution),
    )
    if (
        item not in buyer.backpack
        and turn.gold + buyer_contribution
        + _sales_value(turn, contributor_sales) < price
    ):
        return None

    buyer_starts = _after_sale_positions(
        turn,
        buyer,
        buyer_sales,
        clock,
        deadline,
        max_expansions,
    )
    contributor_starts = _after_sale_positions(
        turn,
        contributor,
        contributor_sales,
        clock,
        deadline,
        max_expansions,
    )
    if not buyer_starts or not contributor_starts:
        _note_joint_blocker(context, "sale_route_unavailable")
        return None

    preferred_buyer_post = existing[1] if existing is not None else None
    best = None
    purchase_found = False
    distinct_post_found = False
    deadline_rejected = False
    for buyer_vendor, buyer_stand, buyer_sale_rounds in buyer_starts:
        at_buyer_start = replace(buyer, pos=buyer_stand)
        purchase = (
            _held_item_route(
                turn,
                at_buyer_start,
                item,
                clock,
                deadline,
                max_expansions,
                preferred_post_id=preferred_buyer_post,
                preferred_use_target_id=preferred_use_target_id,
            )
            if item in buyer.backpack
            else _purchase_route(
                turn,
                at_buyer_start,
                item,
                clock,
                deadline,
                max_expansions,
                preferred_post_id=preferred_buyer_post,
                preferred_use_target_id=preferred_use_target_id,
                gate_new_maintenance=existing is None,
            )
        )
        if purchase is None:
            continue
        purchase_found = True
        for contributor_vendor, contributor_stand, contributor_sale_rounds in (
            contributor_starts
        ):
            at_contributor_start = replace(contributor, pos=contributor_stand)
            for weapon, post_stand, post_rounds in _post_routes(
                turn,
                at_contributor_start,
                clock,
                deadline,
                max_expansions,
            ):
                if weapon.unit_id == purchase.post_weapon_id:
                    continue
                distinct_post_found = True
                readiness = max(buyer_sale_rounds, contributor_sale_rounds)
                buyer_rounds = readiness + purchase.rounds
                contributor_rounds = contributor_sale_rounds + post_rounds
                if (
                    buyer_rounds > turn.rounds_until_night
                    or contributor_rounds > turn.rounds_until_night
                ):
                    deadline_rejected = True
                    continue
                if (
                    existing is None
                    and not _maintenance_purchase_worthwhile(
                        turn,
                        buyer,
                        item,
                        replace(
                            purchase,
                            rounds=buyer_rounds + contributor_rounds,
                        ),
                    )
                ):
                    continue
                candidate = JointFundingRoute(
                    buyer.unit_id,
                    contributor.unit_id,
                    item,
                    purchase.use_target_id,
                    purchase.post_weapon_id,
                    weapon.unit_id,
                    buyer_sales,
                    contributor_sales,
                    buyer_vendor,
                    buyer_stand if buyer_sales else None,
                    contributor_vendor,
                    contributor_stand if contributor_sales else None,
                    purchase,
                    post_stand,
                    buyer_rounds,
                    contributor_rounds,
                )
                key = (
                    max(buyer_rounds, contributor_rounds),
                    buyer_rounds + contributor_rounds,
                    candidate.buyer_post_id,
                    candidate.contributor_post_id,
                    post_stand.x,
                    post_stand.y,
                )
                if best is None or key < best[0]:
                    best = (key, candidate)
    if best is None:
        if deadline_rejected:
            _note_joint_blocker(context, "return_deadline")
        elif purchase_found and not distinct_post_found:
            _note_joint_blocker(context, "distinct_post_unavailable")
        elif not purchase_found:
            _note_joint_blocker(context, "purchase_route_unavailable")
    return best[1] if best is not None else None


def _joint_route_actions(
    turn: Turn,
    route: JointFundingRoute,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[PlannedAction, ...]:
    hard_deadline = turn.round_no + turn.rounds_until_night - 1
    result = []
    buyer = turn.unit(route.buyer_id)
    contributor = turn.unit(route.contributor_id)
    if buyer is None or contributor is None:
        return ()
    diagnostic = {
        "kind": "jointFunding",
        "purchase": route.item,
        "useTargetId": (
            str(route.use_target_id) if route.use_target_id is not None else None
        ),
        "participants": [str(route.buyer_id), str(route.contributor_id)],
        "expectedContribution": {
            str(route.buyer_id): _sales_value(turn, route.buyer_sales),
            str(route.contributor_id): _sales_value(
                turn, route.contributor_sales,
            ),
        },
        "currentGold": turn.gold,
    }
    for worker, sales, vendor, vendor_stand, post_id, estimated in (
        (
            buyer,
            route.buyer_sales,
            route.buyer_vendor,
            route.buyer_vendor_stand,
            route.buyer_post_id,
            route.buyer_rounds,
        ),
        (
            contributor,
            route.contributor_sales,
            route.contributor_vendor,
            route.contributor_vendor_stand,
            route.contributor_post_id,
            route.contributor_rounds,
        ),
    ):
        reason = (
            f"fund:{route.item}:{post_id}:{route.use_target_id or 0}:"
            f"joint:{route.buyer_id}"
        )
        action = None
        target = None
        if sales:
            target = vendor
            if worker.pos == vendor_stand:
                name, quantity = sales[0]
                action = ActionProposal(
                    worker.unit_id,
                    worker.unit_id,
                    sell_command(name, quantity),
                )
            else:
                action = _move_to_stand(
                    turn,
                    worker,
                    vendor_stand,
                    clock,
                    deadline,
                    max_expansions,
                )
        elif worker.unit_id == route.buyer_id:
            candidate = _funding_action(
                turn,
                worker,
                route.item,
                hard_deadline,
                clock,
                deadline,
                max_expansions,
                preferred_post_id=route.buyer_post_id,
                preferred_use_target_id=route.use_target_id,
                verified_route=route.purchase,
            )
            if candidate is not None:
                action = candidate.proposal
                target = candidate.plan_target
        elif worker.pos != route.contributor_post_stand:
            action = _move_to_stand(
                turn,
                worker,
                route.contributor_post_stand,
                clock,
                deadline,
                max_expansions,
            )
            target = turn.unit(route.contributor_post_id).pos
        if action is not None:
            result.append(PlannedAction(
                action,
                target,
                reason,
                hard_deadline,
                estimated,
                diagnostic,
            ))
    return tuple(result)


def _joint_completion_actions(
    turn: Turn,
    state: SessionState,
    workers: tuple[Unit, ...],
    existing_plans: dict[int, tuple[str, int, int | None, int, bool]],
    existing: tuple[str, int | None, int | None, int],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[tuple[PlannedAction, ...], frozenset[int], dict[int, str]] | None:
    item, _, target_id, buyer_id = existing
    target = turn.unit(target_id) if target_id is not None else None
    completed = next((
        entry
        for entry in reversed(state.action_history)
        if entry.pending.actor_id == buyer_id
        and entry.pending.source_session == state.session_index
        and entry.pending.round_no == turn.round_no - 1
        and entry.pending.action == "use"
        and entry.pending.name == item
        and entry.pending.target == (target.pos if target is not None else None)
    ), None)
    returning = any(plan[4] for plan in existing_plans.values())
    if not returning and (completed is None or completed.success is not True):
        return None
    if set(existing_plans) != {worker.unit_id for worker in workers}:
        return None
    if any(
        (plan[0], plan[2], plan[3]) != (item, target_id, buyer_id)
        for plan in existing_plans.values()
    ):
        return None
    post_ids = {role_id: plan[1] for role_id, plan in existing_plans.items()}
    if len(set(post_ids.values())) != len(post_ids):
        return None

    diagnostic = {
        "kind": "jointFunding",
        "purchase": item,
        "useTargetId": str(target_id) if target_id is not None else None,
        "participants": [str(worker.unit_id) for worker in workers],
        "expectedContribution": {
            str(worker.unit_id): 0 for worker in workers
        },
        "currentGold": turn.gold,
    }
    actions = []
    all_at_posts = True
    for worker in workers:
        post = turn.unit(post_ids[worker.unit_id])
        if post is None or post.kind not in TOWER_TYPES:
            return None
        if distance(worker.pos, post.pos) == 1:
            continue
        all_at_posts = False
        candidate = _post_action(
            turn,
            worker,
            item,
            post.unit_id,
            turn.round_no + turn.rounds_until_night - 1,
            clock,
            deadline,
            max_expansions,
        )
        if candidate is None:
            return None
        actions.append(replace(
            candidate,
            plan_reason=(
                f"fund:{item}:{post.unit_id}:{target_id or 0}:"
                f"joint:{buyer_id}:return"
            ),
            diagnostic=diagnostic,
        ))
    if all_at_posts:
        for role_id in post_ids:
            state.plans.pop(role_id, None)
        return (), frozenset(post_ids), {}
    destinations = [action.proposal.destination for action in actions]
    if len(destinations) != len(set(destinations)):
        return None
    return tuple(actions), frozenset(post_ids), {}


def _joint_cancellation_reason(
    turn: Turn,
    workers: tuple[Unit, ...],
    existing: tuple[str, int | None, int | None, int] | None,
) -> str:
    if existing is None:
        return "route_or_deadline"
    item, _, target_id, buyer_id = existing
    buyer = next((worker for worker in workers if worker.unit_id == buyer_id), None)
    contributor = next((
        worker for worker in workers if worker.unit_id != buyer_id
    ), None)
    if buyer is None or contributor is None:
        return "participant_unavailable"
    price = turn.weapon_prices.get(item)
    if price is None:
        return "purchase_unavailable"
    if item not in buyer.backpack and not _fund_target_still_needs(
        turn, buyer, item, target_id,
    ):
        return "target_no_longer_needs_item"
    if (
        item not in buyer.backpack
        and turn.gold + _inventory_value(turn, buyer)
        + _inventory_value(turn, contributor) < price
    ):
        return "combined_value_insufficient"
    return "route_or_deadline"


def _joint_plan(reason: str) -> tuple[str, int, int | None, int, bool] | None:
    parts = reason.split(":")
    if (
        len(parts) not in (6, 7)
        or parts[0] != "fund"
        or parts[4] != "joint"
        or (len(parts) == 7 and parts[6] != "return")
    ):
        return None
    try:
        return (
            parts[1],
            int(parts[2]),
            int(parts[3]) or None,
            int(parts[5]),
            len(parts) == 7,
        )
    except ValueError:
        return None


def _inventory_value(turn: Turn, worker: Unit) -> int:
    return sum(
        turn.vendor_prices.get(item, 0)
        for item in worker.backpack
        if item in MINERALS
    )


def _sales_for_deficit(
    turn: Turn,
    worker: Unit,
    deficit: int,
) -> tuple[tuple[str, int], ...]:
    if deficit <= 0:
        return ()
    remaining = deficit
    result = []
    minerals = Counter(item for item in worker.backpack if item in MINERALS)
    for name in sorted(
        minerals,
        key=lambda entry: (turn.vendor_prices.get(entry, 0), entry),
        reverse=True,
    ):
        price = turn.vendor_prices.get(name, 0)
        if price <= 0:
            continue
        quantity = min(minerals[name], (remaining + price - 1) // price)
        if quantity:
            result.append((name, quantity))
            remaining -= price * quantity
        if remaining <= 0:
            break
    return tuple(result)


def _sales_value(
    turn: Turn,
    sales: tuple[tuple[str, int], ...],
) -> int:
    return sum(turn.vendor_prices.get(name, 0) * quantity for name, quantity in sales)


def _after_sale_positions(
    turn: Turn,
    worker: Unit,
    sales: tuple[tuple[str, int], ...],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[tuple[Pos | None, Pos, int], ...]:
    if not sales:
        return ((None, worker.pos, 0),)
    result = []
    for vendor in turn.zones_of("vendor"):
        for stand, cost in _routes_to_adjacent(
            turn, worker, vendor, clock, deadline, max_expansions,
        ):
            result.append((vendor, stand, cost + len(sales)))
    return tuple(result)


def _trade_or_mine(
    turn: Turn,
    worker: Unit,
    available_gold: int,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    failed_mines: set[Pos],
    *,
    check_funding: bool = True,
    night_cleared: bool = False,
    allow_procurement_batch: bool = True,
) -> PlannedAction | None:
    minerals = Counter(item for item in worker.backpack if item in MINERALS)
    vendors = turn.zones_of("vendor")
    if check_funding:
        candidate = _single_funding_action(
            turn, worker, available_gold, clock, deadline, max_expansions,
            allow_procurement_batch=allow_procurement_batch,
        )
        if candidate is not None:
            return candidate
    if minerals and vendors:
        adjacent = next((pos for pos in vendors if distance(worker.pos, pos) == 1), None)
        if adjacent is not None:
            return _sell_or_move(
                turn, worker, adjacent, clock, deadline, max_expansions,
            )
        if (
            worker.backpack_full
            or not turn.is_day and not night_cleared
        ):
            target = _nearest(worker.pos, vendors)
            return _sell_or_move(
                turn, worker, target, clock, deadline, max_expansions,
            )

    shops = turn.zones_of("weaponShop")
    if shops:
        for purchase in _purchase_candidates(turn, worker):
            price = turn.weapon_prices.get(purchase)
            if price is None or price > available_gold:
                continue
            if turn.is_day:
                route = _purchase_route(
                    turn, worker, purchase, clock, deadline, max_expansions,
                )
                if route is None or route.rounds > turn.rounds_until_night:
                    continue
                candidate = _funding_action(
                    turn,
                    worker,
                    purchase,
                    turn.round_no + turn.rounds_until_night - 1,
                    clock,
                    deadline,
                    max_expansions,
                    verified_route=route,
                )
                if candidate is not None:
                    return candidate
                continue
            if (
                purchase == "Medicine"
                and not _medicine_purchase_worthwhile(worker, 2)
            ):
                continue
            for target in sorted(
                shops,
                key=lambda pos: (distance(worker.pos, pos), pos.x, pos.y),
            ):
                if not _purchase_is_timely(
                    turn,
                    worker,
                    target,
                    purchase,
                    clock,
                    deadline,
                    max_expansions,
                ):
                    continue
                candidate = _buy_or_move(
                    turn, worker, target, purchase,
                    clock, deadline, max_expansions,
                )
                if candidate is not None:
                    return candidate

    return _mine_action(
        turn,
        worker,
        clock,
        deadline,
        max_expansions,
        excluded_targets=failed_mines,
    )


def daytime_liquidation_actions(
    turn: Turn,
    worker: Unit,
    *,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[PlannedAction, ...]:
    """Return bounded sale candidates for inventory already earned."""
    if not any(item in MINERALS for item in worker.backpack):
        return ()
    result = []
    for vendor in sorted(
        turn.zones_of("vendor"),
        key=lambda pos: (distance(worker.pos, pos), pos.x, pos.y),
    ):
        if clock() >= deadline:
            break
        candidate = _sell_or_move(
            turn, worker, vendor, clock, deadline, max_expansions,
        )
        if candidate is not None:
            result.append(candidate)
    return tuple(result)


def _single_funding_action(
    turn: Turn,
    worker: Unit,
    available_gold: int,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    allow_procurement_batch: bool = True,
) -> PlannedAction | None:
    funded = _fundable_purchase(
        turn, worker, available_gold, clock, deadline, max_expansions,
    )
    if funded is None:
        return None
    item, hard_deadline, route = funded
    batch = _new_procurement_batch_action(
        turn,
        worker,
        available_gold,
        item,
        hard_deadline,
        route,
        clock,
        deadline,
        max_expansions,
    ) if allow_procurement_batch else None
    if batch is not None:
        return batch
    return _funding_action(
        turn,
        worker,
        item,
        hard_deadline,
        clock,
        deadline,
        max_expansions,
        verified_route=route,
    )


def _new_procurement_batch_action(
    turn: Turn,
    worker: Unit,
    available_gold: int,
    item: str,
    hard_deadline: int,
    immediate_route: FundingRoute,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    context = _ROUTE_SEARCH_CONTEXT.get()
    if (
        context is not None
        and worker.unit_id in context.procurement_blocked_role_ids
    ):
        return None
    if (
        item != "WallFixer"
        and not item.startswith("WallUpgradeVoucher")
    ) or immediate_route.use_target_id is None:
        return None
    if turn.station() is None or len(turn.weapons()) < MAX_WEAPONS:
        return None
    if _critical_purchase_available(
        turn, worker, clock, deadline, max_expansions,
    ):
        return None
    primary = turn.unit(immediate_route.use_target_id)
    if primary is None or (
        item == "WallFixer" and _wall_repair_urgent(turn, primary)
    ):
        return None
    if item == "WallFixer" and any(
        _wall_repair_urgent(turn, target)
        for target in _item_targets(turn, item)
    ):
        return None
    adjacent_mines = tuple(sorted(
        (
            pos for pos, kind in turn.zones.items()
            if kind in MINERALS
            and distance(worker.pos, pos) == 1
            and turn.vendor_prices.get(kind, 0) > 0
        ),
        key=lambda pos: (
            -turn.vendor_prices.get(turn.zones[pos], 0), pos.x, pos.y,
        ),
    ))
    if not adjacent_mines or worker.capacity is None:
        return None
    targets = _viable_procurement_targets(
        turn,
        worker,
        item,
        available_gold,
        immediate_route,
        clock,
        deadline,
        max_expansions,
    )
    if len(targets) < 2:
        return None
    price = turn.weapon_prices.get(item)
    if price is None or price <= 0:
        return None
    inventory_value = sum(
        turn.vendor_prices.get(entry, 0)
        for entry in worker.backpack if entry in MINERALS
    )
    free_slots = worker.capacity - len(worker.backpack)
    if free_slots <= 0:
        return None
    for mine in adjacent_mines:
        mineral = turn.zones[mine]
        if any(
            entry in MINERALS and entry != mineral
            for entry in worker.backpack
        ):
            continue
        quote = turn.vendor_prices[mineral]
        observed_need = price * len(targets)
        needed_units = max(
            0,
            (observed_need - available_gold - inventory_value + quote - 1)
            // quote,
        )
        extra = min(free_slots, needed_units)
        if extra <= 0:
            continue
        projected = replace(
            worker, backpack=worker.backpack + (mineral,) * extra,
        )
        projected_route = _batch_funding_chain(
            turn,
            projected,
            item,
            clock,
            deadline,
            max_expansions,
            preferred_post_id=immediate_route.post_weapon_id,
            preferred_use_target_id=primary.unit_id,
        )
        if projected_route is None:
            continue
        feasible_extra = min(
            extra,
            max(0, turn.rounds_until_night - projected_route.rounds),
        )
        if feasible_extra != extra:
            extra = feasible_extra
            if extra <= 0:
                continue
            projected = replace(
                worker, backpack=worker.backpack + (mineral,) * extra,
            )
            projected_route = _batch_funding_chain(
                turn,
                projected,
                item,
                clock,
                deadline,
                max_expansions,
                preferred_post_id=immediate_route.post_weapon_id,
                preferred_use_target_id=primary.unit_id,
            )
            if projected_route is None:
                continue
        committed_rounds = extra + projected_route.rounds
        if committed_rounds > turn.rounds_until_night:
            continue
        sale = _sale_for_deficit(
            turn,
            Counter(entry for entry in worker.backpack if entry in MINERALS),
            price - available_gold,
        )
        if sale is None:
            continue
        immediate_value = turn.vendor_prices.get(sale[0], 0) * sale[1]
        projected_value = inventory_value + extra * quote
        if (
            immediate_value <= 0
            or projected_value * immediate_route.rounds
            <= immediate_value * committed_rounds
        ):
            continue
        affordable = min(
            len(targets), (available_gold + projected_value) // price,
        )
        if affordable < 2:
            continue
        ordered_targets = (
            primary,
            *(target for target in targets if target.unit_id != primary.unit_id),
        )
        reserved_ids = tuple(
            target.unit_id for target in ordered_targets[:affordable]
        )
        batch = ProcurementBatch(
            item,
            immediate_route.post_weapon_id,
            primary.unit_id,
            mineral,
            mine,
            worker.backpack.count(mineral) + extra,
            quote,
            price,
            reserved_ids,
        )
        return PlannedAction(
            ActionProposal(
                worker.unit_id,
                worker.unit_id,
                collect_command(mine),
                destination=mine,
            ),
            mine,
            _procurement_reason(batch),
            hard_deadline,
            committed_rounds,
            diagnostic={
                "kind": "procurementBatch",
                "extraCollections": extra,
                "reservedTargets": len(reserved_ids),
                "immediateRealizedValue": immediate_value,
                "immediateRounds": immediate_route.rounds,
                "batchRealizedValue": projected_value,
                "batchRounds": committed_rounds,
            },
        )
    return None


def _viable_procurement_targets(
    turn: Turn,
    worker: Unit,
    item: str,
    available_gold: int,
    immediate_route: FundingRoute,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[Unit, ...]:
    context = _ROUTE_SEARCH_CONTEXT.get()
    price = turn.weapon_prices.get(item)
    if price is None:
        return ()
    result = []
    for target in _item_targets(turn, item):
        if clock() >= deadline:
            break
        if (
            target.unit_id != immediate_route.use_target_id
            and context is not None
            and target.unit_id in context.investment_targets
        ):
            continue
        route = (
            immediate_route
            if target.unit_id == immediate_route.use_target_id
            else _funding_chain(
                turn,
                worker,
                item,
                price - available_gold,
                clock,
                deadline,
                max_expansions,
                preferred_post_id=immediate_route.post_weapon_id,
                preferred_use_target_id=target.unit_id,
                gate_new_maintenance=True,
            )
        )
        if route is not None and route.rounds <= turn.rounds_until_night:
            result.append(target)
    return tuple(result)


def _batch_funding_chain(
    turn: Turn,
    worker: Unit,
    item: str,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    preferred_post_id: int,
    preferred_use_target_id: int,
) -> FundingRoute | None:
    minerals = Counter(entry for entry in worker.backpack if entry in MINERALS)
    if not minerals:
        return None
    sale_actions = sum(
        count > 0 and turn.vendor_prices.get(name, 0) > 0
        for name, count in minerals.items()
    )
    if not sale_actions:
        return None
    best = None
    for vendor in turn.zones_of("vendor"):
        vendor_route = _best_adjacent_route(
            turn, worker, vendor, clock, deadline, max_expansions,
        )
        if vendor_route is None:
            continue
        vendor_stand, vendor_cost = vendor_route
        purchase = _purchase_route(
            turn,
            replace(worker, pos=vendor_stand),
            item,
            clock,
            deadline,
            max_expansions,
            preferred_post_id=preferred_post_id,
            preferred_use_target_id=preferred_use_target_id,
            gate_new_maintenance=False,
            maintenance_round_prefix=vendor_cost + sale_actions,
        )
        if purchase is None:
            continue
        candidate = FundingRoute(
            vendor_cost + sale_actions + purchase.rounds,
            vendor,
            vendor_stand,
            purchase.shop,
            purchase.shop_stand,
            purchase.use_target_id,
            purchase.use_stand,
            purchase.post_weapon_id,
            purchase.post_stand,
        )
        if best is None or _funding_route_key(candidate) < _funding_route_key(best):
            best = candidate
    return best


def _mine_action(
    turn: Turn,
    worker: Unit,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    preferred_target: Pos | None = None,
    excluded_targets: set[Pos] | frozenset[Pos] = frozenset(),
) -> PlannedAction | None:
    if worker.backpack_full:
        return None
    mines = sorted(
        (
            target for target in turn.zones_of(*MINERALS)
            if target not in excluded_targets
        ),
        key=lambda pos: (
            pos != preferred_target,
            distance(worker.pos, pos),
            -turn.vendor_prices.get(turn.zones[pos], 0),
            pos.x,
            pos.y,
        ),
    )[:MAX_MINE_CANDIDATES]
    opportunities = []
    reachable = []
    for target in mines:
        if clock() >= deadline:
            break
        route = _best_adjacent_route(
            turn, worker, target, clock, deadline, max_expansions,
        )
        if route is None:
            continue
        stand, travel_rounds = route
        reachable.append((travel_rounds, target))
        value = turn.vendor_prices.get(turn.zones[target], 0)
        realization_rounds = _mine_realization_rounds(
            turn,
            worker,
            target,
            stand,
            travel_rounds,
            clock,
            deadline,
            max_expansions,
        )
        if value <= 0 or realization_rounds is None:
            continue
        if turn.is_day and realization_rounds > turn.rounds_until_night:
            continue
        opportunities.append(MiningOpportunity(
            target, stand, value, realization_rounds,
        ))
    if opportunities:
        best = opportunities[0]
        for candidate in opportunities[1:]:
            left = candidate.value * best.rounds
            right = best.value * candidate.rounds
            if left > right or (
                left == right
                and (
                    candidate.target == preferred_target,
                    -candidate.rounds,
                    -candidate.value,
                    -candidate.target.x,
                    -candidate.target.y,
                ) > (
                    best.target == preferred_target,
                    -best.rounds,
                    -best.value,
                    -best.target.x,
                    -best.target.y,
                )
            ):
                best = candidate
        preferred = next((
            candidate for candidate in opportunities
            if candidate.target == preferred_target
        ), None)
        if (
            preferred is not None
            and best.target != preferred.target
            and best.value * preferred.rounds * 100
            <= preferred.value * best.rounds
            * (100 + MINE_SWITCH_MARGIN_PERCENT)
        ):
            best = preferred
        return _collect_or_move(
            turn,
            worker,
            best.target,
            clock,
            deadline,
            max_expansions,
            diagnostic={
                "kind": "mining",
                "mineral": turn.zones[best.target],
                "target": best.target.dump(),
                "estimatedValue": best.value,
                "estimatedRounds": best.rounds,
            },
        )
    if not reachable:
        return None
    _, target = min(
        reachable,
        key=lambda item: (
            item[0],
            item[1] != preferred_target,
            item[1].x,
            item[1].y,
        ),
    )
    return _collect_or_move(
        turn, worker, target, clock, deadline, max_expansions,
    )


def _mine_realization_rounds(
    turn: Turn,
    worker: Unit,
    target: Pos,
    stand: Pos,
    travel_rounds: int,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> int | None:
    mineral = turn.zones[target]
    projected = replace(
        worker,
        pos=stand,
        backpack=(*worker.backpack, mineral),
    )
    projected_value = sum(
        turn.vendor_prices.get(item, 0)
        for item in projected.backpack
        if item in MINERALS
    )
    for item in _purchase_candidates(turn, projected):
        price = turn.weapon_prices.get(item)
        if (
            price is None
            or price <= turn.gold
            or price > turn.gold + projected_value
        ):
            continue
        route = _funding_chain(
            turn,
            projected,
            item,
            price - turn.gold,
            clock,
            deadline,
            max_expansions,
        )
        if route is not None:
            return travel_rounds + 1 + route.rounds

    best_liquidation = None
    for vendor in turn.zones_of("vendor"):
        route = _best_adjacent_route(
            turn, projected, vendor, clock, deadline, max_expansions,
        )
        if route is None:
            continue
        rounds = travel_rounds + 1 + route[1] + 1
        best_liquidation = (
            rounds
            if best_liquidation is None
            else min(best_liquidation, rounds)
        )
    return best_liquidation


def _fundable_purchase(
    turn: Turn,
    worker: Unit,
    available_gold: int,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[str, int, FundingRoute] | None:
    if not turn.is_day or worker.backpack_full:
        return None
    inventory_value = sum(
        turn.vendor_prices.get(item, 0)
        for item in worker.backpack
        if item in MINERALS
    )
    for item in _purchase_candidates(turn, worker):
        price = turn.weapon_prices.get(item)
        if (
            price is None
            or price <= available_gold
            or price > available_gold + inventory_value
        ):
            continue
        route = _funding_chain(
            turn,
            worker,
            item,
            price - available_gold,
            clock,
            deadline,
            max_expansions,
        )
        if route is not None and route.rounds <= turn.rounds_until_night:
            return item, turn.round_no + turn.rounds_until_night - 1, route
    return None


def _funding_build_action(
    turn: Turn,
    worker: Unit,
    target: Pos,
    kind: str,
    hard_deadline: int | None,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    verified_route: BuildingFundingRoute | None = None,
    available_gold: int | None = None,
) -> PlannedAction | None:
    reason = f"fund:build:{kind}"
    if (
        not turn.is_day
        or turn.station() is None
        or len(turn.weapons()) >= MAX_WEAPONS
        or not _build_target_available(turn, target)
        or hard_deadline is None
        or turn.round_no > hard_deadline
    ):
        return None
    cash = turn.gold if available_gold is None else max(available_gold, 0)
    if cash >= 25:
        build_route = _best_adjacent_route(
            turn, worker, target, clock, deadline, max_expansions,
        )
        if build_route is None or build_route[1] + 1 > turn.rounds_until_night:
            return None
        build_stand, route_cost = build_route
        proposal = (
            ActionProposal(
                worker.unit_id,
                worker.unit_id,
                build_command(target, kind),
                destination=target,
            )
            if worker.pos == build_stand
            else _move_to_stand(
                turn,
                worker,
                build_stand,
                clock,
                deadline,
                max_expansions,
            )
        )
        if proposal is None:
            return None
        return PlannedAction(
            proposal, target, reason, hard_deadline,
            route_cost + 1,
        )
    minerals = Counter(entry for entry in worker.backpack if entry in MINERALS)
    chain_rounds = _building_funding_chain(
        turn,
        worker,
        target,
        25 - cash,
        clock,
        deadline,
        max_expansions,
    )
    route = verified_route or chain_rounds
    if route is None or route.rounds > turn.rounds_until_night:
        return None
    sale = _sale_for_deficit(turn, minerals, 25 - cash)
    if sale is None:
        return None
    name, quantity = sale
    vendor = route.vendor
    if worker.pos == route.vendor_stand:
        return PlannedAction(
            ActionProposal(
                worker.unit_id,
                worker.unit_id,
                sell_command(name, quantity),
            ),
            target,
            reason,
            hard_deadline,
            route.rounds,
        )
    proposal = _move_to_stand(
        turn,
        worker,
        route.vendor_stand,
        clock,
        deadline,
        max_expansions,
    )
    if proposal is None:
        return None
    return PlannedAction(
        proposal,
        target,
        reason,
        hard_deadline,
        route.rounds,
    )


def _building_funding_chain(
    turn: Turn,
    worker: Unit,
    target: Pos,
    deficit: int,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> BuildingFundingRoute | None:
    minerals = Counter(entry for entry in worker.backpack if entry in MINERALS)
    sale_actions = 0
    remaining = deficit
    for name in sorted(
        minerals,
        key=lambda entry: (turn.vendor_prices.get(entry, 0), entry),
        reverse=True,
    ):
        price = turn.vendor_prices.get(name, 0)
        if price <= 0:
            continue
        used = min(minerals[name], (remaining + price - 1) // price)
        if used:
            remaining -= used * price
            sale_actions += 1
        if remaining <= 0:
            break
    if remaining > 0:
        return None
    best = None
    for vendor in turn.zones_of("vendor"):
        vendor_route = _best_adjacent_route(
            turn, worker, vendor, clock, deadline, max_expansions,
        )
        if vendor_route is None:
            continue
        vendor_stand, vendor_cost = vendor_route
        from_vendor = replace(worker, pos=vendor_stand)
        build_route = _best_adjacent_route(
            turn,
            from_vendor,
            target,
            clock,
            deadline,
            max_expansions,
        )
        if build_route is None:
            continue
        build_stand, build_cost = build_route
        rounds = vendor_cost + sale_actions + build_cost + 1
        candidate = BuildingFundingRoute(
            rounds, vendor, vendor_stand, build_stand,
        )
        if best is None or (
            candidate.rounds, candidate.vendor.x, candidate.vendor.y
        ) < (
            best.rounds, best.vendor.x, best.vendor.y
        ):
            best = candidate
    return best


def _funding_action(
    turn: Turn,
    worker: Unit,
    item: str,
    hard_deadline: int | None,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    preferred_post_id: int | None = None,
    preferred_use_target_id: int | None = None,
    verified_route: FundingRoute | None = None,
    gate_new_maintenance: bool = True,
) -> PlannedAction | None:
    if hard_deadline is not None and turn.round_no > hard_deadline:
        return None
    if (
        item not in worker.backpack
        and preferred_post_id is not None
        and not _fund_target_still_needs(
            turn, worker, item, preferred_use_target_id,
        )
    ):
        return _post_action(
            turn,
            worker,
            item,
            preferred_post_id,
            hard_deadline,
            clock,
            deadline,
            max_expansions,
        )
    if item in worker.backpack:
        route = verified_route or _held_item_route(
            turn,
            worker,
            item,
            clock,
            deadline,
            max_expansions,
            preferred_post_id=preferred_post_id,
            preferred_use_target_id=preferred_use_target_id,
        )
        if route is None or route.rounds > turn.rounds_until_night:
            return None
        reason = _fund_reason(item, route)
        target = turn.unit(route.use_target_id) if route.use_target_id else None
        if item == "Medicine" and worker.pos == route.use_stand:
            if worker.health >= _max_role_health(worker):
                return None
            return PlannedAction(
                ActionProposal(
                    worker.unit_id,
                    worker.unit_id,
                    use_command(item),
                    item_costs=(item,),
                ),
                worker.pos,
                reason,
                hard_deadline,
                route.rounds,
            )
        if worker.pos == route.use_stand:
            return PlannedAction(
                ActionProposal(
                    worker.unit_id,
                    worker.unit_id,
                    use_command(item, target.pos),
                    item_costs=(item,),
                ),
                target.pos,
                reason,
                hard_deadline,
                route.rounds,
            )
        proposal = _move_to_stand(
            turn,
            worker,
            route.use_stand,
            clock,
            deadline,
            max_expansions,
        )
        if proposal is None:
            return None
        return PlannedAction(
            proposal,
            target.pos,
            reason,
            hard_deadline,
            route.rounds,
        )

    price = turn.weapon_prices.get(item)
    if price is None:
        return None
    if turn.gold >= price:
        route = verified_route or _purchase_route(
            turn,
            worker,
            item,
            clock,
            deadline,
            max_expansions,
            preferred_post_id=preferred_post_id,
            preferred_use_target_id=preferred_use_target_id,
            gate_new_maintenance=gate_new_maintenance,
        )
        if route is None or route.rounds > turn.rounds_until_night:
            return None
        reason = _fund_reason(item, route)
        if worker.pos == route.shop_stand:
            proposal = ActionProposal(
                worker.unit_id,
                worker.unit_id,
                buy_command(item),
            )
        else:
            proposal = _move_to_stand(
                turn,
                worker,
                route.shop_stand,
                clock,
                deadline,
                max_expansions,
            )
        if proposal is None:
            return None
        return PlannedAction(
            proposal,
            route.shop,
            reason,
            hard_deadline,
            route.rounds,
        )

    minerals = Counter(entry for entry in worker.backpack if entry in MINERALS)
    if not minerals:
        return None
    chain = _funding_chain(
        turn,
        worker,
        item,
        price - turn.gold,
        clock,
        deadline,
        max_expansions,
        preferred_post_id=preferred_post_id,
        preferred_use_target_id=preferred_use_target_id,
        gate_new_maintenance=gate_new_maintenance,
    )
    route = verified_route or chain
    if route is None or route.rounds > turn.rounds_until_night:
        return None
    reason = _fund_reason(item, route)
    sale = _sale_for_deficit(turn, minerals, price - turn.gold)
    if sale is None:
        return None
    name, quantity = sale
    vendor = route.vendor
    if vendor is None:
        return None
    if worker.pos == route.vendor_stand:
        return PlannedAction(
            ActionProposal(
                worker.unit_id,
                worker.unit_id,
                sell_command(name, quantity),
            ),
            vendor,
            reason,
            hard_deadline,
            route.rounds,
        )
    proposal = _move_to_stand(
        turn,
        worker,
        route.vendor_stand,
        clock,
        deadline,
        max_expansions,
    )
    if proposal is None:
        return None
    return PlannedAction(
        proposal,
        vendor,
        reason,
        hard_deadline,
        route.rounds,
    )


def _sale_for_deficit(
    turn: Turn,
    minerals: Counter[str],
    deficit: int,
) -> tuple[str, int] | None:
    choices = sorted(
        (
            (turn.vendor_prices.get(name, 0), name, count)
            for name, count in minerals.items()
        ),
        reverse=True,
    )
    for price, name, count in choices:
        if price <= 0:
            continue
        quantity = min(count, (deficit + price - 1) // price)
        if quantity:
            return name, quantity
    return None


def _funding_chain(
    turn: Turn,
    worker: Unit,
    item: str,
    deficit: int,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    preferred_post_id: int | None = None,
    preferred_use_target_id: int | None = None,
    gate_new_maintenance: bool = True,
) -> FundingRoute | None:
    minerals = Counter(entry for entry in worker.backpack if entry in MINERALS)
    sale_actions = 0
    remaining = deficit
    for name in sorted(
        minerals,
        key=lambda entry: (turn.vendor_prices.get(entry, 0), entry),
        reverse=True,
    ):
        price = turn.vendor_prices.get(name, 0)
        if price <= 0:
            continue
        used = min(minerals[name], (remaining + price - 1) // price)
        if used:
            remaining -= used * price
            sale_actions += 1
        if remaining <= 0:
            break
    if remaining > 0:
        return None

    best: FundingRoute | None = None
    for vendor in turn.zones_of("vendor"):
        vendor_route = _best_adjacent_route(
            turn, worker, vendor, clock, deadline, max_expansions,
        )
        if vendor_route is None:
            continue
        vendor_stand, vendor_cost = vendor_route
        at_vendor = replace(worker, pos=vendor_stand)
        purchase = _purchase_route(
            turn,
            at_vendor,
            item,
            clock,
            deadline,
            max_expansions,
            preferred_post_id=preferred_post_id,
            preferred_use_target_id=preferred_use_target_id,
            gate_new_maintenance=gate_new_maintenance,
            maintenance_round_prefix=vendor_cost + sale_actions,
        )
        if purchase is None:
            continue
        candidate = FundingRoute(
            vendor_cost + sale_actions + purchase.rounds,
            vendor,
            vendor_stand,
            purchase.shop,
            purchase.shop_stand,
            purchase.use_target_id,
            purchase.use_stand,
            purchase.post_weapon_id,
            purchase.post_stand,
        )
        if (
            gate_new_maintenance
            and not _maintenance_purchase_worthwhile(
                turn, worker, item, candidate,
            )
        ):
            continue
        if best is None or _funding_route_key(candidate) < _funding_route_key(best):
            best = candidate
    return best


def _purchase_route(
    turn: Turn,
    worker: Unit,
    item: str,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    preferred_post_id: int | None = None,
    preferred_use_target_id: int | None = None,
    gate_new_maintenance: bool = True,
    maintenance_round_prefix: int = 0,
) -> FundingRoute | None:
    if (
        gate_new_maintenance
        and item == "WallFixer"
        and preferred_use_target_id is None
    ):
        routes = tuple(
            route
            for target in _item_targets(turn, item)
            for route in (
                _purchase_route(
                    turn,
                    worker,
                    item,
                    clock,
                    deadline,
                    max_expansions,
                    preferred_post_id=preferred_post_id,
                    preferred_use_target_id=target.unit_id,
                    gate_new_maintenance=False,
                ),
            )
            if route is not None
            and _maintenance_purchase_worthwhile(
                turn,
                worker,
                item,
                replace(
                    route,
                    rounds=maintenance_round_prefix + route.rounds,
                ),
            )
        )
        return min(routes, key=_funding_route_key, default=None)
    best = None
    for shop in turn.zones_of("weaponShop"):
        shop_route = _best_adjacent_route(
            turn, worker, shop, clock, deadline, max_expansions,
        )
        if shop_route is None:
            continue
        shop_stand, shop_cost = shop_route
        at_shop = replace(worker, pos=shop_stand)
        held = _held_item_route(
            turn,
            at_shop,
            item,
            clock,
            deadline,
            max_expansions,
            preferred_post_id=preferred_post_id,
            preferred_use_target_id=preferred_use_target_id,
            gate_new_maintenance=gate_new_maintenance,
        )
        if held is None:
            continue
        candidate = FundingRoute(
            shop_cost + 1 + held.rounds,
            None,
            None,
            shop,
            shop_stand,
            held.use_target_id,
            held.use_stand,
            held.post_weapon_id,
            held.post_stand,
        )
        if (
            gate_new_maintenance
            and not _maintenance_purchase_worthwhile(
                turn,
                worker,
                item,
                replace(
                    candidate,
                    rounds=maintenance_round_prefix + candidate.rounds,
                ),
            )
        ):
            continue
        if best is None or _funding_route_key(candidate) < _funding_route_key(best):
            best = candidate
    return best


def _maintenance_purchase_worthwhile(
    turn: Turn,
    worker: Unit,
    item: str,
    route: FundingRoute,
) -> bool:
    if item == "Medicine":
        return _medicine_purchase_worthwhile(worker, route.rounds)
    elif item == "WallFixer":
        target = (
            turn.unit(route.use_target_id)
            if route.use_target_id is not None else None
        )
        if target is None or target.kind != WALL:
            return False
        recovery = _max_building_health(target) - target.health
        urgent = _wall_repair_urgent(turn, target)
    else:
        return True
    return recovery > 0 and (urgent or recovery >= max(route.rounds, 1))


def _wall_repair_urgent(turn: Turn, target: Unit) -> bool:
    return any(
        robot.target_team == turn.team_type
        and distance(robot.pos, target.pos) <= ROBOT_ATTACK_RANGE
        and robot.attack_power >= target.health
        for robot in turn.robots
    )


def _medicine_purchase_worthwhile(worker: Unit, rounds: int) -> bool:
    recovery = _max_role_health(worker) - worker.health
    return recovery > 0 and (
        worker.health <= EMERGENCY_MEDICINE_HEALTH
        or recovery >= max(rounds, 1)
    )


def _held_item_route(
    turn: Turn,
    worker: Unit,
    item: str,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    preferred_post_id: int | None = None,
    preferred_use_target_id: int | None = None,
    gate_new_maintenance: bool = False,
) -> FundingRoute | None:
    targets = _item_targets(turn, item)
    context = _ROUTE_SEARCH_CONTEXT.get()
    if context is not None:
        targets = tuple(
            target for target in targets
            if context.investment_owners.get(
                target.unit_id, worker.unit_id,
            ) == worker.unit_id
            and (
                target.unit_id not in context.investment_targets
                or context.investment_owners.get(target.unit_id) == worker.unit_id
            )
        )
    if preferred_use_target_id is not None:
        targets = tuple(
            target for target in targets
            if target.unit_id == preferred_use_target_id
        )
    best = None
    target_choices = (None,) if item == "Medicine" else targets
    for target in target_choices:
        use_routes = (
            ((worker.pos, 0, None),)
            if target is None
            else tuple(
                (stand, cost, target.unit_id)
                for stand, cost in _routes_to_adjacent(
                    turn, worker, target.pos, clock, deadline,
                    max_expansions,
                )
            )
        )
        for use_stand, use_cost, target_id in use_routes:
            at_use = replace(worker, pos=use_stand)
            for weapon, post_stand, post_cost in _post_routes(
                turn,
                at_use,
                clock,
                deadline,
                max_expansions,
                preferred_post_id=preferred_post_id,
            ):
                candidate = FundingRoute(
                    use_cost + 1 + post_cost,
                    None,
                    None,
                    None,
                    None,
                    target_id,
                    use_stand,
                    weapon.unit_id,
                    post_stand,
                )
                if (
                    gate_new_maintenance
                    and not _maintenance_purchase_worthwhile(
                        turn, worker, item, candidate,
                    )
                ):
                    continue
                key = (_funding_route_key(candidate), target_id or 0)
                if best is None or key < best[0]:
                    best = (key, candidate)
        context = _ROUTE_SEARCH_CONTEXT.get()
        if (
            best is not None
            and context is not None
            and context.truncated_reason in ("deadline", "search_limit")
        ):
            break
    return best[1] if best is not None else None


def _post_routes(
    turn: Turn,
    worker: Unit,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    preferred_post_id: int | None = None,
) -> tuple[tuple[Unit, Pos, int], ...]:
    result = []
    for weapon in turn.weapons():
        if preferred_post_id is not None and weapon.unit_id != preferred_post_id:
            continue
        staffed_by_other = any(
            role.unit_id != worker.unit_id
            and distance(role.pos, weapon.pos) == 1
            for role in turn.controllable()
        )
        if staffed_by_other:
            continue
        for stand, cost in _routes_to_adjacent(
            turn, worker, weapon.pos, clock, deadline, max_expansions,
        ):
            result.append((weapon, stand, cost))
    return tuple(result)


def _post_action(
    turn: Turn,
    worker: Unit,
    item: str,
    post_weapon_id: int | None,
    hard_deadline: int | None,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    routes = _post_routes(
        turn,
        worker,
        clock,
        deadline,
        max_expansions,
        preferred_post_id=post_weapon_id,
    )
    if not routes:
        return None
    weapon, stand, route_cost = min(
        routes,
        key=lambda route: (route[2], route[0].unit_id, route[1].x, route[1].y),
    )
    if route_cost > turn.rounds_until_night or worker.pos == stand:
        return None
    path = _search_path(
        turn,
        worker,
        stand,
        clock=clock,
        deadline=deadline,
        max_expansions=max_expansions,
    )
    if path.status != "found" or path.step is None:
        return None
    return PlannedAction(
        ActionProposal(
            worker.unit_id,
            worker.unit_id,
            move_command(path.step),
            destination=path.step,
        ),
        weapon.pos,
        f"fund:{item}:{weapon.unit_id}",
        hard_deadline,
        route_cost,
    )


def _funding_route_key(route: FundingRoute) -> tuple[int, int, int, int, int]:
    vendor = route.vendor or Pos(10**9, 10**9)
    shop = route.shop or Pos(10**9, 10**9)
    return (
        route.rounds,
        vendor.x,
        vendor.y,
        shop.x,
        shop.y,
    )


def _fund_reason(item: str, route: FundingRoute) -> str:
    return (
        f"fund:{item}:{route.post_weapon_id}:"
        f"{route.use_target_id or 0}"
    )


def _fund_target_still_needs(
    turn: Turn,
    worker: Unit,
    item: str,
    target_id: int | None,
) -> bool:
    if item == "Medicine":
        return worker.health < _max_role_health(worker)
    target = turn.unit(target_id) if target_id is not None else None
    if target is None:
        return False
    if item == "WallFixer":
        return target.kind == WALL and target.health < _max_building_health(target)
    if not item.endswith(("1", "2")):
        return False
    required_level = int(item[-1])
    if item.startswith("WeaponUpgradeVoucher"):
        return target.kind in TOWER_TYPES and target.level == required_level
    if item.startswith("StationUpgradeVoucher"):
        return target.kind == STATION and target.level == required_level
    if item.startswith("WallUpgradeVoucher"):
        return target.kind == WALL and target.level == required_level
    return False


def _collect_or_move(
    turn: Turn,
    worker: Unit,
    target: Pos,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    diagnostic: dict[str, object] | None = None,
) -> PlannedAction | None:
    reason = f"mine:{turn.zones[target]}"
    if distance(worker.pos, target) == 1:
        return PlannedAction(ActionProposal(
            worker.unit_id,
            worker.unit_id,
            collect_command(target),
        ), target, reason, diagnostic=diagnostic)
    candidate = _move_adjacent(
        turn, worker, target, reason, None,
        clock, deadline, max_expansions,
    )
    return replace(candidate, diagnostic=diagnostic) if candidate is not None else None


def _sell_or_move(
    turn: Turn,
    worker: Unit,
    target: Pos,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    if distance(worker.pos, target) == 1:
        minerals = Counter(item for item in worker.backpack if item in MINERALS)
        if not minerals:
            return None
        name = max(
            minerals,
            key=lambda item: (turn.vendor_prices.get(item, 0), item),
        )
        return PlannedAction(ActionProposal(
            worker.unit_id,
            worker.unit_id,
            sell_command(name, minerals[name]),
        ))
    return _move_adjacent(
        turn, worker, target, "vendor", None,
        clock, deadline, max_expansions,
    )


def _buy_or_move(
    turn: Turn,
    worker: Unit,
    target: Pos,
    item: str,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    if not _planned_purchase_valid(turn, worker, item):
        return None
    if distance(worker.pos, target) == 1:
        return PlannedAction(ActionProposal(
            worker.unit_id,
            worker.unit_id,
            buy_command(item),
        ))
    return _move_adjacent(
        turn, worker, target, f"shop:{item}", None,
        clock, deadline, max_expansions,
    )


def _purchase_candidates(turn: Turn, worker: Unit) -> tuple[str, ...]:
    candidates: list[str] = []
    if worker.health < _max_role_health(worker) and "Medicine" in turn.weapon_prices:
        candidates.append("Medicine")
    damaged_wall = next((
        wall for wall in turn.walls()
        if wall.health < _max_building_health(wall)
    ), None)
    if damaged_wall is not None and "WallFixer" in turn.weapon_prices:
        candidates.append("WallFixer")
    for target in (*turn.weapons(), *((turn.station(),) if turn.station() else ())):
        prefix = "Weapon" if target.kind in TOWER_TYPES else "Station"
        if target.level in (1, 2):
            item = f"{prefix}UpgradeVoucher{target.level}"
            if item in turn.weapon_prices and item not in candidates:
                candidates.append(item)
    wall_positions = {wall.pos for wall in turn.walls()}
    context = _ROUTE_SEARCH_CONTEXT.get()
    layout_walls = context.wall_upgrade_targets if context is not None else ()
    wall_levels = (
        {wall.level for wall in turn.walls()}
        if len(turn.weapons()) >= MAX_WEAPONS
        and bool(layout_walls)
        and set(layout_walls).issubset(wall_positions)
        else set()
    )
    for required_level in (1, 2):
        item = f"WallUpgradeVoucher{required_level}"
        if required_level in wall_levels and item in turn.weapon_prices:
            candidates.append(item)
    return tuple(candidates)


def _purchase_is_timely(
    turn: Turn,
    worker: Unit,
    shop: Pos,
    item: str,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> bool:
    target = _item_target(turn, item)
    if item != "Medicine" and target is None:
        return False
    if not turn.is_day:
        return item == "Medicine" and distance(worker.pos, shop) == 1

    best_rounds: int | None = None
    for shop_stand in _adjacent_stands(turn, worker, shop):
        path = _search_path(
            turn,
            worker,
            shop_stand,
            clock=clock,
            deadline=deadline,
            max_expansions=max_expansions,
        )
        if path.status == "already_there":
            shop_rounds = 0
        elif path.status == "found" and path.cost is not None:
            shop_rounds = path.cost
        else:
            if path.status == "deadline":
                return False
            continue

        use_rounds = 1
        if target is not None:
            at_shop = replace(worker, pos=shop_stand)
            route = _route_cost_to_adjacent(
                turn,
                at_shop,
                target.pos,
                clock,
                deadline,
                max_expansions,
            )
            if route is None:
                continue
            use_rounds += route
        total_rounds = shop_rounds + 1 + use_rounds
        best_rounds = (
            total_rounds
            if best_rounds is None
            else min(best_rounds, total_rounds)
        )
    return best_rounds is not None and best_rounds <= turn.rounds_until_night


def _route_cost_to_adjacent(
    turn: Turn,
    worker: Unit,
    target: Pos,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> int | None:
    routes = _routes_to_adjacent(
        turn, worker, target, clock, deadline, max_expansions,
    )
    return min((cost for _, cost in routes), default=None)


def _search_path(
    turn: Turn,
    worker: Unit,
    stand: Pos,
    *,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PathResult:
    context = _ROUTE_SEARCH_CONTEXT.get()
    turn_id = id(turn)
    key = (turn_id, worker.unit_id, worker.pos, stand, max_expansions)
    if context is not None:
        context.turns[turn_id] = turn
        cached = context.paths.get(key)
        if cached is not None:
            if clock() >= deadline:
                return PathResult("deadline", None, 0, None)
            context.path_cache_hits += 1
            return cached
    # Prefer progress within equal-cost fronts under the same expansion cap.
    result = next_step(
        turn, worker, stand, clock=clock, deadline=deadline,
        max_expansions=max_expansions,
        prefer_deep_ties=True,
    )
    if context is not None:
        context.path_computations += 1
        context.path_expansions += result.expansions
        if result.status != "deadline":
            context.paths[key] = result
    return result


def _routes_to_adjacent(
    turn: Turn,
    worker: Unit,
    target: Pos,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[tuple[Pos, int], ...]:
    context = _ROUTE_SEARCH_CONTEXT.get()
    turn_id = id(turn)
    cache_key = (turn_id, worker.unit_id, worker.pos, target, max_expansions)
    if context is not None:
        context.turns[turn_id] = turn
    if context is not None and cache_key in context.routes:
        if clock() >= deadline:
            context.truncated_reason = "deadline"
            return ()
        context.cache_hits += 1
        return context.routes[cache_key]
    routes = []
    complete = True
    for stand in _adjacent_stands(turn, worker, target):
        if (
            context is not None
            and context.path_searches >= MAX_ECONOMY_PATH_SEARCHES
        ):
            context.truncated_reason = "search_limit"
            return ()
        if context is not None:
            context.path_searches += 1
        path = _search_path(
            turn,
            worker,
            stand,
            clock=clock,
            deadline=deadline,
            max_expansions=max_expansions,
        )
        if path.status == "already_there":
            routes.append((stand, 0))
        elif path.status == "found" and path.cost is not None:
            routes.append((stand, path.cost))
        elif path.status == "deadline":
            if context is not None:
                context.truncated_reason = path.status
            return ()
        elif path.status == "expansion_limit":
            if context is not None:
                context.truncated_reason = path.status
            complete = False
            continue
    result = tuple(routes)
    if context is not None and complete:
        context.routes[cache_key] = result
    return result


def _best_adjacent_route(
    turn: Turn,
    worker: Unit,
    target: Pos,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[Pos, int] | None:
    routes = _routes_to_adjacent(
        turn, worker, target, clock, deadline, max_expansions,
    )
    return min(
        routes,
        key=lambda route: (route[1], route[0].x, route[0].y),
        default=None,
    )


def _move_to_stand(
    turn: Turn,
    worker: Unit,
    stand: Pos | None,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> ActionProposal | None:
    if stand is None or worker.pos == stand:
        return None
    path = _search_path(
        turn,
        worker,
        stand,
        clock=clock,
        deadline=deadline,
        max_expansions=max_expansions,
    )
    if path.status != "found" or path.step is None:
        return None
    return ActionProposal(
        worker.unit_id,
        worker.unit_id,
        move_command(path.step),
        destination=path.step,
    )


def _item_target(turn: Turn, item: str) -> Unit | None:
    return next(iter(_item_targets(turn, item)), None)


def _item_targets(turn: Turn, item: str) -> tuple[Unit, ...]:
    if item == "WallFixer":
        targets = turn.walls()
    elif item.startswith("WeaponUpgradeVoucher"):
        targets = turn.weapons()
    elif item.startswith("StationUpgradeVoucher"):
        targets = (turn.station(),) if turn.station() else ()
    elif item.startswith("WallUpgradeVoucher"):
        targets = turn.walls()
    else:
        return ()
    return tuple(sorted(
        (target for target in targets if _item_matches_target(item, target)),
        key=lambda target: target.unit_id,
    ))


def _item_matches_target(item: str, target: Unit) -> bool:
    if item == "WallFixer":
        return target.kind == WALL and target.health < _max_building_health(target)
    prefixes = {
        "WeaponUpgradeVoucher": TOWER_TYPES,
        "StationUpgradeVoucher": (STATION,),
        "WallUpgradeVoucher": (WALL,),
    }
    prefix = next((
        name for name in prefixes
        if item in (f"{name}1", f"{name}2")
    ), None)
    if prefix is None:
        return False
    required_level = int(item[-1])
    return target.level == required_level and target.kind in prefixes[prefix]


def _move_adjacent(
    turn: Turn,
    worker: Unit,
    target: Pos,
    reason: str,
    deadline_round: int | None,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    for stand in _adjacent_stands(turn, worker, target):
        path = _search_path(
            turn,
            worker,
            stand,
            clock=clock,
            deadline=deadline,
            max_expansions=max_expansions,
        )
        if path.status == "already_there":
            return None
        if path.status == "found" and path.step is not None:
            return PlannedAction(
                ActionProposal(
                    worker.unit_id,
                    worker.unit_id,
                    move_command(path.step),
                    destination=path.step,
                ),
                target,
                reason,
                deadline_round,
            )
        if path.status == "deadline":
            return None
        if path.status == "expansion_limit":
            continue
    return None


def _adjacent_stands(turn: Turn, worker: Unit, target: Pos) -> tuple[Pos, ...]:
    blocked = turn.blocked(worker)
    positions = (
        Pos(target.x + dx, target.y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if dx or dy
    )
    return tuple(sorted(
        (pos for pos in positions if turn.land(pos) and pos not in blocked),
        key=lambda pos: (distance(worker.pos, pos), pos.x, pos.y),
    ))


def _build_target_available(turn: Turn, target: Pos) -> bool:
    return turn.land(target) and target not in turn.occupied_cells()


def _build_plan_valid(
    turn: Turn,
    worker: Unit,
    target: Pos,
    kind: str,
) -> bool:
    if not turn.is_day or not _build_target_available(turn, target):
        return False
    if kind in TOWER_TYPES:
        return (
            turn.station() is not None
            and len(turn.weapons()) < MAX_WEAPONS
            and turn.gold >= 25
        )
    return kind == WALL and "stone" in worker.backpack


def _planned_purchase_valid(turn: Turn, worker: Unit, item: str) -> bool:
    price = turn.weapon_prices.get(item)
    return (
        worker.capacity is not None
        and not worker.backpack_full
        and price is not None
        and price <= turn.gold
        and item in _purchase_candidates(turn, worker)
    )


def _nearest(origin: Pos, positions) -> Pos | None:
    choices = tuple(positions)
    if not choices:
        return None
    return min(choices, key=lambda pos: (distance(origin, pos), pos.x, pos.y))


def _max_role_health(role: Unit) -> int:
    return 200 if role.kind == PIONEER else 220


def _max_building_health(unit: Unit) -> int:
    if unit.kind == STATION:
        return 1500 * max(unit.level, 1)
    return 500 + 500 * max(unit.level, 1)
