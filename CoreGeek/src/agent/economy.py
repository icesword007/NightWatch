import time
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable

from .actions import ActionProposal, PlannedAction
from .grid import next_step
from .protocol import (
    PIONEER,
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
from .state import SessionState

MINERALS = ("stone", "iron", "copper")
TOWER_ORDER = ("gatling", "railgun", "rocket")
MAX_WEAPONS = 3


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
class BuildingFundingRoute:
    rounds: int
    vendor: Pos
    vendor_stand: Pos
    build_stand: Pos


def propose_economy(
    turn: Turn,
    state: SessionState,
    *,
    clock: Callable[[], float] = time.monotonic,
    deadline: float,
    max_expansions: int,
    need_wall: bool = False,
    reserved_role_ids: frozenset[int] = frozenset(),
) -> tuple[PlannedAction, ...]:
    candidates: list[PlannedAction] = []
    maintained_roles: set[int] = set()
    claimed_build_targets: set[Pos] = set()
    claimed_tower_types: set[str] = set()
    wall_trial_claimed = state.wall_trial_started or any(
        plan.reason == "build:wall" for plan in state.plans.values()
    )
    claimed_gold = 0
    failed_builds = {
        completed.pending.target
        for completed in state.action_history
        if completed.pending.action == "build"
        and completed.success is False
        and completed.pending.target is not None
        and completed.pending.source_session == state.session_index
    }

    for role in turn.controllable():
        if clock() >= deadline:
            break
        plan = state.plans.get(role.unit_id)
        candidate = None
        if plan is None or not plan.reason.startswith("fund:"):
            candidate = _maintenance_action(
                turn, role, clock, deadline, max_expansions,
            )
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
            maintained_roles.add(role.unit_id)

    for worker in turn.workers():
        if clock() >= deadline:
            break
        if worker.unit_id in maintained_roles:
            continue
        if worker.unit_id in reserved_role_ids:
            continue
        candidate = _continue_plan(
            turn, state, worker, failed_builds,
            clock, deadline, max_expansions,
        )
        if candidate is None and need_wall and not wall_trial_claimed:
            candidate = _wall_action(
                turn,
                worker,
                failed_builds | claimed_build_targets | {
                    plan.target for plan in state.plans.values()
                },
                clock, deadline, max_expansions,
            )
            if candidate is not None:
                wall_trial_claimed = True
        if candidate is None:
            candidate = _tower_action(
                turn,
                worker,
                failed_builds | claimed_build_targets,
                claimed_tower_types,
                turn.gold - claimed_gold,
                clock,
                deadline,
                max_expansions,
            )
        if candidate is None:
            candidate = _trade_or_mine(
                turn, worker, turn.gold - claimed_gold,
                clock, deadline, max_expansions,
            )
        if candidate is not None:
            candidates.append(candidate)
            if (
                candidate.plan_reason
                and (
                    candidate.plan_reason.startswith("build:")
                    or candidate.plan_reason.startswith("fund:build:")
                )
                and candidate.plan_target is not None
            ):
                claimed_build_targets.add(candidate.plan_target)
                claimed_tower_types.add(candidate.plan_reason.rsplit(":", 1)[1])
            action = candidate.proposal.command["action"]
            if action == "build" and candidate.proposal.command.get("name") in TOWER_TYPES:
                claimed_gold += 25
            elif action == "buy":
                name = candidate.proposal.command["name"]
                claimed_gold += turn.weapon_prices[name]
    return tuple(candidates)


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
) -> PlannedAction | None:
    items = Counter(worker.backpack)
    if items["Medicine"] and worker.health < _max_role_health(worker):
        return PlannedAction(ActionProposal(
            worker.unit_id, worker.unit_id, use_command("Medicine"),
        ))

    for item in worker.backpack:
        target = _item_target(turn, item)
        if target is None:
            continue
        if distance(worker.pos, target.pos) == 1:
            return PlannedAction(ActionProposal(
                worker.unit_id,
                worker.unit_id,
                use_command(item, target.pos),
                item_costs=(item,),
            ))
        return _move_adjacent(
            turn,
            worker,
            target.pos,
            reason=f"use:{item}:{target.unit_id}",
            deadline_round=None,
            clock=clock,
            deadline=deadline,
            max_expansions=max_expansions,
        )
    return None


def _continue_plan(
    turn: Turn,
    state: SessionState,
    worker: Unit,
    failed_builds: set[Pos],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    plan = state.plans.get(worker.unit_id)
    if plan is None or plan.reason == "s0_probe":
        return None
    if plan.deadline_round is not None and turn.round_no > plan.deadline_round:
        return None
    if plan.reason.startswith("build:"):
        kind = plan.reason.split(":", 1)[1]
        if (
            plan.target in failed_builds
            or not _build_plan_valid(turn, worker, plan.target, kind)
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
        if _item_target(turn, parts[1]) != target:
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
        )
    if plan.reason.startswith("mine:"):
        if worker.backpack_full or turn.zones.get(plan.target) not in MINERALS:
            return None
        return _collect_or_move(
            turn, worker, plan.target, clock, deadline, max_expansions,
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


def _wall_action(
    turn: Turn,
    worker: Unit,
    excluded: set[Pos],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    if not turn.is_day or "stone" not in worker.backpack:
        return None
    target = _nearest(worker.pos, (
        pos for pos in wall_build_positions(turn) if pos not in excluded
    ))
    if target is None:
        return None
    return _build_or_move(
        turn, worker, target, WALL, clock, deadline, max_expansions,
    )


def _tower_action(
    turn: Turn,
    worker: Unit,
    excluded: set[Pos],
    claimed_types: set[str],
    available_gold: int,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    weapons = turn.weapons()
    if (
        not turn.is_day
        or turn.station() is None
        or len(weapons) + len(claimed_types) >= MAX_WEAPONS
    ):
        return None
    existing = Counter(weapon.kind for weapon in weapons)
    kind = next((
        candidate for candidate in TOWER_ORDER
        if existing[candidate] == 0 and candidate not in claimed_types
    ), None)
    if kind is None:
        kind = next((
            candidate for candidate in TOWER_ORDER
            if existing[candidate] + (candidate in claimed_types) < 3
        ), None)
    if kind is None:
        return None
    target = _nearest(worker.pos, (
        pos for pos in weapon_build_positions(turn) if pos not in excluded
    ))
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
        )
    return _build_or_move(
        turn, worker, target, kind, clock, deadline, max_expansions,
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


def _trade_or_mine(
    turn: Turn,
    worker: Unit,
    available_gold: int,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    minerals = Counter(item for item in worker.backpack if item in MINERALS)
    vendors = turn.zones_of("vendor")
    funded = _fundable_purchase(
        turn, worker, available_gold, clock, deadline, max_expansions,
    )
    if funded is not None:
        item, hard_deadline, route = funded
        candidate = _funding_action(
            turn,
            worker,
            item,
            hard_deadline,
            clock,
            deadline,
            max_expansions,
            verified_route=route,
        )
        if candidate is not None:
            return candidate
    if minerals and vendors:
        adjacent = next((pos for pos in vendors if distance(worker.pos, pos) == 1), None)
        if adjacent is not None:
            return _sell_or_move(
                turn, worker, adjacent, clock, deadline, max_expansions,
            )
        if worker.backpack_full or turn.rounds_until_night <= 12:
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

    mines = turn.zones_of(*MINERALS)
    if not mines or worker.backpack_full:
        return None
    target = min(
        mines,
        key=lambda pos: (
            distance(worker.pos, pos),
            -turn.vendor_prices.get(turn.zones[pos], 0),
            pos.x,
            pos.y,
        ),
    )
    return _collect_or_move(
        turn, worker, target, clock, deadline, max_expansions,
    )


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
    if turn.gold >= 25:
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
        25 - turn.gold,
        clock,
        deadline,
        max_expansions,
    )
    route = verified_route or chain_rounds
    if route is None or route.rounds > turn.rounds_until_night:
        return None
    sale = _sale_for_deficit(turn, minerals, 25 - turn.gold)
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
        route = _held_item_route(
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
        route = _purchase_route(
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
) -> FundingRoute | None:
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
        if best is None or _funding_route_key(candidate) < _funding_route_key(best):
            best = candidate
    return best


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
) -> FundingRoute | None:
    target = (
        turn.unit(preferred_use_target_id)
        if preferred_use_target_id is not None
        else _item_target(turn, item)
    )
    if item == "Medicine":
        use_routes = ((worker.pos, 0, None),)
    elif target is None or not _fund_target_still_needs(
        turn, worker, item, target.unit_id,
    ):
        return None
    else:
        use_routes = tuple(
            (stand, cost, target.unit_id)
            for stand, cost in _routes_to_adjacent(
                turn, worker, target.pos, clock, deadline, max_expansions,
            )
        )
    best = None
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
            if best is None or _funding_route_key(candidate) < _funding_route_key(best):
                best = candidate
    return best


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
    path = next_step(
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
) -> PlannedAction | None:
    reason = f"mine:{turn.zones[target]}"
    if distance(worker.pos, target) == 1:
        return PlannedAction(ActionProposal(
            worker.unit_id,
            worker.unit_id,
            collect_command(target),
        ), target, reason)
    return _move_adjacent(
        turn, worker, target, reason, None,
        clock, deadline, max_expansions,
    )


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
        path = next_step(
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
            if path.status in ("deadline", "expansion_limit"):
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


def _routes_to_adjacent(
    turn: Turn,
    worker: Unit,
    target: Pos,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[tuple[Pos, int], ...]:
    routes = []
    for stand in _adjacent_stands(turn, worker, target):
        path = next_step(
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
        elif path.status in ("deadline", "expansion_limit"):
            return ()
    return tuple(routes)


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
    path = next_step(
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
    if item == "WallFixer":
        return next((
            wall for wall in turn.walls()
            if wall.health < _max_building_health(wall)
        ), None)
    prefixes = (
        ("WeaponUpgradeVoucher", turn.weapons()),
        ("StationUpgradeVoucher", (turn.station(),) if turn.station() else ()),
        ("WallUpgradeVoucher", turn.walls()),
    )
    for prefix, targets in prefixes:
        if item not in (f"{prefix}1", f"{prefix}2"):
            continue
        required_level = int(item[-1])
        return next((
            target for target in targets
            if target is not None and target.level == required_level
        ), None)
    return None


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
        path = next_step(
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
        if path.status in ("deadline", "expansion_limit"):
            return None
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
