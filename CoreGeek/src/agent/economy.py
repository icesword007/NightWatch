import time
from collections import Counter
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
        if candidate is None and need_wall:
            candidate = _wall_action(
                turn,
                worker,
                failed_builds | claimed_build_targets | {
                    plan.target for plan in state.plans.values()
                },
                clock, deadline, max_expansions,
            )
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
                and candidate.plan_reason.startswith("build:")
                and candidate.plan_target is not None
            ):
                claimed_build_targets.add(candidate.plan_target)
                claimed_tower_types.add(candidate.plan_reason.split(":", 1)[1])
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
        or available_gold < 25
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

    purchase = _needed_purchase(turn, worker)
    shops = turn.zones_of("weaponShop")
    if purchase is not None and shops:
        price = turn.weapon_prices.get(purchase)
        if price is not None and price <= available_gold:
            target = _nearest(worker.pos, shops)
            return _buy_or_move(
                turn, worker, target, purchase,
                clock, deadline, max_expansions,
            )

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


def _needed_purchase(turn: Turn, worker: Unit) -> str | None:
    if worker.health < _max_role_health(worker) and "Medicine" in turn.weapon_prices:
        return "Medicine"
    for target in (*turn.weapons(), *((turn.station(),) if turn.station() else ())):
        prefix = "Weapon" if target.kind in TOWER_TYPES else "Station"
        if target.level in (1, 2):
            item = f"{prefix}UpgradeVoucher{target.level}"
            if item in turn.weapon_prices:
                return item
    damaged_wall = next((wall for wall in turn.walls() if wall.health < _max_building_health(wall)), None)
    if damaged_wall is not None and "WallFixer" in turn.weapon_prices:
        return "WallFixer"
    return None


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
        and _needed_purchase(turn, worker) == item
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
