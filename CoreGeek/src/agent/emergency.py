import time
from typing import Callable, Iterable

from .actions import ActionProposal, PlannedAction
from .defense import _needs_emergency_medicine
from .protocol import (
    ROUNDS_PER_DAY,
    WEAPON_BUILD_COST,
    Pos,
    Robot,
    Turn,
    buy_command,
    distance,
    use_command,
)
from .state import SessionState


EMERGENCY_ITEMS = ("Bomb", "DizzyWeapon")
EMERGENCY_THREAT_DISTANCE = 3
MAX_EMERGENCY_TARGET_CANDIDATES = 128


def propose_held_emergency(
    turn: Turn,
    state: SessionState,
    *,
    defense_actions: Iterable[PlannedAction],
    unavailable_role_ids: frozenset[int] = frozenset(),
    reserved_weapon_ids: frozenset[int] = frozenset(),
    clock: Callable[[], float] = time.monotonic,
    deadline: float,
) -> PlannedAction | None:
    if turn.is_day or clock() >= deadline or reserved_weapon_ids:
        return None
    defense_actions = tuple(defense_actions)
    if any(
        action.proposal.command.get("action") == "attack"
        for action in defense_actions
    ):
        return None
    targets = _urgent_targets(turn, clock, deadline)
    if targets is None:
        return None
    station_cells, urgent, centers = targets
    holders = [
        role for role in turn.controllable()
        if role.unit_id not in unavailable_role_ids
        and not _needs_emergency_medicine(role)
    ]
    bomb_holders = tuple(
        role.unit_id for role in holders if "Bomb" in role.backpack
    )
    dizzy_holders = tuple(
        role.unit_id for role in holders if "DizzyWeapon" in role.backpack
    )
    failed_uses = frozenset(
        (
            completed.pending.actor_id,
            completed.pending.name,
            completed.pending.target,
        )
        for completed in state.action_history
        if completed.success is False
        and completed.pending.action == "use"
        and completed.pending.name in EMERGENCY_ITEMS
        and completed.pending.target is not None
    )
    if bomb_holders:
        selected = _best_bomb(
            centers, urgent, station_cells, bomb_holders, failed_uses,
            clock, deadline,
        )
        if selected is not None:
            role_id, target, affected, killed = selected
            return _proposal(
                role_id, "Bomb", target, affected, killed,
            )
    if dizzy_holders:
        selected = _best_dizzy(
            centers, urgent, station_cells, dizzy_holders, failed_uses,
            clock, deadline,
        )
        if selected is not None:
            role_id, target, affected = selected
            return _proposal(
                role_id,
                "DizzyWeapon",
                target,
                affected,
                0,
            )
    return None


def propose_emergency_purchase(
    turn: Turn,
    state: SessionState,
    *,
    defense_actions: Iterable[PlannedAction],
    unavailable_role_ids: frozenset[int],
    reserved_weapon_ids: frozenset[int],
    clock: Callable[[], float],
    deadline: float,
) -> PlannedAction | None:
    if (
        turn.is_day or turn.round_in_day >= ROUNDS_PER_DAY
        or clock() >= deadline or reserved_weapon_ids
        or state.emergency_purchase_night == (turn.round_no - 1) // ROUNDS_PER_DAY
        or turn.gold < WEAPON_BUILD_COST * max(0, 3 - len(turn.weapons()))
        or any(
            item in role.backpack
            for role in turn.controllable() for item in EMERGENCY_ITEMS
        )
    ):
        return None
    defense_actions = tuple(defense_actions)
    if any(action.proposal.command.get("action") == "attack" for action in defense_actions):
        return None
    unavailable = unavailable_role_ids | frozenset(
        action.proposal.actor_id for action in defense_actions
    )
    shops = turn.zones_of("weaponShop")
    buyers = tuple(
        role.unit_id for role in turn.controllable()
        if role.unit_id not in unavailable
        and role.unit_id not in state.plans
        and not _needs_emergency_medicine(role)
        and role.capacity is not None
        and not role.backpack_full
        and any(distance(role.pos, shop) == 1 for shop in shops)
    )
    if not buyers:
        return None
    targets = _urgent_targets(turn, clock, deadline)
    if targets is None:
        return None
    station_cells, urgent, centers = targets
    reserve = WEAPON_BUILD_COST * max(0, 3 - len(turn.weapons()))
    for item in EMERGENCY_ITEMS:
        price = turn.weapon_prices.get(item)
        if price is None or price <= 0 or turn.gold - price < reserve:
            continue
        if item == "Bomb":
            selected = _best_bomb(
                centers, urgent, station_cells, buyers, frozenset(), clock, deadline,
            )
            affected, killed = (selected[2], selected[3]) if selected else (0, 0)
        else:
            selected = _best_dizzy(
                centers, urgent, station_cells, buyers, frozenset(), clock, deadline,
            )
            affected, killed = (selected[2], 0) if selected else (0, 0)
        if selected is not None:
            role_id = selected[0]
            return PlannedAction(
                ActionProposal(role_id, role_id, buy_command(item), gold_cost=price),
                plan_reason=f"emergency:buy:{item}",
                diagnostic={
                    "kind": "emergencyPurchase",
                    "item": item,
                    "price": price,
                    "affectedUrgent": affected,
                    "killedUrgent": killed,
                },
            )
    return None


def _urgent_targets(
    turn: Turn,
    clock: Callable[[], float],
    deadline: float,
) -> tuple[tuple[Pos, ...], list[Robot], list[Pos]] | None:
    station = turn.station()
    if station is None:
        return None
    station_cells = turn.footprint(station)
    urgent: list[Robot] = []
    for index, robot in enumerate(turn.robots):
        if index % 16 == 0 and clock() >= deadline:
            return None
        if (
            robot.health > 0
            and 0 <= robot.pos.x < turn.width
            and 0 <= robot.pos.y < turn.height
            and robot.target_team == turn.team_type
            and robot.abnormal_state != "dizzy"
            and min(distance(robot.pos, cell) for cell in station_cells)
            <= EMERGENCY_THREAT_DISTANCE
        ):
            urgent.append(robot)
    if not urgent or clock() >= deadline:
        return None
    centers = sorted(
        {robot.pos for robot in urgent},
        key=lambda pos: (
            min(distance(pos, cell) for cell in station_cells),
            pos.x,
            pos.y,
        ),
    )[:MAX_EMERGENCY_TARGET_CANDIDATES]
    return station_cells, urgent, centers


def _best_bomb(
    centers: list[Pos],
    urgent: list[Robot],
    station_cells: tuple[Pos, ...],
    role_ids: tuple[int, ...],
    failed_uses: frozenset[tuple[int, str | None, Pos]],
    clock: Callable[[], float],
    deadline: float,
) -> tuple[int, Pos, int, int] | None:
    choices = []
    for center in centers:
        if clock() >= deadline:
            break
        affected = [robot for robot in urgent if distance(center, robot.pos) <= 1]
        killed = [robot for robot in affected if robot.health <= 100]
        if not killed:
            continue
        role_id = next((
            candidate for candidate in role_ids
            if (candidate, "Bomb", center) not in failed_uses
        ), None)
        if role_id is None:
            continue
        choices.append((
            (
                min(distance(center, cell) for cell in station_cells),
                -len(killed),
                -sum(min(100, robot.health) for robot in affected),
                center.x,
                center.y,
                role_id,
            ),
            center,
            len(affected),
            len(killed),
        ))
    if not choices:
        return None
    score, center, affected, killed = min(choices, key=lambda item: item[0])
    return score[-1], center, affected, killed


def _best_dizzy(
    centers: list[Pos],
    urgent: list[Robot],
    station_cells: tuple[Pos, ...],
    role_ids: tuple[int, ...],
    failed_uses: frozenset[tuple[int, str | None, Pos]],
    clock: Callable[[], float],
    deadline: float,
) -> tuple[int, Pos, int] | None:
    choices = []
    for center in centers:
        if clock() >= deadline:
            break
        affected = [robot for robot in urgent if distance(center, robot.pos) <= 1]
        if not affected:
            continue
        role_id = next((
            candidate for candidate in role_ids
            if (candidate, "DizzyWeapon", center) not in failed_uses
        ), None)
        if role_id is None:
            continue
        choices.append((
            (
                min(distance(center, cell) for cell in station_cells),
                -len(affected),
                -sum(robot.attack_power for robot in affected),
                center.x,
                center.y,
                role_id,
            ),
            center,
            len(affected),
        ))
    if not choices:
        return None
    score, center, affected = min(choices, key=lambda item: item[0])
    return score[-1], center, affected


def _proposal(
    role_id: int,
    item: str,
    target: Pos,
    affected: int,
    killed: int,
) -> PlannedAction:
    return PlannedAction(
        ActionProposal(
            role_id,
            role_id,
            use_command(item, target),
        ),
        target,
        f"emergency:{item}",
        diagnostic={
            "kind": "heldEmergency",
            "item": item,
            "affectedUrgent": affected,
            "killedUrgent": killed,
        },
    )
