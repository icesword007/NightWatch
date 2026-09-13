import time
from typing import Callable

from .actions import ActionProposal, PlannedAction
from .grid import next_step
from .protocol import Pos, Robot, Turn, Unit, distance, move_command
from .state import SessionState

DUSK_POSITIONING_ROUNDS = 12
EMERGENCY_MEDICINE_HEALTH = 40
TASK_RECALL_THREAT_DISTANCE = 3


def protected_gunners(turn: Turn) -> frozenset[int]:
    used: set[int] = set()
    for weapon in turn.weapons():
        controller = _adjacent_controller(turn, weapon, used)
        if controller is not None:
            used.add(controller.unit_id)
    return frozenset(used)


def task_pioneer_recall_action(
    turn: Turn,
    pioneer: Unit,
    *,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> PlannedAction | None:
    if turn.is_day or turn.station() is None:
        return None
    station_cells = turn.footprint(turn.station())
    threats = [
        robot for robot in turn.robots
        if robot.health > 0 and robot.target_team == turn.team_type
    ]
    if not threats:
        return None
    committed_gunners = protected_gunners(turn)
    threats.sort(key=lambda robot: (
        min(distance(robot.pos, cell) for cell in station_cells),
        -robot.attack_power,
        robot.robot_id,
    ))
    for threat in threats:
        threat_distance = min(
            distance(threat.pos, cell) for cell in station_cells
        )
        if threat_distance > TASK_RECALL_THREAT_DISTANCE:
            continue
        for weapon in turn.weapons():
            if (
                weapon.cooldown > 0
                or distance(weapon.pos, threat.pos) > weapon.range_of_attack()
                or any(
                    role.unit_id != pioneer.unit_id
                    and distance(role.pos, weapon.pos) == 1
                    for role in turn.controllable()
                )
            ):
                continue
            route = _gunner_route(
                turn, pioneer, weapon, clock, deadline, max_expansions,
            )
            if route is None:
                continue
            route_cost = route[2]
            if route_cost > threat_distance or route_cost + 1 <= threat_distance:
                continue
            other_can_arrive = False
            for role in turn.controllable():
                if (
                    role.unit_id == pioneer.unit_id
                    or role.unit_id in committed_gunners
                ):
                    continue
                other_route = _gunner_route(
                    turn, role, weapon, clock, deadline, max_expansions,
                )
                if other_route is not None and other_route[2] <= threat_distance:
                    other_can_arrive = True
                    break
            if not other_can_arrive:
                stand, step, _ = route
                if step is None:
                    return None
                return PlannedAction(
                    ActionProposal(
                        pioneer.unit_id,
                        pioneer.unit_id,
                        move_command(step),
                        destination=step,
                    ),
                    stand,
                    f"gunner:{weapon.unit_id}",
                    turn.round_no + threat_distance,
                )
    return None


def propose_defense(
    turn: Turn,
    state: SessionState,
    *,
    clock: Callable[[], float] = time.monotonic,
    deadline: float,
    max_expansions: int,
    unavailable_role_ids: frozenset[int] = frozenset(),
) -> tuple[PlannedAction, ...]:
    candidates: list[PlannedAction] = []
    used_roles: set[int] = set()
    staffed_weapons: set[int] = set()
    projected_damage: dict[int, int] = {}

    for weapon in turn.weapons():
        controller = _adjacent_controller(
            turn, weapon, used_roles | set(unavailable_role_ids),
        )
        if controller is None:
            continue
        used_roles.add(controller.unit_id)
        staffed_weapons.add(weapon.unit_id)
        if turn.is_day or weapon.cooldown > 0:
            continue
        target = _select_target(turn, weapon, projected_damage)
        if target is None:
            continue
        target_count = 1 if weapon.kind == "railgun" else max(weapon.level, 1)
        targets = [target.pos for _ in range(target_count)]
        candidates.append(PlannedAction(
            ActionProposal(
                command_owner_id=weapon.unit_id,
                actor_id=controller.unit_id,
                command={
                    "action": "attack",
                    "controllerId": str(controller.unit_id),
                    "targetPos": [pos.dump() for pos in targets],
                },
            ),
            controller.pos,
            f"gunner:{weapon.unit_id}",
        ))
        _apply_projected_damage(
            turn, weapon, target, target_count, projected_damage,
        )

    if clock() >= deadline:
        return tuple(candidates)
    candidates.extend(_position_gunners(
        turn,
        state,
        used_roles,
        staffed_weapons,
        clock,
        deadline,
        max_expansions,
        unavailable_role_ids,
    ))
    return tuple(candidates)


def threat_score(turn: Turn, robot: Robot) -> tuple[int, int, int, int]:
    station = turn.station()
    anchors = [weapon.pos for weapon in turn.weapons()]
    if station is not None:
        anchors.extend(turn.footprint(station))
    proximity = min((distance(robot.pos, anchor) for anchor in anchors), default=9999)
    team_type = getattr(turn, "team_type", "")
    targets_us = int(robot.target_team == team_type)
    return (targets_us, -proximity, robot.attack_power, -robot.health)


def _select_target(
    turn: Turn,
    weapon: Unit,
    projected_damage: dict[int, int],
) -> Robot | None:
    in_range = [
        robot for robot in turn.robots
        if robot.health > 0
        and distance(weapon.pos, robot.pos) <= weapon.range_of_attack()
    ]
    if not in_range:
        return None
    uncovered = [
        robot for robot in in_range
        if projected_damage.get(robot.robot_id, 0) < robot.health
    ]
    choices = uncovered or in_range
    return max(
        choices,
        key=lambda robot: (threat_score(turn, robot), -robot.robot_id),
    )


def _apply_projected_damage(
    turn: Turn,
    weapon: Unit,
    target: Robot,
    target_count: int,
    projected: dict[int, int],
) -> None:
    if weapon.kind == "gatling":
        ray = _robots_on_ray(turn.robots, weapon.pos, target.pos)
        victim = ray[0] if ray else target
        projected[victim.robot_id] = projected.get(victim.robot_id, 0) + 10 * target_count
        return
    if weapon.kind == "railgun":
        energy = weapon.attack_power or 10 * max(weapon.level, 1)
        for victim in _robots_on_ray(turn.robots, weapon.pos, target.pos):
            damage = min(energy, victim.health)
            projected[victim.robot_id] = projected.get(victim.robot_id, 0) + damage
            energy -= damage
            if energy <= 0:
                break
        return
    if weapon.kind == "rocket":
        for robot in turn.robots:
            splash_distance = distance(robot.pos, target.pos)
            if splash_distance > 1:
                continue
            per_missile = 20 if splash_distance == 0 else 10
            projected[robot.robot_id] = (
                projected.get(robot.robot_id, 0) + per_missile * target_count
            )


def _robots_on_ray(
    robots: tuple[Robot, ...],
    origin: Pos,
    target: Pos,
) -> tuple[Robot, ...]:
    dx = target.x - origin.x
    dy = target.y - origin.y
    length_squared = dx * dx + dy * dy
    on_ray = []
    for robot in robots:
        rx = robot.pos.x - origin.x
        ry = robot.pos.y - origin.y
        dot = rx * dx + ry * dy
        if rx * dy == ry * dx and 0 < dot <= length_squared:
            on_ray.append((distance(origin, robot.pos), robot.robot_id, robot))
    return tuple(item[2] for item in sorted(on_ray))


def _adjacent_controller(
    turn: Turn,
    weapon: Unit,
    used_roles: set[int],
) -> Unit | None:
    controllers = [
        role for role in turn.controllable()
        if role.unit_id not in used_roles
        and not _needs_emergency_medicine(role)
        and distance(role.pos, weapon.pos) == 1
    ]
    if not controllers:
        return None
    return min(
        controllers,
        key=lambda role: (role.kind != "worker", role.unit_id),
    )


def _position_gunners(
    turn: Turn,
    state: SessionState,
    used_roles: set[int],
    staffed_weapons: set[int],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    unavailable_role_ids: frozenset[int],
) -> list[PlannedAction]:
    result = []
    available = [
        role for role in turn.controllable()
        if role.unit_id not in used_roles
        and role.unit_id not in unavailable_role_ids
        and not _needs_emergency_medicine(role)
    ]
    available.sort(key=lambda role: (role.kind != "worker", role.unit_id))
    unstaffed = [
        weapon for weapon in turn.weapons()
        if weapon.unit_id not in staffed_weapons
    ]
    unstaffed.sort(key=lambda weapon: weapon.unit_id)

    for weapon in unstaffed:
        if not available or clock() >= deadline:
            break
        planned = _planned_controller(state, available, weapon)
        routes = []
        for role in available:
            route = _gunner_route(
                turn, role, weapon, clock, deadline, max_expansions,
            )
            if route is not None:
                routes.append((role, route))
            if clock() >= deadline:
                break
        if not routes:
            continue
        role, route = min(
            routes,
            key=lambda item: (
                item[1][2] > turn.rounds_until_night,
                item[0] != planned,
                item[1][2],
                item[0].kind != "worker",
                item[0].unit_id,
            ),
        )
        stand, step, route_cost = route
        if (
            turn.is_day
            and turn.rounds_until_night > DUSK_POSITIONING_ROUNDS
            and route_cost < turn.rounds_until_night
            and planned is None
        ):
            continue
        if step is None:
            used_roles.add(role.unit_id)
            available.remove(role)
            continue
        result.append(PlannedAction(
            ActionProposal(
                role.unit_id,
                role.unit_id,
                move_command(step),
                destination=step,
            ),
            stand,
            f"gunner:{weapon.unit_id}",
            turn.round_no + max(turn.rounds_until_night, route_cost, 1),
        ))
        used_roles.add(role.unit_id)
        available.remove(role)
    return result


def _planned_controller(
    state: SessionState,
    roles: list[Unit],
    weapon: Unit,
) -> Unit | None:
    reason = f"gunner:{weapon.unit_id}"
    return next((
        role for role in roles
        if state.plans.get(role.unit_id) is not None
        and state.plans[role.unit_id].reason == reason
    ), None)


def _gunner_route(
    turn: Turn,
    role: Unit,
    weapon: Unit,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[Pos, Pos | None, int] | None:
    blocked = turn.blocked(role)
    choices = [
        Pos(weapon.pos.x + dx, weapon.pos.y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx or dy)
    ]
    legal = [
        pos for pos in choices if turn.land(pos) and pos not in blocked
    ]
    legal.sort(key=lambda pos: (distance(role.pos, pos), pos.x, pos.y))
    for stand in legal:
        path = next_step(
            turn,
            role,
            stand,
            clock=clock,
            deadline=deadline,
            max_expansions=max_expansions,
        )
        if path.status == "already_there":
            return stand, None, 0
        if (
            path.status == "found"
            and path.step is not None
            and path.cost is not None
        ):
            return stand, path.step, path.cost
        if path.status in ("deadline", "expansion_limit"):
            return None
    return None


def _needs_emergency_medicine(role: Unit) -> bool:
    return (
        role.health <= EMERGENCY_MEDICINE_HEALTH
        and "Medicine" in role.backpack
    )
