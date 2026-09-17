import heapq
import time
from dataclasses import replace
from typing import Callable

from .actions import ActionProposal, PlannedAction
from .fortification import gunner_stand_sort_key
from .grid import next_step
from .layout import preferred_gunner_stand
from .protocol import (
    DUSK_POSITIONING_ROUNDS,
    PlayerTask,
    Pos,
    Robot,
    Turn,
    Unit,
    distance,
    move_command,
)
from .state import SessionState

EMERGENCY_MEDICINE_HEALTH = 40
TASK_RECALL_THREAT_DISTANCE = 3
MIN_TASK_INTERACTION_ROUNDS = 5
MAX_ROCKET_TARGET_CANDIDATES = 128


def protected_gunners(turn: Turn) -> frozenset[int]:
    assignments = _adjacent_assignments(
        turn, set(), include_emergency_medicine=True,
    )
    return frozenset(role.unit_id for role in assignments.values())


def task_pioneer_recall_action(
    turn: Turn,
    pioneer: Unit,
    *,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    state: SessionState | None = None,
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
                state=state,
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
                    state=state,
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


def task_pioneer_day_return_action(
    turn: Turn,
    state: SessionState,
    pioneer: Unit,
    *,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    reserved_role_ids: frozenset[int] = frozenset(),
    reserved_weapon_ids: frozenset[int] = frozenset(),
) -> tuple[PlannedAction, int] | None:
    _, action, _ = _task_pioneer_day_return_assessment(
        turn,
        state,
        pioneer,
        clock=clock,
        deadline=deadline,
        max_expansions=max_expansions,
        reserved_role_ids=reserved_role_ids,
        reserved_weapon_ids=reserved_weapon_ids,
    )
    return action


def _task_pioneer_day_return_assessment(
    turn: Turn,
    state: SessionState,
    pioneer: Unit,
    *,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    reserved_role_ids: frozenset[int] = frozenset(),
    reserved_weapon_ids: frozenset[int] = frozenset(),
) -> tuple[str, tuple[PlannedAction, int] | None, int]:
    if not turn.is_day or len(turn.weapons()) < 3:
        return "not_required", None, 0
    adjacent = _adjacent_assignments(
        turn,
        {pioneer.unit_id, *reserved_role_ids},
        excluded_weapon_ids=reserved_weapon_ids,
    )
    committed_roles = {role.unit_id for role in adjacent.values()}
    staffed_weapons = set(adjacent)
    committed_roles.update(reserved_role_ids)
    staffed_weapons.update(reserved_weapon_ids)
    for role_id, plan in state.plans.items():
        if role_id == pioneer.unit_id or not plan.reason.startswith("gunner:"):
            continue
        try:
            weapon_id = int(plan.reason.split(":", 1)[1])
        except ValueError:
            continue
        if (
            weapon_id not in staffed_weapons
            and turn.unit(role_id) is not None
            and turn.unit(weapon_id) is not None
        ):
            committed_roles.add(role_id)
            staffed_weapons.add(weapon_id)
    projected_roles, projected_weapons = _project_daytime_gunners(
        turn,
        state,
        pioneer,
        committed_roles,
        staffed_weapons,
        clock,
        deadline,
        max_expansions,
    )
    committed_roles.update(projected_roles)
    staffed_weapons.update(projected_weapons)
    if len(staffed_weapons) < 2:
        return "not_required", None, 0

    planned = state.plans.get(pioneer.unit_id)
    planned_weapon_id = None
    if planned is not None and planned.reason.startswith("gunner:"):
        try:
            planned_weapon_id = int(planned.reason.split(":", 1)[1])
        except ValueError:
            planned_weapon_id = None
    options = []
    unavailable = False
    for weapon in turn.weapons():
        if weapon.unit_id in staffed_weapons:
            continue
        route = _gunner_route(
            turn, pioneer, weapon, clock, deadline, max_expansions,
            state=state,
        )
        other_can_arrive = False
        for role in turn.controllable():
            if role.unit_id == pioneer.unit_id or role.unit_id in committed_roles:
                continue
            other_route = _gunner_route(
                turn, role, weapon, clock, deadline, max_expansions,
                state=state,
            )
            if (
                other_route is not None
                and other_route[2] <= turn.rounds_until_night
            ):
                other_can_arrive = True
                break
        if other_can_arrive:
            continue
        if route is None:
            unavailable = True
            continue
        options.append((weapon, route))
    if not options:
        return (
            "unavailable" if unavailable else "not_required",
            None,
            0,
        )
    weapon, route = min(
        options,
        key=lambda item: (
            item[0].unit_id != planned_weapon_id,
            item[1][2],
            item[0].unit_id,
        ),
    )
    stand, step, route_cost = route
    if step is None:
        return "required", None, route_cost
    action = PlannedAction(
        ActionProposal(
            pioneer.unit_id,
            pioneer.unit_id,
            move_command(step),
            destination=step,
        ),
        stand,
        f"gunner:{weapon.unit_id}",
        turn.round_no + max(turn.rounds_until_night, route_cost, 1),
        route_cost,
    ), turn.rounds_until_night - route_cost
    return "required", action, route_cost


def task_start_skip_reason(
    turn: Turn,
    state: SessionState,
    pioneer: Unit,
    task: PlayerTask,
    stand: Pos,
    arrival_rounds: int,
    *,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    reserved_role_ids: frozenset[int] = frozenset(),
    reserved_weapon_ids: frozenset[int] = frozenset(),
) -> str | None:
    timeout_rounds = getattr(task, "timeout_rounds", None)
    if (
        timeout_rounds is not None
        and timeout_rounds < MIN_TASK_INTERACTION_ROUNDS
    ):
        return "insufficient_solver_window"
    if not turn.is_day:
        return None
    projected_pioneer = replace(pioneer, pos=stand)
    projected_turn = replace(
        turn,
        ours=tuple(
            projected_pioneer if role.unit_id == pioneer.unit_id else role
            for role in turn.ours
        ),
    )
    return_status, _, return_rounds = _task_pioneer_day_return_assessment(
        projected_turn,
        state,
        projected_pioneer,
        clock=clock,
        deadline=deadline,
        max_expansions=max_expansions,
        reserved_role_ids=reserved_role_ids,
        reserved_weapon_ids=reserved_weapon_ids,
    )
    if return_status == "unavailable":
        return "return_route_unavailable"
    if return_status == "not_required":
        return None
    available_rounds = max(turn.rounds_until_night - 1, 0)
    required_rounds = (
        arrival_rounds + MIN_TASK_INTERACTION_ROUNDS + return_rounds
    )
    if required_rounds > available_rounds:
        return "insufficient_solver_return_window"
    return None


def _project_daytime_gunners(
    turn: Turn,
    state: SessionState,
    pioneer: Unit,
    committed_roles: set[int],
    staffed_weapons: set[int],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> tuple[set[int], set[int]]:
    roles = [
        role for role in turn.controllable()
        if role.unit_id != pioneer.unit_id
        and role.unit_id not in committed_roles
    ]
    weapons = [
        weapon for weapon in turn.weapons()
        if weapon.unit_id not in staffed_weapons
    ]
    assignments = _route_assignments(
        turn,
        state,
        roles,
        weapons,
        clock,
        deadline,
        max_expansions,
        respect_positioning_window=False,
    )
    on_time = {
        weapon_id: role
        for weapon_id, (role, route) in assignments.items()
        if route[2] <= turn.rounds_until_night
    }
    return (
        {role.unit_id for role in on_time.values()},
        set(on_time),
    )


def propose_defense(
    turn: Turn,
    state: SessionState,
    *,
    clock: Callable[[], float] = time.monotonic,
    deadline: float,
    max_expansions: int,
    unavailable_role_ids: frozenset[int] = frozenset(),
    reserved_weapon_ids: frozenset[int] = frozenset(),
) -> tuple[PlannedAction, ...]:
    candidates: list[PlannedAction] = []
    used_roles: set[int] = set()
    staffed_weapons: set[int] = set(reserved_weapon_ids)
    projected_damage: dict[int, int] = {}

    adjacent = _adjacent_assignments(
        turn,
        used_roles | set(unavailable_role_ids),
        state,
        reserved_weapon_ids,
    )
    for weapon in turn.weapons():
        controller = adjacent.get(weapon.unit_id)
        if controller is None:
            continue
        used_roles.add(controller.unit_id)
        staffed_weapons.add(weapon.unit_id)
        if turn.is_day or weapon.cooldown > 0:
            continue
        target_count = 1 if weapon.kind == "railgun" else max(weapon.level, 1)
        if weapon.kind == "rocket":
            rocket_targets = _select_rocket_targets(
                turn, weapon, target_count, projected_damage,
                clock=clock, deadline=deadline, state=state,
            )
            if not rocket_targets:
                continue
            targets = [target.pos for target in rocket_targets]
        else:
            target = _select_target(turn, weapon, projected_damage)
            if target is None:
                continue
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
        if weapon.kind != "rocket":
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


def _select_rocket_targets(
    turn: Turn,
    weapon: Unit,
    count: int,
    projected_damage: dict[int, int],
    *,
    clock: Callable[[], float] = time.monotonic,
    deadline: float = float("inf"),
    state: SessionState | None = None,
) -> tuple[Robot, ...]:
    if count <= 0 or clock() >= deadline:
        return ()
    in_range = []
    robots_by_pos: dict[Pos, list[Robot]] = {}
    anchors = _rocket_threat_anchors(turn)
    for index, robot in enumerate(turn.robots):
        if index % 16 == 0 and clock() >= deadline:
            return ()
        if robot.health <= 0:
            continue
        robots_by_pos.setdefault(robot.pos, []).append(robot)
        if distance(weapon.pos, robot.pos) <= weapon.range_of_attack():
            in_range.append(robot)
    if not in_range:
        return ()
    shortlisted = heapq.nsmallest(
        MAX_ROCKET_TARGET_CANDIDATES,
        in_range,
        key=lambda robot: (
            -_rocket_threat_layer(turn, robot, anchors, state)[0],
            -_rocket_threat_layer(turn, robot, anchors, state)[1],
            *(-value for value in threat_score(turn, robot)),
            robot.robot_id,
        ),
    )
    selected: list[Robot] = []
    local_projected = dict(projected_damage)
    for _ in range(count):
        if clock() >= deadline:
            if not selected:
                return ()
            filler = selected[-1]
            while len(selected) < count:
                selected.append(filler)
                _apply_rocket_projected_damage(
                    robots_by_pos, filler.pos, local_projected,
                )
            break
        uncovered = [
            robot for robot in shortlisted
            if local_projected.get(robot.robot_id, 0) < robot.health
        ]
        if not uncovered:
            if selected:
                filler = selected[-1]
                while len(selected) < count:
                    selected.append(filler)
                    _apply_rocket_projected_damage(
                        robots_by_pos, filler.pos, local_projected,
                    )
            break
        best_layer = max(
            _rocket_threat_layer(turn, robot, anchors, state)
            for robot in uncovered
        )
        choices = [
            robot for robot in uncovered
            if _rocket_threat_layer(turn, robot, anchors, state) == best_layer
        ]
        target = max(
            choices,
            key=lambda robot: (
                _rocket_effective_damage(
                    turn, robot.pos, local_projected, robots_by_pos,
                ),
                threat_score(turn, robot),
                -robot.robot_id,
            ),
        )
        selected.append(target)
        _apply_rocket_projected_damage(
            robots_by_pos, target.pos, local_projected,
        )
    if len(selected) != count:
        return ()
    projected_damage.update(local_projected)
    return tuple(selected)


def _rocket_threat_anchors(turn: Turn) -> tuple[Pos, ...]:
    station = turn.station()
    anchors = [weapon.pos for weapon in turn.weapons()]
    if station is not None:
        anchors.extend(turn.footprint(station))
    anchors.extend(
        role.pos
        for role in turn.controllable()
        if any(distance(role.pos, weapon.pos) == 1 for weapon in turn.weapons())
    )
    return tuple(dict.fromkeys(anchors))


def _rocket_threat_layer(
    turn: Turn,
    robot: Robot,
    anchors: tuple[Pos, ...] | None = None,
    state: SessionState | None = None,
) -> tuple[int, int]:
    station = turn.station()
    anchors = anchors if anchors is not None else _rocket_threat_anchors(turn)
    proximity = min((distance(robot.pos, anchor) for anchor in anchors), default=9999)
    behind = False
    if station is not None:
        enemy_station = next(
            (unit for unit in turn.enemies if unit.kind == "station"), None,
        )
        direction_x = (
            state.layout_direction[0]
            if state is not None and state.layout_initialized
            else 1 if enemy_station is not None and enemy_station.pos.x > station.pos.x
            else -1 if enemy_station is not None else 1
        )
        behind = (robot.pos.x - station.pos.x) * direction_x < 0
    return (
        int(robot.target_team == turn.team_type),
        int(proximity <= TASK_RECALL_THREAT_DISTANCE or behind),
    )


def _rocket_effective_damage(
    turn: Turn,
    target: Pos,
    projected_damage: dict[int, int],
    robots_by_pos: dict[Pos, list[Robot]] | None = None,
) -> int:
    total = 0
    if robots_by_pos is None:
        victims = turn.robots
    else:
        victims = tuple(
            robot
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for robot in robots_by_pos.get(Pos(target.x + dx, target.y + dy), ())
        )
    for robot in victims:
        splash_distance = distance(robot.pos, target)
        if splash_distance <= 1:
            remaining = max(
                robot.health - projected_damage.get(robot.robot_id, 0), 0,
            )
            damage = 20 if splash_distance == 0 else 10
            total += min(damage, remaining)
    return total


def _apply_rocket_projected_damage(
    robots_by_pos: dict[Pos, list[Robot]],
    target: Pos,
    projected: dict[int, int],
) -> None:
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            pos = Pos(target.x + dx, target.y + dy)
            per_missile = 20 if dx == 0 and dy == 0 else 10
            for robot in robots_by_pos.get(pos, ()):
                projected[robot.robot_id] = (
                    projected.get(robot.robot_id, 0) + per_missile
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


def _adjacent_assignments(
    turn: Turn,
    excluded_role_ids: set[int],
    state: SessionState | None = None,
    excluded_weapon_ids: frozenset[int] = frozenset(),
    include_emergency_medicine: bool = False,
) -> dict[int, Unit]:
    roles = tuple(
        role for role in turn.controllable()
        if role.unit_id not in excluded_role_ids
        and (
            include_emergency_medicine
            or not _needs_emergency_medicine(role)
        )
    )
    weapons = tuple(
        weapon for weapon in turn.weapons()
        if weapon.unit_id not in excluded_weapon_ids
    )
    best: dict[int, Unit] = {}
    best_score: tuple[int, int, int] | None = None
    best_pairs: tuple[tuple[int, int], ...] | None = None

    def search(index: int, used_roles: set[int], assigned: dict[int, Unit]) -> None:
        nonlocal best, best_score, best_pairs
        if index == len(weapons):
            planned = 0
            if state is not None:
                planned = sum(
                    state.plans.get(role.unit_id) is not None
                    and state.plans[role.unit_id].reason == f"gunner:{weapon_id}"
                    for weapon_id, role in assigned.items()
                )
            score = (
                len(assigned),
                planned,
                sum(role.kind == "worker" for role in assigned.values()),
            )
            pairs = tuple(sorted(
                (weapon_id, role.unit_id) for weapon_id, role in assigned.items()
            ))
            if (
                best_score is None
                or score > best_score
                or (score == best_score and (best_pairs is None or pairs < best_pairs))
            ):
                best = dict(assigned)
                best_score = score
                best_pairs = pairs
            return
        weapon = weapons[index]
        search(index + 1, used_roles, assigned)
        for role in roles:
            if (
                role.unit_id in used_roles
                or distance(role.pos, weapon.pos) != 1
            ):
                continue
            used_roles.add(role.unit_id)
            assigned[weapon.unit_id] = role
            search(index + 1, used_roles, assigned)
            assigned.pop(weapon.unit_id)
            used_roles.remove(role.unit_id)

    search(0, set(), {})
    return best


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
    unstaffed = [
        weapon for weapon in turn.weapons()
        if weapon.unit_id not in staffed_weapons
    ]
    assignments = _route_assignments(
        turn, state, available, unstaffed, clock, deadline, max_expansions,
    )
    for weapon in unstaffed:
        assigned = assignments.get(weapon.unit_id)
        if assigned is None:
            continue
        role, route = assigned
        stand, step, route_cost = route
        if step is None:
            used_roles.add(role.unit_id)
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
    return result


def _route_assignments(
    turn: Turn,
    state: SessionState,
    roles: list[Unit],
    weapons: list[Unit],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    respect_positioning_window: bool = True,
) -> dict[int, tuple[Unit, tuple[Pos, Pos | None, int]]]:
    routes: dict[tuple[int, int], tuple[Pos, Pos | None, int]] = {}
    positioning_window = DUSK_POSITIONING_ROUNDS + max(len(weapons) - 1, 0)
    for role in roles:
        for weapon in weapons:
            route = _gunner_route(
                turn, role, weapon, clock, deadline, max_expansions,
                state=state,
            )
            if route is None:
                continue
            planned = state.plans.get(role.unit_id)
            if (
                respect_positioning_window
                and turn.is_day
                and turn.rounds_until_night > positioning_window
                and route[2] < turn.rounds_until_night
                and (
                    planned is None
                    or planned.reason != f"gunner:{weapon.unit_id}"
                )
            ):
                continue
            routes[(role.unit_id, weapon.unit_id)] = route
            if clock() >= deadline:
                break
        if clock() >= deadline:
            break

    best: dict[int, tuple[Unit, tuple[Pos, Pos | None, int]]] = {}
    best_score: tuple[int, int, int, int, int] | None = None
    best_pairs: tuple[tuple[int, int], ...] | None = None

    def search(
        index: int,
        used_roles: set[int],
        assigned: dict[int, tuple[Unit, tuple[Pos, Pos | None, int]]],
    ) -> None:
        nonlocal best, best_score, best_pairs
        if index == len(weapons):
            score = (
                sum(route[2] <= turn.rounds_until_night for _, route in assigned.values()),
                len(assigned),
                sum(
                    state.plans.get(role.unit_id) is not None
                    and state.plans[role.unit_id].reason == f"gunner:{weapon_id}"
                    for weapon_id, (role, _) in assigned.items()
                ),
                -sum(route[2] for _, route in assigned.values()),
                sum(role.kind == "worker" for role, _ in assigned.values()),
            )
            pairs = tuple(sorted(
                (weapon_id, role.unit_id) for weapon_id, (role, _) in assigned.items()
            ))
            if (
                best_score is None
                or score > best_score
                or (score == best_score and (best_pairs is None or pairs < best_pairs))
            ):
                best = dict(assigned)
                best_score = score
                best_pairs = pairs
            return
        weapon = weapons[index]
        search(index + 1, used_roles, assigned)
        for role in roles:
            route = routes.get((role.unit_id, weapon.unit_id))
            if route is None or role.unit_id in used_roles:
                continue
            used_roles.add(role.unit_id)
            assigned[weapon.unit_id] = (role, route)
            search(index + 1, used_roles, assigned)
            assigned.pop(weapon.unit_id)
            used_roles.remove(role.unit_id)

    search(0, set(), {})
    return best


def _gunner_route(
    turn: Turn,
    role: Unit,
    weapon: Unit,
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    state: SessionState | None = None,
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
    preferred = preferred_gunner_stand(turn, weapon.pos, state)
    legal.sort(key=lambda pos: (
        pos != preferred, distance(role.pos, pos), pos.x, pos.y,
    ))
    found = []
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
            candidate = (stand, path.step, path.cost)
            if not turn.is_day:
                return candidate
            found.append(candidate)
            if path.cost == 0:
                return candidate
        if path.status == "deadline":
            return None
        if path.status == "expansion_limit":
            continue
    if not found:
        return None
    timely = [route for route in found if route[2] <= turn.rounds_until_night]
    if timely:
        def timely_key(route: tuple[Pos, Pos | None, int]) -> tuple[int, ...]:
            protection = gunner_stand_sort_key(turn, weapon.pos, route[0])
            return (int(route[0] != preferred),) + protection[:1] + (
                route[2],
            ) + protection[1:]

        return min(timely, key=timely_key)
    return min(found, key=lambda route: (
        route[2], gunner_stand_sort_key(turn, weapon.pos, route[0]),
    ))


def _needs_emergency_medicine(role: Unit) -> bool:
    return (
        role.health <= EMERGENCY_MEDICINE_HEALTH
        and "Medicine" in role.backpack
    )
