import hashlib
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from .grid import next_step
from .protocol import STATION, Pos, Turn, distance

MAX_WALL_TARGETS = 6
MAX_WALL_BATCH = 2
MAX_TEMPORARY_RECHECKS = 2


@dataclass(frozen=True, slots=True)
class DirectionPrior:
    dx: int
    dy: int
    source: str


def direction_prior(turn: Turn) -> DirectionPrior:
    station = turn.station()
    if station is None:
        return DirectionPrior(1, 0, "current_map_horizontal_no_station")
    enemy_station = next(
        (
            unit for unit in turn.enemies
            if unit.kind == STATION and unit.health > 0
        ),
        None,
    )
    if enemy_station is not None and enemy_station.pos.x != station.pos.x:
        return DirectionPrior(
            1 if enemy_station.pos.x > station.pos.x else -1,
            0,
            "current_map_horizontal_enemy_x",
        )
    station_center_x2 = station.pos.x * 2 + 1
    map_center_x2 = turn.width - 1
    if station_center_x2 != map_center_x2:
        return DirectionPrior(
            1 if station_center_x2 < map_center_x2 else -1,
            0,
            "current_map_horizontal_map_center_x",
        )
    return DirectionPrior(1, 0, "current_map_horizontal_equal_x_fallback")


def ordered_wall_targets(
    turn: Turn,
    candidates: tuple[Pos, ...],
) -> tuple[Pos, ...]:
    station = turn.station()
    if station is None:
        return ()
    direction = direction_prior(turn)
    center_x2 = station.pos.x * 2 + 1
    center_y2 = station.pos.y * 2 - 1

    def coordinates(pos: Pos) -> tuple[int, int]:
        relative_x2 = pos.x * 2 - center_x2
        relative_y2 = pos.y * 2 - center_y2
        forward = relative_x2 * direction.dx + relative_y2 * direction.dy
        cross = relative_y2 if direction.dx else relative_x2
        return forward, cross

    scored = [(pos, *coordinates(pos)) for pos in candidates]
    if not scored:
        return ()
    front_edge = max(forward for _, forward, _ in scored)
    front = sorted(
        (entry for entry in scored if entry[1] == front_edge),
        key=lambda entry: (abs(entry[2]), entry[2], entry[0].x, entry[0].y),
    )[:4]
    chosen = [entry[0] for entry in front]
    for side in (-1, 1):
        flank = [
            entry for entry in scored
            if entry[0] not in chosen
            and entry[1] < front_edge
            and entry[1] >= 0
            and (entry[2] < 0 if side < 0 else entry[2] > 0)
        ]
        if flank:
            chosen.append(max(
                flank,
                key=lambda entry: (
                    abs(entry[2]), entry[1], -entry[0].x, -entry[0].y,
                ),
            )[0])
    return tuple(chosen[:MAX_WALL_TARGETS])


def safe_wall_targets(
    turn: Turn,
    candidates: tuple[Pos, ...],
) -> tuple[Pos, ...]:
    accepted = []
    projected_zones = dict(turn.zones)
    for target in ordered_wall_targets(turn, candidates):
        trial_zones = dict(projected_zones)
        trial_zones[target] = "wall"
        if not _has_distinct_weapon_stands(turn, trial_zones):
            continue
        accepted.append(target)
        projected_zones = trial_zones
    return tuple(accepted)


def _has_distinct_weapon_stands(
    turn: Turn,
    zones: dict[Pos, str],
) -> bool:
    controllable_ids = {role.unit_id for role in turn.controllable()}
    static_occupied = {
        cell
        for unit in (*turn.ours, *turn.enemies)
        if (
            unit.health > 0
            and (
                unit in turn.enemies
                or unit.unit_id not in controllable_ids
            )
        )
        for cell in turn.footprint(unit)
    }
    choices = []
    for weapon in turn.weapons():
        legal = tuple(
            pos
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx or dy)
            for pos in (Pos(weapon.pos.x + dx, weapon.pos.y + dy),)
            if (
                0 <= pos.x < turn.width
                and 0 <= pos.y < turn.height
                and zones.get(pos, "land") == "land"
                and pos not in static_occupied
            )
        )
        if not legal:
            return False
        choices.append(legal)

    roles = turn.controllable()
    reachable = {
        role.unit_id: _reachable_cells(turn, role.pos, zones, static_occupied)
        for role in roles
    }
    station = turn.station()
    footprint = (
        frozenset(turn.footprint(station))
        if station is not None
        else frozenset()
    )
    direction = direction_prior(turn)
    if footprint:
        center_x2 = station.pos.x * 2 + 1
        center_y2 = station.pos.y * 2 - 1
        back_passages = {
            pos
            for x in range(turn.width)
            for y in range(turn.height)
            for pos in (Pos(x, y),)
            if (
                min(distance(pos, cell) for cell in footprint) == 2
                and zones.get(pos, "land") == "land"
                and pos not in static_occupied
                and (
                    (pos.x * 2 - center_x2) * direction.dx
                    + (pos.y * 2 - center_y2) * direction.dy
                ) < 0
            )
        }
        if not back_passages or any(
            not cells.intersection(back_passages)
            for cells in reachable.values()
        ):
            return False

    target_count = min(len(roles), len(choices))

    def assign(
        index: int,
        used_roles: set[int],
        used_stands: set[Pos],
        count: int,
    ) -> bool:
        if count >= target_count:
            return True
        if index == len(choices) or count + len(choices) - index < target_count:
            return False
        if assign(index + 1, used_roles, used_stands, count):
            return True
        for role in roles:
            if role.unit_id in used_roles:
                continue
            for stand in choices[index]:
                if (
                    stand in used_stands
                    or stand not in reachable[role.unit_id]
                ):
                    continue
                if assign(
                    index + 1,
                    used_roles | {role.unit_id},
                    used_stands | {stand},
                    count + 1,
                ):
                    return True
        return False

    choices.sort(key=len)
    return assign(0, set(), set(), 0)


def _reachable_cells(
    turn: Turn,
    start: Pos,
    zones: dict[Pos, str],
    static_occupied: set[Pos],
) -> set[Pos]:
    blocked = {
        pos for pos, kind in zones.items() if kind != "land"
    } | static_occupied | {robot.pos for robot in turn.robots}
    blocked.discard(start)
    reached = {start}
    frontier = [start]
    while frontier and len(reached) <= turn.width * turn.height:
        current = frontier.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if not (dx or dy):
                    continue
                candidate = Pos(current.x + dx, current.y + dy)
                if (
                    candidate in reached
                    or candidate in blocked
                    or not 0 <= candidate.x < turn.width
                    or not 0 <= candidate.y < turn.height
                ):
                    continue
                reached.add(candidate)
                frontier.append(candidate)
    return reached


def prepare_fortification(
    turn: Turn,
    state: Any,
    candidates: tuple[Pos, ...],
    *,
    reserved_role_ids: frozenset[int] = frozenset(),
    clock: Callable[[], float] = time.monotonic,
    deadline: float = float("inf"),
    max_expansions: int = 256,
) -> int | None:
    if not turn.is_day or len(turn.weapons()) < 3 or turn.station() is None:
        state.fortification_phase = "waiting"
        state.fortification_skip_reason = "tower_line_or_day"
        return None
    if not state.fortification_initialized:
        remaining = max(MAX_WALL_TARGETS - len(turn.walls()), 0)
        state.fortification_targets = safe_wall_targets(
            turn, candidates,
        )[:remaining]
        state.fortification_initialized = True
        if not state.fortification_targets:
            state.fortification_phase = "complete" if remaining == 0 else "waiting"
            state.fortification_skip_reason = (
                None if remaining == 0 else "no_safe_targets"
            )
            return None
    occupied_walls = {wall.pos for wall in turn.walls()}
    state.fortification_completed.update(
        occupied_walls.intersection(state.fortification_targets)
    )
    remaining_targets = tuple(
        target for target in state.fortification_targets
        if target not in state.fortification_completed
        and target not in state.fortification_failed
    )
    if not remaining_targets:
        state.fortification_phase = "complete"
        state.fortification_skip_reason = None
        return None
    builder = turn.unit(state.fortification_builder_id) if (
        state.fortification_builder_id is not None
    ) else None
    if builder is None and state.fortification_builder_id is not None:
        state.fortification_builder_id = None
        state.fortification_batch_targets = ()
        if turn.round_in_day != 1:
            state.fortification_phase = "waiting"
            state.fortification_skip_reason = "builder_unavailable"
            return None
    if builder is not None and _worker_protected(
        builder, state, reserved_role_ids,
    ):
        state.fortification_phase = "waiting"
        state.fortification_skip_reason = "worker_protected"
        return None
    if builder is None:
        eligible = [
            worker for worker in turn.workers()
            if not _worker_protected(worker, state, reserved_role_ids)
        ]
        if not eligible:
            state.fortification_phase = "waiting"
            state.fortification_skip_reason = "worker_protected"
            return None
        first_target = remaining_targets[0]
        builder = min(
            eligible,
            key=lambda worker: (
                distance(worker.pos, first_target), worker.unit_id,
            ),
        )
        state.fortification_builder_id = builder.unit_id
    batch_targets = tuple(
        target for target in state.fortification_batch_targets
        if target in remaining_targets
    )
    if not batch_targets:
        state.fortification_batch_targets = ()
        batch_targets = _current_safe_prefix(
            turn, state, remaining_targets, candidates,
        )
    else:
        batch_targets = _current_safe_prefix(
            turn, state, batch_targets, candidates,
        )
    if not batch_targets:
        temporarily_blocked = any(
            target in state.fortification_deferred
            for target in remaining_targets
            if target not in state.fortification_failed
        )
        state.fortification_phase = (
            "waiting" if temporarily_blocked else "complete"
        )
        state.fortification_skip_reason = (
            "target_temporarily_blocked"
            if temporarily_blocked
            else "target_unsafe"
        )
        return None
    failed_mines = frozenset(
        completed.pending.target
        for completed in state.action_history
        if completed.pending.action == "collect"
        and completed.success is False
        and completed.pending.target is not None
        and completed.pending.source_session == state.session_index
    )
    if (
        "stone" not in builder.backpack
        and fortification_stone_target(turn, builder, failed_mines) is None
    ):
        state.fortification_phase = "waiting"
        state.fortification_skip_reason = "stone_unavailable"
        return None
    feasible_targets = ()
    available_slots = (
        MAX_WALL_BATCH
        if builder.capacity is None
        else max(builder.capacity - len(builder.backpack), 0)
    )
    max_batch = min(
        len(batch_targets),
        MAX_WALL_BATCH,
        builder.backpack.count("stone") + available_slots,
    )
    for size in range(max_batch, 0, -1):
        trial = batch_targets[:size]
        if _can_build_and_return(
            turn,
            builder,
            trial,
            failed_mines,
            clock,
            deadline,
            max_expansions,
        ):
            feasible_targets = trial
            break
    if not feasible_targets:
        state.fortification_phase = "waiting"
        state.fortification_skip_reason = "return_deadline"
        return None
    state.fortification_batch_targets = feasible_targets
    state.fortification_phase = (
        "building"
        if builder.backpack.count("stone") >= len(feasible_targets)
        else "mining"
    )
    state.fortification_skip_reason = None
    return builder.unit_id


def _current_safe_prefix(
    turn: Turn,
    state: Any,
    targets: tuple[Pos, ...],
    candidates: tuple[Pos, ...],
) -> tuple[Pos, ...]:
    accepted = []
    for target in targets:
        trial = tuple((*accepted, target))
        validation_signature = _wall_condition_signature(
            turn, trial, target in candidates,
        )
        deferred = state.fortification_deferred.get(target)
        if deferred is not None and deferred[0] == validation_signature:
            continue
        if target in candidates and safe_wall_targets(turn, trial) == trial:
            state.fortification_deferred.pop(target, None)
            accepted.append(target)
            if len(accepted) >= MAX_WALL_BATCH:
                break
            continue
        blocker_signature = _temporary_blocker_signature(
            turn, target, candidates,
        )
        if blocker_signature is not None:
            if deferred is None:
                attempts = 1
            elif deferred[1] == blocker_signature:
                attempts = deferred[2]
            else:
                attempts = deferred[2] + 1
            state.fortification_deferred[target] = (
                validation_signature, blocker_signature, attempts,
            )
            if attempts <= MAX_TEMPORARY_RECHECKS:
                continue
        state.fortification_deferred.pop(target, None)
        state.fortification_failed.add(target)
    return tuple(accepted)


def _wall_condition_signature(
    turn: Turn,
    trial: tuple[Pos, ...],
    target_available: bool,
) -> str:
    parts = [
        f"map:{turn.width}:{turn.height}",
        f"available:{int(target_available)}",
        "trial:" + ",".join(f"{pos.x}:{pos.y}" for pos in trial),
    ]
    parts.extend(
        f"zone:{pos.x}:{pos.y}:{kind}"
        for pos, kind in sorted(
            turn.zones.items(), key=lambda entry: (entry[0].x, entry[0].y)
        )
    )
    parts.extend(
        f"unit:{side}:{unit.unit_id}:{unit.kind}:"
        f"{unit.pos.x}:{unit.pos.y}:{unit.health}"
        for side, units in (("our", turn.ours), ("enemy", turn.enemies))
        for unit in sorted(units, key=lambda entry: entry.unit_id)
    )
    parts.extend(
        f"robot:{robot.robot_id}:{robot.pos.x}:{robot.pos.y}:{robot.health}"
        for robot in sorted(turn.robots, key=lambda entry: entry.robot_id)
    )
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _temporary_blocker_signature(
    turn: Turn,
    target: Pos,
    candidates: tuple[Pos, ...],
) -> str | None:
    if turn.zones.get(target, "land") != "land":
        return None
    movable_units = tuple(
        unit for unit in (*turn.controllable(), *turn.enemies)
        if unit.kind != STATION
    )
    movable = {unit.pos for unit in movable_units} | {
        robot.pos for robot in turn.robots
    }
    if target not in candidates:
        blockers = [
            f"unit:{unit.unit_id}:{unit.kind}"
            for unit in movable_units if unit.pos == target
        ]
        blockers.extend(
            f"robot:{robot.robot_id}"
            for robot in turn.robots if robot.pos == target
        )
        if not blockers:
            return None
        return hashlib.sha256(
            "\0".join(sorted(blockers)).encode("utf-8")
        ).hexdigest()
    if not movable:
        return None
    relaxed = replace(
        turn,
        enemies=tuple(unit for unit in turn.enemies if unit.kind == STATION),
        robots=(),
    )
    if safe_wall_targets(relaxed, (target,)) != (target,):
        return None
    blockers = [
        f"unit:{unit.unit_id}:{unit.kind}:{unit.pos.x}:{unit.pos.y}"
        for unit in movable_units
    ]
    blockers.extend(
        f"robot:{robot.robot_id}:{robot.pos.x}:{robot.pos.y}"
        for robot in turn.robots
    )
    return hashlib.sha256(
        "\0".join(sorted(blockers)).encode("utf-8")
    ).hexdigest()


def _worker_protected(
    worker: Any,
    state: Any,
    reserved_role_ids: frozenset[int],
) -> bool:
    if worker.unit_id in reserved_role_ids:
        return True
    plan = state.plans.get(worker.unit_id)
    if plan is not None and plan.reason.startswith(("fund:", "use:")):
        return True
    if any(
        item.startswith((
            "StationUpgradeVoucher",
            "WeaponUpgradeVoucher",
            "WallUpgradeVoucher",
        ))
        for item in worker.backpack
    ):
        return True
    return "Medicine" in worker.backpack and worker.health < 220


def remaining_wall_targets(state: Any) -> tuple[Pos, ...]:
    return tuple(
        target for target in state.fortification_targets
        if target not in state.fortification_completed
        and target not in state.fortification_failed
    )


def fortification_diagnostic(turn: Turn, state: Any) -> dict[str, Any]:
    direction = direction_prior(turn)
    gunner_stands = [
        plan.target.dump()
        for _, plan in sorted(state.plans.items())
        if plan.reason.startswith("gunner:")
    ][:3]
    return {
        "direction": {"x": direction.dx, "y": direction.dy},
        "directionSource": direction.source,
        "targets": [target.dump() for target in state.fortification_targets[:6]],
        "batchTargets": [
            target.dump() for target in state.fortification_batch_targets[:2]
        ],
        "completed": len(state.fortification_completed),
        "failed": len(state.fortification_failed),
        "builderId": (
            str(state.fortification_builder_id)
            if state.fortification_builder_id is not None
            else None
        ),
        "phase": state.fortification_phase,
        "skipReason": state.fortification_skip_reason,
        "gunnerStands": gunner_stands,
    }


def _can_build_and_return(
    turn: Turn,
    builder: Any,
    wall_targets: tuple[Pos, ...],
    failed_mines: frozenset[Pos],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
) -> bool:
    projected_turn = turn
    projected_builder = builder
    rounds = 0
    stone_count = builder.backpack.count("stone")
    needed_stone = max(len(wall_targets) - stone_count, 0)
    if needed_stone:
        mine = fortification_stone_target(turn, builder, failed_mines)
        if mine is None:
            return False
        mine_route = _best_adjacent_route(
            projected_turn,
            projected_builder,
            (mine,),
            clock,
            deadline,
            max_expansions,
            [8],
        )
        if mine_route is None:
            return False
        mine_stand, mine_cost = mine_route
        rounds += mine_cost + needed_stone
        projected_builder, projected_turn = _project_role(
            projected_turn,
            projected_builder,
            mine_stand,
            (*projected_builder.backpack, *(("stone",) * needed_stone)),
        )
    for wall_target in wall_targets:
        wall_route = _best_adjacent_route(
            projected_turn,
            projected_builder,
            (wall_target,),
            clock,
            deadline,
            max_expansions,
            [8],
        )
        if wall_route is None:
            return False
        wall_stand, wall_cost = wall_route
        rounds += wall_cost + 1
        backpack = list(projected_builder.backpack)
        backpack.remove("stone")
        projected_builder, projected_turn = _project_role(
            projected_turn,
            projected_builder,
            wall_stand,
            tuple(backpack),
        )
        zones = dict(projected_turn.zones)
        zones[wall_target] = "wall"
        projected_turn = replace(projected_turn, zones=zones)
    post_targets = tuple(
        Pos(weapon.pos.x + dx, weapon.pos.y + dy)
        for weapon in projected_turn.weapons()
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx or dy)
    )
    post_route = _best_route(
        projected_turn,
        projected_builder,
        post_targets,
        clock,
        deadline,
        max_expansions,
        [16],
    )
    if post_route is None:
        return False
    return rounds + post_route[1] <= turn.rounds_until_night


def _best_adjacent_route(
    turn: Turn,
    role: Any,
    targets: tuple[Pos, ...],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    searches: list[int],
) -> tuple[Pos, int] | None:
    stands = tuple(
        Pos(target.x + dx, target.y + dy)
        for target in targets
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx or dy)
    )
    return _best_route(
        turn, role, stands, clock, deadline, max_expansions, searches,
    )


def _best_route(
    turn: Turn,
    role: Any,
    targets: tuple[Pos, ...],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    searches: list[int],
) -> tuple[Pos, int] | None:
    blocked = turn.blocked(role)
    legal = sorted(
        {
            target for target in targets
            if turn.land(target) and target not in blocked
        },
        key=lambda pos: (distance(role.pos, pos), pos.x, pos.y),
    )
    best = None
    for target in legal:
        if searches[0] <= 0 or clock() >= deadline:
            break
        searches[0] -= 1
        route = next_step(
            turn,
            role,
            target,
            clock=clock,
            deadline=deadline,
            max_expansions=max_expansions,
        )
        if route.status == "already_there":
            return target, 0
        elif route.status == "found" and route.cost is not None:
            candidate = (target, route.cost)
        elif route.status == "deadline":
            return None
        else:
            continue
        if best is None or candidate[1] < best[1]:
            best = candidate
        if candidate[1] == distance(role.pos, target):
            return candidate
    return best


def fortification_stone_target(
    turn: Turn,
    builder: Any,
    excluded: frozenset[Pos] | set[Pos] = frozenset(),
) -> Pos | None:
    return min(
        (pos for pos in turn.stone_mines() if pos not in excluded),
        key=lambda pos: (distance(builder.pos, pos), pos.x, pos.y),
        default=None,
    )


def _project_role(
    turn: Turn,
    role: Any,
    pos: Pos,
    backpack: tuple[str, ...],
) -> tuple[Any, Turn]:
    projected = replace(role, pos=pos, backpack=backpack)
    ours = tuple(
        projected if unit.unit_id == role.unit_id else unit
        for unit in turn.ours
    )
    return projected, replace(turn, ours=ours)


def gunner_stand_sort_key(
    turn: Turn,
    weapon_pos: Pos,
    pos: Pos,
) -> tuple[int, int, int, int]:
    direction = direction_prior(turn)
    station = turn.station()
    if station is None:
        return (0, 0, pos.x, pos.y)
    center_x2 = station.pos.x * 2 + 1
    center_y2 = station.pos.y * 2 - 1
    forward = (
        (pos.x * 2 - center_x2) * direction.dx
        + (pos.y * 2 - center_y2) * direction.dy
    )
    lateral = abs(pos.y - weapon_pos.y) if direction.dx else abs(
        pos.x - weapon_pos.x
    )
    return (forward, lateral, pos.x, pos.y)
