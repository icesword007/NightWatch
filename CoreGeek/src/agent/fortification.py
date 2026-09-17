import hashlib
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from .grid import next_step
from .layout import (
    MAX_LAYOUT_WALLS,
    layout_diagnostic,
    plan_defense_layout,
)
from .protocol import (
    ROUNDS_PER_DAY,
    STATION,
    Pos,
    Turn,
    distance,
)

MAX_WALL_TARGETS = MAX_LAYOUT_WALLS
UNKNOWN_CAPACITY_WALL_BATCH = 2
MAX_TEMPORARY_RECHECKS = 2


@dataclass(frozen=True, slots=True)
class DirectionPrior:
    dx: int
    dy: int
    source: str


def direction_prior(turn: Turn) -> DirectionPrior:
    plan = plan_defense_layout(turn)
    return DirectionPrior(
        plan.direction.x, plan.direction.y, plan.direction_source,
    )


def ordered_wall_targets(
    turn: Turn,
    candidates: tuple[Pos, ...],
) -> tuple[Pos, ...]:
    del candidates
    return plan_defense_layout(turn).wall_targets


def safe_wall_targets(
    turn: Turn,
    candidates: tuple[Pos, ...],
) -> tuple[Pos, ...]:
    accepted = []
    projected_zones = dict(turn.zones)
    planned = ordered_wall_targets(turn, candidates)
    requested = set(candidates)
    targets = (
        tuple(target for target in planned if target in requested)
        if requested and requested.issubset(set(planned))
        else planned
    )
    for target in targets:
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
    weapon_positions = list(dict.fromkeys(
        [weapon.pos for weapon in turn.weapons()]
        + list(plan_defense_layout(turn).tower_targets)
    ))[:3]
    for weapon_pos in weapon_positions:
        legal = tuple(
            pos
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx or dy)
            for pos in (Pos(weapon_pos.x + dx, weapon_pos.y + dy),)
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
    reserved_rounds: int = 0,
) -> int | None:
    if not turn.is_day or len(turn.weapons()) < 2 or turn.station() is None:
        state.fortification_phase = "waiting"
        state.fortification_skip_reason = "tower_line_or_day"
        return None
    partial_tower_line = len(turn.weapons()) == 2
    partial_builders = [
        worker for worker in turn.workers()
        if "stone" in worker.backpack
        and not _worker_protected(worker, state, reserved_role_ids)
    ]
    if partial_tower_line and (
        len(turn.workers()) < 2 or not partial_builders
    ):
        state.fortification_phase = "waiting"
        state.fortification_skip_reason = "third_tower_priority"
        return None
    if not state.fortification_initialized:
        state.fortification_planning_day = (
            turn.round_no - 1
        ) // ROUNDS_PER_DAY + 1
        state.fortification_targets = safe_wall_targets(
            turn, candidates,
        )[:MAX_WALL_TARGETS]
        if len(state.fortification_targets) < len(state.layout_wall_targets):
            state.layout_complete = False
            if state.layout_degraded_reason is None:
                state.layout_degraded_reason = "layout_access_or_construction_gap"
        state.fortification_initialized = True
        if not state.fortification_targets:
            state.fortification_phase = "waiting"
            state.fortification_skip_reason = "no_safe_targets"
            return None
    occupied_walls = {wall.pos for wall in turn.walls()}
    state.fortification_completed.update(
        occupied_walls.intersection(state.fortification_targets)
    )
    remaining_targets = tuple(
        target for target in (
            tuple(
                target for target in state.fortification_targets
                if target in state.fortification_recovery_targets
            )
            + tuple(
                target for target in state.fortification_targets
                if target not in state.fortification_recovery_targets
            )
        )
        if target not in state.fortification_completed
        and target not in state.fortification_failed
        and state.fortification_attempt_days.get(target)
        != (turn.round_no - 1) // ROUNDS_PER_DAY + 1
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
        state.fortification_batch_signature = None
        state.fortification_builder_snapshot = None
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
            and (not partial_tower_line or "stone" in worker.backpack)
        ]
        if not eligible:
            state.fortification_phase = "waiting"
            state.fortification_skip_reason = "worker_protected"
            return None
        first_target = remaining_targets[0]
        builder = min(
            eligible,
            key=lambda worker: (
                "stone" not in worker.backpack,
                distance(worker.pos, first_target), worker.unit_id,
            ),
        )
        state.fortification_builder_id = builder.unit_id
    batch_targets = tuple(
        target for target in state.fortification_batch_targets
        if target in remaining_targets
    )
    if (
        batch_targets
        and remaining_targets[0] in state.fortification_recovery_targets
        and remaining_targets[0] not in batch_targets
    ):
        batch_targets = ()
        state.fortification_batch_targets = ()
        state.fortification_batch_signature = None
        state.fortification_builder_snapshot = None
    continuing_batch = bool(batch_targets)
    progress_confirmed = (
        continuing_batch
        and _batch_progress_confirmed(turn, state, builder.unit_id)
    )
    current_batch_signature = _batch_environment_signature(
        turn, state, builder.unit_id,
    )
    cache_environment_valid = (
        continuing_batch
        and progress_confirmed
        and state.fortification_batch_signature == current_batch_signature
    )
    if not batch_targets:
        state.fortification_batch_targets = ()
        batch_targets = _current_safe_prefix(
            turn, state, remaining_targets, candidates,
        )
    elif not (
        cache_environment_valid
        and all(target in candidates for target in batch_targets)
    ):
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
    available_slots = (
        UNKNOWN_CAPACITY_WALL_BATCH
        if builder.capacity is None
        else max(builder.capacity - len(builder.backpack), 0)
    )
    max_batch = min(
        len(batch_targets),
        builder.backpack.count("stone") + available_slots,
    )
    if not continuing_batch and builder.backpack.count("stone"):
        max_batch = min(max_batch, builder.backpack.count("stone"))
    can_continue_without_replan = (
        continuing_batch
        and max_batch >= len(batch_targets)
        and cache_environment_valid
        and (
            builder.backpack.count("stone") >= len(batch_targets)
            or fortification_stone_target(turn, builder, failed_mines)
            is not None
        )
    )
    feasible_targets = (
        batch_targets
        if can_continue_without_replan
        else _largest_feasible_wall_prefix(
            turn,
            builder,
            batch_targets[:max_batch],
            failed_mines,
            clock,
            deadline,
            max_expansions,
            reserved_rounds,
        )
    )
    if not feasible_targets:
        state.fortification_phase = "waiting"
        state.fortification_skip_reason = "return_deadline"
        return None
    state.fortification_batch_targets = feasible_targets
    state.fortification_batch_signature = current_batch_signature
    state.fortification_builder_snapshot = (
        builder.pos, tuple(sorted(builder.backpack)),
    )
    state.fortification_phase = (
        "building"
        if builder.backpack.count("stone") >= len(feasible_targets)
        else "mining"
    )
    state.fortification_skip_reason = None
    return builder.unit_id


def _batch_progress_confirmed(
    turn: Turn,
    state: Any,
    builder_id: int,
) -> bool:
    builder = turn.unit(builder_id)
    snapshot = state.fortification_builder_snapshot
    if builder is None or snapshot is None:
        return False
    completed = next((
        completed
        for completed in reversed(state.action_history)
        if completed.pending.actor_id == builder_id
        and completed.pending.round_no == turn.round_no - 1
        and completed.pending.source_session == state.session_index
    ), None)
    if completed is None or completed.success is not True:
        return False
    pending = completed.pending
    previous_pos, previous_backpack = snapshot
    expected_pos = previous_pos
    expected_backpack = list(previous_backpack)
    if pending.action == "move":
        plan = state.plans.get(builder_id)
        if (
            pending.target is None
            or plan is None
            or plan.source_session != state.session_index
            or not (
                plan.reason == "build:wall"
                and plan.target in state.fortification_targets
                or plan.reason == "mine:stone"
                and turn.zones.get(plan.target) == "stone"
            )
        ):
            return False
        expected_pos = pending.target
    elif pending.action == "collect":
        if pending.target is None or turn.zones.get(pending.target) != "stone":
            return False
        expected_backpack.append("stone")
    elif pending.action == "build":
        if (
            pending.name != "wall"
            or pending.target not in state.fortification_targets
            or "stone" not in expected_backpack
        ):
            return False
        expected_backpack.remove("stone")
    else:
        return False
    return (
        builder.pos == expected_pos
        and tuple(sorted(builder.backpack)) == tuple(sorted(expected_backpack))
    )


def _batch_environment_signature(
    turn: Turn,
    state: Any,
    builder_id: int,
) -> str:
    planned = set(state.fortification_targets)
    parts = [
        f"zone:{pos.x}:{pos.y}:{kind}"
        for pos, kind in sorted(
            turn.zones.items(), key=lambda entry: (entry[0].x, entry[0].y)
        )
    ]
    parts.extend(
        f"unit:{side}:{unit.unit_id}:{unit.kind}:{unit.pos.x}:{unit.pos.y}"
        for side, units in (("our", turn.ours), ("enemy", turn.enemies))
        for unit in sorted(units, key=lambda entry: entry.unit_id)
        if unit.unit_id != builder_id
        and not (unit.kind == "wall" and unit.pos in planned)
    )
    parts.extend(
        f"robot:{robot.robot_id}:{robot.pos.x}:{robot.pos.y}:{robot.health}"
        for robot in sorted(turn.robots, key=lambda entry: entry.robot_id)
    )
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


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
    day = (turn.round_no - 1) // ROUNDS_PER_DAY + 1
    wall_positions = {wall.pos for wall in turn.walls()}
    gunner_stands = [
        plan.target.dump()
        for _, plan in sorted(state.plans.items())
        if plan.reason.startswith("gunner:")
    ][:3]
    result = {
        "direction": {"x": direction.dx, "y": direction.dy},
        "directionSource": direction.source,
        "targets": [
            target.dump() for target in state.fortification_targets[:MAX_WALL_TARGETS]
        ],
        "batchTargets": [
            target.dump()
            for target in state.fortification_batch_targets[:MAX_WALL_TARGETS]
        ],
        "completed": len(state.fortification_completed),
        "observedFixedWalls": sum(
            target in wall_positions
            for target in state.fortification_targets[:MAX_WALL_TARGETS]
        ),
        "missingConfirmedWalls": sum(
            target not in wall_positions
            for target in state.fortification_recovery_targets
        ),
        "recoveryTargets": [
            target.dump()
            for target in state.fortification_targets[:MAX_WALL_TARGETS]
            if target in state.fortification_recovery_targets
        ],
        "attemptsToday": sum(
            attempt_day == day
            for attempt_day in state.fortification_attempt_days.values()
        ),
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
    result["layout"] = layout_diagnostic(turn, state)
    return result


def _can_build_and_return(
    turn: Turn,
    builder: Any,
    wall_targets: tuple[Pos, ...],
    failed_mines: frozenset[Pos],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    reserved_rounds: int = 0,
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
    return rounds + post_route[1] <= max(
        turn.rounds_until_night - reserved_rounds, 0,
    )


def _largest_feasible_wall_prefix(
    turn: Turn,
    builder: Any,
    wall_targets: tuple[Pos, ...],
    failed_mines: frozenset[Pos],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    reserved_rounds: int = 0,
) -> tuple[Pos, ...]:
    if not wall_targets:
        return ()
    projected_turn = turn
    projected_builder = builder
    rounds = 0
    initial_stone = builder.backpack.count("stone")
    needed_stone = max(
        len(wall_targets) - initial_stone, 0,
    )
    if needed_stone:
        mine = fortification_stone_target(turn, builder, failed_mines)
        if mine is None:
            return _largest_existing_stone_prefix(
                turn, builder, wall_targets, failed_mines, clock, deadline,
                max_expansions, reserved_rounds=reserved_rounds,
            )
        mine_route = _best_adjacent_route(
            projected_turn, projected_builder, (mine,), clock, deadline,
            max_expansions, [8],
        )
        if mine_route is None:
            return ()
        mine_stand, mine_cost = mine_route
        rounds += mine_cost
        projected_builder, projected_turn = _project_role(
            projected_turn,
            projected_builder,
            mine_stand,
            (*projected_builder.backpack, *(("stone",) * needed_stone)),
        )

    projections = []
    for index, wall_target in enumerate(wall_targets, start=1):
        wall_route = (
            (projected_builder.pos, 0)
            if (
                distance(projected_builder.pos, wall_target) == 1
                and projected_turn.land(projected_builder.pos)
                and projected_builder.pos
                not in projected_turn.blocked(projected_builder)
            )
            else _best_adjacent_route(
                projected_turn, projected_builder, (wall_target,), clock,
                deadline, max_expansions, [8],
            )
        )
        if wall_route is None:
            break
        wall_stand, wall_cost = wall_route
        rounds += wall_cost + 1 + (index > initial_stone)
        backpack = list(projected_builder.backpack)
        backpack.remove("stone")
        projected_builder, projected_turn = _project_role(
            projected_turn, projected_builder, wall_stand, tuple(backpack),
        )
        zones = dict(projected_turn.zones)
        zones[wall_target] = "wall"
        projected_turn = replace(projected_turn, zones=zones)
        projections.append((index, rounds, projected_turn, projected_builder))
        if rounds > max(turn.rounds_until_night - reserved_rounds, 0):
            break

    feasible = ()
    for size, spent, trial_turn, trial_builder in reversed(projections):
        post_targets = tuple(
            Pos(weapon.pos.x + dx, weapon.pos.y + dy)
            for weapon in trial_turn.weapons()
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx or dy)
        )
        post_route = (
            (trial_builder.pos, 0)
            if trial_builder.pos in post_targets
            else _best_route(
                trial_turn, trial_builder, post_targets, clock, deadline,
                max_expansions, [16],
            )
        )
        if (
            post_route is not None
            and spent + post_route[1]
            <= max(turn.rounds_until_night - reserved_rounds, 0)
        ):
            feasible = wall_targets[:size]
            break
    if initial_stone and len(feasible) <= initial_stone:
        existing = _largest_existing_stone_prefix(
            turn, builder, wall_targets, failed_mines, clock, deadline,
            max_expansions, minimum_size=len(feasible) + 1,
            reserved_rounds=reserved_rounds,
        )
        if existing:
            return existing
    return feasible


def _largest_existing_stone_prefix(
    turn: Turn,
    builder: Any,
    wall_targets: tuple[Pos, ...],
    failed_mines: frozenset[Pos],
    clock: Callable[[], float],
    deadline: float,
    max_expansions: int,
    *,
    minimum_size: int = 1,
    reserved_rounds: int = 0,
) -> tuple[Pos, ...]:
    limit = min(builder.backpack.count("stone"), len(wall_targets))
    for size in range(limit, minimum_size - 1, -1):
        trial = wall_targets[:size]
        if _can_build_and_return(
            turn, builder, trial, failed_mines, clock, deadline,
            max_expansions, reserved_rounds,
        ):
            return trial
    return ()


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
