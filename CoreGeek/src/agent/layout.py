from dataclasses import dataclass
from typing import Any

from .protocol import STATION, TOWER_TYPES, WALL, Pos, Turn


MAX_LAYOUT_WALLS = 14


@dataclass(frozen=True, slots=True)
class DefenseLayout:
    direction: Pos
    direction_source: str
    tower_targets: tuple[Pos, ...]
    gunner_stands: tuple[Pos, ...]
    wall_targets: tuple[Pos, ...]
    exit_cells: tuple[Pos, ...]
    complete: bool
    degraded_reason: str | None = None


def direction_prior(turn: Turn) -> tuple[Pos, str]:
    station = turn.station()
    if station is None:
        return Pos(1, 0), "current_map_horizontal_no_station"
    enemy_station = next(
        (unit for unit in turn.enemies if unit.kind == STATION and unit.health > 0),
        None,
    )
    if enemy_station is not None and enemy_station.pos.x != station.pos.x:
        return (
            Pos(1 if enemy_station.pos.x > station.pos.x else -1, 0),
            "current_map_horizontal_enemy_x",
        )
    center_x2 = station.pos.x * 2 + 1
    map_center_x2 = turn.width - 1
    if center_x2 != map_center_x2:
        return (
            Pos(1 if center_x2 < map_center_x2 else -1, 0),
            "current_map_horizontal_map_center_x",
        )
    return Pos(1, 0), "current_map_horizontal_equal_x_fallback"


def plan_defense_layout(turn: Turn) -> DefenseLayout:
    direction, source = direction_prior(turn)
    station = turn.station()
    if station is None:
        return DefenseLayout(direction, source, (), (), (), (), False, "station_missing")
    footprint = turn.footprint(station)
    min_x = min(pos.x for pos in footprint)
    max_x = max(pos.x for pos in footprint)
    min_y = min(pos.y for pos in footprint)
    max_y = max(pos.y for pos in footprint)
    front_x = max_x + 2 if direction.x > 0 else min_x - 2
    back_tower_x = min_x - 1 if direction.x > 0 else max_x + 1
    back_stand_x = min_x - 2 if direction.x > 0 else max_x + 2
    center_ys = _center_out(min_y - 1, max_y + 1)
    tower_targets = tuple(Pos(back_tower_x, y) for y in center_ys[:3])
    gunner_stands = tuple(Pos(back_stand_x, y) for y in center_ys[:3])
    exit_cells = tuple(Pos(back_stand_x, y) for y in range(min_y - 2, max_y + 3))

    front = tuple(Pos(front_x, y) for y in _center_out(min_y - 2, max_y + 2))
    flank_xs = (
        range(front_x - 1, min_x - 2, -1)
        if direction.x > 0
        else range(front_x + 1, max_x + 2)
    )
    top = tuple(Pos(x, min_y - 2) for x in flank_xs)
    bottom = tuple(Pos(x, max_y + 2) for x in flank_xs)
    walls = tuple((*front, *_interleave(top, bottom)))[:MAX_LAYOUT_WALLS]

    static_occupied = {
        cell
        for role in (*turn.ours, *turn.enemies)
        if role.health > 0 and role.kind not in ("worker", "pioneer")
        for cell in turn.footprint(role)
    }

    def available(pos: Pos, allowed_kinds: tuple[str, ...] = ()) -> bool:
        if not turn.land(pos):
            return False
        occupant = next(
            (
                role for role in turn.ours
                if role.health > 0 and pos in turn.footprint(role)
            ),
            None,
        )
        return pos not in static_occupied or (
            occupant is not None and occupant.kind in allowed_kinds
        )

    valid_pairs = tuple(
        (tower, stand)
        for tower, stand in zip(tower_targets, gunner_stands)
        if available(tower, TOWER_TYPES) and available(stand)
    )
    valid_towers = tuple(tower for tower, _ in valid_pairs)
    valid_stands = tuple(stand for _, stand in valid_pairs)
    valid_exits = tuple(pos for pos in exit_cells if available(pos))
    valid_walls = tuple(pos for pos in walls if available(pos, (WALL,)))
    off_plan_weapons = tuple(
        weapon.pos for weapon in turn.weapons()
        if weapon.pos not in tower_targets
    )
    complete = (
        len(valid_towers) == 3
        and len(valid_stands) == 3
        and len(valid_exits) >= 2
        and len(valid_walls) == MAX_LAYOUT_WALLS
        and len(set(valid_towers) | set(valid_stands) | set(valid_walls))
        == 3 + 3 + MAX_LAYOUT_WALLS
        and not off_plan_weapons
    )
    degraded_reason = None
    if off_plan_weapons:
        degraded_reason = "existing_weapon_outside_layout"
    elif not complete:
        degraded_reason = "insufficient_legal_layout_cells"
    return DefenseLayout(
        direction,
        source,
        valid_towers,
        valid_stands,
        valid_walls,
        valid_exits,
        complete,
        degraded_reason,
    )


def ensure_defense_layout(turn: Turn, state: Any) -> DefenseLayout:
    if not state.layout_initialized:
        plan = plan_defense_layout(turn)
        state.layout_initialized = True
        state.layout_direction = (plan.direction.x, plan.direction.y)
        state.layout_direction_source = plan.direction_source
        state.layout_tower_targets = plan.tower_targets
        state.layout_gunner_stands = plan.gunner_stands
        state.layout_wall_targets = plan.wall_targets
        state.layout_exit_cells = plan.exit_cells
        state.layout_complete = plan.complete
        state.layout_degraded_reason = plan.degraded_reason
    _observe_direction(turn, state)
    return layout_from_state(state)


def layout_from_state(state: Any) -> DefenseLayout:
    return DefenseLayout(
        Pos(*state.layout_direction),
        state.layout_direction_source,
        state.layout_tower_targets,
        state.layout_gunner_stands,
        state.layout_wall_targets,
        state.layout_exit_cells,
        state.layout_complete,
        state.layout_degraded_reason,
    )


def preferred_gunner_stand(
    turn: Turn,
    weapon_pos: Pos,
    state: Any | None = None,
) -> Pos | None:
    if state is not None and state.layout_initialized:
        tower_targets = state.layout_tower_targets
        gunner_stands = state.layout_gunner_stands
    else:
        plan = plan_defense_layout(turn)
        tower_targets = plan.tower_targets
        gunner_stands = plan.gunner_stands
    try:
        index = tower_targets.index(weapon_pos)
    except ValueError:
        return None
    return gunner_stands[index] if index < len(gunner_stands) else None


def layout_diagnostic(turn: Turn, state: Any) -> dict[str, Any]:
    weapons = {weapon.pos: weapon.kind for weapon in turn.weapons()}
    walls = {wall.pos for wall in turn.walls()}
    built_towers = [
        pos for pos in state.layout_tower_targets if pos in weapons
    ]
    built_walls = [
        pos for pos in state.layout_wall_targets if pos in walls
    ]
    return {
        "direction": {"x": state.layout_direction[0], "y": state.layout_direction[1]},
        "directionSource": state.layout_direction_source,
        "observedDeviation": state.layout_observed_deviation,
        "complete": state.layout_complete,
        "degradedReason": state.layout_degraded_reason,
        "towerTargets": [pos.dump() for pos in state.layout_tower_targets[:3]],
        "builtTowerTargets": [pos.dump() for pos in built_towers[:3]],
        "towerGaps": [
            pos.dump() for pos in state.layout_tower_targets[:3]
            if pos not in weapons
        ],
        "rocketCount": sum(
            weapon.kind == "rocket" for weapon in turn.weapons()
        ),
        "gunnerStands": [pos.dump() for pos in state.layout_gunner_stands[:3]],
        "wallTargets": [pos.dump() for pos in state.layout_wall_targets[:MAX_LAYOUT_WALLS]],
        "builtWallTargets": [pos.dump() for pos in built_walls[:MAX_LAYOUT_WALLS]],
        "wallGaps": [
            pos.dump() for pos in state.layout_wall_targets[:MAX_LAYOUT_WALLS]
            if pos not in walls
        ],
        "exitCells": [pos.dump() for pos in state.layout_exit_cells[:6]],
    }


def _observe_direction(turn: Turn, state: Any) -> None:
    station = turn.station()
    if station is None:
        return
    direction_x = state.layout_direction[0]
    signs = {
        1 if (robot.pos.x - station.pos.x) * direction_x > 0 else -1
        for robot in turn.robots
        if robot.health > 0
        and robot.target_team == turn.team_type
        and robot.pos.x != station.pos.x
    }
    if signs == {1}:
        state.layout_observed_deviation = "aligned"
    elif signs == {-1}:
        state.layout_observed_deviation = "opposite"
    elif signs:
        state.layout_observed_deviation = "mixed"


def _center_out(start: int, stop: int) -> tuple[int, ...]:
    center2 = start + stop
    return tuple(sorted(
        range(start, stop + 1),
        key=lambda value: (abs(value * 2 - center2), value),
    ))


def _interleave(first: tuple[Pos, ...], second: tuple[Pos, ...]) -> tuple[Pos, ...]:
    result = []
    for index in range(max(len(first), len(second))):
        if index < len(first):
            result.append(first[index])
        if index < len(second):
            result.append(second[index])
    return tuple(result)
