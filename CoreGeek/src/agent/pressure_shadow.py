"""Read-only, uncalibrated observations of pressure on both bases."""

from dataclasses import dataclass, field
from typing import Any

from .protocol import DAY_ROUNDS, ROUNDS_PER_DAY, Turn, distance, station_footprint


ALGORITHM_VERSION = "heuristic_v1"
ID_LIMIT = 512
SIDES = ("challenger", "defender")


@dataclass(slots=True)
class NightRecord:
    night: int
    prediction: dict[str, Any]
    frames: int = 0
    last_round: int | None = None
    complete: bool = True
    damage: dict[str, bool] = field(default_factory=lambda: {side: False for side in SIDES})
    base_comparable: dict[str, bool] = field(default_factory=lambda: {side: True for side in SIDES})
    unknowns: set[str] = field(default_factory=set)


@dataclass(slots=True)
class ShadowState:
    last_round: int | None = None
    previous_bases: dict[str, tuple[int, int, int] | None] = field(default_factory=dict)
    previous_walls: dict[str, dict[int, tuple[int, int]]] = field(default_factory=dict)
    previous_robots: dict[int, tuple[str, int]] | None = None
    pending_prediction: dict[str, Any] | None = None
    active_night: NightRecord | None = None
    recent_nights: list[dict[str, Any]] = field(default_factory=list)
    last_verification: dict[str, Any] | None = None
    day_observation_incomplete: bool = False
    changed_defense: dict[str, bool] = field(
        default_factory=lambda: {side: False for side in SIDES}
    )
    current: dict[str, Any] | None = None


def _night_number(round_no: int) -> int:
    return (round_no - 1) // ROUNDS_PER_DAY + 1


def _base(roles: tuple) -> Any:
    return next((role for role in roles if role.kind == "station"), None)


def _side_roles(turn: Turn) -> tuple[tuple[str, tuple], tuple[str, tuple]]:
    enemy = "defender" if turn.team_type == "challenger" else "challenger"
    return ((turn.team_type, turn.ours), (enemy, turn.enemies))


def _wall_map(roles: tuple) -> tuple[dict[int, tuple[int, int]], bool]:
    walls = (role for role in roles if role.kind == "wall")
    result = {}
    truncated = False
    for wall in walls:
        if len(result) < ID_LIMIT:
            result[wall.unit_id] = (wall.level, wall.health)
        else:
            truncated = True
    return result, truncated


def _prediction(
    state: ShadowState, night: int, issued_round: int | None, our_team: str,
) -> dict[str, Any]:
    previous = state.recent_nights[-1] if state.recent_nights else None
    current_unknowns = list(state.current["unknowns"]) if state.current else []
    if state.day_observation_incomplete:
        current_unknowns.append("day_observation_incomplete")
    sides = {}
    for side in SIDES:
        baseline = "unknown"
        reasons = []
        if previous is not None and previous["night"] == night - 1 and previous["complete"]:
            actual = previous["sides"][side]["actualBaseDamage"]
            if actual is True:
                baseline = "elevated"
            elif actual is False:
                baseline = "low"
        if baseline == "unknown":
            reasons.append("no_complete_previous_night")
        if state.changed_defense[side]:
            reasons.append("changed_defense")
        if current_unknowns:
            reasons.append("incomplete_current_observation")
        sides[side] = {
            "risk": "unknown" if state.changed_defense[side] or current_unknowns else baseline,
            "baselineRisk": baseline,
            "changedDefense": state.changed_defense[side],
            "criticalWallBreachRisk": "unknown",
            "criticalWallBreachReason": "wall_topology_unverified",
            "reasons": reasons,
        }
    return {
        "status": "issued",
        "night": night,
        "issuedRound": issued_round,
        "algorithmVersion": ALGORITHM_VERSION,
        "basedOnNight": previous["night"] if previous is not None else None,
        "completeEvidence": previous is not None and previous["night"] == night - 1
        and previous["complete"] and not current_unknowns,
        "unknowns": current_unknowns,
        "risk": sides[our_team]["risk"],
        "baselineRisk": sides[our_team]["baselineRisk"],
        "changedDefense": sides[our_team]["changedDefense"],
        "sides": sides,
    }


def _score(risk: str, actual: bool | None) -> str:
    if actual is None or risk == "unknown":
        return "unscorable"
    if risk == "elevated":
        return "hit" if actual else "false_positive"
    return "false_negative" if actual else "correct_negative"


def _finish_night(state: ShadowState, our_team: str, boundary_round: int) -> None:
    record = state.active_night
    if record is None:
        return
    expected_end = record.night * ROUNDS_PER_DAY
    if boundary_round != expected_end + 1:
        record.unknowns.add("missing_day_boundary")
    complete = record.complete and record.frames == ROUNDS_PER_DAY - DAY_ROUNDS
    complete = complete and record.last_round == expected_end and not record.unknowns
    sides = {}
    for side in SIDES:
        actual = True if record.damage[side] else (
            False if complete and record.base_comparable[side] else None
        )
        risk = record.prediction["sides"][side]["risk"]
        sides[side] = {"actualBaseDamage": actual, "result": _score(risk, actual)}
    verification = {
        "night": record.night,
        "issuedRound": record.prediction["issuedRound"],
        "algorithmVersion": ALGORITHM_VERSION,
        "complete": complete,
        "unknowns": sorted(record.unknowns),
        "sides": sides,
        "actualBaseDamage": sides[our_team]["actualBaseDamage"],
        "result": sides[our_team]["result"],
    }
    state.last_verification = verification
    state.recent_nights.append(verification)
    del state.recent_nights[:-3]
    state.active_night = None


def observe_shadow(turn: Turn, state: ShadowState) -> None:
    """Observe one new round without changing Turn or decision state."""
    if state.last_round is not None and turn.round_no <= state.last_round:
        return
    night = _night_number(turn.round_no)
    first_night_round = (night - 1) * ROUNDS_PER_DAY + DAY_ROUNDS + 1
    consecutive = state.last_round == turn.round_no - 1
    if turn.is_day and state.active_night is not None:
        record = state.active_night
        if turn.round_no == record.night * ROUNDS_PER_DAY + 1:
            for side, roles in _side_roles(turn):
                base = _base(roles)
                previous = state.previous_bases.get(side)
                if (base is None or previous is None
                        or (base.unit_id, base.level) != previous[:2]):
                    record.base_comparable[side] = False
                elif base.health < previous[2]:
                    record.damage[side] = True
        _finish_night(state, turn.team_type, turn.round_no)
        state.changed_defense = {side: False for side in SIDES}
    elif state.active_night is not None and state.active_night.night != night:
        _finish_night(state, turn.team_type, turn.round_no)
        state.changed_defense = {side: False for side in SIDES}
    if (turn.is_day and (state.pending_prediction is None
                         or state.pending_prediction["night"] != night)):
        state.day_observation_incomplete = False
    if not turn.is_day and state.active_night is None:
        frozen = (state.pending_prediction
                  if state.pending_prediction is not None
                  and state.pending_prediction["night"] == night else None)
        if frozen is None:
            frozen = _prediction(state, night, None, turn.team_type)
            frozen["status"] = "unavailable"
            frozen["risk"] = "unknown"
            for side in SIDES:
                frozen["sides"][side]["risk"] = "unknown"
                frozen["sides"][side]["reasons"].append("no_daytime_prediction")
        state.active_night = NightRecord(night, frozen)
        if turn.round_no != first_night_round:
            state.active_night.complete = False
            state.active_night.unknowns.add("night_started_late")
        if not consecutive:
            state.active_night.complete = False
            state.active_night.unknowns.add("missing_dusk_boundary")
    bases = {"challenger": None, "defender": None}
    base_units = {}
    walls = {}
    unknowns = []
    if state.last_round is None and turn.round_no != 1:
        unknowns.append("history_starts_late")
    elif state.last_round is not None and not consecutive:
        unknowns.append("round_gap")
    for side, roles in _side_roles(turn):
        base = _base(roles)
        base_units[side] = base
        bases[side] = (base.unit_id, base.level, base.health) if base else None
        walls[side], limited = _wall_map(roles)
        if base is None:
            unknowns.append(f"{side}_base_missing")
        if limited:
            unknowns.append(f"{side}_wall_id_limit")
    if not turn.robot_roles_observed:
        unknowns.append("robots_missing")
    robots = {side: {"count": 0, "types": {}, "totalHp": 0,
                     "observedHpLoss": 0, "damagedIds": 0,
                     "disappeared": 0,
                     "nearestBaseChebyshev": None} for side in SIDES}
    robot_hp = {}
    unknown_target = 0
    for robot in turn.robots:
        if len(robot_hp) < ID_LIMIT:
            robot_hp[robot.robot_id] = (robot.target_team, robot.health)
        if robot.target_team not in SIDES:
            unknown_target += 1
            continue
        group = robots[robot.target_team]
        group["count"] += 1
        by_type = group["types"].setdefault(robot.kind, {"count": 0, "totalHp": 0})
        by_type["count"] += 1
        by_type["totalHp"] += robot.health
        group["totalHp"] += robot.health
        base = base_units[robot.target_team]
        if base is not None:
            nearest = min(distance(robot.pos, cell) for cell in station_footprint(base.pos))
            current = group["nearestBaseChebyshev"]
            group["nearestBaseChebyshev"] = nearest if current is None else min(current, nearest)
        if consecutive and state.previous_robots is not None:
            previous = state.previous_robots.get(robot.robot_id)
            if (previous is not None
                    and previous[0] == robot.target_team
                    and robot.health < previous[1]):
                group["observedHpLoss"] += previous[1] - robot.health
                group["damagedIds"] += 1
    if len(turn.robots) > ID_LIMIT:
        unknowns.append("robot_id_limit")
    if unknown_target:
        unknowns.append("unknown_target_robots")
    if turn.is_day and unknowns:
        state.day_observation_incomplete = True
    if not turn.robot_roles_observed:
        for group in robots.values():
            group.update(count=None, types=None, totalHp=None,
                         observedHpLoss=None, damagedIds=None,
                         disappeared=None)
    if (consecutive and not turn.is_day and state.previous_robots is not None
            and turn.robot_roles_observed):
        for robot_id, (target, _) in state.previous_robots.items():
            if robot_id not in robot_hp and target in SIDES:
                robots[target]["disappeared"] += 1
    current = {"roundNo": turn.round_no, "night": night,
               "phase": "day" if turn.is_day else "night",
               "ourTeam": turn.team_type, "complete": not unknowns,
               "unknowns": unknowns,
               "unknownTargetRobots": unknown_target if turn.robot_roles_observed else None,
               "sides": {}}
    for side in SIDES:
        previous_base = state.previous_bases.get(side)
        base = bases[side]
        same_base = (consecutive and previous_base is not None and base is not None
                     and previous_base[:2] == base[:2])
        base_loss = max(0, previous_base[2] - base[2]) if same_base else 0
        old_walls = state.previous_walls.get(side, {}) if consecutive else {}
        new_walls = walls[side]
        shared = old_walls.keys() & new_walls.keys()
        changes = {
            "count": len(new_walls),
            "damaged": sum(new_walls[i][1] < old_walls[i][1] and
                           new_walls[i][0] == old_walls[i][0] for i in shared),
            "disappeared": len(old_walls.keys() - new_walls.keys()),
            "new": len(new_walls.keys() - old_walls.keys()) if consecutive else 0,
            "upgraded": sum(new_walls[i][0] > old_walls[i][0] for i in shared),
            "repairedSameLevel": sum(new_walls[i][1] > old_walls[i][1] and
                                     new_walls[i][0] == old_walls[i][0] for i in shared),
        }
        current["sides"][side] = {
            "base": {"id": base[0], "level": base[1], "hp": base[2],
                     "observedHpLoss": base_loss} if base else None,
            "walls": changes, "robots": robots[side],
            "criticalWallBreach": "unknown",
        }
        if turn.is_day and consecutive:
            if (previous_base is not None and base is not None
                    and (previous_base[:2] != base[:2] or base[2] > previous_base[2])):
                state.changed_defense[side] = True
            if any(changes[name] for name in (
                "new", "disappeared", "upgraded", "repairedSameLevel",
            )):
                state.changed_defense[side] = True
        if state.active_night is not None and not turn.is_day:
            if base_loss:
                state.active_night.damage[side] = True
            if not same_base:
                state.active_night.base_comparable[side] = False
    state.current = current
    if turn.is_day:
        state.pending_prediction = _prediction(
            state, night, turn.round_no, turn.team_type,
        )
    else:
        record = state.active_night
        if record.last_round is not None and turn.round_no != record.last_round + 1:
            record.complete = False
            record.unknowns.add("missing_night_frame")
        if unknowns:
            record.unknowns.update(unknowns)
        record.frames += 1
        record.last_round = turn.round_no
    state.previous_bases = bases
    state.previous_walls = walls
    state.previous_robots = (
        robot_hp if turn.robot_roles_observed and len(turn.robots) <= ID_LIMIT
        else None
    )
    state.last_round = turn.round_no


def shadow_diagnostic(state: ShadowState) -> dict[str, Any]:
    prediction = (state.active_night.prediction if state.active_night is not None
                  else state.pending_prediction)
    return {"current": state.current,
            "prediction": prediction, "verification": state.last_verification,
            "recentNights": list(state.recent_nights)}
