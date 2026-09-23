"""Read-only, uncalibrated observations of pressure on both bases."""

import copy
from dataclasses import dataclass, field
from typing import Any

from .protocol import DAY_ROUNDS, ROUNDS_PER_DAY, Pos, Turn, distance, station_footprint


ALGORITHM_VERSION = "shadow_v3"
ID_LIMIT = 512
SIDES = ("challenger", "defender")
BASE_MAX_HEALTH = (1500, 3000, 4500)


def _night_metrics() -> dict[str, Any]:
    return {
        "base": {"firstDamageRound": None, "cumulativeHpLoss": 0,
                 "longestConsecutiveDamageRounds": 0,
                 "currentDamageStreak": 0},
        "robots": {"firstThreat": None, "lastThreat": None,
                   "peakCount": 0, "peakHp": 0,
                   "cumulativeObservedHpLoss": 0,
                   "nearestBaseChebyshev": None},
        "walls": {name: 0 for name in (
            "damaged", "disappeared", "new", "upgraded",
            "repairedSameLevel", "positionChanged",
        )},
        "criticalWalls": {
            "damagedWallFrames": 0, "nearBaseDamagedWallFrames": 0,
            "sideDamagedWallFrames": 0,
            "damagedWallFramesWithAdjacentRobots": 0,
            "adjacentRobotObservations": 0,
            "longestConsecutiveDamageRounds": 0,
            "damagedWallFramesObservedLowerBound": 0,
            "nearBaseDamagedWallFramesObservedLowerBound": 0,
            "sideDamagedWallFramesObservedLowerBound": 0,
            "damagedWallFramesWithAdjacentRobotsObservedLowerBound": 0,
            "adjacentRobotObservationsLowerBound": 0,
            "coverageComplete": True, "unknowns": [],
            "dawnObserved": False, "dawnWallObservationComplete": None,
            "dawnPriorAdjacentRobotObservations": 0,
        },
        "completeness": {"base": True, "robots": True, "walls": True,
                         "unknowns": []},
    }


def _day_context() -> dict[str, Any]:
    return {
        "base": {"levelChanges": 0, "hpIncrease": 0,
                 "hpDecrease": 0, "identityChanges": 0},
        "walls": {name: 0 for name in (
            "damaged", "disappeared", "new", "upgraded",
            "repairedSameLevel", "positionChanged",
        )},
    }


@dataclass(slots=True)
class NightRecord:
    night: int
    prediction: dict[str, Any]
    frames: int = 0
    first_round: int | None = None
    last_round: int | None = None
    complete: bool = True
    damage: dict[str, bool] = field(default_factory=lambda: {side: False for side in SIDES})
    base_comparable: dict[str, bool] = field(default_factory=lambda: {side: True for side in SIDES})
    unknowns: set[str] = field(default_factory=set)
    metrics: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {side: _night_metrics() for side in SIDES}
    )
    investment_start_base: tuple[int, int, int] | None = None
    investment_start_defense: dict[str, Any] | None = None
    investment_end_defense: dict[str, Any] | None = None
    investment_unknowns: set[str] = field(default_factory=set)
    investment_defense_changes: set[str] = field(default_factory=set)
    first_night_robot_difference: dict[str, Any] | None = None
    wall_damage_streaks: dict[str, dict[int, int]] = field(
        default_factory=lambda: {side: {} for side in SIDES})


@dataclass(slots=True)
class ShadowState:
    last_round: int | None = None
    previous_bases: dict[str, tuple[int, int, int] | None] = field(default_factory=dict)
    previous_walls: dict[str, dict[int, tuple[int, int, int, int]]] = field(
        default_factory=dict)
    previous_walls_complete: dict[str, bool] = field(default_factory=dict)
    previous_robots: dict[int, tuple[str, int]] | None = None
    previous_robot_positions: dict[str, dict[tuple[int, int], int]] | None = None
    pending_prediction: dict[str, Any] | None = None
    active_night: NightRecord | None = None
    recent_nights: list[dict[str, Any]] = field(default_factory=list)
    last_verification: dict[str, Any] | None = None
    day_observation_incomplete: bool = False
    day_context: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {side: _day_context() for side in SIDES}
    )
    changed_defense: dict[str, bool] = field(
        default_factory=lambda: {side: False for side in SIDES}
    )
    current: dict[str, Any] | None = None
    current_pressure: dict[str, Any] | None = None
    previous_our_defense: dict[str, Any] | None = None


def _night_number(round_no: int) -> int:
    return (round_no - 1) // ROUNDS_PER_DAY + 1


def _base(roles: tuple) -> Any:
    return next((role for role in roles if role.kind == "station"), None)


def _side_roles(turn: Turn) -> tuple[tuple[str, tuple], tuple[str, tuple]]:
    enemy = "defender" if turn.team_type == "challenger" else "challenger"
    return ((turn.team_type, turn.ours), (enemy, turn.enemies))


def _our_defense_snapshot(
    turn: Turn, task_role_ids: frozenset[int] | None,
) -> dict[str, Any]:
    return {
        "towers": tuple(sorted(
            (unit.unit_id, unit.kind, unit.level, unit.pos.x, unit.pos.y)
            for unit in turn.weapons()
        )),
        "availableRoles": tuple(sorted(
            unit.unit_id for unit in turn.controllable() if unit.health > 0
        )),
        "taskReservedRoles": (
            tuple(sorted(task_role_ids)) if task_role_ids is not None else None
        ),
    }


def _defense_changes(old: dict[str, Any] | None,
                     new: dict[str, Any] | None) -> set[str]:
    if old is None or new is None:
        return {"defense_snapshot_missing"}
    changes = set()
    for field, reason in (
        ("towers", "towers_changed"),
        ("availableRoles", "role_availability_changed"),
        ("taskReservedRoles", "task_occupancy_changed"),
    ):
        if old[field] is None or new[field] is None:
            changes.add(f"{field}_unknown")
        elif old[field] != new[field]:
            changes.add(reason)
    return changes


def _observe_investment_base(
    record: NightRecord, previous: tuple[int, int, int] | None,
    current: tuple[int, int, int] | None,
) -> None:
    if previous is None or current is None:
        record.investment_unknowns.add("base_missing")
    elif previous[0] != current[0]:
        record.investment_unknowns.add("base_identity_changed")
    elif previous[1] != current[1]:
        record.investment_unknowns.add("base_level_changed")
    elif current[2] > previous[2]:
        record.investment_unknowns.add("base_hp_increased")


def _wall_map(roles: tuple) -> tuple[dict[int, tuple[int, int, int, int]], bool]:
    walls = sorted((role for role in roles if role.kind == "wall"),
                   key=lambda role: role.unit_id)
    result = {}
    truncated = False
    for wall in walls:
        if len(result) < ID_LIMIT:
            result[wall.unit_id] = (
                wall.level, wall.health, wall.pos.x, wall.pos.y,
            )
        else:
            truncated = True
    return result, truncated


def _wall_changes(
    old: dict[int, tuple[int, int, int, int]],
    new: dict[int, tuple[int, int, int, int]],
) -> dict[str, int]:
    shared = old.keys() & new.keys()
    comparable = {i for i in shared if old[i][2:] == new[i][2:]}
    return {
        "damaged": sum(new[i][1] < old[i][1] and new[i][0] == old[i][0]
                       for i in comparable),
        "disappeared": len(old.keys() - new.keys()),
        "new": len(new.keys() - old.keys()),
        "upgraded": sum(new[i][0] > old[i][0] for i in comparable),
        "repairedSameLevel": sum(
            new[i][1] > old[i][1] and new[i][0] == old[i][0]
            for i in comparable
        ),
        "positionChanged": sum(old[i][2:] != new[i][2:] for i in shared),
    }


def _prediction(
    state: ShadowState, night: int, issued_round: int | None, our_team: str,
) -> dict[str, Any]:
    previous = state.recent_nights[-1] if state.recent_nights else None
    based_on = previous["night"] if previous and previous["night"] == night - 1 else None
    current_unknowns = list(state.current["unknowns"]) if state.current else []
    if state.day_observation_incomplete:
        current_unknowns.append("day_observation_incomplete")
    sides = {}
    for side in SIDES:
        baseline = "unknown"
        evidence = []
        if based_on is not None:
            actual = previous["sides"][side]["actualBaseDamage"]
            if actual is True:
                baseline = "elevated"
                evidence.append("prior_base_damage")
            elif actual is False:
                baseline = "low"
                evidence.append("prior_complete_night_without_base_damage")
        context = copy.deepcopy(state.day_context[side])
        changed = state.changed_defense[side]
        reasons = ["new_wave_not_modeled", "defense_effect_not_calibrated"]
        if baseline == "unknown":
            reasons.append("no_scored_previous_night")
        if changed:
            reasons.append("defense_changed")
        if current_unknowns:
            reasons.append("incomplete_current_observation")
        sides[side] = {
            "risk": "unknown",
            "baselineRisk": baseline,
            "persistenceBaseline": {"risk": baseline,
                                    "basedOnNight": based_on},
            "assessment": {"risk": "unknown", "evidence": evidence,
                           "unknowns": reasons},
            "contextChanges": context,
            "changedDefense": changed,
            "criticalWallBreachRisk": "unknown",
            "criticalWallBreachReason": "wall_topology_unverified",
            "reasons": reasons,
        }
    return {
        "status": "issued",
        "night": night,
        "issuedRound": issued_round,
        "algorithmVersion": ALGORITHM_VERSION,
        "basedOnNight": based_on,
        "completeEvidence": based_on is not None
        and previous["sides"][our_team]["actualBaseDamage"] is not None,
        "unknowns": current_unknowns,
        "risk": "unknown",
        "baselineRisk": sides[our_team]["baselineRisk"],
        "persistenceBaseline": sides[our_team]["persistenceBaseline"],
        "assessment": sides[our_team]["assessment"],
        "changedDefense": sides[our_team]["changedDefense"],
        "sides": sides,
    }


def _score(risk: str, actual: bool | None) -> str:
    if actual is None or risk == "unknown":
        return "unscorable"
    if risk == "elevated":
        return "hit" if actual else "false_positive"
    return "false_negative" if actual else "correct_negative"


def _record_base_loss(record: NightRecord, side: str, round_no: int, loss: int) -> bool:
    metrics = record.metrics[side]["base"]
    if not loss:
        metrics["currentDamageStreak"] = 0
        return False
    first = metrics["firstDamageRound"] is None
    if first:
        metrics["firstDamageRound"] = round_no
    metrics["cumulativeHpLoss"] += loss
    metrics["currentDamageStreak"] += 1
    metrics["longestConsecutiveDamageRounds"] = max(
        metrics["longestConsecutiveDamageRounds"], metrics["currentDamageStreak"],
    )
    record.damage[side] = True
    return first


def _verification(
    record: NightRecord, our_team: str, *, status: str, complete: bool,
) -> dict[str, Any]:
    sides = {}
    for side in SIDES:
        metrics = copy.deepcopy(record.metrics[side])
        metrics["criticalWalls"]["unknowns"] = sorted(set(
            metrics["criticalWalls"]["unknowns"]))
        if record.unknowns:
            metrics["completeness"]["base"] = False
        metrics["completeness"]["unknowns"] = sorted(set(
            metrics["completeness"]["unknowns"] + list(record.unknowns)
        ))
        actual = True if record.damage[side] else (
            False if complete and record.base_comparable[side] else None
        )
        frozen = record.prediction["sides"][side]
        baseline_risk = (
            frozen["persistenceBaseline"]["risk"]
            if record.prediction["status"] == "issued" else "unknown"
        )
        sides[side] = {
            "actualBaseDamage": actual,
            "result": _score(frozen["assessment"]["risk"], actual),
            "baselineResult": _score(baseline_risk, actual),
            "firstDamageRound": metrics["base"]["firstDamageRound"],
            "metrics": metrics,
        }
    return {
        "night": record.night,
        "status": status,
        "issuedRound": record.prediction["issuedRound"],
        "algorithmVersion": ALGORITHM_VERSION,
        "complete": complete,
        "completeScope": "round_sequence",
        "observedRange": {"firstRound": record.first_round,
                          "lastRound": record.last_round,
                          "frames": record.frames},
        "unknowns": sorted(record.unknowns),
        "firstNightRobotDifference": copy.deepcopy(
            record.first_night_robot_difference),
        "sides": sides,
        "actualBaseDamage": sides[our_team]["actualBaseDamage"],
        "result": sides[our_team]["result"],
        "baselineResult": sides[our_team]["baselineResult"],
        "firstDamageRound": sides[our_team]["firstDamageRound"],
    }


def _finish_night(
    state: ShadowState, our_team: str, boundary_round: int,
    collect_investment_history: bool,
) -> None:
    record = state.active_night
    if record is None:
        return
    expected_end = record.night * ROUNDS_PER_DAY
    if boundary_round != expected_end + 1:
        record.unknowns.add("missing_day_boundary")
        for side in SIDES:
            critical = record.metrics[side]["criticalWalls"]
            critical["coverageComplete"] = False
            critical["unknowns"].append("dawn_missing")
    complete = record.complete and record.frames == ROUNDS_PER_DAY - DAY_ROUNDS
    complete = complete and record.last_round == expected_end and not record.unknowns
    verification = _verification(
        record, our_team, status="final", complete=complete,
    )
    if collect_investment_history:
        unknowns = set(record.investment_unknowns)
        if not complete:
            unknowns.add("night_or_dawn_sequence_incomplete")
        if record.investment_start_base is None:
            unknowns.add("dusk_base_missing")
        if record.investment_end_defense is None:
            unknowns.add("dawn_defense_missing")
        verification["investmentHistory"] = {
            "night": record.night,
            "damage": (
                record.metrics[our_team]["base"]["cumulativeHpLoss"]
                if not unknowns else None
            ),
            "observedPositiveHpDrops": record.metrics[our_team]["base"][
                "cumulativeHpLoss"
            ],
            "baseId": (
                record.investment_start_base[0]
                if record.investment_start_base else None
            ),
            "baseLevel": (
                record.investment_start_base[1]
                if record.investment_start_base else None
            ),
            "unknowns": sorted(unknowns),
            "defenseAtDusk": record.investment_start_defense,
            "defenseAtDawn": record.investment_end_defense,
            "nightDefenseChanges": sorted(record.investment_defense_changes),
            "nightWallChanges": copy.deepcopy(record.metrics[our_team]["walls"]),
        }
    state.last_verification = verification
    state.recent_nights.append(verification)
    del state.recent_nights[:-3]
    state.active_night = None


def observe_shadow(
    turn: Turn, state: ShadowState,
    task_role_ids: frozenset[int] | None = None,
    *, collect_investment_history: bool = True,
) -> None:
    """Observe one new round without changing Turn or decision state."""
    if state.last_round is not None and turn.round_no <= state.last_round:
        return
    night = _night_number(turn.round_no)
    first_night_round = (night - 1) * ROUNDS_PER_DAY + DAY_ROUNDS + 1
    consecutive = state.last_round == turn.round_no - 1
    our_defense = (
        _our_defense_snapshot(turn, task_role_ids)
        if collect_investment_history else None
    )
    if turn.is_day and state.active_night is not None:
        record = state.active_night
        if turn.round_no == record.night * ROUNDS_PER_DAY + 1:
            if collect_investment_history:
                current_base = _base(turn.ours)
                _observe_investment_base(
                    record, state.previous_bases.get(turn.team_type),
                    (current_base.unit_id, current_base.level, current_base.health)
                    if current_base is not None else None,
                )
                record.investment_end_defense = our_defense
                record.investment_defense_changes.update(_defense_changes(
                    state.previous_our_defense, our_defense,
                ))
            for side, roles in _side_roles(turn):
                base = _base(roles)
                previous = state.previous_bases.get(side)
                if (base is None or previous is None
                        or (base.unit_id, base.level) != previous[:2]):
                    record.base_comparable[side] = False
                    record.metrics[side]["completeness"]["base"] = False
                else:
                    _record_base_loss(
                        record, side, turn.round_no,
                        max(0, previous[2] - base.health),
                    )
                current_walls, limited = _wall_map(roles)
                critical = record.metrics[side]["criticalWalls"]
                critical["dawnObserved"] = True
                critical["dawnWallObservationComplete"] = not limited
                if limited:
                    record.metrics[side]["completeness"]["walls"] = False
                    critical["coverageComplete"] = False
                    critical["unknowns"].append("dawn_wall_id_limit")
                changes = _wall_changes(
                    state.previous_walls.get(side, {}), current_walls,
                )
                for name, count in changes.items():
                    record.metrics[side]["walls"][name] += count
                previous_walls = state.previous_walls.get(side, {})
                damaged_ids = [wall_id for wall_id in
                               previous_walls.keys() & current_walls.keys()
                               if previous_walls[wall_id][0]
                               == current_walls[wall_id][0]
                               and previous_walls[wall_id][2:]
                               == current_walls[wall_id][2:]
                               and current_walls[wall_id][1]
                               < previous_walls[wall_id][1]] if consecutive else []
                if damaged_ids:
                    wall_complete = (not limited
                        and state.previous_walls_complete.get(side, False))
                    critical["damagedWallFramesObservedLowerBound"] += len(
                        damaged_ids)
                    if wall_complete and critical["damagedWallFrames"] is not None:
                        critical["damagedWallFrames"] += len(damaged_ids)
                    else:
                        critical["damagedWallFrames"] = None
                    near_base = 0
                    if base is not None:
                        for wall_id in damaged_ids:
                            _, _, x, y = current_walls[wall_id]
                            if min(distance(Pos(x, y), cell) for cell in
                                   station_footprint(base.pos)) <= 1:
                                near_base += 1
                        critical["nearBaseDamagedWallFramesObservedLowerBound"] += near_base
                        critical["sideDamagedWallFramesObservedLowerBound"] += (
                            len(damaged_ids) - near_base)
                    if wall_complete and base is not None:
                        if critical["nearBaseDamagedWallFrames"] is not None:
                            critical["nearBaseDamagedWallFrames"] += near_base
                        if critical["sideDamagedWallFrames"] is not None:
                            critical["sideDamagedWallFrames"] += (
                                len(damaged_ids) - near_base)
                    else:
                        critical["nearBaseDamagedWallFrames"] = None
                        critical["sideDamagedWallFrames"] = None
                    prior_positions = state.previous_robot_positions
                    if prior_positions is None:
                        critical["dawnPriorAdjacentRobotObservations"] = None
                    else:
                        critical["dawnPriorAdjacentRobotObservations"] += sum(
                            prior_positions[side].get((x + dx, y + dy), 0)
                            for wall_id in damaged_ids
                            for _, _, x, y in (current_walls[wall_id],)
                            for dx in (-1, 0, 1) for dy in (-1, 0, 1))
                    critical["damagedWallFramesWithAdjacentRobots"] = None
                    critical["adjacentRobotObservations"] = None
                    critical["coverageComplete"] = False
                    critical["unknowns"].append(
                        "dawn_robot_association_unknown")
                    next_streaks = {}
                    for wall_id in damaged_ids:
                        streak = record.wall_damage_streaks[side].get(
                            wall_id, 0) + 1
                        next_streaks[wall_id] = streak
                        critical["longestConsecutiveDamageRounds"] = max(
                            critical["longestConsecutiveDamageRounds"], streak)
                    record.wall_damage_streaks[side] = next_streaks
        _finish_night(
            state, turn.team_type, turn.round_no, collect_investment_history,
        )
        state.changed_defense = {side: False for side in SIDES}
    elif state.active_night is not None and state.active_night.night != night:
        _finish_night(
            state, turn.team_type, turn.round_no, collect_investment_history,
        )
        state.changed_defense = {side: False for side in SIDES}
    if (turn.is_day and (state.pending_prediction is None
                         or state.pending_prediction["night"] != night)):
        state.day_observation_incomplete = False
        state.day_context = {side: _day_context() for side in SIDES}
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
        if collect_investment_history and consecutive:
            state.active_night.investment_start_base = (
                state.previous_bases.get(turn.team_type)
            )
            state.active_night.investment_start_defense = (
                state.previous_our_defense
            )
        if (collect_investment_history
                and state.active_night.investment_start_defense is None):
            state.active_night.investment_unknowns.add("dusk_defense_missing")
        if turn.round_no != first_night_round:
            state.active_night.complete = False
            state.active_night.unknowns.add("night_started_late")
        if not consecutive:
            state.active_night.complete = False
            state.active_night.unknowns.add("missing_dusk_boundary")
    bases = {"challenger": None, "defender": None}
    base_units = {}
    walls = {}
    walls_complete = {}
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
        walls_complete[side] = not limited
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
    robot_positions = {side: {} for side in SIDES}
    unknown_target = 0
    for robot_index, robot in enumerate(sorted(
            turn.robots, key=lambda item: item.robot_id)):
        if robot_index < ID_LIMIT:
            robot_hp[robot.robot_id] = (robot.target_team, robot.health)
        if robot.target_team not in SIDES:
            unknown_target += 1
            continue
        group = robots[robot.target_team]
        if robot_index < ID_LIMIT:
            position_key = (robot.pos.x, robot.pos.y)
            robot_positions[robot.target_team][position_key] = (
                robot_positions[robot.target_team].get(position_key, 0) + 1
            )
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
    if state.active_night is not None and not turn.is_day \
            and state.active_night.first_night_robot_difference is None:
        reasons = []
        if turn.round_no != first_night_round:
            reasons.append("night_started_late")
        if not turn.robot_roles_observed:
            reasons.append("robots_missing")
        if "robot_id_limit" in unknowns:
            reasons.append("robot_id_limit")
        if unknown_target:
            reasons.append("unknown_target_robots")
        enemy = "defender" if turn.team_type == "challenger" else "challenger"
        complete = not reasons
        opponent = robots[turn.team_type]["count"] if complete else None
        ours = robots[enemy]["count"] if complete else None
        raw_difference = opponent - ours if complete else None
        inconsistent = raw_difference is not None and raw_difference < 0
        if inconsistent:
            reasons.append("negative_difference")
        state.active_night.first_night_robot_difference = {
            "status": ("inconsistent_observation" if inconsistent else
                       "conditional_observation" if complete else "unknown"),
            "roundNo": turn.round_no,
            "opponentRobotsTargetingUs": opponent,
            "ourRobotsTargetingOpponent": ours,
            "rawDifference": raw_difference,
            "ourAppliedAdditions": 0,
            "ourAppliedAdditionsBasis": "program_does_not_summon",
            "inferredOpponentAdditions": (
                None if inconsistent else raw_difference),
            "inferenceScope": "current_program_only",
            "assumptions": ["equal_natural_robot_totals"],
            "unknowns": sorted(set(reasons + [
                "external_or_deployment_additions_unknown",
                "timing_unverified",
            ])),
            "typeAttribution": "unknown",
        }
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
        changes = _wall_changes(old_walls, new_walls) if consecutive else {
            name: 0 for name in state.day_context[side]["walls"]
        }
        damaged_ids = []
        if consecutive:
            damaged_ids = [wall_id for wall_id in old_walls.keys() & new_walls.keys()
                           if old_walls[wall_id][0] == new_walls[wall_id][0]
                           and old_walls[wall_id][2:] == new_walls[wall_id][2:]
                           and new_walls[wall_id][1] < old_walls[wall_id][1]]
        adjacent_robots = 0
        with_adjacent = 0
        near_base = 0
        for wall_id in damaged_ids:
            _, _, x, y = new_walls[wall_id]
            adjacent = sum(robot_positions[side].get((x + dx, y + dy), 0)
                           for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            adjacent_robots += adjacent
            with_adjacent += adjacent > 0
            base_unit = base_units[side]
            if base_unit is not None and min(
                distance(Pos(x, y), cell) for cell in station_footprint(base_unit.pos)
            ) <= 1:
                near_base += 1
        robots_complete = turn.robot_roles_observed and not unknown_target \
            and len(turn.robots) <= ID_LIMIT
        wall_comparison_complete = (
            consecutive and walls_complete[side]
            and state.previous_walls_complete.get(side, False)
        )
        base_geometry_complete = base_units[side] is not None
        pressure_unknowns = []
        if not consecutive:
            pressure_unknowns.append("frame_sequence_incomplete")
        if not wall_comparison_complete:
            pressure_unknowns.append("wall_observation_incomplete")
        if not walls_complete[side] or state.previous_walls_complete.get(side) is False:
            pressure_unknowns.append("wall_id_limit")
        if not base_geometry_complete:
            pressure_unknowns.append("base_missing")
        if not robots_complete:
            pressure_unknowns.extend(reason for reason in (
                "robots_missing", "robot_id_limit", "unknown_target_robots")
                if reason in unknowns)
        wall_pressure = {
            "comparable": wall_comparison_complete,
            "wallObservationComplete": walls_complete[side],
            "baseGeometryComplete": base_geometry_complete,
            "damagedWalls": (len(damaged_ids)
                             if wall_comparison_complete else None),
            "damagedWallsObservedLowerBound": len(damaged_ids),
            "nearBaseDamagedWalls": (near_base if wall_comparison_complete
                                     and base_geometry_complete else None),
            "sideDamagedWalls": (len(damaged_ids) - near_base
                                  if wall_comparison_complete
                                  and base_geometry_complete else None),
            "damagedWallsWithAdjacentRobots": (
                with_adjacent if wall_comparison_complete and robots_complete
                else None),
            "adjacentRobotsToDamagedWalls": (
                adjacent_robots if wall_comparison_complete and robots_complete
                else None),
            "robotObservationComplete": robots_complete,
            "unknowns": sorted(set(pressure_unknowns)),
            "causalAttribution": "unknown",
            "criticalWallBreachRisk": "unknown",
        }
        current["sides"][side] = {
            "base": {"id": base[0], "level": base[1], "hp": base[2],
                     "observedHpLoss": base_loss} if base else None,
            "walls": {"count": len(new_walls), **changes},
            "robots": robots[side],
            "criticalWallBreach": "unknown",
            "criticalWallPressure": wall_pressure,
        }
        if turn.is_day and consecutive and turn.round_in_day != 1:
            context = state.day_context[side]
            if previous_base is not None and base is not None:
                if previous_base[0] != base[0]:
                    context["base"]["identityChanges"] += 1
                if previous_base[1] != base[1]:
                    context["base"]["levelChanges"] += 1
                context["base"]["hpIncrease"] += max(0, base[2] - previous_base[2])
                context["base"]["hpDecrease"] += max(0, previous_base[2] - base[2])
            for name, count in changes.items():
                context["walls"][name] += count
            state.changed_defense[side] = any(context["base"].values()) or any(
                context["walls"].values()
            )
        if state.active_night is not None and not turn.is_day:
            record = state.active_night
            if collect_investment_history and side == turn.team_type:
                _observe_investment_base(record, previous_base, base)
                record.investment_defense_changes.update(_defense_changes(
                    state.previous_our_defense, our_defense,
                ))
            if not same_base:
                record.base_comparable[side] = False
                record.metrics[side]["completeness"]["base"] = False
                reasons = record.metrics[side]["completeness"]["unknowns"]
                if "base_missing_or_not_comparable" not in reasons:
                    reasons.append("base_missing_or_not_comparable")
            if f"{side}_wall_id_limit" in unknowns:
                record.metrics[side]["completeness"]["walls"] = False
                reasons = record.metrics[side]["completeness"]["unknowns"]
                if "wall_id_limit" not in reasons:
                    reasons.append("wall_id_limit")
            if ("robots_missing" in unknowns or "robot_id_limit" in unknowns
                    or "unknown_target_robots" in unknowns):
                record.metrics[side]["completeness"]["robots"] = False
                for reason in ("robots_missing", "robot_id_limit",
                               "unknown_target_robots"):
                    if reason in unknowns and reason not in record.metrics[side]["completeness"]["unknowns"]:
                        record.metrics[side]["completeness"]["unknowns"].append(reason)
            metrics = record.metrics[side]
            critical = metrics["criticalWalls"]
            critical_unknowns = critical["unknowns"]
            if (record.first_round is None
                    and "missing_dusk_boundary" in record.unknowns):
                critical_unknowns.append("missing_dusk_boundary")
            if record.first_round is not None and not consecutive:
                critical_unknowns.append("frame_sequence_incomplete")
            if not walls_complete[side] or (
                    state.previous_walls_complete.get(side) is False):
                critical_unknowns.append("wall_id_limit")
            if not base_geometry_complete:
                critical_unknowns.append("base_missing")
            if not robots_complete:
                critical_unknowns.extend(reason for reason in (
                    "robots_missing", "robot_id_limit", "unknown_target_robots")
                    if reason in unknowns)
            critical["coverageComplete"] = not critical_unknowns
            if not consecutive:
                record.wall_damage_streaks[side].clear()
            next_streaks = {}
            for wall_id in damaged_ids:
                streak = record.wall_damage_streaks[side].get(wall_id, 0) + 1
                next_streaks[wall_id] = streak
                critical["longestConsecutiveDamageRounds"] = max(
                    critical["longestConsecutiveDamageRounds"], streak)
            record.wall_damage_streaks[side] = next_streaks
            critical["damagedWallFramesObservedLowerBound"] += len(damaged_ids)
            first_observed_night_frame = record.first_round is None
            if (first_observed_night_frame and not wall_comparison_complete
                    and walls_complete[side] and not critical_unknowns):
                pass
            elif critical["damagedWallFrames"] is not None and wall_comparison_complete:
                critical["damagedWallFrames"] += len(damaged_ids)
            else:
                critical["damagedWallFrames"] = None
            if base_geometry_complete:
                critical["nearBaseDamagedWallFramesObservedLowerBound"] += near_base
                critical["sideDamagedWallFramesObservedLowerBound"] += (
                    len(damaged_ids) - near_base)
            if (first_observed_night_frame and not wall_comparison_complete
                    and walls_complete[side] and base_geometry_complete
                    and not critical_unknowns):
                pass
            elif wall_comparison_complete and base_geometry_complete:
                if critical["nearBaseDamagedWallFrames"] is not None:
                    critical["nearBaseDamagedWallFrames"] += near_base
                if critical["sideDamagedWallFrames"] is not None:
                    critical["sideDamagedWallFrames"] += len(damaged_ids) - near_base
            else:
                critical["nearBaseDamagedWallFrames"] = None
                critical["sideDamagedWallFrames"] = None
            if turn.robot_roles_observed:
                critical["damagedWallFramesWithAdjacentRobotsObservedLowerBound"] += with_adjacent
                critical["adjacentRobotObservationsLowerBound"] += adjacent_robots
            if (first_observed_night_frame and not wall_comparison_complete
                    and walls_complete[side] and robots_complete
                    and not critical_unknowns):
                pass
            elif wall_comparison_complete and robots_complete:
                if critical["damagedWallFramesWithAdjacentRobots"] is not None:
                    critical["damagedWallFramesWithAdjacentRobots"] += with_adjacent
                if critical["adjacentRobotObservations"] is not None:
                    critical["adjacentRobotObservations"] += adjacent_robots
            else:
                critical["damagedWallFramesWithAdjacentRobots"] = None
                critical["adjacentRobotObservations"] = None
            if robots[side]["count"] is not None:
                threat = {"roundNo": turn.round_no,
                          "count": robots[side]["count"],
                          "hp": robots[side]["totalHp"],
                          "nearestBaseChebyshev": robots[side]["nearestBaseChebyshev"]}
                if threat["count"]:
                    if metrics["robots"]["firstThreat"] is None:
                        metrics["robots"]["firstThreat"] = threat
                    metrics["robots"]["lastThreat"] = threat
                metrics["robots"]["peakCount"] = max(
                    metrics["robots"]["peakCount"], threat["count"],
                )
                metrics["robots"]["peakHp"] = max(
                    metrics["robots"]["peakHp"], threat["hp"],
                )
                metrics["robots"]["cumulativeObservedHpLoss"] += (
                    robots[side]["observedHpLoss"]
                )
                nearest = threat["nearestBaseChebyshev"]
                prior_nearest = metrics["robots"]["nearestBaseChebyshev"]
                if nearest is not None:
                    metrics["robots"]["nearestBaseChebyshev"] = (
                        nearest if prior_nearest is None else min(prior_nearest, nearest)
                    )
            for name, count in changes.items():
                metrics["walls"][name] += count
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
            for side in SIDES:
                record.metrics[side]["base"]["currentDamageStreak"] = 0
        if "round_gap" in unknowns or "history_starts_late" in unknowns:
            record.complete = False
            record.unknowns.update(set(unknowns) & {"round_gap", "history_starts_late"})
        if record.first_round is None:
            record.first_round = turn.round_no
        record.frames += 1
        record.last_round = turn.round_no
        first_damage = False
        for side in SIDES:
            first_damage |= _record_base_loss(
                record, side, turn.round_no,
                current["sides"][side]["base"]["observedHpLoss"]
                if current["sides"][side]["base"] is not None else 0,
            )
        if first_damage:
            state.last_verification = _verification(
                record, turn.team_type, status="event_observed", complete=False,
            )
    pressure_sides = {}
    for side in SIDES:
        observed = current["sides"][side]
        base = observed["base"]
        nearest = observed["robots"]["nearestBaseChebyshev"]
        active_damage = bool(state.active_night and state.active_night.damage[side])
        if active_damage or (base is not None and base["observedHpLoss"]):
            classification = "base_damage_observed"
        elif base is None or not turn.robot_roles_observed or unknown_target:
            classification = "unknown"
        elif nearest is not None and nearest <= 1:
            classification = "threats_near_base"
        else:
            classification = "no_near_threat_observed"
        pressure_sides[side] = {
            "classification": classification,
            "evidence": {"roundNo": turn.round_no, "night": night,
                         "phase": "day" if turn.is_day else "night",
                         "robotCount": observed["robots"]["count"],
                         "nearestBaseChebyshev": nearest,
                         "observedBaseHpLoss": base["observedHpLoss"] if base else None},
            "nearThresholdChebyshev": 1,
        }
    state.current_pressure = {"sides": pressure_sides}
    state.previous_bases = bases
    if collect_investment_history:
        state.previous_our_defense = our_defense
    state.previous_walls = walls
    state.previous_walls_complete = walls_complete
    state.previous_robots = (
        robot_hp if turn.robot_roles_observed and len(turn.robots) <= ID_LIMIT
        else None
    )
    state.previous_robot_positions = (
        robot_positions if turn.robot_roles_observed
        and len(turn.robots) <= ID_LIMIT and not unknown_target else None
    )
    state.last_round = turn.round_no


def shadow_diagnostic(state: ShadowState) -> dict[str, Any]:
    prediction = (state.active_night.prediction if state.active_night is not None
                  else state.pending_prediction)
    record = state.active_night
    night_summary = ({
        "night": record.night,
        "issuedRound": record.prediction["issuedRound"],
        "algorithmVersion": ALGORITHM_VERSION,
        "observedRange": {"firstRound": record.first_round,
                          "lastRound": record.last_round,
                          "frames": record.frames},
        "frameSequenceCompleteSoFar": record.complete,
        "unknowns": sorted(record.unknowns),
        "firstNightRobotDifference": copy.deepcopy(
            record.first_night_robot_difference),
        "sides": copy.deepcopy(record.metrics),
    } if record is not None else None)
    return {"current": state.current,
            "currentPressure": state.current_pressure,
            "prediction": prediction, "verification": state.last_verification,
            "nightSummary": night_summary,
            "recentNights": list(state.recent_nights)}


def historical_investment_assessment(
    turn: Turn, state: ShadowState,
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe one observed loss repeated arithmetically, without predicting."""
    result: dict[str, Any] = {
        "status": "unknown", "baselineNight": None,
        "observedDamage": None, "currentHp": None,
        "upgradeFullHp": None, "currentMargin": None,
        "upgradeMargin": None, "defenseComparison": "unknown",
        "defenseChanges": [], "defenseAtDusk": None,
        "defenseAtDawn": None,
        "defenseCurrent": None, "unknowns": [],
        "candidate": candidate or {"status": "not_evaluated"},
        "assumption": "repeat_observed_loss_arithmetic_only",
    }
    if not turn.is_day:
        result["unknowns"] = ["current_night_in_progress"]
        return result
    previous_night = _night_number(turn.round_no) - 1
    history = state.recent_nights[-1] if state.recent_nights else None
    if history is None or history["night"] != previous_night:
        result["unknowns"] = ["adjacent_complete_night_missing"]
        return result
    source = history.get("investmentHistory")
    result["baselineNight"] = previous_night
    if source is None:
        result["unknowns"] = ["investment_history_missing"]
        return result
    station = turn.station()
    if station is None:
        result["unknowns"] = ["current_base_missing"]
        return result
    result["currentHp"] = station.health
    result["defenseAtDusk"] = source["defenseAtDusk"]
    result["defenseAtDawn"] = source["defenseAtDawn"]
    result["defenseCurrent"] = state.previous_our_defense
    if station.level in (1, 2):
        result["upgradeFullHp"] = BASE_MAX_HEALTH[station.level]
    unknowns = list(source["unknowns"])
    if (station.unit_id, station.level) != (
        source["baseId"], source["baseLevel"]
    ):
        unknowns.append("current_base_identity_or_level_changed")
    if state.day_context[turn.team_type]["base"]["hpIncrease"]:
        unknowns.append("day_base_hp_increased")
    if state.day_observation_incomplete:
        unknowns.append("day_observation_incomplete")
    result["unknowns"] = sorted(set(unknowns))
    if unknowns:
        return result
    damage = source["damage"]
    result["observedDamage"] = damage
    result["currentMargin"] = station.health - damage
    if result["upgradeFullHp"] is not None:
        result["upgradeMargin"] = result["upgradeFullHp"] - damage
    changes = set(source["nightDefenseChanges"])
    changes.update(_defense_changes(
        source["defenseAtDawn"], state.previous_our_defense,
    ))
    if any(source["nightWallChanges"].values()):
        changes.add("night_walls_changed")
    if any(state.day_context[turn.team_type]["walls"].values()):
        changes.add("day_walls_changed")
    if state.day_context[turn.team_type]["base"]["hpDecrease"]:
        changes.add("day_base_hp_decreased")
    result["defenseChanges"] = sorted(changes)
    result["defenseComparison"] = (
        "unknown" if any(reason.endswith("_unknown") or reason.endswith("_missing")
                         for reason in changes)
        else "changed_unquantified" if changes
        else "approximately_comparable"
    )
    result["status"] = (
        "no_upgrade_candidate" if result["upgradeFullHp"] is None
        else "no_observed_damage" if damage == 0
        else "assessed"
    )
    return result
