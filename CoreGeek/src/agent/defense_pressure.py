from dataclasses import dataclass, field
from typing import Any

from .protocol import ROUNDS_PER_DAY, Turn

MAX_WAVE_NIGHTS = 10
MAX_WAVE_ROBOT_IDS = 512
MAX_WAVE_FORECASTS = 10


@dataclass(slots=True)
class DefenseSnapshot:
    station_health: int
    role_health: int
    weapon_health: int
    weapon_count: int
    wall_count: int


@dataclass(slots=True)
class WaveNight:
    day: int
    target_team: str
    robot_types_by_id: dict[int, str] = field(default_factory=dict)
    complete: bool = False
    first_observed_round: int = 0
    last_observed_round: int = 0
    truncated: bool = False
    start_defense: DefenseSnapshot | None = None
    end_defense: DefenseSnapshot | None = None
    loss_finalized: bool = False
    loss_completeness: str = "lower_bound"

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for kind in self.robot_types_by_id.values():
            counts[kind] = counts.get(kind, 0) + 1
        return counts


@dataclass(slots=True)
class WaveForecast:
    target_day: int
    basis_days: tuple[int, ...]
    sample_count: int
    method: str
    counts: dict[str, int]
    growth: dict[str, int]
    uncertain_types: tuple[str, ...] = ()
    actual_counts: dict[str, int] | None = None
    absolute_error: dict[str, int] | None = None


def observe_pressure(turn: Turn, state: Any) -> None:
    day = _day(turn.round_no)
    if turn.is_day:
        _finalize_previous_night(turn, state, day)
        return

    night = next((item for item in state.wave_history if item.day == day), None)
    if night is None:
        night = WaveNight(
            day=day,
            target_team=turn.team_type,
            complete=turn.round_in_day == 71,
            first_observed_round=turn.round_no,
            last_observed_round=turn.round_no,
            start_defense=_snapshot(turn),
        )
        state.wave_history.append(night)
        if len(state.wave_history) > MAX_WAVE_NIGHTS:
            del state.wave_history[:-MAX_WAVE_NIGHTS]
            state.wave_history_truncated = True
    else:
        night.last_observed_round = max(night.last_observed_round, turn.round_no)
    night.end_defense = _snapshot(turn)
    for robot in turn.robots:
        if robot.target_team != turn.team_type:
            continue
        if robot.robot_id in night.robot_types_by_id:
            continue
        if len(night.robot_types_by_id) >= MAX_WAVE_ROBOT_IDS:
            night.truncated = True
            continue
        night.robot_types_by_id[robot.robot_id] = robot.kind
    if night.truncated:
        night.complete = False
    _score_forecast(state, night)
    _store_next_forecast(state, day)


def pressure_diagnostic(turn: Turn, state: Any) -> dict[str, Any]:
    day = _day(turn.round_no)
    current = next((item for item in state.wave_history if item.day == day), None)
    previous = next(
        (item for item in reversed(state.wave_history) if item.day < day), None,
    )
    forecast_target = day if turn.is_day else day + 1
    forecast = next(
        (
            item for item in reversed(state.wave_forecasts)
            if item.target_day == forecast_target
        ),
        None,
    )
    reasons = _assessment_reasons(turn, state, forecast, previous)
    return {
        "historyCount": len(state.wave_history),
        "historyLimit": MAX_WAVE_NIGHTS,
        "historyTruncated": state.wave_history_truncated,
        "robotIdLimitPerNight": MAX_WAVE_ROBOT_IDS,
        "observationMode": "combined_observed",
        "currentNight": _night_dump(current),
        "previousNightLoss": _loss_dump(previous),
        "forecast": _forecast_dump(forecast),
        "forecastHistory": [
            _forecast_dump(item)
            for item in state.wave_forecasts
            if item.target_day < forecast_target
        ][-MAX_WAVE_FORECASTS:],
        "assessment": {
            "reasons": reasons,
            "currentDefense": _defense_dump(turn, state),
            "guaranteesSurvival": False,
        },
    }


def _day(round_no: int) -> int:
    return (round_no - 1) // ROUNDS_PER_DAY + 1


def _snapshot(turn: Turn) -> DefenseSnapshot:
    station = turn.station()
    controllable = turn.controllable()
    weapons = turn.weapons()
    walls = turn.walls()
    return DefenseSnapshot(
        station_health=station.health if station is not None else 0,
        role_health=sum(role.health for role in controllable),
        weapon_health=sum(weapon.health for weapon in weapons),
        weapon_count=len(weapons),
        wall_count=len(walls),
    )


def _finalize_previous_night(turn: Turn, state: Any, day: int) -> None:
    previous = next(
        (item for item in reversed(state.wave_history) if item.day == day - 1),
        None,
    )
    if (
        previous is not None
        and previous.end_defense is not None
        and not previous.loss_finalized
    ):
        if turn.round_in_day == 1:
            previous.end_defense = _snapshot(turn)
            if previous.complete:
                previous.loss_completeness = "complete"
        previous.loss_finalized = True


def _score_forecast(state: Any, night: WaveNight) -> None:
    if not night.complete:
        return
    for forecast in state.wave_forecasts:
        if forecast.target_day != night.day or forecast.actual_counts is not None:
            continue
        actual = night.counts()
        forecast.actual_counts = actual
        kinds = set(forecast.counts) | set(actual)
        forecast.absolute_error = {
            kind: abs(forecast.counts.get(kind, 0) - actual.get(kind, 0))
            for kind in sorted(kinds)
        }


def _store_next_forecast(state: Any, day: int) -> None:
    complete = [item for item in state.wave_history if item.complete]
    latest = complete[-1] if complete else None
    target_day = day + 1
    if latest is None:
        proposed = WaveForecast(target_day, (), 0, "growth_unknown", {}, {})
    elif (
        latest.day == day
        and len(complete) >= 2
        and complete[-2].day + 1 == latest.day
    ):
        earlier = complete[-2]
        previous_counts = earlier.counts()
        latest_counts = latest.counts()
        kinds = set(previous_counts) | set(latest_counts)
        uncertain_types = tuple(sorted(
            kind for kind in kinds
            if latest_counts.get(kind, 0) < previous_counts.get(kind, 0)
        ))
        growth = {
            kind: max(latest_counts.get(kind, 0) - previous_counts.get(kind, 0), 0)
            for kind in sorted(kinds)
        }
        counts = {
            kind: (
                max(previous_counts.get(kind, 0), latest_counts.get(kind, 0))
                if kind in uncertain_types
                else latest_counts.get(kind, 0) + growth[kind]
            )
            for kind in sorted(kinds)
            if max(previous_counts.get(kind, 0), latest_counts.get(kind, 0)) > 0
        }
        proposed = WaveForecast(
            target_day,
            (earlier.day, latest.day),
            2,
            (
                "adjacent_delta_with_decline_unknown"
                if uncertain_types
                else "latest_nonnegative_adjacent_delta"
            ),
            counts,
            growth,
            uncertain_types,
        )
    else:
        proposed = WaveForecast(
            target_day,
            (latest.day,),
            1,
            "growth_unknown",
            latest.counts(),
            {},
        )
    existing = next(
        (item for item in state.wave_forecasts if item.target_day == target_day),
        None,
    )
    if existing is None:
        state.wave_forecasts.append(proposed)
        del state.wave_forecasts[:-MAX_WAVE_FORECASTS]


def _night_dump(night: WaveNight | None) -> dict[str, Any] | None:
    if night is None:
        return None
    return {
        "day": night.day,
        "targetTeam": night.target_team,
        "counts": night.counts(),
        "completeness": "complete" if night.complete else "lower_bound",
        "firstObservedRound": night.first_observed_round,
        "lastObservedRound": night.last_observed_round,
        "robotIdsRetained": len(night.robot_types_by_id),
        "truncated": night.truncated,
        "source": "combined_observed",
    }


def _forecast_dump(forecast: WaveForecast | None) -> dict[str, Any] | None:
    if forecast is None:
        return None
    return {
        "targetDay": forecast.target_day,
        "basisDays": list(forecast.basis_days),
        "sampleCount": forecast.sample_count,
        "method": forecast.method,
        "counts": dict(forecast.counts),
        "growth": dict(forecast.growth),
        "uncertainTypes": list(forecast.uncertain_types),
        "actualCounts": (
            dict(forecast.actual_counts)
            if forecast.actual_counts is not None else None
        ),
        "absoluteError": (
            dict(forecast.absolute_error)
            if forecast.absolute_error is not None else None
        ),
    }


def _loss_dump(night: WaveNight | None) -> dict[str, Any] | None:
    if (
        night is None
        or night.start_defense is None
        or night.end_defense is None
    ):
        return None
    start = night.start_defense
    end = night.end_defense
    return {
        "day": night.day,
        "completeness": night.loss_completeness,
        "stationHealth": max(start.station_health - end.station_health, 0),
        "roleHealth": max(start.role_health - end.role_health, 0),
        "weaponHealth": max(start.weapon_health - end.weapon_health, 0),
        "weaponsLost": max(start.weapon_count - end.weapon_count, 0),
        "wallsLost": max(start.wall_count - end.wall_count, 0),
    }


def _assessment_reasons(
    turn: Turn,
    state: Any,
    forecast: WaveForecast | None,
    previous: WaveNight | None,
) -> list[str]:
    defense = _defense_dump(turn, state)
    previous_loss = _loss_dump(previous)
    observed_loss = previous_loss is not None and any(
        previous_loss[key] > 0
        for key in (
            "stationHealth", "roleHealth", "weaponHealth",
            "weaponsLost", "wallsLost",
        )
    )
    reasons = []
    if (
        defense["missingFixedWalls"]
        or defense["damagedWalls"]
        or defense["weaponCount"] < 3
        or defense["controllableRoles"] < defense["weaponCount"]
        or defense["stationHealth"] <= 0
        or observed_loss
    ):
        reasons.append("known_deficit")
    if forecast is not None and any(value > 0 for value in forecast.growth.values()):
        reasons.append("rising_pressure")
    if (
        forecast is None
        or forecast.method == "growth_unknown"
        or bool(forecast.uncertain_types)
    ):
        reasons.append("unknown")
    if not reasons or reasons == ["unknown"]:
        reasons.append("no_observed_deficit")
    return reasons


def _defense_dump(turn: Turn, state: Any) -> dict[str, Any]:
    walls = turn.walls()
    wall_positions = {wall.pos for wall in walls}
    targets = tuple(state.fortification_targets[:6])
    return {
        "stationHealth": turn.station().health if turn.station() is not None else 0,
        "controllableRoles": len(turn.controllable()),
        "weaponCount": len(turn.weapons()),
        "weaponLevels": [weapon.level for weapon in turn.weapons()],
        "wallCount": len(walls),
        "damagedWalls": sum(wall.health < 1000 for wall in walls),
        "missingFixedWalls": sum(target not in wall_positions for target in targets),
    }
