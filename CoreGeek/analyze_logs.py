#!/usr/bin/env python3
"""Offline, read-only summary of NightWatch server turn records."""

import argparse
from collections import Counter, defaultdict
import json
import math
import re
import sys


LOG_PREFIX = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} \| ({.*)$")
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
TASK_INSTANCE = re.compile(r"^[0-9]{1,10}:[0-9]{1,10}$")
DAY_ROUNDS = 70
ROUNDS_PER_DAY = 130
MISSING = object()
HISTORY_UNKNOWNS = frozenset({
    "adjacent_complete_night_missing", "investment_history_missing",
    "current_base_missing", "current_base_identity_or_level_changed",
    "day_base_hp_increased", "day_observation_incomplete",
    "night_or_dawn_sequence_incomplete", "dusk_base_missing",
    "dawn_defense_missing", "base_missing", "base_identity_changed",
    "base_level_changed", "base_hp_increased", "assessment_error",
    "current_night_in_progress",
})
DEFENSE_CHANGES = frozenset({
    "defense_snapshot_missing", "towers_unknown", "towers_changed",
    "availableRoles_unknown", "role_availability_changed",
    "taskReservedRoles_unknown", "task_occupancy_changed",
    "night_walls_changed", "day_walls_changed", "day_base_hp_decreased",
})
TASK_END_REASONS = frozenset({
    "unknown", "timeout", "death", "left_point", "replaced",
})
SOLVER_STOP_REASONS = frozenset({
    "invalid_envelope_after_correction", "command_after_envelope_correction",
    "command_after_final_request", "command_result_at_deadline",
    "deadline_after_failed_submission", "deadline_repeated_answer",
    "deadline_without_answer", "repeated_cycle_at_deadline",
    "repeated_failed_command", "repeated_identical_tool_result",
    "solver_abandoned", "vacuous_partial_after_correction",
    "vacuous_partial_at_deadline",
})
SIDES = ("challenger", "defender")
RISK = frozenset({"unknown", "low", "elevated"})


def number(value):
    if type(value) is int:
        return True
    return type(value) is float and math.isfinite(value)


def value_at(record, *path):
    value = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return value


def classified(value):
    if value is MISSING:
        return "missing"
    if value is None:
        return "null"
    if not number(value):
        return "invalidType"
    return "known"


def safe_id(value):
    if type(value) in (str, int):
        value = str(value)
        if SAFE_ID.fullmatch(value):
            return value
    return None


def task_instance(value, session):
    if session is None:
        return None
    if not isinstance(value, str) or not TASK_INSTANCE.fullmatch(value):
        return None
    if value.split(":", 1)[0] != str(session):
        return None
    return value


def nonnegative_int(value):
    return value if type(value) is int and value >= 0 else None


def task_summary(records, session, sequence_complete):
    instances = {}
    unattributed = 0

    def entry(instance_id):
        return instances.setdefault(instance_id, {
            "instanceId": instance_id, "firstObservedRound": None,
            "lastObservedRound": None, "submitAnswerRequests": 0,
            "end": None,
            "successStatus": "unknown", "unknowns": [],
        })

    for record in records:
        decision = record.get("decision")
        decision = decision if isinstance(decision, dict) else {}
        instance_id = task_instance(decision.get("taskInstanceId"), session)
        if instance_id is not None:
            item = entry(instance_id)
            round_no = record["roundNo"]
            if item["firstObservedRound"] is None:
                item["firstObservedRound"] = round_no
            item["firstObservedRound"] = min(item["firstObservedRound"], round_no)
            item["lastObservedRound"] = max(item["lastObservedRound"] or 0, round_no)
            commands = record.get("commands")
            if isinstance(commands, dict):
                rows = commands.get("items")
                if isinstance(rows, list):
                    item["submitAnswerRequests"] += sum(
                        isinstance(command, dict)
                        and command.get("action") == "submitAnswer"
                        for command in rows)
                else:
                    item["unknowns"].append("commands_missing")
                if commands.get("truncated") is True:
                    item["unknowns"].append("commands_truncated")
            else:
                item["unknowns"].append("commands_missing")
        elif value_at(record, "task", "active") is True:
            unattributed += 1

        ended = decision.get("taskEnd")
        if not isinstance(ended, dict):
            continue
        ended_id = task_instance(ended.get("instanceId"), session)
        if ended_id is None:
            unattributed += 1
            continue
        item = entry(ended_id)
        if ended.get("endRound") != record["roundNo"]:
            item["unknowns"].append("end_round_mismatch")
            continue
        codes = ended.get("associatedErrorCodes")
        codes = sorted({code for code in codes if type(code) is int and code in (1, 2)}) \
            if isinstance(codes, list) else []
        item["end"] = {
            "acceptedRound": nonnegative_int(ended.get("acceptedRound")),
            "timeoutRound": nonnegative_int(ended.get("timeoutRound")),
            "endRound": nonnegative_int(ended.get("endRound")),
            "endReason": ended.get("endReason")
                if isinstance(ended.get("endReason"), str)
                and ended.get("endReason") in TASK_END_REASONS else "unknown",
            "submissionCount": nonnegative_int(ended.get("submissionCount")),
            "associatedErrorCodes": codes,
            "solverStoppedReason": ended.get("solverStoppedReason")
                if isinstance(ended.get("solverStoppedReason"), str)
                and ended.get("solverStoppedReason") in SOLVER_STOP_REASONS else None,
            "observedTeamScoreDelta": ended.get("observedTeamScoreDelta")
                if type(ended.get("observedTeamScoreDelta")) is int else None,
            "observedGoldDelta": ended.get("observedGoldDelta")
                if type(ended.get("observedGoldDelta")) is int else None,
            "attribution": "unattributed", "successStatus": "unknown",
        }

    for item in instances.values():
        if item["end"] is None:
            item["unknowns"].append("end_not_observed")
        if not sequence_complete:
            item["unknowns"].append("sequence_ambiguous")
        item["unknowns"] = sorted(set(item["unknowns"]))
    return {"instances": [instances[key] for key in sorted(instances)],
            "unattributedFrames": unattributed,
            "source": "turn_commands_visible_only",
            "successStatus": "unknown"}


def pressure_summary(records, identity_unknown):
    nights = {}
    for record in records:
        shadow = value_at(record, "decision", "pressureShadow")
        if not isinstance(shadow, dict):
            continue
        prediction = shadow.get("prediction")
        if isinstance(prediction, dict) and prediction.get("status") == "issued":
            night = nonnegative_int(prediction.get("night"))
            if night:
                nights.setdefault(night, {})["prediction"] = prediction
        verifications = [shadow.get("verification")]
        recent = shadow.get("recentNights")
        if isinstance(recent, list):
            verifications.extend(recent[:3])
        for verification in verifications:
            if not isinstance(verification, dict):
                continue
            night = nonnegative_int(verification.get("night"))
            status = verification.get("status")
            if not night or status not in ("event_observed", "final"):
                continue
            item = nights.setdefault(night, {})
            observed_round = nonnegative_int(value_at(
                verification, "observedRange", "lastRound"))
            if observed_round is None or observed_round > record["roundNo"]:
                observed_round = record["roundNo"]
            current = item.get("verification")
            current_round = item.get("verificationRound", -1)
            if (current is None
                or status == "final" and current.get("status") != "final"
                or status == current.get("status") and observed_round > current_round):
                item["verification"] = verification
                item["verificationRound"] = observed_round

    scores = {name: Counter() for name in ("assessment", "persistenceBaseline")}
    scores_by_side = {side: {name: Counter() for name in scores} for side in SIDES}
    output = []
    for night in sorted(nights):
        item = nights[night]
        prediction = item.get("prediction", {})
        verification = item.get("verification", {})
        predicted_sides = prediction.get("sides") if isinstance(prediction.get("sides"), dict) else {}
        verified_sides = verification.get("sides") if isinstance(verification.get("sides"), dict) else {}
        sides = {}
        for side in SIDES:
            predicted = predicted_sides.get(side, {})
            verified = verified_sides.get(side, {})
            predicted = predicted if isinstance(predicted, dict) else {}
            verified = verified if isinstance(verified, dict) else {}
            assessment = predicted.get("assessment")
            baseline = predicted.get("persistenceBaseline")
            risk_values = {
                "assessment": assessment.get("risk") if isinstance(assessment, dict) else None,
                "persistenceBaseline": baseline.get("risk") if isinstance(baseline, dict) else None,
            }
            actual = verified.get("actualBaseDamage")
            if identity_unknown or (actual is not True and not (
                actual is False and verification.get("status") == "final"
                and verification.get("complete") is True
            )):
                actual = None
            side_summary = {"actualBaseDamage": actual,
                            "observation": verification.get("status")
                                if verification else "missing"}
            for name, risk in risk_values.items():
                risk = risk if isinstance(risk, str) and risk in RISK else "unknown"
                side_summary[name + "Risk"] = risk
                result = (
                    "unscorable" if actual is None or risk == "unknown" else
                    "hit" if risk == "elevated" and actual else
                    "falsePositive" if risk == "elevated" else
                    "falseNegative" if actual else "correctNegative"
                )
                side_summary[name + "Result"] = result
                scores[name][result] += 1
                scores_by_side[side][name][result] += 1
            sides[side] = side_summary
        output.append({"night": night, "verificationStatus":
                       verification.get("status") if verification else "missing",
                       "sides": sides})
    score_keys = ("hit", "falseNegative", "falsePositive",
                  "correctNegative", "unscorable")
    return {"nights": output, "identityUnknown": identity_unknown,
            "coverage": {
        "nights": len(output),
        "predictionNights": sum("prediction" in value for value in nights.values()),
        "finalNights": sum(value.get("verification", {}).get("status") == "final"
                           for value in nights.values()),
        "eventOnlyNights": sum(value.get("verification", {}).get("status") == "event_observed"
                               for value in nights.values()),
    }, "scores": {name: {key: counter[key] for key in score_keys}
        for name, counter in scores.items()},
        "scoresBySide": {side: {name: {key: counter[key] for key in score_keys}
            for name, counter in by_name.items()}
            for side, by_name in scores_by_side.items()}}


def numeric_summary(values):
    categories = Counter(classified(value) for value in values)
    known = [value for value in values if classified(value) == "known"]
    return {
        "known": len(known), "zero": sum(value == 0 for value in known),
        "missing": categories["missing"], "null": categories["null"],
        "invalidType": categories["invalidType"],
        "min": min(known) if known else None,
        "max": max(known) if known else None,
    }


def command_count_summary(values):
    normalized = [value if type(value) is int and value >= 0 else
                  None if value is None else
                  MISSING if value is MISSING else "invalid"
                  for value in values]
    summary = numeric_summary(normalized)
    return summary


def percentile(sorted_values, percent):
    if not sorted_values:
        return None
    return sorted_values[max(0, math.ceil(len(sorted_values) * percent) - 1)]


def latency_summary(values):
    categories = Counter(classified(value) for value in values)
    known = sorted(value for value in values if classified(value) == "known")
    return {
        "n": len(known), "p50": percentile(known, .50),
        "p95": percentile(known, .95),
        "max": known[-1] if known else None,
        "over500": sum(value > 500 for value in known),
        "unknown": {key: categories[key] for key in ("missing", "null", "invalidType")},
    }


def phase(round_no):
    return "day" if (round_no - 1) % ROUNDS_PER_DAY < DAY_ROUNDS else "night"


def base_state(record):
    base = value_at(record, "bases", "our")
    if not isinstance(base, dict) or base.get("present") is not True:
        return None, "base_missing"
    base_id = base.get("id")
    hp = base.get("health", MISSING)
    structures = value_at(record, "ourStructures")
    items = structures.get("items") if isinstance(structures, dict) else None
    if not isinstance(items, list):
        items = []
    matching = [item for item in items if isinstance(item, dict)
                and item.get("type") == "station" and item.get("id") == base_id]
    level = matching[0].get("level", MISSING) if len(matching) == 1 else MISSING
    if type(level) is not int or level < 1:
        level = MISSING
    defense = []
    position_unknown = (not isinstance(structures, dict)
                        or structures.get("truncated") is True
                        or not isinstance(structures.get("items"), list))
    for item in items:
        if not isinstance(item, dict) or item.get("type") == "station":
            continue
        pos = item.get("pos")
        if (isinstance(pos, dict) and type(pos.get("x")) is int
            and type(pos.get("y")) is int
            and pos["x"] >= 0 and pos["y"] >= 0):
            position = (pos["x"], pos["y"])
        else:
            position = None
            position_unknown = True
        defense.append((str(item.get("id")), str(item.get("type")),
                        repr(item.get("level")), repr(item.get("health")), position))
    defense.sort(key=lambda item: (item[:4], repr(item[4])))
    error = (
        "base_identity_or_hp_unknown" if base_id is None or classified(hp) != "known"
        else "base_level_unknown" if level is MISSING else None
    )
    return (base_id, level, hp, defense, position_unknown), error


def history_comparisons(by_round, sequence_complete, identity_unknown):
    comparisons = []
    for night in sorted({(round_no - 1) // ROUNDS_PER_DAY + 1 for round_no in by_round}):
        night_start = (night - 1) * ROUNDS_PER_DAY + DAY_ROUNDS + 1
        night_end = night * ROUNDS_PER_DAY
        dawn = night_end + 1
        prior_day = [round_no for round_no in by_round
                     if night_start - DAY_ROUNDS <= round_no < night_start]
        if not prior_day:
            continue
        candidates = [(round_no, value_at(by_round[round_no], "decision", "historyInvestment"))
                      for round_no in prior_day]
        candidates = [(round_no, value) for round_no, value in candidates
                      if isinstance(value, dict) and value.get("baselineNight") == night - 1]
        if not candidates:
            continue
        issued_round, history = max(candidates)
        baseline = history.get("baselineNight", MISSING)
        if classified(baseline) != "known" or baseline != night - 1:
            continue
        unknowns = []
        if identity_unknown:
            unknowns.append("identity_unknown")
        source_status = history.get("status")
        source_unknowns = history.get("unknowns")
        source_changes = history.get("defenseChanges")
        historical_unknowns = sorted({item for item in source_unknowns
            if isinstance(item, str) and item in HISTORY_UNKNOWNS}) if isinstance(source_unknowns, list) else []
        historical_changes = sorted({item for item in source_changes
            if isinstance(item, str) and item in DEFENSE_CHANGES}) if isinstance(source_changes, list) else []
        if issued_round != night_start - 1 or night_start - 1 not in by_round:
            unknowns.append("dusk_record_missing")
        if not sequence_complete:
            unknowns.append("sequence_ambiguous")
        if any(round_no not in by_round for round_no in range(night_start, night_end + 1)):
            unknowns.append("night_frames_missing")
        if dawn not in by_round:
            unknowns.append("dawn_frame_missing")
        states = []
        if not unknowns:
            for round_no in (issued_round, *range(night_start, dawn + 1)):
                state, error = base_state(by_round[round_no])
                if error is not None:
                    unknowns.append(error)
                    break
                states.append(state)
        damage = 0
        if states:
            first = states[0]
            if any(state[:2] != first[:2] for state in states):
                unknowns.append("base_identity_or_level_changed")
            for previous, current in zip(states, states[1:]):
                if current[2] > previous[2]:
                    unknowns.append("base_healed")
                elif current[2] < previous[2] and current is not states[-1]:
                    damage += previous[2] - current[2]
        defense_changed = bool(states and any(state[3] != states[0][3] for state in states))
        defense_position_unknown = bool(states and any(state[4] for state in states))
        source_comparison = history.get("defenseComparison")
        if source_comparison not in ("unknown", "changed_unquantified", "approximately_comparable"):
            source_comparison = "unknown"
        defense_comparison = (
            "unknown" if unknowns or defense_position_unknown else
            "changed_unquantified" if defense_changed or historical_changes or source_comparison == "changed_unquantified" else
            "unknown" if source_comparison == "unknown" or historical_unknowns else
            "approximately_comparable"
        )
        comparisons.append({
            "issuedRound": issued_round, "roundsBeforeNight": night_start - issued_round,
            "baselineNight": baseline, "targetNight": night,
            "status": "unknown" if unknowns else "observed",
            "unknowns": sorted(set(unknowns)),
            "observedHpDrop": None if unknowns else damage,
            "actualObservedHpDrop": None if unknowns else damage,
            "defenseComparison": defense_comparison,
            "historicalStatus": source_status if source_status in (
                "assessed", "no_observed_damage", "no_upgrade_candidate", "unknown") else "unknown",
            "historicalUnknowns": historical_unknowns,
            "historicalDefenseChanges": historical_changes,
            "comparisonStatus": "unknown" if unknowns or source_status not in (
                "assessed", "no_observed_damage") or defense_comparison != "approximately_comparable"
                else "approximately_comparable",
            "priorObservedDamage": history.get("observedDamage")
                if classified(history.get("observedDamage", MISSING)) == "known" else None,
        })
    return {"comparisons": comparisons}


def summarize_group(key, records):
    team_type, team_id, session, build = key
    identity_unknown = None in key
    rounds = [record["roundNo"] for record in records]
    count = Counter(rounds)
    unique = sorted(count)
    duplicate = sorted(round_no for round_no, total in count.items() if total > 1)
    out_of_order = sum(current <= previous for previous, current in zip(rounds, rounds[1:]))
    gaps = [{"after": left, "before": right, "missing": right - left - 1}
            for left, right in zip(unique, unique[1:]) if right > left + 1]
    sequence_complete = not duplicate and not out_of_order and not gaps and not identity_unknown
    by_round = {record["roundNo"]: record for record in records if count[record["roundNo"]] == 1}
    unique_records = [record for record in records if count[record["roundNo"]] == 1]
    latency = {}
    for part in ("day", "night"):
        subset = [record for record in unique_records if phase(record["roundNo"]) == part]
        latency[part] = {name: latency_summary(
            [value_at(record, "timingMs", name) for record in subset])
            for name in ("processing", "decide")}
    latency["byDay"] = [{
        "dayNumber": day_number,
        **{part: {name: latency_summary([
            value_at(record, "timingMs", name) for record in unique_records
            if (record["roundNo"] - 1) // ROUNDS_PER_DAY + 1 == day_number
            and phase(record["roundNo"]) == part])
            for name in ("processing", "decide")}
            for part in ("day", "night")},
    } for day_number in sorted({(record["roundNo"] - 1) // ROUNDS_PER_DAY + 1
                                for record in unique_records})]
    gold = numeric_summary([value_at(record, "economy", "gold") for record in unique_records])
    base_states = [base_state(record) for record in unique_records]
    base_health = numeric_summary([state[2] if state is not None else MISSING
                                   for state, error in base_states])
    base_levels = numeric_summary([state[1] if state is not None else MISSING
                                   for state, error in base_states])
    base_errors = Counter(error for state, error in base_states if error)
    level_increases = 0
    for left, right in zip(unique_records, unique_records[1:]):
        if right["roundNo"] != left["roundNo"] + 1:
            continue
        previous, _ = base_state(left)
        current, _ = base_state(right)
        if (previous and current and previous[0] == current[0]
            and type(previous[1]) is int and type(current[1]) is int
            and current[1] > previous[1]):
            level_increases += 1
    upgrade_requests = 0
    truncated = Counter()
    for record in unique_records:
        commands = value_at(record, "commands")
        if isinstance(commands, dict):
            if commands.get("truncated") is True:
                truncated["commands"] += 1
            items = commands.get("items")
            if isinstance(items, list):
                upgrade_requests += sum(isinstance(item, dict)
                    and item.get("action") == "use"
                    and item.get("name") in ("StationUpgradeVoucher1", "StationUpgradeVoucher2")
                    for item in items)
        for field in ("ourStructures", "controlledRoles", "robots", "errorCodes"):
            section = record.get(field)
            if isinstance(section, dict) and section.get("truncated") is True:
                truncated[field] += 1
    intervals = []
    current_interval = None
    for record in unique_records:
        round_no = record["roundNo"]
        count_value = value_at(record, "commandCount")
        is_empty = type(count_value) is int and count_value == 0
        day_number = (round_no - 1) // ROUNDS_PER_DAY + 1
        part = phase(round_no)
        if current_interval is not None and (not is_empty
            or round_no != current_interval["endRound"] + 1
            or day_number != current_interval["day"]
            or part != current_interval["phase"]):
            intervals.append(current_interval)
            current_interval = None
        if is_empty:
            if current_interval is None:
                current_interval = {"startRound": round_no, "endRound": round_no,
                                    "length": 1, "day": day_number, "phase": part}
            else:
                current_interval["endRound"] = round_no
                current_interval["length"] += 1
    if current_interval is not None:
        intervals.append(current_interval)
    empty_streak = max((item["length"] for item in intervals), default=0)
    return {
        "team": {"type": team_type, "id": team_id}, "session": session,
        "buildId": build, "frames": len(records),
        "uniqueFrames": len(unique_records),
        "excludedDuplicateFrames": len(records) - len(unique_records),
        "sequence": {"firstRound": unique[0], "lastRound": unique[-1],
            "duplicateRounds": duplicate, "outOfOrderCount": out_of_order,
            "gaps": gaps, "identityUnknown": identity_unknown,
            "complete": sequence_complete,
            "headTailCompleteness": "unknown_without_declared_range"},
        "latencyMs": latency,
        "economy": {"gold": gold, "longestEmptyCommandStreak": empty_streak,
            "emptyCommandIntervals": intervals,
            "commandCount": command_count_summary([
                value_at(record, "commandCount") for record in unique_records]),
            "pathComputations": numeric_summary([
                value_at(record, "decision", "economyPlanning", "pathComputations")
                for record in unique_records]),
            "pathExpansions": numeric_summary([
                value_at(record, "decision", "economyPlanning", "pathExpansions")
                for record in unique_records]),
            "pathSearches": numeric_summary([
                value_at(record, "decision", "economyPlanning", "pathSearches")
                for record in unique_records]),
            "unsafeDayWork": numeric_summary([
                value_at(record, "decision", "defensePlanning", "unsafeDayWork")
                for record in unique_records]),
            "truncatedPlanningRounds": sum(
                value_at(record, "decision", "economyPlanning", "truncatedReason")
                not in (MISSING, None, "") for record in unique_records)},
        "base": {"health": base_health, "level": base_levels,
                 "unknownReasons": dict(sorted(base_errors.items())),
                 "requestedUpgradeUses": upgrade_requests,
                 "observedLevelIncreases": level_increases},
        "truncatedSections": dict(sorted(truncated.items())),
        "errors": {"count": numeric_summary([value_at(record, "errorCount")
                                             for record in unique_records])},
        "historyInvestment": history_comparisons(by_round, sequence_complete, identity_unknown),
        "tasks": task_summary(unique_records, session, sequence_complete),
        "pressure": pressure_summary(unique_records, identity_unknown),
    }


def parse_line(line):
    match = LOG_PREFIX.match(line.rstrip("\r\n"))
    if match is None:
        return None
    try:
        record = json.loads(match.group(1))
    except (ValueError, TypeError):
        return None
    if not isinstance(record, dict):
        return None
    return record


def analyze_lines(lines, *, pk, half):
    if not pk or half not in ("challenger", "defender"):
        raise ValueError("explicit PK and half (challenger/defender) required")
    groups = defaultdict(list)
    unrecognized = invalid_round = ignored_events = half_mismatch = 0
    for line in lines:
        record = parse_line(line)
        if record is None:
            unrecognized += 1
            continue
        if record.get("event") != "turn":
            ignored_events += 1
            continue
        round_no = record.get("roundNo")
        if type(round_no) is not int or round_no < 1:
            invalid_round += 1
            continue
        team = record.get("team")
        if not isinstance(team, dict):
            team = {}
        session = value_at(record, "decision", "newsEvidence", "currentSession")
        if session is MISSING or session is None:
            session = value_at(record, "decision", "pressureShadow", "session")
        if session is MISSING or type(session) is not int:
            session = None
        team_type = team.get("type") if team.get("type") in ("challenger", "defender") else None
        if team_type is not None and team_type != half:
            half_mismatch += 1
        key = (team_type, safe_id(team.get("id")), session,
               safe_id(record.get("buildId")))
        groups[key].append(record)
    output = [summarize_group(key, records) for key, records in groups.items()]
    output.sort(key=lambda group: (str(group["team"]), str(group["session"]),
                                   str(group["buildId"])))
    return {"pk": pk, "half": half,
            "input": {"unrecognizedLines": unrecognized,
                      "ignoredOtherEvents": ignored_events,
                      "invalidRoundRecords": invalid_round,
                      "halfTeamMismatchFrames": half_mismatch},
            "groups": output,
            "acceptanceEvidence": {"fullSha": "unknown", "packageHash": "unknown",
                                   "terminalSignal": "unknown", "platformVerdict": "unknown"},
            "scope": "offline_observation_only; no platform pass or counterfactual conclusion"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="local server stdout log")
    parser.add_argument("--pk", required=True)
    parser.add_argument("--half", required=True, choices=("challenger", "defender"))
    args = parser.parse_args(argv)
    try:
        with open(args.input, encoding="utf-8") as source:
            result = analyze_lines(source, pk=args.pk, half=args.half)
    except (OSError, UnicodeError):
        parser.exit(2, "input read failed\n")
    json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
