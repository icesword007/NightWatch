import json
import unittest

from agent.server import turn_log_record
from analyze_logs import analyze_lines


def row(round_no, *, session=1, build="r-test", hp=100, level=1,
        gold=20, decision=None, processing=100, decide=40):
    return {
        "event": "turn", "buildId": build, "roundNo": round_no,
        "team": {"type": "challenger", "id": "team-a"},
        "decision": {"newsEvidence": {"currentSession": session}, **(decision or {})},
        "timingMs": {"processing": processing, "decide": decide},
        "economy": {"gold": gold}, "commandCount": 0,
        "bases": {"our": {"present": True, "id": "base-a", "health": hp}},
        "ourStructures": {"truncated": False, "items": [
            {"id": "base-a", "type": "station", "level": level, "health": hp},
        ]},
    }


def lines(*records):
    return ["2026-09-21 00:00:00,000 | " + json.dumps(record) for record in records]


class OfflineAnalysisTests(unittest.TestCase):
    def run_rows(self, *records):
        return analyze_lines(lines(*records), pk="PK-synthetic", half="challenger")

    def test_real_server_record_schema_and_sensitive_whitelist(self):
        payload = {
            "roundNo": 1, "teamOur": {"type": "challenger", "teamId": "team-a",
                "goldNum": 0, "roles": [{"id": "base-a", "roleType": "station",
                    "health": 100, "level": 1, "pos": {"x": 1, "y": 1}}]},
            "teamEnemy": {"roles": []}, "errors": [],
        }
        record = turn_log_record(payload, {"roleCommandMap": {}},
            {"processing": 0, "decide": 0},
            decision_trace={"newsEvidence": {"currentSession": 1},
                "executeCmd": "SENSITIVE_SENTINEL"})
        result = analyze_lines(lines(record), pk="PK-synthetic", half="challenger")
        self.assertEqual(result["groups"][0]["frames"], 1)
        self.assertEqual(result["groups"][0]["economy"]["gold"]["zero"], 1)
        self.assertNotIn("SENSITIVE_SENTINEL", json.dumps(result))

    def test_groups_and_sequence_anomalies(self):
        result = self.run_rows(row(1), row(3), row(3), row(2),
            row(1, session=2), row(1, build="r-next"))
        self.assertEqual(len(result["groups"]), 3)
        group = next(group for group in result["groups"]
                     if group["buildId"] == "r-test" and group["session"] == 1)
        self.assertEqual(group["sequence"]["duplicateRounds"], [3])
        self.assertEqual(group["sequence"]["outOfOrderCount"], 2)
        self.assertFalse(group["sequence"]["complete"])

    def test_unknown_number_categories_and_latency(self):
        records = [row(1, gold=0, processing=0), row(2, gold=None, processing=None),
            row(3, gold="bad", processing="bad"), row(4)]
        del records[-1]["economy"]["gold"]
        del records[-1]["timingMs"]["processing"]
        result = self.run_rows(*records)["groups"][0]
        self.assertEqual(result["economy"]["gold"], {
            "known": 1, "zero": 1, "missing": 1, "null": 1, "invalidType": 1,
            "min": 0, "max": 0,
        })
        day = result["latencyMs"]["day"]["processing"]
        self.assertEqual((day["n"], day["p50"], day["over500"]), (1, 0, 0))
        self.assertEqual(day["unknown"], {"missing": 1, "null": 1, "invalidType": 1})

    def test_history_uses_prior_day_and_requires_complete_next_night(self):
        history = {"historyInvestment": {"status": "assessed", "baselineNight": 1,
            "observedDamage": 20}}
        records = [row(n, hp=100, decision=history) for n in range(200, 201)]
        records += [row(n, hp=100 - (n - 200)) for n in range(201, 261)]
        records += [row(261, hp=40)]
        result = self.run_rows(*records)["groups"][0]["historyInvestment"]
        self.assertEqual(result["comparisons"][0]["issuedRound"], 200)
        self.assertEqual(result["comparisons"][0]["targetNight"], 2)
        self.assertEqual(result["comparisons"][0]["observedHpDrop"], 60)
        self.assertEqual(result["comparisons"][0]["status"], "observed")

    def test_gap_heal_level_change_and_missing_base_are_unknown(self):
        history = {"historyInvestment": {"status": "assessed", "baselineNight": 1}}
        records = [row(200, decision=history)]
        records += [row(n, hp=100) for n in range(201, 261) if n != 230]
        records += [row(261, hp=120, level=2)]
        result = self.run_rows(*records)["groups"][0]["historyInvestment"]
        self.assertEqual(result["comparisons"][0]["status"], "unknown")
        self.assertIn("night_frames_missing", result["comparisons"][0]["unknowns"])

    def test_individual_night_discontinuities_are_unknown(self):
        history = {"historyInvestment": {"status": "assessed", "baselineNight": 1}}
        for case, reason in (("heal", "base_healed"),
                             ("level", "base_identity_or_level_changed"),
                             ("missing", "base_missing")):
            with self.subTest(case=case):
                records = [row(200, decision=history)]
                records += [row(n, hp=100 - (n - 201)) for n in range(201, 262)]
                target = records[31]
                if case == "heal":
                    target["bases"]["our"]["health"] = 200
                elif case == "level":
                    target["ourStructures"]["items"][0]["level"] = 2
                else:
                    target["bases"]["our"] = {"present": False}
                comparison = self.run_rows(*records)["groups"][0][
                    "historyInvestment"]["comparisons"][0]
                self.assertEqual(comparison["status"], "unknown")
                self.assertIn(reason, comparison["unknowns"])

    def test_last_prior_assessment_discloses_dusk_gap(self):
        history = {"historyInvestment": {"status": "assessed", "baselineNight": 1}}
        result = self.run_rows(row(199, decision=history), row(200),
                               *[row(n) for n in range(201, 262)])
        comparison = result["groups"][0]["historyInvestment"]["comparisons"][0]
        self.assertEqual(comparison["issuedRound"], 199)
        self.assertEqual(comparison["roundsBeforeNight"], 2)
        self.assertIn("dusk_record_missing", comparison["unknowns"])

    def test_base_observation_and_truncation(self):
        first = row(1, hp=100, level=1)
        second = row(2, hp=130, level=2)
        second["commands"] = {"truncated": False, "items": [
            {"action": "use", "name": "StationUpgradeVoucher1"}]}
        third = row(3)
        third["ourStructures"]["truncated"] = True
        result = self.run_rows(first, second, third)["groups"][0]
        self.assertEqual(result["base"]["observedLevelIncreases"], 1)
        self.assertEqual(result["base"]["requestedUpgradeUses"], 1)
        self.assertEqual(result["truncatedSections"]["ourStructures"], 1)
        self.assertEqual(result["base"]["unknownReasons"],
                         {"structures_missing_or_truncated": 1})

    def test_unrecognized_lines_and_secret_error_never_echo(self):
        result = analyze_lines([
            "SENSITIVE_SENTINEL arbitrary {\"event\":\"turn\"}",
            "2026-09-21 00:00:00,000 | {invalid SENSITIVE_SENTINEL",
            *lines({"event": "task_detail", "prompt": "SENSITIVE_SENTINEL"}),
            *lines(row(1)),
        ], pk="PK-synthetic", half="challenger")
        self.assertEqual(result["input"]["unrecognizedLines"], 2)
        self.assertEqual(result["input"]["ignoredOtherEvents"], 1)
        self.assertNotIn("SENSITIVE_SENTINEL", json.dumps(result))

    def test_malformed_group_ids_are_redacted_without_crashing(self):
        record = row(1)
        record["team"]["id"] = {"token": "SENSITIVE_SENTINEL"}
        record["buildId"] = ["SENSITIVE_SENTINEL"]
        result = self.run_rows(record)
        self.assertIsNone(result["groups"][0]["team"]["id"])
        self.assertIsNone(result["groups"][0]["buildId"])
        self.assertNotIn("SENSITIVE_SENTINEL", json.dumps(result))

    def test_half_mismatch_is_visible(self):
        result = analyze_lines(lines(row(1)), pk="PK-synthetic", half="defender")
        self.assertEqual(result["input"]["halfTeamMismatchFrames"], 1)

    def test_duplicate_rounds_do_not_weight_latency_or_empty_streak(self):
        result = self.run_rows(row(330, processing=10), row(332, processing=900),
                               row(332, processing=900), row(334, processing=20))["groups"][0]
        self.assertEqual(result["frames"], 4)
        self.assertEqual(result["uniqueFrames"], 2)
        self.assertEqual(result["excludedDuplicateFrames"], 2)
        self.assertEqual(result["economy"]["longestEmptyCommandStreak"], 1)
        self.assertEqual(result["latencyMs"]["day"]["processing"]["n"], 1)
        self.assertEqual(result["latencyMs"]["night"]["processing"]["n"], 1)
        self.assertEqual(result["latencyMs"]["day"]["processing"]["over500"], 0)
        self.assertFalse(result["sequence"]["complete"])

    def test_wall_damage_keeps_observed_drop_but_marks_comparison_changed(self):
        history = {"historyInvestment": {"status": "assessed", "baselineNight": 1,
                    "observedDamage": 10, "defenseComparison": "changed_unquantified",
                    "defenseChanges": ["night_walls_changed"],
                    "unknowns": ["day_observation_incomplete"]}}
        records = [row(200, hp=100, decision=history)]
        records += [row(n, hp=100 - (n - 200)) for n in range(201, 262)]
        for record in records:
            record["ourStructures"]["items"].append({
                "id": "wall-a", "type": "wall", "health": 100,
                "level": 1, "pos": {"x": 2, "y": 2}})
        records[31]["ourStructures"]["items"][1]["health"] = 90
        result = self.run_rows(*records)["groups"][0]["historyInvestment"]["comparisons"][0]
        self.assertEqual(result["actualObservedHpDrop"], 60)
        self.assertEqual(result["defenseComparison"], "changed_unquantified")
        self.assertIn("night_walls_changed", result["historicalDefenseChanges"])
        self.assertIn("day_observation_incomplete", result["historicalUnknowns"])

    def test_day_buckets_and_defense_unsafe_count_types(self):
        records = [row(1, decision={"defensePlanning": {"unsafeDayWork": 0}}),
                   row(2, decision={"defensePlanning": {"unsafeDayWork": None}}),
                   row(131, decision={"defensePlanning": {"unsafeDayWork": "bad"}}),
                   row(132)]
        result = self.run_rows(*records)["groups"][0]
        self.assertEqual(len(result["latencyMs"]["byDay"]), 2)
        self.assertEqual(result["latencyMs"]["byDay"][0]["day"]["processing"]["n"], 2)
        self.assertEqual(result["economy"]["unsafeDayWork"], {
            "known": 1, "zero": 1, "missing": 1, "null": 1,
            "invalidType": 1, "min": 0, "max": 0})

    def test_unknown_identity_cannot_close_night(self):
        history = {"historyInvestment": {"status": "no_observed_damage",
                    "baselineNight": 1, "observedDamage": 0}}
        for missing in ("team", "session", "build"):
            with self.subTest(missing=missing):
                records = [row(200, decision=history)]
                records += [row(n) for n in range(201, 262)]
                for record in records:
                    if missing == "team":
                        record["team"].pop("id")
                    elif missing == "session":
                        record["decision"]["newsEvidence"].pop("currentSession")
                    else:
                        record.pop("buildId")
                result = self.run_rows(*records)["groups"][0]
                self.assertFalse(result["sequence"]["complete"])
                comparison = result["historyInvestment"]["comparisons"][0]
                self.assertIn("identity_unknown", comparison["unknowns"])

    def test_zero_damage_history_is_valid(self):
        history = {"historyInvestment": {"status": "no_observed_damage",
                    "baselineNight": 1, "observedDamage": 0}}
        records = [row(200, decision=history)]
        records += [row(n) for n in range(201, 262)]
        comparison = self.run_rows(*records)["groups"][0]["historyInvestment"]["comparisons"][0]
        self.assertEqual(comparison["status"], "observed")
        self.assertEqual(comparison["actualObservedHpDrop"], 0)
        self.assertEqual(comparison["priorObservedDamage"], 0)

    def test_empty_intervals_split_at_phase_day_gap_and_duplicate(self):
        records = [row(n) for n in (69, 70, 71, 72, 74, 131, 132, 134, 134, 135)]
        result = self.run_rows(*records)["groups"][0]["economy"]
        self.assertEqual(result["emptyCommandIntervals"], [
            {"startRound": 69, "endRound": 70, "length": 2,
             "day": 1, "phase": "day"},
            {"startRound": 71, "endRound": 72, "length": 2,
             "day": 1, "phase": "night"},
            {"startRound": 74, "endRound": 74, "length": 1,
             "day": 1, "phase": "night"},
            {"startRound": 131, "endRound": 132, "length": 2,
             "day": 2, "phase": "day"},
            {"startRound": 135, "endRound": 135, "length": 1,
             "day": 2, "phase": "day"},
        ])

    def test_empty_count_requires_nonnegative_integer(self):
        records = [row(n) for n in range(1, 7)]
        for record, value in zip(records, (0, False, 0.0, None, -1, "0")):
            record["commandCount"] = value
        records.append(row(7))
        del records[-1]["commandCount"]
        result = self.run_rows(*records)["groups"][0]["economy"]
        self.assertEqual(result["longestEmptyCommandStreak"], 1)
        self.assertEqual(result["commandCount"], {
            "known": 1, "zero": 1, "missing": 1, "null": 1,
            "invalidType": 4, "min": 0, "max": 0})

    def test_defense_position_change_or_missing_preserves_hp_drop(self):
        history = {"historyInvestment": {"status": "assessed", "baselineNight": 1,
                    "defenseComparison": "approximately_comparable"}}
        for case, expected in (("move", "changed_unquantified"),
                               ("missing", "unknown")):
            with self.subTest(case=case):
                records = [row(200, hp=100, decision=history)]
                records += [row(n, hp=100 - (n - 200)) for n in range(201, 262)]
                for record in records:
                    record["ourStructures"]["items"].append({
                        "id": "tower-a", "type": "gatling", "level": 1,
                        "health": 100, "pos": {"x": 2, "y": 2}})
                if case == "move":
                    records[31]["ourStructures"]["items"][1]["pos"] = {"x": 3, "y": 2}
                else:
                    del records[31]["ourStructures"]["items"][1]["pos"]
                comparison = self.run_rows(*records)["groups"][0][
                    "historyInvestment"]["comparisons"][0]
                self.assertEqual(comparison["actualObservedHpDrop"], 60)
                self.assertEqual(comparison["defenseComparison"], expected)


if __name__ == "__main__":
    unittest.main()
