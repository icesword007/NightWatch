import copy
import itertools
import json
import unittest
from pathlib import Path

from agent.pressure_shadow import ShadowState, observe_shadow, shadow_diagnostic
from agent.protocol import Turn
from agent.server import task_log_record, turn_log_record
from analyze_logs import analyze_lines


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


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

    def test_real_task_and_turn_records_deduplicate_same_submit(self):
        payload = {
            "roundNo": 5, "phaseTask": "SENSITIVE_TASK", "errors": [],
            "lastCmdResult": "[exitCode:0]\nSENSITIVE_RESULT",
            "teamOur": {"type": "challenger", "teamId": "team-a",
                        "goldNum": 10, "totalScore": 0, "roles": []},
            "teamEnemy": {"roles": []},
        }
        response = {"roleCommandMap": {"7": {
            "action": "submitAnswer", "taskAnswer": "SENSITIVE_ANSWER"}}}
        trace = {"newsEvidence": {"currentSession": 1},
                 "taskInstanceId": "1:5", "solverState": "waiting_llm",
                 "solverReason": "llm_pending",
                 "taskEnvelopeCorrectionRequested": True}
        turn = turn_log_record(payload, response, {"processing": 1},
                               decision_trace=trace)
        task = task_log_record(payload, response, decision_trace=trace)
        result = self.run_rows(turn, task)
        summary = result["groups"][0]["tasks"]
        instance = summary["instances"][0]
        self.assertEqual(instance["submitAnswerRequests"], 1)
        self.assertEqual(instance["evidenceSources"], ["task", "turn"])
        self.assertEqual(instance["solverStates"], ["waiting_llm"])
        self.assertEqual(instance["solverReasons"], ["llm_pending"])
        self.assertEqual(instance["commandResultClasses"], ["completed"])
        self.assertEqual(instance["envelopeCorrectionRequested"], True)
        self.assertIn("envelopeCorrectionRequested", instance["diagnosticFlags"])
        self.assertEqual(result["input"]["taskEventRecords"], 1)
        self.assertEqual(instance["timeline"][0]["roundNo"], 5)
        self.assertEqual(instance["timeline"][0]["sources"], ["task", "turn"])
        self.assertEqual(instance["timeline"][0]["submitObservations"], 2)
        self.assertNotIn("SENSITIVE_", json.dumps(result))

    def test_task_only_record_is_grouped_and_missing_identity_is_not_merged(self):
        base = {"event": "task", "buildId": "r-test", "roundNo": 8,
                "team": {"type": "challenger", "id": "team-a"},
                "sessionIndex": 1, "taskInstanceId": "1:8",
                "solverState": "waiting", "requestFingerprint": "a" * 64,
                "input": {"cmdResult": {"class": "completed", "value": "SECRET"},
                          "errors": [{"code": 2, "description": "SECRET"}]}}
        unknown = copy.deepcopy(base)
        unknown["sessionIndex"] = None
        result = self.run_rows(base, unknown)
        self.assertEqual(len(result["groups"]), 2)
        known = next(group for group in result["groups"] if group["session"] == 1)
        instance = known["tasks"]["instances"][0]
        self.assertEqual(instance["commandResultClasses"], ["completed"])
        self.assertEqual(instance["relatedErrorCodes"], [])
        self.assertIn("error_association_unknown", instance["unknowns"])
        self.assertEqual(instance["successStatus"], "unknown")
        self.assertIn("turn_record_missing", instance["unknowns"])
        self.assertNotIn("SECRET", json.dumps(result))

    def test_production_ended_errors_attach_to_ended_instance(self):
        payload = {"roundNo": 9, "phaseTask": "NEW_SECRET",
            "errors": [{"errorCode": 2, "description": "SECRET_ERROR"}],
            "teamOur": {"type": "challenger", "teamId": "team-a",
                        "goldNum": 0, "totalScore": 0, "roles": []}}
        trace = {"newsEvidence": {"currentSession": 1},
                 "taskInstanceId": "1:2", "solverState": "solving",
                 "taskEnd": {"instanceId": "1:1", "endRound": 9,
                             "associatedErrorCodes": [2]}}
        task = task_log_record(payload, {"roleCommandMap": {}},
                               decision_trace=trace)
        result = self.run_rows(task)["groups"][0]["tasks"]
        by_id = {item["instanceId"]: item for item in result["instances"]}
        self.assertEqual(by_id["1:1"]["relatedErrorCodes"], [2])
        self.assertEqual(by_id["1:2"]["relatedErrorCodes"], [])
        self.assertNotIn("SECRET", json.dumps(result))

    def test_cross_source_submit_without_shared_fingerprint_is_unknown(self):
        turn = row(4, decision={"taskInstanceId": "1:4"})
        turn["commands"] = {"items": [{"action": "submitAnswer"}],
                            "truncated": False}
        task = {"event": "task", "buildId": "r-test", "roundNo": 4,
                "team": turn["team"], "sessionIndex": 1,
                "taskInstanceId": "1:4", "solverState": "submit_pending",
                "output": {"taskActions": [{"action": "submitAnswer"}]}}
        instance = self.run_rows(turn, task)["groups"][0]["tasks"]["instances"][0]
        self.assertIsNone(instance["submitAnswerRequests"])
        self.assertEqual(instance["submitAnswerRequestBounds"], {"min": 1, "max": 2})
        self.assertIn("submit_association_unknown", instance["unknowns"])

    def test_malformed_task_metadata_is_redacted_and_does_not_crash(self):
        task = {"event": "task", "buildId": "r-test", "roundNo": 4,
                "team": {"type": "challenger", "id": "team-a"},
                "sessionIndex": 1, "taskInstanceId": "1:4",
                "requestFingerprint": "b" * 64,
                "solverState": {"secret": "SENSITIVE_SENTINEL"},
                "solverReason": ["SENSITIVE_SENTINEL"],
                "endedTaskInfo": {"instanceId": "1:3", "endRound": 4,
                    "endReason": {}, "solverStoppedReason": [],
                    "associatedErrorCodes": {}},
                "input": {"cmdResult": {"class": []},
                          "errors": [{"code": {"secret": "SENSITIVE_SENTINEL"}}]}}
        output = self.run_rows(task)
        self.assertEqual(len(output["groups"]), 1)
        instances = output["groups"][0]["tasks"]["instances"]
        unknowns = next(item["unknowns"] for item in instances
                        if item["instanceId"] == "1:4")
        self.assertIn("solver_state_invalid", unknowns)
        self.assertIn("solver_reason_invalid", unknowns)
        self.assertIn("command_result_class_invalid", unknowns)
        self.assertNotIn("SENSITIVE_SENTINEL", json.dumps(output))

    def test_task_timeline_is_bounded_and_conflicts_reduce_trust(self):
        records = [{"event": "task", "buildId": "r-test", "roundNo": index,
                    "team": {"type": "challenger", "id": "team-a"},
                    "sessionIndex": 1, "taskInstanceId": "1:4",
                    "requestFingerprint": f"{index:064x}",
                    "solverState": "solving"} for index in range(1, 67)]
        conflict = copy.deepcopy(records[0])
        conflict["solverState"] = "submit_pending"
        result = self.run_rows(*records, conflict)["groups"][0]["tasks"]
        instance = result["instances"][0]
        self.assertEqual(len(instance["timeline"]), 64)
        self.assertTrue(instance["timelineTruncated"])
        self.assertIn("conflicting_task_record", instance["unknowns"])
        self.assertEqual(result["coverage"]["conflictingDuplicateTaskRecords"], 1)

    def test_submit_bounds_deduplicate_repeated_task_before_cross_source_range(self):
        turn = row(2, decision={"taskInstanceId": "1:1"})
        turn["commands"] = {"items": [{"action": "submitAnswer"}],
                            "truncated": False}
        task = {"event": "task", "buildId": "r-test", "roundNo": 2,
                "team": turn["team"], "sessionIndex": 1,
                "taskInstanceId": "1:1", "requestFingerprint": "a" * 64,
                "solverState": "submit_pending",
                "output": {"taskActions": [{"action": "submitAnswer"}]}}
        instance = self.run_rows(turn, task, copy.deepcopy(task))["groups"][0]
        instance = instance["tasks"]["instances"][0]
        self.assertIsNone(instance["submitAnswerRequests"])
        self.assertEqual(instance["submitAnswerRequestBounds"], {"min": 1, "max": 2})
        self.assertEqual(instance["observedSubmitRecords"], 3)

    def test_conflicting_same_fingerprint_submit_count_is_not_exact(self):
        submitted = {"event": "task", "buildId": "r-test", "roundNo": 2,
            "team": {"type": "challenger", "id": "team-a"},
            "sessionIndex": 1, "taskInstanceId": "1:1",
            "requestFingerprint": "a" * 64, "solverState": "submit_pending",
            "output": {"taskActions": [{"action": "submitAnswer"}]}}
        absent = copy.deepcopy(submitted)
        absent["output"]["taskActions"] = []
        instance = self.run_rows(submitted, absent)["groups"][0]
        instance = instance["tasks"]["instances"][0]
        self.assertIsNone(instance["submitAnswerRequests"])
        self.assertEqual(instance["submitAnswerRequestBounds"], {"min": 0, "max": 1})
        self.assertIn("conflicting_task_record", instance["unknowns"])

    def test_submit_count_fingerprint_groups_table_and_order_invariance(self):
        def task(fingerprint, count):
            value = {"event": "task", "buildId": "r-test", "roundNo": 2,
                "team": {"type": "challenger", "id": "team-a"},
                "sessionIndex": 1, "taskInstanceId": "1:1",
                "solverState": "submit_pending",
                "output": {"taskActions": [
                    {"action": "submitAnswer"} for _ in range(count)]}}
            if fingerprint is not None:
                value["requestFingerprint"] = fingerprint * 64
            return value

        def turn(count, fingerprint=None):
            value = row(2, decision={"taskInstanceId": "1:1"})
            value["commands"] = {"items": [
                {"action": "submitAnswer"} for _ in range(count)],
                "truncated": False}
            if fingerprint is not None:
                value["requestFingerprint"] = fingerprint * 64
            return value

        cases = (
            ("same_duplicate", [task("a", 1), task("a", 1)],
             {"min": 1, "max": 1}, 1, (1,)),
            ("same_conflict", [task("a", 0), task("a", 1)],
             {"min": 0, "max": 1}, None, (0, 1)),
            ("different", [task("a", 1), task("b", 1)],
             {"min": 2, "max": 2}, 2, (2,)),
            ("cross_source_same", [turn(1, "a"), task("a", 1)],
             {"min": 1, "max": 1}, 1, (1,)),
            ("mixed_missing", [task("a", 1), task(None, 1)],
             {"min": 1, "max": 2}, None, (1, 2)),
            ("single_missing", [task(None, 1)],
             {"min": 1, "max": 1}, 1, (1,)),
        )
        for name, records, bounds, exact, feasible in cases:
            for order in itertools.permutations(records):
                with self.subTest(name=name, order=[r["event"] for r in order]):
                    instance = self.run_rows(*order)["groups"][0]
                    instance = instance["tasks"]["instances"][0]
                    self.assertEqual(instance["submitAnswerRequestBounds"], bounds)
                    self.assertEqual(instance["submitAnswerRequests"], exact)
                    self.assertTrue(all(bounds["min"] <= value <= bounds["max"]
                                        for value in feasible))

    def test_tasks_end_event_uses_ended_instance_and_never_infers_success(self):
        first = row(1, decision={"taskInstanceId": "1:1"})
        first["commands"] = {"items": [{"action": "submitAnswer"}],
                             "truncated": False}
        ended = row(2, decision={"taskInstanceId": "1:2", "taskEnd": {
            "instanceId": "1:1", "endRound": 2, "endReason": "unknown",
            "submissionCount": 1, "associatedErrorCodes": [],
            "solverStoppedReason": None, "observedTeamScoreDelta": 10,
            "observedGoldDelta": 5, "attribution": "unattributed",
            "successStatus": "unknown", "acceptedRound": 1,
            "timeoutRound": 20, "answer": "SENSITIVE_SENTINEL",
        }})
        result = self.run_rows(first, ended)["groups"][0]["tasks"]
        by_id = {item["instanceId"]: item for item in result["instances"]}
        self.assertEqual(by_id["1:1"]["submitAnswerRequests"], 1)
        self.assertEqual(by_id["1:1"]["end"]["endRound"], 2)
        self.assertEqual(by_id["1:1"]["successStatus"], "unknown")
        self.assertIsNone(by_id["1:2"]["end"])
        self.assertNotIn("SENSITIVE_SENTINEL", json.dumps(result))

    def test_task_end_retains_production_envelope_stop_reasons(self):
        for reason in ("invalid_envelope_after_correction",
                       "command_after_envelope_correction"):
            with self.subTest(reason=reason):
                record = row(2, decision={"taskEnd": {
                    "instanceId": "1:1", "endRound": 2,
                    "solverStoppedReason": reason}})
                instance = self.run_rows(record)["groups"][0]
                self.assertEqual(instance["tasks"]["instances"][0]
                                 ["end"]["solverStoppedReason"], reason)

    def test_task_detail_submitted_items_are_ignored_not_request_count(self):
        turn = row(1, decision={"taskInstanceId": "1:1"})
        turn["requestFingerprint"] = "f" * 64
        detail = {"event": "task_detail", "roundNo": 1,
                  "buildId": "r-test", "team": turn["team"],
                  "requestFingerprint": "f" * 64, "taskInstanceId": "1:1",
                  "submittedAnswers": {"items": [
                      {"text": "SENSITIVE_SENTINEL"}], "truncated": True},
                  "text": {"prompt": "SENSITIVE_SENTINEL"}}
        output = self.run_rows(turn, detail)
        result = output["groups"][0]["tasks"]
        instance = result["instances"][0]
        self.assertEqual(instance["submitAnswerRequests"], 0)
        self.assertEqual(output["input"]["ignoredOtherEvents"], 1)
        self.assertNotIn("SENSITIVE_SENTINEL", json.dumps(result))

    def test_pressure_scores_two_sides_once_per_night_with_event_final_priority(self):
        prediction = {"status": "issued", "night": 1, "sides": {
            "challenger": {"assessment": {"risk": "unknown"},
                           "persistenceBaseline": {"risk": "elevated"}},
            "defender": {"assessment": {"risk": "unknown"},
                         "persistenceBaseline": {"risk": "low"}},
        }}
        event = {"status": "event_observed", "night": 1, "complete": False,
                 "sides": {"challenger": {"actualBaseDamage": True},
                           "defender": {"actualBaseDamage": None}}}
        final = {"status": "final", "night": 1, "complete": True,
                 "sides": {"challenger": {"actualBaseDamage": True},
                           "defender": {"actualBaseDamage": False}}}
        records = [row(71, decision={"pressureShadow": {
            "prediction": prediction, "verification": event}}),
            row(131, decision={"pressureShadow": {
                "prediction": prediction, "verification": final,
                "recentNights": [final]}}),
            row(132, decision={"pressureShadow": {
                "prediction": prediction, "verification": final,
                "recentNights": [final]}})]
        result = self.run_rows(*records)["groups"][0]["pressure"]
        self.assertEqual(result["coverage"]["nights"], 1)
        self.assertEqual(result["coverage"]["finalNights"], 1)
        self.assertEqual(result["scores"]["assessment"]["unscorable"], 2)
        self.assertEqual(result["scores"]["persistenceBaseline"]["hit"], 1)
        self.assertEqual(result["scores"]["persistenceBaseline"]["correctNegative"], 1)
        self.assertEqual(result["nights"][0]["sides"]["defender"]["actualBaseDamage"], False)

    def test_pressure_event_only_and_missing_tail_do_not_create_no_damage(self):
        prediction = {"status": "issued", "night": 1, "sides": {
            "challenger": {"assessment": {"risk": "unknown"},
                           "persistenceBaseline": {"risk": "low"}},
            "defender": {"assessment": {"risk": "unknown"},
                         "persistenceBaseline": {"risk": "low"}},
        }}
        event = {"status": "event_observed", "night": 1, "complete": False,
                 "sides": {"challenger": {"actualBaseDamage": True},
                           "defender": {"actualBaseDamage": False}}}
        record = row(71, decision={"pressureShadow": {
            "prediction": prediction, "verification": event}})
        result = self.run_rows(record)["groups"][0]["pressure"]
        self.assertEqual(result["coverage"]["eventOnlyNights"], 1)
        self.assertIsNone(result["nights"][0]["sides"]["defender"]["actualBaseDamage"])
        self.assertEqual(result["scores"]["persistenceBaseline"]["falseNegative"], 1)
        self.assertEqual(result["scores"]["persistenceBaseline"]["unscorable"], 1)

    def test_pressure_later_event_updates_both_sides_without_final(self):
        prediction = {"status": "issued", "night": 1, "sides": {
            side: {"assessment": {"risk": "unknown"},
                   "persistenceBaseline": {"risk": "elevated"}}
            for side in ("challenger", "defender")}}
        earlier = {"status": "event_observed", "night": 1,
            "observedRange": {"lastRound": 75}, "complete": False,
            "sides": {"challenger": {"actualBaseDamage": True},
                      "defender": {"actualBaseDamage": None}}}
        later = {"status": "event_observed", "night": 1,
            "observedRange": {"lastRound": 80}, "complete": False,
            "sides": {"challenger": {"actualBaseDamage": True},
                      "defender": {"actualBaseDamage": True}}}
        first = row(75, decision={"pressureShadow": {
            "prediction": prediction, "verification": earlier}})
        second = row(80, decision={"pressureShadow": {
            "prediction": prediction, "verification": later}})
        for records in ((first, second), (second, first)):
            with self.subTest(order=[item["roundNo"] for item in records]):
                result = self.run_rows(*records)["groups"][0]["pressure"]
                self.assertEqual(result["coverage"]["eventOnlyNights"], 1)
                self.assertIs(result["nights"][0]["sides"]["defender"]
                              ["actualBaseDamage"], True)
                self.assertEqual(result["scores"]["persistenceBaseline"]["hit"], 2)

    def test_three_nights_from_production_shadow_diagnostic(self):
        template = json.loads(FIXTURE.read_text(encoding="utf-8"))
        state = ShadowState()
        records = []
        for round_no in range(1, 392):
            payload = copy.deepcopy(template)
            payload["roundNo"] = round_no
            payload["teamOur"].update(type="challenger", teamId="team-a")
            our_hp = 1000 if round_no < 75 else 900 if round_no < 335 else 850
            enemy_hp = 1000 if round_no < 80 else 950 if round_no < 205 else 900
            payload["teamOur"]["roles"] = [{"id": 10013,
                "roleType": "station", "pos": {"x": 2, "y": 2},
                "health": our_hp, "level": 1}]
            payload["teamEnemy"]["roles"] = [{"id": 20013,
                "roleType": "station", "pos": {"x": 18, "y": 18},
                "health": enemy_hp, "level": 1}]
            payload["robot"] = {"roles": []}
            observe_shadow(Turn.load(payload), state, frozenset())
            records.append(turn_log_record(payload, {"roleCommandMap": {}},
                {"processing": 1}, decision_trace={"newsEvidence": {
                    "currentSession": 1},
                    "pressureShadow": shadow_diagnostic(state)}))
        result = self.run_rows(*records)["groups"][0]["pressure"]
        self.assertEqual(result["coverage"]["nights"], 4)
        self.assertEqual(result["coverage"]["finalNights"], 3)
        self.assertEqual(result["scores"]["assessment"]["unscorable"], 8)
        self.assertEqual(result["scores"]["persistenceBaseline"]["hit"], 1)
        self.assertEqual(result["scores"]["persistenceBaseline"]["falseNegative"], 1)
        self.assertEqual(result["scores"]["persistenceBaseline"]["falsePositive"], 2)

    def test_production_turn_record_task_and_pressure_fields(self):
        payload = {"roundNo": 131, "teamOur": {"type": "challenger",
            "teamId": "team-a", "roles": []}, "teamEnemy": {"roles": []}}
        trace = {"newsEvidence": {"currentSession": 1},
                 "taskInstanceId": "1:2", "taskEnd": {
                     "instanceId": "1:1", "endRound": 131,
                     "endReason": "unknown", "submissionCount": 1,
                     "associatedErrorCodes": [2], "successStatus": "unknown",
                     "attribution": "unattributed"},
                 "pressureShadow": {"session": 1, "prediction": {
                     "status": "issued", "night": 1, "sides": {
                         "challenger": {"assessment": {"risk": "unknown"},
                                        "persistenceBaseline": {"risk": "low"}},
                         "defender": {"assessment": {"risk": "unknown"},
                                      "persistenceBaseline": {"risk": "low"}}}},
                     "verification": {"status": "final", "night": 1,
                         "complete": True, "sides": {
                             "challenger": {"actualBaseDamage": False},
                             "defender": {"actualBaseDamage": True}}}}}
        record = turn_log_record(payload, {"roleCommandMap": {}},
                                 {"processing": 1}, decision_trace=trace)
        group = self.run_rows(record)["groups"][0]
        self.assertEqual(group["tasks"]["instances"][0]["instanceId"], "1:1")
        self.assertEqual(group["tasks"]["instances"][0]["end"]["associatedErrorCodes"], [2])
        self.assertEqual(group["pressure"]["scoresBySide"]["challenger"]
                         ["persistenceBaseline"]["correctNegative"], 1)

    def test_offline_pressure_whitelists_night_start_and_wall_counts(self):
        record = row(72, decision={"pressureShadow": {
            "session": 1,
            "nightSummary": {"night": 1,
                "observedRange": {"firstRound": 71, "lastRound": 72, "frames": 2},
                "firstNightRobotDifference": {
                    "status": "conditional_observation", "rawDifference": -2,
                    "inferredOpponentAdditions": -2,
                    "ourAppliedAdditions": 0,
                    "ourAppliedAdditionsBasis": "program_does_not_summon",
                    "unknowns": ["timing_unverified"],
                    "secret": "SENSITIVE_SENTINEL"},
                "sides": {"challenger": {"criticalWalls": {
                    "damagedWallFrames": 2, "nearBaseDamagedWallFrames": 1,
                    "sideDamagedWallFrames": 1,
                    "damagedWallFramesWithAdjacentRobots": 1,
                    "adjacentRobotObservations": None,
                    "adjacentRobotObservationsLowerBound": 3,
                    "longestConsecutiveDamageRounds": 2,
                    "coverageComplete": False,
                    "unknowns": ["robots_missing"]}},
                    "defender": {"criticalWalls": {}}}}}})
        result = self.run_rows(record)["groups"][0]["pressure"]["nights"][0]
        self.assertEqual(result["firstNightRobotDifference"]["rawDifference"], -2)
        self.assertEqual(result["criticalWalls"]["challenger"]
                         ["damagedWallFrames"], 2)
        self.assertIsNone(result["criticalWalls"]["challenger"]
                          ["adjacentRobotObservations"])
        self.assertEqual(result["criticalWalls"]["challenger"]
                         ["adjacentRobotObservationsLowerBound"], 3)
        self.assertFalse(result["criticalWalls"]["challenger"]
                         ["coverageComplete"])
        self.assertIn("robots_missing", result["criticalWalls"]["challenger"]
                      ["unknowns"])
        self.assertNotIn("SENSITIVE_SENTINEL", json.dumps(result))

    def test_final_only_production_record_keeps_frozen_night_observation(self):
        state = ShadowState()
        for round_no in range(70, 132):
            value = json.loads(FIXTURE.read_text(encoding="utf-8"))
            value["roundNo"] = round_no
            value["teamOur"].update(type="challenger", teamId="team-a")
            value["teamOur"]["roles"] = [{"id": 10013,
                "roleType": "station", "pos": {"x": 2, "y": 2},
                "health": 100, "level": 1}, {"id": 10014,
                "roleType": "wall", "pos": {"x": 5, "y": 5},
                "health": 90 if round_no == 131 else 100, "level": 1}]
            value["teamEnemy"]["roles"] = [{"id": 20013,
                "roleType": "station", "pos": {"x": 18, "y": 18},
                "health": 100, "level": 1}]
            value["robot"] = {"roles": ([
                {"id": 1, "roleType": "smallRobot", "pos": {"x": 3, "y": 3},
                 "health": 50, "targetTeam": "challenger"},
            ] if round_no == 71 else [{"id": 2, "roleType": "smallRobot",
                "pos": {"x": 4, "y": 5}, "health": 50,
                "targetTeam": "challenger"}] if round_no == 130 else [])}
            observe_shadow(Turn.load(value), state)
        shadow = shadow_diagnostic(state)
        self.assertIsNone(shadow["nightSummary"])
        final_payload = value
        record = turn_log_record(final_payload, {"roleCommandMap": {}},
            {"processing": 1}, decision_trace={
                "newsEvidence": {"currentSession": 1},
                "pressureShadow": shadow})
        night = self.run_rows(record)["groups"][0]["pressure"]["nights"][0]
        self.assertEqual(night["firstNightRobotDifference"]["rawDifference"], 1)
        self.assertEqual(night["firstNightRobotDifference"]["assumptions"],
                         ["equal_natural_robot_totals"])
        self.assertEqual(night["firstNightRobotDifference"]["timingStatus"],
                         "unknown")
        self.assertTrue(night["criticalWalls"]["challenger"]["dawnObserved"])
        self.assertEqual(night["criticalWalls"]["challenger"]
                         ["damagedWallFrames"], 1)
        self.assertIsNone(night["criticalWalls"]["challenger"]
                          ["adjacentRobotObservations"])
        self.assertEqual(night["criticalWalls"]["challenger"]
                         ["dawnPriorAdjacentRobotObservations"], 1)

    def test_task_and_pressure_keep_unknown_on_legacy_duplicate_and_bad_fields(self):
        legacy = row(1, decision={"taskInstanceId": "1:1"})
        legacy["commands"] = {"items": [{"action": "submitAnswer"}],
                              "truncated": True}
        bad = row(3, decision={"taskInstanceId": "1:1", "taskEnd": {
            "instanceId": "1:1", "endRound": 3,
            "endReason": ["SENSITIVE_SENTINEL"],
            "solverStoppedReason": ["SENSITIVE_SENTINEL"],
            "associatedErrorCodes": [2, "SENSITIVE_SENTINEL"]},
            "pressureShadow": {"prediction": {"status": "issued",
                "night": 1, "sides": {"challenger": {
                    "assessment": {"risk": ["SENSITIVE_SENTINEL"]}}}}}})
        output = self.run_rows(legacy, bad, copy.deepcopy(bad))
        group = output["groups"][0]
        instance = group["tasks"]["instances"][0]
        self.assertIsNone(instance["end"])
        self.assertEqual(instance["submitAnswerRequests"], 1)
        self.assertIn("commands_truncated", instance["unknowns"])
        self.assertIn("sequence_ambiguous", instance["unknowns"])
        self.assertEqual(group["pressure"]["nights"], [])
        self.assertNotIn("SENSITIVE_SENTINEL", json.dumps(output))

    def test_malformed_pressure_risk_and_unknown_session_stay_unscorable(self):
        record = row(71, decision={"taskInstanceId": "1:1",
            "pressureShadow": {"prediction": {"status": "issued", "night": 1,
                "sides": {"challenger": {
                    "assessment": {"risk": ["SENSITIVE_SENTINEL"]},
                    "persistenceBaseline": {"risk": "low"}}}},
                "verification": {"status": "event_observed", "night": 1,
                    "complete": False, "sides": {
                        "challenger": {"actualBaseDamage": True}}}}})
        record["decision"]["newsEvidence"].pop("currentSession")
        output = self.run_rows(record)
        group = output["groups"][0]
        self.assertTrue(group["sequence"]["identityUnknown"])
        self.assertEqual(group["tasks"]["instances"], [])
        self.assertIsNone(group["pressure"]["nights"][0]
                          ["sides"]["challenger"]["actualBaseDamage"])
        self.assertEqual(group["pressure"]["scores"]["assessment"]["unscorable"], 2)
        self.assertNotIn("SENSITIVE_SENTINEL", json.dumps(output))

    def test_task_instances_do_not_cross_session_or_build_groups(self):
        first = row(1, session=1, decision={"taskInstanceId": "1:1"})
        second = row(1, session=2, decision={"taskInstanceId": "2:1"})
        third = row(1, session=1, build="r-next",
                    decision={"taskInstanceId": "1:1"})
        groups = self.run_rows(first, second, third)["groups"]
        self.assertEqual(len(groups), 3)
        self.assertEqual(sorted(group["tasks"]["instances"][0]["instanceId"]
                                for group in groups), ["1:1", "1:1", "2:1"])
        self.assertTrue(all(len(group["tasks"]["instances"]) == 1
                            for group in groups))

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
        self.assertEqual(result["base"]["health"]["known"], 3)
        self.assertEqual(result["base"]["level"]["known"], 3)
        self.assertEqual(result["base"]["unknownReasons"], {})

    def test_truncated_real_record_keeps_visible_base_hp_and_level(self):
        roles = [{"id": "base-a", "roleType": "station", "health": 1480,
                  "level": 2, "pos": {"x": 1, "y": 1}}]
        roles += [{"id": f"wall-{i:02d}", "roleType": "wall",
                   "health": 100, "level": 1, "pos": {"x": i + 2, "y": 2}}
                  for i in range(20)]
        record = turn_log_record({"roundNo": 330,
            "teamOur": {"type": "challenger", "teamId": "team-a",
                        "roles": roles}, "teamEnemy": {"roles": []}},
            {"roleCommandMap": {}}, {"processing": 1, "decide": 1},
            decision_trace={"newsEvidence": {"currentSession": 1}})
        self.assertTrue(record["ourStructures"]["truncated"])
        self.assertTrue(any(item["type"] == "station"
                            for item in record["ourStructures"]["items"]))
        result = self.run_rows(record)["groups"][0]["base"]
        self.assertEqual(result["health"]["known"], 1)
        self.assertEqual(result["health"]["min"], 1480)
        self.assertEqual(result["level"]["min"], 2)

    def test_truncated_record_without_visible_base_keeps_hp_only(self):
        record = row(1, hp=1480, level=2)
        record["ourStructures"] = {"truncated": True, "items": [
            {"id": f"wall-{i:02d}", "type": "wall", "level": 1,
             "health": 100, "pos": {"x": i, "y": 2}}
            for i in range(16)]}
        result = self.run_rows(record)["groups"][0]["base"]
        self.assertEqual(result["health"]["known"], 1)
        self.assertEqual(result["level"]["known"], 0)
        self.assertEqual(result["unknownReasons"], {"base_level_unknown": 1})

    def test_truncated_night_keeps_observed_hp_drop_but_not_defense_comparison(self):
        history = {"historyInvestment": {"status": "assessed", "baselineNight": 1,
                    "defenseComparison": "approximately_comparable"}}
        records = [row(200, hp=100, decision=history)]
        records += [row(n, hp=100 - (n - 200)) for n in range(201, 262)]
        for record in records:
            record["ourStructures"]["truncated"] = True
        comparison = self.run_rows(*records)["groups"][0][
            "historyInvestment"]["comparisons"][0]
        self.assertEqual(comparison["actualObservedHpDrop"], 60)
        self.assertEqual(comparison["defenseComparison"], "unknown")
        self.assertEqual(comparison["status"], "observed")

    def test_truncated_night_missing_level_remains_unknown(self):
        history = {"historyInvestment": {"status": "assessed", "baselineNight": 1}}
        records = [row(200, hp=100, decision=history)]
        records += [row(n, hp=100 - (n - 200)) for n in range(201, 262)]
        for record in records:
            record["ourStructures"]["truncated"] = True
        records[31]["ourStructures"]["items"] = []
        comparison = self.run_rows(*records)["groups"][0][
            "historyInvestment"]["comparisons"][0]
        self.assertEqual(comparison["status"], "unknown")
        self.assertIn("base_level_unknown", comparison["unknowns"])

    def test_truncated_night_gap_heal_or_upgrade_still_unknown(self):
        history = {"historyInvestment": {"status": "assessed", "baselineNight": 1}}
        for case, reason in (("gap", "night_frames_missing"),
                             ("heal", "base_healed"),
                             ("upgrade", "base_identity_or_level_changed")):
            with self.subTest(case=case):
                records = [row(200, hp=100, decision=history)]
                records += [row(n, hp=100 - (n - 200)) for n in range(201, 262)]
                for record in records:
                    record["ourStructures"]["truncated"] = True
                if case == "gap":
                    records = [record for record in records if record["roundNo"] != 230]
                elif case == "heal":
                    records[31]["bases"]["our"]["health"] = 200
                else:
                    records[31]["ourStructures"]["items"][0]["level"] = 2
                comparison = self.run_rows(*records)["groups"][0][
                    "historyInvestment"]["comparisons"][0]
                self.assertEqual(comparison["status"], "unknown")
                self.assertIn(reason, comparison["unknowns"])

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
