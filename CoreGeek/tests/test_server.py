import io
import json
import socket
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from unittest import mock

from agent import server as server_module
from agent.brain import decide


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "s0_request.json"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.port = free_port()
        self.process = subprocess.Popen(
            ["bash", "run.sh", str(self.port)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.addCleanup(self.stop_process)
        self.url = f"http://127.0.0.1:{self.port}/"
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                output = self.process.stdout.read() if self.process.stdout else ""
                self.fail(f"server exited during startup:\n{output}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    return
            except OSError:
                time.sleep(0.02)
        self.fail("server did not listen within 3 seconds")

    def stop_process(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.process.stdout:
            self.process.stdout.close()

    def post(self, body, content_type="application/json; charset=utf-8"):
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        return urllib.request.urlopen(request, timeout=2)

    def test_run_script_serves_two_requests_with_complete_json_response(self):
        # Break caught: missing deployment entrypoint or stateful one-shot HTTP handling.
        body = FIXTURE.read_bytes()

        for _ in range(2):
            with self.post(body) as response:
                payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    response.headers.get_content_type(), "application/json"
                )
                self.assertEqual(
                    response.headers.get_content_charset(), "utf-8"
                )
                self.assertEqual(payload["prompt"], "")
                self.assertEqual(payload["executeCmd"], "")
                self.assertEqual(
                    payload["roleCommandMap"]["10010"]["targetPos"],
                    [{"x": 1, "y": 1}],
                )
                self.assertEqual(
                    payload["roleCommandMap"]["10010"]["action"], "collect"
                )

    def test_malformed_json_is_not_disguised_as_a_legal_empty_turn(self):
        # Break caught: a decoder or decision exception is swallowed as roleCommandMap={}.
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post(b"not-json")

        self.assertEqual(caught.exception.code, 400)
        body = json.loads(caught.exception.read().decode("utf-8"))
        self.assertEqual(body, {"error": "invalid request"})

    def test_turn_log_keeps_commands_and_feedback_correlated_by_role(self):
        # Break caught: per-role outcomes collapse into aggregate success/failure counts.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["lastRoundRoleActionResults"] = {
            "10011": False,
            "10010": True,
        }
        response = decide(payload)
        builder = getattr(server_module, "turn_log_record", None)
        self.assertTrue(callable(builder), "structured turn log builder is missing")

        record = builder(
            payload,
            response,
            {
                "parse": 1.0,
                "decide": 2.0,
                "encode": 3.0,
                "processing": 6.0,
                "serverWriteComplete": 7.0,
            },
        )

        self.assertEqual(record["buildId"], "nightwatch-s1-r9")
        self.assertEqual(
            record["team"], {"type": "challenger", "id": "s0-our"}
        )
        self.assertEqual(
            record["commands"],
            {
                "items": [{
                    "roleId": "10010",
                    "action": "collect",
                    "targetPos": [{"x": 1, "y": 1}],
                    "targetPosTruncated": False,
                }],
                "truncated": False,
            },
        )
        self.assertEqual(
            record["actionFeedback"],
            {
                "items": [
                    {"roleId": "10010", "ok": True},
                    {"roleId": "10011", "ok": False},
                ],
                "truncated": False,
            },
        )
        self.assertEqual(
            record["timingScope"],
            "server-side only; not judger end-to-end",
        )
        serialized = json.dumps(record, ensure_ascii=False)
        for sensitive_field in (
            "phaseTask",
            "worldNews",
            "lastCmdResult",
        ):
            self.assertNotIn(sensitive_field, serialized)

    def test_turn_log_records_utc_roles_and_both_base_states(self):
        # Break caught: platform logs cannot reconstruct positions or base survival.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["teamOur"]["roles"].append({
            "id": 10013,
            "pos": {"x": 0, "y": 4},
            "roleType": "station",
            "health": 1400,
            "attackPower": 0,
            "attackRange": 0,
            "backPackCapability": 0,
            "backpack": [],
            "level": 1,
        })
        payload["teamEnemy"]["roles"].append({
            "id": 20013,
            "pos": {"x": 4, "y": 4},
            "roleType": "station",
            "health": 1300,
            "attackPower": 0,
            "attackRange": 0,
            "level": 1,
        })
        response = decide(payload)

        record = server_module.turn_log_record(payload, response, {})

        self.assertIn("timestampUtc", record)
        timestamp = datetime.fromisoformat(record["timestampUtc"])
        self.assertIsNotNone(timestamp.tzinfo)
        self.assertEqual(timestamp.utcoffset().total_seconds(), 0)
        self.assertEqual(record["controlledRoles"], {
            "items": [{
                "id": "10010",
                "type": "worker",
                "pos": {"x": 2, "y": 2},
                "health": 220,
                "capacity": 100,
                "backpack": {},
            }],
            "truncated": False,
        })
        self.assertEqual(record["bases"], {
            "our": {
                "present": True,
                "id": "10013",
                "pos": {"x": 0, "y": 4},
                "health": 1400,
            },
            "enemy": {
                "present": True,
                "id": "20013",
                "pos": {"x": 4, "y": 4},
                "health": 1300,
            },
        })

    def test_turn_log_marks_missing_bases_without_crashing(self):
        # Break caught: station absence crashes logging or is mistaken for zero HP.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

        record = server_module.turn_log_record(payload, decide(payload), {})

        self.assertEqual(record["bases"], {
            "our": {"present": False},
            "enemy": {"present": False},
        })

    def test_turn_log_keeps_c_action_name_and_quantity(self):
        # Break caught: build/trade evidence loses the tested item and quantity.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        response = {
            "roleCommandMap": {
                "10010": {
                    "action": "buy",
                    "name": "Medicine",
                    "num": 2,
                }
            },
            "prompt": "",
            "executeCmd": "",
        }

        record = server_module.turn_log_record(payload, response, {})

        self.assertEqual(record["commands"], {
            "items": [{
                "roleId": "10010",
                "action": "buy",
                "name": "Medicine",
                "num": 2,
            }],
            "truncated": False,
        })

    def test_turn_log_exposes_task_phase_without_private_contents(self):
        # Break caught: task observability logs the prompt, command, or answer text.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["phaseTask"] = "private task text"
        response = {
            "roleCommandMap": {
                "10011": {
                    "action": "submitAnswer",
                    "taskAnswer": "private answer",
                }
            },
            "prompt": "private prompt",
            "executeCmd": "private command",
        }

        record = server_module.turn_log_record(
            payload, response, {"processing": 1.0},
        )
        encoded = json.dumps(record)

        self.assertEqual(record["task"], {
            "phase": "active",
            "active": True,
            "promptRequested": True,
            "commandRequested": True,
            "points": {"items": [], "truncated": False},
            "tool": {
                "llmResponsePresent": False,
                "envelopeKind": None,
                "answerComplete": None,
                "commandResultClass": "none",
                "commandResultTruncated": False,
                "answerSubmitted": True,
            },
        })
        for private in (
            "private task text",
            "private answer",
            "private prompt",
            "private command",
        ):
            self.assertNotIn(private, encoded)

    def test_task_detail_log_records_whitelisted_interaction_text(self):
        # Break caught: downloadable logs expose only presence flags, not semantics.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload.update({
            "phaseTask": "synthetic task body",
            "llmResp": '{"kind":"command","content":"ls input.txt"}',
            "lastCmdResult": "[exitCode:0]\nsynthetic result",
            "privateSecret": "must-not-leak",
        })
        payload["errors"] = [{"errorCode": 2, "description": "try again"}]
        response = {
            "roleCommandMap": {
                "10011": {
                    "action": "submitAnswer",
                    "taskAnswer": "synthetic answer",
                },
            },
            "prompt": "synthetic solver prompt",
            "executeCmd": "ls input.txt",
        }
        trace = {"taskInstanceId": "task-7"}

        record = server_module.task_detail_log_record(
            payload, response, decision_trace=trace,
        )

        self.assertEqual(record["event"], "task_detail")
        self.assertEqual(record["taskInstanceId"], "task-7")
        self.assertEqual(record["inputAssociation"], {
            "phaseTask": "current_active_task",
            "llmResp": "unknown_previous_request",
            "lastCmdResult": "unknown_previous_request",
            "prompt": "current_active_task",
            "executeCmd": "current_active_task",
            "submittedAnswers": "current_active_task",
        })
        self.assertEqual(record["text"]["phaseTask"]["value"], "synthetic task body")
        self.assertEqual(record["text"]["prompt"]["value"], "synthetic solver prompt")
        self.assertEqual(record["text"]["llmResp"]["value"], payload["llmResp"])
        self.assertEqual(
            record["text"]["lastCmdResult"]["value"], payload["lastCmdResult"],
        )
        self.assertEqual(record["text"]["executeCmd"]["value"], "ls input.txt")
        self.assertEqual(record["submittedAnswers"]["items"][0]["text"]["value"], "synthetic answer")
        encoded = json.dumps(record, ensure_ascii=False)
        self.assertNotIn("must-not-leak", encoded)
        self.assertNotIn("privateSecret", encoded)

    def test_task_detail_log_marks_own_and_platform_truncation(self):
        # Break caught: downloaded evidence silently loses the tail of large fields.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["phaseTask"] = "x" * (server_module.MAX_TASK_DETAIL_CHARS + 7)
        payload["lastCmdResult"] = "[exitCode:0]\nvalue\n[TRUNCATED]"

        record = server_module.task_detail_log_record(
            payload,
            {"roleCommandMap": {}, "prompt": "", "executeCmd": ""},
        )

        phase = record["text"]["phaseTask"]
        self.assertEqual(len(phase["value"]), server_module.MAX_TASK_DETAIL_CHARS)
        self.assertEqual(phase["originalLength"], server_module.MAX_TASK_DETAIL_CHARS + 7)
        self.assertTrue(phase["truncated"])
        self.assertTrue(record["text"]["lastCmdResult"]["platformTruncated"])

    def test_turn_log_records_bounded_decision_trace_without_private_text(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["phaseTask"] = "private task text"
        payload["lastCmdResult"] = "private command output"
        response = {
            "roleCommandMap": {},
            "prompt": "private prompt",
            "executeCmd": "private command",
        }
        trace = {
            "roundNo": payload["roundNo"],
            "taskInstanceId": "task-7",
            "taskRemainingRounds": 4,
            "solverState": "solving",
            "solverReason": "awaiting_result",
            "leaveReason": None,
            "commandFingerprint": "a1b2c3d4e5f6",
            "resultFingerprint": "0f1e2d3c4b5a",
            "cycleFingerprint": "112233445566",
            "coordinationReason": "task_active",
            "actions": [{
                "roleId": "10010",
                "domain": "tasks",
                "reason": "task:task-7",
                "estimatedRounds": 1,
                "deadlineRound": 9,
            }],
        }

        record = server_module.turn_log_record(
            payload, response, {}, decision_trace=trace,
        )
        encoded = json.dumps(record, ensure_ascii=False)

        self.assertEqual(record["decision"], trace)
        self.assertEqual(set(response), {"roleCommandMap", "prompt", "executeCmd"})
        for private in (
            "private task text",
            "private command output",
            "private prompt",
            "private command",
        ):
            self.assertNotIn(private, encoded)

    def test_turn_log_has_bounded_s1_evidence_without_sensitive_text(self):
        # Break caught: intranet cannot verify economy, defense, or task tool shape.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["teamOur"].update({
            "goldNum": 37,
            "totalScore": 81,
            "playerTasks": [
                {
                    "taskType": f"自进化类{index}",
                    "taskPosition": {"x": index, "y": 4},
                    "coldDownRounds": index,
                    "scoreReward": 50,
                    "goldReward": 30,
                    "isValid": True,
                    "timeoutRounds": 20,
                }
                for index in range(server_module.MAX_LOG_ITEMS + 1)
            ],
        })
        payload["teamOur"]["roles"][0].update({
            "backPackCapability": 8,
            "backpack": ["stone", "stone", "Medicine", "private-item"],
        })
        payload["teamOur"]["roles"].extend([
            {
                "id": 10013,
                "pos": {"x": 0, "y": 4},
                "roleType": "station",
                "health": 1400,
                "level": 2,
                "cooldown": 0,
            },
            {
                "id": 10020,
                "pos": {"x": 2, "y": 4},
                "roleType": "gatling",
                "health": 900,
                "level": 1,
                "cooldown": 2,
            },
            {
                "id": 10030,
                "pos": {"x": 3, "y": 4},
                "roleType": "wall",
                "health": 700,
                "level": 1,
                "cooldown": 0,
            },
        ])
        payload["robot"]["roles"] = [
            {
                "id": 30000 + index,
                "pos": {"x": index, "y": 8},
                "roleType": "smallRobot",
                "health": 40,
                "attackPower": 5,
                "abnormalState": "Dizzy" if index == 0 else "",
                "targetTeam": "challenger",
            }
            for index in range(server_module.MAX_LOG_ITEMS + 1)
        ]
        payload["vendorShopList"] = [
            {"name": "stone", "price": 4},
            {"name": "private-good", "price": 999},
        ]
        payload["weaponShopList"] = [
            {"name": "Medicine", "price": 10},
            {"name": "WeaponUpgradeVoucher1", "price": 20},
            {"name": "private-tool", "price": 999},
        ]
        payload["phaseTask"] = "private task text"
        payload["llmResp"] = (
            '{"kind":"command","content":"private command"}'
        )
        payload["lastCmdResult"] = "[exitCode:0]\nprivate output"
        response = {
            "roleCommandMap": {
                "10011": {
                    "action": "submitAnswer",
                    "taskAnswer": "private answer",
                }
            },
            "prompt": "private prompt",
            "executeCmd": "private execute",
        }

        record = server_module.turn_log_record(payload, response, {})
        encoded = json.dumps(record, ensure_ascii=False)

        self.assertIn("economy", record)
        self.assertEqual(record["economy"], {"gold": 37, "score": 81})
        worker = record["controlledRoles"]["items"][0]
        self.assertEqual(worker["capacity"], 8)
        self.assertEqual(worker["backpack"], {"Medicine": 1, "stone": 2})
        self.assertFalse(record["controlledRoles"]["truncated"])
        self.assertEqual(
            {item["type"] for item in record["ourStructures"]["items"]},
            {"station", "gatling", "wall"},
        )
        tower = next(
            item for item in record["ourStructures"]["items"]
            if item["type"] == "gatling"
        )
        self.assertEqual(tower, {
            "id": "10020",
            "type": "gatling",
            "pos": {"x": 2, "y": 4},
            "health": 900,
            "level": 1,
            "cooldown": 2,
        })
        self.assertTrue(record["robots"]["truncated"])
        self.assertEqual(record["robots"]["items"][0]["targetTeam"], "challenger")
        self.assertEqual(record["shops"], {
            "vendor": {"stone": 4},
            "weapon": {"Medicine": 10, "WeaponUpgradeVoucher1": 20},
        })
        self.assertTrue(record["task"]["points"]["truncated"])
        self.assertEqual(record["task"]["points"]["items"][0], {
            "type": "自进化类0",
            "pos": {"x": 0, "y": 4},
            "cooldownRounds": 0,
            "scoreReward": 50,
            "goldReward": 30,
            "valid": True,
            "timeoutRounds": 20,
        })
        self.assertEqual(record["task"]["tool"], {
            "llmResponsePresent": True,
            "envelopeKind": "command",
            "answerComplete": None,
            "commandResultClass": "success",
            "commandResultTruncated": False,
            "answerSubmitted": True,
        })
        answer_payload = dict(payload)
        answer_payload["llmResp"] = (
            '{"kind":"answer","content":"private llm answer",'
            '"complete":false}'
        )
        answer_record = server_module.turn_log_record(
            answer_payload, response, {},
        )
        self.assertEqual(answer_record["task"]["tool"]["envelopeKind"], "answer")
        self.assertFalse(answer_record["task"]["tool"]["answerComplete"])
        self.assertNotIn(
            "private llm answer",
            json.dumps(answer_record, ensure_ascii=False),
        )
        for raw, expected in (
            ("", "none"),
            ("[exitCode:1]\nprivate", "failed"),
            ("[TIMEOUT]\nprivate", "timeout"),
            ("[JUDGER_ERROR]\nprivate", "judger_error"),
            ("other private output", "unknown"),
            ("[exitCode:not-an-int]\nprivate", "unknown"),
            ("[exitCode:0]\nprivate\n[TRUNCATED]", "truncated"),
        ):
            self.assertEqual(server_module._command_result_class(raw), expected)
        for private in (
            "private-item",
            "private-good",
            "private-tool",
            "private task text",
            "private command",
            "private output",
            "private answer",
            "private prompt",
            "private execute",
        ):
            self.assertNotIn(private, encoded)

    def test_slow_turn_log_runs_after_response_is_sent(self):
        # Break caught: synchronous log output delays the response write.
        body = FIXTURE.read_bytes()
        handler = object.__new__(server_module.Handler)
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        events = []
        records = []

        def send_body(status, response_body):
            events.append(("send", time.monotonic()))
            self.assertEqual(status, 200)
            self.assertTrue(response_body)

        def slow_log(*args, **kwargs):
            events.append(("log_start", time.monotonic()))
            records.append(json.loads(args[1]))
            time.sleep(0.05)
            events.append(("log_end", time.monotonic()))

        handler._send_body = send_body
        with mock.patch.object(server_module.LOGGER, "info", side_effect=slow_log):
            server_module.Handler.do_POST(handler)

        self.assertEqual([event for event, _ in events], [
            "send",
            "log_start",
            "log_end",
        ])
        self.assertGreaterEqual(
            records[0]["timingMs"]["serverWriteComplete"],
            records[0]["timingMs"]["processing"],
        )

    def test_malformed_request_shape_is_reported_as_client_error(self):
        # Break caught: wrong JSON container types escape validation as server faults.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["teamOur"]["roles"][0] = []

        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post(json.dumps(payload).encode("utf-8"))

        self.assertEqual(caught.exception.code, 400)
        body = json.loads(caught.exception.read().decode("utf-8"))
        self.assertEqual(body, {"error": "invalid request"})


if __name__ == "__main__":
    unittest.main()
