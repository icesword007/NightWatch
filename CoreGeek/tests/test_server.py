import io
import json
import socket
import subprocess
import time
import unittest
import urllib.error
import urllib.request
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
                    [{"x": 2, "y": 1}],
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

        self.assertEqual(record["buildId"], "nightwatch-s0-a2")
        self.assertEqual(
            record["team"], {"type": "challenger", "id": "s0-our"}
        )
        self.assertEqual(
            record["commands"],
            [
                {
                    "roleId": "10010",
                    "action": "move",
                    "targetPos": [{"x": 2, "y": 1}],
                }
            ],
        )
        self.assertEqual(
            record["actionFeedback"],
            [
                {"roleId": "10010", "ok": True},
                {"roleId": "10011", "ok": False},
            ],
        )
        self.assertEqual(
            record["timingScope"],
            "server-side only; not judger end-to-end",
        )
        serialized = json.dumps(record, ensure_ascii=False)
        for sensitive_field in (
            "phaseTask",
            "llmResp",
            "worldNews",
            "lastCmdResult",
        ):
            self.assertNotIn(sensitive_field, serialized)

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
