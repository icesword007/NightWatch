import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .brain import decide

LOGGER = logging.getLogger(__name__)
BUILD_ID = "nightwatch-s0-a2"
MAX_LOG_ITEMS = 16
MAX_LOG_TARGETS = 3


def turn_log_record(
    payload: dict[str, Any],
    response: dict[str, Any],
    timing_ms: dict[str, float],
) -> dict[str, Any]:
    team = payload.get("teamOur")
    if not isinstance(team, dict):
        team = {}

    raw_commands = response.get("roleCommandMap")
    if not isinstance(raw_commands, dict):
        raw_commands = {}
    commands = []
    for role_id, command in sorted(
        raw_commands.items(), key=lambda item: str(item[0]),
    )[:MAX_LOG_ITEMS]:
        if not isinstance(command, dict):
            continue
        summary = {
            "roleId": str(role_id),
            "action": command.get("action"),
        }
        controller_id = command.get("controllerId")
        if controller_id is not None:
            summary["controllerId"] = str(controller_id)
        targets = command.get("targetPos")
        if isinstance(targets, list):
            summary["targetPos"] = targets[:MAX_LOG_TARGETS]
        commands.append(summary)

    raw_feedback = payload.get("lastRoundRoleActionResults")
    if not isinstance(raw_feedback, dict):
        raw_feedback = {}
    feedback = [
        {"roleId": str(role_id), "ok": result}
        for role_id, result in sorted(
            raw_feedback.items(), key=lambda item: str(item[0]),
        )[:MAX_LOG_ITEMS]
        if isinstance(result, bool)
    ]

    raw_errors = payload.get("errors")
    if not isinstance(raw_errors, list):
        raw_errors = []
    error_codes = [
        error.get("errorCode")
        for error in raw_errors[:MAX_LOG_ITEMS]
        if isinstance(error, dict) and "errorCode" in error
    ]

    return {
        "event": "turn",
        "buildId": BUILD_ID,
        "roundNo": payload.get("roundNo"),
        "team": {"type": team.get("type"), "id": team.get("teamId")},
        "commandCount": len(raw_commands),
        "commands": commands,
        "feedbackCount": len(raw_feedback),
        "actionFeedback": feedback,
        "errorCount": len(raw_errors),
        "errorCodes": error_codes,
        "timingMs": timing_ms,
        "timingScope": "server-side only; not judger end-to-end",
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        started = time.monotonic()
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise ValueError("request body is empty")
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            parsed = time.monotonic()
            response = decide(payload)
            decided = time.monotonic()
            body = json.dumps(
                response, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
            encoded = time.monotonic()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            self._send_json(400, {"error": "invalid request"})
            LOGGER.warning("invalid request buildId=%s", BUILD_ID)
            return
        except Exception:
            self._send_json(500, {"error": "internal error"})
            LOGGER.exception("decision failed buildId=%s", BUILD_ID)
            return

        processing_ms = {
            "parse": (parsed - started) * 1000,
            "decide": (decided - parsed) * 1000,
            "encode": (encoded - decided) * 1000,
            "processing": (encoded - started) * 1000,
        }
        self._send_body(200, body)
        sent = time.monotonic()
        processing_ms["serverWriteComplete"] = (sent - started) * 1000
        record = turn_log_record(payload, response, processing_ms)
        LOGGER.info(
            "%s", json.dumps(record, ensure_ascii=False, separators=(",", ":")),
        )

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        self._send_body(status, body)

    def _send_body(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(port: int) -> None:
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
