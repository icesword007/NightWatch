import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .brain import decide
from .tasks import parse_llm_envelope

LOGGER = logging.getLogger(__name__)
BUILD_ID = "nightwatch-s1-r7"
MAX_LOG_ITEMS = 16
MAX_LOG_TARGETS = 3
LOG_INVENTORY_ITEMS = (
    "stone",
    "iron",
    "copper",
    "Medicine",
    "WallFixer",
    "StationUpgradeVoucher1",
    "StationUpgradeVoucher2",
    "WeaponUpgradeVoucher1",
    "WeaponUpgradeVoucher2",
    "WallUpgradeVoucher1",
    "WallUpgradeVoucher2",
)
LOG_STRUCTURE_TYPES = frozenset(("station", "gatling", "railgun", "rocket", "wall"))


def turn_log_record(
    payload: dict[str, Any],
    response: dict[str, Any],
    timing_ms: dict[str, float],
    *,
    decision_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    team = payload.get("teamOur")
    if not isinstance(team, dict):
        team = {}
    enemy_team = payload.get("teamEnemy")
    if not isinstance(enemy_team, dict):
        enemy_team = {}

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
        if isinstance(command.get("name"), str):
            summary["name"] = command["name"]
        number = command.get("num")
        if isinstance(number, int) and not isinstance(number, bool):
            summary["num"] = number
        targets = command.get("targetPos")
        if isinstance(targets, list):
            summary["targetPos"] = [
                _position(target) for target in targets[:MAX_LOG_TARGETS]
            ]
            summary["targetPosTruncated"] = len(targets) > MAX_LOG_TARGETS
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
        "timestampUtc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "roundNo": payload.get("roundNo"),
        "team": {"type": team.get("type"), "id": team.get("teamId")},
        "economy": {
            "gold": team.get("goldNum"),
            "score": team.get("totalScore"),
        },
        "controlledRoles": _controlled_role_states(team),
        "ourStructures": _structure_states(team),
        "robots": _robot_states(payload),
        "shops": _shop_prices(payload),
        "bases": {
            "our": _base_state(team),
            "enemy": _base_state(enemy_team),
        },
        "commandCount": len(raw_commands),
        "commands": _limited(commands, len(raw_commands)),
        "feedbackCount": len(raw_feedback),
        "actionFeedback": _limited(feedback, len(raw_feedback)),
        "errorCount": len(raw_errors),
        "errorCodes": _limited(error_codes, len(raw_errors)),
        "task": {
            "phase": "active" if payload.get("phaseTask") else "idle",
            "active": bool(payload.get("phaseTask")),
            "promptRequested": bool(response.get("prompt")),
            "commandRequested": bool(response.get("executeCmd")),
            "points": _task_points(team),
            "tool": _task_tool_shape(payload, response),
        },
        "decision": decision_trace,
        "timingMs": timing_ms,
        "timingScope": "server-side only; not judger end-to-end",
    }


def _controlled_role_states(team: dict[str, Any]) -> dict[str, Any]:
    roles = team.get("roles")
    if not isinstance(roles, list):
        return _limited([], 0)
    result = []
    for role in sorted(
        (role for role in roles if isinstance(role, dict)),
        key=lambda role: str(role.get("id")),
    ):
        if role.get("roleType") not in ("worker", "pioneer"):
            continue
        result.append({
            "id": _short(role.get("id")),
            "type": role.get("roleType"),
            "pos": _position(role.get("pos")),
            "health": role.get("health"),
            "capacity": role.get("backPackCapability"),
            "backpack": _inventory_counts(role.get("backpack")),
        })
        if len(result) == MAX_LOG_ITEMS:
            break
    total = sum(
        isinstance(role, dict)
        and role.get("roleType") in ("worker", "pioneer")
        for role in roles
    )
    return _limited(result, total)


def _structure_states(team: dict[str, Any]) -> dict[str, Any]:
    roles = team.get("roles")
    if not isinstance(roles, list):
        return _limited([], 0)
    matching = [
        role for role in roles
        if isinstance(role, dict) and role.get("roleType") in LOG_STRUCTURE_TYPES
    ]
    matching.sort(key=lambda role: _short(role.get("id")))
    items = [{
        "id": _short(role.get("id")),
        "type": role.get("roleType"),
        "pos": _position(role.get("pos")),
        "health": role.get("health"),
        "level": role.get("level"),
        "cooldown": role.get("cooldown"),
    } for role in matching[:MAX_LOG_ITEMS]]
    return _limited(items, len(matching))


def _robot_states(payload: dict[str, Any]) -> dict[str, Any]:
    robot_section = payload.get("robot")
    roles = robot_section.get("roles") if isinstance(robot_section, dict) else None
    if not isinstance(roles, list):
        return _limited([], 0)
    matching = [role for role in roles if isinstance(role, dict)]
    matching.sort(key=lambda role: _short(role.get("id")))
    items = [{
        "id": _short(role.get("id")),
        "type": _short(role.get("roleType")),
        "pos": _position(role.get("pos")),
        "health": role.get("health"),
        "attackPower": role.get("attackPower"),
        "abnormalState": _short(role.get("abnormalState")),
        "targetTeam": _short(role.get("targetTeam")),
    } for role in matching[:MAX_LOG_ITEMS]]
    return _limited(items, len(matching))


def _shop_prices(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    allowed = frozenset(LOG_INVENTORY_ITEMS)
    return {
        "vendor": _prices(payload.get("vendorShopList"), frozenset((
            "stone", "iron", "copper",
        ))),
        "weapon": _prices(payload.get("weaponShopList"), allowed),
    }


def _prices(raw: Any, allowed: frozenset[str]) -> dict[str, Any]:
    if not isinstance(raw, list):
        return {}
    result = {}
    for item in raw:
        if not isinstance(item, dict) or item.get("name") not in allowed:
            continue
        if isinstance(item.get("price"), int) and not isinstance(item["price"], bool):
            result[item["name"]] = item["price"]
    return dict(sorted(result.items()))


def _task_points(team: dict[str, Any]) -> dict[str, Any]:
    raw = team.get("playerTasks")
    if not isinstance(raw, list):
        return _limited([], 0)
    tasks = [task for task in raw if isinstance(task, dict)]
    items = [{
        "type": _short(task.get("taskType")),
        "pos": _position(task.get("taskPosition")),
        "cooldownRounds": task.get("coldDownRounds"),
        "scoreReward": task.get("scoreReward"),
        "goldReward": task.get("goldReward"),
        "valid": task.get("isValid"),
        "timeoutRounds": task.get("timeoutRounds"),
    } for task in tasks[:MAX_LOG_ITEMS]]
    return _limited(items, len(tasks))


def _task_tool_shape(
    payload: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    raw_llm = payload.get("llmResp")
    raw_llm = raw_llm if isinstance(raw_llm, str) else ""
    envelope = parse_llm_envelope(raw_llm)
    raw_result = payload.get("lastCmdResult")
    raw_result = raw_result if isinstance(raw_result, str) else ""
    commands = response.get("roleCommandMap")
    commands = commands if isinstance(commands, dict) else {}
    answer_submitted = any(
        isinstance(command, dict)
        and command.get("action") == "submitAnswer"
        and bool(command.get("taskAnswer"))
        for command in commands.values()
    )
    return {
        "llmResponsePresent": bool(raw_llm),
        "envelopeKind": envelope.kind if envelope is not None else None,
        "answerComplete": (
            envelope.complete
            if envelope is not None and envelope.kind == "answer"
            else None
        ),
        "commandResultClass": _command_result_class(raw_result),
        "commandResultTruncated": "[TRUNCATED]" in raw_result,
        "answerSubmitted": answer_submitted,
    }


def _command_result_class(result: str) -> str:
    if not result:
        return "none"
    if "[TRUNCATED]" in result:
        return "truncated"
    first_line = result.splitlines()[0]
    if first_line == "[TIMEOUT]":
        return "timeout"
    if first_line == "[JUDGER_ERROR]":
        return "judger_error"
    if first_line == "[exitCode:0]":
        return "success"
    if first_line.startswith("[exitCode:") and first_line.endswith("]"):
        try:
            exit_code = int(first_line[len("[exitCode:"):-1])
        except ValueError:
            return "unknown"
        return "success" if exit_code == 0 else "failed"
    return "unknown"


def _inventory_counts(raw: Any) -> dict[str, int]:
    if not isinstance(raw, list):
        return {}
    counts = Counter(item for item in raw if item in LOG_INVENTORY_ITEMS)
    return {name: counts[name] for name in sorted(counts)}


def _limited(items: list[Any], total: int) -> dict[str, Any]:
    return {"items": items[:MAX_LOG_ITEMS], "truncated": total > MAX_LOG_ITEMS}


def _position(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    return {"x": raw.get("x"), "y": raw.get("y")}


def _short(value: Any, limit: int = 64) -> str:
    text = str(value) if value is not None else ""
    return text if len(text) <= limit else f"{text[:limit - 11]}[TRUNCATED]"


def _base_state(team: dict[str, Any]) -> dict[str, Any]:
    roles = team.get("roles")
    if isinstance(roles, list):
        for role in roles:
            if isinstance(role, dict) and role.get("roleType") == "station":
                return {
                    "present": True,
                    "id": _short(role.get("id")),
                    "pos": _position(role.get("pos")),
                    "health": role.get("health"),
                }
    return {"present": False}


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
            decision_traces = []
            response = decide(payload, trace_sink=decision_traces.append)
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
        decision_trace = decision_traces[0] if decision_traces else None
        record = turn_log_record(
            payload,
            response,
            processing_ms,
            decision_trace=decision_trace,
        )
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
