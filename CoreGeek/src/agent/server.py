import hashlib
import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

from .brain import decide
from .tasks import parse_llm_envelope

LOGGER = logging.getLogger(__name__)
BUILD_ID = "nightwatch-s2-integrated-r20"
MAX_TASK_DETAIL_CHARS = 131_072
MAX_NEWS_DETAIL_CHARS = 4_096
MAX_LOG_ITEMS = 16
MAX_LOG_TARGETS = 3

# --- Task diagnostic logging constants ---
MAX_PROMPT_CHARS = 32_768
MAX_LLM_RESP_CHARS = 20_000
MAX_CMD_RESULT_CHARS = 32_768
MAX_EXECUTE_CMD_CHARS = 4_096
MAX_ANSWER_CHARS = 16_384
MAX_PHASE_TASK_CHARS = 32_768

# Task economy snapshot cache — per taskInstanceId, records economy state
# when the task first appears, used to compute scoreDelta / goldDelta on end.
_task_economy_snapshots: dict[tuple[Any, ...], dict[str, Any]] = {}
_task_economy_lock = Lock()
MAX_TASK_ECONOMY_SNAPSHOTS = 32
LOG_INVENTORY_ITEMS = (
    "stone",
    "iron",
    "copper",
    "Medicine",
    "WallFixer",
    "Bomb",
    "DizzyWeapon",
    "StationUpgradeVoucher1",
    "StationUpgradeVoucher2",
    "WeaponUpgradeVoucher1",
    "WeaponUpgradeVoucher2",
    "WallUpgradeVoucher1",
    "WallUpgradeVoucher2",
)
LOG_STRUCTURE_TYPES = frozenset(("station", "gatling", "railgun", "rocket", "wall"))


def _trace_session_index(trace: dict[str, Any]) -> Any:
    evidence = trace.get("newsEvidence")
    return evidence.get("currentSession") if isinstance(evidence, dict) else None


def _valid_task_tool_inputs(trace: dict[str, Any], round_no: Any) -> list[dict[str, Any]]:
    values = trace.get("taskToolInputs")
    if not isinstance(values, list):
        return []
    return [
        value for value in values[:MAX_LOG_ITEMS]
        if isinstance(value, dict)
        and value.get("kind") in ("llm", "cmd")
        and value.get("receivedRound") == round_no
    ]


def task_log_record(
    payload: dict[str, Any],
    response: dict[str, Any],
    *,
    decision_trace: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a task-focused diagnostic log record.

    Returns None if no task-relevant activity occurred this round.
    Includes endedTaskInfo, answerResult, scoreDelta, and solverInternals
    for task debugging and post-game analysis.
    """
    trace = decision_trace if isinstance(decision_trace, dict) else {}
    team = payload.get("teamOur")
    team = team if isinstance(team, dict) else {}
    round_no = payload.get("roundNo")

    task_instance_id = trace.get("taskInstanceId")
    ended_task_info = trace.get("taskEnd")
    if not isinstance(ended_task_info, dict):
        ended_task_info = trace.get("endedTaskInfo")
    if not isinstance(ended_task_info, dict):
        ended_task_info = None
    phase_task = payload.get("phaseTask")
    phase_task = phase_task if isinstance(phase_task, str) else ""
    has_phase_task = bool(phase_task)
    solver_state = trace.get("solverState", "idle")
    solver_reason = trace.get("solverReason")
    task_start_skip = trace.get("taskStartSkipReason")
    coordination_reason = trace.get("coordinationReason")

    commands = response.get("roleCommandMap")
    commands = commands if isinstance(commands, dict) else {}
    task_actions = _extract_task_commands(commands)
    prompt_sent = response.get("prompt")
    prompt_sent = prompt_sent if isinstance(prompt_sent, str) else ""
    execute_cmd = response.get("executeCmd")
    execute_cmd = execute_cmd if isinstance(execute_cmd, str) else ""

    llm_resp = payload.get("llmResp")
    llm_resp = llm_resp if isinstance(llm_resp, str) else ""
    cmd_result = payload.get("lastCmdResult")
    cmd_result = cmd_result if isinstance(cmd_result, str) else ""

    raw_feedback = payload.get("lastRoundRoleActionResults")
    raw_feedback = raw_feedback if isinstance(raw_feedback, dict) else {}
    raw_errors = payload.get("errors")
    raw_errors = raw_errors if isinstance(raw_errors, list) else []
    task_role_ids = _task_related_role_ids(commands, trace, team)

    task_points = _task_point_states(team)

    task_prompt = bool(prompt_sent) and (
        has_phase_task or task_instance_id is not None
    )
    task_cmd = bool(execute_cmd) and (
        has_phase_task or task_instance_id is not None
    )
    is_task_relevant = (
        task_instance_id is not None
        or ended_task_info is not None
        or has_phase_task
        or task_actions
        or task_prompt
        or task_cmd
        or task_start_skip is not None
        or solver_state not in ("idle",)
        or (
            coordination_reason is not None
            and coordination_reason != "normal"
        )
        or _has_task_accept(commands)
    )
    if not is_task_relevant:
        return None

    # Filter out idle move-only rounds (noise reduction)
    if (
        task_instance_id is None
        and not has_phase_task
        and solver_state == "idle"
        and task_start_skip is None
        and not _has_task_accept(commands)
        and not task_prompt
        and not task_cmd
        and task_actions
        and all(a["action"] == "move" for a in task_actions)
    ):
        return None

    record: dict[str, Any] = {
        "event": "task",
        "buildId": BUILD_ID,
        "timestampUtc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds",
        ),
        "roundNo": round_no,
        "team": {
            "type": team.get("type"),
            "id": team.get("teamId"),
        },
        "sessionIndex": _trace_session_index(trace),
        "taskInstanceId": task_instance_id,
        "solverState": solver_state,
        "solverReason": solver_reason,
        "taskRemainingRounds": trace.get("taskRemainingRounds"),
        "leaveReason": trace.get("leaveReason"),
        "coordinationReason": coordination_reason,
        "taskStartSkipReason": task_start_skip,
        "taskPoints": task_points,
    }

    if ended_task_info is not None:
        record["endedTaskInfo"] = ended_task_info

    # Solver internal state from trace
    solver_internals = {
        "taskEntryReadAttempted": trace.get("taskEntryReadAttempted"),
        "finalOnlyCorrectionRequested": trace.get(
            "taskFinalOnlyCorrectionRequested",
        ),
        "repeatedCommandCorrectionRequested": trace.get(
            "taskRepeatedCommandCorrectionRequested",
        ),
        "envelopeCorrectionRequested": trace.get(
            "taskEnvelopeCorrectionRequested",
        ),
        "envelopeCorrectionPending": trace.get(
            "taskEnvelopeCorrectionPending",
        ),
        "envelopeRejectionClass": trace.get("taskEnvelopeRejectionClass"),
        "consecutiveNonzeroFailures": trace.get(
            "taskConsecutiveNonzeroFailures",
        ),
        "cycleFingerprint": trace.get("cycleFingerprint"),
        "commandFingerprint": trace.get("commandFingerprint"),
        "resultFingerprint": trace.get("resultFingerprint"),
        "toolInputs": _valid_task_tool_inputs(trace, round_no),
    }
    solver_internals = {
        key: value for key, value in solver_internals.items()
        if value is not None
    }
    if solver_internals:
        record["solverInternals"] = solver_internals

    # Input from judger
    input_section: dict[str, Any] = {}
    if has_phase_task:
        input_section["phaseTask"] = _bounded_text(
            phase_task, MAX_PHASE_TASK_CHARS,
        )
    if llm_resp:
        envelope = parse_llm_envelope(llm_resp)
        input_section["llmResp"] = _bounded_text(llm_resp, MAX_LLM_RESP_CHARS)
        if envelope is not None:
            input_section["llmEnvelope"] = {
                "kind": envelope.kind,
                "content": _bounded_text(
                    envelope.content, MAX_ANSWER_CHARS,
                ),
                "complete": envelope.complete,
            }
        elif llm_resp:
            input_section["llmEnvelopeParseFailed"] = True
    if cmd_result:
        input_section["cmdResult"] = {
            "class": _command_result_class(cmd_result),
            "truncated": "[TRUNCATED]" in cmd_result,
            **_bounded_text(cmd_result, MAX_CMD_RESULT_CHARS),
        }
    if raw_errors:
        input_section["errors"] = [
            {
                "code": error.get("errorCode"),
                "description": _bounded_text(
                    error.get("description"), 1024,
                )["value"],
            }
            for error in raw_errors[:16]
            if isinstance(error, dict)
        ]
        associated_codes = (
            ended_task_info.get("associatedErrorCodes")
            if ended_task_info is not None else None
        )
        answer_result = _classify_answer_result(
            associated_codes if isinstance(associated_codes, list) else [],
        )
        association = "ended_task"
        if answer_result is None and any(
            result is False and str(role_id) in task_role_ids
            for role_id, result in raw_feedback.items()
        ):
            answer_result = _classify_answer_result([
                error for error in raw_errors
                if isinstance(error, dict)
                and str(error.get("errorCode")) in ("1", "2")
            ])
            association = "current_task_action_feedback"
        if answer_result is not None:
            answer_result["association"] = association
            input_section["answerResult"] = answer_result
    if input_section:
        record["input"] = input_section

    # Output to judger
    output_section: dict[str, Any] = {}
    if prompt_sent:
        output_section["prompt"] = _bounded_text(
            prompt_sent, MAX_PROMPT_CHARS,
        )
    if execute_cmd:
        output_section["executeCmd"] = _bounded_text(
            execute_cmd, MAX_EXECUTE_CMD_CHARS,
        )
    if task_actions:
        output_section["taskActions"] = task_actions
    if output_section:
        record["output"] = output_section

    # Action feedback
    relevant_feedback = {
        str(role_id): result
        for role_id, result in raw_feedback.items()
        if str(role_id) in task_role_ids
    }
    if relevant_feedback:
        record["actionFeedback"] = {
            str(role_id): result
            for role_id, result in relevant_feedback.items()
        }

    # Economy context with scoreDelta attribution
    current_gold = team.get("goldNum")
    current_score = team.get("totalScore")
    record["economy"] = {
        "gold": current_gold,
        "score": current_score,
    }

    session_index = _trace_session_index(trace)
    team_key = (team.get("type"), team.get("teamId"), session_index)
    snapshot = None
    with _task_economy_lock:
        if task_instance_id is not None and solver_state != "ended":
            key = (*team_key, task_instance_id)
            if key not in _task_economy_snapshots:
                _task_economy_snapshots[key] = {
                    "startRound": round_no,
                    "gold": current_gold,
                    "score": current_score,
                }
                while len(_task_economy_snapshots) > MAX_TASK_ECONOMY_SNAPSHOTS:
                    _task_economy_snapshots.pop(next(iter(_task_economy_snapshots)))
        if ended_task_info is not None:
            ended_id = ended_task_info.get("instanceId") or task_instance_id
            if ended_id is not None:
                snapshot = _task_economy_snapshots.pop((*team_key, ended_id), None)
    if ended_task_info is not None:
        if snapshot is not None:
            record["scoreDelta"] = {
                "startRound": snapshot["startRound"],
                "endRound": round_no,
                "goldStart": snapshot["gold"],
                "goldEnd": current_gold,
                "goldDelta": (
                    current_gold - snapshot["gold"]
                    if isinstance(current_gold, int)
                    and isinstance(snapshot["gold"], int)
                    else None
                ),
                "scoreStart": snapshot["score"],
                "scoreEnd": current_score,
                "scoreDelta": (
                    current_score - snapshot["score"]
                    if isinstance(current_score, int)
                    and isinstance(snapshot["score"], int)
                    else None
                ),
                "attribution": "team_observation",
            }

    legacy = task_detail_log_record(
        payload, response, decision_trace=trace,
    )
    if legacy is not None:
        record["requestFingerprint"] = legacy.get("requestFingerprint")
        record["inputAssociation"] = legacy.get("inputAssociation")
        record["acceptedToolInputs"] = legacy.get("acceptedToolInputs", [])

    record["pioneer"] = _pioneer_state(team)

    return record


def _extract_task_commands(
    commands: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract task-related commands from the response."""
    result = []
    for role_id, command in sorted(
        commands.items(), key=lambda item: str(item[0]),
    ):
        if not isinstance(command, dict):
            continue
        action = command.get("action")
        if action not in ("acceptTask", "submitAnswer", "move"):
            continue
        entry: dict[str, Any] = {
            "roleId": str(role_id),
            "action": action,
        }
        if action == "submitAnswer":
            entry["taskAnswer"] = _bounded_text(
                command.get("taskAnswer"), MAX_ANSWER_CHARS,
            )
        targets = command.get("targetPos")
        if isinstance(targets, list) and targets:
            entry["targetPos"] = {
                "x": targets[0].get("x") if isinstance(targets[0], dict) else None,
                "y": targets[0].get("y") if isinstance(targets[0], dict) else None,
            }
        result.append(entry)
    return result


def _has_task_accept(commands: dict[str, Any]) -> bool:
    return any(
        isinstance(cmd, dict) and cmd.get("action") == "acceptTask"
        for cmd in commands.values()
    )


def _task_related_role_ids(
    commands: dict[str, Any],
    trace: dict[str, Any],
    team: dict[str, Any],
) -> set[str]:
    """Identify role IDs involved in task actions."""
    ids = set()
    roles = team.get("roles")
    if isinstance(roles, list):
        for role in roles:
            if isinstance(role, dict) and role.get("roleType") == "pioneer":
                ids.add(str(role.get("id")))
    for role_id, command in commands.items():
        if isinstance(command, dict) and command.get("action") in (
            "acceptTask", "submitAnswer",
        ):
            ids.add(str(role_id))
    return ids


def _task_point_states(team: dict[str, Any]) -> list[dict[str, Any]]:
    raw = team.get("playerTasks")
    if not isinstance(raw, list):
        return []
    result = []
    for task in raw:
        if not isinstance(task, dict):
            continue
        result.append({
            "type": task.get("taskType"),
            "pos": {
                "x": task.get("taskPosition", {}).get("x")
                if isinstance(task.get("taskPosition"), dict)
                else None,
                "y": task.get("taskPosition", {}).get("y")
                if isinstance(task.get("taskPosition"), dict)
                else None,
            },
            "cooldownRounds": task.get("coldDownRounds"),
            "scoreReward": task.get("scoreReward"),
            "goldReward": task.get("goldReward"),
            "valid": task.get("isValid"),
            "timeoutRounds": task.get("timeoutRounds"),
        })
    return result


def _pioneer_state(team: dict[str, Any]) -> dict[str, Any] | None:
    roles = team.get("roles")
    if not isinstance(roles, list):
        return None
    for role in roles:
        if not isinstance(role, dict) or role.get("roleType") != "pioneer":
            continue
        pos = role.get("pos")
        return {
            "id": str(role.get("id")),
            "pos": {
                "x": pos.get("x") if isinstance(pos, dict) else None,
                "y": pos.get("y") if isinstance(pos, dict) else None,
            },
            "health": role.get("health"),
            "backpackSize": len(role.get("backpack", []))
            if isinstance(role.get("backpack"), list)
            else 0,
        }
    return None


def _bounded_text(raw: Any, limit: int) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {
            "value": None,
            "originalLength": None,
            "truncated": False,
            "platformTruncated": False,
            "fingerprint": None,
        }
    return {
        "value": raw[:limit],
        "originalLength": len(raw),
        "truncated": len(raw) > limit,
        "platformTruncated": "[TRUNCATED]" in raw,
        "fingerprint": hashlib.sha256(
            raw.encode("utf-8"),
        ).hexdigest()[:16],
    }


def _classify_answer_result(
    raw_errors: list[Any],
) -> dict[str, Any] | None:
    """Infer answer result from error codes.

    error code=1 -> timeout, code=2 -> wrong, other -> error.
    """
    if not raw_errors:
        return None
    codes: list[int] = []
    for error in raw_errors:
        code = error.get("errorCode") if isinstance(error, dict) else error
        try:
            code_int = int(code) if code is not None else None
        except (TypeError, ValueError):
            code_int = None
        if code_int is not None:
            codes.append(code_int)
    if not codes:
        return None
    if 1 in codes:
        result_class = "timeout"
    elif 2 in codes:
        result_class = "wrong"
    else:
        result_class = "error"
    return {
        "class": result_class,
        "errorCodes": codes,
    }


def task_detail_log_record(
    payload: dict[str, Any],
    response: dict[str, Any],
    *,
    decision_trace: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    commands = response.get("roleCommandMap")
    commands = commands if isinstance(commands, dict) else {}
    submitted = []
    for role_id, command in sorted(
        commands.items(), key=lambda item: str(item[0]),
    ):
        if (
            not isinstance(command, dict)
            or command.get("action") != "submitAnswer"
        ):
            continue
        submitted.append({
            "roleId": str(role_id),
            "text": _detail_text(command.get("taskAnswer")),
        })
    trace = decision_trace if isinstance(decision_trace, dict) else {}
    task_instance_id = trace.get("taskInstanceId")
    tool_inputs = trace.get("taskToolInputs")
    accepted_tool_inputs = [
        value for value in tool_inputs[:2]
        if isinstance(value, dict)
        and value.get("kind") in ("llm", "cmd")
        and value.get("receivedRound") == payload.get("roundNo")
    ] if isinstance(tool_inputs, list) and task_instance_id is not None else []
    accepted_kinds = {value["kind"] for value in accepted_tool_inputs}
    news = trace.get("newsInterpretation")
    news = news if isinstance(news, dict) else {}
    news_events = news.get("events")
    news_events = news_events if isinstance(news_events, list) else []
    has_news_response = any(
        isinstance(event, dict)
        and event.get("kind") in ("response_accepted", "response_rejected")
        for event in news_events
    )
    has_news_prompt = any(
        isinstance(event, dict) and event.get("kind") == "request_issued"
        for event in news_events
    )
    raw_text = {
        "phaseTask": payload.get("phaseTask"),
        "llmResp": None if has_news_response else payload.get("llmResp"),
        "lastCmdResult": payload.get("lastCmdResult"),
        "prompt": None if has_news_prompt else response.get("prompt"),
        "executeCmd": response.get("executeCmd"),
    }
    relevant = (
        any(isinstance(value, str) and value for value in raw_text.values())
        or bool(submitted)
        or task_instance_id is not None
        or (
            not news_events
            and isinstance(payload.get("errors"), list)
            and bool(payload.get("errors"))
        )
    )
    if not relevant:
        return None
    team = payload.get("teamOur")
    team = team if isinstance(team, dict) else {}
    raw_feedback = payload.get("lastRoundRoleActionResults")
    raw_feedback = raw_feedback if isinstance(raw_feedback, dict) else {}
    raw_errors = payload.get("errors")
    raw_errors = raw_errors if isinstance(raw_errors, list) else []
    errors = []
    for error in raw_errors[:MAX_LOG_ITEMS]:
        if not isinstance(error, dict):
            continue
        errors.append({
            "code": error.get("errorCode"),
            "description": _detail_text(error.get("description")),
        })
    return {
        "event": "task_detail",
        "buildId": BUILD_ID,
        "timestampUtc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "roundNo": payload.get("roundNo"),
        "team": {"type": team.get("type"), "id": team.get("teamId")},
        "requestFingerprint": _payload_fingerprint(payload),
        "taskInstanceId": task_instance_id,
        "inputAssociation": {
            "phaseTask": (
                "current_active_task"
                if task_instance_id is not None
                and isinstance(payload.get("phaseTask"), str)
                and bool(payload.get("phaseTask"))
                else "unknown"
            ),
            "llmResp": (
                "accepted_current_task" if "llm" in accepted_kinds
                else "unknown_previous_request"
            ),
            "lastCmdResult": (
                "accepted_current_task" if "cmd" in accepted_kinds
                else "unknown_previous_request"
            ),
            "prompt": (
                "current_active_task"
                if task_instance_id is not None else "unknown"
            ),
            "executeCmd": (
                "current_active_task"
                if task_instance_id is not None else "unknown"
            ),
            "submittedAnswers": (
                "current_active_task"
                if task_instance_id is not None else "unknown"
            ),
        },
        "flow": {
            "phaseTask": "request_current",
            "llmResp": "request_from_prior_prompt",
            "lastCmdResult": "request_from_prior_command",
            "prompt": "response_current",
            "executeCmd": "response_current",
            "submittedAnswers": "response_current",
        },
        "acceptedToolInputs": accepted_tool_inputs,
        "text": {
            name: _detail_text(value) for name, value in raw_text.items()
        },
        "submittedAnswers": _limited(submitted, len(submitted)),
        "actionFeedback": _limited([
            {"roleId": str(role_id), "ok": result}
            for role_id, result in sorted(
                raw_feedback.items(), key=lambda item: str(item[0]),
            )[:MAX_LOG_ITEMS]
            if isinstance(result, bool)
        ], len(raw_feedback)),
        "errors": _limited(errors, len(raw_errors)),
        "economy": {
            "gold": team.get("goldNum"),
            "score": team.get("totalScore"),
        },
    }


def news_detail_log_record(
    payload: dict[str, Any],
    *,
    decision_trace: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    trace = decision_trace if isinstance(decision_trace, dict) else {}
    news = trace.get("newsInterpretation")
    news = news if isinstance(news, dict) else {}
    raw_events = news.get("events")
    raw_events = raw_events if isinstance(raw_events, list) else []
    events = [
        _news_event_detail(event)
        for event in raw_events[:MAX_LOG_ITEMS]
        if isinstance(event, dict)
    ]
    if not events:
        return None
    team = payload.get("teamOur")
    team = team if isinstance(team, dict) else {}
    return {
        "event": "news_detail",
        "buildId": BUILD_ID,
        "timestampUtc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "roundNo": payload.get("roundNo"),
        "team": {"type": team.get("type"), "id": team.get("teamId")},
        "requestFingerprint": _payload_fingerprint(payload),
        "events": events,
    }


def _news_event_detail(raw: dict[str, Any]) -> dict[str, Any]:
    detail = {
        name: raw.get(name)
        for name in (
            "kind",
            "requestId",
            "session",
            "issuedRound",
            "issuedDay",
            "requestDay",
            "reason",
            "candidateCount",
        )
        if name in raw
    }
    rejection = raw.get("rejectionDetail")
    if isinstance(rejection, dict):
        detail["rejectionDetail"] = {
            name: rejection[name]
            for name in ("candidateIndex", "field", "code")
            if name in rejection
        }
    for name in ("sourceIds", "newSourceIds", "contextSourceIds"):
        source_ids = raw.get(name)
        if isinstance(source_ids, list):
            detail[name] = [
                value for value in source_ids[:8] if isinstance(value, str)
            ]
    for name in ("prompt", "response"):
        text = raw.get(name)
        if not isinstance(text, dict):
            continue
        value = text.get("value")
        value = value if isinstance(value, str) else ""
        detail[name] = {
            "value": value[:MAX_NEWS_DETAIL_CHARS],
            "originalLength": text.get("originalLength"),
            "truncated": bool(text.get("truncated"))
            or len(value) > MAX_NEWS_DETAIL_CHARS,
            "fingerprint": text.get("fingerprint"),
        }
    candidates = raw.get("candidates")
    if isinstance(candidates, list):
        detail["candidates"] = candidates[:8]
    return detail


def _detail_text(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {
            "value": None,
            "inputType": type(raw).__name__,
            "originalLength": None,
            "truncated": False,
            "platformTruncated": False,
            "fingerprint": None,
        }
    return {
        "value": raw[:MAX_TASK_DETAIL_CHARS],
        "inputType": "str",
        "originalLength": len(raw),
        "truncated": len(raw) > MAX_TASK_DETAIL_CHARS,
        "platformTruncated": "[TRUNCATED]" in raw,
        "fingerprint": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        "requestFingerprint": _payload_fingerprint(payload),
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
        "decision": _summary_decision_trace(decision_trace),
        "timingMs": timing_ms,
        "timingScope": "server-side only; not judger end-to-end",
    }


def _summary_decision_trace(
    decision_trace: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(decision_trace, dict):
        return None
    summary = dict(decision_trace)
    raw_news = decision_trace.get("newsInterpretation")
    if not isinstance(raw_news, dict):
        return summary
    news = dict(raw_news)
    raw_events = raw_news.get("events")
    if isinstance(raw_events, list):
        events = []
        for raw_event in raw_events[:MAX_LOG_ITEMS]:
            if not isinstance(raw_event, dict):
                continue
            event = _news_event_detail(raw_event)
            event.pop("prompt", None)
            event.pop("response", None)
            event.pop("candidates", None)
            events.append(event)
        news["events"] = events
    summary["newsInterpretation"] = news
    return summary

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
        return "completed"
    if first_line.startswith("[exitCode:") and first_line.endswith("]"):
        try:
            exit_code = int(first_line[len("[exitCode:"):-1])
        except ValueError:
            return "unknown"
        return "completed" if exit_code == 0 else "failed"
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
        try:
            decision_trace = decision_traces[0] if decision_traces else None
            record = turn_log_record(
                payload,
                response,
                processing_ms,
                decision_trace=decision_trace,
            )
            LOGGER.info(
                "%s",
                json.dumps(record, ensure_ascii=False, separators=(",", ":")),
            )
            task_record = task_log_record(
                payload, response, decision_trace=decision_trace,
            )
            if task_record is not None:
                LOGGER.info(
                    "%s",
                    json.dumps(
                        task_record, ensure_ascii=False, separators=(",", ":"),
                    ),
                )
            news_detail = news_detail_log_record(
                payload, decision_trace=decision_trace,
            )
            if news_detail is not None:
                LOGGER.info(
                    "%s",
                    json.dumps(
                        news_detail, ensure_ascii=False, separators=(",", ":"),
                    ),
                )
        except Exception:
            LOGGER.exception("post-response logging failed buildId=%s", BUILD_ID)

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
