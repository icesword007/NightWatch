from collections import Counter
from typing import Iterable

from .intelligence import NewsCandidate, TreasureConditions
from .protocol import PIONEER, Turn


MAX_TREASURE_CANDIDATES = 32


def evaluate_treasure_candidates(
    turn: Turn,
    candidates: Iterable[NewsCandidate],
    *,
    session_index: int,
) -> tuple[dict, ...]:
    current = tuple(
        candidate for candidate in candidates
        if candidate.kind == "treasure"
    )[-MAX_TREASURE_CANDIDATES:]
    conflicting_indexes = _conflicting_candidate_indexes(
        current, session_index=session_index,
    )
    return tuple(
        _evaluate_candidate(
            turn,
            candidate,
            candidate_index=index,
            session_index=session_index,
            alternatives=index in conflicting_indexes,
        )
        for index, candidate in enumerate(current)
    )


def _evaluate_candidate(
    turn: Turn,
    candidate: NewsCandidate,
    *,
    candidate_index: int,
    session_index: int,
    alternatives: bool,
) -> dict:
    reasons: list[str] = []
    result = {
        "candidateIndex": candidate_index,
        "requestId": candidate.request_id,
        "status": "pending_validation",
        "localChecksPassed": False,
        "actionEnabled": False,
        "reasons": reasons,
        "timeStatus": "unknown",
        "missingItems": {},
        "pioneerId": None,
    }
    conditions = candidate.treasure_conditions
    if conditions is None:
        reasons.append("unstructured")
        return result

    if conditions.location is None:
        reasons.append("missing_location")
    if conditions.window is None:
        reasons.append("missing_window")
    if conditions.items is None:
        reasons.append("missing_items")
    if alternatives:
        reasons.append("alternative_candidates")
    if (
        candidate.source_session != session_index
        or any(
            source_session != session_index
            for source_session in candidate.citation_source_sessions
        )
        or any(
            source_session != session_index
            for field in _fields(conditions)
            for source_session in field.source_sessions
        )
    ):
        reasons.append("source_session_mismatch")
    if candidate.citation_source_truncated or any(
        field.source_truncated for field in _fields(conditions)
    ):
        reasons.append("source_truncated")
    if candidate.missing_conditions:
        reasons.append("unresolved_conditions")
    if candidate.conflicts:
        reasons.append("conflicts")

    if conditions.location is not None:
        location = conditions.location.value
        if not (
            0 <= location["x"] < turn.width
            and 0 <= location["y"] < turn.height
        ):
            reasons.append("location_out_of_bounds")

    if conditions.window is not None:
        window = conditions.window.value
        if turn.round_no < window["startRound"]:
            result["timeStatus"] = "future"
            reasons.append("window_future")
        elif turn.round_no > window["endRound"]:
            result["timeStatus"] = "expired"
            reasons.append("window_expired")
        else:
            result["timeStatus"] = "open"

    if conditions.items is not None:
        required = Counter(conditions.items.value)
        alive = turn.pioneers()
        any_pioneer = any(unit.kind == PIONEER for unit in turn.ours)
        if not alive:
            reasons.append("pioneer_dead" if any_pioneer else "pioneer_missing")
        else:
            known_items = set(turn.weapon_prices)
            for pioneer in alive:
                known_items.update(pioneer.backpack)
            if any(item not in known_items for item in required):
                reasons.append("item_unavailable_or_unknown")
            pioneer, missing = min(
                (
                    (
                        role,
                        required - Counter(role.backpack),
                    )
                    for role in alive
                ),
                key=lambda item: (sum(item[1].values()), item[0].unit_id),
            )
            result["pioneerId"] = str(pioneer.unit_id)
            result["missingItems"] = dict(sorted(missing.items()))
            if missing:
                reasons.append("missing_items")
        if turn.phase_task:
            reasons.append("active_task")

    result["localChecksPassed"] = not reasons
    return result


def _fields(conditions: TreasureConditions) -> tuple:
    return tuple(
        field for field in (
            conditions.location, conditions.window, conditions.items,
        )
        if field is not None
    )


def _conflicting_candidate_indexes(
    candidates: tuple[NewsCandidate, ...],
    *,
    session_index: int,
) -> frozenset[int]:
    eligible = tuple(
        (index, candidate.treasure_conditions)
        for index, candidate in enumerate(candidates)
        if candidate.source_session == session_index
        and candidate.treasure_conditions is not None
    )
    conflicts = set()
    for offset, (left_index, left) in enumerate(eligible):
        for right_index, right in eligible[offset + 1:]:
            if _conditions_conflict(left, right):
                conflicts.update((left_index, right_index))
    return frozenset(conflicts)


def _conditions_conflict(
    left: TreasureConditions,
    right: TreasureConditions,
) -> bool:
    for field_name in ("location", "window", "items"):
        left_field = getattr(left, field_name)
        right_field = getattr(right, field_name)
        if left_field is None or right_field is None:
            continue
        if _normalized_value(field_name, left_field.value) != _normalized_value(
            field_name, right_field.value,
        ):
            return True
    return False


def _normalized_value(field_name: str, value: object) -> tuple:
    if field_name == "items":
        return tuple(sorted(Counter(value).items()))
    return tuple(sorted(value.items()))
