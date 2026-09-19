from collections import Counter
from dataclasses import replace
from typing import Callable, Iterable

from .defense import DAY_WORK_RETURN_MARGIN
from .grid import next_step
from .intelligence import NewsCandidate, TreasureConditions
from .protocol import PIONEER, TOWER_TYPES, Pos, Turn, Unit, distance


MAX_TREASURE_CANDIDATES = 32
MAX_TREASURE_ROUTE_CANDIDATES = 4
MAX_TREASURE_ROUTE_EXPANSIONS = 256
TREASURE_ROUTE_SECONDS = 0.01


def evaluate_treasure_candidates(
    turn: Turn,
    candidates: Iterable[NewsCandidate],
    *,
    session_index: int,
    daytime_assignments: dict[
        int, tuple[Unit, tuple[Pos, Pos | None, int]]
    ] | None = None,
    clock: Callable[[], float] | None = None,
    deadline: float | None = None,
    allocation_context: dict | None = None,
) -> tuple[dict, ...]:
    current = tuple(
        candidate for candidate in candidates
        if candidate.kind == "treasure"
    )[-MAX_TREASURE_CANDIDATES:]
    conflicting_indexes = _conflicting_candidate_indexes(
        current, session_index=session_index,
    )
    route_budget = (
        {
            "remaining": MAX_TREASURE_ROUTE_EXPANSIONS,
            "evaluated": 0,
            "deadline": min(deadline, clock() + TREASURE_ROUTE_SECONDS),
        }
        if clock is not None and deadline is not None else None
    )
    return tuple(
        _evaluate_candidate(
            turn,
            candidate,
            candidate_index=index,
            session_index=session_index,
            alternatives=index in conflicting_indexes,
            daytime_assignments=daytime_assignments,
            clock=clock,
            route_budget=route_budget,
            allocation_context=allocation_context,
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
    daytime_assignments: dict[
        int, tuple[Unit, tuple[Pos, Pos | None, int]]
    ] | None,
    clock: Callable[[], float] | None,
    route_budget: dict | None,
    allocation_context: dict | None,
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
        "route": {
            "status": "not_evaluated",
            "outboundMoveRounds": None,
            "waitRounds": None,
            "earliestActionRound": None,
            "returnMoveRounds": None,
            "returnWithMargin": "unknown",
            "expansions": 0,
        },
        "procurement": {
            "missingItemCount": None,
            "missingCost": None,
            "backpackFreeSlots": None,
            "shopAvailable": "unknown",
            "currentCashCoversMissingCost": "unknown",
            "backpackFitsMissing": "unknown",
            "routeScope": "missing_item_procurement",
            "spendingBudgetEvaluated": False,
            "spendableBudgetCoversMissingCost": "unknown",
            "routeEvaluated": False,
            "returnWindowEvaluated": False,
            "semanticValidationEvaluated": False,
            "currentAllocation": {
                "scope": "current_response_only",
                "evaluated": False,
                "status": "context_unknown",
                "committedGold": None,
                "cashAfterCommitments": None,
                "cashAfterCommitmentsCoversMissingCost": "unknown",
                "pioneerHasAcceptedAction": "unknown",
                "pioneerPriorityReserved": "unknown",
                "reasons": [],
            },
            "route": {
                "status": "not_evaluated",
                "shopMoveRounds": None,
                "purchaseRounds": None,
                "altarMoveRounds": None,
                "waitRounds": None,
                "earliestActionRound": None,
                "returnMoveRounds": None,
                "returnWithMargin": "unknown",
                "expansions": 0,
            },
        },
    }
    conditions = candidate.treasure_conditions
    result["evidenceAudit"] = {
        name: _field_audit(
            getattr(conditions, name) if conditions is not None else None,
            session_index=session_index,
            is_window=name == "window",
        )
        for name in ("location", "window", "items")
    }
    if conditions is None:
        reasons.append("unstructured")
        _assess_current_allocation(turn, result, session_index, allocation_context)
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
            procurement = result["procurement"]
            missing_count = sum(missing.values())
            procurement["missingItemCount"] = missing_count
            procurement["shopAvailable"] = all(
                item in turn.weapon_prices for item in missing
            )
            if procurement["shopAvailable"]:
                cost = sum(
                    count * turn.weapon_prices[item]
                    for item, count in missing.items()
                )
                procurement["missingCost"] = cost
                procurement["currentCashCoversMissingCost"] = turn.gold >= cost
            if pioneer.capacity is not None:
                free_slots = max(0, pioneer.capacity - len(pioneer.backpack))
                procurement["backpackFreeSlots"] = free_slots
                procurement["backpackFitsMissing"] = free_slots >= missing_count
            if missing:
                reasons.append("missing_items")
        if turn.phase_task:
            reasons.append("active_task")

    result["localChecksPassed"] = not reasons
    _assess_current_allocation(turn, result, session_index, allocation_context)
    if route_budget is not None and clock is not None:
        _assess_route(
            turn, conditions, result, daytime_assignments, clock,
            route_budget,
        )
    return result


def _assess_current_allocation(
    turn: Turn, result: dict, session_index: int, context: dict | None,
) -> None:
    allocation = result["procurement"]["currentAllocation"]
    if context is None:
        return
    if (
        not isinstance(context, dict)
        or context.get("teamId") != turn.team_id
        or type(context.get("roundNo")) is not int
        or context["roundNo"] != turn.round_no
        or type(context.get("sessionIndex")) is not int
        or context["sessionIndex"] != session_index
        or type(context.get("currentGold")) is not int
        or context["currentGold"] != turn.gold
        or turn.gold < 0
        or type(context.get("goldRemaining")) is not int
        or not 0 <= context["goldRemaining"] <= turn.gold
        or any(
            not isinstance(context.get(key), frozenset)
            or any(type(role_id) is not int for role_id in context[key])
            for key in ("acceptedActors", "taskReservedActors", "fundingReservedActors")
        )
    ):
        allocation["status"] = "context_mismatch"
        return
    remaining = context["goldRemaining"]
    allocation["evaluated"] = True
    allocation["status"] = "evaluated"
    allocation["committedGold"] = turn.gold - remaining
    allocation["cashAfterCommitments"] = remaining
    cost = result["procurement"]["missingCost"]
    if cost is not None:
        covers = remaining >= cost
        allocation["cashAfterCommitmentsCoversMissingCost"] = covers
        if not covers and turn.gold >= cost:
            allocation["reasons"].append("cash_committed_elsewhere")
    pioneer_id = result["pioneerId"]
    if pioneer_id is not None:
        pioneer = int(pioneer_id)
        acted = pioneer in context["acceptedActors"]
        reserved = (
            pioneer in context["taskReservedActors"]
            or pioneer in context["fundingReservedActors"]
        )
        allocation["pioneerHasAcceptedAction"] = acted
        allocation["pioneerPriorityReserved"] = reserved
        if acted:
            allocation["reasons"].append("actor_has_action")
        if reserved:
            allocation["reasons"].append("actor_priority_reserved")


def _field_audit(field, *, session_index: int, is_window: bool) -> dict:
    audit = {
        "status": "missing_field",
        "citationCount": 0,
        "distinctSourceCount": "unknown",
        "duplicateSourceCount": "unknown",
        "reasons": [],
    }
    if is_window:
        audit["timeAnchorStatus"] = "anchor_unknown"
    if field is None:
        return audit
    audit["citationCount"] = len(field.citations)
    fingerprints = field.source_fingerprints
    if fingerprints is not None and len(fingerprints) == len(field.citations):
        distinct = len(set(fingerprints))
        audit["distinctSourceCount"] = distinct
        audit["duplicateSourceCount"] = len(fingerprints) - distinct
        if len(fingerprints) > distinct:
            audit["reasons"].append("duplicate_source")
    else:
        audit["reasons"].append("source_metadata_unknown")
    if field.source_truncated:
        audit["reasons"].append("source_truncated")
    if any(source != session_index for source in field.source_sessions):
        audit["reasons"].append("source_session_mismatch")
    derivation = field.derivation
    if derivation is None:
        audit["status"] = "audit_missing"
    else:
        audit["status"] = {
            "direct": "direct_unverified",
            "derived": "derived_unverified",
            "unknown": "unknown",
        }[derivation.kind]
        if derivation.unresolved:
            audit["reasons"].append("unresolved_present")
        if is_window and derivation.time_basis == "absolute_rounds":
            audit["timeAnchorStatus"] = "absolute_anchor_unverified"
    return audit


def _assess_route(
    turn: Turn,
    conditions: TreasureConditions,
    result: dict,
    assignments: dict[int, tuple[Unit, tuple[Pos, Pos | None, int]]] | None,
    clock: Callable[[], float],
    budget: dict,
) -> None:
    route = result["route"]
    if not turn.is_day:
        route["status"] = "night"
        if result["procurement"]["missingItemCount"] not in (None, 0):
            result["procurement"]["route"]["status"] = "night"
        return
    if result["procurement"]["missingItemCount"] not in (None, 0):
        route["status"] = "procurement_route_not_evaluated"
        _assess_shopping_route(
            turn, conditions, result, assignments, clock, budget,
        )
        return
    if any(reason != "window_future" for reason in result["reasons"]):
        route["status"] = "prerequisite_blocked"
        return
    if (
        conditions.location is None or conditions.window is None
        or conditions.items is None or result["pioneerId"] is None
    ):
        route["status"] = "prerequisite_blocked"
        return
    if budget["evaluated"] >= MAX_TREASURE_ROUTE_CANDIDATES:
        route["status"] = "candidate_limit"
        return
    budget["evaluated"] += 1
    if clock() >= budget["deadline"]:
        route["status"] = "deadline"
        return
    pioneer = turn.unit(int(result["pioneerId"]))
    if pioneer is None:
        route["status"] = "prerequisite_blocked"
        return
    post = _assigned_post(turn, pioneer, assignments)
    location = Pos(**conditions.location.value)
    window = conditions.window.value
    day_end = turn.round_no + turn.rounds_until_night - 1
    blocked = turn.blocked(pioneer)
    stands = sorted(
        (
            Pos(location.x + dx, location.y + dy)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if dx or dy
        ),
        key=lambda stand: (distance(pioneer.pos, stand), stand.x, stand.y),
    )
    legal = [stand for stand in stands if turn.land(stand) and stand not in blocked]
    if not legal:
        route["status"] = "unreachable"
        route["returnWithMargin"] = False if post is not None else "unknown"
        return
    first_failure = None
    for stand in legal:
        if clock() >= budget["deadline"]:
            route["status"] = "deadline"
            route["returnWithMargin"] = "unknown"
            return
        if budget["remaining"] <= 0:
            route["status"] = "expansion_limit"
            route["returnWithMargin"] = "unknown"
            return
        outward = next_step(
            turn, pioneer, stand, clock=clock,
            deadline=budget["deadline"],
            max_expansions=budget["remaining"],
        )
        budget["remaining"] -= outward.expansions
        route["expansions"] += outward.expansions
        if outward.status in ("deadline", "expansion_limit"):
            route["status"] = outward.status
            route["returnWithMargin"] = "unknown"
            return
        if outward.status == "unreachable" or outward.cost is None:
            continue
        move_rounds = outward.cost
        action_round = max(turn.round_no + move_rounds, window["startRound"])
        attempt = {
            "outboundMoveRounds": move_rounds,
            "waitRounds": action_round - turn.round_no - move_rounds,
            "earliestActionRound": action_round,
            "returnMoveRounds": None,
            "returnWithMargin": "unknown",
        }
        if action_round > window["endRound"] or action_round > day_end:
            attempt.update(
                status="candidate_window_missed",
                returnWithMargin=False if post is not None else "unknown",
            )
            if first_failure is None:
                first_failure = attempt
            continue
        if post is None:
            route.update(attempt, status="return_post_unknown")
            return
        projected = replace(pioneer, pos=stand)
        projected_turn = replace(
            turn,
            ours=tuple(
                projected if role.unit_id == pioneer.unit_id else role
                for role in turn.ours
            ),
        )
        if clock() >= budget["deadline"]:
            route["status"] = "deadline"
            route["returnWithMargin"] = "unknown"
            return
        if budget["remaining"] <= 0:
            route["status"] = "expansion_limit"
            route["returnWithMargin"] = "unknown"
            return
        home = next_step(
            projected_turn, projected, post, clock=clock,
            deadline=budget["deadline"],
            max_expansions=budget["remaining"],
        )
        budget["remaining"] -= home.expansions
        route["expansions"] += home.expansions
        if home.status in ("deadline", "expansion_limit"):
            route["status"] = home.status
            route["returnWithMargin"] = "unknown"
            return
        if home.status == "unreachable" or home.cost is None:
            attempt.update(status="candidate_return_unreachable", returnWithMargin=False)
            if first_failure is None:
                first_failure = attempt
            continue
        attempt["returnMoveRounds"] = home.cost
        timely = (
            action_round - turn.round_no + 1 + home.cost
            + DAY_WORK_RETURN_MARGIN <= turn.rounds_until_night
        )
        attempt["returnWithMargin"] = timely
        attempt["status"] = "feasible" if timely else "candidate_return_too_late"
        if timely:
            route.update(attempt)
            return
        if first_failure is None:
            first_failure = attempt
    if first_failure is not None:
        route.update(first_failure)
    else:
        route["status"] = "unreachable"
        route["returnWithMargin"] = False if post is not None else "unknown"


def _assess_shopping_route(
    turn: Turn,
    conditions: TreasureConditions,
    result: dict,
    assignments: dict[int, tuple[Unit, tuple[Pos, Pos | None, int]]] | None,
    clock: Callable[[], float],
    budget: dict,
) -> None:
    procurement = result["procurement"]
    route = procurement["route"]
    if any(
        reason not in ("missing_items", "window_future", "item_unavailable_or_unknown")
        for reason in result["reasons"]
    ):
        route["status"] = "prerequisite_blocked"
        return
    if (
        conditions.location is None or conditions.window is None
        or conditions.items is None or result["pioneerId"] is None
    ):
        route["status"] = "prerequisite_blocked"
        return
    if not procurement["shopAvailable"]:
        route["status"] = "price_missing"
        return
    if procurement["backpackFitsMissing"] is not True:
        route["status"] = (
            "capacity_insufficient"
            if procurement["backpackFitsMissing"] is False
            else "capacity_unknown"
        )
        return
    if not procurement["currentCashCoversMissingCost"]:
        route["status"] = "cash_insufficient"
        return
    shops = turn.zones_of("weaponShop")
    if not shops:
        route["status"] = "shop_missing"
        return
    if budget["evaluated"] >= MAX_TREASURE_ROUTE_CANDIDATES:
        route["status"] = "candidate_limit"
        return
    budget["evaluated"] += 1
    if clock() >= budget["deadline"]:
        route["status"] = "deadline"
        return
    pioneer = turn.unit(int(result["pioneerId"]))
    if pioneer is None:
        route["status"] = "prerequisite_blocked"
        return
    procurement["routeEvaluated"] = True
    post = _assigned_post(turn, pioneer, assignments)
    location = Pos(**conditions.location.value)
    window = conditions.window.value
    day_end = turn.round_no + turn.rounds_until_night - 1
    blocked = turn.blocked(pioneer)
    purchase_rounds = len(result["missingItems"])
    purchased = tuple(
        item for item, count in result["missingItems"].items()
        for _ in range(count)
    )
    first_failure = None
    for shop in sorted(shops, key=lambda pos: (distance(pioneer.pos, pos), pos.x, pos.y)):
        for shop_stand in _legal_stands(turn, shop, blocked, pioneer.pos):
            shop_status, shop_moves = _shopping_path(
                turn, pioneer, shop_stand, clock, budget, route,
            )
            if shop_status in ("deadline", "expansion_limit"):
                route["status"] = shop_status
                return
            if shop_moves is None:
                continue
            buy_end = turn.round_no + shop_moves + purchase_rounds - 1
            if buy_end > day_end:
                attempt = {
                    "status": "candidate_window_missed",
                    "shopMoveRounds": shop_moves,
                    "purchaseRounds": purchase_rounds,
                    "altarMoveRounds": None,
                    "waitRounds": None,
                    "earliestActionRound": None,
                    "returnMoveRounds": None,
                    "returnWithMargin": False if post is not None else "unknown",
                }
                if first_failure is None:
                    first_failure = attempt
                continue
            at_shop = replace(
                pioneer, pos=shop_stand,
                backpack=pioneer.backpack + purchased,
            )
            shop_turn = _projected_turn(turn, at_shop)
            for altar_stand in _legal_stands(
                shop_turn, location, shop_turn.blocked(at_shop), shop_stand,
            ):
                altar_status, altar_moves = _shopping_path(
                    shop_turn, at_shop, altar_stand, clock, budget, route,
                )
                if altar_status in ("deadline", "expansion_limit"):
                    route["status"] = altar_status
                    return
                if altar_moves is None:
                    continue
                action_round = max(
                    turn.round_no + shop_moves + purchase_rounds + altar_moves,
                    window["startRound"],
                )
                attempt = {
                    "shopMoveRounds": shop_moves,
                    "purchaseRounds": purchase_rounds,
                    "altarMoveRounds": altar_moves,
                    "waitRounds": action_round - turn.round_no - shop_moves
                    - purchase_rounds - altar_moves,
                    "earliestActionRound": action_round,
                    "returnMoveRounds": None,
                    "returnWithMargin": "unknown",
                }
                if action_round > window["endRound"] or action_round > day_end:
                    attempt.update(
                        status="candidate_window_missed",
                        returnWithMargin=False if post is not None else "unknown",
                    )
                    if first_failure is None:
                        first_failure = attempt
                    continue
                if post is None:
                    route.update(attempt, status="return_post_unknown")
                    return
                at_altar = replace(at_shop, pos=altar_stand)
                altar_turn = _projected_turn(turn, at_altar)
                home_status, home_moves = _shopping_path(
                    altar_turn, at_altar, post, clock, budget, route,
                )
                if home_status in ("deadline", "expansion_limit"):
                    route["status"] = home_status
                    return
                if home_moves is None:
                    attempt.update(
                        status="candidate_return_unreachable",
                        returnWithMargin=False,
                    )
                    if first_failure is None:
                        first_failure = attempt
                    continue
                attempt["returnMoveRounds"] = home_moves
                timely = (
                    action_round - turn.round_no + 1 + home_moves
                    + DAY_WORK_RETURN_MARGIN <= turn.rounds_until_night
                )
                attempt["returnWithMargin"] = timely
                attempt["status"] = (
                    "feasible" if timely else "candidate_return_too_late"
                )
                if timely:
                    route.update(attempt)
                    procurement["returnWindowEvaluated"] = True
                    return
                if first_failure is None:
                    first_failure = attempt
    if first_failure is not None:
        route.update(first_failure)
        procurement["returnWindowEvaluated"] = (
            first_failure["returnMoveRounds"] is not None
        )
    else:
        route["status"] = "unreachable"
        route["returnWithMargin"] = False if post is not None else "unknown"


def _legal_stands(
    turn: Turn, target: Pos, blocked: frozenset[Pos], origin: Pos,
) -> tuple[Pos, ...]:
    stands = (
        Pos(target.x + dx, target.y + dy)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        if dx or dy
    )
    return tuple(sorted(
        (stand for stand in stands if turn.land(stand) and stand not in blocked),
        key=lambda stand: (distance(origin, stand), stand.x, stand.y),
    ))


def _projected_turn(turn: Turn, projected: Unit) -> Turn:
    return replace(
        turn,
        ours=tuple(
            projected if role.unit_id == projected.unit_id else role
            for role in turn.ours
        ),
    )


def _shopping_path(
    turn: Turn, pioneer: Unit, goal: Pos,
    clock: Callable[[], float], budget: dict, route: dict,
) -> tuple[str, int | None]:
    if clock() >= budget["deadline"]:
        return "deadline", None
    if budget["remaining"] <= 0:
        return "expansion_limit", None
    path = next_step(
        turn, pioneer, goal, clock=clock,
        deadline=budget["deadline"],
        max_expansions=budget["remaining"],
    )
    budget["remaining"] -= path.expansions
    route["expansions"] += path.expansions
    return path.status, path.cost


def _assigned_post(
    turn: Turn,
    pioneer: Unit,
    assignments: dict[int, tuple[Unit, tuple[Pos, Pos | None, int]]] | None,
) -> Pos | None:
    for weapon_id, (role, route) in sorted((assignments or {}).items()):
        weapon = turn.unit(weapon_id)
        stand = route[0]
        if (
            role.unit_id == pioneer.unit_id
            and role.pos == pioneer.pos
            and weapon is not None
            and weapon.kind in TOWER_TYPES
            and distance(stand, weapon.pos) == 1
            and turn.land(stand)
            and stand not in turn.blocked(pioneer)
        ):
            return stand
    return None


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
