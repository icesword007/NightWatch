import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent.brain import DecisionEngine
from agent.intelligence import (
    CitedTreasureValue,
    NewsCandidate,
    NewsCitation,
    TreasureConditions,
)
from agent.protocol import Pos, Turn
from agent import server as server_module
from agent.treasure import evaluate_treasure_candidates


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def payload(round_no=1, *, team_id="treasure-tests"):
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value["roundNo"] = round_no
    value["teamOur"]["teamId"] = team_id
    value["worldNews"] = {
        "officialNews": "",
        "folkLegends": "At (4,4), bring two StarSand from round 2 to 20.",
    }
    value["teamOur"]["roles"].append({
        "id": 10011,
        "pos": {"x": 3, "y": 3},
        "roleType": "pioneer",
        "health": 200,
        "attackPower": 0,
        "attackRange": 0,
        "backPackCapability": 40,
        "backpack": [],
    })
    value["weaponShopList"] = [{"name": "StarSand", "price": 15}]
    return value


def cited(value, *, session=1, truncated=False, source="news-s1-folk-abc"):
    return CitedTreasureValue(
        value=value,
        citations=(NewsCitation(source, "evidence"),),
        source_sessions=(session,),
        source_truncated=truncated,
    )


def candidate(
    *,
    location=None,
    window=None,
    items=None,
    session=1,
    missing=(),
    conflicts=(),
):
    conditions = None
    if location is not None or window is not None or items is not None:
        conditions = TreasureConditions(
            location=cited(location, session=session) if location is not None else None,
            window=cited(window, session=session) if window is not None else None,
            items=cited(tuple(items), session=session) if items is not None else None,
        )
    return NewsCandidate(
        request_id="news-s1-r1-request",
        kind="treasure",
        interpretation="hypothesis",
        citations=(NewsCitation("news-s1-folk-abc", "evidence"),),
        missing_conditions=tuple(missing),
        conflicts=tuple(conflicts),
        source_session=session,
        issued_round=1,
        treasure_conditions=conditions,
    )


def response_for(pending, *, location=(4, 4), window=(2, 20), items=None):
    source = pending.sources[0]
    items = ["StarSand"] if items is None else items

    def field(value):
        return {
            "value": value,
            "citations": [{
                "sourceId": source.source_id,
                "excerpt": "At (4,4), bring two StarSand from round 2 to 20.",
            }],
        }

    return json.dumps({
        "requestId": pending.request_id,
        "candidates": [{
            "type": "treasure",
            "interpretation": "A bounded hypothesis.",
            "citations": [{
                "sourceId": source.source_id,
                "excerpt": "At (4,4), bring two StarSand from round 2 to 20.",
            }],
            "missingConditions": [],
            "conflicts": [],
            "treasureConditions": {
                "location": field({"x": location[0], "y": location[1]}),
                "window": field({
                    "startRound": window[0], "endRound": window[1],
                }),
                "items": field(items),
            },
        }],
    })


def simple_route_case(*, round_no=10, target=(2, 2), window=(10, 20)):
    current = payload(round_no)
    current["mapInfo"].update({"width": 6, "height": 6, "zones": []})
    current["teamEnemy"]["roles"] = []
    current["robot"]["roles"] = []
    current["teamOur"]["roles"][0]["pos"] = {"x": 4, "y": 4}
    pioneer = current["teamOur"]["roles"][1]
    pioneer["pos"] = {"x": 1, "y": 1}
    pioneer["backpack"] = ["StarSand"]
    current["teamOur"]["roles"].append({
        "id": 10020, "pos": {"x": 1, "y": 2},
        "roleType": "gatling", "health": 1000, "level": 1,
        "cooldown": 0, "backpack": [],
    })
    turn = Turn.load(current)
    hypothesis = candidate(
        location={"x": target[0], "y": target[1]},
        window={"startRound": window[0], "endRound": window[1]},
        items=("StarSand",),
    )
    assignments = {
        10020: (turn.pioneers()[0], (Pos(1, 1), None, 0)),
    }
    return current, turn, hypothesis, assignments


class TreasureEvaluationTests(unittest.TestCase):
    def test_route_zero_step_wait_and_window_end_are_counted_by_action_round(self):
        _, turn, hypothesis, assignments = simple_route_case(
            window=(12, 12),
        )
        route = evaluate_treasure_candidates(
            turn, (hypothesis,), session_index=1,
            daytime_assignments=assignments,
            clock=lambda: 0.0, deadline=1.0,
        )[0]["route"]
        self.assertEqual(route["status"], "feasible")
        self.assertEqual(route["outboundMoveRounds"], 0)
        self.assertEqual(route["waitRounds"], 2)
        self.assertEqual(route["earliestActionRound"], 12)
        self.assertEqual(route["returnMoveRounds"], 0)

        _, turn, hypothesis, assignments = simple_route_case(
            target=(3, 1), window=(10, 11),
        )
        route = evaluate_treasure_candidates(
            turn, (hypothesis,), session_index=1,
            daytime_assignments=assignments,
            clock=lambda: 0.0, deadline=1.0,
        )[0]["route"]
        self.assertEqual(route["status"], "feasible")
        self.assertEqual(route["earliestActionRound"], 11)

        _, turn, hypothesis, assignments = simple_route_case(
            target=(3, 1), window=(10, 10),
        )
        route = evaluate_treasure_candidates(
            turn, (hypothesis,), session_index=1,
            daytime_assignments=assignments,
            clock=lambda: 0.0, deadline=1.0,
        )[0]["route"]
        self.assertEqual(route["status"], "candidate_window_missed")
        self.assertFalse(route["returnWithMargin"])

    def test_route_return_margin_exact_boundary_and_missing_post_unknown(self):
        _, turn, hypothesis, assignments = simple_route_case(
            round_no=68, window=(68, 68),
        )
        route = evaluate_treasure_candidates(
            turn, (hypothesis,), session_index=1,
            daytime_assignments=assignments,
            clock=lambda: 0.0, deadline=1.0,
        )[0]["route"]
        self.assertEqual(route["status"], "feasible")
        self.assertTrue(route["returnWithMargin"])

        _, turn, hypothesis, assignments = simple_route_case(
            round_no=69, window=(69, 69),
        )
        route = evaluate_treasure_candidates(
            turn, (hypothesis,), session_index=1,
            daytime_assignments=assignments,
            clock=lambda: 0.0, deadline=1.0,
        )[0]["route"]
        self.assertEqual(route["status"], "candidate_return_too_late")
        self.assertFalse(route["returnWithMargin"])
        self.assertEqual(route["outboundMoveRounds"], 0)
        self.assertEqual(route["returnMoveRounds"], 0)

        _, turn, hypothesis, _ = simple_route_case()
        route = evaluate_treasure_candidates(
            turn, (hypothesis,), session_index=1,
            daytime_assignments=None,
            clock=lambda: 0.0, deadline=1.0,
        )[0]["route"]
        self.assertEqual(route["status"], "return_post_unknown")
        self.assertEqual(route["returnWithMargin"], "unknown")

        _, turn, hypothesis, _ = simple_route_case(
            target=(3, 1), window=(10, 10),
        )
        route = evaluate_treasure_candidates(
            turn, (hypothesis,), session_index=1,
            daytime_assignments=None,
            clock=lambda: 0.0, deadline=1.0,
        )[0]["route"]
        self.assertEqual(route["status"], "candidate_window_missed")
        self.assertEqual(route["returnWithMargin"], "unknown")

    def test_route_prerequisites_block_search_without_claiming_reachability(self):
        cases = []
        current, _, hypothesis, assignments = simple_route_case()
        missing = copy.deepcopy(current)
        missing["teamOur"]["roles"][1]["backpack"] = []
        cases.append(("missing", missing, hypothesis, "procurement_route_not_evaluated"))
        active = copy.deepcopy(current)
        active["phaseTask"] = "active task"
        cases.append(("active", active, hypothesis, "prerequisite_blocked"))
        night = copy.deepcopy(current)
        night["roundNo"] = 71
        cases.append(("night", night, hypothesis, "night"))
        expired = candidate(
            location={"x": 2, "y": 2},
            window={"startRound": 1, "endRound": 9},
            items=("StarSand",),
        )
        cases.append(("expired", current, expired, "prerequisite_blocked"))
        other_session = replace(hypothesis, source_session=2)
        cases.append(("source", current, other_session, "prerequisite_blocked"))
        truncated = replace(
            hypothesis,
            citation_source_truncated=True,
        )
        cases.append(("truncated", current, truncated, "prerequisite_blocked"))
        incomplete = candidate(items=("StarSand",))
        cases.append(("incomplete", current, incomplete, "prerequisite_blocked"))
        for label, raw, value, expected in cases:
            with self.subTest(label=label):
                result = evaluate_treasure_candidates(
                    Turn.load(raw), (value,), session_index=1,
                    daytime_assignments=assignments,
                    clock=lambda: 0.0, deadline=1.0,
                )[0]
                self.assertEqual(result["route"]["status"], expected)
                self.assertEqual(result["route"]["expansions"], 0)
                self.assertFalse(result["actionEnabled"])

        conflicting = candidate(
            location={"x": 3, "y": 2},
            window={"startRound": 10, "endRound": 20},
            items=("StarSand",),
        )
        results = evaluate_treasure_candidates(
            Turn.load(current), (hypothesis, conflicting), session_index=1,
            daytime_assignments=assignments,
            clock=lambda: 0.0, deadline=1.0,
        )
        self.assertTrue(all(
            result["route"]["status"] == "prerequisite_blocked"
            and result["route"]["expansions"] == 0
            for result in results
        ))

    def test_route_distinguishes_exhausted_map_from_shared_search_cutoff(self):
        current, _, hypothesis, _ = simple_route_case(target=(4, 4))
        current["mapInfo"]["zones"] = [
            {"pos": {"x": 2, "y": y}, "neutralType": "stone"}
            for y in range(6)
        ]
        turn = Turn.load(current)
        assignments = {10020: (turn.pioneers()[0], (Pos(1, 1), None, 0))}
        unreachable = evaluate_treasure_candidates(
            turn, (hypothesis,), session_index=1,
            daytime_assignments=assignments,
            clock=lambda: 0.0, deadline=1.0,
        )[0]["route"]
        self.assertEqual(unreachable["status"], "unreachable")
        self.assertGreater(unreachable["expansions"], 0)

        occupied = copy.deepcopy(current)
        occupied["mapInfo"]["zones"] = [
            {"pos": {"x": x, "y": y}, "neutralType": "stone"}
            for x in (3, 4, 5) for y in (3, 4, 5)
            if (x, y) != (4, 4)
        ]
        no_stand = evaluate_treasure_candidates(
            Turn.load(occupied), (hypothesis,), session_index=1,
            daytime_assignments=assignments,
            clock=lambda: 0.0, deadline=1.0,
        )[0]["route"]
        self.assertEqual(no_stand["status"], "unreachable")
        self.assertEqual(no_stand["expansions"], 0)

        with patch("agent.treasure.MAX_TREASURE_ROUTE_EXPANSIONS", 1):
            routes = [item["route"] for item in evaluate_treasure_candidates(
                turn, (hypothesis,) * 5, session_index=1,
                daytime_assignments=assignments,
                clock=lambda: 0.0, deadline=1.0,
            )]
        self.assertEqual(routes[0]["status"], "expansion_limit")
        self.assertEqual(routes[-1]["status"], "candidate_limit")
        self.assertLessEqual(sum(route["expansions"] for route in routes), 1)
        self.assertTrue(all(
            route["status"] != "unreachable" for route in routes
        ))

        deadline = evaluate_treasure_candidates(
            turn, (hypothesis,), session_index=1,
            daytime_assignments=assignments,
            clock=lambda: 2.0, deadline=1.0,
        )[0]["route"]
        self.assertEqual(deadline["status"], "deadline")
        self.assertEqual(deadline["expansions"], 0)

        ticks = iter((0.0, 0.02))
        short_budget = evaluate_treasure_candidates(
            turn, (hypothesis,), session_index=1,
            daytime_assignments=assignments,
            clock=lambda: next(ticks), deadline=10.0,
        )[0]["route"]
        self.assertEqual(short_budget["status"], "deadline")
        self.assertEqual(short_budget["expansions"], 0)

    def test_held_recipe_route_uses_legal_adjacent_stand_and_returns_to_post(self):
        current = payload(10)
        current["mapInfo"].update({
            "width": 10, "height": 10,
            "zones": [
                {"pos": {"x": x, "y": 1}, "neutralType": "stone"}
                for x in (2, 3, 4)
            ],
        })
        pioneer = current["teamOur"]["roles"][1]
        pioneer["pos"] = {"x": 1, "y": 1}
        pioneer["backpack"] = ["StarSand"]
        current["teamEnemy"]["roles"] = []
        current["robot"]["roles"] = []
        current["teamOur"]["roles"].append({
            "id": 10020, "pos": {"x": 1, "y": 4},
            "roleType": "gatling", "health": 1000, "level": 1,
            "cooldown": 0, "backpack": [],
        })
        turn = Turn.load(current)
        hero = turn.pioneers()[0]
        hypothesis = candidate(
            location={"x": 5, "y": 1},
            window={"startRound": 10, "endRound": 20},
            items=("StarSand",),
        )

        result = evaluate_treasure_candidates(
            turn, (hypothesis,), session_index=1,
            daytime_assignments={10020: (hero, (Pos(1, 3), None, 0))},
            clock=lambda: 0.0, deadline=1.0,
        )[0]

        self.assertEqual(result["route"]["status"], "feasible")
        self.assertGreater(result["route"]["outboundMoveRounds"], 0)
        self.assertGreater(result["route"]["returnMoveRounds"], 0)
        self.assertTrue(result["route"]["returnWithMargin"])
        self.assertFalse(result["actionEnabled"])

    def test_missing_recipe_cost_uses_selected_pioneer_current_cash_and_real_capacity(self):
        current = payload(5)
        current["teamOur"]["goldNum"] = 20
        current["teamOur"]["roles"][0]["backpack"] = ["StarSand", "StarSand"]
        pioneer = current["teamOur"]["roles"][1]
        pioneer["backpack"] = ["StarSand", "other-a", "other-b"]
        pioneer["backPackCapability"] = 4
        hypothesis = candidate(
            location={"x": 4, "y": 4},
            window={"startRound": 2, "endRound": 20},
            items=("StarSand", "StarSand", "StarSand"),
        )

        result = evaluate_treasure_candidates(
            Turn.load(current), (hypothesis,), session_index=1,
        )[0]

        self.assertEqual(result["missingItems"], {"StarSand": 2})
        self.assertEqual(result["procurement"], {
            "missingItemCount": 2,
            "missingCost": 30,
            "backpackFreeSlots": 1,
            "shopAvailable": True,
            "currentCashCoversMissingCost": False,
            "backpackFitsMissing": False,
            "routeScope": "missing_item_procurement",
            "spendingBudgetEvaluated": False,
            "routeEvaluated": False,
            "returnWindowEvaluated": False,
            "semanticValidationEvaluated": False,
        })
        self.assertFalse(result["localChecksPassed"])
        self.assertFalse(result["actionEnabled"])

    def test_procurement_unknown_and_already_held_cases_stay_distinct(self):
        base = candidate(
            location={"x": 4, "y": 4},
            window={"startRound": 2, "endRound": 20},
            items=("StarSand", "StarSand"),
        )
        unknown_recipe = replace(
            base,
            treasure_conditions=replace(base.treasure_conditions, items=None),
        )
        result = evaluate_treasure_candidates(
            Turn.load(payload(5)), (unknown_recipe,), session_index=1,
        )[0]["procurement"]
        self.assertIsNone(result["missingItemCount"])
        self.assertIsNone(result["missingCost"])
        self.assertEqual(result["shopAvailable"], "unknown")

        dead = payload(5)
        dead["teamOur"]["roles"][1]["health"] = 0
        result = evaluate_treasure_candidates(
            Turn.load(dead), (base,), session_index=1,
        )[0]["procurement"]
        self.assertIsNone(result["missingItemCount"])
        self.assertIsNone(result["missingCost"])

        missing_price = payload(5)
        missing_price["weaponShopList"] = []
        result = evaluate_treasure_candidates(
            Turn.load(missing_price), (base,), session_index=1,
        )[0]["procurement"]
        self.assertEqual(result["missingItemCount"], 2)
        self.assertFalse(result["shopAvailable"])
        self.assertIsNone(result["missingCost"])
        self.assertEqual(result["currentCashCoversMissingCost"], "unknown")

        partly_listed = candidate(items=("StarSand", "UnknownRelic"))
        result = evaluate_treasure_candidates(
            Turn.load(payload(5)), (partly_listed,), session_index=1,
        )[0]["procurement"]
        self.assertEqual(result["missingItemCount"], 2)
        self.assertFalse(result["shopAvailable"])
        self.assertIsNone(result["missingCost"])

        held = payload(5)
        held["weaponShopList"] = []
        held["teamOur"]["roles"][1]["backpack"] = ["StarSand", "StarSand"]
        result = evaluate_treasure_candidates(
            Turn.load(held), (base,), session_index=1,
        )[0]["procurement"]
        self.assertEqual(result["missingItemCount"], 0)
        self.assertEqual(result["missingCost"], 0)
        self.assertTrue(result["shopAvailable"])
        self.assertTrue(result["currentCashCoversMissingCost"])

        empty = replace(
            base,
            treasure_conditions=replace(
                base.treasure_conditions, items=cited(()),
            ),
        )
        result = evaluate_treasure_candidates(
            Turn.load(payload(5)), (empty,), session_index=1,
        )[0]["procurement"]
        self.assertEqual(result["missingItemCount"], 0)
        self.assertEqual(result["missingCost"], 0)

    def test_procurement_uses_current_prices_and_does_not_merge_pioneer_bags(self):
        current = payload(5)
        current["teamOur"]["roles"][1]["backpack"] = ["StarSand"]
        current["teamOur"]["roles"][1]["backPackCapability"] = None
        current["teamOur"]["roles"].append({
            "id": 10015, "pos": {"x": 4, "y": 3},
            "roleType": "pioneer", "health": 200,
            "backPackCapability": 40, "backpack": ["StarSand"],
        })
        current["weaponShopList"] = [{"name": "StarSand", "price": 23}]
        hypothesis = candidate(items=("StarSand", "StarSand", "StarSand"))

        result = evaluate_treasure_candidates(
            Turn.load(current), (hypothesis,), session_index=1,
        )[0]

        self.assertEqual(result["pioneerId"], "10011")
        self.assertEqual(result["procurement"]["missingItemCount"], 2)
        self.assertEqual(result["procurement"]["missingCost"], 46)
        self.assertIsNone(result["procurement"]["backpackFreeSlots"])
        self.assertEqual(result["procurement"]["backpackFitsMissing"], "unknown")

    def test_unstructured_and_unknown_fields_stay_blocked(self):
        # Break caught: an interpretation-only candidate is treated as actionable.
        turn = Turn.load(payload(5))
        values = (
            candidate(),
            candidate(location={"x": 4, "y": 4}),
        )

        result = evaluate_treasure_candidates(turn, values, session_index=1)

        self.assertEqual(result[0]["reasons"], ["unstructured"])
        self.assertEqual(
            result[1]["reasons"], ["missing_window", "missing_items"],
        )
        self.assertFalse(result[0]["localChecksPassed"])
        self.assertFalse(result[1]["actionEnabled"])

    def test_local_checks_use_one_living_pioneer_and_preserve_item_counts(self):
        # Break caught: worker inventory or a single copy satisfies a duplicate recipe.
        current = payload(5)
        current["teamOur"]["roles"][0]["backpack"] = ["StarSand"]
        current["teamOur"]["roles"][1]["backpack"] = ["StarSand"]
        hypothesis = candidate(
            location={"x": 4, "y": 4},
            window={"startRound": 2, "endRound": 20},
            items=("StarSand", "StarSand"),
        )

        result = evaluate_treasure_candidates(
            Turn.load(current), (hypothesis,), session_index=1,
        )[0]

        self.assertEqual(result["timeStatus"], "open")
        self.assertEqual(result["pioneerId"], "10011")
        self.assertEqual(result["missingItems"], {"StarSand": 1})
        self.assertIn("missing_items", result["reasons"])
        self.assertFalse(result["localChecksPassed"])
        self.assertFalse(result["actionEnabled"])

        current["teamOur"]["roles"][1]["backpack"].append("StarSand")
        ready = evaluate_treasure_candidates(
            Turn.load(current), (hypothesis,), session_index=1,
        )[0]
        self.assertEqual(ready["missingItems"], {})
        self.assertEqual(ready["reasons"], [])
        self.assertTrue(ready["localChecksPassed"])
        self.assertFalse(ready["actionEnabled"])
        self.assertEqual(ready["status"], "pending_validation")

    def test_world_constraints_are_reported_without_declaring_recipe_truth(self):
        # Break caught: invalid location, time, source, or item evidence is ignored.
        current = payload(1)
        current["weaponShopList"] = []
        base = candidate(
            location={"x": -1, "y": 5},
            window={"startRound": 2, "endRound": 3},
            items=("UnknownRelic",),
            missing=("meaning unresolved",),
            conflicts=("accounts conflict",),
        )
        truncated = replace(
            base,
            source_session=2,
            treasure_conditions=replace(
                base.treasure_conditions,
                location=replace(
                    base.treasure_conditions.location,
                    source_sessions=(2,),
                    source_truncated=True,
                ),
            ),
        )

        result = evaluate_treasure_candidates(
            Turn.load(current), (truncated,), session_index=1,
        )[0]

        self.assertEqual(result["timeStatus"], "future")
        self.assertEqual(result["missingItems"], {"UnknownRelic": 1})
        self.assertEqual(set(result["reasons"]), {
            "location_out_of_bounds",
            "window_future",
            "item_unavailable_or_unknown",
            "missing_items",
            "source_truncated",
            "source_session_mismatch",
            "unresolved_conditions",
            "conflicts",
        })
        self.assertLessEqual(len(result["reasons"]), 12)

    def test_distinct_hypotheses_are_alternatives_but_duplicate_evidence_is_not(self):
        # Break caught: a newer candidate silently replaces a distinct hypothesis.
        turn = Turn.load(payload(5))
        first = candidate(
            location={"x": 4, "y": 4},
            window={"startRound": 2, "endRound": 20},
            items=(),
        )
        duplicate = replace(first, request_id="news-s1-r2-duplicate")
        second = replace(
            first,
            request_id="news-s1-r3-alternative",
            treasure_conditions=replace(
                first.treasure_conditions,
                location=cited({"x": 3, "y": 4}),
            ),
        )

        duplicates = evaluate_treasure_candidates(
            turn, (first, duplicate), session_index=1,
        )
        alternatives = evaluate_treasure_candidates(
            turn, (first, duplicate, second), session_index=1,
        )

        self.assertNotIn("alternative_candidates", duplicates[0]["reasons"])
        self.assertTrue(all(
            "alternative_candidates" in item["reasons"]
            for item in alternatives
        ))

    def test_item_order_is_not_an_alternative_but_quantity_is(self):
        # Break caught: recipe tuple order is mistaken for different Counter semantics.
        current = payload(5)
        current["weaponShopList"].append({"name": "MapRelic", "price": 15})
        current["teamOur"]["roles"][1]["backpack"] = [
            "StarSand", "MapRelic", "MapRelic",
        ]
        first = candidate(
            location={"x": 4, "y": 4},
            window={"startRound": 2, "endRound": 20},
            items=("StarSand", "MapRelic"),
        )
        reversed_items = replace(
            first,
            request_id="news-s1-r2-reversed",
            treasure_conditions=replace(
                first.treasure_conditions,
                items=cited(("MapRelic", "StarSand")),
            ),
        )
        extra_item = replace(
            first,
            request_id="news-s1-r3-extra",
            treasure_conditions=replace(
                first.treasure_conditions,
                items=cited(("StarSand", "MapRelic", "MapRelic")),
            ),
        )

        same = evaluate_treasure_candidates(
            Turn.load(current), (first, reversed_items), session_index=1,
        )
        different = evaluate_treasure_candidates(
            Turn.load(current), (first, extra_item), session_index=1,
        )

        self.assertTrue(all(item["localChecksPassed"] for item in same))
        self.assertTrue(all(
            "alternative_candidates" not in item["reasons"] for item in same
        ))
        self.assertTrue(all(
            "alternative_candidates" in item["reasons"] for item in different
        ))

    def test_old_session_hypothesis_does_not_make_current_candidate_alternative(self):
        # Break caught: a stale recipe blocks an otherwise valid current-session check.
        current = payload(5)
        current_candidate = candidate(
            location={"x": 4, "y": 4},
            window={"startRound": 2, "endRound": 20},
            items=(),
        )
        stale = candidate(
            location={"x": 3, "y": 4},
            window={"startRound": 2, "endRound": 20},
            items=("OldRelic",),
            session=2,
        )

        result = evaluate_treasure_candidates(
            Turn.load(current), (current_candidate, stale), session_index=1,
        )

        self.assertTrue(result[0]["localChecksPassed"])
        self.assertNotIn("alternative_candidates", result[0]["reasons"])
        self.assertIn("source_session_mismatch", result[1]["reasons"])

    def test_partial_then_complete_candidate_is_compatible_without_backfill(self):
        # Break caught: a later compatible completion is permanently blocked by nulls.
        engine = DecisionEngine()
        first = payload(1, team_id="treasure-compatible-completion")
        engine.decide(first)
        first_request = engine.state.state.pending_news_request
        first_source = first_request.sources[0]

        def citation(source):
            return [{
                "sourceId": source.source_id,
                "excerpt": source.text,
            }]

        partial = copy.deepcopy(first)
        partial["roundNo"] = 2
        partial["llmResp"] = json.dumps({
            "requestId": first_request.request_id,
            "candidates": [{
                "type": "treasure",
                "interpretation": "Only the location is known.",
                "citations": citation(first_source),
                "missingConditions": ["Window and items remain unknown."],
                "conflicts": [],
                "treasureConditions": {
                    "location": {
                        "value": {"x": 4, "y": 4},
                        "citations": citation(first_source),
                    },
                    "window": None,
                    "items": None,
                },
            }],
        })
        partial["worldNews"]["folkLegends"] = "Second complete account."
        engine.decide(partial)
        second_request = engine.state.state.pending_news_request
        second_source = next(
            source for source in second_request.sources
            if source.evidence_role == "new_evidence"
        )

        complete = copy.deepcopy(partial)
        complete["roundNo"] = 3
        complete["llmResp"] = json.dumps({
            "requestId": second_request.request_id,
            "candidates": [{
                "type": "treasure",
                "interpretation": "The same location now has complete conditions.",
                "citations": citation(second_source),
                "missingConditions": [],
                "conflicts": [],
                "treasureConditions": {
                    "location": {
                        "value": {"x": 4, "y": 4},
                        "citations": citation(second_source),
                    },
                    "window": {
                        "value": {"startRound": 2, "endRound": 20},
                        "citations": citation(second_source),
                    },
                    "items": {
                        "value": [],
                        "citations": citation(second_source),
                    },
                },
            }],
        })
        traces = []

        engine.decide(complete, trace_sink=traces.append)

        retained = engine.state.state.news_candidates
        diagnostics = traces[-1]["treasureConditions"]["candidates"]
        self.assertEqual(len(retained), 2)
        self.assertIsNone(retained[0].treasure_conditions.window)
        self.assertIsNone(retained[0].treasure_conditions.items)
        self.assertNotIn("alternative_candidates", diagnostics[0]["reasons"])
        self.assertNotIn("alternative_candidates", diagnostics[1]["reasons"])
        self.assertTrue(diagnostics[1]["localChecksPassed"])

    def test_pioneer_presence_task_and_inventory_are_separate_constraints(self):
        # Break caught: absent/dead/task-busy pioneers collapse into recipe failure.
        hypothesis = candidate(
            location={"x": 4, "y": 4},
            window={"startRound": 2, "endRound": 20},
            items=("MapOnlyRelic",),
        )
        absent = payload(5)
        absent["teamOur"]["roles"] = [absent["teamOur"]["roles"][0]]
        dead = payload(5)
        dead["teamOur"]["roles"][1]["health"] = 0
        held = payload(5)
        held["weaponShopList"] = []
        held["teamOur"]["roles"][1]["backpack"] = ["MapOnlyRelic"]
        active = copy.deepcopy(held)
        active["phaseTask"] = "active task"

        absent_result = evaluate_treasure_candidates(
            Turn.load(absent), (hypothesis,), session_index=1,
        )[0]
        dead_result = evaluate_treasure_candidates(
            Turn.load(dead), (hypothesis,), session_index=1,
        )[0]
        held_result = evaluate_treasure_candidates(
            Turn.load(held), (hypothesis,), session_index=1,
        )[0]
        active_result = evaluate_treasure_candidates(
            Turn.load(active), (hypothesis,), session_index=1,
        )[0]

        self.assertIn("pioneer_missing", absent_result["reasons"])
        self.assertNotIn("pioneer_dead", absent_result["reasons"])
        self.assertIn("pioneer_dead", dead_result["reasons"])
        self.assertNotIn("item_unavailable_or_unknown", held_result["reasons"])
        self.assertTrue(held_result["localChecksPassed"])
        self.assertEqual(active_result["reasons"], ["active_task"])
        self.assertFalse(active_result["localChecksPassed"])

    def test_real_engine_exposes_bounded_checks_without_changing_actions(self):
        # Break caught: accepting a treasure candidate changes economy/task commands.
        candidate_engine = DecisionEngine()
        control_engine = DecisionEngine()
        first = payload(1, team_id="treasure-integration-candidate")
        control_first = payload(1, team_id="treasure-integration-control")
        candidate_engine.decide(first)
        control_engine.decide(control_first)
        pending = candidate_engine.state.state.pending_news_request
        control_pending = control_engine.state.state.pending_news_request

        returned = copy.deepcopy(first)
        returned["roundNo"] = 2
        returned["llmResp"] = response_for(pending, items=[])
        control = copy.deepcopy(control_first)
        control["roundNo"] = 2
        control["llmResp"] = json.dumps({
            "requestId": control_pending.request_id, "candidates": [],
        })
        candidate_traces = []
        control_traces = []

        actual = candidate_engine.decide(returned, trace_sink=candidate_traces.append)
        baseline = control_engine.decide(control, trace_sink=control_traces.append)

        self.assertEqual(actual["roleCommandMap"], baseline["roleCommandMap"])
        self.assertEqual(actual["executeCmd"], baseline["executeCmd"])
        self.assertFalse(any(
            command.get("action") in {"buy", "move", "summonTreasure"}
            for command in actual["roleCommandMap"].values()
        ))
        diagnostics = candidate_traces[-1]["treasureConditions"]
        self.assertEqual(diagnostics["candidateCount"], 1)
        self.assertEqual(diagnostics["candidates"][0]["timeStatus"], "open")
        self.assertTrue(diagnostics["candidates"][0]["localChecksPassed"])
        self.assertFalse(diagnostics["candidates"][0]["actionEnabled"])
        self.assertNotIn("treasureConditions", control_traces[-1])

    def test_turn_summary_does_not_expose_recipe_item_names(self):
        # Break caught: missingItems copies the inferred recipe into event=turn.
        engine = DecisionEngine()
        first = payload(1, team_id="treasure-turn-redaction")
        engine.decide(first)
        pending = engine.state.state.pending_news_request
        returned = copy.deepcopy(first)
        returned["roundNo"] = 2
        returned["llmResp"] = response_for(
            pending, items=["PrivateMapRelic", "PrivateMapRelic"],
        )
        traces = []

        response = engine.decide(returned, trace_sink=traces.append)
        turn_record = server_module.turn_log_record(
            returned, response, {}, decision_trace=traces[-1],
        )
        news_record = server_module.news_detail_log_record(
            returned, decision_trace=traces[-1],
        )
        decision = turn_record["decision"]["treasureConditions"]

        self.assertNotIn("PrivateMapRelic", json.dumps(decision))
        self.assertEqual(decision["candidates"][0]["missingItemCount"], 2)
        self.assertEqual(decision["candidates"][0]["missingItemKinds"], 1)
        self.assertNotIn("missingItems", decision["candidates"][0])
        self.assertIn("PrivateMapRelic", json.dumps(news_record))

    def test_news_response_procurement_summary_is_read_only_and_redacted(self):
        candidate_engine = DecisionEngine()
        control_engine = DecisionEngine()
        first = payload(1, team_id="treasure-price-candidate")
        first["teamOur"]["goldNum"] = 20
        first["teamOur"]["roles"][1]["backpack"] = ["PrivateMapRelic"]
        first["weaponShopList"] = [{"name": "PrivateMapRelic", "price": 17}]
        control_first = copy.deepcopy(first)
        control_first["teamOur"]["teamId"] = "treasure-price-control"
        candidate_engine.decide(first)
        control_engine.decide(control_first)
        pending = candidate_engine.state.state.pending_news_request
        control_pending = control_engine.state.state.pending_news_request
        returned = copy.deepcopy(first)
        returned["roundNo"] = 2
        returned["llmResp"] = response_for(
            pending, items=["PrivateMapRelic"] * 3,
        )
        control = copy.deepcopy(control_first)
        control["roundNo"] = 2
        control["llmResp"] = json.dumps({
            "requestId": control_pending.request_id, "candidates": [],
        })
        traces = []

        actual = candidate_engine.decide(returned, trace_sink=traces.append)
        baseline = control_engine.decide(control)

        self.assertEqual(actual["roleCommandMap"], baseline["roleCommandMap"])
        self.assertEqual(actual["executeCmd"], baseline["executeCmd"])
        summary = traces[-1]["treasureConditions"]["candidates"][0]
        self.assertFalse(summary["actionEnabled"])
        self.assertEqual(summary["procurement"]["missingItemCount"], 2)
        self.assertEqual(summary["procurement"]["missingCost"], 34)
        self.assertFalse(summary["procurement"]["currentCashCoversMissingCost"])
        self.assertNotIn("PrivateMapRelic", json.dumps(summary))
        self.assertNotIn("At (4,4)", json.dumps(summary))

    def test_news_response_route_uses_current_assignment_without_changing_actions(self):
        candidate_engine = DecisionEngine()
        control_engine = DecisionEngine()
        first, _, _, _ = simple_route_case(round_no=1)
        first["teamOur"]["teamId"] = "treasure-route-candidate"
        control_first = copy.deepcopy(first)
        control_first["teamOur"]["teamId"] = "treasure-route-control"
        candidate_engine.decide(first)
        control_engine.decide(control_first)
        pending = candidate_engine.state.state.pending_news_request
        control_pending = control_engine.state.state.pending_news_request
        returned = copy.deepcopy(first)
        returned["roundNo"] = 2
        returned["llmResp"] = response_for(
            pending, location=(2, 2), window=(2, 20), items=["StarSand"],
        )
        control = copy.deepcopy(control_first)
        control["roundNo"] = 2
        control["llmResp"] = json.dumps({
            "requestId": control_pending.request_id, "candidates": [],
        })
        traces = []

        actual = candidate_engine.decide(returned, trace_sink=traces.append)
        baseline = control_engine.decide(control)

        self.assertEqual(actual["roleCommandMap"], baseline["roleCommandMap"])
        self.assertEqual(actual["executeCmd"], baseline["executeCmd"])
        summary = traces[-1]["treasureConditions"]["candidates"][0]
        self.assertEqual(summary["route"]["status"], "feasible")
        self.assertTrue(summary["route"]["returnWithMargin"])
        self.assertFalse(summary["actionEnabled"])
        self.assertEqual(summary["status"], "pending_validation")
        self.assertEqual(
            summary["procurement"]["routeScope"],
            "missing_item_procurement",
        )
        self.assertFalse(summary["procurement"]["routeEvaluated"])
        self.assertNotIn("StarSand", json.dumps(summary))
        self.assertNotIn('"x"', json.dumps(summary))

    def test_window_does_not_slide_and_session_switch_drops_candidates(self):
        # Break caught: absolute rounds slide by day or prior-session candidates leak.
        engine = DecisionEngine()
        first = payload(1, team_id="treasure-window-session")
        engine.decide(first)
        pending = engine.state.state.pending_news_request

        accepted = copy.deepcopy(first)
        accepted["roundNo"] = 2
        accepted["llmResp"] = response_for(pending, window=(3, 4), items=[])
        future_traces = []
        engine.decide(accepted, trace_sink=future_traces.append)

        opened = copy.deepcopy(accepted)
        opened["roundNo"] = 3
        opened["llmResp"] = ""
        opened_traces = []
        engine.decide(opened, trace_sink=opened_traces.append)

        expired = copy.deepcopy(opened)
        expired["roundNo"] = 5
        expired_traces = []
        engine.decide(expired, trace_sink=expired_traces.append)

        switched = copy.deepcopy(expired)
        switched["roundNo"] = 6
        switched["teamOur"]["teamId"] = "treasure-new-session"
        switched_traces = []
        engine.decide(switched, trace_sink=switched_traces.append)

        self.assertEqual(
            future_traces[-1]["treasureConditions"]["candidates"][0]["timeStatus"],
            "future",
        )
        self.assertEqual(
            opened_traces[-1]["treasureConditions"]["candidates"][0]["timeStatus"],
            "open",
        )
        self.assertEqual(
            expired_traces[-1]["treasureConditions"]["candidates"][0]["timeStatus"],
            "expired",
        )
        self.assertNotIn("treasureConditions", switched_traces[-1])


if __name__ == "__main__":
    unittest.main()
