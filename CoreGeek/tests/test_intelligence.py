import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path

from agent import intelligence
from agent import economy
from agent import server as server_module
from agent.brain import DecisionEngine
from agent.protocol import Pos, Turn
from agent.state import MAX_NEWS_DAY_RECORDS, StateStore, request_fingerprint
from agent.treasure import evaluate_treasure_candidates
from tests.test_economy import economy_payload, with_completed_wall_line


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"
OFFICIAL_COLLAPSE_NEWS = (
    "矿业管理局紧急通报：北部铁矿区昨夜发生严重矿井塌方事故，主巷道结构受损，"
    "部分作业面被掩埋。安全监察部门已下达通知：为保障矿工安全，矿区将于明日全面停工，"
    "进行巷道加固和主矿脉修复。矿区领班表示：'今天浅层矿面还能抢采一些，"
    "明天的全面停工不可避免。'工程队评估：类似规模的塌方事故，"
    "修复工程通常需要2天左右才能完成并恢复开采。"
)


def payload(round_no=1, *, team_id="intelligence-tests"):
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value["roundNo"] = round_no
    value["teamOur"]["teamId"] = team_id
    value["worldNews"] = {
        "officialNews": "Tomorrow the western market may close.",
        "folkLegends": "",
    }
    return value


def valid_response(pending, *, candidates=None):
    return json.dumps({
        "requestId": pending.request_id,
        "candidates": [] if candidates is None else candidates,
    })


def treasure_candidate(pending, *, conditions=None):
    source = pending.sources[0]
    candidate = {
        "type": "treasure",
        "interpretation": "A treasure hypothesis needs local validation.",
        "citations": [{
            "sourceId": source.source_id,
            "excerpt": "western market may close",
        }],
        "missingConditions": [],
        "conflicts": [],
    }
    if conditions is not None:
        candidate["treasureConditions"] = conditions
    return candidate


def cited(source, value, excerpt="western market may close"):
    return {
        "value": value,
        "citations": [{
            "sourceId": source.source_id,
            "excerpt": excerpt,
        }],
    }


def economic_reply(request, resource, effect, start, end, basis="absolute_day"):
    source = request.sources[0]
    citation = {"sourceId": source.source_id, "excerpt": source.text}
    return valid_response(request, candidates=[{
        "type": "news", "interpretation": "Economic window",
        "citations": [citation], "missingConditions": [], "conflicts": [],
        "economicEvents": [{
            "resource": resource, "effect": effect,
            "startDay": start, "endDay": end, "timeBasis": basis,
            "citations": [citation],
        }],
    }])


def collapse_reply(request):
    source = request.sources[0]
    citation = {"sourceId": source.source_id, "excerpt": source.text}
    return valid_response(request, candidates=[{
        "type": "news", "interpretation": "Iron mine supply window",
        "citations": [citation], "missingConditions": [], "conflicts": [],
        "economicEvents": [{
            "resource": "iron", "effect": effect,
            "startDay": 2, "endDay": 3,
            "timeBasis": "relative_publication", "citations": [citation],
        } for effect in ("mining_halt", "price_rise")],
    }])


def task_payload(round_no=1, *, phase=""):
    value = payload(round_no, team_id="intelligence-task-priority")
    value["mapInfo"].update({
        "width": 20,
        "height": 20,
        "zones": [{
            "pos": {"x": 4, "y": 4},
            "neutralType": "challengerTaskPoint1",
        }],
    })
    value["teamOur"]["playerTasks"] = [{
        "taskType": "自进化类1",
        "taskPosition": {"x": 4, "y": 4},
        "coldDownRounds": 0,
        "scoreReward": 50,
        "goldReward": 30,
        "isValid": True,
        "timeoutRounds": 20,
    }]
    value["teamOur"]["roles"] = [{
        "id": 10011,
        "pos": {"x": 3, "y": 3},
        "roleType": "pioneer",
        "health": 200,
        "attackPower": 0,
        "attackRange": 0,
        "backPackCapability": 40,
        "backpack": [],
    }]
    value["teamEnemy"]["roles"] = []
    value["robot"]["roles"] = []
    value["phaseTask"] = phase
    return value


class IntelligenceTests(unittest.TestCase):
    def test_official_collapse_news_changes_real_mining_and_sale_decisions(self):
        mine_first = with_completed_wall_line(economy_payload(gold=100))
        mine_first["teamOur"]["teamId"] = "official-collapse-mine"
        mine_first["mapInfo"]["zones"].append({
            "pos": {"x": 3, "y": 2}, "neutralType": "iron",
        })
        mine_first["vendorShopList"].append({"name": "iron", "price": 200})
        mine_first["weaponShopList"] = []
        mine_first["worldNews"] = {
            "officialNews": OFFICIAL_COLLAPSE_NEWS, "folkLegends": "",
        }
        mining = DecisionEngine(clock=lambda: 0.0)
        mining.decide(mine_first)
        mine_pending = mining.state.state.pending_news_request
        mine_second = copy.deepcopy(mine_first)
        mine_second["roundNo"] = 2
        mine_second["llmResp"] = collapse_reply(mine_pending)
        today = mining.decide(mine_second)["roleCommandMap"]["10010"]
        events = mining.state.state.news_candidates[-1].economic_events
        self.assertEqual([(event.effect, event.effect_basis) for event in events], [
            ("mining_halt", "quoted_news"),
            ("price_rise", "taskbook_5_1_collapse_supply"),
        ])
        self.assertEqual((today["action"], today["targetPos"]),
                         ("collect", [{"x": 3, "y": 2}]))
        mine_active = copy.deepcopy(mine_second)
        mine_active["roundNo"] = 131
        mine_active.pop("llmResp")
        halted = mining.decide(mine_active)["roleCommandMap"]["10010"]
        self.assertFalse(halted["action"] == "collect" and
                         halted.get("targetPos") == [{"x": 3, "y": 2}])

        hold_first = with_completed_wall_line(economy_payload(
            worker_pos=(3, 2), items=("iron",), gold=100,
        ))
        hold_first["teamOur"]["teamId"] = "official-collapse-hold"
        hold_first["weaponShopList"] = []
        hold_first["vendorShopList"] = [{"name": "iron", "price": 100}]
        hold_first["worldNews"] = {
            "officialNews": OFFICIAL_COLLAPSE_NEWS, "folkLegends": "",
        }
        holding = DecisionEngine(clock=lambda: 0.0)
        holding.decide(hold_first)
        hold_pending = holding.state.state.pending_news_request
        hold_second = copy.deepcopy(hold_first)
        hold_second["roundNo"] = 2
        hold_second["llmResp"] = collapse_reply(hold_pending)
        self.assertNotEqual(holding.decide(hold_second)["roleCommandMap"]["10010"]["action"],
                            "sell")
        risen = copy.deepcopy(hold_second)
        risen["roundNo"] = 131
        risen.pop("llmResp")
        risen["vendorShopList"][0]["price"] = 101
        self.assertEqual(holding.decide(risen)["roleCommandMap"]["10010"]["action"],
                         "sell")

    def test_collapse_inference_rejects_negation_missing_duration_and_other_disaster(self):
        variants = (
            "北部铁矿区矿井塌方，矿区明日不会全面停工，修复工程通常需要2天左右才能恢复开采。",
            "北部铁矿区矿井塌方，矿区明日全面停工，修复工程日期尚未确定，恢复开采待通知。",
            "北部铁矿区矿井塌方，矿区明日全面停工，修复工程通常需要2天左右才能恢复开采，但铁价不会上涨。",
            "北部铁矿区附近仓库火灾，矿区明日全面停工，修复工程通常需要2天左右才能恢复开采。",
            "北部铁矿区矿井塌方，矿区明日全面停工，修复期限未定；两天后天气转晴，何时恢复开采待通知。",
            "北部铁矿区矿井塌方，矿区明日可能全面停工，修复工程通常需要2天左右才能恢复开采。",
        )
        for index, message in enumerate(variants):
            with self.subTest(message=message):
                first = payload(team_id=f"collapse-negative-{index}")
                first["worldNews"]["officialNews"] = message
                store = StateStore()
                turn = Turn.load(first)
                store.observe(turn, first, request_fingerprint(first))
                request = store.prepare_news_request(turn)
                parsed = intelligence.parse_news_response(
                    request, collapse_reply(request),
                )
                self.assertEqual(parsed.rejection_reason, "invalid_candidate")
        uncertain = payload(team_id="collapse-possible-shutdown")
        uncertain["worldNews"]["officialNews"] = (
            "北部铁矿区矿井塌方，明日可能全面停工，修复工程通常需要2天左右才能恢复开采。"
        )
        store = StateStore()
        turn = Turn.load(uncertain)
        store.observe(turn, uncertain, request_fingerprint(uncertain))
        request = store.prepare_news_request(turn)
        parsed = intelligence.parse_news_response(
            request, economic_reply(
                request, "iron", "mining_halt", 2, 2,
                "relative_publication",
            ),
        )
        self.assertEqual(parsed.rejection_detail["code"], "uncertain_effect")

    def test_official_collapse_requires_proven_publication_and_official_source(self):
        for round_no, category, code in (
            (2, "officialNews", "publication_unproven"),
            (1, "folkLegends", "untrusted_source"),
        ):
            with self.subTest(round_no=round_no, category=category):
                first = payload(
                    round_no=round_no,
                    team_id=f"collapse-source-{round_no}-{category}",
                )
                first["worldNews"] = {
                    "officialNews": OFFICIAL_COLLAPSE_NEWS if category == "officialNews" else "",
                    "folkLegends": OFFICIAL_COLLAPSE_NEWS if category == "folkLegends" else "",
                }
                store = StateStore()
                turn = Turn.load(first)
                store.observe(turn, first, request_fingerprint(first))
                request = store.prepare_news_request(turn)
                parsed = intelligence.parse_news_response(
                    request, collapse_reply(request),
                )
                self.assertEqual(parsed.rejection_detail["code"], code)

    def test_real_decision_holds_existing_iron_until_observed_rise(self):
        first = with_completed_wall_line(economy_payload(
            worker_pos=(3, 2), items=("iron",), gold=100,
        ))
        first["teamOur"]["teamId"] = "economic-decision-hold"
        first["weaponShopList"] = []
        first["vendorShopList"] = [{"name": "iron", "price": 100}]
        first["worldNews"] = {
            "officialNews": "On game day 2 iron prices rise.",
            "folkLegends": "",
        }
        without_news = copy.deepcopy(first)
        without_news["worldNews"] = {}
        baseline = DecisionEngine(clock=lambda: 0.0).decide(without_news)
        self.assertEqual(baseline["roleCommandMap"]["10010"]["action"],
                         "sell")
        engine = DecisionEngine(clock=lambda: 0.0)
        engine.decide(first)
        pending = engine.state.state.pending_news_request
        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["llmResp"] = economic_reply(
            pending, "iron", "price_rise", 2, 2,
        )
        held = engine.decide(second)["roleCommandMap"]["10010"]
        self.assertNotIn(held["action"], ("sell", "collect"))
        self.assertEqual(engine.state.state.news_price_baselines[
            ("iron", "price_rise", 2, 2, pending.sources[0].fingerprint)
        ], 100)
        third = copy.deepcopy(second)
        third["roundNo"] = 3
        third.pop("llmResp")
        third["vendorShopList"][0]["price"] = 101
        sold = engine.decide(third)["roleCommandMap"]["10010"]
        self.assertEqual(sold["action"], "sell")
        self.assertEqual(sold["name"], "iron")
        fourth = copy.deepcopy(third)
        fourth["roundNo"] = 4
        fourth["teamOur"]["roles"][0]["backpack"] = []
        fourth["lastRoundRoleActionResults"] = {"10010": True}
        engine.decide(fourth)
        fifth = copy.deepcopy(fourth)
        fifth["roundNo"] = 5
        fifth["teamOur"]["roles"][0]["backpack"] = ["iron"]
        fifth["vendorShopList"][0]["price"] = 100
        fifth.pop("lastRoundRoleActionResults")
        self.assertEqual(engine.decide(fifth)["roleCommandMap"]["10010"]["action"],
                         "sell")

    def test_economic_hold_expires_and_full_backpack_can_sell(self):
        first = with_completed_wall_line(economy_payload(
            worker_pos=(3, 2), items=("iron",), gold=100,
        ))
        first["teamOur"]["teamId"] = "economic-hold-expiry"
        first["weaponShopList"] = []
        first["vendorShopList"] = [{"name": "iron", "price": 100}]
        first["worldNews"] = {
            "officialNews": "On game day 2 iron prices rise.",
            "folkLegends": "",
        }
        engine = DecisionEngine(clock=lambda: 0.0)
        engine.decide(first)
        pending = engine.state.state.pending_news_request
        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["llmResp"] = economic_reply(
            pending, "iron", "price_rise", 2, 2,
        )
        engine.decide(second)
        full = copy.deepcopy(second)
        full["roundNo"] = 3
        full.pop("llmResp")
        full["teamOur"]["roles"][0]["backPackCapability"] = 1
        self.assertEqual(engine.decide(full)["roleCommandMap"]["10010"]["action"],
                         "sell")
        expired = copy.deepcopy(second)
        expired["roundNo"] = 261
        expired.pop("llmResp")
        self.assertEqual(engine.decide(expired)["roleCommandMap"]["10010"]["action"],
                         "sell")

    def test_funding_deficit_sells_despite_future_price_news(self):
        first = economy_payload(
            worker_pos=(3, 2), items=("iron",) * 10, gold=0,
        )
        first["teamOur"]["teamId"] = "economic-funding-overrides-hold"
        first["vendorShopList"] = [{"name": "iron", "price": 10}]
        first["worldNews"] = {
            "officialNews": "On game day 2 iron prices rise.",
            "folkLegends": "",
        }
        engine = DecisionEngine(clock=lambda: 0.0)
        engine.decide(first)
        pending = engine.state.state.pending_news_request
        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["llmResp"] = economic_reply(
            pending, "iron", "price_rise", 2, 2,
        )
        command = engine.decide(second)["roleCommandMap"]["10010"]
        self.assertEqual(command["action"], "sell")
        self.assertTrue(engine.state.state.plans[10010].reason.startswith("fund:"))

    def test_holding_iron_still_sells_unheld_copper(self):
        first = with_completed_wall_line(economy_payload(
            worker_pos=(3, 2), items=("iron", "copper"), gold=100,
        ))
        first["teamOur"]["teamId"] = "economic-mixed-inventory"
        first["weaponShopList"] = []
        first["vendorShopList"] = [
            {"name": "iron", "price": 200},
            {"name": "copper", "price": 100},
        ]
        first["worldNews"] = {
            "officialNews": "On game day 2 iron prices rise.",
            "folkLegends": "",
        }
        engine = DecisionEngine(clock=lambda: 0.0)
        engine.decide(first)
        pending = engine.state.state.pending_news_request
        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["llmResp"] = economic_reply(
            pending, "iron", "price_rise", 2, 2,
        )
        command = engine.decide(second)["roleCommandMap"]["10010"]
        self.assertEqual((command["action"], command["name"]),
                         ("sell", "copper"))

    def test_relative_economic_window_needs_proven_publication_day(self):
        first = payload(team_id="relative-economic-window")
        first["worldNews"]["officialNews"] = (
            "Tomorrow and the day after tomorrow iron mines halt."
        )
        store = StateStore()
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        parsed = intelligence.parse_news_response(
            request, economic_reply(request, "iron", "mining_halt", 2, 3,
                                    "relative_publication"),
        )
        self.assertIsNone(parsed.rejection_reason)
        delayed = copy.deepcopy(first)
        delayed["roundNo"] = 2
        late_store = StateStore()
        late_turn = Turn.load(delayed)
        late_store.observe(late_turn, delayed, request_fingerprint(delayed))
        late_request = late_store.prepare_news_request(late_turn)
        rejected = intelligence.parse_news_response(
            late_request, economic_reply(
                late_request, "iron", "mining_halt", 2, 3,
                "relative_publication",
            ),
        )
        self.assertEqual(rejected.rejection_detail["code"],
                         "publication_unproven")

    def test_economic_event_rejects_folk_truncation_other_session_and_fake_day(self):
        first = payload(team_id="economic-trust-boundary")
        first["worldNews"]["officialNews"] = (
            "On game day 1 iron mines halt by 2 or 3 units."
        )
        store = StateStore()
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        for altered, expected in (
            (replace(request, sources=(replace(request.sources[0],
                                               truncated=True),)), "untrusted_source"),
            (replace(request, sources=(replace(request.sources[0],
                                               category="folkLegends"),)), "untrusted_source"),
            (replace(request, sources=(replace(request.sources[0],
                                               first_session=2),)), "untrusted_source"),
            (request, "time_uncited"),
        ):
            with self.subTest(expected=expected, source=altered.sources[0]):
                result = intelligence.parse_news_response(
                    altered, economic_reply(
                        altered, "iron", "mining_halt", 2, 3,
                    ),
                )
                self.assertEqual(result.rejection_detail["code"], expected)

    def test_real_decision_switches_away_from_cited_halted_mine(self):
        first = with_completed_wall_line(economy_payload(gold=100))
        first["teamOur"]["teamId"] = "economic-decision-mine"
        first["mapInfo"]["zones"].append({
            "pos": {"x": 3, "y": 2}, "neutralType": "iron",
        })
        first["vendorShopList"].append({"name": "iron", "price": 200})
        first["weaponShopList"] = []
        first["worldNews"] = {
            "officialNews": "On game day 1 iron mines halt.",
            "folkLegends": "",
        }
        engine = DecisionEngine(clock=lambda: 0.0)
        engine.decide(first)
        pending = engine.state.state.pending_news_request
        source = pending.sources[0]
        citation = {"sourceId": source.source_id, "excerpt": source.text}
        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["llmResp"] = valid_response(pending, candidates=[{
            "type": "news", "interpretation": "Iron halt",
            "citations": [citation], "missingConditions": [], "conflicts": [],
            "economicEvents": [{
                "resource": "iron", "effect": "mining_halt",
                "startDay": 1, "endDay": 1, "timeBasis": "absolute_day",
                "citations": [citation],
            }],
        }])
        response = engine.decide(second)
        command = response["roleCommandMap"]["10010"]
        self.assertFalse(command["action"] == "collect" and
                         command.get("targetPos") == [{"x": 3, "y": 2}])

    def test_future_halt_keeps_today_collect_then_blocks_active_day(self):
        first = with_completed_wall_line(economy_payload(gold=100))
        first["teamOur"]["teamId"] = "economic-future-halt"
        first["mapInfo"]["zones"] = [
            {"pos": {"x": 2, "y": 2}, "neutralType": "iron"},
            {"pos": {"x": 4, "y": 2}, "neutralType": "vendor"},
        ]
        first["vendorShopList"] = [{"name": "iron", "price": 100}]
        first["weaponShopList"] = []
        first["worldNews"] = {
            "officialNews": "On game day 2 iron mines halt.",
            "folkLegends": "",
        }
        engine = DecisionEngine(clock=lambda: 0.0)
        engine.decide(first)
        pending = engine.state.state.pending_news_request
        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["llmResp"] = economic_reply(
            pending, "iron", "mining_halt", 2, 2,
        )
        today = engine.decide(second)["roleCommandMap"]["10010"]
        self.assertEqual(today["action"], "collect")
        active = copy.deepcopy(second)
        active["roundNo"] = 131
        active.pop("llmResp")
        blocked = engine.decide(active)["roleCommandMap"]["10010"]
        self.assertNotEqual(blocked["action"], "collect")
        plan = engine.state.state.plans.get(10010)
        self.assertFalse(plan is not None and plan.reason.startswith("mine:"))

    def test_halt_checks_actual_collect_round_at_day_boundary(self):
        first = economy_payload(gold=100)
        first["teamOur"]["teamId"] = "economic-collect-boundary"
        first["mapInfo"]["zones"][0]["neutralType"] = "iron"
        first["vendorShopList"] = [{"name": "iron", "price": 100}]
        first["worldNews"] = {
            "officialNews": "On game day 2 iron mines halt.",
            "folkLegends": "",
        }
        engine = DecisionEngine(clock=lambda: 0.0)
        engine.decide(first)
        pending = engine.state.state.pending_news_request
        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["llmResp"] = economic_reply(
            pending, "iron", "mining_halt", 2, 2,
        )
        engine.decide(second)
        near = copy.deepcopy(first)
        near["roundNo"] = 130
        near_turn = Turn.load(near)
        far = copy.deepcopy(near)
        far["teamOur"]["roles"][0]["pos"] = {"x": 2, "y": 0}
        far_turn = Turn.load(far)
        context = economy.RouteSearchContext({}, state=engine.state.state)
        token = economy._ROUTE_SEARCH_CONTEXT.set(context)
        try:
            collect = economy._collect_or_move(
                near_turn, near_turn.unit(10010), Pos(2, 2),
                lambda: 0.0, 1.0, 256,
            )
            crossing = economy._collect_or_move(
                far_turn, far_turn.unit(10010), Pos(2, 2),
                lambda: 0.0, 1.0, 256,
            )
        finally:
            economy._ROUTE_SEARCH_CONTEXT.reset(token)
        self.assertEqual(collect.proposal.command["action"], "collect")
        self.assertIsNone(crossing)

    def test_conflicting_official_windows_disable_news_mining_override(self):
        first = with_completed_wall_line(economy_payload(gold=100))
        first["teamOur"]["teamId"] = "economic-conflict"
        first["mapInfo"]["zones"][0]["neutralType"] = "iron"
        first["vendorShopList"] = [{"name": "iron", "price": 100}]
        first["weaponShopList"] = []
        first["worldNews"] = {
            "officialNews": "On game day 1 iron mines halt.",
            "folkLegends": "",
        }
        engine = DecisionEngine(clock=lambda: 0.0)
        engine.decide(first)
        pending = engine.state.state.pending_news_request
        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["llmResp"] = economic_reply(
            pending, "iron", "mining_halt", 1, 1,
        )
        self.assertNotEqual(engine.decide(second)["roleCommandMap"]["10010"]["action"],
                            "collect")
        third = copy.deepcopy(first)
        third["roundNo"] = 3
        third["worldNews"]["officialNews"] = (
            "On game days 1 and 2 iron mines halt."
        )
        engine.decide(third)
        second_pending = engine.state.state.pending_news_request
        fourth = copy.deepcopy(third)
        fourth["roundNo"] = 4
        fourth["llmResp"] = economic_reply(
            second_pending, "iron", "mining_halt", 1, 2,
        )
        self.assertEqual(engine.decide(fourth)["roleCommandMap"]["10010"]["action"],
                         "collect")

    def test_official_economic_event_requires_cited_time_anchor(self):
        first = payload(team_id="economic-event")
        first["worldNews"]["officialNews"] = (
            "On game days 2 and 3, iron mines halt and iron prices rise."
        )
        store = StateStore()
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        source = request.sources[0]
        citation = {"sourceId": source.source_id, "excerpt": source.text}
        claim = {
            "type": "news", "interpretation": "Iron economy window",
            "citations": [citation], "missingConditions": [], "conflicts": [],
            "economicEvents": [{
                "resource": "iron", "effect": "mining_halt",
                "startDay": 2, "endDay": 3,
                "timeBasis": "absolute_day", "citations": [citation],
            }],
        }
        parsed = intelligence.parse_news_response(
            request, valid_response(request, candidates=[claim]),
        )
        self.assertIsNone(parsed.rejection_reason)
        self.assertEqual(parsed.candidates[0].economic_events[0].start_day, 2)

    def test_citation_prompt_explains_contiguous_source_excerpts(self):
        first = payload(team_id="intelligence-citation-prompt")
        first["worldNews"]["officialNews"] = "North ... south; gate opens."
        store = StateStore()
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        source = request.sources[0]
        prompt = request.prompt
        self.assertIn("top-level and field citations", prompt)
        self.assertIn("contiguous substring", prompt)
        self.assertIn("JSON decoding", prompt)
        self.assertIn("Do not insert ellipses or join separated passages", prompt)
        self.assertIn("separate citations", prompt)
        self.assertIn("Literal ellipses already in the source", prompt)
        self.assertIn("citation excerpts at most 512 characters", prompt)
        self.assertLessEqual(len(prompt), intelligence.MAX_NEWS_PROMPT_CHARS)

        claim = treasure_candidate(request, conditions={
            "location": cited(source, {"x": 4, "y": 5}, "North ... south"),
            "window": None, "items": None,
        })
        claim["citations"] = [
            {"sourceId": source.source_id, "excerpt": "North ... south"},
            {"sourceId": source.source_id, "excerpt": "gate opens"},
        ]
        good = intelligence.parse_news_response(
            request, valid_response(request, candidates=[claim]))
        self.assertIsNone(good.rejection_reason)
        bad_top = {**claim, "citations": [
            {"sourceId": source.source_id, "excerpt": "North ... gate opens"}]}
        self.assertEqual(intelligence.parse_news_response(
            request, valid_response(request, candidates=[bad_top])
        ).rejection_reason, "invalid_candidate")
        bad_field = {**claim, "treasureConditions": {
            **claim["treasureConditions"], "location": cited(
                source, {"x": 4, "y": 5}, "North ... gate opens")}}
        self.assertEqual(intelligence.parse_news_response(
            request, valid_response(request, candidates=[bad_field])
        ).rejection_reason, "invalid_candidate")

    def test_citation_prompt_keeps_source_selection_within_hard_limit(self):
        from types import SimpleNamespace

        sources = [SimpleNamespace(
            category=f"synthetic{i}", value="x" * 1024,
            value_fingerprint=f"{i:064x}", source_session=1,
            source_round=1, source_day=1, value_truncated=False,
        ) for i in range(8)]
        request = intelligence.build_news_request(
            sources, session=1, round_no=1, day=1,
        )
        self.assertEqual(len(request.sources), 7)
        self.assertTrue(all(len(source.text) == 1024 for source in request.sources))
        self.assertLessEqual(len(request.prompt), intelligence.MAX_NEWS_PROMPT_CHARS)

    def test_news_rejection_detail_identifies_first_invalid_candidate(self):
        # Break caught: one malformed candidate hides which field rejected the batch.
        store = StateStore()
        first = payload(team_id="intelligence-rejection-detail")
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        source = request.sources[0]
        good = treasure_candidate(request, conditions={
            "location": None, "window": None, "items": None,
        })
        cases = [
            ({"type": "treasure", "private-key": "secret"}, "candidate", "invalid_shape"),
            ({**good, "type": "private-type"}, "type", "invalid_type"),
            ({**good, "interpretation": ""}, "interpretation", "invalid_text"),
            ({**good, "missingConditions": ["", "secret"]},
             "missingConditions", "invalid_list"),
            ({**good, "citations": [{"sourceId": "private-source", "excerpt": "secret"}]},
             "citations", "unknown_source"),
            ({**good, "citations": [{"sourceId": source.source_id, "excerpt": "private-secret"}]},
             "citations", "excerpt_mismatch"),
            ({**good, "treasureConditions": {"location": {"value": {"x": "secret", "y": 1},
                "citations": cited(source, {"x": 1, "y": 1})["citations"]},
                "window": None, "items": None}},
             "treasureConditions.location", "invalid_value"),
            ({**good, "treasureConditions": {"location": None, "window": None,
                "items": cited(source, ["", "secret"])}},
             "treasureConditions.items", "invalid_value"),
            ({**good, "treasureConditions": {"location": None,
                "window": cited(source, {"startRound": 30, "endRound": 20}), "items": None}},
             "treasureConditions.window", "out_of_range"),
            ({**good, "treasureConditions": {"location": None,
                "window": {**cited(source, {"startRound": 2, "endRound": 20}),
                    "derivation": {"kind": "derived", "explanation": "secret",
                        "unresolved": [], "timeBasis": "private-basis"}}, "items": None}},
             "treasureConditions.window.derivation.timeBasis", "invalid_time_basis"),
            ({**good, "treasureConditions": {"location": None,
                "window": {**cited(source, {"startRound": 2, "endRound": 20}),
                    "derivation": {"kind": "private-kind", "explanation": "secret",
                        "unresolved": [], "timeBasis": "absolute_rounds"}}, "items": None}},
             "treasureConditions.window.derivation.kind", "invalid_kind"),
        ]
        for bad, field, code in cases:
            with self.subTest(field=field, code=code):
                result = intelligence.parse_news_response(
                    request, valid_response(request, candidates=[good, bad]),
                )
                self.assertEqual(result.rejection_reason, "invalid_candidate")
                self.assertEqual(result.rejection_detail, {
                    "candidateIndex": 1, "field": field, "code": code,
                })
                self.assertEqual(result.candidates, ())
                self.assertNotIn("secret", json.dumps(result.rejection_detail))

    def test_news_rejection_detail_reaches_next_round_news_log(self):
        # Break caught: parser detail disappears before the existing news log.
        engine = DecisionEngine()
        first = payload(team_id="intelligence-rejection-log")
        engine.decide(first)
        pending = engine.state.state.pending_news_request
        returned = copy.deepcopy(first)
        returned["roundNo"] = 2
        returned["llmResp"] = valid_response(pending, candidates=[{
            **treasure_candidate(pending),
            "treasureConditions": {"location": None,
                "window": cited(pending.sources[0], {"startRound": 9, "endRound": 2}),
                "items": None},
        }])
        traces = []
        engine.decide(returned, trace_sink=traces.append)
        event = traces[-1]["newsInterpretation"]["events"][0]
        record = server_module.news_detail_log_record(
            returned, decision_trace=traces[-1],
        )
        expected = {"candidateIndex": 0, "field": "treasureConditions.window",
                    "code": "out_of_range"}
        self.assertEqual(event["reason"], "invalid_candidate")
        self.assertEqual(event["rejectionDetail"], expected)
        self.assertEqual(record["events"][0]["rejectionDetail"], expected)
        self.assertEqual(engine.state.state.news_candidates, [])

    def test_derivation_audit_deduplicates_source_text_without_verifying_claims(self):
        store = StateStore()
        first = payload(team_id="intelligence-audit-same-text")
        first["worldNews"]["folkLegends"] = first["worldNews"]["officialNews"]
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        self.assertEqual(len(request.sources), 2)
        left, right = request.sources
        self.assertNotEqual(left.source_id, right.source_id)
        self.assertEqual(left.fingerprint, right.fingerprint)
        location = cited(left, {"x": 4, "y": 5})
        location["citations"].append({
            "sourceId": right.source_id,
            "excerpt": "western market may close",
        })
        location["derivation"] = {
            "kind": "direct", "explanation": "The model claims a direct mention.",
            "unresolved": [], "timeBasis": "not_applicable",
        }
        window = cited(left, {"startRound": 2, "endRound": 20})
        window["derivation"] = {
            "kind": "derived", "explanation": "Tomorrow is ambiguous.",
            "unresolved": ["Publication time is unknown."],
            "timeBasis": "first_observed",
        }
        items = cited(left, ["PrivateRelic"])
        items["derivation"] = {
            "kind": "unknown", "explanation": "The recipe is not established.",
            "unresolved": [], "timeBasis": "not_applicable",
        }
        parsed = intelligence.parse_news_response(
            request, valid_response(request, candidates=[treasure_candidate(
                request,
                conditions={"location": location, "window": window, "items": items},
            )]),
        )
        self.assertIsNone(parsed.rejection_reason)

        result = evaluate_treasure_candidates(
            turn, parsed.candidates, session_index=1,
        )[0]
        audit = result["evidenceAudit"]
        self.assertEqual(audit["location"]["status"], "direct_unverified")
        self.assertEqual(audit["location"]["citationCount"], 2)
        self.assertEqual(audit["location"]["distinctSourceCount"], 1)
        self.assertEqual(audit["location"]["duplicateSourceCount"], 1)
        self.assertIn("duplicate_source", audit["location"]["reasons"])
        self.assertEqual(audit["window"]["status"], "derived_unverified")
        self.assertEqual(audit["window"]["timeAnchorStatus"], "anchor_unknown")
        self.assertIn("unresolved_present", audit["window"]["reasons"])
        self.assertEqual(audit["items"]["status"], "unknown")
        self.assertFalse(result["actionEnabled"])
        self.assertNotIn("PrivateRelic", json.dumps(audit))
        self.assertNotIn("Tomorrow", json.dumps(audit))

    def test_distinct_news_texts_count_as_sources_not_independent_proof(self):
        store = StateStore()
        first = payload(team_id="intelligence-audit-distinct-text")
        first["worldNews"]["folkLegends"] = "A different market clue."
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        left, right = request.sources
        field = {
            "value": {"x": 4, "y": 5},
            "citations": [{
                "sourceId": source.source_id,
                "excerpt": source.text,
            } for source in (left, right)],
        }
        field["derivation"] = {
            "kind": "direct", "explanation": "The model claims both clues agree.",
            "unresolved": [], "timeBasis": "not_applicable",
        }
        claim = treasure_candidate(request, conditions={
            "location": field, "window": None, "items": None,
        })
        claim["citations"] = [{
            "sourceId": left.source_id,
            "excerpt": left.text,
        }]
        parsed = intelligence.parse_news_response(
            request, valid_response(request, candidates=[claim]),
        )
        self.assertIsNone(parsed.rejection_reason)
        audit = evaluate_treasure_candidates(
            turn, parsed.candidates, session_index=1,
        )[0]["evidenceAudit"]["location"]
        self.assertEqual(audit["citationCount"], 2)
        self.assertEqual(audit["distinctSourceCount"], 2)
        self.assertEqual(audit["duplicateSourceCount"], 0)
        self.assertEqual(audit["status"], "direct_unverified")


    def test_treasure_derivation_is_parsed_with_actual_source_fingerprints(self):
        store = StateStore()
        first = payload(team_id="intelligence-derivation")
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        source = request.sources[0]
        location = cited(source, {"x": 4, "y": 5})
        location["derivation"] = {
            "kind": "derived",
            "explanation": "An unverified inference from the cited clue.",
            "unresolved": ["The clue may refer to another market."],
            "timeBasis": "not_applicable",
        }

        parsed = intelligence.parse_news_response(
            request,
            valid_response(request, candidates=[treasure_candidate(
                request,
                conditions={"location": location, "window": None, "items": None},
            )]),
        )

        self.assertIsNone(parsed.rejection_reason)
        field = parsed.candidates[0].treasure_conditions.location
        self.assertEqual(field.derivation.kind, "derived")
        self.assertEqual(field.derivation.unresolved, (
            "The clue may refer to another market.",
        ))
        self.assertEqual(field.source_fingerprints, (source.fingerprint,))

    def test_treasure_conditions_are_strictly_parsed_with_field_evidence(self):
        # Break caught: treasure details are accepted as prose without field citations.
        store = StateStore()
        first = payload(team_id="intelligence-treasure-structure")
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        source = request.sources[0]
        response = valid_response(request, candidates=[treasure_candidate(
            request,
            conditions={
                "location": cited(source, {"x": 4, "y": 5}),
                "window": cited(source, {"startRound": 10, "endRound": 20}),
                "items": cited(source, ["StarSand", "StarSand"]),
            },
        )])

        parsed = intelligence.parse_news_response(request, response)

        self.assertIsNone(parsed.rejection_reason)
        conditions = parsed.candidates[0].treasure_conditions
        self.assertEqual(conditions.location.value, {"x": 4, "y": 5})
        self.assertEqual(
            conditions.window.value,
            {"startRound": 10, "endRound": 20},
        )
        self.assertEqual(conditions.items.value, ("StarSand", "StarSand"))
        self.assertEqual(conditions.location.source_sessions, (1,))
        self.assertFalse(conditions.location.source_truncated)
        self.assertEqual(parsed.candidates[0].status, "pending_validation")

    def test_legacy_candidates_and_null_or_empty_treasure_fields_remain_valid(self):
        # Break caught: extending treasure parsing rejects old six-field responses.
        store = StateStore()
        first = payload(team_id="intelligence-treasure-compatible")
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        source = request.sources[0]
        legacy = treasure_candidate(request)
        unknown = treasure_candidate(request, conditions={
            "location": None,
            "window": None,
            "items": cited(source, []),
        })

        parsed = intelligence.parse_news_response(
            request, valid_response(request, candidates=[legacy, unknown]),
        )

        self.assertIsNone(parsed.rejection_reason)
        self.assertIsNone(parsed.candidates[0].treasure_conditions)
        self.assertIsNone(parsed.candidates[1].treasure_conditions.location)
        self.assertIsNone(parsed.candidates[1].treasure_conditions.window)
        self.assertEqual(parsed.candidates[1].treasure_conditions.items.value, ())
        self.assertIsNone(parsed.candidates[1].treasure_conditions.items.derivation)
        self.assertEqual(
            parsed.candidates[1].treasure_conditions.items.source_fingerprints,
            (source.fingerprint,),
        )

    def test_treasure_conditions_reject_malformed_values_and_field_citations(self):
        # Break caught: bools, invented fields, fake quotes, or unsafe recipes survive.
        store = StateStore()
        first = payload(team_id="intelligence-treasure-invalid")
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        source = request.sources[0]
        base = {
            "location": cited(source, {"x": 4, "y": 5}),
            "window": cited(source, {"startRound": 10, "endRound": 20}),
            "items": cited(source, ["StarSand"]),
        }
        malformed = []
        cases = [
            ("unknown condition key", {**base, "route": None}),
            ("bool coordinate", {**base, "location": cited(source, {"x": True, "y": 5})}),
            ("float coordinate", {**base, "location": cited(source, {"x": 4.0, "y": 5})}),
            ("reversed window", {**base, "window": cited(source, {"startRound": 20, "endRound": 10})}),
            ("window below one", {**base, "window": cited(source, {"startRound": 0, "endRound": 10})}),
            ("window above match", {**base, "window": cited(source, {"startRound": 10, "endRound": 1301})}),
            ("too many items", {**base, "items": cited(source, ["StarSand"] * 9)}),
            ("empty item name", {**base, "items": cited(source, [""])}),
            ("long item name", {**base, "items": cited(source, ["x" * 129])}),
            ("fake field quote", {**base, "items": cited(source, ["StarSand"], "not present")}),
            ("empty field citations", {**base, "items": {"value": ["StarSand"], "citations": []}}),
            ("too many field citations", {**base, "items": {
                "value": ["StarSand"],
                "citations": cited(source, ["StarSand"])["citations"] * 9,
            }}),
            ("unknown field key", {**base, "items": {
                **cited(source, ["StarSand"]), "confidence": 1,
            }}),
            ("non-string item", {**base, "items": cited(source, [1])}),
        ]
        for name, conditions in cases:
            with self.subTest(name=name):
                result = intelligence.parse_news_response(
                    request,
                    valid_response(request, candidates=[treasure_candidate(
                        request, conditions=conditions,
                    )]),
                )
                malformed.append(result.rejection_reason)
        news_with_conditions = treasure_candidate(request, conditions=base)
        news_with_conditions["type"] = "news"
        malformed.append(intelligence.parse_news_response(
            request,
            valid_response(request, candidates=[news_with_conditions]),
        ).rejection_reason)

        self.assertEqual(malformed, ["invalid_candidate"] * len(malformed))

    def test_treasure_derivation_rejects_bad_shape_types_and_time_basis(self):
        store = StateStore()
        first = payload(team_id="intelligence-derivation-invalid")
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        source = request.sources[0]
        base = {
            "kind": "derived", "explanation": "A bounded summary.",
            "unresolved": [], "timeBasis": "not_applicable",
        }
        invalid = (
            None,
            {**base, "kind": True},
            {**base, "kind": "verified"},
            {**base, "explanation": ""},
            {**base, "explanation": "x" * 513},
            {**base, "unresolved": "unknown"},
            {**base, "unresolved": ["x"] * 9},
            {**base, "unresolved": [""]},
            {**base, "timeBasis": "first_observed"},
            {**base, "confidence": 1},
            {key: value for key, value in base.items() if key != "explanation"},
        )
        for value in invalid:
            with self.subTest(value=value):
                field = cited(source, {"x": 4, "y": 5})
                field["derivation"] = value
                response = valid_response(request, candidates=[treasure_candidate(
                    request, conditions={
                        "location": field, "window": None, "items": None,
                    },
                )])
                self.assertEqual(
                    intelligence.parse_news_response(request, response).rejection_reason,
                    "invalid_candidate",
                )
        for basis in ("not_applicable", "verified", 2, True):
            with self.subTest(window_basis=basis):
                field = cited(source, {"startRound": 2, "endRound": 20})
                field["derivation"] = {**base, "timeBasis": basis}
                response = valid_response(request, candidates=[treasure_candidate(
                    request, conditions={
                        "location": None, "window": field, "items": None,
                    },
                )])
                self.assertEqual(
                    intelligence.parse_news_response(request, response).rejection_reason,
                    "invalid_candidate",
                )

    def test_treasure_detail_retains_response_bound_field_evidence(self):
        # Break caught: accepted field evidence loses its request/session snapshot.
        engine = DecisionEngine()
        first = payload(team_id="intelligence-treasure-detail")
        first["worldNews"]["officialNews"] = (
            "western market may close" + "x" * 2_000
        )
        engine.decide(first)
        pending = engine.state.state.pending_news_request
        source = pending.sources[0]
        returned = copy.deepcopy(first)
        returned["roundNo"] = 2
        location_field = cited(source, {"x": 4, "y": 4})
        location_field["derivation"] = {
            "kind": "derived", "explanation": "A short unverified inference.",
            "unresolved": ["The market may refer to another place."],
            "timeBasis": "not_applicable",
        }
        returned["llmResp"] = valid_response(
            pending,
            candidates=[treasure_candidate(pending, conditions={
                "location": location_field,
                "window": cited(source, {"startRound": 2, "endRound": 20}),
                "items": cited(source, ["StarSand"]),
            })],
        )

        engine.decide(returned)

        detail = engine.state.state.news_events[-1]["candidates"][0]
        location = detail["treasureConditions"]["location"]
        self.assertEqual(location["value"], {"x": 4, "y": 4})
        self.assertEqual(location["sourceSessions"], [1])
        self.assertTrue(location["sourceTruncated"])
        self.assertEqual(location["citations"], [{
            "sourceId": source.source_id,
            "excerpt": "western market may close",
        }])
        self.assertEqual(location["derivation"], {
            "kind": "derived",
            "explanation": "A short unverified inference.",
            "unresolved": ["The market may refer to another place."],
            "timeBasis": "not_applicable",
        })
        self.assertEqual(detail["status"], "pending_validation")

    def test_top_level_truncated_citation_blocks_complete_field_hypothesis(self):
        # Break caught: only field citations are checked for truncation/session origin.
        engine = DecisionEngine()
        first = payload(team_id="intelligence-treasure-top-citation")
        first["worldNews"] = {
            "officialNews": "TOP EVIDENCE " + "x" * 2_000,
            "folkLegends": "FIELD EVIDENCE",
        }
        engine.decide(first)
        pending = engine.state.state.pending_news_request
        top_source = next(source for source in pending.sources if source.truncated)
        field_source = next(source for source in pending.sources if not source.truncated)

        def field(value):
            return {
                "value": value,
                "citations": [{
                    "sourceId": field_source.source_id,
                    "excerpt": "FIELD EVIDENCE",
                }],
            }

        returned = copy.deepcopy(first)
        returned["roundNo"] = 2
        returned["llmResp"] = valid_response(pending, candidates=[{
            "type": "treasure",
            "interpretation": "Top-level evidence is truncated.",
            "citations": [{
                "sourceId": top_source.source_id,
                "excerpt": "TOP EVIDENCE",
            }],
            "missingConditions": [],
            "conflicts": [],
            "treasureConditions": {
                "location": field({"x": 4, "y": 4}),
                "window": field({"startRound": 2, "endRound": 20}),
                "items": field([]),
            },
        }])
        traces = []

        engine.decide(returned, trace_sink=traces.append)

        stored = engine.state.state.news_candidates[0]
        diagnostic = traces[-1]["treasureConditions"]["candidates"][0]
        self.assertTrue(stored.citation_source_truncated)
        self.assertEqual(stored.citation_source_sessions, (1,))
        self.assertIn("source_truncated", diagnostic["reasons"])
        self.assertFalse(diagnostic["localChecksPassed"])

    def test_news_result_is_consumed_before_a_new_task_starts(self):
        # Break caught: a news reply becomes a task answer when phaseTask appears.
        engine = DecisionEngine()
        first = payload()

        issued = engine.decide(first)

        self.assertTrue(issued["prompt"])
        self.assertEqual(
            issued["roleCommandMap"]["10010"]["action"], "collect",
        )
        pending = engine.state.state.pending_news_request
        self.assertIsNotNone(pending)
        self.assertEqual(engine.state.state.news_calls_by_day, {1: 1})
        self.assertIn(pending.request_id, issued["prompt"])

        result = copy.deepcopy(first)
        result["roundNo"] = 2
        result["phaseTask"] = "Solve a different task."
        result["llmResp"] = json.dumps({
            "requestId": pending.request_id,
            "candidates": [{
                "type": "news",
                "interpretation": "The western market could be unavailable.",
                "citations": [{
                    "sourceId": pending.sources[0].source_id,
                    "excerpt": "western market may close",
                }],
                "missingConditions": ["Publication time is unknown."],
                "conflicts": [],
            }],
        })

        engine.decide(result)

        self.assertIsNone(engine.state.state.pending_news_request)
        self.assertEqual(len(engine.state.state.news_candidates), 1)
        self.assertEqual(
            engine.state.state.news_candidates[0].status,
            "pending_validation",
        )
        self.assertEqual(engine.state.state.active_task.tool_results, [])
        self.assertEqual(engine.state.state.late_tool_results, 0)

    def test_task_prompt_and_accept_both_preempt_news(self):
        # Break caught: an unrelated news prompt consumes the single task prompt slot.
        active_engine = DecisionEngine()
        active = task_payload(1, phase="Solve this task now.")

        active_response = active_engine.decide(active)

        self.assertTrue(active_response["prompt"])
        self.assertNotIn("UNTRUSTED_NEWS_DATA", active_response["prompt"])
        self.assertEqual(active_engine.state.state.news_calls_by_day, {})

        accept_engine = DecisionEngine()
        accept = task_payload(1)
        accept_response = accept_engine.decide(accept)

        self.assertEqual(
            accept_response["roleCommandMap"]["10011"]["action"],
            "acceptTask",
        )
        self.assertEqual(accept_response["prompt"], "")
        self.assertEqual(accept_engine.state.state.news_calls_by_day, {})

    def test_daily_limit_counts_only_actual_prompts_and_refreshes_for_new_news(self):
        # Break caught: cached or skipped prompts consume quota, or old news is retried.
        engine = DecisionEngine()
        first = payload()
        first_response = engine.decide(first)
        first_pending = engine.state.state.pending_news_request

        cached = engine.decide(copy.deepcopy(first))
        self.assertEqual(cached, first_response)
        self.assertEqual(engine.state.state.news_calls_by_day, {1: 1})

        same_round_changed = copy.deepcopy(first)
        same_round_changed["worldNews"]["officialNews"] = "same-round change"
        changed_response = engine.decide(same_round_changed)
        self.assertEqual(changed_response["prompt"], "")
        self.assertEqual(
            engine.state.state.pending_news_request.request_id,
            first_pending.request_id,
        )
        self.assertEqual(engine.state.state.news_calls_by_day, {1: 1})

        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["llmResp"] = valid_response(first_pending)
        second["worldNews"]["officialNews"] = "second bulletin"
        second_response = engine.decide(second)
        second_pending = engine.state.state.pending_news_request
        self.assertTrue(second_response["prompt"])
        self.assertNotEqual(second_pending.request_id, first_pending.request_id)
        self.assertEqual(engine.state.state.news_calls_by_day, {1: 2})

        third = copy.deepcopy(second)
        third["roundNo"] = 3
        third["llmResp"] = valid_response(second_pending)
        third["worldNews"]["officialNews"] = "third bulletin"
        third_response = engine.decide(third)
        self.assertEqual(third_response["prompt"], "")
        self.assertEqual(engine.state.state.news_calls_by_day, {1: 2})

        next_day = copy.deepcopy(third)
        next_day["roundNo"] = 131
        next_day["llmResp"] = ""
        next_day["worldNews"]["officialNews"] = "new day bulletin"
        next_day_response = engine.decide(next_day)
        self.assertTrue(next_day_response["prompt"])
        self.assertEqual(engine.state.state.news_calls_by_day[2], 1)

    def test_cross_day_response_is_charged_to_issue_day_and_quota_error_is_local(self):
        # Break caught: an R130 request is charged to day two or blocks every later day.
        engine = DecisionEngine()
        first = payload(130, team_id="intelligence-cross-day")
        issued = engine.decide(first)
        old_pending = engine.state.state.pending_news_request
        self.assertTrue(issued["prompt"])

        next_day = copy.deepcopy(first)
        next_day["roundNo"] = 131
        next_day["llmResp"] = ""
        next_day["errors"] = [{
            "errorCode": 5,
            "description": "daily LLM quota exceeded",
        }]
        next_day["worldNews"]["officialNews"] = "day two new bulletin"
        response = engine.decide(next_day)

        self.assertIn(1, engine.state.state.news_blocked_days)
        self.assertNotIn(2, engine.state.state.news_blocked_days)
        self.assertEqual(engine.state.state.news_calls_by_day[1], 1)
        self.assertEqual(engine.state.state.news_calls_by_day[2], 1)
        self.assertTrue(response["prompt"])
        self.assertNotEqual(
            engine.state.state.pending_news_request.request_id,
            old_pending.request_id,
        )

    def test_mismatched_late_and_cross_session_results_are_rejected(self):
        # Break caught: a stale or foreign requestId creates a current candidate.
        mismatch_engine = DecisionEngine()
        first = payload(team_id="intelligence-mismatch")
        mismatch_engine.decide(first)
        pending = mismatch_engine.state.state.pending_news_request
        mismatch = copy.deepcopy(first)
        mismatch["roundNo"] = 2
        mismatch["llmResp"] = json.dumps({
            "requestId": pending.request_id + "-wrong",
            "candidates": [],
        })
        mismatch_response = mismatch_engine.decide(mismatch)
        self.assertEqual(mismatch_response["prompt"], "")
        self.assertEqual(mismatch_engine.state.state.news_candidates, [])

        late_engine = DecisionEngine()
        late_first = payload(team_id="intelligence-late")
        late_engine.decide(late_first)
        late_pending = late_engine.state.state.pending_news_request
        late = copy.deepcopy(late_first)
        late["roundNo"] = 3
        late["llmResp"] = valid_response(late_pending)
        late_engine.decide(late)
        self.assertEqual(late_engine.state.state.news_candidates, [])
        self.assertIsNone(late_engine.state.state.pending_news_request)

        switched_engine = DecisionEngine()
        switched_first = payload(team_id="intelligence-old-session")
        switched_engine.decide(switched_first)
        switched_pending = switched_engine.state.state.pending_news_request
        switched = copy.deepcopy(switched_first)
        switched["roundNo"] = 2
        switched["teamOur"]["teamId"] = "intelligence-new-session"
        switched["llmResp"] = valid_response(switched_pending)
        switched_engine.decide(switched)
        self.assertEqual(switched_engine.state.state.news_candidates, [])
        self.assertEqual(switched_engine.state.state.session_index, 2)
        self.assertNotEqual(
            switched_engine.state.state.pending_news_request.request_id,
            switched_pending.request_id,
        )

    def test_parser_rejects_unverified_or_action_shaped_candidates(self):
        # Break caught: requestId guesses, fake quotes, or commands become facts.
        store = StateStore()
        first = payload(team_id="intelligence-parser")
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        source = request.sources[0]
        valid = {
            "type": "news",
            "interpretation": "A closure is possible.",
            "citations": [{
                "sourceId": source.source_id,
                "excerpt": "western market",
            }],
            "missingConditions": ["Publication time is unknown."],
            "conflicts": [],
        }
        self.assertIsNone(
            intelligence.parse_news_response(
                request, valid_response(request, candidates=[]),
            ).rejection_reason,
        )
        fake_quote = copy.deepcopy(valid)
        fake_quote["citations"][0]["excerpt"] = "not in the source"
        self.assertEqual(
            intelligence.parse_news_response(
                request,
                valid_response(request, candidates=[fake_quote]),
            ).rejection_reason,
            "invalid_candidate",
        )
        command = copy.deepcopy(valid)
        command["command"] = "buy everything"
        self.assertEqual(
            intelligence.parse_news_response(
                request,
                valid_response(request, candidates=[command]),
            ).rejection_reason,
            "invalid_candidate",
        )
        unknown = copy.deepcopy(valid)
        unknown["citations"][0]["sourceId"] = "unknown-source"
        self.assertEqual(
            intelligence.parse_news_response(
                request,
                valid_response(request, candidates=[unknown]),
            ).rejection_reason,
            "invalid_candidate",
        )

    def test_request_selects_at_most_eight_current_day_unattempted_sources(self):
        # Break caught: overflow evidence is marked attempted without being sent.
        store = StateStore()
        for index in range(5):
            current = payload(index + 1, team_id="intelligence-source-limit")
            current["worldNews"] = {
                "officialNews": f"official-{index}",
                "folkLegends": f"folk-{index}",
            }
            turn = Turn.load(current)
            store.observe(turn, current, request_fingerprint(current))

        request = store.prepare_news_request(turn)
        self.assertEqual(len(request.sources), 8)
        self.assertLessEqual(len(request.prompt), intelligence.MAX_NEWS_PROMPT_CHARS)
        self.assertIn("UNTRUSTED_NEWS_DATA_BEGIN", request.prompt)
        self.assertIn("publication time", request.prompt)
        store.record_news_request(request)
        self.assertEqual(len(store.state.news_attempted_source_ids), 8)
        self.assertEqual(len(store.state.history), 10)

    def test_invalid_response_is_not_retried_without_new_evidence(self):
        # Break caught: a rejected response spends the same ordinary call forever.
        engine = DecisionEngine()
        first = payload(team_id="intelligence-invalid-no-retry")
        engine.decide(first)

        rejected = copy.deepcopy(first)
        rejected["roundNo"] = 2
        rejected["llmResp"] = "not-json"
        response = engine.decide(rejected)

        self.assertEqual(response["prompt"], "")
        self.assertIsNone(engine.state.state.pending_news_request)
        self.assertEqual(engine.state.state.news_calls_by_day, {1: 1})
        self.assertEqual(
            engine.state.state.news_events[-1]["reason"], "invalid_json",
        )

    def test_quota_error_blocks_only_more_requests_from_the_issue_day(self):
        # Break caught: error 5 immediately triggers another ordinary call that day.
        engine = DecisionEngine()
        first = payload(team_id="intelligence-quota-day")
        engine.decide(first)

        failed = copy.deepcopy(first)
        failed["roundNo"] = 2
        failed["errors"] = [{"errorCode": 5, "description": "quota"}]
        failed["worldNews"]["officialNews"] = "new evidence after quota"
        response = engine.decide(failed)

        self.assertEqual(response["prompt"], "")
        self.assertIn(1, engine.state.state.news_blocked_days)
        self.assertEqual(engine.state.state.news_calls_by_day, {1: 1})

    def test_cross_day_refresh_does_not_retry_attempted_old_news(self):
        # Break caught: a daily reset treats analyzed evidence as newly publishable.
        store = StateStore()
        first = payload(1, team_id="intelligence-old-evidence")
        turn = Turn.load(first)
        store.observe(turn, first, request_fingerprint(first))
        request = store.prepare_news_request(turn)
        store.record_news_request(request)
        store.state.pending_news_request = None

        next_day = copy.deepcopy(first)
        next_day["roundNo"] = 131
        next_turn = Turn.load(next_day)
        store.observe(next_turn, next_day, request_fingerprint(next_day))

        self.assertIsNone(store.prepare_news_request(next_turn))
        self.assertEqual(store.state.news_skip_reason, "no_new_evidence")

    def test_unattempted_backlog_survives_daily_reset_with_bounded_context(self):
        # Break caught: day filtering permanently strands new evidence after quota.
        engine = DecisionEngine()
        first = payload(1, team_id="intelligence-backlog")
        first["worldNews"] = {"officialNews": "message one", "folkLegends": ""}
        engine.decide(first)
        first_pending = engine.state.state.pending_news_request

        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["llmResp"] = valid_response(first_pending)
        second["worldNews"]["officialNews"] = "message two"
        engine.decide(second)
        second_pending = engine.state.state.pending_news_request

        third = copy.deepcopy(second)
        third["roundNo"] = 3
        third["llmResp"] = valid_response(second_pending)
        third["worldNews"]["officialNews"] = "message three"
        self.assertEqual(engine.decide(third)["prompt"], "")

        next_day = copy.deepcopy(third)
        next_day["roundNo"] = 131
        next_day["llmResp"] = ""
        response = engine.decide(next_day)
        pending = engine.state.state.pending_news_request

        self.assertTrue(response["prompt"])
        self.assertEqual(engine.state.state.news_calls_by_day[2], 1)
        self.assertEqual(
            [source.evidence_role for source in pending.sources],
            ["new_evidence", "context", "context"],
        )
        self.assertEqual(pending.sources[0].text, "message three")
        self.assertIn('"evidenceRole":"new_evidence"', pending.prompt)
        self.assertIn('"evidenceRole":"context"', pending.prompt)
        event = engine.state.state.news_events[-1]
        self.assertEqual(event["newSourceIds"], [pending.sources[0].source_id])
        self.assertEqual(len(event["contextSourceIds"]), 2)

    def test_news_day_accounting_is_bounded(self):
        # Break caught: per-day call and blocked maps grow for a ten-day match.
        store = StateStore()
        for day in range(1, 5):
            current = payload(
                (day - 1) * 130 + 1,
                team_id="intelligence-day-bound",
            )
            current["worldNews"]["officialNews"] = f"day-{day} bulletin"
            turn = Turn.load(current)
            store.observe(turn, current, request_fingerprint(current))
            request = store.prepare_news_request(turn)
            self.assertIsNotNone(request)
            store.record_news_request(request)
            store.state.pending_news_request = None

        self.assertLessEqual(
            len(set(store.state.news_calls_by_day) | store.state.news_blocked_days),
            MAX_NEWS_DAY_RECORDS,
        )

    def test_task_result_is_not_consumed_as_news(self):
        # Break caught: a task prompt result is routed by JSON shape into news state.
        engine = DecisionEngine()
        active = task_payload(1, phase="Solve this task now.")
        response = engine.decide(active)
        self.assertTrue(response["prompt"])
        self.assertIsNone(engine.state.state.pending_news_request)

        returned = copy.deepcopy(active)
        returned["roundNo"] = 2
        returned["llmResp"] = json.dumps({
            "requestId": "news-s1-r1-guessed",
            "candidates": [],
        })
        engine.decide(returned)

        self.assertEqual(engine.state.state.news_candidates, [])
        self.assertTrue(engine.state.state.active_task.tool_results)

    def test_news_prompt_survives_action_fallback_and_merges_two_sources(self):
        # Break caught: final fallback erases a registered prompt or splits evidence.
        engine = DecisionEngine()
        current = payload(team_id="intelligence-fallback")
        current["worldNews"]["folkLegends"] = "A second untrusted account."
        current["teamOur"]["roles"] = []
        current["teamEnemy"]["roles"] = []
        current["robot"]["roles"] = []

        response = engine.decide(current)

        self.assertEqual(response["roleCommandMap"], {})
        self.assertTrue(response["prompt"])
        self.assertEqual(
            len(engine.state.state.pending_news_request.sources), 2,
        )
        self.assertEqual(engine.state.state.news_calls_by_day, {1: 1})

    def test_response_and_candidate_history_limits_are_enforced(self):
        # Break caught: model output or accepted history grows without a fixed cap.
        engine = DecisionEngine()
        candidate = None
        for day in range(1, 6):
            issued_payload = payload(
                (day - 1) * 130 + 1,
                team_id="intelligence-candidate-bound",
            )
            issued_payload["worldNews"]["officialNews"] = f"day-{day} bulletin"
            engine.decide(issued_payload)
            pending = engine.state.state.pending_news_request
            candidate = {
                "type": "news",
                "interpretation": f"candidate-{day}",
                "citations": [{
                    "sourceId": pending.sources[0].source_id,
                    "excerpt": f"day-{day} bulletin",
                }],
                "missingConditions": [],
                "conflicts": [],
            }
            returned = copy.deepcopy(issued_payload)
            returned["roundNo"] += 1
            returned["llmResp"] = valid_response(
                pending, candidates=[candidate] * 8,
            )
            engine.decide(returned)

        self.assertEqual(
            len(engine.state.state.news_candidates),
            intelligence.MAX_NEWS_CANDIDATES,
        )
        self.assertEqual(
            intelligence.parse_news_response(
                pending,
                "x" * (intelligence.MAX_NEWS_RESPONSE_CHARS + 1),
            ).rejection_reason,
            "response_too_long",
        )
        self.assertEqual(
            intelligence.parse_news_response(
                pending,
                valid_response(pending, candidates=[candidate] * 9),
            ).rejection_reason,
            "invalid_candidates",
        )

    def test_non_string_citation_id_is_rejected_without_breaking_engine(self):
        # Break caught: a list sourceId raises TypeError during the real R+1 path.
        engine = DecisionEngine()
        first = payload(team_id="intelligence-source-id-type")
        engine.decide(first)
        pending = engine.state.state.pending_news_request
        malformed = copy.deepcopy(first)
        malformed["roundNo"] = 2
        malformed["llmResp"] = valid_response(pending, candidates=[{
            "type": "news",
            "interpretation": "candidate",
            "citations": [{"sourceId": [], "excerpt": "western market"}],
            "missingConditions": [],
            "conflicts": [],
        }])

        response = engine.decide(malformed)

        self.assertEqual(set(response), {"roleCommandMap", "prompt", "executeCmd"})
        self.assertEqual(engine.state.state.news_candidates, [])
        self.assertEqual(
            engine.state.state.news_events[-1]["reason"], "invalid_candidate",
        )
        follow_up = copy.deepcopy(malformed)
        follow_up["roundNo"] = 3
        follow_up["llmResp"] = ""
        follow_up["worldNews"]["officialNews"] = "fresh follow-up"
        self.assertTrue(engine.decide(follow_up)["prompt"])

    def test_oversized_json_integer_is_rejected_without_breaking_engine(self):
        # Break caught: json.loads raises ValueError before envelope validation.
        engine = DecisionEngine()
        first = payload(team_id="intelligence-json-integer-limit")
        engine.decide(first)
        oversized = copy.deepcopy(first)
        oversized["roundNo"] = 2
        oversized["llmResp"] = "1" * 5_000

        response = engine.decide(oversized)

        self.assertEqual(set(response), {"roleCommandMap", "prompt", "executeCmd"})
        self.assertIsNone(engine.state.state.pending_news_request)
        self.assertEqual(
            engine.state.state.news_events[-1]["reason"], "invalid_json",
        )
        follow_up = copy.deepcopy(oversized)
        follow_up["roundNo"] = 3
        follow_up["llmResp"] = ""
        follow_up["worldNews"]["officialNews"] = "fresh after integer limit"
        self.assertTrue(engine.decide(follow_up)["prompt"])

    def test_prompt_budget_shrinks_long_escaped_batch_without_marking_overflow(self):
        # Break caught: eight individually valid sources make the whole prompt vanish.
        store = StateStore()
        for index in range(5):
            current = payload(index + 1, team_id="intelligence-prompt-budget")
            current["worldNews"] = {
                "officialNews": "\n" * 1_020 + f"official-{index}",
                "folkLegends": "\n" * 1_020 + f"folk-{index}",
            }
            turn = Turn.load(current)
            store.observe(turn, current, request_fingerprint(current))

        request = store.prepare_news_request(turn)

        self.assertIsNotNone(request)
        self.assertGreaterEqual(len(request.sources), 1)
        self.assertLess(len(request.sources), intelligence.MAX_NEWS_REQUEST_SOURCES)
        self.assertLessEqual(len(request.prompt), intelligence.MAX_NEWS_PROMPT_CHARS)
        self.assertIn("1024 characters", request.prompt)
        self.assertIn("truncated source may omit conditions", request.prompt)
        store.record_news_request(request)
        attempted = set(store.state.news_attempted_source_ids)
        self.assertEqual(
            attempted,
            {
                source.source_id
                for source in request.sources
                if source.evidence_role == "new_evidence"
            },
        )
        self.assertLess(len(attempted), len(store.state.history))


if __name__ == "__main__":
    unittest.main()
