import copy
import json
import unittest
from pathlib import Path

from agent import intelligence
from agent.brain import DecisionEngine
from agent.protocol import Turn
from agent.state import MAX_NEWS_DAY_RECORDS, StateStore, request_fingerprint


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


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
