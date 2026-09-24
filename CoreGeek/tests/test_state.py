import copy
import concurrent.futures
import hashlib
import importlib
import json
import threading
import time
import unittest
from pathlib import Path

from agent import brain
from agent.protocol import Pos, Turn


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def load_state_module(test_case):
    try:
        return importlib.import_module("agent.state")
    except ModuleNotFoundError:
        test_case.fail("agent.state is missing")


class StateTests(unittest.TestCase):
    def test_wall_breach_history_tracks_position_across_rebuilt_ids_and_nights(self):
        state_module = load_state_module(self)
        store = state_module.StateStore()
        payload = load_fixture()
        payload["roundNo"] = 99
        wall = {
            "id": 10100, "pos": {"x": 5, "y": 5}, "roleType": "wall",
            "health": 1000, "attackPower": 0, "attackRange": 0,
            "backPackCapability": 0, "backpack": [], "level": 1,
            "cooldown": 0,
        }
        payload["teamOur"]["roles"].append(wall)
        for round_no, wall_id in ((99, 10100), (100, None),
                                  (130, None), (131, 20100),
                                  (229, 20100), (230, None)):
            payload["roundNo"] = round_no
            payload["teamOur"]["roles"] = [
                role for role in payload["teamOur"]["roles"]
                if role["roleType"] != "wall"
            ]
            if wall_id is not None:
                payload["teamOur"]["roles"].append({**wall, "id": wall_id})
            store.observe(Turn.load(payload), payload,
                          state_module.request_fingerprint(payload))

        self.assertEqual(store.state.wall_breaches[Pos(5, 5)], (1, 2))

    def test_wall_breach_history_does_not_infer_from_observation_gap(self):
        state_module = load_state_module(self)
        store = state_module.StateStore()
        payload = load_fixture()
        payload["roundNo"] = 99
        payload["teamOur"]["roles"].append({
            "id": 10100, "pos": {"x": 5, "y": 5}, "roleType": "wall",
            "health": 1000, "attackPower": 0, "attackRange": 0,
            "backPackCapability": 0, "backpack": [], "level": 1,
            "cooldown": 0,
        })
        store.observe(Turn.load(payload), payload,
                      state_module.request_fingerprint(payload))
        payload["roundNo"] = 101
        payload["teamOur"]["roles"].pop()
        store.observe(Turn.load(payload), payload,
                      state_module.request_fingerprint(payload))

        self.assertEqual(store.state.wall_breaches, {})

    def test_wall_damage_history_requires_contiguous_same_wall_observations(self):
        state_module = load_state_module(self)
        store = state_module.StateStore()
        payload = load_fixture()
        payload["roundNo"] = 10
        wall = {
            "id": 10100,
            "pos": {"x": 5, "y": 5},
            "roleType": "wall",
            "health": 1000,
            "attackPower": 0,
            "attackRange": 0,
            "backPackCapability": 0,
            "backpack": [],
            "level": 1,
            "cooldown": 0,
        }
        payload["teamOur"]["roles"].append(wall)
        store.observe(
            Turn.load(payload), payload, state_module.request_fingerprint(payload),
        )

        damaged = copy.deepcopy(payload)
        damaged["roundNo"] = 11
        damaged["teamOur"]["roles"][-1]["health"] = 800
        store.observe(
            Turn.load(damaged), damaged,
            state_module.request_fingerprint(damaged),
        )
        self.assertEqual(
            store.state.wall_recent_damage[Pos(5, 5)], ((11, 200),),
        )

        missing = copy.deepcopy(damaged)
        missing["roundNo"] = 12
        missing["teamOur"]["roles"] = missing["teamOur"]["roles"][:-1]
        store.observe(
            Turn.load(missing), missing,
            state_module.request_fingerprint(missing),
        )

        returned = copy.deepcopy(damaged)
        returned["roundNo"] = 13
        returned["teamOur"]["roles"][-1]["health"] = 700
        store.observe(
            Turn.load(returned), returned,
            state_module.request_fingerprint(returned),
        )
        self.assertEqual(
            store.state.wall_recent_damage[Pos(5, 5)], ((11, 200),),
        )

        rebuilt = copy.deepcopy(returned)
        rebuilt["roundNo"] = 14
        rebuilt["teamOur"]["roles"][-1]["id"] = 10101
        rebuilt["teamOur"]["roles"][-1]["health"] = 600
        store.observe(
            Turn.load(rebuilt), rebuilt,
            state_module.request_fingerprint(rebuilt),
        )
        self.assertEqual(
            store.state.wall_recent_damage[Pos(5, 5)], ((11, 200),),
        )

    def test_same_news_is_recorded_again_for_a_new_session(self):
        # Break caught: retained old-session history suppresses current evidence.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        first = load_fixture()
        first["roundNo"] = 10
        first["worldNews"] = {
            "officialNews": "same official bulletin",
            "folkLegends": "",
        }
        store.observe(
            Turn.load(first), first, state_module.request_fingerprint(first),
        )

        next_session = copy.deepcopy(first)
        next_session["roundNo"] = 1
        next_session["teamOur"]["teamId"] = "s2-news-next-session"
        store.observe(
            Turn.load(next_session),
            next_session,
            state_module.request_fingerprint(next_session),
        )

        facts = [
            fact for fact in store.state.history
            if fact.category == "officialNews"
            and fact.value == "same official bulletin"
        ]
        self.assertEqual([fact.source_session for fact in facts], [1, 2])

    def test_same_session_repeat_keeps_first_observation_across_days(self):
        # Break caught: seeing the same text later rewrites its first-seen date.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        first = load_fixture()
        first["roundNo"] = 2
        first["worldNews"] = {
            "officialNews": "unchanged bulletin",
            "folkLegends": "",
        }
        store.observe(
            Turn.load(first), first, state_module.request_fingerprint(first),
        )

        later = copy.deepcopy(first)
        later["roundNo"] = 132
        store.observe(
            Turn.load(later), later, state_module.request_fingerprint(later),
        )

        facts = [
            fact for fact in store.state.history
            if fact.category == "officialNews"
        ]
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].source_session, 1)
        self.assertEqual(facts[0].source_round, 2)
        self.assertEqual(facts[0].source_day, 1)

    def test_news_source_and_distinct_text_are_not_merged(self):
        # Break caught: equal text from two sources or changed text overwrites evidence.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        first = load_fixture()
        first["roundNo"] = 3
        first["worldNews"] = {
            "officialNews": "shared wording",
            "folkLegends": "shared wording",
        }
        store.observe(
            Turn.load(first), first, state_module.request_fingerprint(first),
        )
        changed = copy.deepcopy(first)
        changed["roundNo"] = 4
        changed["worldNews"]["officialNews"] = "changed wording"
        store.observe(
            Turn.load(changed),
            changed,
            state_module.request_fingerprint(changed),
        )

        self.assertEqual(
            [(fact.category, fact.value) for fact in store.state.history],
            [
                ("officialNews", "shared wording"),
                ("folkLegends", "shared wording"),
                ("officialNews", "changed wording"),
            ],
        )

    def test_long_news_uses_full_digest_without_unbounded_text(self):
        # Break caught: two long messages with one preview prefix collapse together.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        prefix = "x" * state_module.MAX_NEWS_TEXT_CHARS
        payload = load_fixture()
        payload["roundNo"] = 5
        payload["worldNews"] = {
            "officialNews": prefix + "a",
            "folkLegends": "",
        }
        store.observe(
            Turn.load(payload), payload, state_module.request_fingerprint(payload),
        )
        changed = copy.deepcopy(payload)
        changed["roundNo"] = 6
        changed["worldNews"]["officialNews"] = prefix + "b"
        store.observe(
            Turn.load(changed),
            changed,
            state_module.request_fingerprint(changed),
        )

        facts = [
            fact for fact in store.state.history
            if fact.category == "officialNews"
        ]
        self.assertEqual(len(facts), 2)
        self.assertTrue(all(fact.value == prefix for fact in facts))
        self.assertTrue(all(fact.value_truncated for fact in facts))
        self.assertNotEqual(facts[0].value_fingerprint, facts[1].value_fingerprint)

    def test_news_history_and_invalid_input_are_bounded(self):
        # Break caught: malformed or unique news grows retained state without limit.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        payload = load_fixture()
        payload["worldNews"] = {
            "officialNews": None,
            "folkLegends": [],
        }
        store.observe(
            Turn.load(payload), payload, state_module.request_fingerprint(payload),
        )
        self.assertEqual(store.state.history, [])

        for index in range(state_module.MAX_HISTORY_FACTS + 5):
            current = copy.deepcopy(payload)
            current["roundNo"] = index + 2
            current["worldNews"] = {
                "officialNews": f"bulletin-{index}",
                "folkLegends": "",
            }
            store.observe(
                Turn.load(current),
                current,
                state_module.request_fingerprint(current),
            )

        self.assertEqual(len(store.state.history), state_module.MAX_HISTORY_FACTS)
        self.assertEqual(store.state.history[0].value, "bulletin-5")

    def test_lock_timeout_returns_current_request_without_mutating_state(self):
        # Break caught: lock contention ignores the decision budget.
        engine_type = getattr(brain, "DecisionEngine", None)
        self.assertTrue(callable(engine_type), "DecisionEngine is missing")
        engine = engine_type(budget_seconds=0.02)
        first = load_fixture()
        engine.decide(first)
        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["teamOur"]["roles"][0]["pos"] = {"x": 3, "y": 3}

        engine._lock.acquire()
        timer = threading.Timer(0.15, engine._lock.release)
        timer.start()
        started = time.monotonic()
        try:
            response = engine.decide(second)
            elapsed = time.monotonic() - started
        finally:
            timer.join()
            if engine._lock.locked():
                engine._lock.release()

        self.assertLess(elapsed, 0.08)
        self.assertEqual(
            response["roleCommandMap"]["10010"]["targetPos"],
            [{"x": 2, "y": 2}],
        )
        self.assertEqual(engine.state.state.last_round_no, 1)

    def test_decision_engine_serializes_concurrent_duplicate_requests(self):
        # Break caught: ThreadingHTTPServer races advance one observation twice.
        engine_type = getattr(brain, "DecisionEngine", None)
        self.assertTrue(callable(engine_type), "DecisionEngine is missing")
        engine = engine_type()
        payload = load_fixture()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(
                engine.decide,
                [copy.deepcopy(payload) for _ in range(8)],
            ))

        self.assertTrue(all(response == responses[0] for response in responses))
        self.assertEqual(engine.state.state.observation_count, 1)
        self.assertEqual(len(engine.state.state.pending_actions), 1)

    def test_duplicate_request_reuses_matching_response_trace(self):
        engine = brain.DecisionEngine()
        payload = load_fixture()
        traces = []

        first = engine.decide(payload, trace_sink=traces.append)
        second = engine.decide(copy.deepcopy(payload), trace_sink=traces.append)

        self.assertEqual(first, second)
        self.assertEqual(len(traces), 2)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[0]["roundNo"], payload["roundNo"])

    def test_concurrent_duplicate_traces_do_not_mix_between_requests(self):
        engine = brain.DecisionEngine()
        payload = load_fixture()

        def decide_once(_):
            local_trace = []
            response = engine.decide(
                copy.deepcopy(payload), trace_sink=local_trace.append,
            )
            return response, local_trace

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(decide_once, range(8)))

        first_response, first_trace = results[0]
        self.assertEqual(len(first_trace), 1)
        for response, trace in results:
            self.assertEqual(response, first_response)
            self.assertEqual(trace, first_trace)

    def test_task_trace_hashes_private_command_and_result_with_known_budget(self):
        engine = brain.DecisionEngine()
        payload = load_fixture()
        payload["teamOur"]["teamId"] = "trace-task"
        payload["teamOur"]["roles"][0]["roleType"] = "pioneer"
        payload["teamOur"]["playerTasks"] = [{
            "taskType": "private type",
            "taskPosition": {"x": 1, "y": 1},
            "coldDownRounds": 0,
            "scoreReward": 10,
            "goldReward": 10,
            "isValid": True,
            "timeoutRounds": 5,
        }]
        accept_traces = []
        accepted = engine.decide(payload, trace_sink=accept_traces.append)
        self.assertEqual(
            accepted["roleCommandMap"]["10010"]["action"], "acceptTask",
        )
        self.assertNotIn(
            "private type", json.dumps(accept_traces[0], ensure_ascii=False),
        )

        active = copy.deepcopy(payload)
        active["roundNo"] = 2
        active["phaseTask"] = "private task text"
        active["lastRoundRoleActionResults"] = {"10010": True}
        engine.decide(active)

        command_text = "private command text"
        command_turn = copy.deepcopy(active)
        command_turn["roundNo"] = 3
        command_turn["llmResp"] = json.dumps({
            "kind": "command", "content": command_text,
        })
        command_traces = []
        response = engine.decide(command_turn, trace_sink=command_traces.append)
        self.assertEqual(response["executeCmd"], command_text)

        result_text = "[exitCode:0]\nprivate output text"
        result_turn = copy.deepcopy(active)
        result_turn["roundNo"] = 4
        result_turn["llmResp"] = ""
        result_turn["lastCmdResult"] = result_text
        result_traces = []
        engine.decide(result_turn, trace_sink=result_traces.append)
        trace = result_traces[0]
        encoded = json.dumps(trace, ensure_ascii=False)

        self.assertEqual(trace["taskInstanceId"], "1:1")
        self.assertEqual(trace["taskRemainingRounds"], 2)
        self.assertEqual(
            command_traces[0]["commandFingerprint"],
            hashlib.sha256(command_text.encode()).hexdigest()[:16],
        )
        self.assertEqual(
            trace["resultFingerprint"],
            hashlib.sha256(result_text.encode()).hexdigest()[:16],
        )
        self.assertIsNotNone(trace["cycleFingerprint"])
        for private in ("private task text", command_text, "private output text"):
            self.assertNotIn(private, encoded)

    def test_late_command_result_is_not_fingerprinted_as_current_task_evidence(self):
        engine = brain.DecisionEngine()
        payload = load_fixture()
        payload["phaseTask"] = "private task"
        payload["lastCmdResult"] = "late private output"
        traces = []

        engine.decide(payload, trace_sink=traces.append)

        self.assertIsNone(traces[0]["resultFingerprint"])

    def test_expired_improvement_budget_keeps_complete_basic_response(self):
        # Break caught: deadline expiry returns a partial or malformed response.
        engine_type = getattr(brain, "DecisionEngine", None)
        self.assertTrue(callable(engine_type), "DecisionEngine is missing")
        times = iter((0.0, 10.0, 10.0, 10.0))
        engine = engine_type(clock=lambda: next(times), budget_seconds=1.0)

        response = engine.decide(load_fixture())

        self.assertEqual(set(response), {"roleCommandMap", "prompt", "executeCmd"})
        self.assertEqual(
            response["roleCommandMap"]["10010"]["action"], "move"
        )

    def test_pending_task_tool_does_not_block_probe_action(self):
        # Break caught: waiting on a cross-round tool stalls unrelated role planning.
        engine_type = getattr(brain, "DecisionEngine", None)
        self.assertTrue(callable(engine_type), "DecisionEngine is missing")
        engine = engine_type()
        first = load_fixture()
        first["phaseTask"] = "active task"
        engine.decide(first)
        engine.state.state.active_task.pending_llm_round = 1

        second = copy.deepcopy(first)
        second["roundNo"] = 2
        response = engine.decide(second)

        self.assertTrue(response["roleCommandMap"])
        self.assertEqual(response["prompt"], "")
        self.assertEqual(response["executeCmd"], "")

    def test_same_round_tool_result_cannot_confirm_new_request(self):
        # Break caught: changed same-round content confirms a tool call too early.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        payload = load_fixture()
        payload["phaseTask"] = "active task"
        turn = Turn.load(payload)
        fingerprint = state_module.request_fingerprint(payload)
        store.observe(turn, payload, fingerprint)
        store.record_response(turn, fingerprint, {
            "roleCommandMap": {},
            "prompt": "solve this",
            "executeCmd": "",
        })

        changed = copy.deepcopy(payload)
        changed["llmResp"] = "too early"
        store.observe(
            Turn.load(changed),
            changed,
            state_module.request_fingerprint(changed),
        )

        self.assertEqual(store.state.active_task.tool_results, [])
        self.assertEqual(store.state.active_task.pending_llm_round, 1)
        self.assertEqual(store.state.late_tool_results, 1)

    def test_skipped_round_action_and_tool_feedback_become_unknown(self):
        # Break caught: round 3 results are guessed to confirm round 1 requests.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        first = load_fixture()
        first["phaseTask"] = "active task"
        turn = Turn.load(first)
        fingerprint = state_module.request_fingerprint(first)
        store.observe(turn, first, fingerprint)
        store.record_response(turn, fingerprint, {
            "roleCommandMap": {
                "10010": {
                    "action": "move",
                    "targetPos": [{"x": 2, "y": 1}],
                }
            },
            "prompt": "solve this",
            "executeCmd": "",
        })

        third = copy.deepcopy(first)
        third["roundNo"] = 3
        third["lastRoundRoleActionResults"] = {"10010": True}
        third["llmResp"] = "not from round 1"
        store.observe(Turn.load(third), third, state_module.request_fingerprint(third))

        self.assertEqual(store.state.pending_actions, {})
        self.assertIsNone(store.state.action_history[-1].success)
        self.assertEqual(store.state.active_task.tool_results, [])
        self.assertIsNone(store.state.active_task.pending_llm_round)
        self.assertEqual(store.state.late_tool_results, 1)

    def test_same_round_replacement_withdraws_actions_and_tool_requests(self):
        # Break caught: a changed same-round response leaves withdrawn work pending.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        payload = load_fixture()
        payload["phaseTask"] = "active task"
        turn = Turn.load(payload)
        fingerprint = state_module.request_fingerprint(payload)
        store.observe(turn, payload, fingerprint)
        store.record_response(turn, fingerprint, {
            "roleCommandMap": {
                "10010": {
                    "action": "move",
                    "targetPos": [{"x": 2, "y": 1}],
                }
            },
            "prompt": "solve this",
            "executeCmd": "pwd",
        })

        changed = copy.deepcopy(payload)
        changed["teamOur"]["roles"][0]["health"] = 219
        changed_fingerprint = state_module.request_fingerprint(changed)
        store.observe(Turn.load(changed), changed, changed_fingerprint)
        store.record_response(Turn.load(changed), changed_fingerprint, {
            "roleCommandMap": {}, "prompt": "", "executeCmd": "",
        })

        self.assertEqual(store.state.pending_actions, {})
        self.assertEqual(store.state.plans, {})
        self.assertIsNone(store.state.active_task.pending_llm_round)
        self.assertIsNone(store.state.active_task.pending_cmd_round)

    def test_identical_request_reuses_response_without_advancing_state(self):
        # Break caught: a judge retry creates a second pending action.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        payload = load_fixture()
        turn = Turn.load(payload)
        fingerprint = state_module.request_fingerprint(payload)

        first = store.observe(turn, payload, fingerprint)
        self.assertIsNone(first.cached_response)
        response = {
            "roleCommandMap": {"10010": {"action": "move"}},
            "prompt": "",
            "executeCmd": "",
        }
        store.record_response(turn, fingerprint, response)
        duplicate = store.observe(
            Turn.load(copy.deepcopy(payload)),
            copy.deepcopy(payload),
            state_module.request_fingerprint(copy.deepcopy(payload)),
        )

        self.assertEqual(duplicate.cached_response, response)
        self.assertEqual(store.state.observation_count, 1)
        self.assertEqual(len(store.state.pending_actions), 1)

    def test_same_round_changed_content_is_processed(self):
        # Break caught: cache keys only on roundNo and hides a changed observation.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        payload = load_fixture()
        fingerprint = state_module.request_fingerprint(payload)
        turn = Turn.load(payload)
        store.observe(turn, payload, fingerprint)
        store.record_response(turn, fingerprint, {
            "roleCommandMap": {}, "prompt": "", "executeCmd": "",
        })

        changed = copy.deepcopy(payload)
        changed["teamOur"]["roles"][0]["health"] = 219
        result = store.observe(
            Turn.load(changed), changed, state_module.request_fingerprint(changed)
        )

        self.assertIsNone(result.cached_response)
        self.assertEqual(store.state.observation_count, 2)

    def test_round_rewind_resets_dynamic_state_but_keeps_sourced_history(self):
        # Break caught: a suspected new half inherits roles, budget, or task state.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        payload = load_fixture()
        payload["roundNo"] = 10
        payload["phaseTask"] = "same task text"
        turn = Turn.load(payload)
        fingerprint = state_module.request_fingerprint(payload)
        store.observe(turn, payload, fingerprint)
        store.state.plans[10010] = state_module.PlanState(
            role_id=10010,
            target=Pos(4, 4),
            reason="probe",
            deadline_round=20,
            source_session=store.state.session_index,
        )
        store.state.fortification_initialized = True
        store.state.fortification_builder_id = 10010
        store.state.fortification_targets = (Pos(5, 5),)
        store.state.fortification_batch_targets = (Pos(5, 5),)
        store.state.fortification_completed.add(Pos(5, 5))
        store.state.fortification_deferred[Pos(6, 5)] = (
            "validation", "blocker", 1,
        )
        store.state.task_environment_paths.append("/sandbox/verified.txt")
        store.record_response(turn, fingerprint, {
            "roleCommandMap": {"10010": {"action": "move"}},
            "prompt": "",
            "executeCmd": "",
        })
        old_task_id = store.state.active_task.instance_id
        old_history = tuple(store.state.history)

        rewound = copy.deepcopy(payload)
        rewound["roundNo"] = 1
        result = store.observe(
            Turn.load(rewound), rewound, state_module.request_fingerprint(rewound)
        )

        self.assertEqual(result.boundary, "suspected_round_rewind")
        self.assertEqual(store.state.session_index, 2)
        self.assertEqual(store.state.plans, {})
        self.assertEqual(store.state.pending_actions, {})
        self.assertFalse(store.state.fortification_initialized)
        self.assertIsNone(store.state.fortification_builder_id)
        self.assertEqual(store.state.fortification_targets, ())
        self.assertEqual(store.state.fortification_batch_targets, ())
        self.assertEqual(store.state.fortification_completed, set())
        self.assertEqual(store.state.fortification_deferred, {})
        self.assertEqual(store.state.task_environment_paths, [])
        self.assertEqual(
            tuple(store.state.history[:len(old_history)]), old_history,
        )
        self.assertTrue(any(
            fact.source_session == 2
            for fact in store.state.history[len(old_history):]
        ))
        self.assertNotEqual(store.state.active_task.instance_id, old_task_id)

    def test_team_identity_change_starts_new_dynamic_session(self):
        # Break caught: switching sides inherits the previous side's live plan.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        payload = load_fixture()
        turn = Turn.load(payload)
        fingerprint = state_module.request_fingerprint(payload)
        store.observe(turn, payload, fingerprint)
        store.state.plans[10010] = state_module.PlanState(
            role_id=10010,
            target=Pos(4, 4),
            reason="probe",
            deadline_round=20,
            source_session=store.state.session_index,
        )
        store.record_response(turn, fingerprint, {
            "roleCommandMap": {"10010": {"action": "move"}},
            "prompt": "",
            "executeCmd": "",
        })

        switched = copy.deepcopy(payload)
        switched["teamOur"]["teamId"] = "s0-switched"
        switched["teamOur"]["type"] = "defender"
        result = store.observe(
            Turn.load(switched),
            switched,
            state_module.request_fingerprint(switched),
        )

        self.assertEqual(result.boundary, "team_identity_changed")
        self.assertEqual(store.state.session_index, 2)
        self.assertEqual(store.state.team_id, "s0-switched")
        self.assertEqual(store.state.team_type, "defender")
        self.assertEqual(store.state.plans, {})
        self.assertEqual(store.state.pending_actions, {})

    def test_failed_buy_stays_pending_history_not_inventory(self):
        # Break caught: sending or failing a buy fabricates an owned voucher.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        payload = load_fixture()
        turn = Turn.load(payload)
        fingerprint = state_module.request_fingerprint(payload)
        store.observe(turn, payload, fingerprint)
        store.record_response(turn, fingerprint, {
            "roleCommandMap": {
                "10010": {
                    "action": "buy",
                    "name": "WeaponUpgradeVoucher1",
                }
            },
            "prompt": "",
            "executeCmd": "",
        })

        following = copy.deepcopy(payload)
        following["roundNo"] = 2
        following["lastRoundRoleActionResults"] = {"10010": False}
        store.observe(
            Turn.load(following),
            following,
            state_module.request_fingerprint(following),
        )

        self.assertEqual(store.state.pending_actions, {})
        self.assertFalse(store.state.action_history[-1].success)
        self.assertNotIn(
            "WeaponUpgradeVoucher1",
            Turn.load(following).controllable()[0].backpack,
        )

    def test_dead_role_releases_plan(self):
        # Break caught: a dead role continues owning a job in the next observation.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        payload = load_fixture()
        store.observe(
            Turn.load(payload), payload, state_module.request_fingerprint(payload)
        )
        store.state.plans[10010] = state_module.PlanState(
            role_id=10010,
            target=Pos(4, 4),
            reason="probe",
            deadline_round=20,
            source_session=store.state.session_index,
        )

        dead = copy.deepcopy(payload)
        dead["roundNo"] = 2
        dead["teamOur"]["roles"][0]["health"] = 0
        store.observe(Turn.load(dead), dead, state_module.request_fingerprint(dead))

        self.assertNotIn(10010, store.state.plans)

    def test_disappeared_mine_releases_long_range_plan(self):
        # Break caught: invalid C goals survive observation and block replanning.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        payload = load_fixture()
        store.observe(
            Turn.load(payload), payload, state_module.request_fingerprint(payload)
        )
        store.state.plans[10010] = state_module.PlanState(
            role_id=10010,
            target=Pos(1, 1),
            reason="mine:stone",
            deadline_round=None,
            source_session=store.state.session_index,
        )

        changed = copy.deepcopy(payload)
        changed["roundNo"] = 2
        changed["mapInfo"]["zones"] = []
        store.observe(
            Turn.load(changed), changed, state_module.request_fingerprint(changed)
        )

        self.assertNotIn(10010, store.state.plans)

    def test_late_tool_result_does_not_attach_to_repeated_task_text(self):
        # Break caught: a result from an ended task is reused by a new same-text task.
        state_module = load_state_module(self)
        store = state_module.StateStore()
        active = load_fixture()
        active["phaseTask"] = "same task text"
        store.observe(
            Turn.load(active), active, state_module.request_fingerprint(active)
        )
        first_id = store.state.active_task.instance_id
        store.state.active_task.pending_llm_round = 1

        ended = copy.deepcopy(active)
        ended["roundNo"] = 2
        ended["phaseTask"] = ""
        ended["llmResp"] = "late result"
        store.observe(Turn.load(ended), ended, state_module.request_fingerprint(ended))
        self.assertIsNone(store.state.active_task)

        repeated = copy.deepcopy(active)
        repeated["roundNo"] = 3
        repeated["llmResp"] = ""
        store.observe(
            Turn.load(repeated),
            repeated,
            state_module.request_fingerprint(repeated),
        )

        self.assertNotEqual(store.state.active_task.instance_id, first_id)
        self.assertEqual(store.state.active_task.tool_results, [])
        self.assertEqual(store.state.late_tool_results, 1)


if __name__ == "__main__":
    unittest.main()
