import copy
import concurrent.futures
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
        self.assertEqual(tuple(store.state.history), old_history)
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
