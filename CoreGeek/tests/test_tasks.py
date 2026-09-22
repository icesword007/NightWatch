import copy
import importlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import agent.economy as economy
from agent.actions import ActionAllocator, ActionProposal
from agent.brain import DecisionEngine
from agent.protocol import Pos, Turn
from agent.state import StateStore, TaskMemory, request_fingerprint
from agent.tasks import _prior_task_experience, _record_sop_step


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def unit(unit_id, kind, x, y, *, health=200, cooldown=0):
    return {
        "id": unit_id,
        "pos": {"x": x, "y": y},
        "roleType": kind,
        "health": health,
        "attackPower": 10 if kind in ("gatling", "railgun") else 0,
        "attackRange": 0,
        "backPackCapability": 40 if kind == "pioneer" else 100,
        "backpack": [],
        "level": 1,
        "cooldown": cooldown,
    }


def task_payload(*, round_no=1, pioneer_pos=(1, 1), phase_task=""):
    payload = load_fixture()
    payload["roundNo"] = round_no
    payload["mapInfo"].update({
        "width": 20,
        "height": 20,
        "zones": [
            {
                "pos": {"x": 4, "y": 4},
                "neutralType": "challengerTaskPoint1",
            },
            {
                "pos": {"x": 7, "y": 7},
                "neutralType": "defenderTaskPoint1",
            },
        ],
    })
    payload["teamOur"].update({
        "type": "challenger",
        "teamId": "task-tests",
        "goldNum": 0,
        "totalScore": 0,
        "playerTasks": [{
            "taskType": "自进化类1",
            "taskPosition": {"x": 4, "y": 4},
            "coldDownRounds": 0,
            "scoreReward": 50,
            "goldReward": 30,
            "isValid": True,
            "timeoutRounds": 20,
        }],
        "roles": [unit(10011, "pioneer", *pioneer_pos)],
    })
    payload["teamEnemy"]["roles"] = []
    payload["robot"]["roles"] = []
    payload["phaseTask"] = phase_task
    payload["llmResp"] = ""
    payload["lastCmdResult"] = ""
    payload["errors"] = []
    payload["weaponShopList"] = []
    payload["vendorShopList"] = []
    return payload


def state_for(payload):
    store = StateStore()
    turn = Turn.load(payload)
    store.observe(turn, payload, request_fingerprint(payload))
    return store.state


class TaskTests(unittest.TestCase):
    def test_explicit_file_task_starts_with_one_bounded_read_command(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1,
            pioneer_pos=(3, 3),
            phase_task="请阅读task_3_gamma.md并回答问题",
        )

        first = engine.decide(active)

        self.assertEqual(first["prompt"], "")
        self.assertIn("python3", first["executeCmd"])
        self.assertIn("task_3_gamma.md", first["executeCmd"])
        self.assertLessEqual(len(first["executeCmd"]), 4_096)
        task = engine.state.state.active_task
        self.assertTrue(task.entry_read_attempted)
        self.assertEqual(task.command_count, 1)
        self.assertEqual(task.pending_cmd_round, 1)

        result = copy.deepcopy(active)
        result["roundNo"] = 2
        result["lastCmdResult"] = (
            "[exitCode:0]\n[TASK_INPUT_PATH]\n/tmp/task_3_gamma.md\n"
            "[TASK_INPUT_CONTENT]\nverified instructions"
        )
        followup = engine.decide(result)
        self.assertEqual(followup["executeCmd"], "")
        self.assertIn("verified instructions", followup["prompt"])
        self.assertIn("untrusted task material", followup["prompt"])

    def test_real_single_file_suffix_starts_bounded_read_and_injects_content(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1,
            pioneer_pos=(3, 3),
            phase_task="\n请阅读task_1_alpha.md，获取任务信息\n",
        )

        first = engine.decide(active)

        self.assertEqual(first["prompt"], "")
        self.assertIn("task_1_alpha.md", first["executeCmd"])
        result = copy.deepcopy(active)
        result["roundNo"] = 2
        result["lastCmdResult"] = (
            "[exitCode:0]\n[TASK_INPUT_PATH]\n"
            "/tmp/selfEvolutionTask/example/task_1_alpha.md\n"
            "[TASK_INPUT_CONTENT]\nverified real-format instructions"
        )

        followup = engine.decide(result)

        self.assertEqual(followup["executeCmd"], "")
        self.assertIn("verified real-format instructions", followup["prompt"])

    def test_entry_read_falls_back_when_unsafe_or_deadline_is_short(self):
        for text in (
            "比较a.md和b.md",
            "请阅读a.md并打开b.md",
            "分析工程并给出答案，参考可能在task.md",
            "readme.md",
            "openapi.md",
        ):
            with self.subTest(text=text):
                response = DecisionEngine().decide(task_payload(
                    round_no=1, pioneer_pos=(3, 3), phase_task=text,
                ))
                self.assertEqual(response["executeCmd"], "")
                self.assertTrue(response["prompt"])

        tasks = importlib.import_module("agent.tasks")
        short = task_payload(
            round_no=2,
            pioneer_pos=(3, 3),
            phase_task="请阅读task.md并回答",
        )
        state = state_for(short)
        state.active_task.timeout_round = 3
        proposal = tasks.propose_tasks(
            Turn.load(short),
            state,
            deadline=100.0,
            max_expansions=10,
        )
        self.assertEqual(proposal.execute_cmd, "")
        self.assertTrue(proposal.prompt)

    def test_final_only_command_gets_exactly_one_correction_without_execution(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="solve from evidence",
        )
        engine.decide(active)
        task = engine.state.state.active_task
        task.timeout_round = 4
        task.final_answer_requested = True

        violation = copy.deepcopy(active)
        violation["roundNo"] = 2
        violation["llmResp"] = '{"kind":"command","content":"unsafe-step"}'
        corrected = engine.decide(violation)

        self.assertEqual(corrected["executeCmd"], "")
        self.assertTrue(corrected["prompt"])
        self.assertIn("final-answer-only correction", corrected["prompt"])
        self.assertTrue(task.final_only_correction_requested)
        self.assertIsNone(task.solver_stopped_reason)

        repeated = copy.deepcopy(active)
        repeated["roundNo"] = 3
        repeated["llmResp"] = '{"kind":"command","content":"unsafe-again"}'
        left = engine.decide(repeated)
        self.assertEqual(left["executeCmd"], "")
        self.assertEqual(task.solver_stopped_reason, "command_after_final_request")

    def test_remaining_two_correction_allows_partial_submit_at_remaining_one(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="solve from evidence",
        )
        engine.decide(active)
        task = engine.state.state.active_task
        task.timeout_round = 4
        task.final_answer_requested = True
        violation = copy.deepcopy(active)
        violation["roundNo"] = 2
        violation["llmResp"] = '{"kind":"command","content":"too-late"}'

        correction = engine.decide(violation)

        self.assertIn("Known remaining task rounds: 2", correction["prompt"])
        self.assertEqual(correction["executeCmd"], "")
        answer = copy.deepcopy(active)
        answer["roundNo"] = 3
        answer["llmResp"] = (
            '{"kind":"answer","content":"verified partial","complete":false}'
        )
        submitted = engine.decide(answer)
        self.assertEqual(submitted["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": "verified partial",
        })
        self.assertIsNone(task.solver_stopped_reason)

    def test_coordination_deadline_is_not_extended_by_final_correction(self):
        engine = DecisionEngine()
        active = self._third_post_payload(59)
        first = engine.decide(active)
        self.assertIn("Known remaining task rounds: 3", first["prompt"])
        task = engine.state.state.active_task
        self.assertEqual(task.coordination_deadline_round, 62)
        violation = self._third_post_payload(60)
        violation["llmResp"] = '{"kind":"command","content":"too-late"}'

        response = engine.decide(violation)

        self.assertTrue(response["prompt"])
        self.assertEqual(response["executeCmd"], "")
        self.assertIn("Known remaining task rounds: 2", response["prompt"])
        self.assertEqual(task.coordination_deadline_round, 62)
        self.assertTrue(task.coordination_final_requested)
        self.assertFalse(task.final_answer_requested)

    def test_final_only_correction_preserves_partial_answer_and_abandon(self):
        for envelope, expected_action, expected_reason in (
            (
                '{"kind":"answer","content":"verified partial","complete":false}',
                "submitAnswer",
                None,
            ),
            (
                '{"kind":"abandon","reason":"insufficient evidence"}',
                "move",
                "solver_abandoned",
            ),
        ):
            with self.subTest(envelope=envelope):
                engine = DecisionEngine()
                active = task_payload(
                    round_no=1,
                    pioneer_pos=(3, 3),
                    phase_task="solve from evidence",
                )
                engine.decide(active)
                task = engine.state.state.active_task
                task.final_answer_requested = True
                task.final_only_correction_requested = True
                response_payload = copy.deepcopy(active)
                response_payload["roundNo"] = 2
                response_payload["llmResp"] = envelope

                response = engine.decide(response_payload)

                self.assertEqual(
                    response["roleCommandMap"]["10011"]["action"],
                    expected_action,
                )
                self.assertEqual(task.solver_stopped_reason, expected_reason)

    def test_final_only_violation_with_one_round_left_gets_one_correction(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="solve from evidence",
        )
        engine.decide(active)
        task = engine.state.state.active_task
        task.timeout_round = 3
        task.final_answer_requested = True
        violation = copy.deepcopy(active)
        violation["roundNo"] = 2
        violation["llmResp"] = '{"kind":"command","content":"too-late"}'

        response = engine.decide(violation)

        self.assertEqual(response["executeCmd"], "")
        self.assertTrue(task.final_only_correction_requested)
        self.assertTrue(response["prompt"])
        self.assertIsNone(task.solver_stopped_reason)
        final = copy.deepcopy(active)
        final["roundNo"] = 3
        final["llmResp"] = (
            '{"kind":"answer","content":"verified partial","complete":false}'
        )
        self.assertEqual(engine.decide(final)["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": "verified partial",
        })

    def test_automatic_read_duplicate_request_and_new_task_are_isolated(self):
        engine = DecisionEngine()
        first_task = task_payload(
            round_no=1,
            pioneer_pos=(3, 3),
            phase_task="请阅读first.md并回答问题",
        )
        first = engine.decide(first_task)
        first_memory = engine.state.state.active_task
        duplicate = engine.decide(copy.deepcopy(first_task))

        self.assertEqual(duplicate, first)
        self.assertEqual(first_memory.command_count, 1)
        self.assertEqual(len(first_memory.solver_history), 1)

        replacement = copy.deepcopy(first_task)
        replacement["roundNo"] = 2
        replacement["phaseTask"] = "请阅读second.md并回答问题"
        replacement["lastCmdResult"] = (
            "[exitCode:0]\n[TASK_INPUT_PATH]\n/tmp/first.md\n"
            "[TASK_INPUT_CONTENT]\nstale first task content"
        )
        second = engine.decide(replacement)
        second_memory = engine.state.state.active_task

        self.assertTrue(second["executeCmd"])
        self.assertIn("second.md", second["executeCmd"])
        self.assertNotEqual(second_memory.instance_id, first_memory.instance_id)
        self.assertEqual(second_memory.command_count, 1)
        self.assertEqual(second_memory.solver_evidence, [])
        self.assertNotIn(
            "stale first task content", "\n".join(second_memory.solver_history),
        )
        self.assertEqual(engine.state.state.late_tool_results, 1)

    def test_failed_automatic_read_is_not_verified_or_repeated(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1,
            pioneer_pos=(3, 3),
            phase_task="请阅读missing.md并回答",
        )
        first = engine.decide(active)
        self.assertTrue(first["executeCmd"])

        failed = copy.deepcopy(active)
        failed["roundNo"] = 2
        failed["lastCmdResult"] = (
            "[exitCode:2]\n[TASK_INPUT_STATUS:error] not_found"
        )
        followup = engine.decide(failed)

        task = engine.state.state.active_task
        self.assertEqual(followup["executeCmd"], "")
        self.assertTrue(followup["prompt"])
        self.assertEqual(task.solver_evidence, [])
        self.assertEqual(task.command_count, 1)
        self.assertTrue(task.entry_read_attempted)
        self.assertIn("not_found", followup["prompt"])

        invalid = copy.deepcopy(active)
        invalid["roundNo"] = 3
        invalid["llmResp"] = "invalid"
        retried = engine.decide(invalid)
        self.assertEqual(retried["executeCmd"], "")

    def test_task_trace_exposes_only_bounded_entry_and_correction_flags(self):
        traces = []
        engine = DecisionEngine()
        active = task_payload(
            round_no=1,
            pioneer_pos=(3, 3),
            phase_task="请阅读task.md并回答",
        )

        engine.decide(active, trace_sink=traces.append)

        self.assertTrue(traces[-1]["taskEntryReadAttempted"])
        self.assertFalse(traces[-1]["taskFinalOnlyCorrectionRequested"])
        self.assertNotIn("task.md", json.dumps(traces[-1]))

    def test_dead_task_owner_does_not_revive_old_instance_next_day(self):
        payload = task_payload(round_no=69, pioneer_pos=(3, 3))
        payload["teamOur"]["teamId"] = "s2-dead-task-owner-next-day"
        payload["worldNews"] = {}
        engine = DecisionEngine()

        accepted = engine.decide(payload)
        self.assertEqual(
            accepted["roleCommandMap"]["10011"]["action"], "acceptTask",
        )
        active = copy.deepcopy(payload)
        active["roundNo"] = 70
        active["phaseTask"] = "first-day task"
        active["lastRoundRoleActionResults"] = {"10011": True}
        self.assertTrue(engine.decide(active)["prompt"])
        old_instance = engine.state.state.active_task.instance_id

        night = copy.deepcopy(active)
        night["roundNo"] = 71
        night["phaseTask"] = ""
        night["llmResp"] = '{"kind":"answer","content":"late"}'
        night["teamOur"]["roles"] = []
        night["worldNews"] = {}
        response = engine.decide(night)
        self.assertEqual(response["prompt"], "")
        self.assertIsNone(engine.state.state.active_task)
        self.assertEqual(
            engine.state.state.ended_tasks[-1].instance_id, old_instance,
        )
        self.assertEqual(engine.state.state.ended_tasks[-1].end_reason, "death")

        dawn = copy.deepcopy(payload)
        dawn["roundNo"] = 131
        dawn["llmResp"] = ""
        dawn["lastRoundRoleActionResults"] = {}
        next_task = engine.decide(dawn)

        self.assertEqual(
            next_task["roleCommandMap"]["10011"]["action"], "acceptTask",
        )
        self.assertIsNone(engine.state.state.active_task)

    def _idle_third_gunner_payload(self, round_no=65, *, rocket_cooldown=0):
        payload = task_payload(round_no=round_no, pioneer_pos=(12, 5))
        payload["teamOur"]["teamId"] = "task-idle-third-gunner"
        payload["teamOur"]["roles"].extend([
            unit(10010, "worker", 6, 5),
            unit(10012, "worker", 9, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 9, 6, health=1000),
            unit(
                10040, "rocket", 12, 6,
                health=1000, cooldown=rocket_cooldown,
            ),
        ])
        return payload

    def test_required_third_gunner_releases_only_after_safe_night_observation(self):
        # Daytime and the spawn frame keep the post; a complete later empty
        # observation releases the pioneer into the existing task route.
        engine = DecisionEngine()
        payload = self._idle_third_gunner_payload()
        traces = []

        for round_no in range(65, 75):
            payload["roundNo"] = round_no
            response = engine.decide(payload, trace_sink=traces.append)
            if round_no <= 71:
                self.assertNotIn("10011", response["roleCommandMap"])
                self.assertEqual(
                    traces[-1]["coordinationReason"], "gunner_hold",
                )
            else:
                self.assertEqual(
                    response["roleCommandMap"]["10011"]["action"], "move",
                )
                self.assertTrue(traces[-1]["nightClearance"]["released"])
            for role_id, command in response["roleCommandMap"].items():
                if command["action"] != "move":
                    continue
                role = next(
                    entry for entry in payload["teamOur"]["roles"]
                    if entry["id"] == int(role_id)
                )
                role["pos"] = copy.deepcopy(command["targetPos"][0])
            payload["lastRoundRoleActionResults"] = {
                role_id: True for role_id in response["roleCommandMap"]
            }

        pioneer = next(
            role for role in payload["teamOur"]["roles"]
            if role["id"] == 10011
        )
        self.assertEqual(pioneer["pos"], {"x": 9, "y": 2})

    def test_required_third_gunner_attacks_when_target_appears(self):
        payload = self._idle_third_gunner_payload(round_no=71)
        payload["robot"]["roles"] = [{
            "id": 30001,
            "pos": {"x": 12, "y": 8},
            "roleType": "smallRobot",
            "health": 40,
            "abnormalState": "",
            "targetTeam": "challenger",
        }]

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10040"], {
            "action": "attack",
            "controllerId": "10011",
            "targetPos": [{"x": 12, "y": 8}],
        })

    def test_cooling_required_third_gunner_stays_at_post(self):
        payload = self._idle_third_gunner_payload(
            round_no=71, rocket_cooldown=2,
        )

        response = DecisionEngine().decide(payload)

        self.assertNotIn("10011", response["roleCommandMap"])

    def test_required_third_gunner_uses_medicine_instead_of_leaving_for_task(self):
        payload = self._idle_third_gunner_payload(round_no=71)
        pioneer = next(
            role for role in payload["teamOur"]["roles"]
            if role["id"] == 10011
        )
        pioneer["health"] = 40
        pioneer["backpack"] = ["Medicine"]

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10011"], {
            "action": "use", "name": "Medicine",
        })

    def test_night_gunner_immediately_uses_held_upgrade_voucher(self):
        # Break caught: protected-gunner filtering hides a legal held investment.
        payload = self._idle_third_gunner_payload(round_no=71)
        worker = next(
            role for role in payload["teamOur"]["roles"]
            if role["id"] == 10010
        )
        worker["backpack"] = ["WeaponUpgradeVoucher1"]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "use",
            "name": "WeaponUpgradeVoucher1",
            "targetPos": [{"x": 6, "y": 6}],
        })

    def test_immediate_held_voucher_is_not_hidden_by_stale_single_fund_plan(self):
        # Break caught: an obsolete return post hides a currently legal use.
        payload = self._idle_third_gunner_payload(round_no=70)
        worker = next(
            role for role in payload["teamOur"]["roles"]
            if role["id"] == 10010
        )
        worker["backpack"] = ["WeaponUpgradeVoucher1"]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        engine = DecisionEngine()
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.set_plan(
            10010,
            Pos(6, 6),
            "fund:WeaponUpgradeVoucher1:10030:10020",
            70,
        )

        response = engine.decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "use",
            "name": "WeaponUpgradeVoucher1",
            "targetPos": [{"x": 6, "y": 6}],
        })

    def test_night_gunner_does_not_chase_remote_held_upgrade_voucher(self):
        payload = self._idle_third_gunner_payload(round_no=71)
        worker = next(
            role for role in payload["teamOur"]["roles"]
            if role["id"] == 10010
        )
        worker["pos"] = {"x": 1, "y": 1}
        worker["backpack"] = ["WeaponUpgradeVoucher1"]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]

        engine = DecisionEngine()
        traces = []
        response = engine.decide(payload, trace_sink=traces.append)

        plan = engine.state.state.plans.get(10010)
        self.assertFalse(plan is not None and plan.reason.startswith("use:"))
        self.assertNotEqual(
            response["roleCommandMap"].get("10010", {}).get("name"),
            "WeaponUpgradeVoucher1",
        )
        self.assertEqual(
            traces[0]["economyPlanning"]["heldInvestment"], "pending",
        )

    def test_remote_held_voucher_resumes_upgrade_route_at_next_dawn(self):
        payload = self._idle_third_gunner_payload(round_no=71)
        payload["teamOur"]["teamId"] = "s2-held-voucher-next-dawn"
        worker = next(
            role for role in payload["teamOur"]["roles"]
            if role["id"] == 10010
        )
        worker["pos"] = {"x": 1, "y": 1}
        worker["backpack"] = ["WeaponUpgradeVoucher1"]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        engine = DecisionEngine()

        night_response = engine.decide(payload)
        self.assertNotEqual(
            night_response["roleCommandMap"].get("10010", {}).get("name"),
            "WeaponUpgradeVoucher1",
        )

        dawn = copy.deepcopy(payload)
        dawn["roundNo"] = 131
        dawn_response = engine.decide(dawn)

        self.assertEqual(
            dawn_response["roleCommandMap"]["10010"]["action"], "move",
        )
        self.assertTrue(
            engine.state.state.plans[10010].reason.startswith("use:"),
        )

    def test_held_voucher_arbitration_rejection_is_diagnosed(self):
        payload = self._idle_third_gunner_payload(round_no=60)
        worker = next(
            role for role in payload["teamOur"]["roles"]
            if role["id"] == 10010
        )
        worker["pos"] = {"x": 4, "y": 5}
        worker["backpack"] = ["StationUpgradeVoucher1"]
        payload["weaponShopList"] = [
            {"name": "StationUpgradeVoucher1", "price": 100},
        ]
        traces = []

        engine = DecisionEngine()
        response = engine.decide(payload, trace_sink=traces.append)

        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")
        self.assertTrue(
            engine.state.state.plans[10010].reason.startswith("gunner:"),
        )
        self.assertEqual(
            traces[0]["economyPlanning"]["heldInvestment"], "move",
        )
        self.assertGreaterEqual(
            traces[0]["economyPlanning"]["rejectedActions"], 1,
        )

    def test_task_defense_return_coexists_with_held_voucher_use(self):
        payload = self._third_post_payload(62)
        worker = next(
            role for role in payload["teamOur"]["roles"]
            if role["id"] == 10010
        )
        worker["pos"] = {"x": 7, "y": 10}
        worker["backpack"] = ["StationUpgradeVoucher1"]
        payload["weaponShopList"] = [
            {"name": "StationUpgradeVoucher1", "price": 100},
        ]
        traces = []

        response = DecisionEngine().decide(payload, trace_sink=traces.append)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "use",
            "name": "StationUpgradeVoucher1",
            "targetPos": [{"x": 8, "y": 9}],
        })
        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")
        self.assertEqual(traces[-1]["coordinationReason"], "task_defense_return")

    def test_economy_soft_deadline_still_delivers_defense_response(self):
        payload = self._idle_third_gunner_payload(round_no=60)
        pioneer = next(
            role for role in payload["teamOur"]["roles"]
            if role["id"] == 10011
        )
        pioneer["pos"] = {"x": 15, "y": 16}
        for role in payload["teamOur"]["roles"]:
            if role["roleType"] == "worker":
                role["backpack"] = ["copper"] * 10
        payload["mapInfo"]["zones"].extend([
            {"pos": {"x": 3, "y": 2}, "neutralType": "vendor"},
            {"pos": {"x": 5, "y": 2}, "neutralType": "weaponShop"},
        ])
        payload["vendorShopList"] = [{"name": "copper", "price": 5}]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]

        class ControlledClock:
            now = 10.0

            def __call__(self):
                return self.now

        clock = ControlledClock()
        real_next_step = economy.next_step
        economy_searches = 0

        def exhaust_economy_budget(*args, **kwargs):
            nonlocal economy_searches
            economy_searches += 1
            clock.now = 13.0
            return real_next_step(*args, **kwargs)

        traces = []
        with patch.object(economy, "next_step", exhaust_economy_budget):
            response = DecisionEngine(
                clock=clock, budget_seconds=4.0,
            ).decide(payload, trace_sink=traces.append)

        self.assertGreaterEqual(economy_searches, 1)
        self.assertEqual(
            traces[0]["economyPlanning"]["truncatedReason"], "deadline",
        )
        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")
        self.assertTrue(any(
            action["domain"] == "defense"
            for action in traces[0]["actions"]
        ))

    def test_dawn_releases_idle_third_gunner_for_new_task(self):
        payload = self._idle_third_gunner_payload(round_no=1)

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")

    def test_two_worker_staffed_towers_leave_pioneer_free_for_new_task(self):
        payload = self._idle_third_gunner_payload()
        payload["teamOur"]["roles"] = [
            role for role in payload["teamOur"]["roles"]
            if role["id"] != 10040
        ]

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")

    def _third_post_payload(self, round_no):
        payload = task_payload(
            round_no=round_no,
            pioneer_pos=(15, 16),
            phase_task="active task",
        )
        payload["teamOur"]["roles"].extend([
            unit(10010, "worker", 6, 5),
            unit(10012, "worker", 9, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 9, 6, health=1000),
            unit(10040, "rocket", 12, 6, health=1000),
        ])
        return payload

    def test_task_keeps_working_with_margin_but_returns_for_required_third_post(self):
        enough = self._third_post_payload(50)
        response = DecisionEngine().decide(enough)
        self.assertTrue(response["prompt"])
        self.assertIn("Known remaining task rounds: 12", response["prompt"])
        self.assertNotIn("10011", response["roleCommandMap"])

        tight = self._third_post_payload(62)
        engine = DecisionEngine()
        response = engine.decide(tight)
        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")
        self.assertEqual(engine.state.state.plans[10011].reason, "gunner:10040")
        self.assertEqual(response["prompt"], "")

    def test_reliable_task_answer_submits_before_last_feasible_return(self):
        engine = DecisionEngine()
        first = self._third_post_payload(60)
        engine.decide(first)

        answer = self._third_post_payload(61)
        answer["llmResp"] = (
            '{"kind":"answer","content":"supported","complete":false}'
        )
        response = engine.decide(answer)
        self.assertEqual(response["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": "supported",
        })

        return_turn = self._third_post_payload(62)
        return_turn["lastRoundRoleActionResults"] = {"10011": True}
        response = engine.decide(return_turn)
        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")
        self.assertEqual(engine.state.state.plans[10011].reason, "gunner:10040")

    def test_disappeared_third_post_releases_task_pioneer(self):
        engine = DecisionEngine()
        tight = self._third_post_payload(62)
        engine.decide(tight)

        changed = self._third_post_payload(63)
        changed["teamOur"]["roles"] = [
            role for role in changed["teamOur"]["roles"]
            if role["id"] != 10040
        ]
        changed["lastRoundRoleActionResults"] = {"10011": True}
        response = engine.decide(changed)

        self.assertTrue(response["prompt"])
        self.assertNotIn("10011", response["roleCommandMap"])

    def test_disappeared_third_post_revokes_only_coordination_final_request(self):
        engine = DecisionEngine()
        constrained = self._third_post_payload(60)
        first = engine.decide(constrained)
        self.assertTrue(first["prompt"])

        changed = self._third_post_payload(61)
        changed["teamOur"]["roles"] = [
            role for role in changed["teamOur"]["roles"]
            if role["id"] != 10040
        ]
        changed["llmResp"] = (
            '{"kind":"command","content":"safe private command"}'
        )
        response = engine.decide(changed)

        self.assertEqual(response["executeCmd"], "safe private command")

    def test_remote_workers_projected_to_two_posts_start_pioneer_budget_early(self):
        payload = self._third_post_payload(50)
        payload["teamOur"]["teamId"] = "task-projected-workers"
        payload["teamOur"]["roles"][1]["pos"] = {"x": 1, "y": 1}
        payload["teamOur"]["roles"][2]["pos"] = {"x": 2, "y": 1}

        response = DecisionEngine().decide(payload)

        self.assertTrue(response["prompt"])
        self.assertNotIn("Known remaining task rounds: unknown", response["prompt"])

    def test_allocator_requires_pioneer_and_own_valid_task_point(self):
        # Break caught: acceptTask is only format-checked by the B allocator.
        adjacent = task_payload(pioneer_pos=(3, 3))
        turn = Turn.load(adjacent)
        self.assertTrue(ActionAllocator(turn).try_add(ActionProposal(
            10011, 10011, {"action": "acceptTask"},
        )))

        far = task_payload(pioneer_pos=(1, 1))
        self.assertFalse(ActionAllocator(Turn.load(far)).try_add(ActionProposal(
            10011, 10011, {"action": "acceptTask"},
        )))

        worker = task_payload(pioneer_pos=(3, 3))
        worker["teamOur"]["roles"][0]["roleType"] = "worker"
        self.assertFalse(ActionAllocator(Turn.load(worker)).try_add(ActionProposal(
            10011, 10011, {"action": "acceptTask"},
        )))

    def test_submit_requires_active_task_and_nonempty_bounded_answer(self):
        # Break caught: inactive, empty, or unbounded answers reach the response.
        inactive = Turn.load(task_payload(pioneer_pos=(3, 3)))
        self.assertFalse(ActionAllocator(inactive).try_add(ActionProposal(
            10011, 10011, {"action": "submitAnswer", "taskAnswer": "answer"},
        )))

        active = Turn.load(task_payload(
            pioneer_pos=(3, 3), phase_task="real task",
        ))
        self.assertFalse(ActionAllocator(active).try_add(ActionProposal(
            10011, 10011, {"action": "submitAnswer", "taskAnswer": ""},
        )))
        self.assertTrue(ActionAllocator(active).try_add(ActionProposal(
            10011, 10011, {"action": "submitAnswer", "taskAnswer": "answer"},
        )))

    def test_strict_llm_envelope_parser(self):
        # Break caught: prose or fenced JSON is forwarded as a shell command.
        tasks = importlib.import_module("agent.tasks")
        command = tasks.parse_llm_envelope(
            '{"kind":"command","content":"python3 solve.py"}'
        )
        answer = tasks.parse_llm_envelope(
            '{"kind":"answer","content":"42","complete":false}'
        )
        abandon = tasks.parse_llm_envelope(
            '{"kind":"abandon","reason":"insufficient evidence"}'
        )

        self.assertEqual(command.kind, "command")
        self.assertEqual(command.content, "python3 solve.py")
        self.assertEqual(answer.kind, "answer")
        self.assertFalse(answer.complete)
        self.assertIsNotNone(abandon)
        self.assertEqual(abandon.kind, "abandon")
        self.assertEqual(abandon.content, "insufficient evidence")
        fenced = (
            '```json\n{"kind":"answer","content":"42",'
            '"complete":true}\n```'
        )
        self.assertEqual(tasks.parse_llm_envelope(fenced),
                         tasks.LlmEnvelope("answer", "42", True))
        for label in ("python", "jsonl"):
            wrapped = fenced.replace("```json\n", f"```{label}\n", 1)
            self.assertEqual(tasks.parse_llm_envelope(wrapped),
                             tasks.LlmEnvelope("answer", "42", True))
        for invalid in (
            "prose " + fenced,
            fenced + " trailing prose",
            fenced + "\n" + fenced,
            "```json\n" + fenced.split("\n", 1)[1].removesuffix("\n```"),
            '{"kind":"command","content":""}',
            '{"kind":"answer","content":"42","complete":true,"extra":1}',
            '{"kind":"abandon","reason":""}',
            "not json",
        ):
            self.assertIsNone(tasks.parse_llm_envelope(invalid))

    def test_complete_unknown_label_fence_at_deadline_submits_inner_answer(self):
        engine = DecisionEngine()
        active = task_payload(round_no=1, pioneer_pos=(3, 3),
                              phase_task="return synthetic result")
        engine.decide(active)
        engine.state.state.active_task.timeout_round = 2
        final = copy.deepcopy(active)
        final["roundNo"] = 2
        final["llmResp"] = '```python\n{"kind":"answer","content":"{}","complete":true}\n```'
        response = engine.decide(final)
        self.assertEqual(response["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": "{}",
        })

    def test_vacuous_partial_gets_one_correction_without_submission(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="synthetic task",
        )
        engine.decide(active)
        task = engine.state.state.active_task
        task.timeout_round = 4
        blank = copy.deepcopy(active)
        blank["roundNo"] = 2
        blank["llmResp"] = json.dumps({
            "kind": "answer", "content": '{"a":"unknown","b":null}',
            "complete": False,
        })
        correction = engine.decide(blank)
        self.assertTrue(correction["prompt"])
        self.assertNotIn("10011", correction["roleCommandMap"])
        self.assertTrue(task.vacuous_partial_correction_requested)
        usable = copy.deepcopy(active)
        usable["roundNo"] = 3
        usable["llmResp"] = json.dumps({
            "kind": "answer", "content": '{"a":"unknown","b":0}',
            "complete": False,
        })
        self.assertEqual(engine.decide(usable)["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": '{"a":"unknown","b":0}',
        })

    def test_vacuous_partial_boundary_and_repetition(self):
        tasks = importlib.import_module("agent.tasks")
        for text in (" ", "unknown", "NULL", "none", "N/A", "not found",
                     "[]", "{}", '{"x":"unknown","y":null}',
                     '{"x":"N/A","y":[]}'):
            self.assertTrue(tasks._is_vacuous_partial_answer(text))
        for text in ("0", "false", '{"x":"unknown","y":false}',
                     '{"x":"unknown","y":"real"}'):
            self.assertFalse(tasks._is_vacuous_partial_answer(text))
        engine = DecisionEngine()
        active = task_payload(round_no=1, pioneer_pos=(3, 3), phase_task="synthetic")
        engine.decide(active)
        task = engine.state.state.active_task
        task.timeout_round = 4
        for round_no in (2, 3):
            bad = copy.deepcopy(active)
            bad["roundNo"] = round_no
            bad["llmResp"] = json.dumps({
                "kind": "answer", "content": "unknown", "complete": False,
            })
            result = engine.decide(bad)
            self.assertFalse(any(
                command["action"] == "submitAnswer"
                for command in result["roleCommandMap"].values()
            ))
        self.assertEqual(task.solver_stopped_reason,
                         "vacuous_partial_after_correction")

    def test_empty_result_can_be_confirmed_as_complete_after_one_correction(self):
        engine = DecisionEngine()
        active = task_payload(round_no=1, pioneer_pos=(3, 3),
                              phase_task="Return an empty JSON list if no matches")
        engine.decide(active)
        task = engine.state.state.active_task
        task.timeout_round = 3
        partial = copy.deepcopy(active)
        partial["roundNo"] = 2
        partial["llmResp"] = json.dumps({
            "kind": "answer", "content": "[]", "complete": False,
        })
        correction = engine.decide(partial)
        self.assertNotIn("10011", correction["roleCommandMap"])
        self.assertIn("complete:true", correction["prompt"])
        confirmed = copy.deepcopy(active)
        confirmed["roundNo"] = 3
        confirmed["llmResp"] = json.dumps({
            "kind": "answer", "content": "[]", "complete": True,
        })
        self.assertEqual(engine.decide(confirmed)["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": "[]",
        })

    def test_task_end_trace_is_bounded_unattributed_and_action_neutral(self):
        engine = DecisionEngine()
        active = task_payload(round_no=1, pioneer_pos=(3, 3), phase_task="synthetic")
        active["teamOur"]["totalScore"] = 10
        active["teamOur"]["goldNum"] = 5
        engine.decide(active)
        answer = copy.deepcopy(active)
        answer["roundNo"] = 2
        answer["llmResp"] = json.dumps({
            "kind": "answer", "content": "SECRET_SYNTHETIC_ANSWER", "complete": True,
        })
        engine.decide(answer)
        ended = copy.deepcopy(active)
        ended["roundNo"] = 3
        ended["phaseTask"] = ""
        ended["teamOur"]["totalScore"] = 20
        ended["teamOur"]["goldNum"] = 9
        ended["lastRoundRoleActionResults"] = {"10011": True}
        ended["errors"] = [{"errorCode": 2, "description": "PRIVATE_SYNTHETIC_ERROR"}]
        without_trace = engine.decide(ended)
        trace = []
        with_trace = engine.decide(ended, trace_sink=trace.append)
        self.assertEqual(without_trace, with_trace)
        event = trace[0]["taskEnd"]
        self.assertEqual(event["instanceId"], "1:1")
        self.assertEqual(event["endRound"], 3)
        self.assertEqual(event["submissionCount"], 1)
        self.assertEqual(event["associatedErrorCodes"], [2])
        self.assertEqual(event["observedTeamScoreDelta"], 10)
        self.assertEqual(event["observedGoldDelta"], 4)
        self.assertEqual(event["attribution"], "unattributed")
        self.assertEqual(event["successStatus"], "unknown")
        self.assertNotIn("SECRET_SYNTHETIC_ANSWER", json.dumps(trace))
        self.assertNotIn("PRIVATE_SYNTHETIC_ERROR", json.dumps(event))
        later = copy.deepcopy(ended)
        later["roundNo"] = 4
        next_trace = []
        engine.decide(later, trace_sink=next_trace.append)
        self.assertIsNone(next_trace[0]["taskEnd"])

    def test_task_end_keeps_prior_submission_error_and_timeout_code(self):
        engine = DecisionEngine()
        active = task_payload(round_no=1, pioneer_pos=(3, 3),
                              phase_task="synthetic")
        engine.decide(active)
        answer = copy.deepcopy(active)
        answer["roundNo"] = 2
        answer["llmResp"] = json.dumps({
            "kind": "answer", "content": "partial", "complete": False,
        })
        engine.decide(answer)
        feedback = copy.deepcopy(active)
        feedback["roundNo"] = 3
        feedback["lastRoundRoleActionResults"] = {"10011": True}
        feedback["errors"] = [{"errorCode": 2, "description": "synthetic"}]
        engine.decide(feedback)
        ended = copy.deepcopy(active)
        ended["roundNo"] = 4
        ended["phaseTask"] = ""
        ended["errors"] = [{"errorCode": 1, "description": "timeout"}]
        trace = []
        engine.decide(ended, trace_sink=trace.append)
        self.assertEqual(trace[0]["taskEnd"]["associatedErrorCodes"], [1, 2])

    def test_crlf_hint_requires_current_task_evidence_and_is_once_only(self):
        for result, expected in (
            ("[exitCode:126]\n/bin/sh^M: bad interpreter", True),
            ("[exitCode:126]\npermission denied", False),
            ("[exitCode:126]\r\n./check: bad interpreter: No such file or directory\r\n", False),
        ):
            with self.subTest(result=result):
                engine = DecisionEngine()
                active = task_payload(
                    round_no=1, pioneer_pos=(3, 3), phase_task="synthetic",
                )
                engine.decide(active)
                task = engine.state.state.active_task
                task.pending_llm_round = None
                task.pending_cmd_round = 1
                task.last_command = "./check"
                received = copy.deepcopy(active)
                received["roundNo"] = 2
                received["lastCmdResult"] = result
                response = engine.decide(received)
                self.assertEqual("CRLF interpreter" in response["prompt"], expected)
                self.assertEqual(response["executeCmd"], "")
                if expected:
                    self.assertTrue(task.crlf_hint_requested)
                    task.pending_cmd_round = 2
                    repeated = copy.deepcopy(received)
                    repeated["roundNo"] = 3
                    later = engine.decide(repeated)
                    self.assertNotIn("CRLF interpreter", later["prompt"])

    def test_crlf_auto_repair_targets_exact_script_and_preserves_cd(self):
        for command, script in (
            ("./check.sh", "./check.sh"),
            ("cd '/tmp/work space' && './my check.sh'", "./my check.sh"),
        ):
            with self.subTest(command=command):
                engine = DecisionEngine()
                active = task_payload(round_no=1, pioneer_pos=(3, 3),
                                      phase_task="repair current script")
                engine.decide(active)
                task = engine.state.state.active_task
                task.timeout_round = 8
                task.pending_llm_round = None
                task.pending_cmd_round = 1
                task.last_command = command
                received = copy.deepcopy(active)
                received["roundNo"] = 2
                received["lastCmdResult"] = (
                    f"[exitCode:126]\nbash: {script}: /bin/sh^M: bad interpreter"
                )
                response = engine.decide(received)
                self.assertEqual(response["prompt"], "")
                self.assertIn("python3 -c", response["executeCmd"])
                self.assertIn("check.sh", response["executeCmd"])
                self.assertTrue(response["executeCmd"].endswith(
                    " && " + ("'./my check.sh'" if "my check" in command else script)
                ))
                if command.startswith("cd "):
                    self.assertTrue(response["executeCmd"].startswith(
                        "cd '/tmp/work space' && "
                    ))
                self.assertNotIn("find .", response["executeCmd"])

    def test_crlf_auto_repair_rejects_ambiguous_or_late_error(self):
        for command, result, timeout in (
            ("./check.sh", "[exitCode:126]\n./other.sh: /bin/sh^M: bad interpreter", 8),
            ("./check.sh | cat", "[exitCode:126]\n./check.sh: /bin/sh^M: bad interpreter", 8),
            ("./check.sh", "[exitCode:126]\n./check.sh: /bin/sh^M: bad interpreter", 4),
        ):
            with self.subTest(command=command, timeout=timeout):
                engine = DecisionEngine()
                active = task_payload(round_no=1, pioneer_pos=(3, 3),
                                      phase_task="repair current script")
                engine.decide(active)
                task = engine.state.state.active_task
                task.timeout_round = timeout
                task.pending_llm_round = None
                task.pending_cmd_round = 1
                task.last_command = command
                received = copy.deepcopy(active)
                received["roundNo"] = 2
                received["lastCmdResult"] = result
                response = engine.decide(received)
                self.assertEqual(response["executeCmd"], "")

    def test_prior_task_experience_is_structured_unverified_and_same_type_only(self):
        engine = DecisionEngine()
        first = task_payload(round_no=1, pioneer_pos=(3, 3), phase_task="first synthetic")
        engine.decide(first)
        answer = copy.deepcopy(first)
        answer["roundNo"] = 2
        answer["llmResp"] = json.dumps({
            "kind": "answer", "content": "PRIVATE_SYNTHETIC_ANSWER", "complete": True,
        })
        engine.decide(answer)
        ended = copy.deepcopy(first)
        ended["roundNo"] = 3
        ended["phaseTask"] = ""
        engine.decide(ended)
        next_task = copy.deepcopy(first)
        next_task["roundNo"] = 4
        next_task["phaseTask"] = "second synthetic"
        prompt = engine.decide(next_task)["prompt"]
        self.assertIn("Unverified prior same-type workflow", prompt)
        self.assertIn("re-check this task", prompt)
        self.assertNotIn("PRIVATE_SYNTHETIC_ANSWER", prompt)
        self.assertNotIn("commands:", prompt)
        other = DecisionEngine()
        other.decide(first)
        other.decide(answer)
        other.decide(ended)
        different = copy.deepcopy(next_task)
        different["teamOur"]["playerTasks"][0]["taskType"] = "自进化类2"
        self.assertNotIn("Unverified prior same-type workflow",
                         other.decide(different)["prompt"])

    def test_prior_workflow_keeps_last_useful_steps_without_values(self):
        state = state_for(task_payload(phase_task="synthetic"))
        prior = TaskMemory("1:prior", "synthetic", task_type="自进化类1")
        prior.end_reason = "unknown"
        prior.submission_count = 1
        for command, result in (
            ("cat /tmp/old-input", "[exitCode:0]\nold data"),
            ("curl -H 'X-API-Key: PRIVATE_KEY' 'https://example.test/a?city=SECRET'",
             "[exitCode:0]\n" + json.dumps({"error": "bad city"})),
            ("curl 'https://example.test/fail'",
             "[exitCode:0]\n" + json.dumps({"status": 401})),
            ("curl -X POST 'https://example.test/write'",
             "[exitCode:0]\n" + json.dumps({"ok": True})),
            ("curl -XPOST 'https://example.test/write'",
             "[exitCode:0]\n" + json.dumps({"ok": True})),
            ("curl --request=POST 'https://example.test/write'",
             "[exitCode:0]\n" + json.dumps({"ok": True})),
            ("curl -dsecret 'https://example.test/write'",
             "[exitCode:0]\n" + json.dumps({"ok": True})),
            ("cd /tmp && curl 'https://example.test/read'",
             "[exitCode:0]\n" + json.dumps({"data": [1]})),
            ("curl 'https://example.test/b?city=SECRET'",
             "[exitCode:0]\n" + json.dumps({"data": [1]})),
            ("python3 /tmp/old-parser.py", "[exitCode:0]\nparsed"),
            ("./check", "[exitCode:0]\nCHECK PASS"),
            ("cd '/tmp/current workspace' && ./check",
             "[exitCode:0]\nCHECK PASS"),
        ):
            _record_sop_step(prior, command, result)
        self.assertEqual(len(prior.sop_steps), 4)
        state.ended_tasks.append(prior)
        current = TaskMemory("1:current", "synthetic", task_type="自进化类1")
        hint = _prior_task_experience(current, state)
        self.assertIn("local check", hint)
        self.assertIn("API read", hint)
        for sensitive in ("PRIVATE_KEY", "SECRET", "/tmp/", "bad city", "old data"):
            self.assertNotIn(sensitive, hint)
        prior.submission_count = 0
        self.assertEqual(_prior_task_experience(current, state), "")

    def test_engine_sop_chain_keeps_last_four_safe_steps(self):
        engine = DecisionEngine()
        active = task_payload(round_no=1, pioneer_pos=(3, 3),
                              phase_task="synthetic work")
        engine.decide(active)
        engine.state.state.active_task.timeout_round = 30
        round_no = 2
        commands = (
            ("cat /tmp/old-input", "[exitCode:0]\ninput"),
            ("curl -XPOST 'https://example.test/write?secret=SECRET'",
             "[exitCode:0]\n" + json.dumps({"ok": True})),
            ("curl 'https://example.test/read?secret=SECRET'",
             "[exitCode:0]\n" + json.dumps({"data": [1]})),
            ("python3 /tmp/old-parser.py", "[exitCode:0]\nparsed"),
            ("cd '/tmp/current workspace' && ./check", "[exitCode:0]\nPASS"),
        )
        for command, result in commands:
            request = copy.deepcopy(active)
            request["roundNo"] = round_no
            request["llmResp"] = json.dumps({"kind": "command", "content": command})
            self.assertEqual(engine.decide(request)["executeCmd"], command)
            round_no += 1
            received = copy.deepcopy(active)
            received["roundNo"] = round_no
            received["lastCmdResult"] = result
            engine.decide(received)
            round_no += 1
        answer = copy.deepcopy(active)
        answer["roundNo"] = round_no
        answer["llmResp"] = json.dumps({"kind": "answer", "content": "private result",
                                         "complete": True})
        engine.decide(answer)
        ended = copy.deepcopy(active)
        ended["roundNo"] = round_no + 1
        ended["phaseTask"] = ""
        engine.decide(ended)
        next_task = copy.deepcopy(active)
        next_task["roundNo"] = round_no + 2
        next_task["phaseTask"] = "next synthetic work"
        prompt = engine.decide(next_task)["prompt"]
        self.assertIn("API read", prompt)
        self.assertIn("local check", prompt)
        self.assertIn("file inspection", prompt)
        self.assertNotIn("SECRET", prompt)
        self.assertNotIn("/tmp/current workspace", prompt)
        self.assertNotIn("private result", prompt)

    def test_pagination_hint_requires_complete_consistent_result_and_budget(self):
        valid = json.dumps({
            "pagination": {"total_count": 5, "offset": 0, "limit": 2},
            "data": [{"id": 1}, {"id": 2}],
        })
        cases = (
            ("[exitCode:0]\n" + valid, True, None),
            ("[exitCode:0]\n" + valid + "\n[TRUNCATED]", False, None),
            ("[exitCode:1]\n" + valid, False, None),
            ("[exitCode:0]\n" + json.dumps({
                "pagination": {"total_count": 5, "offset": 0, "limit": 2},
                "data": [{"id": 1}],
            }), False, None),
            ("[exitCode:0]\n" + json.dumps({
                "error": "failed", "pagination": {"total_count": 5,
                    "offset": 0, "limit": 2}, "data": [{"id": 1}, {"id": 2}],
            }), False, None),
            ("[exitCode:0]\n" + valid, False, 4),
        )
        for result, expected, timeout_round in cases:
            with self.subTest(result=result, timeout=timeout_round):
                engine = DecisionEngine()
                active = task_payload(round_no=1, pioneer_pos=(3, 3),
                                      phase_task="synthetic")
                engine.decide(active)
                task = engine.state.state.active_task
                task.pending_llm_round = None
                task.pending_cmd_round = 1
                task.last_command = "inspect current API"
                task.timeout_round = timeout_round
                received = copy.deepcopy(active)
                received["roundNo"] = 2
                received["lastCmdResult"] = result
                response = engine.decide(received)
                self.assertEqual("Pagination metadata" in response["prompt"],
                                 expected)
                self.assertEqual(response["executeCmd"], "")
                if expected:
                    self.assertTrue(task.pagination_checked)
                    task.pending_cmd_round = 2
                    repeated = copy.deepcopy(received)
                    repeated["roundNo"] = 3
                    self.assertNotIn("Pagination metadata",
                                     engine.decide(repeated)["prompt"])

    def test_nested_pagination_hint_uses_real_result_chain_without_execution(self):
        valid = {"code": 0, "data": {"records": [{"id": 1}, {"id": 2}],
            "pagination": {"total_count": 5, "offset": 0, "limit": 2}}}
        cases = (
            (valid, True, None),
            ({**valid, "code": 500}, False, None),
            ({**valid, "code": 201}, True, None),
            ({**valid, "success": False}, False, None),
            ({**valid, "data": {**valid["data"], "error": "bad"}}, False, None),
            ({**valid, "data": {**valid["data"], "status": "failed"}}, False, None),
            ({**valid, "data": {**valid["data"], "records": [{"id": 1}]}}, False, None),
            ({**valid, "pagination": {"total_count": 5, "offset": 0, "limit": 2}},
             False, None),
            (valid, False, 4),
        )
        for body, expected, timeout_round in cases:
            with self.subTest(body=body, timeout=timeout_round):
                engine = DecisionEngine()
                active = task_payload(round_no=1, pioneer_pos=(3, 3),
                                      phase_task="synthetic nested API")
                engine.decide(active)
                task = engine.state.state.active_task
                task.pending_llm_round = None
                task.pending_cmd_round = 1
                task.last_command = "inspect current API"
                task.timeout_round = timeout_round
                received = copy.deepcopy(active)
                received["roundNo"] = 2
                received["lastCmdResult"] = "[exitCode:0]\n" + json.dumps(body)
                response = engine.decide(received)
                self.assertEqual("Pagination metadata" in response["prompt"],
                                 expected)
                self.assertEqual(response["executeCmd"], "")
                if expected:
                    task.pending_cmd_round = 2
                    repeated = copy.deepcopy(received)
                    repeated["roundNo"] = 3
                    self.assertNotIn("Pagination metadata",
                                     engine.decide(repeated)["prompt"])

    def test_nested_pagination_long_result_keeps_hint_and_rejects_truncation(self):
        body = {"code": 0, "data": {"records": [
            {"blob": "x" * 5000}, {"blob": "y" * 5000}],
            "pagination": {"total_count": 3, "offset": 0, "limit": 2}}}
        for suffix, expected in (("", True), ("\n[TRUNCATED]", False)):
            with self.subTest(suffix=suffix):
                engine = DecisionEngine()
                active = task_payload(round_no=1, pioneer_pos=(3, 3),
                                      phase_task="long nested API")
                engine.decide(active)
                task = engine.state.state.active_task
                task.pending_llm_round = None
                task.pending_cmd_round = 1
                received = copy.deepcopy(active)
                received["roundNo"] = 2
                received["lastCmdResult"] = "[exitCode:0]\n" + json.dumps(body) + suffix
                response = engine.decide(received)
                self.assertEqual("Pagination metadata" in response["prompt"],
                                 expected)
                self.assertEqual(response["executeCmd"], "")

    def test_auto_pagination_get_preserves_auth_and_prior_page_evidence(self):
        engine = DecisionEngine()
        active = task_payload(round_no=1, pioneer_pos=(3, 3),
                              phase_task="synthetic paged API")
        engine.decide(active)
        task = engine.state.state.active_task
        task.timeout_round = 8
        task.pending_llm_round = None
        task.pending_cmd_round = 1
        task.last_command = (
            "curl -sS -H 'X-API-Key: SYNTHETIC_AUTH' "
            "'https://example.test/records?city=New%20York&offset=0&limit=2'"
        )
        page0 = copy.deepcopy(active)
        page0["roundNo"] = 2
        page0["lastCmdResult"] = "[exitCode:0]\n" + json.dumps({
            "code": 0, "data": {"records": [
                {"id": "FIRST_PAGE_MARKER"}, {"id": 2}],
                "pagination": {"total_count": 5, "offset": 0, "limit": 2}},
        })
        first = engine.decide(page0)
        self.assertEqual(first["prompt"], "")
        self.assertIn("offset=2", first["executeCmd"])
        self.assertIn("limit=2", first["executeCmd"])
        self.assertIn("city=New%20York", first["executeCmd"])
        self.assertIn("SYNTHETIC_AUTH", first["executeCmd"])
        page2 = copy.deepcopy(active)
        page2["roundNo"] = 3
        page2["lastCmdResult"] = "[exitCode:0]\n" + json.dumps({
            "code": 0, "data": {"records": [{"id": 3}, {"id": 4}],
                "pagination": {"total_count": 5, "offset": 2, "limit": 2}},
        })
        second = engine.decide(page2)
        self.assertIn("offset=4", second["executeCmd"])
        page4 = copy.deepcopy(active)
        page4["roundNo"] = 4
        page4["lastCmdResult"] = "[exitCode:0]\n" + json.dumps({
            "code": 0, "data": {"records": [{"id": 5}],
                "pagination": {"total_count": 5, "offset": 4, "limit": 2}},
        })
        final_prompt = engine.decide(page4)
        self.assertEqual(final_prompt["executeCmd"], "")
        self.assertIn("FIRST_PAGE_MARKER", final_prompt["prompt"])

    def test_auto_pagination_updates_only_missing_cursor_parts(self):
        body = "[exitCode:0]\n" + json.dumps({
            "data": {"records": [{"id": 1}, {"id": 2}],
                "pagination": {"total_count": 5, "offset": 0, "limit": 2}},
        })
        for query, expected in (
            ("city=New%20York&offset=0", "city=New%20York&offset=2&limit=2"),
            ("city=New%20York&limit=2", "city=New%20York&limit=2&offset=2"),
        ):
            with self.subTest(query=query):
                engine = DecisionEngine()
                active = task_payload(round_no=1, pioneer_pos=(3, 3),
                                      phase_task="paged API")
                engine.decide(active)
                task = engine.state.state.active_task
                task.timeout_round = 8
                task.pending_llm_round = None
                task.pending_cmd_round = 1
                task.last_command = f"curl 'https://example.test/data?{query}'"
                received = copy.deepcopy(active)
                received["roundNo"] = 2
                received["lastCmdResult"] = body
                response = engine.decide(received)
                self.assertIn(expected, response["executeCmd"])

    def test_auto_pagination_rejects_unsafe_commands_duplicate_and_deadline(self):
        body = "[exitCode:0]\n" + json.dumps({
            "data": {"records": [{"id": 1}, {"id": 2}],
                "pagination": {"total_count": 5, "offset": 0, "limit": 2}},
        })
        commands = (
            "curl -X POST 'https://example.test/data?offset=0&limit=2'",
            "curl 'https://example.test/data?offset=0&limit=2' | cat",
            "cd /tmp/ws && curl 'https://example.test/data?offset=0&limit=2'",
            "curl 'https://example.test/data?page=1&limit=2'",
            "curl 'https://example.test/data?offset=0&offset=0&limit=2'",
            "curl 'https://example.test/data?signature=abc&offset=0&limit=2'",
            "curl -H 'X-API-Key: $TOKEN' 'https://example.test/data?offset=0&limit=2'",
            "curl 'https://example.test/data?Offset=0&limit=2'",
            "curl 'https://example.test/data?%6fffset=0&limit=2'",
            "curl 'https://[bad?offset=0&limit=2'",
        )
        for command in commands:
            with self.subTest(command=command):
                engine = DecisionEngine()
                active = task_payload(round_no=1, pioneer_pos=(3, 3),
                                      phase_task="paged API")
                engine.decide(active)
                task = engine.state.state.active_task
                task.timeout_round = 8
                task.pending_llm_round = None
                task.pending_cmd_round = 1
                task.last_command = command
                received = copy.deepcopy(active)
                received["roundNo"] = 2
                received["lastCmdResult"] = body
                response = engine.decide(received)
                self.assertEqual(response["executeCmd"], "")
                self.assertIn("Pagination metadata", response["prompt"])
        engine = DecisionEngine()
        active = task_payload(round_no=1, pioneer_pos=(3, 3),
                              phase_task="paged API")
        engine.decide(active)
        task = engine.state.state.active_task
        task.timeout_round = 6
        task.pending_llm_round = None
        task.pending_cmd_round = 1
        task.last_command = "curl 'https://example.test/data?offset=0&limit=2'"
        received = copy.deepcopy(active)
        received["roundNo"] = 2
        received["lastCmdResult"] = body
        self.assertIn("offset=2", engine.decide(received)["executeCmd"])
        task.pending_cmd_round = 2
        duplicate = copy.deepcopy(received)
        duplicate["roundNo"] = 3
        repeated = engine.decide(duplicate)
        self.assertEqual(repeated["executeCmd"], "")

    def test_long_tool_results_keep_detected_hints_in_final_prompt(self):
        long_results = (
            ("[exitCode:0]\n" + json.dumps({
                "pagination": {"total_count": 3, "offset": 0, "limit": 2},
                "data": [{"blob": "x" * 5000}, {"blob": "y" * 5000}],
            }), "Pagination metadata"),
            ("[exitCode:126]\n./check: /bin/sh^M: bad interpreter\n"
             + "x" * 10000, "CRLF interpreter"),
        )
        for result, needle in long_results:
            with self.subTest(needle=needle):
                engine = DecisionEngine()
                active = task_payload(round_no=1, pioneer_pos=(3, 3),
                                      phase_task="synthetic long output")
                engine.decide(active)
                task = engine.state.state.active_task
                task.pending_llm_round = None
                task.pending_cmd_round = 1
                task.last_command = "synthetic inspect"
                received = copy.deepcopy(active)
                received["roundNo"] = 2
                received["lastCmdResult"] = result
                response = engine.decide(received)
                self.assertIn(needle, response["prompt"])
                self.assertIn("[TRUNCATED MIDDLE]", response["prompt"])
                self.assertEqual(response["executeCmd"], "")
                task.pending_cmd_round = 2
                repeated = copy.deepcopy(received)
                repeated["roundNo"] = 3
                self.assertNotIn(needle, engine.decide(repeated)["prompt"])

    def test_oversized_json_integer_is_rejected_without_poisoning_next_response(self):
        # Break caught: Python's integer digit limit raises ValueError outside
        # the JSONDecodeError branch and crashes the shared task solver entry.
        tasks = importlib.import_module("agent.tasks")
        oversized = (
            '{"kind":"command","content":"inspect","extra":'
            + "9" * 5_000
            + "}"
        )

        self.assertIsNone(tasks.parse_llm_envelope(oversized))
        self.assertEqual(
            tasks.parse_llm_envelope(
                '{"kind":"answer","content":"42","complete":true}'
            ),
            tasks.LlmEnvelope("answer", "42", True),
        )

        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        engine.decide(active)
        invalid = copy.deepcopy(active)
        invalid["roundNo"] = 2
        invalid["llmResp"] = oversized
        self.assertIn("one format-only correction", engine.decide(invalid)["prompt"])
        valid = copy.deepcopy(active)
        valid["roundNo"] = 3
        valid["llmResp"] = (
            '{"kind":"answer","content":"42","complete":true}'
        )
        self.assertEqual(engine.decide(valid)["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": "42",
        })

    def test_missing_kind_gets_one_format_correction_then_submits_json_answer(self):
        # Break caught: a likely answer object receives only generic JSON advice,
        # then repetition protection ends the task before a legal answer can submit.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="synthetic task",
        )
        normal_prompt = engine.decide(active)["prompt"]
        self.assertIn("Command exploration is allowed.", normal_prompt)
        self.assertIn("Maximum remaining tool cycles: 4", normal_prompt)
        self.assertIn('{"kind":"command"', normal_prompt)

        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = (
            '{"kind":"command","content":"inspect-input"}'
        )
        self.assertEqual(engine.decide(command)["executeCmd"], "inspect-input")
        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = "[exitCode:0]\nsynthetic verified evidence"
        self.assertTrue(engine.decide(result)["prompt"])

        invalid = copy.deepcopy(active)
        invalid["roundNo"] = 4
        invalid["llmResp"] = '{"result":"synthetic-value"}'
        traces = []
        correction = engine.decide(invalid, trace_sink=traces.append)

        self.assertEqual(correction["executeCmd"], "")
        self.assertNotIn("10011", correction["roleCommandMap"])
        self.assertIn("one format-only correction", correction["prompt"])
        self.assertIn('"kind":"answer"', correction["prompt"])
        self.assertIn("content must be a JSON string", correction["prompt"])
        self.assertNotIn("Command exploration is allowed.", correction["prompt"])
        self.assertIn(
            "Command exploration is not allowed", correction["prompt"],
        )
        self.assertIn("Maximum remaining tool cycles: 0", correction["prompt"])
        self.assertNotIn('{"kind":"command"', correction["prompt"])
        self.assertNotIn("One command request consumes", correction["prompt"])
        task = engine.state.state.active_task
        self.assertTrue(task.envelope_correction_requested)
        self.assertTrue(task.envelope_correction_pending)
        self.assertEqual(task.last_envelope_rejection, "missing_kind")
        self.assertTrue(traces[-1]["taskEnvelopeCorrectionRequested"])
        self.assertTrue(traces[-1]["taskEnvelopeCorrectionPending"])
        self.assertEqual(
            traces[-1]["taskEnvelopeRejectionClass"], "missing_kind",
        )

        answer_text = '{"result":"synthetic-value"}'
        corrected = copy.deepcopy(active)
        corrected["roundNo"] = 5
        corrected["llmResp"] = json.dumps({
            "kind": "answer",
            "content": answer_text,
            "complete": True,
        })
        submitted = engine.decide(corrected)

        self.assertEqual(submitted["executeCmd"], "")
        self.assertEqual(submitted["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": answer_text,
        })
        self.assertFalse(task.envelope_correction_pending)

        finished = copy.deepcopy(active)
        finished["roundNo"] = 6
        finished["phaseTask"] = ""
        finished["lastRoundRoleActionResults"] = {"10011": True}
        first_finish = engine.decide(finished)
        replay_finish = engine.decide(copy.deepcopy(finished))
        self.assertFalse(any(
            command.get("action") == "submitAnswer"
            for command in first_finish["roleCommandMap"].values()
        ))
        self.assertEqual(replay_finish, first_finish)

    def test_format_correction_rejects_command_and_any_second_invalid_envelope(self):
        # Break caught: format correction can be converted into another command,
        # or changing an invalid response fingerprint opens an unbounded prompt loop.
        for followup, stopped_reason in (
            (
                '{"kind":"command","content":"echo unsafe"}',
                "command_after_envelope_correction",
            ),
            (
                '{"kind":"unknown","content":"changed"}',
                "invalid_envelope_after_correction",
            ),
            (
                '{"kind":"answer","content":7,"complete":true}',
                "invalid_envelope_after_correction",
            ),
        ):
            with self.subTest(followup=followup):
                engine = DecisionEngine()
                active = task_payload(
                    round_no=1, pioneer_pos=(3, 3), phase_task="synthetic task",
                )
                engine.decide(active)
                invalid = copy.deepcopy(active)
                invalid["roundNo"] = 2
                invalid["llmResp"] = '{"answer":"synthetic"}'
                self.assertTrue(engine.decide(invalid)["prompt"])

                repeated = copy.deepcopy(active)
                repeated["roundNo"] = 3
                repeated["llmResp"] = followup
                response = engine.decide(repeated)

                self.assertEqual(response["executeCmd"], "")
                self.assertEqual(response["prompt"], "")
                self.assertFalse(any(
                    command.get("action") == "submitAnswer"
                    for command in response["roleCommandMap"].values()
                ))
                self.assertEqual(
                    engine.state.state.active_task.solver_stopped_reason,
                    stopped_reason,
                )

    def test_format_correction_retains_tail_answer_from_long_invalid_envelope(self):
        # Break caught: a missing-kind object with a long explanation loses its
        # answer tail before the model gets its one format-only correction.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="synthetic task",
        )
        engine.decide(active)
        invalid = copy.deepcopy(active)
        invalid["roundNo"] = 2
        invalid["llmResp"] = json.dumps({
            "explanation": "x" * 5_000,
            "content": "TAIL_ANSWER_41",
        })

        correction = engine.decide(invalid)

        self.assertIn("TAIL_ANSWER_41", correction["prompt"])
        self.assertIn("[TRUNCATED MIDDLE]", correction["prompt"])
        self.assertNotIn("submitAnswer", str(correction["roleCommandMap"]))

    def test_long_tool_result_marks_omitted_range_without_guessing_middle(self):
        # Break caught: a bounded output hides its middle without telling the
        # solver how much is unknown, encouraging a false negative inference.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="inspect API evidence",
        )
        engine.decide(active)
        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"inspect"}'
        engine.decide(command)
        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = (
            "[exitCode:0]\n" + "a" * 9_000 + "TOKEN_SYNTHETIC_MIDDLE"
            + "z" * 9_000
        )

        prompt = engine.decide(result)["prompt"]

        self.assertIn("[TRUNCATED MIDDLE]", prompt)
        self.assertIn("omitted chars", prompt)
        self.assertNotIn("TOKEN_SYNTHETIC_MIDDLE", prompt)
        self.assertEqual(engine.state.state.active_task.tool_results[-1][1], result["lastCmdResult"])

    def test_tool_input_trace_proves_current_task_request_association(self):
        # Break caught: logs call an accepted prior prompt/command result
        # unknown even though pending request state proves the association.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="API evidence task",
        )
        engine.decide(active)
        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"inspect-api"}'
        traces = []
        engine.decide(command, trace_sink=traces.append)
        self.assertEqual(traces[-1]["taskToolInputs"], [{
            "kind": "llm", "issuedRound": 1, "receivedRound": 2,
            "originalChars": len(command["llmResp"]),
        }])
        server = importlib.import_module("agent.server")
        llm_detail = server.task_detail_log_record(
            command,
            {"roleCommandMap": {}, "prompt": "", "executeCmd": "inspect-api"},
            decision_trace=traces[-1],
        )
        self.assertEqual(
            llm_detail["inputAssociation"]["llmResp"],
            "accepted_current_task",
        )

        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = "[exitCode:0]\nHTTP/1.1 404 Not Found"
        engine.decide(result, trace_sink=traces.append)
        self.assertEqual(traces[-1]["taskToolInputs"], [{
            "kind": "cmd", "issuedRound": 2, "receivedRound": 3,
            "originalChars": len(result["lastCmdResult"]),
        }])
        detail = server.task_detail_log_record(
            result, {"roleCommandMap": {}, "prompt": "", "executeCmd": ""},
            decision_trace=traces[-1],
        )
        self.assertEqual(
            detail["inputAssociation"]["lastCmdResult"],
            "accepted_current_task",
        )
        self.assertEqual(detail["acceptedToolInputs"], traces[-1]["taskToolInputs"])

        replacement = copy.deepcopy(active)
        replacement["roundNo"] = 4
        replacement["phaseTask"] = "different task"
        replacement["lastCmdResult"] = "late result"
        engine.decide(replacement, trace_sink=traces.append)
        self.assertEqual(traces[-1]["taskToolInputs"], [])
        late_detail = server.task_detail_log_record(
            replacement,
            {"roleCommandMap": {}, "prompt": "", "executeCmd": ""},
            decision_trace=traces[-1],
        )
        self.assertEqual(
            late_detail["inputAssociation"]["lastCmdResult"],
            "unknown_previous_request",
        )

    def test_crlf_failure_can_be_followed_by_model_repair_and_checked_answer(self):
        # Characterization: no local shell rewrite or inferred success; only
        # the model's revised command and later real check permit submission.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="repair script and report check",
        )
        engine.decide(active)
        first = copy.deepcopy(active)
        first["roundNo"] = 2
        first["llmResp"] = json.dumps({
            "kind": "command", "content": "./check.sh",
        })
        self.assertEqual(engine.decide(first)["executeCmd"], "./check.sh")
        failed = copy.deepcopy(active)
        failed["roundNo"] = 3
        failed["lastCmdResult"] = (
            "[exitCode:126]\n/bin/sh^M: bad interpreter"
        )
        prompt = engine.decide(failed)["prompt"]
        self.assertIn("bad interpreter", prompt)
        self.assertIn("./check.sh", prompt)
        revised = copy.deepcopy(active)
        revised["roundNo"] = 4
        revised["llmResp"] = json.dumps({
            "kind": "command", "content": "python3 check.py",
        })
        self.assertEqual(engine.decide(revised)["executeCmd"], "python3 check.py")
        checked = copy.deepcopy(active)
        checked["roundNo"] = 5
        checked["lastCmdResult"] = "[exitCode:0]\nCHECK PASS synthetic"
        self.assertIn("CHECK PASS synthetic", engine.decide(checked)["prompt"])
        answer = copy.deepcopy(active)
        answer["roundNo"] = 6
        answer["llmResp"] = json.dumps({
            "kind": "answer", "content": "check passed", "complete": True,
        })
        self.assertEqual(
            engine.decide(answer)["roleCommandMap"]["10011"],
            {"action": "submitAnswer", "taskAnswer": "check passed"},
        )

    def test_api_404_does_not_become_success_or_block_revised_request(self):
        # Characterization: exit 0 only completes the process; a different
        # model request can follow HTTP 404, and abandon stays available.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="inspect API and answer",
        )
        engine.decide(active)
        first = copy.deepcopy(active)
        first["roundNo"] = 2
        first["llmResp"] = json.dumps({
            "kind": "command", "content": "curl -i /documented",
        })
        self.assertEqual(engine.decide(first)["executeCmd"], "curl -i /documented")
        failed = copy.deepcopy(active)
        failed["roundNo"] = 3
        failed["lastCmdResult"] = "[exitCode:0]\nHTTP/1.1 404 Not Found"
        prompt = engine.decide(failed)["prompt"]
        self.assertIn("HTTP/1.1 404", prompt)
        self.assertIn("tool process completed with exit code 0", prompt)
        second = copy.deepcopy(active)
        second["roundNo"] = 4
        second["llmResp"] = json.dumps({
            "kind": "command", "content": "curl -i /observed",
        })
        self.assertEqual(engine.decide(second)["executeCmd"], "curl -i /observed")
        later = copy.deepcopy(active)
        later["roundNo"] = 5
        later["lastCmdResult"] = "[exitCode:0]\nHTTP/1.1 404 Not Found"
        self.assertIn("HTTP/1.1 404", engine.decide(later)["prompt"])
        abandon = copy.deepcopy(active)
        abandon["roundNo"] = 6
        abandon["llmResp"] = json.dumps({
            "kind": "abandon", "reason": "No verified answer",
        })
        response = engine.decide(abandon)
        self.assertEqual(response["executeCmd"], "")
        self.assertFalse(any(
            command.get("action") == "submitAnswer"
            for command in response["roleCommandMap"].values()
        ))

    def test_valid_answer_with_placeholder_word_keeps_strict_submit_path(self):
        # Break caught: a global keyword veto rejects an otherwise legal
        # answer merely because the task's literal output contains this word.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="report the literal label",
        )
        engine.decide(active)
        answer = copy.deepcopy(active)
        answer["roundNo"] = 2
        answer["llmResp"] = json.dumps({
            "kind": "answer", "content": "placeholder is the label",
            "complete": True,
        })
        self.assertEqual(engine.decide(answer)["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": "placeholder is the label",
        })

    def test_format_correction_uses_last_round_but_not_elapsed_deadline(self):
        # Break caught: a fixed two-round guard discards a legal last-round submit,
        # or a correction prompt is emitted when no response can arrive in time.
        for remaining in (0, 1, 2):
            with self.subTest(remaining=remaining):
                engine = DecisionEngine()
                active = task_payload(
                    round_no=1, pioneer_pos=(3, 3), phase_task="synthetic task",
                )
                engine.decide(active)
                task = engine.state.state.active_task
                task.timeout_round = 2 + remaining
                invalid = copy.deepcopy(active)
                invalid["roundNo"] = 2
                invalid["llmResp"] = '{"answer":"synthetic"}'

                correction = engine.decide(invalid)

                if remaining == 0:
                    self.assertEqual(correction["prompt"], "")
                    self.assertFalse(task.envelope_correction_requested)
                    self.assertEqual(
                        task.solver_stopped_reason, "deadline_without_answer",
                    )
                    continue
                self.assertIn("one format-only correction", correction["prompt"])
                answer = copy.deepcopy(active)
                answer["roundNo"] = 3
                answer["llmResp"] = (
                    '{"kind":"answer","content":"supported",'
                    '"complete":false}'
                )
                submitted = engine.decide(answer)
                self.assertEqual(submitted["roleCommandMap"]["10011"], {
                    "action": "submitAnswer", "taskAnswer": "supported",
                })

    def test_format_correction_accepts_abandon_and_resets_on_task_replacement(self):
        # Break caught: correction state leaks across task instances or blocks the
        # existing explicit abandon outcome.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="first task",
        )
        engine.decide(active)
        invalid = copy.deepcopy(active)
        invalid["roundNo"] = 2
        invalid["llmResp"] = '{"answer":"synthetic"}'
        engine.decide(invalid)

        abandon = copy.deepcopy(active)
        abandon["roundNo"] = 3
        abandon["llmResp"] = (
            '{"kind":"abandon","reason":"insufficient evidence"}'
        )
        response = engine.decide(abandon)
        self.assertEqual(response["executeCmd"], "")
        self.assertEqual(
            engine.state.state.active_task.solver_stopped_reason,
            "solver_abandoned",
        )

        replacement = copy.deepcopy(active)
        replacement["roundNo"] = 4
        replacement["phaseTask"] = "second task"
        replacement["llmResp"] = ""
        replaced = engine.decide(replacement)
        task = engine.state.state.active_task
        self.assertTrue(replaced["prompt"])
        self.assertFalse(task.envelope_correction_requested)
        self.assertFalse(task.envelope_correction_pending)
        self.assertIsNone(task.last_envelope_rejection)

    def test_late_invalid_envelope_is_not_misassociated_with_current_task_prompt(self):
        # Break caught: a response arriving later than the existing next-round
        # association consumes the one correction chance for the current prompt.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="synthetic task",
        )
        engine.decide(active)
        late = copy.deepcopy(active)
        late["roundNo"] = 3
        late["llmResp"] = '{"answer":"late synthetic"}'

        response = engine.decide(late)

        task = engine.state.state.active_task
        self.assertTrue(response["prompt"])
        self.assertNotIn("one format-only correction", response["prompt"])
        self.assertFalse(task.envelope_correction_requested)
        self.assertEqual(engine.state.state.late_tool_results, 1)

    def test_coordination_last_round_allows_corrected_answer_submission(self):
        # Break caught: the coordination return deadline is either extended by
        # correction or treated as requiring two remaining rounds.
        engine = DecisionEngine()
        active = self._third_post_payload(59)
        engine.decide(active)
        task = engine.state.state.active_task
        self.assertEqual(task.coordination_deadline_round, 62)
        invalid = self._third_post_payload(60)
        invalid["llmResp"] = '{"answer":"synthetic"}'

        correction = engine.decide(invalid)

        self.assertIn("Known remaining task rounds: 2", correction["prompt"])
        self.assertEqual(task.coordination_deadline_round, 62)
        answer = self._third_post_payload(61)
        answer["llmResp"] = (
            '{"kind":"answer","content":"supported",'
            '"complete":true}'
        )
        submitted = engine.decide(answer)
        self.assertEqual(submitted["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": "supported",
        })

    def test_corrected_answer_deferred_by_combat_retries_without_second_llm_call(self):
        # Break caught: allocator rejection loses a corrected answer or prompts the
        # model again instead of retrying the existing deferred submission.
        engine = DecisionEngine()
        active = task_payload(
            round_no=71, pioneer_pos=(3, 3), phase_task="synthetic task",
        )
        active["teamOur"]["roles"].append(
            unit(10020, "gatling", 3, 2, health=1000)
        )
        active["robot"]["roles"] = [{
            "id": 30001,
            "pos": {"x": 3, "y": 5},
            "roleType": "smallRobot",
            "health": 40,
            "abnormalState": "",
            "targetTeam": "challenger",
        }]
        engine.decide(active)
        invalid = copy.deepcopy(active)
        invalid["roundNo"] = 72
        invalid["lastRoundRoleActionResults"] = {"10020": True}
        invalid["llmResp"] = '{"answer":"synthetic"}'
        self.assertTrue(engine.decide(invalid)["prompt"])

        corrected = copy.deepcopy(active)
        corrected["roundNo"] = 73
        corrected["lastRoundRoleActionResults"] = {"10020": True}
        corrected["llmResp"] = (
            '{"kind":"answer","content":"supported",'
            '"complete":true}'
        )
        blocked = engine.decide(corrected)
        self.assertEqual(blocked["roleCommandMap"]["10020"]["action"], "attack")
        self.assertNotIn("10011", blocked["roleCommandMap"])
        self.assertEqual(
            engine.state.state.active_task.deferred_answer, "supported",
        )

        safe = copy.deepcopy(active)
        safe["roundNo"] = 74
        safe["robot"]["roles"] = []
        safe["lastRoundRoleActionResults"] = {"10020": True}
        retried = engine.decide(safe)
        self.assertEqual(retried["prompt"], "")
        self.assertEqual(retried["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": "supported",
        })

    def test_real_engine_runs_accept_prompt_command_result_answer_chain(self):
        # Break caught: task state exists but no real brain path drives the tools.
        engine = DecisionEngine()

        first = task_payload(round_no=1, pioneer_pos=(1, 1))
        response = engine.decide(first)
        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")

        second = task_payload(round_no=2, pioneer_pos=(3, 3))
        second["lastRoundRoleActionResults"] = {"10011": True}
        response = engine.decide(second)
        self.assertEqual(
            response["roleCommandMap"]["10011"], {"action": "acceptTask"}
        )

        third = task_payload(
            round_no=3, pioneer_pos=(3, 3), phase_task="Use the sandbox and answer.",
        )
        third["lastRoundRoleActionResults"] = {"10011": True}
        response = engine.decide(third)
        self.assertTrue(response["prompt"])
        self.assertEqual(response["executeCmd"], "")

        fourth = copy.deepcopy(third)
        fourth["roundNo"] = 4
        fourth["lastRoundRoleActionResults"] = {}
        fourth["llmResp"] = (
            '{"kind":"command","content":"python3 solve.py"}'
        )
        response = engine.decide(fourth)
        self.assertEqual(response["executeCmd"], "python3 solve.py")
        self.assertEqual(response["prompt"], "")

        fifth = copy.deepcopy(third)
        fifth["roundNo"] = 5
        fifth["lastCmdResult"] = "[exitCode:0]\nvalue=42"
        response = engine.decide(fifth)
        self.assertTrue(response["prompt"])
        self.assertIn("exitCode:0", response["prompt"])

        sixth = copy.deepcopy(third)
        sixth["roundNo"] = 6
        sixth["llmResp"] = (
            '{"kind":"answer","content":"42","complete":true}'
        )
        response = engine.decide(sixth)
        self.assertEqual(response["roleCommandMap"]["10011"], {
            "action": "submitAnswer",
            "taskAnswer": "42",
        })

        seventh = task_payload(round_no=7, pioneer_pos=(3, 3))
        seventh["teamOur"]["goldNum"] = 30
        seventh["lastRoundRoleActionResults"] = {"10011": True}
        engine.decide(seventh)
        self.assertIsNone(engine.state.state.active_task)
        self.assertEqual(engine.state.state.ended_tasks[-1].end_reason, "unknown")

    def test_empty_and_failed_tool_results_do_not_become_answers(self):
        # Break caught: tool timeout or truncation is treated as solved output.
        tasks = importlib.import_module("agent.tasks")
        for result in (
            "",
            "[TIMEOUT]\npartial",
            "[JUDGER_ERROR]\nfailed",
            "[exitCode:1]\nfailed",
            "[exitCode:0]\npartial\n[TRUNCATED]",
        ):
            self.assertFalse(tasks.command_result_complete(result))
        self.assertTrue(tasks.command_result_complete("[exitCode:0]\nvalue=42"))

    def test_followup_prompt_carries_bounded_command_and_result_history(self):
        # Break caught: a later prompt assumes an undocumented shared LLM conversation.
        engine = DecisionEngine()
        first = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="find the answer",
        )
        engine.decide(first)

        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["llmResp"] = (
            '{"kind":"command","content":"python3 solve.py --stage one"}'
        )
        self.assertEqual(
            engine.decide(second)["executeCmd"],
            "python3 solve.py --stage one",
        )

        third = copy.deepcopy(first)
        third["roundNo"] = 3
        third["lastCmdResult"] = "[exitCode:0]\nvalue=42"
        prompt = engine.decide(third)["prompt"]

        self.assertIn("python3 solve.py --stage one", prompt)
        self.assertIn("[exitCode:0]\nvalue=42", prompt)

    def test_successful_tool_evidence_survives_recent_history_truncation(self):
        # Break caught: later retries evict the only successful specification read.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="use the specification",
        )
        engine.decide(active)
        engine.state.state.active_task.timeout_round = 100

        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"read-spec"}'
        engine.decide(command)

        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = (
            "[exitCode:0]\nVerified constraint: output must be decimal."
        )
        prompt = engine.decide(result)["prompt"]
        for index in range(3):
            command = copy.deepcopy(active)
            command["roundNo"] = 4 + index * 2
            command["llmResp"] = (
                '{"kind":"command","content":"followup-'
                f'{index}'
                '"}'
            )
            engine.decide(command)
            later_result = copy.deepcopy(active)
            later_result["roundNo"] = 5 + index * 2
            later_result["lastCmdResult"] = (
                f"[exitCode:0]\nlater evidence {index}"
            )
            prompt = engine.decide(later_result)["prompt"]
        for index in range(5):
            command = copy.deepcopy(active)
            command["roundNo"] = 10 + index * 2
            command["llmResp"] = json.dumps({
                "kind": "command", "content": f"evict-history-{index}",
            })
            engine.decide(command)
            later_result = copy.deepcopy(active)
            later_result["roundNo"] = 11 + index * 2
            later_result["lastCmdResult"] = (
                f"[exitCode:0]\neviction evidence {index}"
            )
            prompt = engine.decide(later_result)["prompt"]

        self.assertIn("Verified constraint: output must be decimal.", prompt)
        self.assertLessEqual(
            len(engine.state.state.active_task.solver_evidence), 3,
        )

    def test_read_specification_survives_find_and_later_successes(self):
        # Break caught: keeping only the first and latest success preserves the
        # path discovery but evicts the specification read that defines output.
        engine = DecisionEngine()
        accepted = task_payload(round_no=1, pioneer_pos=(3, 3))
        accepted["teamOur"]["playerTasks"][0]["timeoutRounds"] = 30
        engine.decide(accepted)
        active = copy.deepcopy(accepted)
        active["roundNo"] = 2
        active["phaseTask"] = "discover, read, and solve"
        active["lastRoundRoleActionResults"] = {"10011": True}
        engine.decide(active)
        exchanges = (
            ("find-input", "[exitCode:0]\n/sandbox/input/specification.md"),
            (
                "read-specification",
                "[exitCode:0]\nRequired format: one decimal number.",
            ),
            ("calculate-first", "[exitCode:0]\nintermediate=40"),
            ("calculate-second", "[exitCode:0]\nintermediate=42"),
            ("verify-first", "[exitCode:0]\nverified-part-a"),
            ("verify-second", "[exitCode:0]\nverified-part-b"),
        )
        for index, (command_text, result_text) in enumerate(exchanges):
            command = copy.deepcopy(active)
            command["roundNo"] = 3 + index * 2
            command["llmResp"] = json.dumps({
                "kind": "command", "content": command_text,
            })
            engine.decide(command)
            result = copy.deepcopy(active)
            result["roundNo"] = 4 + index * 2
            result["lastCmdResult"] = result_text
            engine.decide(result)

        partial = copy.deepcopy(active)
        partial["roundNo"] = 15
        partial["llmResp"] = (
            '{"kind":"answer","content":"42", "complete":false}'
        )
        engine.decide(partial)
        rejected = copy.deepcopy(active)
        rejected["roundNo"] = 16
        rejected["lastRoundRoleActionResults"] = {"10011": False}
        rejected["errors"] = [
            {"errorCode": 2, "description": "format incomplete"},
        ]
        prompt = engine.decide(rejected)["prompt"]

        self.assertIn("Required format: one decimal number.", prompt)
        self.assertLessEqual(
            len(engine.state.state.active_task.solver_evidence), 3,
        )

    def test_new_task_reuses_only_paths_from_successful_sandbox_evidence(self):
        # Break caught: verified environment paths are forgotten between task instances.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="first task",
        )
        engine.decide(active)
        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"discover"}'
        engine.decide(command)
        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = (
            "[exitCode:0]\n/opt/task-input/specification.md\n"
            "https://example.test/api\n"
        )
        engine.decide(result)

        next_task = copy.deepcopy(active)
        next_task["roundNo"] = 4
        next_task["phaseTask"] = "second task using specification.md"
        prompt = engine.decide(next_task)["prompt"]

        self.assertIn("/opt/task-input/specification.md", prompt)
        self.assertIn("verify each path for the current task", prompt)
        self.assertNotIn("//example.test/api", prompt)

    def test_success_output_prose_after_path_is_not_cached_as_a_path(self):
        # Break caught: an inline sentence is injected as a verified path clue.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="first task",
        )
        engine.decide(active)
        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"discover"}'
        engine.decide(command)
        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = (
            "[exitCode:0]\n"
            "/tmp/example/workspace/README.md：必须先读取规范，再提交答案"
        )

        engine.decide(result)

        self.assertEqual(engine.state.state.task_environment_paths, [])
        self.assertEqual(engine.state.state.active_task.environment_paths, ())

    def test_standalone_non_ascii_path_remains_current_task_evidence(self):
        # Break caught: conservative filtering rejects every non-ASCII path.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="读取输入文件",
        )
        engine.decide(active)
        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"discover"}'
        engine.decide(command)
        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = "[exitCode:0]\n/sandbox/输入/数据.csv"

        prompt = engine.decide(result)["prompt"]

        self.assertIn("/sandbox/输入/数据.csv", prompt)
        self.assertEqual(
            engine.state.state.task_environment_paths,
            ["/sandbox/输入/数据.csv"],
        )

    def test_unrelated_new_task_does_not_receive_old_specific_path(self):
        # Break caught: every prior engineering path is injected into an API task.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="inspect project files",
        )
        engine.decide(active)
        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"discover"}'
        engine.decide(command)
        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = "[exitCode:0]\n/workspace/project/README.md"
        engine.decide(result)

        next_task = copy.deepcopy(active)
        next_task["roundNo"] = 4
        next_task["phaseTask"] = "Call the documented API and report its status."
        prompt = engine.decide(next_task)["prompt"]

        self.assertNotIn("/workspace/project/README.md", prompt)

    def test_explicit_path_wins_when_old_paths_share_a_filename(self):
        # Break caught: ambiguous same-name history injects both old directories.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="inspect inputs",
        )
        engine.decide(active)
        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"discover"}'
        engine.decide(command)
        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = (
            "[exitCode:0]\n/a/input/specification.md\n/b/input/specification.md"
        )
        engine.decide(result)

        next_task = copy.deepcopy(active)
        next_task["roundNo"] = 4
        next_task["phaseTask"] = "Use the input at /b/input/specification.md."
        prompt = engine.decide(next_task)["prompt"]

        self.assertIn("/b/input/specification.md", prompt)
        self.assertNotIn("/a/input/specification.md", prompt)

    def test_different_explicit_path_does_not_reuse_same_basename(self):
        # Break caught: an explicit new path falls back to an old same-name path.
        tasks = importlib.import_module("agent.tasks")
        memory = TaskMemory(
            "path-instance",
            "prompt",
            environment_paths=("/tmp/old/task.md",),
        )

        text = tasks._environment_path_text(
            memory, "Read /tmp/new/task.md",
        )

        self.assertEqual(text, "(none)")

    def test_explicit_path_matching_uses_whole_path_boundaries(self):
        # Break caught: /foo is treated as explicitly named inside /foobar.
        tasks = importlib.import_module("agent.tasks")
        memory = TaskMemory(
            "path-boundary",
            "prompt",
            environment_paths=("/foo",),
        )

        text = tasks._environment_path_text(memory, "Read /foobar")

        self.assertEqual(text, "(none)")

    def test_failed_sandbox_output_does_not_create_reusable_path_clue(self):
        # Break caught: a failed command's guessed path becomes trusted next-task input.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="first task",
        )
        engine.decide(active)
        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"discover"}'
        engine.decide(command)
        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = "[exitCode:1]\n/guessed/not-real.txt"
        engine.decide(result)

        next_task = copy.deepcopy(active)
        next_task["roundNo"] = 4
        next_task["phaseTask"] = "second task"
        prompt = engine.decide(next_task)["prompt"]

        self.assertNotIn("/guessed/not-real.txt", prompt)

    def test_solver_prompt_gives_unambiguous_file_location_rules(self):
        # Break caught: the solver scans broadly despite an explicit input path.
        payload = task_payload(
            round_no=1,
            pioneer_pos=(3, 3),
            phase_task="The explicit input is /sandbox/input/data.csv; compute the total.",
        )

        prompt = DecisionEngine().decide(payload)["prompt"]

        self.assertIn("explicit file path", prompt)
        self.assertIn("inspect that exact path directly", prompt)
        self.assertIn("only a filename", prompt)
        self.assertIn("bounded filename search", prompt)
        self.assertIn("exactly one task-relevant input", prompt)
        self.assertIn("Do not assume the entire sandbox contains only one file", prompt)
        self.assertIn("consumes two game-round transitions", prompt)
        self.assertIn("combine bounded discovery and the necessary read", prompt)
        self.assertIn("line endings only when sandbox evidence", prompt)
        self.assertIn("change the scope or method using the new evidence", prompt)
        self.assertIn("compare every request field", prompt)
        self.assertIn("exit code 0 does not prove API success", prompt)
        self.assertIn("Do not invent credentials", prompt)
        self.assertIn("outer JSON envelope", prompt)
        self.assertIn("actual check or TOKEN evidence", prompt)
        self.assertIn("actual API response conflict", prompt)
        self.assertNotIn("Bearer", prompt)
        self.assertNotIn("location=", prompt)
        self.assertIn("Observed environment path clues", prompt)
        self.assertNotIn("Previously verified environment paths", prompt)

    def test_solver_contract_explains_raw_json_and_preserves_nested_answer(self):
        nested_answer = '{"result":"synthetic","count":2}'
        for corrected in (False, True):
            with self.subTest(corrected=corrected):
                engine = DecisionEngine()
                active = task_payload(round_no=1, pioneer_pos=(3, 3),
                                      phase_task="return JSON")
                prompt = engine.decide(active)["prompt"]
                if corrected:
                    invalid = copy.deepcopy(active)
                    invalid["roundNo"] = 2
                    invalid["llmResp"] = "not JSON"
                    prompt = engine.decide(invalid)["prompt"]
                self.assertIn("raw JSON", prompt)
                self.assertIn("no markdown fences", prompt)
                self.assertIn("no extra fields", prompt)
                self.assertIn("escape", prompt)
                answer = copy.deepcopy(active)
                answer["roundNo"] = 3 if corrected else 2
                answer["llmResp"] = json.dumps({
                    "kind": "answer", "content": nested_answer,
                    "complete": True,
                })
                response = engine.decide(answer)
                self.assertEqual(response["roleCommandMap"]["10011"], {
                    "action": "submitAnswer", "taskAnswer": nested_answer,
                })

    def test_known_deadline_prompts_report_state_machine_tool_cycle_ceiling(self):
        # Break caught: the prompt reports raw rounds as if each were a usable
        # command cycle, ignoring command/result transitions and final submission.
        for timeout, expected_remaining, expected_cycles in (
            (5, 4, 1),
            (6, 5, 1),
            (7, 6, 2),
        ):
            with self.subTest(timeout=timeout):
                engine = DecisionEngine()
                accepted = task_payload(round_no=1, pioneer_pos=(3, 3))
                accepted["teamOur"]["teamId"] = f"cycle-budget-{timeout}"
                accepted["teamOur"]["playerTasks"][0]["timeoutRounds"] = timeout
                engine.decide(accepted)
                active = copy.deepcopy(accepted)
                active["roundNo"] = 2
                active["phaseTask"] = "bounded task"
                active["lastRoundRoleActionResults"] = {"10011": True}

                prompt = engine.decide(active)["prompt"]

                self.assertIn(
                    f"Known remaining task rounds: {expected_remaining}", prompt,
                )
                self.assertIn(
                    f"Maximum remaining tool cycles: {expected_cycles}", prompt,
                )

    def test_coordination_deadline_and_final_only_reduce_tool_cycle_ceiling(self):
        # Break caught: cycle advice ignores an earlier return deadline or a
        # final-answer state even though execution already honors both.
        engine = DecisionEngine()
        active = self._third_post_payload(50)
        prompt = engine.decide(active)["prompt"]
        self.assertIn("Known remaining task rounds: 12", prompt)
        self.assertIn("Maximum remaining tool cycles: 5", prompt)

        engine.state.state.active_task.final_answer_requested = True
        retry = self._third_post_payload(51)
        retry["llmResp"] = "invalid-one"
        prompt = engine.decide(retry)["prompt"]
        self.assertIn("Maximum remaining tool cycles: 0", prompt)

    def test_unknown_deadline_tool_cycle_ceiling_tracks_existing_four_cycle_cap(self):
        # Break caught: unknown-deadline guidance advertises a fixed four cycles
        # after some of that existing safety allowance has already been consumed.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        self.assertIn("Maximum remaining tool cycles: 4", engine.decide(active)["prompt"])

        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"inspect"}'
        engine.decide(command)
        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = "[exitCode:0]\nevidence"

        prompt = engine.decide(result)["prompt"]

        self.assertIn("Maximum remaining tool cycles: 3", prompt)

    def test_failed_observations_survive_history_eviction_but_remain_task_local(self):
        # Break caught: failed command/result pairs disappear after eight recent
        # events, or leak into a replacement task as trusted cross-task context.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="first task",
        )
        task = engine.decide(active)
        self.assertTrue(task["prompt"])
        for index in range(3):
            command = copy.deepcopy(active)
            command["roundNo"] = 2 + index * 2
            command["llmResp"] = json.dumps({
                "kind": "command", "content": f"failed-command-{index}",
            })
            engine.decide(command)
            result = copy.deepcopy(active)
            result["roundNo"] = 3 + index * 2
            result["lastCmdResult"] = f"[exitCode:1]\nfailed-result-{index}"
            prompt = engine.decide(result)["prompt"]

        tasks = importlib.import_module("agent.tasks")
        task_memory = engine.state.state.active_task
        for index in range(9):
            tasks._remember(
                task_memory,
                f"History filler {index}",
                "x" * 1_000,
            )
        prompt = tasks._solver_prompt(
            Turn.load(active), task_memory, "continue from retained evidence",
        )

        self.assertIn("Recent failed or incomplete command observations", prompt)
        self.assertNotIn("failed-command-0", prompt)
        self.assertNotIn("failed-result-0", prompt)
        self.assertIn("failed-command-1", prompt)
        self.assertIn("failed-result-1", prompt)
        self.assertIn("failed-command-2", prompt)
        self.assertIn("failed-result-2", prompt)
        self.assertEqual(engine.state.state.active_task.solver_evidence, [])
        self.assertEqual(engine.state.state.active_task.environment_paths, ())

        replacement = copy.deepcopy(active)
        replacement["roundNo"] = 8
        replacement["phaseTask"] = "replacement task"
        replacement["llmResp"] = ""
        prompt = engine.decide(replacement)["prompt"]
        self.assertNotIn("failed-command-1", prompt)
        self.assertNotIn("failed-result-2", prompt)

    def test_failed_observations_are_deduplicated_and_separately_bounded(self):
        # Break caught: a long command consumes the whole pair budget, duplicate
        # pairs multiply, or a successful result enters the failure-only memory.
        tasks = importlib.import_module("agent.tasks")
        memory = TaskMemory("failure-bounds", "prompt")
        long_command = "c" * 4_096
        long_result = "[exitCode:1]\n" + "r" * 20_000

        tasks._remember_failed_observation(memory, long_command, long_result)
        tasks._remember_failed_observation(memory, long_command, long_result)

        self.assertEqual(len(memory.failed_tool_observations), 1)
        command, result = memory.failed_tool_observations[0]
        self.assertTrue(command.endswith("[TRUNCATED]"))
        self.assertIn("[TRUNCATED MIDDLE]", result)
        self.assertTrue(result.endswith("r" * 100))
        rendered = tasks._failed_observations_text(memory)
        self.assertLessEqual(len(rendered), tasks.MAX_FAILED_OBSERVATION_CHARS)
        self.assertIn("c" * 100, rendered)
        self.assertIn("r" * 100, rendered)

        tasks._remember_failed_observation(
            memory, "timed-out", "[TIMEOUT]\npartial",
        )
        self.assertIn("[TIMEOUT]", tasks._failed_observations_text(memory))
        tasks._remember_failed_observation(
            memory, "empty-output", "",
        )
        tasks._remember_failed_observation(
            memory, "truncated-output", "[exitCode:0]\npartial\n[TRUNCATED]",
        )
        self.assertEqual(len(memory.failed_tool_observations), 2)
        self.assertNotIn(long_command[:100], tasks._failed_observations_text(memory))
        self.assertIn("empty-output", tasks._failed_observations_text(memory))
        self.assertIn("truncated-output", tasks._failed_observations_text(memory))

    def test_long_failed_result_preserves_head_and_terminal_error(self):
        # A progress-heavy command must retain both its exit status and the
        # actionable terminal error within the existing bounded prompt budget.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="repair the script",
        )
        engine.decide(active)
        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"python3 solve.py"}'
        engine.decide(command)
        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = (
            "[exitCode:1]\nstarting solver\n"
            + "progress\n" * 2_000
            + "IndentationError: unexpected indent at solve.py:19"
        )

        prompt = engine.decide(result)["prompt"]
        task = engine.state.state.active_task
        tasks = importlib.import_module("agent.tasks")
        context = prompt.split(
            "The platform sandbox result was empty, failed, timed out, or truncated.",
            1,
        )[1].split("\nObserved environment path clues", 1)[0]
        history = tasks._solver_history_text(task)
        failed_observations = tasks._failed_observations_text(task)
        rendered_context = tasks._tool_result_context(
            "The platform sandbox result was empty, failed, timed out, or truncated.",
            result["lastCmdResult"],
        )

        self.assertIn("[exitCode:1]", prompt)
        self.assertIn("starting solver", prompt)
        self.assertIn("[TRUNCATED MIDDLE]", prompt)
        terminal_error = "IndentationError: unexpected indent at solve.py:19"
        self.assertIn(terminal_error, context)
        self.assertIn(terminal_error, history)
        self.assertIn(terminal_error, failed_observations)
        self.assertLessEqual(len(rendered_context), tasks.MAX_TOOL_CONTEXT_CHARS)
        self.assertLessEqual(
            len(task.failed_tool_observations[-1][1]),
            tasks.MAX_FAILED_RESULT_CHARS,
        )

    def test_short_tool_result_context_is_unchanged(self):
        tasks = importlib.import_module("agent.tasks")
        status = "tool status"
        result = "[exitCode:1]\nshort error"

        self.assertEqual(
            tasks._tool_result_context(status, result),
            f"{status}\nPlatform result:\n{result}",
        )

    def test_two_nonzero_failures_block_third_identical_command_for_one_correction(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="repair then answer",
        )
        engine.decide(active)

        for index, detail in enumerate(("first failure", "changed second failure")):
            command = copy.deepcopy(active)
            command["roundNo"] = 2 + index * 2
            command["llmResp"] = (
                '{"kind":"command","content":"python3 solve.py"}'
            )
            self.assertEqual(engine.decide(command)["executeCmd"], "python3 solve.py")
            result = copy.deepcopy(active)
            result["roundNo"] = 3 + index * 2
            result["lastCmdResult"] = f"[exitCode:1]\n{detail}"
            engine.decide(result)

        third = copy.deepcopy(active)
        third["roundNo"] = 6
        third["llmResp"] = '{"kind":"command","content":"python3 solve.py"}'
        correction = engine.decide(third)
        task = engine.state.state.active_task

        self.assertEqual(correction["executeCmd"], "")
        self.assertIn("one repeated-command correction", correction["prompt"])
        self.assertTrue(task.repeated_command_correction_requested)
        self.assertEqual(task.command_count, 2)

        repeated = copy.deepcopy(active)
        repeated["roundNo"] = 7
        repeated["llmResp"] = '{"kind":"command","content":"python3 solve.py"}'
        stopped = engine.decide(repeated)
        self.assertEqual(stopped["executeCmd"], "")
        self.assertEqual(task.solver_stopped_reason, "repeated_failed_command")

    def test_different_executed_command_resets_nonzero_failure_chain(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="repair then answer",
        )
        engine.decide(active)

        round_no = 2
        for detail in ("failure one", "failure two"):
            command = copy.deepcopy(active)
            command["roundNo"] = round_no
            command["llmResp"] = '{"kind":"command","content":"python3 solve.py"}'
            engine.decide(command)
            result = copy.deepcopy(active)
            result["roundNo"] = round_no + 1
            result["lastCmdResult"] = f"[exitCode:2]\n{detail}"
            engine.decide(result)
            round_no += 2

        changed = copy.deepcopy(active)
        changed["roundNo"] = 6
        changed["llmResp"] = '{"kind":"command","content":"sed -n 1,80p solve.py"}'
        self.assertEqual(
            engine.decide(changed)["executeCmd"], "sed -n 1,80p solve.py",
        )
        fixed = copy.deepcopy(active)
        fixed["roundNo"] = 7
        fixed["lastCmdResult"] = "[exitCode:1]\ninspection command failed too"
        engine.decide(fixed)
        retried = copy.deepcopy(active)
        retried["roundNo"] = 8
        retried["llmResp"] = '{"kind":"command","content":"python3 solve.py"}'
        self.assertEqual(engine.decide(retried)["executeCmd"], "python3 solve.py")

    def test_repeated_command_correction_can_submit_reliable_partial_answer(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="bounded task",
        )
        engine.decide(active)
        task = engine.state.state.active_task
        task.consecutive_nonzero_command = "same"
        task.consecutive_nonzero_count = 2

        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"same"}'
        correction = engine.decide(command)
        self.assertIn("one repeated-command correction", correction["prompt"])

        answer = copy.deepcopy(active)
        answer["roundNo"] = 3
        answer["llmResp"] = (
            '{"kind":"answer","content":"supported partial",'
            '"complete":false}'
        )
        response = engine.decide(answer)
        self.assertEqual(response["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": "supported partial",
        })

    def test_unknown_result_does_not_count_as_second_nonzero_failure(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="bounded task",
        )
        engine.decide(active)
        first = copy.deepcopy(active)
        first["roundNo"] = 2
        first["llmResp"] = '{"kind":"command","content":"same"}'
        engine.decide(first)
        failed = copy.deepcopy(active)
        failed["roundNo"] = 3
        failed["lastCmdResult"] = "[exitCode:1]\nfailed"
        engine.decide(failed)
        second = copy.deepcopy(active)
        second["roundNo"] = 4
        second["llmResp"] = '{"kind":"command","content":"same"}'
        engine.decide(second)
        unknown = copy.deepcopy(active)
        unknown["roundNo"] = 5
        unknown["lastCmdResult"] = "runner status unavailable"
        engine.decide(unknown)
        third = copy.deepcopy(active)
        third["roundNo"] = 6
        third["llmResp"] = '{"kind":"command","content":"same"}'

        self.assertEqual(engine.decide(third)["executeCmd"], "same")
        self.assertFalse(
            engine.state.state.active_task.repeated_command_correction_requested,
        )

    def test_exit_zero_is_tool_completion_not_verified_business_success(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="call the API",
        )
        engine.decide(active)
        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"curl endpoint"}'
        engine.decide(command)
        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = "[exitCode:0]\nHTTP 401 Authorization failed"

        prompt = engine.decide(result)["prompt"]

        self.assertIn("tool process completed with exit code 0", prompt)
        self.assertIn("Complete tool output or observation", prompt)
        self.assertNotIn("Verified sandbox result", prompt)
        self.assertNotIn("completed successfully", prompt)

    def test_bounded_exit_zero_output_is_not_labeled_complete_evidence(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="inspect long output",
        )
        engine.decide(active)
        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"inspect"}'
        engine.decide(command)
        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = (
            "[exitCode:0]\nhead\n" + "middle\n" * 1_000 + "tail"
        )

        prompt = engine.decide(result)["prompt"]

        self.assertIn("Bounded tool output or observation", prompt)
        self.assertIn("[TRUNCATED MIDDLE]", prompt)
        self.assertNotIn("Complete tool output or observation", prompt)

    def test_deadline_final_only_precedes_repeated_command_correction(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="deadline task",
        )
        engine.decide(active)
        task = engine.state.state.active_task
        task.timeout_round = 4
        task.consecutive_nonzero_command = "same"
        task.consecutive_nonzero_count = 2
        task.final_answer_requested = True

        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = '{"kind":"command","content":"same"}'
        response = engine.decide(command)

        self.assertEqual(response["executeCmd"], "")
        self.assertIn("final-answer-only correction", response["prompt"])
        self.assertFalse(task.repeated_command_correction_requested)

    def test_failed_observations_do_not_cross_session_and_exit_zero_is_not_reclassified(self):
        # Break caught: failure-only memory survives a team/session boundary, or
        # brittle text matching reclassifies an exit-zero HTTP response as failure.
        tasks = importlib.import_module("agent.tasks")
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="API task",
        )
        engine.decide(active)
        tasks._remember_failed_observation(
            engine.state.state.active_task,
            "curl documented-endpoint",
            "[exitCode:1]\nconnection failed",
        )

        command = copy.deepcopy(active)
        command["roundNo"] = 2
        command["llmResp"] = (
            '{"kind":"command","content":"curl documented-endpoint"}'
        )
        engine.decide(command)
        result = copy.deepcopy(active)
        result["roundNo"] = 3
        result["lastCmdResult"] = "[exitCode:0]\nHTTP/1.1 401 Unauthorized"
        prompt = engine.decide(result)["prompt"]
        self.assertIn("HTTP/1.1 401 Unauthorized", prompt)
        self.assertEqual(
            len(engine.state.state.active_task.failed_tool_observations), 1,
        )

        restarted = copy.deepcopy(active)
        restarted["roundNo"] = 1
        restarted["teamOur"]["teamId"] = "task-tests-new-session"
        restarted["llmResp"] = ""
        prompt = engine.decide(restarted)["prompt"]
        self.assertEqual(engine.state.state.session_index, 2)
        self.assertEqual(
            engine.state.state.active_task.failed_tool_observations, [],
        )
        self.assertNotIn("connection failed", prompt)

    def test_known_deadline_is_in_every_prompt_and_blocks_late_commands(self):
        # Break caught: normal command/result branches bypass the deadline policy.
        engine = DecisionEngine()
        accepted = task_payload(round_no=1, pioneer_pos=(3, 3))
        accepted["teamOur"]["playerTasks"][0]["timeoutRounds"] = 6
        engine.decide(accepted)

        active = copy.deepcopy(accepted)
        active["roundNo"] = 2
        active["phaseTask"] = "active task"
        active["lastRoundRoleActionResults"] = {"10011": True}
        prompt = engine.decide(active)["prompt"]
        self.assertIn("Known remaining task rounds: 5", prompt)
        self.assertIn("Command exploration is allowed", prompt)

        command = copy.deepcopy(active)
        command["roundNo"] = 3
        command["llmResp"] = '{"kind":"command","content":"step-one"}'
        self.assertEqual(engine.decide(command)["executeCmd"], "step-one")

        result = copy.deepcopy(active)
        result["roundNo"] = 4
        result["lastCmdResult"] = "[exitCode:0]\nprogress"
        prompt = engine.decide(result)["prompt"]
        self.assertIn("Known remaining task rounds: 3", prompt)
        self.assertIn("Command exploration is not allowed", prompt)

        answer = copy.deepcopy(active)
        answer["roundNo"] = 5
        answer["llmResp"] = (
            '{"kind":"answer","content":"supported partial",'
            '"complete":false}'
        )
        response = engine.decide(answer)
        self.assertEqual(response["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": "supported partial",
        })

    def test_command_requested_with_four_rounds_left_is_not_dropped_in_flight(self):
        # Break caught: the remaining-three cutoff rejects a command requested
        # by the immediately preceding, still-valid remaining-four prompt.
        engine = DecisionEngine()
        accepted = task_payload(round_no=1, pioneer_pos=(3, 3))
        accepted["teamOur"]["playerTasks"][0]["timeoutRounds"] = 6
        engine.decide(accepted)

        active = copy.deepcopy(accepted)
        active["roundNo"] = 2
        active["phaseTask"] = "active task"
        active["lastRoundRoleActionResults"] = {"10011": True}
        prompt = engine.decide(active)["prompt"]
        self.assertIn("Known remaining task rounds: 5", prompt)
        self.assertIn("Command exploration is allowed", prompt)

        in_flight = copy.deepcopy(active)
        in_flight["roundNo"] = 3
        in_flight["llmResp"] = (
            '{"kind":"command","content":"final-evidence-read"}'
        )
        self.assertEqual(
            engine.decide(in_flight)["executeCmd"], "final-evidence-read",
        )

    def test_unknown_deadline_stops_after_four_distinct_command_cycles(self):
        # Break caught: changing output permits unbounded tool exploration.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        engine.decide(active)

        prompt = ""
        for index in range(4):
            command = copy.deepcopy(active)
            command["roundNo"] = 2 + index * 2
            command["llmResp"] = (
                '{"kind":"command","content":"step-'
                f'{index}'
                '"}'
            )
            engine.decide(command)
            result = copy.deepcopy(active)
            result["roundNo"] = 3 + index * 2
            result["lastCmdResult"] = f"[exitCode:0]\nresult-{index}"
            prompt = engine.decide(result)["prompt"]

        self.assertIn("No more command exploration", prompt)

    def test_known_deadline_allows_a_fifth_cycle_when_budget_is_sufficient(self):
        # Break caught: the unknown-deadline safety cap also truncates a known,
        # still-feasible task whose evidence is continuing to change.
        engine = DecisionEngine()
        accepted = task_payload(round_no=1, pioneer_pos=(3, 3))
        accepted["teamOur"]["playerTasks"][0]["timeoutRounds"] = 20
        engine.decide(accepted)

        active = copy.deepcopy(accepted)
        active["roundNo"] = 2
        active["phaseTask"] = "known long task"
        active["lastRoundRoleActionResults"] = {"10011": True}
        engine.decide(active)

        for index in range(4):
            command = copy.deepcopy(active)
            command["roundNo"] = 3 + index * 2
            command["llmResp"] = (
                '{"kind":"command","content":"known-step-'
                f'{index}'
                '"}'
            )
            self.assertEqual(
                engine.decide(command)["executeCmd"], f"known-step-{index}",
            )
            result = copy.deepcopy(active)
            result["roundNo"] = 4 + index * 2
            result["lastCmdResult"] = f"[exitCode:0]\nchanged-{index}"
            prompt = engine.decide(result)["prompt"]

        self.assertIn("Known remaining task rounds: 11", prompt)
        self.assertIn("Command exploration is allowed", prompt)

        fifth = copy.deepcopy(active)
        fifth["roundNo"] = 11
        fifth["llmResp"] = (
            '{"kind":"command","content":"known-step-4"}'
        )
        self.assertEqual(engine.decide(fifth)["executeCmd"], "known-step-4")

    def test_fourteen_round_chain_converges_and_preserves_partial_feedback(self):
        # Break caught: a changing command loop consumes the whole task window,
        # or an incomplete submission prevents a final evidence-based revision.
        engine = DecisionEngine()
        accepted = task_payload(round_no=1, pioneer_pos=(3, 3))
        accepted["teamOur"]["playerTasks"][0]["timeoutRounds"] = 12
        engine.decide(accepted)

        active = copy.deepcopy(accepted)
        active["roundNo"] = 2
        active["phaseTask"] = "synthetic bounded task"
        active["lastRoundRoleActionResults"] = {"10011": True}
        engine.decide(active)

        for index in range(4):
            command = copy.deepcopy(active)
            command["roundNo"] = 3 + index * 2
            command["llmResp"] = (
                '{"kind":"command","content":"bounded-step-'
                f'{index}'
                '"}'
            )
            self.assertEqual(
                engine.decide(command)["executeCmd"], f"bounded-step-{index}",
            )
            result = copy.deepcopy(active)
            result["roundNo"] = 4 + index * 2
            result["lastCmdResult"] = f"[exitCode:0]\nevidence-{index}"
            response = engine.decide(result)

        self.assertEqual(result["roundNo"], 10)
        self.assertIn("Known remaining task rounds: 3", response["prompt"])
        self.assertIn("Command exploration is not allowed", response["prompt"])

        partial = copy.deepcopy(active)
        partial["roundNo"] = 11
        partial["llmResp"] = (
            '{"kind":"answer","content":"supported partial",'
            '"complete":false}'
        )
        response = engine.decide(partial)
        self.assertEqual(
            response["roleCommandMap"]["10011"]["action"], "submitAnswer",
        )

        rejected = copy.deepcopy(active)
        rejected["roundNo"] = 12
        rejected["lastRoundRoleActionResults"] = {"10011": False}
        rejected["errors"] = [
            {"errorCode": 2, "description": "incomplete"},
        ]
        response = engine.decide(rejected)
        self.assertIn("previous submission failed or was incomplete", response["prompt"])
        self.assertIn("supported partial", response["prompt"])

        revised = copy.deepcopy(active)
        revised["roundNo"] = 13
        revised["llmResp"] = (
            '{"kind":"answer","content":"supported revision",'
            '"complete":true}'
        )
        response = engine.decide(revised)
        self.assertEqual(response["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": "supported revision",
        })

        finished = copy.deepcopy(active)
        finished["roundNo"] = 14
        finished["phaseTask"] = ""
        finished["lastRoundRoleActionResults"] = {"10011": True}
        response = engine.decide(finished)
        self.assertEqual(response["executeCmd"], "")

    def test_evidence_based_answer_can_submit_on_last_known_round(self):
        # Break caught: deadline handling stops a previously requested answer too early.
        engine = DecisionEngine()
        accepted = task_payload(round_no=1, pioneer_pos=(3, 3))
        accepted["teamOur"]["playerTasks"][0]["timeoutRounds"] = 6
        engine.decide(accepted)

        active = copy.deepcopy(accepted)
        active["roundNo"] = 2
        active["phaseTask"] = "active task"
        active["lastRoundRoleActionResults"] = {"10011": True}
        engine.decide(active)

        near_deadline = copy.deepcopy(active)
        near_deadline["roundNo"] = 6
        prompt = engine.decide(near_deadline)["prompt"]
        self.assertIn("Known remaining task rounds: 1", prompt)

        final = copy.deepcopy(active)
        final["roundNo"] = 7
        final["llmResp"] = (
            '{"kind":"answer","content":"supported partial",'
            '"complete":false}'
        )
        response = engine.decide(final)
        self.assertEqual(response["roleCommandMap"]["10011"], {
            "action": "submitAnswer",
            "taskAnswer": "supported partial",
        })

    def test_repeated_command_result_cycle_converges_then_leaves(self):
        # Break caught: alternating LLM and command result kinds reset repetition,
        # or a third identical explicitly failing command reaches the sandbox.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        self.assertIn("remaining task rounds: unknown", engine.decide(active)["prompt"])

        for command_round, result_round in ((2, 3), (4, 5)):
            command = copy.deepcopy(active)
            command["roundNo"] = command_round
            command["llmResp"] = '{"kind":"command","content":"pwd"}'
            self.assertEqual(engine.decide(command)["executeCmd"], "pwd")

            result = copy.deepcopy(active)
            result["roundNo"] = result_round
            result["lastCmdResult"] = "[exitCode:1]\nsame failure"
            engine.decide(result)

        third = copy.deepcopy(active)
        third["roundNo"] = 6
        third["llmResp"] = '{"kind":"command","content":"pwd"}'
        response = engine.decide(third)
        self.assertEqual(response["executeCmd"], "")
        self.assertIn("one repeated-command correction", response["prompt"])

        ignored = copy.deepcopy(active)
        ignored["roundNo"] = 7
        ignored["llmResp"] = '{"kind":"command","content":"pwd"}'
        response = engine.decide(ignored)
        self.assertEqual(response["executeCmd"], "")
        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")
        self.assertIsNotNone(engine.state.state.active_task.solver_stopped_reason)

    def test_unissued_leave_is_retried_after_task_exit_unblocks(self):
        # Break caught: a temporarily impossible leave consumes the only attempt.
        engine = DecisionEngine()
        blocked = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        blocked["mapInfo"]["zones"].extend(
            {"pos": {"x": x, "y": y}, "neutralType": "vendor"}
            for x, y in ((2, 2), (2, 3), (2, 4), (3, 2), (4, 2))
        )
        engine.decide(blocked)
        engine.state.state.active_task.solver_stopped_reason = "test_stop"

        still_blocked = copy.deepcopy(blocked)
        still_blocked["roundNo"] = 2
        response = engine.decide(still_blocked)
        self.assertNotIn("10011", response["roleCommandMap"])
        self.assertFalse(engine.state.state.active_task.abandon_move_attempted)

        opened = task_payload(
            round_no=3, pioneer_pos=(3, 3), phase_task="active task",
        )
        response = engine.decide(opened)
        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")
        self.assertTrue(engine.state.state.active_task.abandon_move_attempted)

    def test_leave_retries_after_same_round_defense_uses_owner(self):
        # Break caught: an allocator conflict counts an unissued task leave as attempted.
        engine = DecisionEngine()
        active = task_payload(
            round_no=71, pioneer_pos=(3, 3), phase_task="active task",
        )
        active["teamOur"]["roles"].extend([
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 3, 2, health=1000),
        ])
        active["robot"]["roles"] = [{
            "id": 30001,
            "pos": {"x": 3, "y": 5},
            "roleType": "smallRobot",
            "health": 40,
            "abnormalState": "",
            "targetTeam": "challenger",
        }]
        engine.decide(active)
        engine.state.state.active_task.solver_stopped_reason = "test_stop"

        defended = copy.deepcopy(active)
        defended["roundNo"] = 72
        response = engine.decide(defended)
        self.assertEqual(response["roleCommandMap"]["10020"]["action"], "attack")
        self.assertFalse(engine.state.state.active_task.abandon_move_attempted)

        safe = copy.deepcopy(active)
        safe["roundNo"] = 73
        safe["robot"]["roles"] = []
        response = engine.decide(safe)
        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")
        self.assertTrue(engine.state.state.active_task.abandon_move_attempted)

    def test_changed_command_result_is_progress_not_a_repeated_cycle(self):
        # Break caught: any repeated command is stopped despite changing evidence.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        engine.decide(active)
        for index, output in enumerate(("first", "second", "first")):
            command = copy.deepcopy(active)
            command["roundNo"] = 2 + index * 2
            command["llmResp"] = '{"kind":"command","content":"probe"}'
            engine.decide(command)
            result = copy.deepcopy(active)
            result["roundNo"] = 3 + index * 2
            result["lastCmdResult"] = f"[exitCode:0]\n{output}"
            response = engine.decide(result)

        self.assertTrue(response["prompt"])
        self.assertNotIn("No more command exploration", response["prompt"])
        self.assertIsNone(engine.state.state.active_task.solver_stopped_reason)

    def test_task_prompt_marks_task_text_truncation(self):
        # Break caught: an oversized task is silently clipped into a different problem.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1,
            pioneer_pos=(3, 3),
            phase_task="x" * 40_000,
        )

        prompt = engine.decide(active)["prompt"]

        self.assertIn("[TRUNCATED]", prompt)

    def test_solver_history_is_bounded_and_marks_omitted_events(self):
        # Break caught: self-contained prompts grow without a strict memory ceiling.
        tasks = importlib.import_module("agent.tasks")
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        engine.decide(active)
        engine.state.state.active_task.timeout_round = 100
        for index in range(10):
            command = copy.deepcopy(active)
            command["roundNo"] = 2 + index * 2
            command["llmResp"] = json.dumps({
                "kind": "command",
                "content": f"history-{index}-" + "x" * 1_000,
            })
            engine.decide(command)
            result = copy.deepcopy(active)
            result["roundNo"] = 3 + index * 2
            result["lastCmdResult"] = (
                f"[exitCode:0]\nhistory-result-{index}-" + "y" * 5_000
            )
            engine.decide(result)

        task = engine.state.state.active_task
        self.assertLessEqual(len(task.solver_history), tasks.MAX_SOLVER_EVENTS)
        self.assertTrue(task.solver_history_truncated)
        self.assertTrue(all(
            len(event) <= tasks.MAX_SOLVER_EVENT_CHARS
            for event in task.solver_history
        ))
        history = tasks._solver_history_text(task)
        self.assertLessEqual(len(history), tasks.MAX_SOLVER_HISTORY_CHARS)
        self.assertIn("[TRUNCATED", history)

    def test_submission_feedback_prompt_carries_previous_answer(self):
        # Break caught: an improvement call cannot see what answer was rejected.
        engine = DecisionEngine()
        first = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        engine.decide(first)

        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["llmResp"] = (
            '{"kind":"answer","content":"candidate-42","complete":false}'
        )
        engine.decide(second)

        third = copy.deepcopy(first)
        third["roundNo"] = 3
        third["lastRoundRoleActionResults"] = {"10011": False}
        third["errors"] = [{"errorCode": 2, "description": "wrong answer"}]
        prompt = engine.decide(third)["prompt"]

        self.assertIn("candidate-42", prompt)
        self.assertIn("wrong answer", prompt)

        fourth = copy.deepcopy(first)
        fourth["roundNo"] = 4
        fourth["llmResp"] = "invalid follow-up"
        prompt = engine.decide(fourth)["prompt"]
        self.assertIn("candidate-42", prompt)
        self.assertIn("wrong answer", prompt)

    def test_task_end_reasons_separate_timeout_death_and_left_point(self):
        # Break caught: every disappearing phaseTask is mislabeled successful.
        cases = (
            ("timeout", [{"errorCode": 1, "description": "timeout"}], 200, (3, 3)),
            ("death", [], 0, (3, 3)),
            ("left_point", [], 200, (10, 10)),
        )
        for expected, errors, health, position in cases:
            engine = DecisionEngine()
            active = task_payload(
                round_no=1, pioneer_pos=(3, 3), phase_task="same real task",
            )
            engine.decide(active)
            ended = task_payload(round_no=2, pioneer_pos=position)
            ended["teamOur"]["roles"][0]["health"] = health
            ended["errors"] = errors
            engine.decide(ended)
            self.assertEqual(
                engine.state.state.ended_tasks[-1].end_reason,
                expected,
            )

    def test_three_towers_do_not_take_pioneer_from_daytime_task(self):
        # Break caught: C's third-gunner rule permanently blocks task collection.
        payload = task_payload(round_no=1, pioneer_pos=(1, 1))
        payload["teamOur"]["roles"].extend([
            unit(10010, "worker", 6, 5),
            unit(10012, "worker", 9, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 9, 6, health=1000),
            unit(10040, "rocket", 12, 6, health=1000),
        ])

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")

    def test_task_is_not_accepted_without_solver_and_return_window(self):
        # Break caught: R58 accepts a task that cannot finish one cycle and return.
        payload = task_payload(round_no=58, pioneer_pos=(3, 3))
        payload["teamOur"]["teamId"] = "task-preaccept-window"
        payload["teamOur"]["roles"].extend([
            unit(10010, "worker", 6, 5),
            unit(10012, "worker", 9, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 9, 6, health=1000),
            unit(10040, "rocket", 16, 6, health=1000),
        ])
        traces = []

        response = DecisionEngine().decide(payload, trace_sink=traces.append)

        pioneer_command = response["roleCommandMap"].get("10011")
        self.assertTrue(
            pioneer_command is None or pioneer_command["action"] != "acceptTask"
        )
        self.assertEqual(
            traces[0]["taskStartSkipReason"], "insufficient_solver_return_window",
        )

    def test_timeout_four_task_is_rejected_before_first_final_only_prompt(self):
        # Break caught: R1 accept with timeout 4 makes the first R2 prompt final-only.
        payload = task_payload(round_no=1, pioneer_pos=(3, 3))
        payload["teamOur"]["teamId"] = "task-preaccept-timeout-four"
        payload["teamOur"]["playerTasks"][0]["timeoutRounds"] = 4
        traces = []

        response = DecisionEngine().decide(payload, trace_sink=traces.append)

        pioneer_command = response["roleCommandMap"].get("10011")
        self.assertTrue(
            pioneer_command is None or pioneer_command["action"] != "acceptTask"
        )
        self.assertEqual(
            traces[0]["taskStartSkipReason"], "insufficient_solver_window",
        )

    def test_timeout_five_allows_one_command_result_and_answer_chain(self):
        # Break caught: the corrected minimum blocks the first feasible boundary.
        engine = DecisionEngine()
        accepted = task_payload(round_no=1, pioneer_pos=(3, 3))
        accepted["teamOur"]["teamId"] = "task-preaccept-timeout-five"
        accepted["teamOur"]["playerTasks"][0]["timeoutRounds"] = 5

        response = engine.decide(accepted)
        self.assertEqual(
            response["roleCommandMap"]["10011"]["action"], "acceptTask",
        )

        active = copy.deepcopy(accepted)
        active["roundNo"] = 2
        active["phaseTask"] = "active task"
        active["lastRoundRoleActionResults"] = {"10011": True}
        prompt = engine.decide(active)["prompt"]
        self.assertIn("Known remaining task rounds: 4", prompt)
        self.assertIn("Command exploration is allowed", prompt)

        command = copy.deepcopy(active)
        command["roundNo"] = 3
        command["llmResp"] = '{"kind":"command","content":"inspect"}'
        self.assertEqual(engine.decide(command)["executeCmd"], "inspect")

        result = copy.deepcopy(active)
        result["roundNo"] = 4
        result["lastCmdResult"] = "[exitCode:0]\nevidence"
        prompt = engine.decide(result)["prompt"]
        self.assertIn("Command exploration is not allowed", prompt)

        answer = copy.deepcopy(active)
        answer["roundNo"] = 5
        answer["llmResp"] = (
            '{"kind":"answer","content":"supported",'
            '"complete":true}'
        )
        response = engine.decide(answer)
        self.assertEqual(response["roleCommandMap"]["10011"], {
            "action": "submitAnswer", "taskAnswer": "supported",
        })

    def test_rejected_task_window_still_allows_news_and_defense_move(self):
        # Integration break caught: a rejected S1 task suppresses S2 news or defense.
        payload = task_payload(round_no=58, pioneer_pos=(3, 3))
        payload["teamOur"]["teamId"] = "integration-task-news-defense"
        payload["teamOur"]["roles"].extend([
            unit(10010, "worker", 6, 5),
            unit(10012, "worker", 9, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 9, 6, health=1000),
            unit(10040, "rocket", 16, 6, health=1000),
        ])
        payload["worldNews"] = {"officialNews": "integration bulletin"}
        traces = []

        response = DecisionEngine().decide(payload, trace_sink=traces.append)

        self.assertEqual(
            traces[0]["taskStartSkipReason"],
            "insufficient_solver_return_window",
        )
        self.assertIn("UNTRUSTED_NEWS_DATA_BEGIN", response["prompt"])
        self.assertTrue(any(
            command["action"] == "move"
            for command in response["roleCommandMap"].values()
        ))
        self.assertFalse(any(
            command["action"] == "acceptTask"
            for command in response["roleCommandMap"].values()
        ))

    def test_news_response_is_consumed_before_new_task_prompt(self):
        # Integration break caught: an S2 news result enters the S1 task parser.
        engine = DecisionEngine()
        first = task_payload(round_no=1, pioneer_pos=(3, 3))
        first["teamOur"]["teamId"] = "integration-news-new-task"
        first["teamOur"]["playerTasks"] = []
        first["worldNews"] = {"officialNews": "stable source text"}

        news_response = engine.decide(first)
        self.assertIn("UNTRUSTED_NEWS_DATA_BEGIN", news_response["prompt"])
        pending = engine.state.state.pending_news_request
        self.assertIsNotNone(pending)
        source = pending.sources[0]

        active = task_payload(
            round_no=2, pioneer_pos=(3, 3), phase_task="new active task",
        )
        active["teamOur"]["teamId"] = "integration-news-new-task"
        active["worldNews"] = {}
        active["llmResp"] = json.dumps({
            "requestId": pending.request_id,
            "candidates": [{
                "type": "news",
                "interpretation": "candidate only",
                "citations": [{
                    "sourceId": source.source_id,
                    "excerpt": "stable source text",
                }],
                "missingConditions": [],
                "conflicts": [],
            }],
        })

        response = engine.decide(active)

        self.assertIn("Solve the following competition task", response["prompt"])
        self.assertNotIn("UNTRUSTED_NEWS_DATA_BEGIN", response["prompt"])
        self.assertEqual(engine.state.state.active_task.tool_results, [])
        self.assertEqual(len(engine.state.state.news_candidates), 1)

    def test_task_is_not_accepted_when_required_return_post_is_unreachable(self):
        # Break caught: a missing return route is mistaken for no return need.
        payload = task_payload(round_no=50, pioneer_pos=(3, 3))
        payload["teamOur"]["teamId"] = "task-preaccept-unreachable-return"
        payload["teamOur"]["roles"].extend([
            unit(10010, "worker", 6, 5),
            unit(10012, "worker", 9, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 9, 6, health=1000),
            unit(10040, "rocket", 16, 6, health=1000),
        ])
        payload["mapInfo"]["zones"].extend(
            {"pos": {"x": 11, "y": y}, "neutralType": "vendor"}
            for y in range(20)
        )
        traces = []

        response = DecisionEngine().decide(payload, trace_sink=traces.append)

        pioneer_command = response["roleCommandMap"].get("10011")
        self.assertTrue(
            pioneer_command is None or pioneer_command["action"] != "acceptTask"
        )
        self.assertEqual(
            traces[0]["taskStartSkipReason"], "return_route_unavailable",
        )

    def test_late_task_is_allowed_when_other_roles_cover_every_weapon(self):
        # Break caught: a fixed dusk cutoff blocks work that needs no pioneer return.
        payload = task_payload(round_no=69, pioneer_pos=(3, 3))
        payload["teamOur"]["teamId"] = "task-preaccept-covered"
        payload["teamOur"]["roles"].extend([
            unit(10010, "worker", 6, 5),
            unit(10012, "worker", 9, 5),
            unit(10014, "worker", 12, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 9, 6, health=1000),
            unit(10040, "rocket", 12, 6, health=1000),
        ])

        response = DecisionEngine().decide(payload)

        self.assertEqual(
            response["roleCommandMap"]["10011"]["action"], "acceptTask",
        )

    def test_task_is_accepted_with_one_cycle_and_return_window(self):
        # Break caught: the new guard blocks every task before dusk positioning.
        payload = task_payload(round_no=50, pioneer_pos=(3, 3))
        payload["teamOur"]["teamId"] = "task-preaccept-window-valid"
        payload["teamOur"]["roles"].extend([
            unit(10010, "worker", 6, 5),
            unit(10012, "worker", 9, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 9, 6, health=1000),
            unit(10040, "rocket", 16, 6, health=1000),
        ])

        response = DecisionEngine().decide(payload)

        self.assertEqual(
            response["roleCommandMap"]["10011"]["action"], "acceptTask",
        )

    def test_short_timeout_task_does_not_hide_feasible_second_point(self):
        # Break caught: the first blocked task causes all other points to be skipped.
        payload = task_payload(round_no=1, pioneer_pos=(3, 3))
        payload["teamOur"]["teamId"] = "task-preaccept-two-points"
        payload["teamOur"]["playerTasks"] = [{
            "taskType": "自进化类1",
            "taskPosition": {"x": 4, "y": 4},
            "coldDownRounds": 0,
            "scoreReward": 500,
            "goldReward": 500,
            "isValid": True,
            "timeoutRounds": 3,
        }, {
            "taskType": "自进化类2",
            "taskPosition": {"x": 7, "y": 7},
            "coldDownRounds": 0,
            "scoreReward": 50,
            "goldReward": 30,
            "isValid": True,
            "timeoutRounds": 20,
        }]

        engine = DecisionEngine()
        response = engine.decide(payload)

        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")
        self.assertEqual(
            engine.state.state.plans[10011].target,
            Pos(7, 7),
        )

    def test_active_task_pioneer_is_reserved_unless_support_is_immediate(self):
        # Break caught: any third tower pulls the pioneer away from an active task.
        payload = task_payload(
            round_no=71, pioneer_pos=(3, 3), phase_task="active task",
        )
        payload["teamOur"]["roles"].extend([
            unit(10010, "worker", 6, 5),
            unit(10012, "worker", 9, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 9, 6, health=1000),
            unit(10040, "rocket", 12, 6, health=1000),
        ])
        payload["robot"]["roles"] = [{
            "id": 30001,
            "pos": {"x": 12, "y": 9},
            "roleType": "smallRobot",
            "health": 40,
            "abnormalState": "",
            "targetTeam": "challenger",
        }]

        response = DecisionEngine().decide(payload)

        self.assertTrue(response["prompt"])
        pioneer_command = response["roleCommandMap"].get("10011")
        self.assertTrue(
            pioneer_command is None or pioneer_command["action"] != "move"
        )

    def test_active_task_reservation_blocks_maintenance_move_but_allows_medicine(self):
        # Break caught: economy moves the task pioneer away to spend a repair item.
        repair = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        repair["teamOur"]["roles"][0]["backpack"] = ["WallFixer"]
        repair["teamOur"]["roles"].append(
            unit(10050, "wall", 8, 8, health=500)
        )
        response = DecisionEngine().decide(repair)
        self.assertNotIn("10011", response["roleCommandMap"])

        medicine = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        medicine["teamOur"]["roles"][0]["health"] = 100
        medicine["teamOur"]["roles"][0]["backpack"] = ["Medicine"]
        response = DecisionEngine().decide(medicine)
        self.assertEqual(response["roleCommandMap"]["10011"], {
            "action": "use",
            "name": "Medicine",
        })

    def test_urgent_recall_moves_only_when_one_round_delay_misses_defense(self):
        # Break caught: an active task pioneer is never allowed to rescue the base.
        def active_payload(robot_pos):
            payload = task_payload(
                round_no=71, pioneer_pos=(3, 3), phase_task="active task",
            )
            payload["teamOur"]["roles"].extend([
                unit(10013, "station", 9, 9, health=1500),
                unit(10020, "gatling", 6, 5, health=1000),
            ])
            payload["robot"]["roles"] = [{
                "id": 30001,
                "pos": {"x": robot_pos[0], "y": robot_pos[1]},
                "roleType": "smallRobot",
                "health": 40,
                "abnormalState": "",
                "targetTeam": "challenger",
            }]
            return payload

        urgent_engine = DecisionEngine()
        urgent = active_payload((7, 7))
        urgent_engine.decide(urgent)
        urgent_answer = copy.deepcopy(urgent)
        urgent_answer["roundNo"] = 72
        urgent_answer["llmResp"] = (
            '{"kind":"answer","content":"42","complete":true}'
        )
        response = urgent_engine.decide(urgent_answer)
        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")

        spare_engine = DecisionEngine()
        spare = active_payload((6, 8))
        spare_engine.decide(spare)
        spare_answer = copy.deepcopy(spare)
        spare_answer["roundNo"] = 72
        spare_answer["llmResp"] = urgent_answer["llmResp"]
        response = spare_engine.decide(spare_answer)
        self.assertEqual(
            response["roleCommandMap"]["10011"]["action"],
            "submitAnswer",
        )

    def test_active_task_does_not_recall_without_effective_weapon_post(self):
        # Break caught: generic base pressure recalls a pioneer that cannot help.
        engine = DecisionEngine()
        first = task_payload(
            round_no=71, pioneer_pos=(3, 3), phase_task="active task",
        )
        first["teamOur"]["roles"].append(
            unit(10013, "station", 9, 9, health=1500)
        )
        first["robot"]["roles"] = [{
            "id": 30001,
            "pos": {"x": 7, "y": 7},
            "roleType": "smallRobot",
            "health": 40,
            "abnormalState": "",
            "targetTeam": "challenger",
        }]
        engine.decide(first)

        second = copy.deepcopy(first)
        second["roundNo"] = 72
        second["llmResp"] = (
            '{"kind":"answer","content":"42","complete":true}'
        )
        response = engine.decide(second)

        self.assertEqual(
            response["roleCommandMap"]["10011"]["action"],
            "submitAnswer",
        )

    def test_staffed_other_weapon_does_not_count_as_recall_alternative(self):
        # Break caught: moving an existing gunner would only swap which tower is idle.
        engine = DecisionEngine()
        first = task_payload(
            round_no=71, pioneer_pos=(3, 3), phase_task="active task",
        )
        first["teamOur"]["roles"].extend([
            unit(10010, "worker", 8, 5),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 6, 5, health=1000),
            unit(10021, "railgun", 8, 6, health=1000),
        ])
        first["robot"]["roles"] = [{
            "id": 30001,
            "pos": {"x": 7, "y": 7},
            "roleType": "smallRobot",
            "health": 40,
            "abnormalState": "",
            "targetTeam": "challenger",
        }]
        engine.decide(first)

        second = copy.deepcopy(first)
        second["roundNo"] = 72
        second["llmResp"] = (
            '{"kind":"answer","content":"42","complete":true}'
        )
        response = engine.decide(second)

        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")
        self.assertEqual(response["roleCommandMap"]["10021"]["action"], "attack")

    def test_urgent_recall_is_bound_to_the_weapon_that_can_help(self):
        # Break caught: a boolean release lets generic defense choose a lower-ID bad post.
        engine = DecisionEngine()
        first = task_payload(
            round_no=71, pioneer_pos=(3, 3), phase_task="active task",
        )
        first["teamOur"]["roles"].extend([
            unit(10013, "station", 9, 9, health=1500),
            unit(10019, "gatling", 2, 6, health=1000, cooldown=1),
            unit(10020, "gatling", 6, 5, health=1000),
        ])
        first["robot"]["roles"] = [{
            "id": 30001,
            "pos": {"x": 7, "y": 7},
            "roleType": "smallRobot",
            "health": 40,
            "abnormalState": "",
            "targetTeam": "challenger",
        }]
        engine.decide(first)

        second = copy.deepcopy(first)
        second["roundNo"] = 72
        second["llmResp"] = (
            '{"kind":"answer","content":"42","complete":true}'
        )
        response = engine.decide(second)

        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")
        self.assertEqual(
            engine.state.state.plans[10011].reason,
            "gunner:10020",
        )

    def test_partial_answer_submits_once_and_identical_result_stops(self):
        # Break caught: partial answers are discarded or repeated every round.
        engine = DecisionEngine()
        first = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        engine.decide(first)

        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["llmResp"] = (
            '{"kind":"answer","content":"partial","complete":false}'
        )
        response = engine.decide(second)
        self.assertEqual(response["roleCommandMap"]["10011"], {
            "action": "submitAnswer",
            "taskAnswer": "partial",
        })

        third = copy.deepcopy(first)
        third["roundNo"] = 3
        third["lastRoundRoleActionResults"] = {"10011": True}
        response = engine.decide(third)
        self.assertTrue(response["prompt"])

        fourth = copy.deepcopy(first)
        fourth["roundNo"] = 4
        fourth["llmResp"] = (
            '{"kind":"answer","content":"partial","complete":false}'
        )
        response = engine.decide(fourth)
        self.assertNotIn("10011", response["roleCommandMap"])
        self.assertTrue(response["prompt"])

        fifth = copy.deepcopy(first)
        fifth["roundNo"] = 5
        fifth["llmResp"] = fourth["llmResp"]
        response = engine.decide(fifth)
        self.assertEqual(response["prompt"], "")
        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")

    def test_error_code_two_requests_improvement_without_fabricated_resubmit(self):
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        engine.decide(active)

        answer = copy.deepcopy(active)
        answer["roundNo"] = 2
        answer["llmResp"] = (
            '{"kind":"answer","content":"supported partial","complete":false}'
        )
        self.assertEqual(
            engine.decide(answer)["roleCommandMap"]["10011"]["action"],
            "submitAnswer",
        )

        rejected = copy.deepcopy(active)
        rejected["roundNo"] = 3
        rejected["lastRoundRoleActionResults"] = {"10011": False}
        rejected["errors"] = [{"errorCode": 2, "description": "incomplete"}]
        response = engine.decide(rejected)
        self.assertTrue(response["prompt"])
        self.assertIn("failed or was incomplete", response["prompt"])
        self.assertNotIn("10011", response["roleCommandMap"])

        unchanged = copy.deepcopy(active)
        unchanged["roundNo"] = 4
        unchanged["llmResp"] = answer["llmResp"]
        response = engine.decide(unchanged)
        self.assertTrue(response["prompt"])
        self.assertNotIn("10011", response["roleCommandMap"])

    def test_immediate_task_pioneer_attack_excludes_same_round_submit(self):
        # Break caught: one pioneer is promised to both a weapon and submitAnswer.
        engine = DecisionEngine()
        first = task_payload(
            round_no=71, pioneer_pos=(3, 3), phase_task="active task",
        )
        first["teamOur"]["roles"].append(
            unit(10020, "gatling", 3, 2, health=1000)
        )
        first["robot"]["roles"] = [{
            "id": 30001,
            "pos": {"x": 3, "y": 5},
            "roleType": "smallRobot",
            "health": 40,
            "abnormalState": "",
            "targetTeam": "challenger",
        }]
        response = engine.decide(first)
        self.assertEqual(response["roleCommandMap"]["10020"]["action"], "attack")
        self.assertTrue(response["prompt"])

        second = copy.deepcopy(first)
        second["roundNo"] = 72
        second["lastRoundRoleActionResults"] = {"10020": True}
        second["llmResp"] = (
            '{"kind":"answer","content":"42","complete":true}'
        )
        response = engine.decide(second)

        self.assertEqual(response["roleCommandMap"]["10020"]["action"], "attack")
        self.assertNotIn("10011", response["roleCommandMap"])

        third = copy.deepcopy(first)
        third["roundNo"] = 73
        third["robot"]["roles"] = []
        third["lastRoundRoleActionResults"] = {"10020": True}
        response = engine.decide(third)

        self.assertEqual(
            response["roleCommandMap"]["10011"],
            {"action": "submitAnswer", "taskAnswer": "42"},
        )

    def test_observed_phase_uses_unconfirmed_prior_accept_round(self):
        # Break caught: omitted action feedback shifts the known task deadline.
        engine = DecisionEngine()
        accepted = task_payload(round_no=1, pioneer_pos=(3, 3))
        response = engine.decide(accepted)
        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "acceptTask")

        active = copy.deepcopy(accepted)
        active["roundNo"] = 2
        active["phaseTask"] = "active task"
        active["lastRoundRoleActionResults"] = {}
        engine.decide(active)

        self.assertEqual(engine.state.state.active_task.accepted_round, 1)

    def test_timeout_round_ends_task_without_success_claim(self):
        # Break caught: an elapsed observed timeout is recorded as an unknown success.
        engine = DecisionEngine()
        accepted = task_payload(round_no=1, pioneer_pos=(3, 3))
        accepted["teamOur"]["playerTasks"][0]["timeoutRounds"] = 5
        engine.decide(accepted)

        active = copy.deepcopy(accepted)
        active["roundNo"] = 2
        active["phaseTask"] = "active task"
        active["lastRoundRoleActionResults"] = {"10011": True}
        engine.decide(active)

        ended = task_payload(round_no=7, pioneer_pos=(3, 3))
        ended["teamOur"]["playerTasks"][0]["timeoutRounds"] = 5
        engine.decide(ended)

        self.assertEqual(
            engine.state.state.ended_tasks[-1].end_reason,
            "timeout",
        )

    def test_repeated_empty_llm_results_stop_reissuing_prompt(self):
        # Break caught: empty cross-round results trigger an infinite identical loop.
        engine = DecisionEngine()
        payload = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        self.assertTrue(engine.decide(payload)["prompt"])

        correction = copy.deepcopy(payload)
        correction["roundNo"] = 2
        self.assertTrue(engine.decide(correction)["prompt"])

        stopped = copy.deepcopy(payload)
        stopped["roundNo"] = 3
        response = engine.decide(stopped)
        self.assertEqual(response["prompt"], "")
        self.assertEqual(response["executeCmd"], "")
        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")

        for round_no in range(4, 9):
            still_stopped = copy.deepcopy(payload)
            still_stopped["roundNo"] = round_no
            response = engine.decide(still_stopped)
            self.assertEqual(response["prompt"], "")
            self.assertEqual(response["executeCmd"], "")
            self.assertNotIn("10011", response["roleCommandMap"])

        replacement = copy.deepcopy(payload)
        replacement["roundNo"] = 9
        replacement["phaseTask"] = "replacement task"
        self.assertTrue(engine.decide(replacement)["prompt"])

    def test_old_accept_history_does_not_set_new_task_deadline(self):
        # Break caught: any old acceptTask is reused as the new task's start round.
        engine = DecisionEngine()
        accepted = task_payload(round_no=1, pioneer_pos=(3, 3))
        engine.decide(accepted)

        ended = task_payload(round_no=2, pioneer_pos=(3, 3))
        ended["lastRoundRoleActionResults"] = {"10011": True}
        engine.decide(ended)

        later = task_payload(
            round_no=10, pioneer_pos=(3, 3), phase_task="new task",
        )
        engine.decide(later)

        task = engine.state.state.active_task
        self.assertIsNone(task.accepted_round)
        self.assertIsNone(task.timeout_round)


if __name__ == "__main__":
    unittest.main()
