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
from agent.state import StateStore, request_fingerprint


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
    def test_dead_task_owner_does_not_revive_old_instance_next_day(self):
        payload = task_payload(round_no=69, pioneer_pos=(3, 3))
        payload["teamOur"]["teamId"] = "s2-dead-task-owner-next-day"
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

    def test_required_third_gunner_stays_at_post_without_targets_across_dusk(self):
        # Break caught: idle defense releases the pioneer to a distant task every round.
        engine = DecisionEngine()
        payload = self._idle_third_gunner_payload()
        traces = []

        for round_no in range(65, 75):
            payload["roundNo"] = round_no
            response = engine.decide(payload, trace_sink=traces.append)
            self.assertNotIn("10011", response["roleCommandMap"])
            self.assertEqual(traces[-1]["coordinationReason"], "gunner_hold")
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
        self.assertEqual(pioneer["pos"], {"x": 12, "y": 5})

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
        for invalid in (
            "```json\n{\"kind\":\"answer\",\"content\":\"42\",\"complete\":true}\n```",
            '{"kind":"command","content":""}',
            '{"kind":"answer","content":"42","complete":true,"extra":1}',
            '{"kind":"abandon","reason":""}',
            "not json",
        ):
            self.assertIsNone(tasks.parse_llm_envelope(invalid))

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
        for round_no in range(10, 15):
            invalid = copy.deepcopy(active)
            invalid["roundNo"] = round_no
            invalid["llmResp"] = f"invalid-response-{round_no}"
            prompt = engine.decide(invalid)["prompt"]

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
        next_task["phaseTask"] = "second task"
        prompt = engine.decide(next_task)["prompt"]

        self.assertIn("/opt/task-input/specification.md", prompt)
        self.assertIn("verify each path for the current task", prompt)
        self.assertNotIn("//example.test/api", prompt)

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
            phase_task="Read /sandbox/input/data.csv and compute the total.",
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
        engine.decide(active)

        retry = copy.deepcopy(active)
        retry["roundNo"] = 3
        retry["llmResp"] = "invalid-response"
        prompt = engine.decide(retry)["prompt"]
        self.assertIn("Known remaining task rounds: 4", prompt)
        self.assertIn("Command exploration is allowed", prompt)

        in_flight = copy.deepcopy(active)
        in_flight["roundNo"] = 4
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
        # Break caught: alternating LLM and command result kinds reset repetition.
        engine = DecisionEngine()
        active = task_payload(
            round_no=1, pioneer_pos=(3, 3), phase_task="active task",
        )
        self.assertIn("remaining task rounds: unknown", engine.decide(active)["prompt"])

        for command_round, result_round in ((2, 3), (4, 5), (6, 7)):
            command = copy.deepcopy(active)
            command["roundNo"] = command_round
            command["llmResp"] = '{"kind":"command","content":"pwd"}'
            self.assertEqual(engine.decide(command)["executeCmd"], "pwd")

            result = copy.deepcopy(active)
            result["roundNo"] = result_round
            result["lastCmdResult"] = "[exitCode:1]\nsame failure"
            response = engine.decide(result)

        self.assertIn("No more command exploration", response["prompt"])

        ignored = copy.deepcopy(active)
        ignored["roundNo"] = 8
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
        for round_no in range(2, 12):
            repeated = copy.deepcopy(active)
            repeated["roundNo"] = round_no
            repeated["llmResp"] = f"invalid-{round_no}-" + "x" * 5_000
            engine.decide(repeated)

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
        accepted["teamOur"]["playerTasks"][0]["timeoutRounds"] = 1
        engine.decide(accepted)

        active = copy.deepcopy(accepted)
        active["roundNo"] = 2
        active["phaseTask"] = "active task"
        active["lastRoundRoleActionResults"] = {"10011": True}
        engine.decide(active)

        ended = task_payload(round_no=3, pioneer_pos=(3, 3))
        ended["teamOur"]["playerTasks"][0]["timeoutRounds"] = 1
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

        for round_no in (2, 3):
            repeated = copy.deepcopy(payload)
            repeated["roundNo"] = round_no
            self.assertTrue(engine.decide(repeated)["prompt"])

        stopped = copy.deepcopy(payload)
        stopped["roundNo"] = 4
        response = engine.decide(stopped)
        self.assertEqual(response["prompt"], "")
        self.assertEqual(response["executeCmd"], "")
        self.assertEqual(response["roleCommandMap"]["10011"]["action"], "move")

        for round_no in range(5, 9):
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
