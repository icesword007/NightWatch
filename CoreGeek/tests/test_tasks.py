import copy
import importlib
import json
import unittest
from pathlib import Path

from agent.actions import ActionAllocator, ActionProposal
from agent.brain import DecisionEngine
from agent.protocol import Turn
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

        self.assertEqual(command.kind, "command")
        self.assertEqual(command.content, "python3 solve.py")
        self.assertEqual(answer.kind, "answer")
        self.assertFalse(answer.complete)
        for invalid in (
            "```json\n{\"kind\":\"answer\",\"content\":\"42\",\"complete\":true}\n```",
            '{"kind":"command","content":""}',
            '{"kind":"answer","content":"42","complete":true,"extra":1}',
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

        for round_no in range(5, 9):
            still_stopped = copy.deepcopy(payload)
            still_stopped["roundNo"] = round_no
            response = engine.decide(still_stopped)
            self.assertEqual(response["prompt"], "")
            self.assertEqual(response["executeCmd"], "")

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
