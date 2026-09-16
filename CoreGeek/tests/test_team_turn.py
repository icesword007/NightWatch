import io
import json
import unittest
from pathlib import Path
from unittest import mock

from agent import server as server_module
from agent.brain import DecisionEngine
from agent.protocol import Pos, Turn
from agent.state import PlanState, request_fingerprint


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def unit(unit_id, kind, x, y, *, health=220, level=1, cooldown=0):
    return {
        "id": unit_id,
        "pos": {"x": x, "y": y},
        "roleType": kind,
        "health": health,
        "attackPower": 10,
        "attackRange": 0,
        "backPackCapability": 100 if kind == "worker" else 40,
        "backpack": [],
        "level": level,
        "cooldown": cooldown,
    }


def base_payload(*, round_no, team_id):
    payload = load_fixture()
    payload["roundNo"] = round_no
    payload["mapInfo"].update({"width": 20, "height": 20, "zones": []})
    payload["teamOur"].update({"teamId": team_id, "roles": []})
    payload["teamEnemy"]["roles"] = []
    payload["robot"]["roles"] = []
    return payload


class TeamTurnTests(unittest.TestCase):
    def test_destroyed_confirmed_wall_is_rebuilt_on_the_next_day(self):
        # Break caught: historical completion permanently hides a destroyed wall.
        payload = base_payload(round_no=5, team_id="s2-wall-next-day-recovery")
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 6, 9),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        engine = DecisionEngine()
        engine.decide(payload)
        target = engine.state.state.fortification_targets[0]

        confirmed = json.loads(json.dumps(payload))
        confirmed["roundNo"] = 6
        confirmed["teamOur"]["roles"].append(
            unit(10050, "wall", target.x, target.y, health=1000),
        )
        engine.decide(confirmed)
        self.assertIn(target, engine.state.state.fortification_completed)

        destroyed_at_night = json.loads(json.dumps(confirmed))
        destroyed_at_night["roundNo"] = 80
        destroyed_at_night["teamOur"]["roles"] = [
            role for role in destroyed_at_night["teamOur"]["roles"]
            if role["id"] != 10050
        ]
        night_response = engine.decide(destroyed_at_night)
        self.assertFalse(any(
            command.get("name") == "wall"
            for command in night_response["roleCommandMap"].values()
        ))

        next_day = json.loads(json.dumps(destroyed_at_night))
        next_day["roundNo"] = 132
        worker = next(
            role for role in next_day["teamOur"]["roles"]
            if role["id"] == 10010
        )
        worker["pos"] = {"x": target.x - 1, "y": target.y}
        worker["backpack"] = ["stone"]
        traces = []
        command = engine.decide(
            next_day, trace_sink=traces.append,
        )["roleCommandMap"]["10010"]

        self.assertEqual(command, {
            "action": "build",
            "targetPos": [target.dump()],
            "name": "wall",
        })
        fortification = traces[0]["economyPlanning"]["fortification"]
        self.assertEqual(fortification["recoveryTargets"], [target.dump()])
        self.assertEqual(fortification["observedFixedWalls"], 0)
        self.assertEqual(fortification["missingConfirmedWalls"], 1)
        self.assertEqual(traces[0]["defensePressure"]["forecast"]["targetDay"], 2)

        attempts_before = dict(engine.state.state.fortification_attempt_days)
        replayed = engine.decide(next_day)
        self.assertEqual(replayed["roleCommandMap"]["10010"], command)
        self.assertEqual(
            engine.state.state.fortification_attempt_days, attempts_before,
        )

        failed = json.loads(json.dumps(next_day))
        failed["roundNo"] = 133
        failed["lastRoundRoleActionResults"] = {"10010": False}
        engine.decide(failed)
        self.assertIn(target, engine.state.state.fortification_failed)
        later = json.loads(json.dumps(failed))
        later["roundNo"] = 262
        later["lastRoundRoleActionResults"] = {}
        later["teamOur"]["roles"][0]["backpack"] = ["stone"]
        response = engine.decide(later)
        self.assertFalse(any(
            candidate.get("name") == "wall"
            and candidate.get("targetPos") == [target.dump()]
            for candidate in response["roleCommandMap"].values()
        ))

    def test_confirmed_wall_missing_on_the_same_day_is_not_rebuilt(self):
        # Break caught: a transient same-day omission is treated as night damage.
        payload = base_payload(round_no=5, team_id="s2-wall-same-day")
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 6, 9),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        engine = DecisionEngine()
        engine.decide(payload)
        target = engine.state.state.fortification_targets[0]
        present = json.loads(json.dumps(payload))
        present["roundNo"] = 6
        present["teamOur"]["roles"].append(
            unit(10050, "wall", target.x, target.y, health=100),
        )
        engine.decide(present)
        missing = json.loads(json.dumps(payload))
        missing["roundNo"] = 7
        missing["teamOur"]["roles"][0]["pos"] = {
            "x": target.x - 1, "y": target.y,
        }
        missing["teamOur"]["roles"][0]["backpack"] = ["stone"]

        response = engine.decide(missing)

        self.assertIn(target, engine.state.state.fortification_completed)
        self.assertFalse(any(
            command.get("name") == "wall"
            and command.get("targetPos") == [target.dump()]
            for command in response["roleCommandMap"].values()
        ))

    def test_unknown_wall_feedback_does_not_create_a_later_day_retry(self):
        # Break caught: an unconfirmed build is treated as a destroyed real wall.
        payload = base_payload(round_no=5, team_id="s2-wall-unknown-day-limit")
        worker = unit(10010, "worker", 11, 8)
        worker["backpack"] = ["stone"]
        payload["teamOur"]["roles"] = [
            worker,
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        engine = DecisionEngine()
        first = engine.decide(payload)["roleCommandMap"]["10010"]
        target = Pos.load(first["targetPos"][0])
        self.assertEqual(first.get("name"), "wall")

        unknown = json.loads(json.dumps(payload))
        unknown["roundNo"] = 6
        unknown["lastRoundRoleActionResults"] = {}
        same_day = engine.decide(unknown)
        self.assertFalse(any(
            command.get("name") == "wall"
            and command.get("targetPos") == [target.dump()]
            for command in same_day["roleCommandMap"].values()
        ))
        self.assertIn(target, engine.state.state.fortification_failed)

        next_day = json.loads(json.dumps(unknown))
        next_day["roundNo"] = 132
        next_day["teamOur"]["roles"][0]["backpack"] = ["stone"]
        retried = engine.decide(next_day)
        self.assertFalse(any(
            command.get("name") == "wall"
            and command.get("targetPos") == [target.dump()]
            for command in retried["roleCommandMap"].values()
        ))

    def test_dawn_release_and_short_task_gate_coexist(self):
        # Integration break caught: S2 dawn release bypasses the S1 task gate.
        payload = base_payload(
            round_no=70, team_id="integration-dawn-task-gate",
        )
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 2, "y": 2}, "neutralType": "stone"},
            {
                "pos": {"x": 4, "y": 4},
                "neutralType": "challengerTaskPoint1",
            },
        ]
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 5, 4),
            unit(10011, "pioneer", 3, 3),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
        ]
        payload["teamOur"]["playerTasks"] = []
        engine = DecisionEngine()

        dusk_response = engine.decide(payload)
        self.assertEqual(
            dusk_response["roleCommandMap"]["10010"]["action"], "move",
        )
        night = json.loads(json.dumps(payload))
        night["roundNo"] = 71
        night["lastRoundRoleActionResults"] = {"10010": True}
        night["teamOur"]["roles"][0]["pos"] = json.loads(json.dumps(
            dusk_response["roleCommandMap"]["10010"]["targetPos"][0]
        ))
        engine.decide(night)

        dawn = json.loads(json.dumps(night))
        dawn["roundNo"] = 131
        dawn["lastRoundRoleActionResults"] = {}
        dawn["teamOur"]["roles"][0]["pos"] = {"x": 2, "y": 1}
        dawn["teamOur"]["playerTasks"] = [{
            "taskType": "自进化类1",
            "taskPosition": {"x": 4, "y": 4},
            "coldDownRounds": 0,
            "scoreReward": 50,
            "goldReward": 30,
            "isValid": True,
            "timeoutRounds": 4,
        }]
        traces = []

        response = engine.decide(dawn, trace_sink=traces.append)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "collect", "targetPos": [{"x": 2, "y": 2}],
        })
        self.assertNotEqual(
            response["roleCommandMap"].get("10011", {}).get("action"),
            "acceptTask",
        )
        self.assertEqual(
            traces[0]["taskStartSkipReason"], "insufficient_solver_window",
        )
        self.assertIsNone(traces[0]["newsEvidence"]["sessionBoundary"])
        self.assertEqual(traces[0]["newsEvidence"]["currentSession"], 1)

    def test_missing_worker_reappears_with_real_inventory_and_resumes_trade(self):
        payload = base_payload(
            round_no=5, team_id="s2-reappeared-worker-trade",
        )
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 2, "y": 2}, "neutralType": "copper"},
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 25}]
        payload["teamOur"]["roles"] = [unit(10010, "worker", 2, 1)]
        engine = DecisionEngine()

        self.assertEqual(
            engine.decide(payload)["roleCommandMap"]["10010"]["action"],
            "collect",
        )
        night = json.loads(json.dumps(payload))
        night["roundNo"] = 71
        night["lastRoundRoleActionResults"] = {}
        night["teamOur"]["roles"] = []
        engine.decide(night)
        self.assertNotIn(10010, engine.state.state.plans)

        dawn = json.loads(json.dumps(night))
        dawn["roundNo"] = 131
        dawn["mapInfo"]["zones"] = [
            {"pos": {"x": 2, "y": 2}, "neutralType": "vendor"},
        ]
        reappeared = unit(10010, "worker", 2, 1)
        reappeared["backpack"] = ["copper"]
        dawn["teamOur"]["roles"] = [reappeared]
        response = engine.decide(dawn)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "sell", "name": "copper", "num": 1,
        })

    def test_idle_night_gunner_plan_does_not_suppress_dawn_economy(self):
        # Break caught: a worker that reached an idle tower keeps its gunner
        # plan forever because no attack action exists to retire that plan.
        payload = base_payload(
            round_no=70, team_id="s2-idle-gunner-dawn-release",
        )
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 2, "y": 2}, "neutralType": "stone"},
        ]
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 5, 4),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
        ]
        engine = DecisionEngine()

        dusk_response = engine.decide(payload)
        self.assertEqual(
            dusk_response["roleCommandMap"]["10010"]["action"], "move",
        )
        night = json.loads(json.dumps(payload))
        night["roundNo"] = 71
        night["lastRoundRoleActionResults"] = {"10010": True}
        night["teamOur"]["roles"][0]["pos"] = json.loads(json.dumps(
            dusk_response["roleCommandMap"]["10010"]["targetPos"][0]
        ))
        self.assertEqual(engine.decide(night)["roleCommandMap"], {})
        self.assertTrue(
            engine.state.state.plans[10010].reason.startswith("gunner:"),
        )

        dawn = json.loads(json.dumps(night))
        dawn["roundNo"] = 131
        dawn["teamOur"]["roles"][0]["pos"] = {"x": 2, "y": 1}
        response = engine.decide(dawn)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "collect", "targetPos": [{"x": 2, "y": 2}],
        })

    def test_two_day_cycle_releases_night_posts_and_restores_them_at_dusk(self):
        # Break caught: a no-deadline night gunner plan survives dawn and keeps
        # suppressing daytime economy across every later day.
        payload = base_payload(
            round_no=70, team_id="s2-two-day-gunner-cycle",
        )
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 2, "y": 2}, "neutralType": "stone"},
        ]
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 6, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
        ]
        engine = DecisionEngine()

        engine.decide(payload)
        night = json.loads(json.dumps(payload))
        night["roundNo"] = 71
        night["robot"]["roles"] = [{
            "id": 30001,
            "pos": {"x": 6, "y": 9},
            "roleType": "smallRobot",
            "health": 40,
            "abnormalState": "",
            "targetTeam": "challenger",
        }]
        night_response = engine.decide(night)
        self.assertEqual(
            night_response["roleCommandMap"]["10020"]["action"], "attack",
        )

        dawn = json.loads(json.dumps(night))
        dawn["roundNo"] = 131
        dawn["robot"]["roles"] = []
        dawn["teamOur"]["roles"][0]["pos"] = {"x": 2, "y": 1}
        dawn_response = engine.decide(dawn)
        self.assertEqual(dawn_response["roleCommandMap"]["10010"], {
            "action": "collect", "targetPos": [{"x": 2, "y": 2}],
        })

        second_dusk = json.loads(json.dumps(dawn))
        second_dusk["roundNo"] = 200
        second_dusk["teamOur"]["roles"][0]["pos"] = {"x": 5, "y": 4}
        dusk_response = engine.decide(second_dusk)
        self.assertEqual(
            dusk_response["roleCommandMap"]["10010"]["action"], "move",
        )
        self.assertEqual(
            dusk_response["roleCommandMap"]["10010"]["targetPos"][0],
            {"x": 5, "y": 5},
        )
        second_night = json.loads(json.dumps(second_dusk))
        second_night["roundNo"] = 201
        second_night["lastRoundRoleActionResults"] = {"10010": True}
        second_night["teamOur"]["roles"][0]["pos"] = json.loads(json.dumps(
            dusk_response["roleCommandMap"]["10010"]["targetPos"][0]
        ))
        second_night["robot"]["roles"] = json.loads(json.dumps(
            night["robot"]["roles"]
        ))
        second_night_response = engine.decide(second_night)
        self.assertEqual(
            second_night_response["roleCommandMap"]["10020"]["action"],
            "attack",
        )

        second_dawn = json.loads(json.dumps(second_night))
        second_dawn["roundNo"] = 261
        second_dawn["robot"]["roles"] = []
        second_dawn["teamOur"]["roles"][0]["pos"] = {"x": 2, "y": 1}
        second_dawn_response = engine.decide(second_dawn)
        self.assertEqual(
            second_dawn_response["roleCommandMap"]["10010"]["action"],
            "collect",
        )
        self.assertEqual(engine.state.state.session_index, 1)

    def test_missing_wall_builder_is_reassigned_on_the_next_day(self):
        # Break caught: a dead first-day builder leaves its fixed ID in state,
        # permanently preventing another worker from continuing the wall line.
        payload = base_payload(
            round_no=5, team_id="s2-next-day-builder-reassignment",
        )
        first_builder = unit(10010, "worker", 11, 8)
        first_builder["backpack"] = ["stone"]
        replacement = unit(10011, "worker", 11, 9)
        replacement["backpack"] = ["stone", "stone"]
        payload["teamOur"]["roles"] = [
            first_builder,
            replacement,
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 17, 9, health=1500),
        ]
        engine = DecisionEngine()

        engine.decide(payload)
        self.assertEqual(engine.state.state.fortification_builder_id, 10010)

        night = json.loads(json.dumps(payload))
        night["roundNo"] = 71
        night["lastRoundRoleActionResults"] = {}
        night["teamOur"]["roles"] = [
            role for role in night["teamOur"]["roles"]
            if role["id"] != 10010
        ]
        engine.decide(night)

        next_day = json.loads(json.dumps(night))
        next_day["roundNo"] = 131
        response = engine.decide(next_day)

        self.assertEqual(engine.state.state.fortification_builder_id, 10011)
        self.assertEqual(
            response["roleCommandMap"]["10011"].get("name"), "wall",
        )

    def test_completed_tower_line_actively_collects_stone_for_fortification(self):
        # Break caught: ordinary high-value mining can starve the approved wall line.
        payload = base_payload(round_no=5, team_id="brain-r8-active-stone")
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 5, "y": 8}, "neutralType": "stone"},
            {"pos": {"x": 5, "y": 10}, "neutralType": "copper"},
            {"pos": {"x": 2, "y": 2}, "neutralType": "vendor"},
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 100}]
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 5, 9),
            unit(10011, "worker", 15, 15),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "collect", "targetPos": [{"x": 5, "y": 8}],
        })

    def test_extra_far_stone_does_not_consume_build_and_return_search_budget(self):
        payload = base_payload(round_no=5, team_id="brain-r8-stone-budget")
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 5, "y": 8}, "neutralType": "stone"},
            {"pos": {"x": 3, "y": 8}, "neutralType": "stone"},
        ]
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 5, 9),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 17, 9, health=1500),
        ]
        traces = []

        response = DecisionEngine().decide(payload, trace_sink=traces.append)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "collect", "targetPos": [{"x": 5, "y": 8}],
        })
        self.assertIsNone(
            traces[0]["economyPlanning"]["fortification"]["skipReason"],
        )

    def test_left_base_uses_enemy_facing_wall_targets_not_nearest_back_cell(self):
        # Break caught: worker proximity overrides the observed horizontal attack side.
        payload = base_payload(round_no=5, team_id="brain-r8-left-front")
        worker = unit(10010, "worker", 6, 8)
        worker["backpack"] = ["stone"]
        payload["teamOur"]["roles"] = [
            worker,
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 17, 9, health=1500),
        ]

        response = DecisionEngine().decide(payload)

        self.assertEqual(
            response["roleCommandMap"]["10010"]["action"], "move",
        )

    def test_successful_wall_continues_to_a_second_fixed_target(self):
        # Break caught: the former one-shot wall flag ends construction after one wall.
        payload = base_payload(round_no=5, team_id="brain-r8-wall-chain")
        worker = unit(10010, "worker", 11, 8)
        worker["backpack"] = ["stone", "stone"]
        payload["teamOur"]["roles"] = [
            worker,
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 17, 9, health=1500),
        ]
        engine = DecisionEngine()

        first = engine.decide(payload)["roleCommandMap"]["10010"]
        self.assertEqual(first["action"], "build")
        first_target = first["targetPos"][0]

        following = json.loads(json.dumps(payload))
        following["roundNo"] = 6
        following["lastRoundRoleActionResults"] = {"10010": True}
        following["teamOur"]["roles"][0]["backpack"] = ["stone"]
        following["teamOur"]["roles"].append(
            unit(10050, "wall", first_target["x"], first_target["y"], health=1000),
        )
        second = engine.decide(following)["roleCommandMap"]["10010"]

        self.assertEqual(second["action"], "build")
        self.assertEqual(second["name"], "wall")
        self.assertNotEqual(second["targetPos"][0], first_target)

    def test_fortification_collects_a_two_wall_batch_before_construction(self):
        # Break caught: the builder walks back after every single stone.
        payload = base_payload(round_no=5, team_id="brain-r8-mine-build")
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 5, "y": 8}, "neutralType": "stone"},
        ]
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 5, 9),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 17, 9, health=1500),
        ]
        engine = DecisionEngine()
        first = engine.decide(payload)["roleCommandMap"]["10010"]
        self.assertEqual(first["action"], "collect")

        following = json.loads(json.dumps(payload))
        following["roundNo"] = 6
        following["lastRoundRoleActionResults"] = {"10010": True}
        following["teamOur"]["roles"][0]["backpack"] = ["stone"]
        second = engine.decide(following)["roleCommandMap"]["10010"]
        self.assertEqual(second, {
            "action": "collect", "targetPos": [{"x": 5, "y": 8}],
        })

        ready = json.loads(json.dumps(following))
        ready["roundNo"] = 7
        ready["lastRoundRoleActionResults"] = {"10010": True}
        ready["teamOur"]["roles"][0]["backpack"] = ["stone", "stone"]
        third = engine.decide(ready)["roleCommandMap"]["10010"]

        self.assertIn(third["action"], ("move", "build"))
        self.assertNotEqual(third["action"], "collect")

    def test_fortification_shrinks_batch_when_stone_disappears(self):
        payload = base_payload(round_no=5, team_id="brain-r9-stone-disappears")
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 5, "y": 8}, "neutralType": "stone"},
        ]
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 5, 9),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 17, 9, health=1500),
        ]
        engine = DecisionEngine()
        self.assertEqual(
            engine.decide(payload)["roleCommandMap"]["10010"]["action"],
            "collect",
        )

        following = json.loads(json.dumps(payload))
        following["roundNo"] = 6
        following["lastRoundRoleActionResults"] = {"10010": True}
        following["mapInfo"]["zones"] = []
        following["teamOur"]["roles"][0]["backpack"] = ["stone"]
        command = engine.decide(following)["roleCommandMap"]["10010"]

        self.assertIn(command["action"], ("move", "build"))
        self.assertEqual(len(engine.state.state.fortification_batch_targets), 1)

    def test_two_workers_build_and_switch_old_mine_to_full_funding_chain(self):
        # Break caught: the active builder and an old mine plan together starve
        # the independent worker's now-affordable upgrade route.
        payload = base_payload(round_no=15, team_id="brain-r9-build-and-fund")
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 5, "y": 8}, "neutralType": "stone"},
            {"pos": {"x": 2, "y": 2}, "neutralType": "copper"},
            {"pos": {"x": 4, "y": 2}, "neutralType": "vendor"},
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 100}]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        builder = unit(10010, "worker", 5, 9)
        economic_worker = unit(10011, "worker", 2, 1)
        economic_worker["backpack"] = ["copper"]
        payload["teamOur"]["roles"] = [
            builder,
            economic_worker,
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 2, health=1000),
            unit(10030, "railgun", 8, 8, health=1000),
            unit(10040, "rocket", 11, 8, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 17, 9, health=1500),
        ]
        engine = DecisionEngine()
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.state.plans[10011] = PlanState(
            10011,
            Pos(2, 2),
            "mine:copper",
            None,
            engine.state.state.session_index,
        )
        builder_actions = []
        economic_actions = []

        for _ in range(12):
            response = engine.decide(payload)
            commands = response["roleCommandMap"]
            if "10010" in commands:
                builder_actions.append(commands["10010"]["action"])
            if "10011" in commands:
                economic_actions.append(commands["10011"]["action"])
            for raw_id, command in commands.items():
                actor = next(
                    role for role in payload["teamOur"]["roles"]
                    if role["id"] == int(raw_id)
                )
                action = command["action"]
                if action == "move":
                    actor["pos"] = json.loads(json.dumps(command["targetPos"][0]))
                elif action == "collect":
                    actor["backpack"].append(
                        "stone" if int(raw_id) == 10010 else "copper"
                    )
                elif action == "sell":
                    actor["backpack"].remove(command["name"])
                    payload["teamOur"]["goldNum"] += 100
                elif action == "buy":
                    payload["teamOur"]["goldNum"] -= 100
                    actor["backpack"].append(command["name"])
                elif action == "use":
                    actor["backpack"].remove(command["name"])
                    target = next(
                        role for role in payload["teamOur"]["roles"]
                        if role["pos"] == command["targetPos"][0]
                    )
                    target["level"] += 1
                elif action == "build":
                    actor["backpack"].remove("stone")
                    target = command["targetPos"][0]
                    payload["teamOur"]["roles"].append(unit(
                        10100 + len(builder_actions),
                        "wall",
                        target["x"],
                        target["y"],
                        health=1000,
                    ))
            payload["lastRoundRoleActionResults"] = {
                raw_id: True for raw_id in commands
            }
            payload["roundNo"] += 1
            if "build" in builder_actions and "use" in economic_actions:
                break

        self.assertEqual(economic_actions[0], "move")
        self.assertIn("collect", builder_actions)
        self.assertIn("build", builder_actions)
        self.assertIn("sell", economic_actions)
        self.assertIn("buy", economic_actions)
        self.assertIn("use", economic_actions)

    def test_failed_fortification_mine_moves_to_another_visible_stone(self):
        # Break caught: dedicated stone mining retries one failed mine forever.
        payload = base_payload(round_no=5, team_id="brain-r8-mine-failure")
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 5, "y": 8}, "neutralType": "stone"},
            {"pos": {"x": 5, "y": 10}, "neutralType": "stone"},
        ]
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 5, 9),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 17, 9, health=1500),
        ]
        engine = DecisionEngine()
        first = engine.decide(payload)["roleCommandMap"]["10010"]
        self.assertEqual(first["targetPos"], [{"x": 5, "y": 8}])

        following = json.loads(json.dumps(payload))
        following["roundNo"] = 6
        following["lastRoundRoleActionResults"] = {"10010": False}
        second = engine.decide(following)["roleCommandMap"]["10010"]

        self.assertEqual(second, {
            "action": "collect", "targetPos": [{"x": 5, "y": 10}],
        })

    def test_old_wall_plan_stops_when_current_safety_check_rejects(self):
        payload = base_payload(round_no=5, team_id="brain-r8-wall-safety-stop")
        worker = unit(10010, "worker", 6, 8)
        worker["backpack"] = ["stone"]
        payload["teamOur"]["roles"] = [
            worker,
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 17, 9, health=1500),
        ]
        engine = DecisionEngine()
        first = engine.decide(payload)["roleCommandMap"]["10010"]
        self.assertEqual(first["action"], "move")
        self.assertEqual(engine.state.state.plans[10010].reason, "build:wall")

        following = json.loads(json.dumps(payload))
        following["roundNo"] = 6
        following["lastRoundRoleActionResults"] = {"10010": True}
        following["teamOur"]["roles"][0]["pos"] = {"x": 11, "y": 8}
        with mock.patch(
            "agent.fortification.safe_wall_targets", return_value=(),
        ):
            response = engine.decide(following)

        self.assertFalse(any(
            command.get("name") == "wall"
            for command in response["roleCommandMap"].values()
        ))

    def test_old_wall_plan_stops_when_return_deadline_check_rejects(self):
        payload = base_payload(round_no=5, team_id="brain-r8-wall-deadline-stop")
        worker = unit(10010, "worker", 6, 8)
        worker["backpack"] = ["stone"]
        payload["teamOur"]["roles"] = [
            worker,
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 17, 9, health=1500),
        ]
        engine = DecisionEngine()
        engine.decide(payload)

        following = json.loads(json.dumps(payload))
        following["roundNo"] = 6
        following["lastRoundRoleActionResults"] = {"10010": True}
        following["teamOur"]["roles"][0]["pos"] = {"x": 11, "y": 8}
        with mock.patch(
            "agent.fortification._can_build_and_return", return_value=False,
        ):
            response = engine.decide(following)

        self.assertFalse(any(
            command.get("name") == "wall"
            for command in response["roleCommandMap"].values()
        ))

    def test_old_wall_target_is_not_revived_when_new_target_has_no_action(self):
        payload = base_payload(round_no=5, team_id="brain-r8-old-wall-stop")
        worker = unit(10010, "worker", 6, 8)
        worker["backpack"] = ["stone"]
        payload["teamOur"]["roles"] = [
            worker,
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 17, 9, health=1500),
        ]
        engine = DecisionEngine()
        engine.decide(payload)
        old_target = engine.state.state.plans[10010].target

        following = json.loads(json.dumps(payload))
        following["roundNo"] = 6
        following["lastRoundRoleActionResults"] = {"10010": True}
        following["teamOur"]["roles"][0]["pos"] = {"x": 11, "y": 8}

        def skip_only_old_target(turn, candidates):
            return () if candidates == (old_target,) else candidates

        with mock.patch(
            "agent.fortification.safe_wall_targets",
            side_effect=skip_only_old_target,
        ), mock.patch(
            "agent.fortification._can_build_and_return", return_value=True,
        ), mock.patch(
            "agent.economy._wall_action", return_value=None,
        ):
            response = engine.decide(following)

        self.assertFalse(any(
            command.get("name") == "wall"
            for command in response["roleCommandMap"].values()
        ))

    def test_real_engine_completes_two_walls_across_continuous_feedback(self):
        # Break caught: an in-transit wall plan excludes its own fixed target.
        payload = base_payload(round_no=5, team_id="brain-r8-wall-continuous")
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 5, "y": 8}, "neutralType": "stone"},
        ]
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 5, 9),
            unit(10011, "worker", 3, 3),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 17, 9, health=1500),
        ]
        engine = DecisionEngine()
        built = []

        for _ in range(35):
            response = engine.decide(payload)
            command = response["roleCommandMap"].get("10010")
            if command is None:
                break
            action = command["action"]
            worker = payload["teamOur"]["roles"][0]
            if action == "move":
                worker["pos"] = json.loads(json.dumps(command["targetPos"][0]))
            elif action == "collect":
                worker["backpack"].append("stone")
            elif action == "build" and command.get("name") == "wall":
                target = command["targetPos"][0]
                built.append((target["x"], target["y"]))
                worker["backpack"].remove("stone")
                payload["teamOur"]["roles"].append(
                    unit(10100 + len(built), "wall", target["x"], target["y"], health=1000),
                )
            payload["lastRoundRoleActionResults"] = {"10010": True}
            payload["roundNo"] += 1
            if len(built) == 2:
                break

        self.assertEqual(built, [(12, 8), (12, 9)])

    def test_fortification_does_not_start_when_builder_cannot_build_and_return(self):
        # Break caught: a fixed dusk threshold ignores the real mine/build/return route.
        payload = base_payload(round_no=57, team_id="brain-r8-wall-deadline")
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 0, "y": 1}, "neutralType": "copper"},
            {"pos": {"x": 2, "y": 2}, "neutralType": "vendor"},
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 100}]
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 0, 0),
            unit(10013, "station", 17, 9, health=1500),
            unit(10020, "gatling", 16, 8, health=1000),
            unit(10030, "railgun", 17, 7, health=1000),
            unit(10040, "rocket", 18, 7, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 1, 9, health=1500),
        ]

        response = DecisionEngine().decide(payload)

        self.assertNotEqual(response["roleCommandMap"]["10010"], {
            "action": "collect", "targetPos": [{"x": 1, "y": 1}],
        })

    def test_daytime_http_path_emits_c_batch_build_commands(self):
        # Break caught: economy helpers exist but HTTP still runs the S0 probe.
        payload = base_payload(round_no=1, team_id="http-c-build")
        payload["teamOur"].update({
            "goldNum": 75,
            "roles": [
                unit(10010, "worker", 7, 8),
                unit(10012, "worker", 7, 9),
                unit(10011, "pioneer", 1, 1, health=200),
                unit(10013, "station", 9, 9, health=1500),
            ],
        })
        body = json.dumps(payload).encode("utf-8")
        handler = object.__new__(server_module.Handler)
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        captured = {}

        def send_body(status, response_body):
            captured["status"] = status
            captured["body"] = json.loads(response_body.decode("utf-8"))

        handler._send_body = send_body
        engine = DecisionEngine()
        with mock.patch.object(server_module, "decide", side_effect=engine.decide):
            with mock.patch.object(server_module.LOGGER, "info"):
                server_module.Handler.do_POST(handler)

        self.assertEqual(captured["status"], 200)
        commands = captured["body"]["roleCommandMap"]
        self.assertEqual(len(commands), 2)
        self.assertEqual(
            {command["action"] for command in commands.values()}, {"build"}
        )
        self.assertEqual(
            {command["name"] for command in commands.values()},
            {"gatling", "railgun"},
        )

    def test_night_decision_engine_emits_real_attack(self):
        # Break caught: defense candidates are never selected by the coordinator.
        payload = base_payload(round_no=71, team_id="brain-c-attack")
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 6, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
        ]
        payload["robot"]["roles"] = [{
            "id": 30001,
            "pos": {"x": 6, "y": 9},
            "roleType": "smallRobot",
            "health": 40,
            "abnormalState": "",
            "targetTeam": "challenger",
        }]

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10020"], {
            "action": "attack",
            "controllerId": "10010",
            "targetPos": [{"x": 6, "y": 9}],
        })

    def test_completed_tower_line_starts_one_stable_fortification_builder(self):
        # Break caught: the active wall capability is never reached by the coordinator.
        payload = base_payload(round_no=5, team_id="brain-c-wall")
        worker = unit(10010, "worker", 6, 9)
        worker["backpack"] = ["stone"]
        payload["teamOur"]["roles"] = [
            worker,
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]

        engine = DecisionEngine()
        response = engine.decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")
        self.assertEqual(engine.state.state.fortification_builder_id, 10010)
        self.assertEqual(len(engine.state.state.fortification_targets), 6)

        following = json.loads(json.dumps(payload))
        following["roundNo"] = 6
        following["lastRoundRoleActionResults"] = {"10010": False}
        response = engine.decide(following)
        self.assertEqual(engine.state.state.fortification_builder_id, 10010)
        self.assertEqual(len(engine.state.state.fortification_targets), 6)

    def test_two_workers_share_one_fortification_builder(self):
        # Break caught: both workers are assigned to the same construction chain.
        payload = base_payload(round_no=5, team_id="brain-c-wall-team-limit")
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 6, 9),
            unit(10011, "worker", 12, 9),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["teamOur"]["roles"][0]["backpack"] = ["stone"]
        payload["teamOur"]["roles"][1]["backpack"] = ["stone"]

        response = DecisionEngine().decide(payload)

        wall_builds = [
            command for command in response["roleCommandMap"].values()
            if command.get("action") == "build" and command.get("name") == "wall"
        ]
        self.assertEqual(len(wall_builds), 1)

    def test_in_transit_fortification_keeps_builder_until_owner_death(self):
        # Break caught: another worker takes over after an in-transit move is unknown.
        payload = base_payload(round_no=5, team_id="brain-c-wall-in-transit")
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 0, 0),
            unit(10011, "worker", 14, 0),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["mapInfo"]["width"] = 20
        payload["mapInfo"]["height"] = 20
        payload["teamOur"]["roles"][0]["backpack"] = ["stone"]
        payload["teamOur"]["roles"][1]["backpack"] = ["stone"]
        engine = DecisionEngine()

        engine.decide(payload)
        self.assertEqual(sum(
            plan.reason == "build:wall"
            for plan in engine.state.state.plans.values()
        ), 1)

        unknown = json.loads(json.dumps(payload))
        unknown["roundNo"] = 6
        unknown["lastRoundRoleActionResults"] = {}
        engine.decide(unknown)
        self.assertEqual(sum(
            plan.reason == "build:wall"
            for plan in engine.state.state.plans.values()
        ), 1)

        owner = next(
            role_id for role_id, plan in engine.state.state.plans.items()
            if plan.reason == "build:wall"
        )
        died = json.loads(json.dumps(unknown))
        died["roundNo"] = 7
        next(
            role for role in died["teamOur"]["roles"] if role["id"] == owner
        )["health"] = 0
        response = engine.decide(died)
        self.assertFalse(any(
            command.get("name") == "wall"
            for command in response["roleCommandMap"].values()
        ))
        self.assertFalse(any(
            plan.reason == "build:wall"
            for plan in engine.state.state.plans.values()
        ))

    def test_unknown_wall_build_feedback_does_not_duplicate_the_attempt(self):
        # Break caught: missing build feedback causes an immediate duplicate attempt.
        payload = base_payload(round_no=5, team_id="brain-c-wall-unknown")
        worker = unit(10010, "worker", 6, 9)
        worker["backpack"] = ["stone"]
        payload["teamOur"]["roles"] = [
            worker,
            unit(10011, "worker", 12, 9),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ]
        payload["teamOur"]["roles"][1]["backpack"] = ["stone"]
        engine = DecisionEngine()
        first = engine.decide(payload)
        self.assertTrue(any(
            command.get("name") == "wall"
            for command in first["roleCommandMap"].values()
        ))

        following = json.loads(json.dumps(payload))
        following["roundNo"] = 6
        following["lastRoundRoleActionResults"] = {}
        response = engine.decide(following)
        self.assertFalse(any(
            command.get("name") == "wall"
            for command in response["roleCommandMap"].values()
        ))

    def test_critical_gunner_uses_held_medicine_before_attacking(self):
        # Break caught: protected gunner filtering suppresses emergency self-care.
        payload = base_payload(round_no=71, team_id="brain-c-medicine")
        worker = unit(10010, "worker", 6, 5, health=1)
        worker["backpack"] = ["Medicine"]
        payload["teamOur"]["roles"] = [
            worker,
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
        ]
        payload["robot"]["roles"] = [{
            "id": 30001,
            "pos": {"x": 6, "y": 9},
            "roleType": "smallRobot",
            "health": 40,
            "abnormalState": "",
            "targetTeam": "challenger",
        }]

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"], {
            "10010": {"action": "use", "name": "Medicine"},
        })

    def test_destroyed_base_keeps_non_base_economy_action(self):
        # Break caught: station absence collapses the whole team response.
        payload = base_payload(round_no=5, team_id="brain-c-no-base")
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 2, "y": 2}, "neutralType": "iron"},
        ]
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 2, 1),
        ]

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "collect",
            "targetPos": [{"x": 2, "y": 2}],
        })


if __name__ == "__main__":
    unittest.main()
