import copy
import json
import unittest
from pathlib import Path

from agent.brain import decide
from agent.protocol import Turn, Unit


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def load_fixture():
    with FIXTURE.open(encoding="utf-8") as handle:
        return json.load(handle)


class ProtocolTests(unittest.TestCase):
    def test_c_decision_collects_an_observed_adjacent_mine(self):
        # Break caught: the HTTP coordinator still emits only the S0 probe.
        response = decide(load_fixture())

        self.assertEqual(response["roleCommandMap"], {
            "10010": {
                "action": "collect",
                "targetPos": [{"x": 1, "y": 1}],
            },
        })
        self.assertIn("UNTRUSTED_NEWS_DATA_BEGIN", response["prompt"])
        self.assertEqual(response["executeCmd"], "")

    def test_response_is_a_complete_json_object(self):
        # Break caught: HTTP code receives only the inner command map.
        encoded = json.dumps(decide(load_fixture()), ensure_ascii=False)

        self.assertIn("roleCommandMap", json.loads(encoded))
        self.assertIn("prompt", json.loads(encoded))
        self.assertIn("executeCmd", json.loads(encoded))

    def test_enemy_role_position_is_blocked(self):
        # Break caught: a visible enemy position is treated as an empty move target.
        turn = Turn.load(load_fixture())
        worker = turn.controllable()[0]

        enemies = getattr(turn, "enemies", ())
        self.assertTrue(enemies, "visible enemy roles were not parsed")
        self.assertIn(enemies[0].pos, turn.blocked(worker))

    def test_economy_prices_and_robot_threat_fields_are_parsed(self):
        # Break caught: C planners guess prices or lose robot target identity.
        payload = load_fixture()
        payload["vendorShopList"] = [{"name": "copper", "price": 5}]
        payload["weaponShopList"] = [{"name": "Medicine", "price": 10}]

        turn = Turn.load(payload)

        self.assertEqual(turn.vendor_prices, {"copper": 5})
        self.assertEqual(turn.weapon_prices, {"Medicine": 10})
        self.assertEqual(turn.robots[0].kind, "smallRobot")
        self.assertEqual(turn.robots[0].target_team, "challenger")
        self.assertEqual(turn.robots[0].abnormal_state, "")

    def test_missing_robot_target_team_remains_unknown(self):
        # Break caught: official sample omits targetTeam despite the field table.
        payload = load_fixture()
        del payload["robot"]["roles"][0]["targetTeam"]

        turn = Turn.load(payload)

        self.assertEqual(turn.robots[0].target_team, "")

    def test_player_task_timeout_is_optional_but_preserved_when_present(self):
        # Break caught: the field table and official sample disagree on timeoutRounds.
        payload = load_fixture()
        payload["teamOur"]["playerTasks"] = [
            {
                "taskType": "自进化类1",
                "taskPosition": {"x": 4, "y": 4},
                "coldDownRounds": 0,
                "scoreReward": 50,
                "goldReward": 30,
                "isValid": True,
                "timeoutRounds": 20,
            },
            {
                "taskType": "自进化类2",
                "taskPosition": {"x": 7, "y": 7},
                "coldDownRounds": 0,
                "scoreReward": 60,
                "goldReward": 40,
                "isValid": True,
            },
        ]

        turn = Turn.load(payload)

        self.assertEqual(turn.player_tasks[0].timeout_rounds, 20)
        self.assertIsNone(turn.player_tasks[1].timeout_rounds)
        self.assertEqual(turn.phase_task, "")

    def test_player_task_validity_rejects_non_boolean_values(self):
        # Break caught: the string "false" is truthy and makes an invalid task selectable.
        payload = load_fixture()
        payload["teamOur"]["playerTasks"] = [{
            "taskType": "自进化类1",
            "taskPosition": {"x": 4, "y": 4},
            "coldDownRounds": 0,
            "scoreReward": 50,
            "goldReward": 30,
            "isValid": "false",
        }]

        with self.assertRaises(ValueError):
            Turn.load(payload)

    def test_missing_role_id_is_rejected_instead_of_becoming_zero(self):
        # Break caught: a missing critical identifier silently becomes role ID 0.
        raw_role = copy.deepcopy(load_fixture()["teamOur"]["roles"][0])
        del raw_role["id"]

        with self.assertRaises(ValueError):
            Unit.load(raw_role)

    def test_missing_role_position_is_rejected_instead_of_becoming_origin(self):
        # Break caught: a role without a position is silently moved from (0, 0).
        payload = load_fixture()
        del payload["teamOur"]["roles"][0]["pos"]

        try:
            decide(payload)
        except Exception as error:
            self.assertIsInstance(error, ValueError)
        else:
            self.fail("missing role position was accepted")


if __name__ == "__main__":
    unittest.main()
