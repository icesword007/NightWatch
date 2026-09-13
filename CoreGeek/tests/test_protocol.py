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
    def test_minimal_decision_moves_first_role_to_observed_empty_neighbour(self):
        # Break caught: enabling the old demo strategy or choosing an observed blocker.
        response = decide(load_fixture())

        self.assertEqual(
            response,
            {
                "roleCommandMap": {
                    "10010": {
                        "action": "move",
                        "targetPos": [{"x": 2, "y": 1}],
                    }
                },
                "prompt": "",
                "executeCmd": "",
            },
        )

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
