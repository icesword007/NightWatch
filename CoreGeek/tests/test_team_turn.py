import io
import json
import unittest
from pathlib import Path
from unittest import mock

from agent import server as server_module
from agent.brain import DecisionEngine


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

    def test_completed_tower_line_can_emit_one_early_wall_trial(self):
        # Break caught: the wall capability is never reachable from the coordinator.
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

        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "build")
        self.assertEqual(response["roleCommandMap"]["10010"]["name"], "wall")

        following = json.loads(json.dumps(payload))
        following["roundNo"] = 6
        following["lastRoundRoleActionResults"] = {"10010": False}
        response = engine.decide(following)
        self.assertFalse(any(
            command.get("action") == "build" and command.get("name") == "wall"
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
