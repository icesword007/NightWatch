import json
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import agent.economy as economy_module
import agent.fortification as fortification_module
from agent.brain import DecisionEngine
from agent.economy import wall_build_positions
from agent.fortification import (
    direction_prior,
    ordered_wall_targets,
    prepare_fortification,
)
from agent.protocol import Pos, Turn
from agent.state import PlanState, SessionState, request_fingerprint


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def unit(unit_id, kind, x, y, *, health=1000):
    return {
        "id": unit_id,
        "pos": {"x": x, "y": y},
        "roleType": kind,
        "health": health,
        "attackPower": 10,
        "attackRange": 0,
        "backPackCapability": 100 if kind == "worker" else 0,
        "backpack": [],
        "level": 1,
        "cooldown": 0,
    }


def layout_turn(*, our_x, enemy_x):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["roundNo"] = 5
    payload["mapInfo"].update({"width": 20, "height": 20, "zones": []})
    payload["teamOur"].update({
        "teamId": "fortification-layout",
        "roles": [unit(10013, "station", our_x, 9)],
    })
    payload["teamEnemy"]["roles"] = [unit(20013, "station", enemy_x, 9)]
    payload["robot"]["roles"] = []
    return Turn.load(payload)


class FortificationTests(unittest.TestCase):
    def test_left_base_uses_four_front_and_two_side_targets(self):
        # Break caught: filling the whole front edge omits the required side wings.
        turn = layout_turn(our_x=9, enemy_x=17)

        targets = ordered_wall_targets(turn, wall_build_positions(turn))

        self.assertEqual(targets, (
            # Hand-derived from the 2x2 station footprint and ring-two boundary.
            Pos(12, 8),
            Pos(12, 9),
            Pos(12, 7),
            Pos(12, 10),
            Pos(11, 6),
            Pos(11, 11),
        ))

    def test_right_base_is_horizontal_mirror_and_stays_at_six_targets(self):
        turn = layout_turn(our_x=9, enemy_x=1)

        targets = ordered_wall_targets(turn, wall_build_positions(turn))

        self.assertEqual(targets, (
            Pos(7, 8), Pos(7, 9), Pos(7, 7), Pos(7, 10),
            Pos(8, 6), Pos(8, 11),
        ))
        self.assertEqual(len(targets), 6)

    def test_missing_enemy_uses_current_map_horizontal_x_fallback(self):
        turn = layout_turn(our_x=2, enemy_x=17)
        turn = replace(turn, enemies=())

        direction = direction_prior(turn)

        self.assertEqual((direction.dx, direction.dy), (1, 0))
        self.assertEqual(
            direction.source, "current_map_horizontal_map_center_x",
        )

    def test_wall_target_that_removes_a_towers_only_control_stand_is_skipped(self):
        # Break caught: wall geometry is accepted without a usable gunner stand.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 5
        payload["mapInfo"].update({
            "width": 20,
            "height": 20,
            "zones": [
                {"pos": {"x": x, "y": y}, "neutralType": "stone"}
                for x, y in ((10, 7), (11, 7), (11, 9), (12, 7), (12, 9))
            ],
        })
        builder = unit(10010, "worker", 11, 10)
        builder["backpack"] = ["stone"]
        payload["teamOur"].update({
            "teamId": "fortification-control-stand",
            "roles": [
                builder,
                unit(10011, "worker", 8, 11),
                unit(10012, "pioneer", 9, 11),
                unit(10013, "station", 9, 9),
                unit(10020, "gatling", 11, 8),
                unit(10030, "railgun", 8, 8),
                unit(10040, "rocket", 9, 7),
            ],
        })
        payload["teamEnemy"]["roles"] = [unit(20013, "station", 17, 9)]
        payload["robot"]["roles"] = []

        engine = DecisionEngine()
        engine.decide(payload)

        self.assertNotIn(Pos(12, 8), engine.state.state.fortification_targets)

    def test_decision_trace_has_bounded_current_map_fortification_diagnostic(self):
        # Break caught: intranet cannot correlate wall intent with map direction.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 5
        payload["mapInfo"].update({
            "width": 20,
            "height": 20,
            "zones": [{
                "pos": {"x": 5, "y": 8}, "neutralType": "stone",
            }],
        })
        payload["teamOur"].update({
            "teamId": "fortification-diagnostic",
            "roles": [
                unit(10010, "worker", 5, 9),
                unit(10013, "station", 9, 9),
                unit(10020, "gatling", 8, 8),
                unit(10030, "railgun", 9, 7),
                unit(10040, "rocket", 10, 7),
            ],
        })
        payload["teamEnemy"]["roles"] = [unit(20013, "station", 17, 9)]
        payload["robot"]["roles"] = []
        traces = []

        DecisionEngine().decide(payload, trace_sink=traces.append)

        diagnostic = traces[0]["economyPlanning"]["fortification"]
        self.assertEqual(diagnostic["direction"], {"x": 1, "y": 0})
        self.assertEqual(
            diagnostic["directionSource"], "current_map_horizontal_enemy_x",
        )
        self.assertLessEqual(len(diagnostic["targets"]), 6)
        self.assertEqual(diagnostic["builderId"], "10010")
        self.assertEqual(diagnostic["phase"], "mining")

    def test_missing_stone_has_specific_bounded_skip_reason(self):
        # Break caught: a missing resource is mislabeled as a return deadline.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 5
        payload["mapInfo"].update({"width": 20, "height": 20, "zones": []})
        payload["teamOur"].update({
            "teamId": "fortification-no-stone",
            "roles": [
                unit(10010, "worker", 5, 9),
                unit(10013, "station", 9, 9),
                unit(10020, "gatling", 8, 8),
                unit(10030, "railgun", 9, 7),
                unit(10040, "rocket", 10, 7),
            ],
        })
        payload["teamEnemy"]["roles"] = [unit(20013, "station", 17, 9)]
        payload["robot"]["roles"] = []
        traces = []

        DecisionEngine().decide(payload, trace_sink=traces.append)

        self.assertEqual(
            traces[0]["economyPlanning"]["fortification"]["skipReason"],
            "stone_unavailable",
        )

    def test_existing_builder_waits_when_a_fund_plan_becomes_active(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 5
        payload["mapInfo"].update({
            "width": 20,
            "height": 20,
            "zones": [{"pos": {"x": 5, "y": 8}, "neutralType": "stone"}],
        })
        payload["teamOur"].update({
            "teamId": "fortification-builder-fund",
            "roles": [
                unit(10010, "worker", 5, 9),
                unit(10013, "station", 9, 9),
                unit(10020, "gatling", 8, 8),
                unit(10030, "railgun", 9, 7),
                unit(10040, "rocket", 10, 7),
            ],
        })
        payload["teamEnemy"]["roles"] = [unit(20013, "station", 17, 9)]
        payload["robot"]["roles"] = []
        turn = Turn.load(payload)
        state = SessionState("fortification-builder-fund", "challenger")
        state.fortification_builder_id = 10010
        state.fortification_initialized = True
        state.fortification_targets = (Pos(12, 8),)
        state.plans[10010] = PlanState(
            10010,
            Pos(4, 2),
            "fund:WeaponUpgradeVoucher1:10020:10020",
            60,
            1,
        )

        builder = prepare_fortification(
            turn,
            state,
            wall_build_positions(turn),
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=64,
        )

        self.assertIsNone(builder)
        self.assertEqual(state.fortification_skip_reason, "worker_protected")

    def test_current_occupancy_revalidates_the_next_fixed_wall_target(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 5
        payload["mapInfo"].update({"width": 20, "height": 20, "zones": []})
        builder = unit(10010, "worker", 11, 10)
        builder["backpack"] = ["stone"]
        payload["teamOur"].update({
            "teamId": "fortification-revalidate",
            "roles": [
                builder,
                unit(10011, "worker", 8, 11),
                unit(10012, "pioneer", 9, 11),
                unit(10013, "station", 9, 9),
                unit(10020, "gatling", 11, 8),
                unit(10030, "railgun", 8, 8),
                unit(10040, "rocket", 9, 7),
            ],
        })
        payload["teamEnemy"]["roles"] = [unit(20013, "station", 17, 9)]
        payload["robot"]["roles"] = []
        engine = DecisionEngine()
        engine.decide(payload)
        self.assertEqual(engine.state.state.fortification_targets[0], Pos(12, 8))

        changed = json.loads(json.dumps(payload))
        changed["roundNo"] = 6
        changed["lastRoundRoleActionResults"] = {"10010": True}
        changed["mapInfo"]["zones"] = [
            {"pos": {"x": x, "y": y}, "neutralType": "stone"}
            for x, y in ((10, 7), (11, 7), (11, 9), (12, 7), (12, 9))
        ]
        engine.decide(changed)

        self.assertIn(Pos(12, 8), engine.state.state.fortification_failed)

    def test_dusk_does_not_pass_stable_builder_as_economy_reservation(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 65
        payload["mapInfo"].update({"width": 20, "height": 20, "zones": []})
        builder = unit(10010, "worker", 11, 8)
        builder["backpack"] = ["stone"]
        payload["teamOur"].update({
            "teamId": "fortification-dusk-release",
            "roles": [
                builder,
                unit(10013, "station", 9, 9),
                unit(10020, "gatling", 8, 8),
                unit(10030, "railgun", 9, 7),
                unit(10040, "rocket", 10, 7),
            ],
        })
        payload["teamEnemy"]["roles"] = [unit(20013, "station", 17, 9)]
        payload["robot"]["roles"] = []
        engine = DecisionEngine()
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.state.fortification_builder_id = 10010
        captured = {}

        def capture_economy(*args, **kwargs):
            captured["builder"] = kwargs["fortification_builder_id"]
            kwargs["diagnostic_sink"]({})
            return ()

        with patch("agent.brain.propose_economy", side_effect=capture_economy):
            engine.decide(payload)

        self.assertIsNone(captured["builder"])

    def test_large_map_fortification_search_and_response_stay_bounded(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 30
        payload["mapInfo"].update({
            "width": 41,
            "height": 32,
            "zones": [
                {"pos": {"x": 3, "y": 1}, "neutralType": "stone"},
                *[
                    {
                        "pos": {"x": 10, "y": y},
                        "neutralType": "defenderTaskPoint1",
                    }
                    for y in range(32) if y != 15
                ],
            ],
        })
        payload["teamOur"].update({
            "teamId": "fortification-large-map",
            "roles": [
                unit(10010, "worker", 2, 1),
                unit(10011, "worker", 2, 3),
                unit(10012, "pioneer", 2, 5),
                unit(10013, "station", 20, 16),
                unit(10020, "gatling", 19, 15),
                unit(10030, "railgun", 20, 14),
                unit(10040, "rocket", 21, 14),
            ],
        })
        payload["teamEnemy"]["roles"] = [unit(20013, "station", 35, 16)]
        payload["robot"]["roles"] = []
        calls = 0
        real_fortification_step = fortification_module.next_step
        real_economy_step = economy_module.next_step

        def counted_fortification_step(*args, **kwargs):
            nonlocal calls
            calls += 1
            return real_fortification_step(*args, **kwargs)

        def counted_economy_step(*args, **kwargs):
            nonlocal calls
            calls += 1
            return real_economy_step(*args, **kwargs)

        started = time.monotonic()
        with patch.object(
            fortification_module, "next_step", counted_fortification_step,
        ), patch.object(economy_module, "next_step", counted_economy_step):
            response = DecisionEngine().decide(payload)
        elapsed = time.monotonic() - started

        self.assertIn("10010", response["roleCommandMap"])
        self.assertLessEqual(calls, 32)
        self.assertLess(elapsed, 4.0)


if __name__ == "__main__":
    unittest.main()
