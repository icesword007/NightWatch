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
    def test_left_base_uses_continuous_front_and_two_side_targets(self):
        # Break caught: a six-wall prefix omits the required continuous side wings.
        turn = layout_turn(our_x=9, enemy_x=17)

        targets = ordered_wall_targets(turn, wall_build_positions(turn))

        self.assertEqual(targets, (
            # Hand-derived from the 2x2 station footprint and ring-two boundary.
            Pos(12, 8),
            Pos(12, 9),
            Pos(12, 7),
            Pos(12, 10),
            Pos(12, 6), Pos(12, 11),
            Pos(11, 6), Pos(11, 11),
            Pos(10, 6), Pos(10, 11),
            Pos(9, 6), Pos(9, 11),
            Pos(8, 6), Pos(8, 11),
        ))

    def test_right_base_is_horizontal_mirror_with_fourteen_targets(self):
        turn = layout_turn(our_x=9, enemy_x=1)

        targets = ordered_wall_targets(turn, wall_build_positions(turn))

        self.assertEqual(targets, (
            Pos(7, 8), Pos(7, 9), Pos(7, 7), Pos(7, 10),
            Pos(7, 6), Pos(7, 11),
            Pos(8, 6), Pos(8, 11),
            Pos(9, 6), Pos(9, 11),
            Pos(10, 6), Pos(10, 11),
            Pos(11, 6), Pos(11, 11),
        ))
        self.assertEqual(len(targets), 14)

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
        self.assertFalse(engine.state.state.layout_complete)
        self.assertIsNotNone(engine.state.state.layout_degraded_reason)

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
        self.assertLessEqual(len(diagnostic["targets"]), 14)
        self.assertIn("towerGaps", diagnostic["layout"])
        self.assertIn("wallGaps", diagnostic["layout"])
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

    def test_movable_worker_on_fixed_target_is_rechecked_then_recovers(self):
        # Break caught: one transient friendly occupancy permanently creates a gap.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 5
        payload["mapInfo"].update({
            "width": 20,
            "height": 20,
            "zones": [{"pos": {"x": 5, "y": 8}, "neutralType": "stone"}],
        })
        builder = unit(10010, "worker", 11, 10)
        builder["backpack"] = ["stone"]
        payload["teamOur"].update({
            "teamId": "fortification-transient-worker",
            "roles": [
                builder,
                unit(10011, "worker", 12, 8),
                unit(10013, "station", 9, 9),
                unit(10020, "gatling", 8, 8),
                unit(10030, "railgun", 9, 7),
                unit(10040, "rocket", 10, 7),
            ],
        })
        payload["teamEnemy"]["roles"] = [unit(20013, "station", 17, 9)]
        payload["robot"]["roles"] = []
        state = SessionState("fortification-transient-worker", "challenger")
        state.fortification_initialized = True
        state.fortification_builder_id = 10010
        state.fortification_targets = (Pos(12, 8),)
        blocked_turn = Turn.load(payload)

        builder_id = prepare_fortification(
            blocked_turn,
            state,
            wall_build_positions(blocked_turn),
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=64,
        )

        self.assertIsNone(builder_id)
        self.assertNotIn(Pos(12, 8), state.fortification_failed)
        self.assertEqual(
            state.fortification_skip_reason, "target_temporarily_blocked",
        )

        payload["roundNo"] = 6
        payload["teamOur"]["roles"][1]["pos"] = {"x": 8, "y": 11}
        clear_turn = Turn.load(payload)
        builder_id = prepare_fortification(
            clear_turn,
            state,
            wall_build_positions(clear_turn),
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=64,
        )

        self.assertEqual(builder_id, 10010)
        self.assertEqual(state.fortification_batch_targets, (Pos(12, 8),))
        self.assertNotIn(Pos(12, 8), state.fortification_failed)

    def test_same_movable_blocker_can_clear_after_more_than_three_rounds(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 5
        payload["mapInfo"].update({"width": 20, "height": 20, "zones": []})
        builder = unit(10010, "worker", 11, 10)
        builder["backpack"] = ["stone"]
        payload["teamOur"].update({
            "teamId": "fortification-persistent-worker",
            "roles": [
                builder,
                unit(10011, "worker", 12, 8),
                unit(10013, "station", 9, 9),
                unit(10020, "gatling", 8, 8),
                unit(10030, "railgun", 9, 7),
                unit(10040, "rocket", 10, 7),
            ],
        })
        payload["teamEnemy"]["roles"] = [unit(20013, "station", 17, 9)]
        payload["robot"]["roles"] = []
        turn = Turn.load(payload)
        state = SessionState("fortification-persistent-worker", "challenger")
        state.fortification_initialized = True
        state.fortification_builder_id = 10010
        state.fortification_targets = (Pos(12, 8),)

        for round_no in range(5, 10):
            payload["roundNo"] = round_no
            turn = Turn.load(payload)
            prepare_fortification(
                turn,
                state,
                wall_build_positions(turn),
                clock=lambda: 0.0,
                deadline=1.0,
                max_expansions=64,
            )
            self.assertNotIn(Pos(12, 8), state.fortification_failed)

        payload["roundNo"] = 10
        payload["teamOur"]["roles"][1]["pos"] = {"x": 8, "y": 11}
        clear_turn = Turn.load(payload)
        builder_id = prepare_fortification(
            clear_turn,
            state,
            wall_build_positions(clear_turn),
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=64,
        )
        self.assertEqual(builder_id, 10010)
        self.assertNotIn(Pos(12, 8), state.fortification_failed)

    def test_unrelated_unit_changes_do_not_consume_direct_blocker_rechecks(self):
        # Break caught: global unit position/HP changes exhaust retries while the
        # same role continues to occupy the wall target.
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 5
        payload["mapInfo"].update({"width": 20, "height": 20, "zones": []})
        payload["teamOur"].update({
            "teamId": "fortification-unrelated-unit-change",
            "roles": [
                unit(10010, "worker", 5, 9),
                unit(10011, "pioneer", 12, 8),
                unit(10013, "station", 9, 9),
                unit(10020, "gatling", 8, 8),
                unit(10030, "railgun", 9, 7),
                unit(10040, "rocket", 10, 7),
            ],
        })
        payload["teamEnemy"]["roles"] = [unit(20013, "station", 17, 9)]
        payload["robot"]["roles"] = []
        target = Pos(12, 8)
        state = SessionState(
            "fortification-unrelated-unit-change", "challenger",
        )
        state.fortification_initialized = True
        state.fortification_builder_id = 10010
        state.fortification_targets = (target,)

        for index, worker_x in enumerate((5, 6, 7, 8), start=5):
            payload["roundNo"] = index
            payload["teamOur"]["roles"][0]["pos"] = {
                "x": worker_x, "y": 9,
            }
            payload["teamOur"]["roles"][0]["health"] = 100 - index
            turn = Turn.load(payload)
            accepted = fortification_module._current_safe_prefix(
                turn,
                state,
                (target,),
                wall_build_positions(turn),
            )
            self.assertEqual(accepted, ())
            self.assertNotIn(target, state.fortification_failed)

        payload["roundNo"] = 9
        payload["teamOur"]["roles"][1]["pos"] = {"x": 8, "y": 11}
        clear_turn = Turn.load(payload)
        accepted = fortification_module._current_safe_prefix(
            clear_turn,
            state,
            (target,),
            wall_build_positions(clear_turn),
        )

        self.assertEqual(accepted, (target,))
        self.assertNotIn(target, state.fortification_failed)

    def test_unchanged_temporary_block_does_not_repeat_safety_search(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 5
        payload["mapInfo"].update({"width": 20, "height": 20, "zones": []})
        builder = unit(10010, "worker", 11, 10)
        builder["backpack"] = ["stone"]
        payload["teamOur"].update({
            "teamId": "fortification-stable-robot-block",
            "roles": [
                builder,
                unit(10013, "station", 9, 9),
                unit(10020, "gatling", 8, 8),
                unit(10030, "railgun", 9, 7),
                unit(10040, "rocket", 10, 7),
            ],
        })
        payload["teamEnemy"]["roles"] = [unit(20013, "station", 17, 9)]
        back_passage = (
            (7, 6), (7, 7), (7, 8), (7, 9), (7, 10),
            (7, 11), (8, 6), (8, 11), (9, 6), (9, 11),
        )
        payload["robot"]["roles"] = [
            {
                "id": 30000 + index,
                "pos": {"x": x, "y": y},
                "roleType": "smallRobot",
                "health": 40,
                "attackPower": 5,
                "abnormalState": "",
                "targetTeam": "challenger",
            }
            for index, (x, y) in enumerate(back_passage)
        ]
        state = SessionState("fortification-stable-robot-block", "challenger")
        state.fortification_initialized = True
        state.fortification_builder_id = 10010
        state.fortification_targets = (Pos(12, 8),)
        real_safe_wall_targets = fortification_module.safe_wall_targets
        calls = 0

        def counted_safe_wall_targets(*args, **kwargs):
            nonlocal calls
            calls += 1
            return real_safe_wall_targets(*args, **kwargs)

        with patch.object(
            fortification_module,
            "safe_wall_targets",
            counted_safe_wall_targets,
        ):
            for round_no in range(5, 10):
                payload["roundNo"] = round_no
                turn = Turn.load(payload)
                prepare_fortification(
                    turn,
                    state,
                    wall_build_positions(turn),
                    clock=lambda: 0.0,
                    deadline=1.0,
                    max_expansions=64,
                )

        self.assertEqual(calls, 2)
        self.assertNotIn(Pos(12, 8), state.fortification_failed)

    def test_short_window_selects_only_feasible_wall_prefix(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 61
        payload["mapInfo"].update({
            "width": 20,
            "height": 20,
            "zones": [{"pos": {"x": 5, "y": 8}, "neutralType": "stone"}],
        })
        payload["teamOur"].update({
            "teamId": "fortification-short-prefix",
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
        state = SessionState("fortification-short-prefix", "challenger")

        builder_id = prepare_fortification(
            turn,
            state,
            wall_build_positions(turn),
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=256,
        )

        self.assertEqual(builder_id, 10010)
        self.assertEqual(len(state.fortification_batch_targets), 1)

    def test_existing_stone_avoids_unneeded_mine_detour_near_dusk(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 61
        payload["mapInfo"].update({
            "width": 20,
            "height": 20,
            "zones": [{"pos": {"x": 0, "y": 0}, "neutralType": "stone"}],
        })
        builder = unit(10010, "worker", 11, 8)
        builder["backpack"] = ["stone"]
        payload["teamOur"].update({
            "teamId": "fortification-held-stone-prefix",
            "roles": [
                builder,
                unit(10013, "station", 9, 9),
                unit(10020, "gatling", 8, 8),
                unit(10030, "railgun", 9, 7),
                unit(10040, "rocket", 10, 7),
            ],
        })
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 17, 9),
        ]
        payload["robot"]["roles"] = []
        state = SessionState(
            "fortification-held-stone-prefix", "challenger",
        )

        builder_id = prepare_fortification(
            Turn.load(payload),
            state,
            wall_build_positions(Turn.load(payload)),
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=256,
        )

        self.assertEqual(builder_id, 10010)
        self.assertEqual(len(state.fortification_batch_targets), 1)

    def test_batch_never_exceeds_current_backpack_capacity(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        payload["roundNo"] = 5
        payload["mapInfo"].update({
            "width": 20,
            "height": 20,
            "zones": [{"pos": {"x": 5, "y": 8}, "neutralType": "stone"}],
        })
        builder = unit(10010, "worker", 5, 9)
        builder["backPackCapability"] = 1
        payload["teamOur"].update({
            "teamId": "fortification-capacity-prefix",
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
        turn = Turn.load(payload)
        state = SessionState("fortification-capacity-prefix", "challenger")

        builder_id = prepare_fortification(
            turn,
            state,
            wall_build_positions(turn),
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=256,
        )

        self.assertEqual(builder_id, 10010)
        self.assertEqual(len(state.fortification_batch_targets), 1)

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
