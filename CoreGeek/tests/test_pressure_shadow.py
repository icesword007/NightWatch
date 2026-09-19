import copy
import json
import unittest
from pathlib import Path

from agent.brain import DecisionEngine
from agent.pressure_shadow import ShadowState, observe_shadow, shadow_diagnostic
from agent.protocol import Turn


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def payload(round_no, *, team="challenger", robots=(), our_hp=100, enemy_hp=100,
            our_level=1, enemy_level=1, our_walls=(), enemy_walls=(),
            robots_observed=True):
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value["roundNo"] = round_no
    value["teamOur"].update(type=team, teamId="shadow-test")
    value["teamOur"]["roles"] = [
        dict(id=10013, roleType="station", pos=dict(x=2, y=2),
             health=our_hp, level=our_level), *our_walls,
    ]
    value["teamEnemy"]["roles"] = [
        dict(id=20013, roleType="station", pos=dict(x=18, y=18),
             health=enemy_hp, level=enemy_level), *enemy_walls,
    ]
    value["robot"] = {"roles": list(robots)} if robots_observed else {}
    return value


def robot(robot_id, target, health=50, *, kind="smallRobot", x=3, y=3):
    return dict(id=robot_id, roleType=kind, pos=dict(x=x, y=y),
                health=health, targetTeam=target)


def wall(wall_id, health=100, level=1):
    return dict(id=wall_id, roleType="wall", pos=dict(x=5, y=5),
                health=health, level=level)


class PressureShadowTests(unittest.TestCase):
    def observe(self, state, round_no, **kwargs):
        observe_shadow(Turn.load(payload(round_no, **kwargs)), state)
        return shadow_diagnostic(state)

    def test_both_sides_and_all_robots_are_counted(self):
        state = ShadowState()
        robots = [robot(i, "challenger", x=3, y=3) for i in range(20)]
        robots += [robot(30, "defender", kind="largeRobot", x=17, y=18),
                   robot(31, "", x=0, y=0)]
        result = self.observe(state, 71, robots=robots)
        self.assertEqual(result["current"]["sides"]["challenger"]["robots"]["count"], 20)
        self.assertEqual(result["current"]["sides"]["defender"]["robots"]["count"], 1)
        self.assertEqual(result["current"]["unknownTargetRobots"], 1)
        self.assertEqual(result["current"]["sides"]["challenger"]["robots"]["nearestBaseChebyshev"], 1)
        self.assertIn("unknown_target_robots", result["current"]["unknowns"])

    def test_missing_robots_and_id_limit_mark_incomplete(self):
        state = ShadowState()
        missing = self.observe(state, 71, robots_observed=False)
        self.assertFalse(missing["current"]["complete"])
        self.assertIsNone(missing["current"]["sides"]["challenger"]["robots"]["count"])
        many = [robot(i, "challenger") for i in range(600)]
        result = self.observe(state, 72, robots=many)
        self.assertEqual(result["current"]["sides"]["challenger"]["robots"]["count"], 600)
        self.assertIn("robot_id_limit", result["current"]["unknowns"])

    def test_wall_disappearance_repair_upgrade_and_robot_hp_are_observations(self):
        state = ShadowState()
        self.observe(state, 71, robots=[robot(1, "challenger", 50)],
                     our_walls=[wall(11, 100), wall(12, 80)])
        result = self.observe(state, 72, robots=[robot(1, "challenger", 30)],
                              our_walls=[wall(11, 120, 2)])
        side = result["current"]["sides"]["challenger"]
        self.assertEqual(side["robots"]["observedHpLoss"], 20)
        self.assertEqual(side["walls"]["disappeared"], 1)
        self.assertEqual(side["walls"]["upgraded"], 1)
        self.assertEqual(side["walls"]["repairedSameLevel"], 0)
        self.assertEqual(side["criticalWallBreach"], "unknown")
        repaired = self.observe(state, 73, our_walls=[wall(11, 130, 2)])
        self.assertEqual(repaired["current"]["sides"]["challenger"]["walls"]["repairedSameLevel"], 1)

    def test_prediction_freezes_before_night_and_scores_complete_positive(self):
        state = ShadowState()
        daytime = copy.deepcopy(self.observe(state, 70)["prediction"])
        first = self.observe(state, 71)
        self.assertEqual(first["prediction"]["risk"], "unknown")
        self.assertEqual(first["prediction"]["issuedRound"], 70)
        self.assertEqual(first["prediction"], daytime)
        for r in range(72, 131):
            self.observe(state, r, our_hp=90 if r >= 75 else 100)
        scored = self.observe(state, 131, our_hp=90)
        self.assertEqual(scored["verification"]["actualBaseDamage"], True)
        self.assertEqual(scored["verification"]["result"], "unscorable")
        self.assertEqual(scored["prediction"]["risk"], "elevated")
        self.assertEqual(scored["prediction"]["basedOnNight"], 1)

    def test_dusk_and_dawn_base_losses_belong_to_that_night(self):
        for loss_round in (71, 131):
            state = ShadowState()
            self.observe(state, 70, our_hp=100)
            for round_no in range(71, 131):
                self.observe(state, round_no,
                             our_hp=90 if round_no >= loss_round else 100)
            result = self.observe(state, 131, our_hp=90)
            self.assertTrue(result["verification"]["actualBaseDamage"])

    def test_enemy_base_damage_is_scored_independently(self):
        state = ShadowState()
        self.observe(state, 70)
        for round_no in range(71, 131):
            self.observe(state, round_no, enemy_hp=90)
        result = self.observe(state, 131, enemy_hp=90)
        self.assertEqual(result["verification"]["sides"]["challenger"]["actualBaseDamage"], False)
        self.assertEqual(result["verification"]["sides"]["defender"]["actualBaseDamage"], True)

    def test_complete_negative_and_changed_defense_unknown(self):
        state = ShadowState()
        for r in range(1, 132):
            self.observe(state, r)
        self.assertEqual(shadow_diagnostic(state)["verification"]["actualBaseDamage"], False)
        self.assertEqual(shadow_diagnostic(state)["prediction"]["risk"], "low")
        changed = self.observe(state, 132, our_walls=[wall(11)])
        self.assertEqual(changed["prediction"]["risk"], "unknown")
        self.assertTrue(changed["prediction"]["changedDefense"])

    def test_missing_frame_and_early_night_do_not_score_negative(self):
        state = ShadowState()
        self.observe(state, 70)
        self.observe(state, 71)
        self.observe(state, 73)
        result = self.observe(state, 131)
        self.assertIsNone(result["verification"]["actualBaseDamage"])
        self.assertEqual(result["verification"]["result"], "unscorable")
        state2 = ShadowState()
        self.observe(state2, 70)
        self.observe(state2, 71)
        result2 = self.observe(state2, 72)
        self.assertIsNone(result2["verification"])

    def test_daytime_gap_keeps_forecast_unknown_after_next_complete_frame(self):
        state = ShadowState()
        for round_no in range(1, 132):
            self.observe(state, round_no)
        self.assertEqual(shadow_diagnostic(state)["prediction"]["risk"], "low")
        self.observe(state, 133)
        later = self.observe(state, 134)
        self.assertEqual(later["prediction"]["baselineRisk"], "low")
        self.assertEqual(later["prediction"]["risk"], "unknown")
        self.assertIn("day_observation_incomplete", later["prediction"]["unknowns"])

    def test_dusk_base_identity_change_excludes_negative_verification(self):
        state = ShadowState()
        self.observe(state, 70, our_level=1)
        for round_no in range(71, 131):
            self.observe(state, round_no, our_level=2)
        result = self.observe(state, 131, our_level=2)
        self.assertIsNone(result["verification"]["actualBaseDamage"])
        self.assertEqual(result["verification"]["result"], "unscorable")

    def test_scoring_all_four_outcomes_with_complete_nights(self):
        state = ShadowState()
        def health(round_no):
            if round_no >= 595:
                return 70
            if round_no >= 205:
                return 80
            if round_no >= 75:
                return 90
            return 100
        results = {}
        for round_no in range(1, 652):
            result = self.observe(state, round_no, our_hp=health(round_no))
            if round_no in (261, 391, 521, 651):
                results[round_no] = result["verification"]["result"]
        self.assertEqual(results, {
            261: "hit", 391: "false_positive",
            521: "correct_negative", 651: "false_negative",
        })
        self.assertEqual(len(state.recent_nights), 3)

    def test_morning_robot_disappearance_is_not_a_kill_claim(self):
        state = ShadowState()
        self.observe(state, 130, robots=[robot(1, "challenger")])
        result = self.observe(state, 131, robots=[])
        self.assertEqual(result["current"]["sides"]["challenger"]["robots"]["disappeared"], 0)
        self.assertNotIn("kills", result["current"]["sides"]["challenger"]["robots"])

    def test_repeat_and_out_of_order_leave_state_unchanged(self):
        state = ShadowState()
        self.observe(state, 71)
        expected = copy.deepcopy(shadow_diagnostic(state))
        self.observe(state, 71, our_hp=1)
        self.observe(state, 70, our_hp=1)
        self.assertEqual(shadow_diagnostic(state), expected)

    def test_engine_response_is_equivalent_with_shadow_disabled(self):
        first = DecisionEngine()
        second = DecisionEngine()
        from unittest import mock
        for round_no in (1, 70, 71, 72, 130, 131):
            value = payload(round_no, robots=[robot(1, "challenger")]
                            if round_no in (71, 72, 130) else [])
            value["teamOur"]["roles"].append(dict(
                id=10011, roleType="pioneer", pos=dict(x=4, y=4),
                health=200, backpack=[], backPackCapability=40,
            ))
            value["teamOur"]["goldNum"] = 100
            value["weaponShopList"] = [{"name": "Medicine", "price": 10}]
            if round_no == 70:
                value["phaseTask"] = "Return the observed marker."
            normal = first.decide(copy.deepcopy(value))
            with mock.patch("agent.brain.observe_shadow"):
                disabled = second.decide(copy.deepcopy(value))
            self.assertEqual(normal, disabled)

    def test_engine_restart_has_fresh_shadow_session(self):
        engine = DecisionEngine()
        traces = []
        engine.decide(payload(71), trace_sink=traces.append)
        changed = payload(1)
        changed["teamOur"]["teamId"] = "another-session"
        engine.decide(changed, trace_sink=traces.append)
        self.assertEqual(traces[-1]["pressureShadow"]["current"]["roundNo"], 1)
        self.assertEqual(traces[-1]["pressureShadow"]["recentNights"], [])
        self.assertIn("elapsedMs", traces[-1]["pressureShadow"])

    def test_action_equivalence_covers_economy_defense_and_task(self):
        from unittest import mock
        from tests.test_defense import defense_payload
        from tests.test_economy import economy_payload, with_completed_wall_line
        from tests.test_intelligence import task_payload

        economy = with_completed_wall_line(
            economy_payload(worker_pos=(11, 7), gold=20)
        )
        economy["mapInfo"]["zones"] = [
            {"pos": {"x": 10, "y": 7}, "neutralType": "weaponShop"},
        ]
        economy["weaponShopList"] = [
            {"name": "WallUpgradeVoucher1", "price": 20},
        ]
        cases = (
            ("economy", economy),
            ("defense", defense_payload()),
            ("task", task_payload(1, phase="Return marker.")),
        )
        for name, value in cases:
            with self.subTest(name=name):
                traces = []
                normal = DecisionEngine().decide(
                    copy.deepcopy(value), trace_sink=traces.append,
                )
                with mock.patch("agent.brain.observe_shadow"):
                    disabled = DecisionEngine().decide(copy.deepcopy(value))
                self.assertEqual(normal, disabled)
                if name == "task":
                    self.assertTrue(normal["prompt"])
                else:
                    self.assertIn(name, [
                        action["domain"] for action in traces[0]["actions"]
                    ])


if __name__ == "__main__":
    unittest.main()
