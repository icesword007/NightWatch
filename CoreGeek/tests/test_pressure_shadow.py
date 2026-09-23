import copy
import json
import unittest
from pathlib import Path

from agent.actions import ActionProposal, PlannedAction
from agent.brain import DecisionEngine
from agent.pressure_shadow import (
    ShadowState, historical_investment_assessment, observe_shadow,
    shadow_diagnostic,
)
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


def wall(wall_id, health=100, level=1, *, x=5, y=5):
    return dict(id=wall_id, roleType="wall", pos=dict(x=x, y=y),
                health=health, level=level)


class PressureShadowTests(unittest.TestCase):
    def test_historical_investment_uses_complete_night_and_dawn(self):
        for team in ("challenger", "defender"):
            with self.subTest(team=team):
                state = ShadowState()
                for round_no in range(1, 132):
                    turn = Turn.load(payload(
                        round_no, team=team,
                        our_hp=1000 if round_no < 75 else 500,
                    ))
                    observe_shadow(turn, state, frozenset())
                    assessment = historical_investment_assessment(turn, state)
                    if round_no == 130:
                        self.assertIn(
                            "current_night_in_progress", assessment["unknowns"],
                        )
                self.assertEqual(assessment["baselineNight"], 1)
                self.assertEqual(assessment["observedDamage"], 500)
                self.assertEqual(assessment["currentMargin"], 0)
                self.assertEqual(assessment["upgradeMargin"], 2500)
                self.assertEqual(
                    assessment["defenseComparison"], "approximately_comparable",
                )
                self.assertEqual(assessment["candidate"]["status"], "not_evaluated")

    def test_historical_investment_rejects_gaps_healing_and_level_changes(self):
        for label, skip, hp_for, level_for, expected in (
            ("gap", {90}, lambda r: 900, lambda r: 1,
             "night_or_dawn_sequence_incomplete"),
            ("healing", set(), lambda r: 900 if r < 90 else 950,
             lambda r: 1, "base_hp_increased"),
            ("level", set(), lambda r: 900, lambda r: 2 if r >= 100 else 1,
             "base_level_changed"),
        ):
            with self.subTest(label=label):
                state = ShadowState()
                for round_no in range(1, 132):
                    if round_no in skip:
                        continue
                    turn = Turn.load(payload(
                        round_no, our_hp=hp_for(round_no),
                        our_level=level_for(round_no),
                    ))
                    observe_shadow(turn, state, frozenset())
                assessment = historical_investment_assessment(turn, state)
                self.assertIsNone(assessment["observedDamage"])
                self.assertIn(expected, assessment["unknowns"])

    def test_historical_investment_marks_defense_changes_without_damage_factor(self):
        state = ShadowState()
        for round_no in range(1, 133):
            value = payload(
                round_no, our_hp=900 if round_no >= 90 else 1000,
                our_walls=(wall(11, health=90 if round_no >= 90 else 100),),
            )
            value["teamOur"]["roles"].extend((
                dict(id=10020, roleType="gatling", pos=dict(x=7, y=7),
                     health=1000, level=2 if round_no == 132 else 1),
                dict(id=10010, roleType="worker", pos=dict(x=6, y=7),
                     health=220, backpack=[], backPackCapability=40),
            ))
            turn = Turn.load(value)
            observe_shadow(
                turn, state,
                frozenset((10010,)) if 90 <= round_no < 100 else frozenset(),
            )
        assessment = historical_investment_assessment(turn, state)
        self.assertEqual(assessment["status"], "assessed")
        self.assertEqual(assessment["observedDamage"], 100)
        self.assertEqual(assessment["defenseComparison"], "changed_unquantified")
        self.assertIn("towers_changed", assessment["defenseChanges"])
        self.assertIn("task_occupancy_changed", assessment["defenseChanges"])
        self.assertIn("night_walls_changed", assessment["defenseChanges"])

    def test_historical_investment_needs_dawn_and_current_base_level(self):
        state = ShadowState()
        for round_no in (*range(1, 131), 132):
            turn = Turn.load(payload(round_no, our_hp=900))
            observe_shadow(turn, state, frozenset())
        assessment = historical_investment_assessment(turn, state)
        self.assertIn("night_or_dawn_sequence_incomplete", assessment["unknowns"])

        state = ShadowState()
        for round_no in range(1, 133):
            turn = Turn.load(payload(
                round_no, our_hp=900,
                our_level=2 if round_no == 132 else 1,
            ))
            observe_shadow(turn, state, frozenset())
        assessment = historical_investment_assessment(turn, state)
        self.assertIn("current_base_identity_or_level_changed", assessment["unknowns"])

        state = ShadowState()
        for round_no in range(1, 133):
            turn = Turn.load(payload(
                round_no, our_hp=950 if round_no == 132 else 900,
            ))
            observe_shadow(turn, state, frozenset())
        assessment = historical_investment_assessment(turn, state)
        self.assertIn("day_base_hp_increased", assessment["unknowns"])

    def test_historical_investment_day_gap_stays_unknown_after_resuming(self):
        for changed_hp in (False, True):
            with self.subTest(changed_hp=changed_hp):
                state = ShadowState()
                for round_no in range(1, 132):
                    turn = Turn.load(payload(
                        round_no, our_hp=1000 if round_no < 90 else 500,
                    ))
                    observe_shadow(turn, state, frozenset())
                for round_no in (140, 141, 142):
                    turn = Turn.load(payload(
                        round_no, our_hp=800 if changed_hp else 500,
                    ))
                    observe_shadow(turn, state, frozenset())
                    assessment = historical_investment_assessment(turn, state)
                    self.assertIsNone(assessment["observedDamage"])
                    self.assertIn(
                        "day_observation_incomplete", assessment["unknowns"],
                    )

    def test_historical_investment_missing_base_and_zero_damage_are_not_safe(self):
        state = ShadowState()
        for round_no in range(1, 132):
            value = payload(round_no, our_hp=900)
            if round_no == 90:
                value["teamOur"]["roles"] = []
            turn = Turn.load(value)
            observe_shadow(turn, state, frozenset())
        self.assertIn(
            "base_missing",
            historical_investment_assessment(turn, state)["unknowns"],
        )

        state = ShadowState()
        for round_no in range(1, 132):
            turn = Turn.load(payload(round_no, our_hp=4500, our_level=3))
            observe_shadow(turn, state, frozenset())
        assessment = historical_investment_assessment(turn, state)
        self.assertEqual(assessment["observedDamage"], 0)
        self.assertEqual(assessment["status"], "no_upgrade_candidate")
        self.assertIsNone(assessment["upgradeFullHp"])

    def test_history_switch_preserves_commands_prompts_and_model_requests(self):
        enabled = DecisionEngine(history_investment_enabled=True)
        disabled = DecisionEngine(history_investment_enabled=False)
        for round_no in (1, 70, 71, 72, 130, 131):
            value = payload(round_no, robots=[robot(1, "challenger")]
                            if round_no in (71, 72, 130) else [])
            value["teamOur"]["roles"].append(dict(
                id=10011, roleType="pioneer", pos=dict(x=4, y=4),
                health=200, backpack=[], backPackCapability=40,
            ))
            if round_no == 70:
                value["phaseTask"] = "Return the observed marker."
            on_traces, off_traces = [], []
            on = enabled.decide(copy.deepcopy(value), trace_sink=on_traces.append)
            off = disabled.decide(copy.deepcopy(value), trace_sink=off_traces.append)
            self.assertEqual(on, off)
            self.assertIn("historyInvestment", on_traces[0])
            self.assertNotIn("historyInvestment", off_traces[0])

    def test_historical_investment_candidate_uses_existing_route_only(self):
        from test_base_upgrade import base_payload, seed_reserve
        traces = []
        DecisionEngine().decide(base_payload(), trace_sink=traces.append)
        candidate = traces[0]["historyInvestment"]["candidate"]
        self.assertEqual(candidate["status"], "allocator_accepted")
        self.assertEqual(candidate["finalResponse"], "selected")
        self.assertEqual(candidate["route"], "evaluated")
        self.assertEqual(candidate["routeScope"], "purchase_use_return")
        self.assertEqual(candidate["totalRounds"], 10)

        engine = DecisionEngine()
        held = base_payload(
            round_no=71, worker_pos=(8, 9), gold=0, health=1000,
            items=("StationUpgradeVoucher1",),
        )
        seed_reserve(engine, held)
        traces = []
        engine.decide(held, trace_sink=traces.append)
        candidate = traces[0]["historyInvestment"]["candidate"]
        self.assertEqual(candidate["status"], "allocator_accepted")
        self.assertEqual(candidate["finalResponse"], "selected")
        self.assertEqual(candidate["action"], "use")
        self.assertEqual(candidate["routeScope"], "use_only")
        self.assertEqual(candidate["execution"], "unconfirmed")

        self.assertEqual(
            DecisionEngine._history_base_candidate((), None, [], False),
            {"status": "not_evaluated", "route": "not_evaluated"},
        )

        pending = PlannedAction(
            ActionProposal(10010, 10010, {
                "action": "buy", "name": "StationUpgradeVoucher1", "num": 1,
            }),
            plan_reason="fund:StationUpgradeVoucher1:10020:10013",
            estimated_rounds=10,
        )
        fallback = DecisionEngine._history_base_candidate(
            (pending,), None, [("economy", pending)], False,
        )
        self.assertEqual(fallback["status"], "allocator_accepted")
        self.assertEqual(fallback["finalResponse"], "not_confirmed")

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
        self.assertEqual(result["nightSummary"]["firstNightRobotDifference"]
                         ["status"], "unknown")

    def test_first_night_robot_difference_is_frozen_and_team_relative(self):
        for team, ours_target, enemy_target in (
            ("challenger", "challenger", "defender"),
            ("defender", "defender", "challenger"),
        ):
            with self.subTest(team=team):
                state = ShadowState()
                first = self.observe(state, 71, team=team, robots=[
                    robot(1, ours_target), robot(2, ours_target),
                    robot(3, ours_target), robot(4, enemy_target),
                ])
                observed = first["nightSummary"]["firstNightRobotDifference"]
                self.assertEqual(observed["status"], "conditional_observation")
                self.assertEqual(observed["rawDifference"], 2)
                self.assertEqual(observed["inferredOpponentAdditions"], 2)
                self.assertEqual(observed["ourAppliedAdditions"], 0)
                self.assertIn("timing_unverified", observed["unknowns"])
                later = self.observe(state, 72, team=team, robots=[])
                self.assertEqual(later["nightSummary"]["firstNightRobotDifference"], observed)

    def test_first_night_robot_difference_missing_or_late_stays_unknown(self):
        state = ShadowState()
        missing = self.observe(state, 71, robots_observed=False)
        observation = missing["nightSummary"]["firstNightRobotDifference"]
        self.assertEqual(observation["status"], "unknown")
        self.assertIsNone(observation["rawDifference"])
        later = self.observe(state, 72, robots=[robot(1, "challenger")])
        self.assertEqual(later["nightSummary"]["firstNightRobotDifference"], observation)
        late = self.observe(ShadowState(), 72, robots=[])
        self.assertIn("night_started_late",
                      late["nightSummary"]["firstNightRobotDifference"]["unknowns"])

    def test_negative_first_night_difference_is_inconsistent_not_normal_additions(self):
        state = ShadowState()
        result = self.observe(state, 71, robots=[
            robot(1, "defender"), robot(2, "defender"),
            robot(3, "challenger"),
        ])
        observed = result["nightSummary"]["firstNightRobotDifference"]
        self.assertEqual(observed["rawDifference"], -1)
        self.assertEqual(observed["status"], "inconsistent_observation")
        self.assertIsNone(observed["inferredOpponentAdditions"])
        self.assertIn("negative_difference", observed["unknowns"])

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
        self.assertEqual(scored["prediction"]["persistenceBaseline"]["risk"], "elevated")
        self.assertEqual(scored["prediction"]["risk"], "unknown")
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
        self.assertEqual(shadow_diagnostic(state)["prediction"]["persistenceBaseline"]["risk"], "low")
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
        self.assertEqual(shadow_diagnostic(state)["prediction"]["persistenceBaseline"]["risk"], "low")
        self.observe(state, 133)
        later = self.observe(state, 134)
        self.assertEqual(later["prediction"]["persistenceBaseline"]["risk"], "low")
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
                results[round_no] = result["verification"]["baselineResult"]
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
        self.assertIn(
            "adjacent_complete_night_missing",
            traces[-1]["historyInvestment"]["unknowns"],
        )

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

    def test_third_night_baseline_miss_is_visible_before_sunrise(self):
        state = ShadowState()
        for round_no in range(1, 341):
            hp = 90 if round_no >= 340 else 100
            result = self.observe(state, round_no, our_hp=hp)
        prediction = result["prediction"]["sides"]["challenger"]
        self.assertEqual(prediction["persistenceBaseline"]["risk"], "low")
        self.assertEqual(prediction["assessment"]["risk"], "unknown")
        verification = result["verification"]
        self.assertEqual(verification["night"], 3)
        self.assertEqual(verification["status"], "event_observed")
        self.assertFalse(verification["complete"])
        self.assertEqual(verification["firstDamageRound"], 340)
        self.assertEqual(verification["sides"]["challenger"]["baselineResult"], "false_negative")
        self.assertEqual(verification["sides"]["challenger"]["result"], "unscorable")

    def test_daytime_repair_keeps_baseline_and_explicit_context(self):
        state = ShadowState()
        for round_no in range(1, 131):
            self.observe(state, round_no, our_walls=[wall(11, 80)])
        self.observe(state, 131, our_walls=[wall(11, 80)])
        day = self.observe(state, 132, our_hp=110,
                           our_walls=[wall(11, 100, 2)])
        side = day["prediction"]["sides"]["challenger"]
        self.assertEqual(side["persistenceBaseline"]["risk"], "low")
        self.assertEqual(side["assessment"]["risk"], "unknown")
        self.assertEqual(side["contextChanges"]["base"]["hpIncrease"], 10)
        self.assertEqual(side["contextChanges"]["walls"]["upgraded"], 1)
        for round_no in range(133, 206):
            hp = 100 if round_no >= 205 else 110
            result = self.observe(state, round_no, our_hp=hp,
                                  our_walls=[wall(11, 100, 2)])
        self.assertEqual(result["verification"]["sides"]["challenger"]["baselineResult"], "false_negative")

    def test_enemy_base_missing_and_robot_id_limit_do_not_erase_our_base_evidence(self):
        state = ShadowState()
        for round_no in range(1, 131):
            value = payload(round_no)
            value["teamEnemy"]["roles"] = []
            observe_shadow(Turn.load(value), state)
        next_day = payload(131)
        next_day["teamEnemy"]["roles"] = []
        observe_shadow(Turn.load(next_day), state)
        self.assertEqual(shadow_diagnostic(state)["prediction"]["sides"]["challenger"]["persistenceBaseline"]["risk"], "low")
        for round_no in range(132, 202):
            value = payload(round_no)
            value["teamEnemy"]["roles"] = []
            observe_shadow(Turn.load(value), state)
        many = [robot(i, "challenger") for i in range(600)]
        damaged = payload(202, our_hp=90, robots=many)
        damaged["teamEnemy"]["roles"] = []
        observe_shadow(Turn.load(damaged), state)
        result = shadow_diagnostic(state)
        self.assertTrue(result["verification"]["sides"]["challenger"]["actualBaseDamage"])
        self.assertEqual(result["verification"]["sides"]["challenger"]["baselineResult"], "false_negative")

    def test_night_summary_accumulates_losses_despite_healing(self):
        state = ShadowState()
        self.observe(state, 70)
        for round_no in range(71, 131):
            hp = 90 if round_no == 72 else 100 if round_no < 80 else 80
            robots = [robot(1, "challenger", health=40 if round_no >= 73 else 50)]
            self.observe(state, round_no, our_hp=hp, robots=robots,
                         our_walls=[wall(11, 90 if round_no >= 74 else 100)])
        final = self.observe(state, 131, our_hp=80, robots=[])
        metrics = final["recentNights"][-1]["sides"]["challenger"]["metrics"]
        self.assertEqual(metrics["base"]["firstDamageRound"], 72)
        self.assertEqual(metrics["base"]["cumulativeHpLoss"], 30)
        self.assertEqual(metrics["robots"]["peakCount"], 1)
        self.assertEqual(metrics["robots"]["cumulativeObservedHpLoss"], 10)
        self.assertEqual(metrics["walls"]["damaged"], 1)

    def test_current_pressure_distinguishes_damage_near_and_unknown(self):
        state = ShadowState()
        self.observe(state, 70)
        near = self.observe(state, 71, robots=[robot(1, "challenger", x=3, y=3)])
        self.assertEqual(near["currentPressure"]["sides"]["challenger"]["classification"], "threats_near_base")
        self.assertEqual(near["nightSummary"]["observedRange"]["lastRound"], 71)
        self.assertEqual(near["nightSummary"]["sides"]["challenger"]["robots"]["peakCount"], 1)
        damaged = self.observe(state, 72, our_hp=90)
        self.assertEqual(damaged["currentPressure"]["sides"]["challenger"]["classification"], "base_damage_observed")
        unknown = ShadowState()
        self.observe(unknown, 70)
        missing = self.observe(unknown, 71, robots_observed=False)
        self.assertEqual(missing["currentPressure"]["sides"]["challenger"]["classification"], "unknown")
        far = self.observe(unknown, 72, robots=[robot(2, "challenger", x=15, y=15)])
        self.assertEqual(far["currentPressure"]["sides"]["challenger"]["classification"], "no_near_threat_observed")

    def test_observed_positive_survives_gap_and_incomplete_final(self):
        state = ShadowState()
        for round_no in range(1, 81):
            self.observe(state, round_no, our_hp=90 if round_no >= 80 else 100)
        self.assertEqual(shadow_diagnostic(state)["verification"]["status"], "event_observed")
        self.observe(state, 82, our_hp=90)
        self.assertTrue(shadow_diagnostic(state)["verification"]["actualBaseDamage"])
        missing_base = payload(83)
        missing_base["teamOur"]["roles"] = []
        observe_shadow(Turn.load(missing_base), state)
        self.assertEqual(shadow_diagnostic(state)["currentPressure"]["sides"]["challenger"]["classification"], "base_damage_observed")
        final = self.observe(state, 131, our_hp=90)
        self.assertFalse(final["verification"]["complete"])
        self.assertTrue(final["verification"]["actualBaseDamage"])
        self.assertEqual(len(final["recentNights"]), 1)

    def test_robot_history_limit_does_not_erase_complete_base_negative(self):
        state = ShadowState()
        many = [robot(i, "challenger") for i in range(600)]
        self.observe(state, 70)
        for round_no in range(71, 131):
            self.observe(state, round_no, robots=many)
        final = self.observe(state, 131)
        ours = final["verification"]["sides"]["challenger"]
        self.assertIs(ours["actualBaseDamage"], False)
        self.assertFalse(ours["metrics"]["completeness"]["robots"])
        self.assertTrue(ours["metrics"]["completeness"]["base"])

    def test_missing_base_never_means_destroyed(self):
        state = ShadowState()
        self.observe(state, 70)
        for round_no in range(71, 131):
            value = payload(round_no)
            if round_no == 80:
                value["teamOur"]["roles"] = []
            observe_shadow(Turn.load(value), state)
        final = self.observe(state, 131)
        self.assertIsNone(final["verification"]["actualBaseDamage"])
        self.assertFalse(final["verification"]["sides"]["challenger"]["metrics"]["completeness"]["base"])

    def test_summary_tracks_threat_range_damage_streak_and_wall_events(self):
        state = ShadowState()
        self.observe(state, 70, our_walls=[wall(11, 100)])
        for round_no in range(71, 131):
            hp = (90 if round_no == 72 else 80 if round_no in (73, 74)
                  else 70 if round_no >= 75 else 100)
            robots = ([robot(1, "challenger", 50, x=3, y=3)] if round_no == 72
                      else [robot(1, "challenger", 30, x=3, y=3)] if round_no == 73
                      else [robot(2, "challenger", 100, x=15, y=15)] if round_no == 76
                      else [])
            walls = ([wall(11, 100)] if round_no == 71
                     else [wall(11, 90)] if round_no == 72
                     else [wall(11, 100)] if round_no == 73
                     else [] if round_no == 75
                     else [wall(11, 100, 2)])
            self.observe(state, round_no, our_hp=hp, robots=robots,
                         our_walls=walls)
        final = self.observe(state, 131, our_hp=70,
                             our_walls=[wall(11, 100, 2)])
        metrics = final["recentNights"][-1]["sides"]["challenger"]["metrics"]
        self.assertEqual(metrics["base"]["cumulativeHpLoss"], 30)
        self.assertEqual(metrics["base"]["longestConsecutiveDamageRounds"], 2)
        self.assertEqual(metrics["robots"]["firstThreat"]["roundNo"], 72)
        self.assertEqual(metrics["robots"]["lastThreat"]["roundNo"], 76)
        self.assertEqual(metrics["robots"]["peakHp"], 100)
        self.assertEqual(metrics["robots"]["nearestBaseChebyshev"], 1)
        self.assertEqual(metrics["robots"]["cumulativeObservedHpLoss"], 20)
        self.assertEqual({name: metrics["walls"][name] for name in (
            "damaged", "repairedSameLevel", "upgraded", "disappeared", "new",
        )}, dict(damaged=1, repairedSameLevel=1, upgraded=1,
                 disappeared=1, new=1))

    def test_critical_wall_pressure_records_geometry_without_causation(self):
        state = ShadowState()
        self.observe(state, 70,
                     our_walls=[wall(11, 100, x=5, y=5),
                                wall(12, 100, x=3, y=2)])
        self.observe(state, 71,
                     robots=[robot(1, "challenger", x=4, y=5)],
                     our_walls=[wall(11, 100, x=5, y=5),
                                wall(12, 100, x=3, y=2)])
        second = self.observe(state, 72,
                              robots=[robot(1, "challenger", x=4, y=5),
                                      robot(2, "challenger", x=3, y=3)],
                              our_walls=[wall(11, 90, x=5, y=5),
                                         wall(12, 90, x=3, y=2)])
        pressure = second["current"]["sides"]["challenger"]["criticalWallPressure"]
        self.assertEqual(pressure["damagedWalls"], 2)
        self.assertEqual(pressure["nearBaseDamagedWalls"], 1)
        self.assertEqual(pressure["sideDamagedWalls"], 1)
        self.assertEqual(pressure["damagedWallsWithAdjacentRobots"], 2)
        self.assertEqual(pressure["causalAttribution"], "unknown")
        third = self.observe(state, 73,
                             robots=[robot(1, "challenger", x=4, y=5)],
                             our_walls=[wall(11, 80, x=5, y=5),
                                        wall(12, 90, x=3, y=2)])
        summary = third["nightSummary"]["sides"]["challenger"]["criticalWalls"]
        self.assertEqual(summary["longestConsecutiveDamageRounds"], 2)
        self.assertEqual(summary["damagedWallFrames"], 3)
        self.assertEqual(third["current"]["sides"]["challenger"]
                         ["criticalWallBreach"], "unknown")
        defender = ShadowState()
        self.observe(defender, 71, team="defender",
                     robots=[robot(5, "defender", x=4, y=5)],
                     our_walls=[wall(21, 100)])
        mapped = self.observe(defender, 72, team="defender",
                              robots=[robot(5, "defender", x=4, y=5)],
                              our_walls=[wall(21, 90)])
        self.assertEqual(mapped["current"]["sides"]["defender"]
                         ["criticalWallPressure"]["damagedWalls"], 1)

    def test_adjacent_robot_without_wall_loss_and_gap_do_not_extend_streak(self):
        state = ShadowState()
        self.observe(state, 71, robots=[robot(1, "challenger", x=4, y=5)],
                     our_walls=[wall(11, 100)])
        unchanged = self.observe(state, 72,
                                 robots=[robot(1, "challenger", x=4, y=5)],
                                 our_walls=[wall(11, 100)])
        pressure = unchanged["current"]["sides"]["challenger"]["criticalWallPressure"]
        self.assertEqual(pressure["damagedWalls"], 0)
        self.assertEqual(pressure["adjacentRobotsToDamagedWalls"], 0)
        gap = self.observe(state, 74,
                           robots=[robot(1, "challenger", x=4, y=5)],
                           our_walls=[wall(11, 90)])
        self.assertFalse(gap["current"]["sides"]["challenger"]
                         ["criticalWallPressure"]["comparable"])
        upgraded = self.observe(state, 75,
                                robots=[robot(1, "challenger", x=4, y=5)],
                                our_walls=[wall(11, 120, 2)])
        self.assertEqual(upgraded["current"]["sides"]["challenger"]
                         ["criticalWallPressure"]["damagedWalls"], 0)

    def test_wall_pressure_marks_missing_base_robot_and_wall_limits(self):
        state = ShadowState()
        self.observe(state, 71, our_walls=[wall(11, 100)])
        missing_base = payload(72, our_walls=[wall(11, 90)],
                               robots_observed=False)
        missing_base["teamOur"]["roles"] = [wall(11, 90)]
        observe_shadow(Turn.load(missing_base), state)
        current = shadow_diagnostic(state)["current"]["sides"]["challenger"]
        self.assertIsNone(current["criticalWallPressure"]["nearBaseDamagedWalls"])
        self.assertIsNone(current["criticalWallPressure"]
                         ["adjacentRobotsToDamagedWalls"])
        summary = shadow_diagnostic(state)["nightSummary"]["sides"]
        self.assertFalse(summary["challenger"]["criticalWalls"]["coverageComplete"])
        self.assertIn("base_missing", summary["challenger"]
                      ["criticalWalls"]["unknowns"])
        self.assertIn("robots_missing", summary["challenger"]
                      ["criticalWalls"]["unknowns"])

        limited = ShadowState()
        walls = [wall(1000 + i, 100, x=5 + i % 10, y=5 + i // 10)
                 for i in range(513)]
        self.observe(limited, 71, our_walls=walls)
        result = self.observe(limited, 72, our_walls=walls)
        pressure = result["current"]["sides"]["challenger"]["criticalWallPressure"]
        self.assertFalse(pressure["wallObservationComplete"])
        self.assertIn("wall_id_limit", pressure["unknowns"])

    def test_final_and_recent_night_keep_frozen_first_night_observation(self):
        state = ShadowState()
        self.observe(state, 70)
        for round_no in range(71, 131):
            robots = ([robot(1, "challenger"), robot(2, "challenger"),
                       robot(3, "defender")] if round_no == 71 else [])
            self.observe(state, round_no, robots=robots)
        final = self.observe(state, 131)
        expected = final["verification"]["firstNightRobotDifference"]
        self.assertEqual(expected["rawDifference"], 1)
        self.assertEqual(final["recentNights"][-1]
                         ["firstNightRobotDifference"], expected)

    def test_dawn_wall_damage_is_counted_with_prior_robot_context_unknown(self):
        state = ShadowState()
        self.observe(state, 70, our_walls=[wall(11, 100)])
        for round_no in range(71, 131):
            robots = [robot(1, "challenger", x=4, y=5)] if round_no == 130 else []
            self.observe(state, round_no, robots=robots,
                         our_walls=[wall(11, 100)])
        final = self.observe(state, 131, robots=[],
                             our_walls=[wall(11, 90)])
        metrics = final["verification"]["sides"]["challenger"]["metrics"]
        self.assertEqual(metrics["walls"]["damaged"], 1)
        critical = metrics["criticalWalls"]
        self.assertEqual(critical["damagedWallFrames"], 1)
        self.assertEqual(critical["sideDamagedWallFrames"], 1)
        self.assertIsNone(critical["adjacentRobotObservations"])
        self.assertEqual(critical["dawnPriorAdjacentRobotObservations"], 1)
        self.assertIn("dawn_robot_association_unknown", critical["unknowns"])


if __name__ == "__main__":
    unittest.main()
