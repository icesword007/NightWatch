import json
import unittest
from pathlib import Path

from agent.brain import DecisionEngine


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def payload(*, round_no, team_id="pressure", team_type="challenger"):
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value["roundNo"] = round_no
    value["mapInfo"].update({"width": 20, "height": 20, "zones": []})
    value["teamOur"].update({
        "teamId": team_id,
        "type": team_type,
        "roles": [unit(10013, "station", 9, 9, health=1500)],
    })
    value["teamEnemy"].update({
        "teamId": f"enemy-{team_id}",
        "type": "defender" if team_type == "challenger" else "challenger",
        "roles": [],
    })
    value["robot"]["roles"] = []
    return value


def unit(unit_id, kind, x, y, *, health=220, level=1):
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
        "cooldown": 0,
    }


def robot(robot_id, kind, *, target="challenger"):
    return {
        "id": robot_id,
        "pos": {"x": robot_id % 10, "y": 15},
        "roleType": kind,
        "health": 40,
        "abnormalState": "",
        "targetTeam": target,
    }


def decide(engine, value):
    traces = []
    engine.decide(value, trace_sink=traces.append)
    return traces[0]["defensePressure"]


def observe_full_night(engine, day, robots):
    start = 71 + (day - 1) * 130
    for round_no in range(start, start + 60):
        value = payload(round_no=round_no)
        value["robot"]["roles"] = robots
        decide(engine, value)
    return decide(engine, payload(round_no=day * 130 + 1))


class DefensePressureTests(unittest.TestCase):
    def test_first_night_snapshot_filters_target_and_deduplicates_robot_ids(self):
        # Break caught: enemy-targeted robots or duplicate IDs inflate our wave.
        value = payload(round_no=71)
        value["robot"]["roles"] = [
            robot(30001, "smallRobot"),
            robot(30001, "smallRobot"),
            robot(30002, "middleRobot"),
            robot(30003, "largeRobot", target="defender"),
            robot(30004, "bossRobot", target=""),
        ]

        diagnostic = decide(DecisionEngine(), value)

        self.assertEqual(diagnostic["currentNight"]["counts"], {
            "smallRobot": 1, "middleRobot": 1,
        })
        self.assertEqual(
            diagnostic["currentNight"]["completeness"], "lower_bound",
        )
        self.assertEqual(diagnostic["historyCount"], 1)
        self.assertIn("unknown", diagnostic["assessment"]["reasons"])

    def test_current_night_accumulates_new_ids_and_refreshes_next_forecast(self):
        # Real-engine regression: R71 small1, then R72 first observes large1.
        engine = DecisionEngine()
        first = payload(round_no=71)
        first["robot"]["roles"] = [robot(30001, "smallRobot")]
        first_diagnostic = decide(engine, first)
        second = payload(round_no=72)
        second["robot"]["roles"] = [
            robot(30001, "smallRobot"), robot(30002, "largeRobot"),
        ]

        second_diagnostic = decide(engine, second)

        self.assertEqual(
            first_diagnostic["forecast"]["counts"], {"smallRobot": 1},
        )
        self.assertEqual(second_diagnostic["currentNight"]["counts"], {
            "smallRobot": 1, "largeRobot": 1,
        })
        self.assertEqual(second_diagnostic["forecast"]["counts"], {
            "smallRobot": 1, "largeRobot": 1,
        })
        self.assertFalse(second_diagnostic["forecast"]["finalized"])
        self.assertEqual(
            second_diagnostic["forecast"]["basisCompleteness"], "lower_bound",
        )
        self.assertEqual(len(engine.state.state.wave_forecasts), 1)

    def test_actual_score_updates_without_rewriting_original_prediction(self):
        engine = DecisionEngine()
        for round_no in range(71, 131):
            first = payload(round_no=round_no)
            first["robot"]["roles"] = [robot(30001, "smallRobot")]
            decide(engine, first)
        dawn = decide(engine, payload(round_no=131))
        original = dawn["forecast"]
        self.assertEqual(original["counts"], {"smallRobot": 1})
        self.assertTrue(original["finalized"])

        second = payload(round_no=201)
        second["robot"]["roles"] = [robot(40001, "smallRobot")]
        decide(engine, second)
        later = payload(round_no=202)
        later["robot"]["roles"] = [
            robot(40001, "smallRobot"), robot(40002, "largeRobot"),
        ]

        diagnostic = decide(engine, later)
        scored = diagnostic["forecastHistory"][-1]

        self.assertEqual(scored["counts"], original["counts"])
        self.assertEqual(scored["actualCounts"], {
            "smallRobot": 1, "largeRobot": 1,
        })
        self.assertEqual(scored["actualCompleteness"], "lower_bound")
        self.assertEqual(scored["absoluteError"], {
            "largeRobot": 1, "smallRobot": 0,
        })

    def test_lower_bound_actual_does_not_score_an_unseen_type_as_zero(self):
        engine = DecisionEngine()
        observe_full_night(engine, 1, [robot(30001, "smallRobot")])
        second = payload(round_no=201)
        second["robot"]["roles"] = [robot(40001, "largeRobot")]

        diagnostic = decide(engine, second)
        scored = diagnostic["forecastHistory"][-1]

        self.assertEqual(scored["actualCounts"], {"largeRobot": 1})
        self.assertEqual(scored["actualCompleteness"], "lower_bound")
        self.assertEqual(scored["absoluteError"], {"largeRobot": 1})

    def test_dawn_finalizes_only_continuously_observed_untruncated_night(self):
        complete_engine = DecisionEngine()
        for round_no in range(71, 131):
            value = payload(round_no=round_no)
            value["robot"]["roles"] = [robot(30001, "smallRobot")]
            decide(complete_engine, value)
        complete = decide(complete_engine, payload(round_no=131))
        self.assertEqual(
            complete["previousNightLoss"]["completeness"], "complete",
        )
        self.assertTrue(complete["forecast"]["finalized"])
        self.assertEqual(complete["forecast"]["basisCompleteness"], "complete")

        skipped_engine = DecisionEngine()
        for round_no in (71, 72, 74, 130):
            value = payload(round_no=round_no)
            value["robot"]["roles"] = [robot(30001, "smallRobot")]
            decide(skipped_engine, value)
        partial = decide(skipped_engine, payload(round_no=131))
        self.assertEqual(
            partial["previousNightLoss"]["completeness"], "lower_bound",
        )
        self.assertTrue(partial["forecast"]["finalized"])
        self.assertEqual(
            partial["forecast"]["basisCompleteness"], "lower_bound",
        )

    def test_next_night_callback_finalizes_a_missed_dawn_as_lower_bound(self):
        engine = DecisionEngine()
        first = payload(round_no=71)
        first["robot"]["roles"] = [robot(30001, "smallRobot")]
        decide(engine, first)
        second = payload(round_no=201)
        second["robot"]["roles"] = [robot(40001, "smallRobot")]

        decide(engine, second)
        prior = engine.state.state.wave_history[0]
        prior_forecast = engine.state.state.wave_forecasts[0]

        self.assertTrue(prior.finalized)
        self.assertFalse(prior.complete)
        self.assertTrue(prior_forecast.finalized)
        self.assertEqual(prior_forecast.basis_completeness, "lower_bound")

    def test_late_observation_is_lower_bound_and_keeps_seen_ids_after_kills(self):
        # Break caught: a later smaller live set is mistaken for a smaller spawn.
        engine = DecisionEngine()
        late = payload(round_no=72)
        late["robot"]["roles"] = [robot(30001, "smallRobot")]
        decide(engine, late)
        later = json.loads(json.dumps(late))
        later["roundNo"] = 73
        later["robot"]["roles"] = [
            robot(30002, "smallRobot"), robot(30003, "middleRobot"),
        ]

        diagnostic = decide(engine, later)

        self.assertEqual(diagnostic["currentNight"]["counts"], {
            "smallRobot": 2, "middleRobot": 1,
        })
        self.assertEqual(
            diagnostic["currentNight"]["completeness"], "lower_bound",
        )
        self.assertEqual(diagnostic["currentNight"]["firstObservedRound"], 72)
        self.assertEqual(diagnostic["currentNight"]["lastObservedRound"], 73)

    def test_adjacent_complete_nights_predict_latest_nonnegative_type_growth(self):
        # Break caught: forecast invents a multiplier or drops newly seen types.
        engine = DecisionEngine()
        first_robots = [
            robot(30001, "smallRobot"), robot(30002, "smallRobot"),
            robot(30003, "middleRobot"),
        ]
        observe_full_night(engine, 1, first_robots)
        second_robots = [
            robot(30001, "smallRobot"),
            robot(30002, "smallRobot"),
            robot(30003, "smallRobot"),
            robot(30004, "middleRobot"),
            robot(30005, "largeRobot"),
        ]

        diagnostic = observe_full_night(engine, 2, second_robots)

        self.assertEqual(diagnostic["forecast"], {
            "targetDay": 3,
            "basisDays": [1, 2],
            "sampleCount": 2,
            "method": "latest_nonnegative_adjacent_delta",
            "counts": {
                "smallRobot": 4,
                "middleRobot": 1,
                "largeRobot": 2,
            },
            "growth": {
                "smallRobot": 1,
                "middleRobot": 0,
                "largeRobot": 1,
            },
            "uncertainTypes": [],
            "basisCompleteness": "complete",
            "finalized": True,
            "actualCounts": None,
            "absoluteError": None,
            "actualCompleteness": None,
            "actualFinalized": False,
        })
        self.assertIn("rising_pressure", diagnostic["assessment"]["reasons"])

    def test_negative_growth_does_not_predict_a_safer_next_night(self):
        # Break caught: a smaller second wave is extrapolated downward.
        engine = DecisionEngine()
        first_robots = [
            robot(30001, "smallRobot"), robot(30002, "smallRobot"),
        ]
        observe_full_night(engine, 1, first_robots)

        diagnostic = observe_full_night(
            engine, 2, [robot(30001, "smallRobot")],
        )

        self.assertEqual(diagnostic["forecast"]["counts"], {"smallRobot": 2})
        self.assertEqual(diagnostic["forecast"]["growth"], {"smallRobot": 0})
        self.assertEqual(
            diagnostic["forecast"]["method"],
            "adjacent_delta_with_decline_unknown",
        )
        self.assertEqual(
            diagnostic["forecast"]["uncertainTypes"], ["smallRobot"],
        )
        self.assertNotIn("rising_pressure", diagnostic["assessment"]["reasons"])
        self.assertIn("unknown", diagnostic["assessment"]["reasons"])

    def test_missing_night_does_not_create_per_day_growth(self):
        # Break caught: non-adjacent complete samples are treated as consecutive.
        engine = DecisionEngine()
        observe_full_night(engine, 1, [robot(30001, "smallRobot")])
        third_robots = [
            robot(30001, "smallRobot"), robot(30002, "smallRobot"),
        ]

        diagnostic = observe_full_night(engine, 3, third_robots)

        self.assertEqual(diagnostic["forecast"]["targetDay"], 4)
        self.assertEqual(diagnostic["forecast"]["basisDays"], [3])
        self.assertEqual(diagnostic["forecast"]["method"], "growth_unknown")
        self.assertIn("unknown", diagnostic["assessment"]["reasons"])

    def test_forecast_is_preserved_and_scored_when_target_night_arrives(self):
        # Break caught: actual night data overwrites the old prediction.
        engine = DecisionEngine()
        observe_full_night(engine, 1, [robot(30001, "smallRobot")])
        second_robots = [
            robot(30001, "smallRobot"), robot(30002, "smallRobot"),
        ]
        predicted = observe_full_night(engine, 2, second_robots)["forecast"]
        third = payload(round_no=331)
        third["robot"]["roles"] = [
            robot(30001, "smallRobot"),
            robot(30002, "smallRobot"),
            robot(30003, "smallRobot"),
            robot(30004, "smallRobot"),
        ]

        diagnostic = decide(engine, third)
        prior = diagnostic["forecastHistory"][-1]

        self.assertEqual(predicted["counts"], {"smallRobot": 3})
        self.assertEqual(prior["counts"], {"smallRobot": 3})
        self.assertEqual(prior["actualCounts"], {"smallRobot": 4})
        self.assertEqual(prior["absoluteError"], {"smallRobot": 1})

    def test_session_reset_drops_wave_history_and_history_is_bounded(self):
        # Break caught: old match pressure leaks into a new team session or grows forever.
        engine = DecisionEngine()
        for day in range(1, 12):
            value = payload(round_no=71 + (day - 1) * 130)
            value["robot"]["roles"] = [robot(30000 + day, "smallRobot")]
            diagnostic = decide(engine, value)
        self.assertEqual(diagnostic["historyCount"], 10)
        self.assertTrue(diagnostic["historyTruncated"])

        reset = payload(round_no=71, team_id="pressure-new-session")
        reset["robot"]["roles"] = [robot(40001, "bossRobot")]
        diagnostic = decide(engine, reset)

        self.assertEqual(diagnostic["historyCount"], 1)
        self.assertFalse(diagnostic["historyTruncated"])
        self.assertEqual(diagnostic["currentNight"]["counts"], {"bossRobot": 1})

    def test_dawn_records_previous_night_defense_losses(self):
        # Break caught: pressure history counts enemies but cannot explain our losses.
        engine = DecisionEngine()
        for round_no in range(71, 131):
            night = payload(round_no=round_no)
            night["teamOur"]["roles"].extend([
                unit(10010, "worker", 7, 8, health=220),
                unit(10020, "gatling", 8, 8, health=1000, level=2),
                unit(10050, "wall", 12, 8, health=1000),
            ])
            night["robot"]["roles"] = [robot(30001, "smallRobot")]
            decide(engine, night)
        dawn = payload(round_no=131)
        dawn["teamOur"]["roles"].extend([
            unit(10010, "worker", 7, 8, health=100),
            unit(10020, "gatling", 8, 8, health=800, level=2),
        ])

        diagnostic = decide(engine, dawn)

        self.assertEqual(diagnostic["previousNightLoss"], {
            "day": 1,
            "completeness": "complete",
            "stationHealth": 0,
            "roleHealth": 120,
            "weaponHealth": 200,
            "weaponsLost": 0,
            "wallsLost": 1,
        })
        self.assertIn("known_deficit", diagnostic["assessment"]["reasons"])

        later = json.loads(json.dumps(dawn))
        later["roundNo"] = 140
        later["teamOur"]["roles"] = [
            unit(10013, "station", 9, 9, health=1000),
        ]
        later_diagnostic = decide(engine, later)
        self.assertEqual(
            later_diagnostic["previousNightLoss"],
            diagnostic["previousNightLoss"],
        )

    def test_daytime_uses_the_stored_forecast_for_the_coming_night(self):
        # Break caught: day 2 drops the target-day-2 forecast until night begins.
        engine = DecisionEngine()
        diagnostic = observe_full_night(
            engine, 1, [robot(30001, "smallRobot")],
        )

        self.assertEqual(diagnostic["forecast"]["targetDay"], 2)
        self.assertEqual(diagnostic["forecast"]["basisDays"], [1])
        self.assertEqual(diagnostic["forecast"]["method"], "growth_unknown")

    def test_partial_current_night_does_not_reuse_an_old_adjacent_growth_pair(self):
        # Break caught: a missed night is hidden by extrapolating days 1 and 2 to day 5.
        engine = DecisionEngine()
        first = payload(round_no=71)
        first["robot"]["roles"] = [robot(30001, "smallRobot")]
        decide(engine, first)
        second = payload(round_no=201)
        second["robot"]["roles"] = [
            robot(30001, "smallRobot"), robot(30002, "smallRobot"),
        ]
        decide(engine, second)
        fourth_late = payload(round_no=462)
        fourth_late["robot"]["roles"] = [
            robot(40001, "smallRobot"),
            robot(40002, "smallRobot"),
            robot(40003, "smallRobot"),
        ]

        diagnostic = decide(engine, fourth_late)

        self.assertEqual(diagnostic["forecast"]["targetDay"], 5)
        self.assertEqual(diagnostic["forecast"]["method"], "growth_unknown")
        self.assertEqual(diagnostic["forecast"]["basisDays"], [4])
        self.assertIn("unknown", diagnostic["assessment"]["reasons"])

    def test_three_towers_with_one_controller_is_a_known_deficit(self):
        # Break caught: any nonzero role count is called adequate for three towers.
        value = payload(round_no=71)
        value["teamOur"]["roles"].extend([
            unit(10010, "worker", 7, 8),
            unit(10020, "gatling", 8, 8, health=1000),
            unit(10030, "railgun", 9, 7, health=1000),
            unit(10040, "rocket", 10, 7, health=1000),
        ])

        diagnostic = decide(DecisionEngine(), value)

        self.assertIn("known_deficit", diagnostic["assessment"]["reasons"])

    def test_previous_night_damage_remains_a_deficit_after_positions_recover(self):
        # Break caught: current counts erase a real base loss from the assessment.
        engine = DecisionEngine()
        for round_no in range(71, 131):
            night = payload(round_no=round_no)
            night["teamOur"]["roles"] = [
                unit(10010, "worker", 7, 8),
                unit(10011, "worker", 7, 9),
                unit(10012, "pioneer", 7, 10),
                unit(10013, "station", 9, 9, health=1500),
                unit(10020, "gatling", 8, 8, health=1000),
                unit(10030, "railgun", 9, 7, health=1000),
                unit(10040, "rocket", 10, 7, health=1000),
            ]
            night["robot"]["roles"] = [robot(30001, "smallRobot")]
            decide(engine, night)
        engine.state.state.fortification_targets = ()
        dawn = json.loads(json.dumps(night))
        dawn["roundNo"] = 131
        dawn["robot"]["roles"] = []
        next(role for role in dawn["teamOur"]["roles"] if role["id"] == 10013)[
            "health"
        ] = 1400

        diagnostic = decide(engine, dawn)

        self.assertEqual(diagnostic["previousNightLoss"]["stationHealth"], 100)
        self.assertIn("known_deficit", diagnostic["assessment"]["reasons"])

    def test_late_day_observation_does_not_impersonate_a_dawn_snapshot(self):
        # Break caught: repairs or losses before a late callback rewrite night-end loss.
        engine = DecisionEngine()
        night = payload(round_no=71)
        night["teamOur"]["roles"].append(
            unit(10050, "wall", 12, 8, health=1000),
        )
        night["robot"]["roles"] = [robot(30001, "smallRobot")]
        decide(engine, night)
        late_day = payload(round_no=140)

        diagnostic = decide(engine, late_day)

        self.assertEqual(
            diagnostic["previousNightLoss"]["completeness"], "lower_bound",
        )
        self.assertEqual(diagnostic["previousNightLoss"]["wallsLost"], 0)

    def test_late_night_start_keeps_loss_lower_bound_even_at_exact_dawn(self):
        # Break caught: an exact dawn callback upgrades a partial night to complete.
        engine = DecisionEngine()
        late_night = payload(round_no=110)
        next(
            role for role in late_night["teamOur"]["roles"]
            if role["id"] == 10013
        )["health"] = 500
        late_night["robot"]["roles"] = [robot(30001, "smallRobot")]
        decide(engine, late_night)
        dawn = payload(round_no=131)
        next(
            role for role in dawn["teamOur"]["roles"]
            if role["id"] == 10013
        )["health"] = 400

        diagnostic = decide(engine, dawn)

        self.assertEqual(
            diagnostic["previousNightLoss"]["completeness"], "lower_bound",
        )
        self.assertEqual(diagnostic["previousNightLoss"]["stationHealth"], 100)

    def test_defender_team_uses_its_own_targeted_wave(self):
        # Break caught: pressure filtering is accidentally hard-coded to challenger.
        value = payload(round_no=71, team_type="defender")
        value["robot"]["roles"] = [
            robot(30001, "smallRobot", target="challenger"),
            robot(30002, "bossRobot", target="defender"),
        ]

        diagnostic = decide(DecisionEngine(), value)

        self.assertEqual(diagnostic["currentNight"]["counts"], {"bossRobot": 1})

    def test_robot_id_storage_cap_marks_wave_as_lower_bound(self):
        # Break caught: visible robot IDs create unbounded per-night state.
        engine = DecisionEngine()
        value = payload(round_no=71)
        value["robot"]["roles"] = [
            robot(40000 + index, "smallRobot") for index in range(520)
        ]

        diagnostic = decide(engine, value)

        self.assertEqual(diagnostic["currentNight"]["robotIdsRetained"], 512)
        self.assertTrue(diagnostic["currentNight"]["truncated"])
        self.assertEqual(
            diagnostic["currentNight"]["completeness"], "lower_bound",
        )

        for round_no in range(72, 131):
            later = payload(round_no=round_no)
            later["robot"]["roles"] = value["robot"]["roles"]
            decide(engine, later)
        dawn = decide(engine, payload(round_no=131))
        self.assertEqual(dawn["forecast"]["basisCompleteness"], "lower_bound")


if __name__ == "__main__":
    unittest.main()
