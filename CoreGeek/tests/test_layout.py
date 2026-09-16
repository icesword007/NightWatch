import copy
import json
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import agent.defense as defense_module
from agent.brain import DecisionEngine
from agent.economy import propose_economy
from agent.protocol import Pos, Turn
from agent.state import (
    CompletedAction,
    PendingAction,
    StateStore,
    request_fingerprint,
)


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def unit(unit_id, kind, x, y, *, health=1000, level=1, cooldown=0):
    return {
        "id": unit_id,
        "pos": {"x": x, "y": y},
        "roleType": kind,
        "health": health,
        "attackPower": 20 if kind == "rocket" else 10,
        "attackRange": 0,
        "backPackCapability": 100 if kind == "worker" else 40,
        "backpack": [],
        "level": level,
        "cooldown": cooldown,
    }


def robot(robot_id, x, y, health, *, target="challenger", kind="smallRobot"):
    return {
        "id": robot_id,
        "pos": {"x": x, "y": y},
        "roleType": kind,
        "health": health,
        "abnormalState": "",
        "targetTeam": target,
    }


def payload(*, our_x=9, enemy_x=17, round_no=5, workers=()):
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value["roundNo"] = round_no
    value["mapInfo"].update({"width": 24, "height": 20, "zones": []})
    value["teamOur"].update({
        "teamId": "layout-tests",
        "goldNum": 0,
        "roles": [
            *(unit(role_id, "worker", x, y) for role_id, x, y in workers),
            unit(10013, "station", our_x, 9, health=1500),
        ],
        "playerTasks": [],
    })
    value["teamEnemy"]["roles"] = [
        unit(20013, "station", enemy_x, 9, health=1500),
    ]
    value["robot"]["roles"] = []
    value["phaseTask"] = ""
    value["llmResp"] = ""
    value["lastCmdResult"] = ""
    value["errors"] = []
    value["weaponShopList"] = []
    value["vendorShopList"] = []
    return value


def state_for(value):
    store = StateStore()
    turn = Turn.load(value)
    store.observe(turn, value, request_fingerprint(value))
    return store.state


class LayoutTests(unittest.TestCase):
    def test_mirrored_layout_uses_rear_towers_three_sided_wall_and_open_back(self):
        # Break caught: tower targets follow workers and the six-wall prefix leaves
        # side gaps instead of a mirrored stable three-sided enclosure.
        from agent.layout import plan_defense_layout

        left = plan_defense_layout(Turn.load(payload(our_x=3, enemy_x=19)))
        right = plan_defense_layout(Turn.load(payload(our_x=19, enemy_x=3)))

        self.assertTrue(left.complete)
        self.assertTrue(right.complete)
        self.assertEqual(len(left.tower_targets), 3)
        self.assertEqual(len(set(left.gunner_stands)), 3)
        self.assertEqual(len(left.wall_targets), 14)
        self.assertTrue(all(pos.x < 3 for pos in left.tower_targets))
        self.assertTrue(all(pos.x > 20 for pos in right.tower_targets))
        self.assertEqual(
            {(23 - pos.x, pos.y) for pos in left.wall_targets},
            {(pos.x, pos.y) for pos in right.wall_targets},
        )
        self.assertTrue(set(left.gunner_stands).isdisjoint(left.wall_targets))
        self.assertTrue(set(left.exit_cells).isdisjoint(left.wall_targets))

    def test_worker_positions_do_not_change_layout(self):
        # Break caught: nearest-worker selection becomes the blueprint itself.
        from agent.layout import plan_defense_layout

        first = Turn.load(payload(workers=((10010, 0, 0), (10012, 23, 19))))
        swapped = Turn.load(payload(workers=((10010, 23, 19), (10012, 0, 0))))

        self.assertEqual(plan_defense_layout(first), plan_defense_layout(swapped))

    def test_static_obstacle_reports_incomplete_layout_without_overlap(self):
        # Break caught: a blocked future tower cell is silently accepted as complete.
        from agent.layout import plan_defense_layout

        value = payload()
        value["mapInfo"]["zones"] = [
            {"pos": {"x": 8, "y": 8}, "neutralType": "stone"},
        ]
        plan = plan_defense_layout(Turn.load(value))

        self.assertFalse(plan.complete)
        self.assertIsNotNone(plan.degraded_reason)
        self.assertNotIn(Pos(8, 8), plan.tower_targets)

    def test_two_workers_can_reserve_distinct_same_round_rockets(self):
        # Break caught: a set of claimed tower types suppresses the second rocket.
        from agent.layout import plan_defense_layout

        initial = Turn.load(payload())
        plan = plan_defense_layout(initial)
        value = payload(workers=(
            (10010, plan.gunner_stands[0].x, plan.gunner_stands[0].y),
            (10012, plan.gunner_stands[1].x, plan.gunner_stands[1].y),
        ))
        value["teamOur"]["goldNum"] = 50
        turn = Turn.load(value)
        candidates = propose_economy(
            turn, state_for(value), clock=lambda: 0.0,
            deadline=1.0, max_expansions=128,
        )
        builds = [
            item for item in candidates
            if item.proposal.command.get("action") == "build"
            and item.proposal.command.get("name") == "rocket"
        ]

        self.assertEqual(len(builds), 2)
        self.assertEqual(len({item.plan_target for item in builds}), 2)

    def test_layout_is_stable_after_first_engine_turn(self):
        # Break caught: changing live robot position recomputes and flips construction.
        first = payload(workers=((10010, 2, 8), (10012, 2, 9)))
        engine = DecisionEngine()
        engine.decide(first)
        targets = engine.state.state.layout_tower_targets

        changed = copy.deepcopy(first)
        changed["roundNo"] = 71
        changed["robot"]["roles"] = [robot(30001, 0, 9, 40)]
        engine.decide(changed)

        self.assertEqual(engine.state.state.layout_tower_targets, targets)
        self.assertEqual(engine.state.state.layout_observed_deviation, "opposite")

    def test_gunner_mapping_and_threat_direction_use_persisted_layout(self):
        from agent.layout import ensure_defense_layout, preferred_gunner_stand

        first = payload()
        state = state_for(first)
        original = ensure_defense_layout(Turn.load(first), state)
        changed = payload(enemy_x=1, round_no=71)
        changed["mapInfo"]["zones"] = [{
            "pos": original.gunner_stands[0].dump(),
            "neutralType": "stone",
        }]
        changed["robot"]["roles"] = [robot(30001, 5, 9, 100)]
        changed_turn = Turn.load(changed)

        self.assertEqual(
            preferred_gunner_stand(
                changed_turn, original.tower_targets[1], state,
            ),
            original.gunner_stands[1],
        )
        self.assertEqual(
            defense_module._rocket_threat_layer(
                changed_turn, changed_turn.robots[0], state=state,
            )[1],
            1,
        )
        self.assertEqual(
            defense_module._rocket_threat_layer(
                changed_turn, changed_turn.robots[0], state=None,
            )[1],
            0,
        )

    def test_wall_order_grows_one_connected_front_then_both_flanks(self):
        from agent.layout import plan_defense_layout

        plan = plan_defense_layout(Turn.load(payload()))
        connected = {plan.wall_targets[0]}
        for target in plan.wall_targets[1:]:
            self.assertTrue(any(
                max(abs(target.x - prior.x), abs(target.y - prior.y)) == 1
                for prior in connected
            ))
            connected.add(target)

    def test_existing_off_blueprint_weapon_is_preserved_and_marks_degraded(self):
        from agent.layout import plan_defense_layout

        value = payload(workers=((10010, 7, 8), (10012, 7, 9)))
        plan = plan_defense_layout(Turn.load(value))
        value["teamOur"]["goldNum"] = 50
        value["teamOur"]["roles"].extend((
            unit(10020, "gatling", 4, 4),
            unit(10030, "rocket", plan.tower_targets[0].x, plan.tower_targets[0].y),
        ))
        turn = Turn.load(value)
        state = state_for(value)
        candidates = propose_economy(
            turn, state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=128,
        )
        builds = [
            item for item in candidates
            if item.proposal.command.get("action") == "build"
        ]

        self.assertFalse(state.layout_complete)
        self.assertEqual(
            state.layout_degraded_reason, "existing_weapon_outside_layout",
        )
        self.assertEqual(len(builds), 1)
        self.assertEqual(builds[0].proposal.command["name"], "rocket")
        self.assertIn(builds[0].plan_target, state.layout_tower_targets[1:])

    def test_existing_one_or_two_rockets_fill_only_remaining_slots(self):
        from agent.layout import plan_defense_layout

        for existing_count, expected_builds in ((1, 2), (2, 1)):
            with self.subTest(existing_count=existing_count):
                initial = payload()
                plan = plan_defense_layout(Turn.load(initial))
                value = payload(workers=(
                    (10010, plan.gunner_stands[0].x, plan.gunner_stands[0].y),
                    (10012, plan.gunner_stands[1].x, plan.gunner_stands[1].y),
                ))
                value["teamOur"]["goldNum"] = 75
                value["teamOur"]["roles"].extend(
                    unit(10020 + index, "rocket", target.x, target.y)
                    for index, target in enumerate(
                        plan.tower_targets[:existing_count]
                    )
                )
                candidates = propose_economy(
                    Turn.load(value), state_for(value), clock=lambda: 0.0,
                    deadline=1.0, max_expansions=128,
                )
                builds = [
                    item for item in candidates
                    if item.proposal.command.get("action") == "build"
                    and item.proposal.command.get("name") == "rocket"
                ]
                self.assertEqual(len(builds), expected_builds)
                self.assertLessEqual(existing_count + len(builds), 3)

    def test_two_towers_without_carried_stone_keep_third_tower_priority(self):
        from agent.layout import plan_defense_layout

        initial = payload()
        plan = plan_defense_layout(Turn.load(initial))
        value = payload(workers=(
            (10010, plan.gunner_stands[0].x, plan.gunner_stands[0].y),
            (10012, plan.gunner_stands[1].x, plan.gunner_stands[1].y),
        ))
        value["teamOur"]["goldNum"] = 25
        value["teamOur"]["roles"].extend(
            unit(10020 + index, "rocket", target.x, target.y)
            for index, target in enumerate(plan.tower_targets[:2])
        )

        response = DecisionEngine().decide(value)

        builds = list(response["roleCommandMap"].values())
        self.assertTrue(any(
            command.get("action") == "build"
            and command.get("name") == "rocket"
            for command in builds
        ))
        self.assertFalse(any(command.get("name") == "wall" for command in builds))

    def test_unknown_build_feedback_does_not_repeat_same_target(self):
        from agent.layout import plan_defense_layout

        initial = payload()
        plan = plan_defense_layout(Turn.load(initial))
        value = payload(workers=(
            (10010, plan.gunner_stands[0].x, plan.gunner_stands[0].y),
            (10012, plan.gunner_stands[1].x, plan.gunner_stands[1].y),
        ))
        value["teamOur"]["goldNum"] = 50
        state = state_for(value)
        unknown_target = plan.tower_targets[0]
        state.action_history.append(CompletedAction(
            PendingAction(
                4, 10010, 10010, "build", unknown_target,
                source_session=state.session_index, name="rocket",
            ),
            None,
        ))

        candidates = propose_economy(
            Turn.load(value), state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=128,
        )
        build_targets = {
            item.plan_target for item in candidates
            if item.proposal.command.get("action") == "build"
        }

        self.assertNotIn(unknown_target, build_targets)
        self.assertEqual(len(build_targets), 2)

    def test_temporary_role_on_last_tower_target_does_not_trigger_fallback_build(self):
        from agent.layout import plan_defense_layout

        initial = payload()
        plan = plan_defense_layout(Turn.load(initial))
        value = payload(workers=((10010, 12, 8), (10012, 12, 9)))
        value["teamOur"]["goldNum"] = 75
        value["teamOur"]["roles"].extend((
            unit(10020, "rocket", plan.tower_targets[0].x, plan.tower_targets[0].y),
            unit(10030, "rocket", plan.tower_targets[1].x, plan.tower_targets[1].y),
            unit(10011, "pioneer", plan.tower_targets[2].x, plan.tower_targets[2].y),
        ))
        state = state_for(value)

        candidates = propose_economy(
            Turn.load(value), state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=128,
        )
        rocket_builds = [
            item for item in candidates
            if item.proposal.command.get("action") == "build"
            and item.proposal.command.get("name") == "rocket"
        ]

        self.assertEqual(state.layout_tower_targets, plan.tower_targets)
        self.assertEqual(rocket_builds, [])

    def test_blocked_first_gunner_stand_keeps_tower_stand_pairs_aligned(self):
        from agent.layout import plan_defense_layout, preferred_gunner_stand

        value = payload()
        value["mapInfo"]["zones"] = [
            {"pos": {"x": 7, "y": 8}, "neutralType": "stone"},
        ]
        turn = Turn.load(value)
        plan = plan_defense_layout(turn)

        self.assertFalse(plan.complete)
        self.assertEqual(plan.tower_targets[0], Pos(8, 9))
        self.assertEqual(plan.gunner_stands[0], Pos(7, 9))
        self.assertEqual(
            preferred_gunner_stand(turn, plan.tower_targets[0]), Pos(7, 9),
        )


class RocketTargetingTests(unittest.TestCase):
    def _night_payload(self, *, level=3, robots=()):
        value = payload(round_no=71)
        value["teamOur"]["roles"] = [
            unit(10010, "worker", 7, 7),
            unit(10012, "worker", 7, 8),
            unit(10011, "pioneer", 7, 9),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "rocket", 8, 7, level=level),
            unit(10030, "rocket", 8, 8, level=level),
            unit(10040, "rocket", 8, 9, level=level),
        ]
        value["robot"]["roles"] = list(robots)
        return value

    def _attacks(self, value):
        response = DecisionEngine().decide(value)
        return [
            command for command in response["roleCommandMap"].values()
            if command["action"] == "attack"
        ]

    def test_rocket_prefers_cluster_yield_within_same_urgency_layer(self):
        attacks = self._attacks(self._night_payload(level=1, robots=(
            robot(30001, 14, 7, 30),
            robot(30002, 14, 8, 30),
            robot(30003, 15, 8, 30),
            robot(30004, 14, 9, 30),
            robot(30005, 14, 12, 30),
        )))

        self.assertEqual(attacks[0]["targetPos"], [{"x": 14, "y": 8}])

    def test_immediate_base_threat_beats_distant_larger_cluster(self):
        attacks = self._attacks(self._night_payload(level=1, robots=(
            robot(30001, 12, 9, 100, kind="largeRobot"),
            robot(30002, 16, 7, 30),
            robot(30003, 16, 8, 30),
            robot(30004, 17, 8, 30),
        )))

        self.assertEqual(attacks[0]["targetPos"], [{"x": 12, "y": 9}])

    def test_level_three_rocket_allocates_missiles_after_projected_damage(self):
        attacks = self._attacks(self._night_payload(level=3, robots=(
            robot(30001, 13, 6, 10),
            robot(30002, 13, 10, 10),
            robot(30003, 15, 8, 200, kind="largeRobot"),
        )))

        first = attacks[0]["targetPos"]
        self.assertEqual(len(first), 3)
        self.assertGreater(len({(pos["x"], pos["y"]) for pos in first}), 1)

    def test_three_rockets_share_projected_damage_instead_of_overkilling(self):
        attacks = self._attacks(self._night_payload(level=1, robots=(
            robot(30001, 13, 6, 10),
            robot(30002, 13, 8, 10),
            robot(30003, 13, 10, 10),
        )))

        targets = [tuple(command["targetPos"][0].values()) for command in attacks]
        self.assertEqual(len(attacks), 3)
        self.assertEqual(len(set(targets)), 3)

    def test_later_rockets_do_not_fire_at_fully_projected_dead_robot(self):
        attacks = self._attacks(self._night_payload(level=3, robots=(
            robot(30001, 13, 8, 10),
        )))

        self.assertEqual(len(attacks), 1)
        self.assertEqual(len(attacks[0]["targetPos"]), 3)
        self.assertEqual(
            attacks[0]["targetPos"], [{"x": 13, "y": 8}] * 3,
        )

    def test_level_two_rocket_fills_protocol_array_when_only_one_target_exists(self):
        attacks = self._attacks(self._night_payload(level=2, robots=(
            robot(30001, 13, 8, 10),
        )))

        self.assertEqual(len(attacks), 1)
        self.assertEqual(
            attacks[0]["targetPos"], [{"x": 13, "y": 8}] * 2,
        )

    def test_level_three_rocket_uses_all_missiles_with_two_targets(self):
        attacks = self._attacks(self._night_payload(level=3, robots=(
            robot(30001, 13, 7, 10),
            robot(30002, 13, 10, 10),
        )))

        self.assertEqual(len(attacks), 1)
        self.assertEqual(len(attacks[0]["targetPos"]), 3)
        self.assertEqual(
            len({tuple(pos.values()) for pos in attacks[0]["targetPos"]}), 2,
        )

    def test_enemy_target_cluster_does_not_distract_from_our_threat(self):
        attacks = self._attacks(self._night_payload(level=1, robots=(
            robot(30001, 12, 9, 100, target="challenger"),
            robot(30002, 14, 6, 30, target="competitor"),
            robot(30003, 14, 7, 30, target="competitor"),
            robot(30004, 14, 8, 30, target="competitor"),
        )))

        self.assertEqual(attacks[0]["targetPos"], [{"x": 12, "y": 9}])

    def test_threat_to_gunner_is_urgent_before_it_reaches_tower_or_base(self):
        value = self._night_payload(level=1, robots=(
            robot(30001, 12, 4, 100),
            robot(30002, 13, 10, 10),
        ))
        value["teamOur"]["roles"] = [
            unit(10010, "worker", 12, 7),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "rocket", 12, 8, level=1),
        ]

        attacks = self._attacks(value)

        self.assertEqual(attacks[0]["targetPos"], [{"x": 12, "y": 4}])

    def test_out_of_range_robot_is_not_selected(self):
        attacks = self._attacks(self._night_payload(level=1, robots=(
            robot(30001, 30, 19, 100),
        )))

        self.assertEqual(attacks, [])

    def test_large_robot_set_stays_bounded_on_41_by_32_map(self):
        value = self._night_payload(level=3, robots=tuple(
            robot(30000 + index, 10 + index % 9, 2 + index % 17, 10 + index % 90)
            for index in range(256)
        ))
        value["mapInfo"].update({"width": 41, "height": 32})

        started = time.perf_counter()
        attacks = self._attacks(value)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 1.0)
        self.assertLessEqual(len(attacks), 3)
        self.assertTrue(all(len(command["targetPos"]) <= 3 for command in attacks))

    def test_expired_rocket_budget_does_not_emit_partial_targets_or_damage(self):
        turn = Turn.load(self._night_payload(level=3, robots=(
            robot(30001, 13, 8, 100),
        )))
        weapon = turn.weapons()[0]
        projected = {99999: 7}

        targets = defense_module._select_rocket_targets(
            turn, weapon, 3, projected,
            clock=lambda: 2.0, deadline=1.0,
        )

        self.assertEqual(targets, ())
        self.assertEqual(projected, {99999: 7})

    def test_mid_selection_deadline_fills_valid_array_atomically(self):
        turn = Turn.load(self._night_payload(level=3, robots=(
            robot(30001, 13, 8, 100),
        )))
        weapon = turn.weapons()[0]
        projected = {}
        readings = iter((0.0, 0.0, 0.0, 2.0))

        targets = defense_module._select_rocket_targets(
            turn, weapon, 3, projected,
            clock=lambda: next(readings, 2.0), deadline=1.0,
        )

        self.assertEqual([target.robot_id for target in targets], [30001] * 3)
        self.assertEqual(projected, {30001: 60})

    def test_large_input_caps_expensive_rocket_yield_evaluations(self):
        value = self._night_payload(level=3, robots=tuple(
            robot(40000 + index, 10 + index % 9, 1 + index % 18, 100)
            for index in range(2048)
        ))
        value["mapInfo"].update({"width": 41, "height": 32})
        calls = 0
        real = defense_module._rocket_effective_damage

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return real(*args, **kwargs)

        with patch.object(
            defense_module, "_rocket_effective_damage", side_effect=counted,
        ):
            self._attacks(value)

        self.assertLessEqual(
            calls, defense_module.MAX_ROCKET_TARGET_CANDIDATES * 9,
        )


if __name__ == "__main__":
    unittest.main()
