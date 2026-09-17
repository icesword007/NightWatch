import importlib
import json
import unittest
from pathlib import Path

from agent.actions import ActionAllocator, ActionProposal, PlannedAction
from agent.brain import DecisionEngine
from agent.defense import daytime_post_assignments, daytime_work_can_return
from agent.protocol import sell_command
from agent.protocol import Pos, Turn
from agent.state import StateStore, request_fingerprint


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def unit(unit_id, kind, x, y, *, health=220, level=1, cooldown=0):
    attack = {"gatling": 10, "railgun": 10 * level, "rocket": 20}.get(kind, 0)
    return {
        "id": unit_id,
        "pos": {"x": x, "y": y},
        "roleType": kind,
        "health": health,
        "attackPower": attack,
        "attackRange": 0,
        "backPackCapability": 100 if kind == "worker" else 40,
        "backpack": [],
        "level": level,
        "cooldown": cooldown,
    }


def robot(robot_id, kind, x, y, health, *, target="challenger"):
    return {
        "id": robot_id,
        "pos": {"x": x, "y": y},
        "roleType": kind,
        "health": health,
        "abnormalState": "",
        "targetTeam": target,
    }


def defense_payload(*, round_no=71):
    payload = load_fixture()
    payload["roundNo"] = round_no
    payload["mapInfo"].update({"width": 20, "height": 20, "zones": []})
    payload["teamOur"].update({
        "teamId": "defense-tests",
        "roles": [
            unit(10010, "worker", 6, 5),
            unit(10012, "worker", 9, 5),
            unit(10011, "pioneer", 12, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000, level=2),
            unit(10030, "railgun", 9, 6, health=1000, level=3),
            unit(10040, "rocket", 12, 6, health=1000, level=2, cooldown=2),
        ],
    })
    payload["teamEnemy"]["roles"] = []
    payload["robot"]["roles"] = [
        robot(30001, "smallRobot", 6, 9, 10),
        robot(30002, "middleRobot", 9, 10, 60),
        robot(30003, "largeRobot", 12, 10, 500),
    ]
    return payload


def state_for(payload):
    store = StateStore()
    turn = Turn.load(payload)
    store.observe(turn, payload, request_fingerprint(payload))
    return store.state


def proposals(payload):
    defense = importlib.import_module("agent.defense")
    turn = Turn.load(payload)
    return defense.propose_defense(
        turn,
        state_for(payload),
        clock=lambda: 0.0,
        deadline=1.0,
        max_expansions=64,
    )


class DefenseTests(unittest.TestCase):
    def test_mixed_inventory_requires_every_sale_before_return(self):
        payload = defense_payload(round_no=67)
        worker = unit(10010, "worker", 5, 5)
        worker["backpack"] = ["stone", "copper"]
        payload["teamOur"]["roles"] = [
            worker,
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 7, 5, health=1000),
        ]
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 5, "y": 6}, "neutralType": "vendor"},
        ]
        payload["vendorShopList"] = [
            {"name": "stone", "price": 1},
            {"name": "copper", "price": 10},
        ]
        payload["robot"]["roles"] = []
        turn = Turn.load(payload)
        parsed_worker = turn.unit(10010)
        candidate = PlannedAction(ActionProposal(
            10010, 10010, sell_command("copper", 1),
        ))
        assignments = {
            10020: (parsed_worker, (Pos(6, 5), Pos(6, 5), 1)),
        }

        self.assertFalse(daytime_work_can_return(
            turn,
            parsed_worker,
            candidate,
            assignments[10020][1],
            assignments,
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=256,
        ))

    def test_daytime_assignments_keep_unique_exact_posts_after_reordering(self):
        payload = defense_payload(round_no=54)
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 5, 5),
            unit(10012, "worker", 5, 7),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 12, 6, health=1000),
        ]
        payload["robot"]["roles"] = []
        turn = Turn.load(payload)
        state = state_for(payload)

        first = daytime_post_assignments(
            turn, state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=256,
        )
        self.assertIsNotNone(first)
        first_by_role = {
            role.unit_id: (weapon_id, route[0])
            for weapon_id, (role, route) in first.items()
        }
        self.assertEqual(len(first_by_role), 2)
        self.assertEqual(len({stand for _, stand in first_by_role.values()}), 2)

        reordered = json.loads(json.dumps(payload))
        reordered["roundNo"] = 55
        reordered["teamOur"]["roles"] = list(reversed(
            reordered["teamOur"]["roles"],
        ))
        second = daytime_post_assignments(
            Turn.load(reordered), state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=256,
        )
        second_by_role = {
            role.unit_id: (weapon_id, route[0])
            for weapon_id, (role, route) in second.items()
        }

        self.assertEqual(second_by_role, first_by_role)

    def test_daytime_gunner_route_prefers_back_side_of_tower(self):
        # Break caught: nearest front-side stands ignore the directional wall plan.
        defense = importlib.import_module("agent.defense")
        payload = defense_payload(round_no=5)
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 10, 3),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 10, 5, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 1, 9, health=1500),
        ]
        payload["robot"]["roles"] = []
        turn = Turn.load(payload)

        route = defense._gunner_route(
            turn,
            turn.unit(10010),
            turn.unit(10020),
            lambda: 0.0,
            1.0,
            64,
        )

        self.assertIsNotNone(route)
        self.assertEqual(route[0], Pos(11, 4))

    def test_timely_back_stand_beats_shorter_exposed_stand(self):
        # Break caught: route length outranks a reachable rear daytime position.
        defense = importlib.import_module("agent.defense")
        payload = defense_payload(round_no=65)
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 8, 5),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 10, 5, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 1, 9, health=1500),
        ]
        payload["robot"]["roles"] = []
        turn = Turn.load(payload)

        route = defense._gunner_route(
            turn, turn.unit(10010), turn.unit(10020), lambda: 0.0, 1.0, 64,
        )

        self.assertEqual(route[0], Pos(11, 5))
        self.assertEqual(route[2], 3)

    def test_late_back_stand_does_not_beat_timely_exposed_stand(self):
        # Break caught: rear preference sends a gunner to a post after night starts.
        defense = importlib.import_module("agent.defense")
        payload = defense_payload(round_no=69)
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 8, 5),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 10, 5, health=1000),
        ]
        payload["teamEnemy"]["roles"] = [
            unit(20013, "station", 1, 9, health=1500),
        ]
        payload["robot"]["roles"] = []
        turn = Turn.load(payload)

        route = defense._gunner_route(
            turn, turn.unit(10010), turn.unit(10020), lambda: 0.0, 1.0, 64,
        )

        self.assertEqual(route[0], Pos(10, 4))
        self.assertEqual(route[2], 2)

    def _limited_route_payload(self):
        payload = defense_payload(round_no=65)
        payload["mapInfo"].update({
            "width": 12,
            "height": 12,
            "zones": [
                {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            ],
        })
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 0, 0),
            unit(10013, "station", 9, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
        ]
        payload["teamEnemy"]["roles"] = []
        payload["robot"]["roles"] = []
        return payload

    def test_gunner_route_continues_after_local_expansion_limit(self):
        # Break caught: one hard stand hides another reachable gunner stand.
        defense = importlib.import_module("agent.defense")
        payload = self._limited_route_payload()
        turn = Turn.load(payload)

        candidates = defense.propose_defense(
            turn,
            state_for(payload),
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=7,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].proposal.command, {
            "action": "move", "targetPos": [{"x": 1, "y": 0}],
        })
        self.assertEqual(candidates[0].plan_target, Pos(6, 5))
        self.assertEqual(candidates[0].plan_reason, "gunner:10020")

    def test_gunner_route_stops_after_global_deadline(self):
        defense = importlib.import_module("agent.defense")
        turn = Turn.load(self._limited_route_payload())
        worker = turn.unit(10010)
        weapon = turn.unit(10020)

        class ExpiredClock:
            calls = 0

            def __call__(self):
                self.calls += 1
                return 10.0

        clock = ExpiredClock()
        route = defense._gunner_route(
            turn,
            worker,
            weapon,
            clock,
            5.0,
            7,
        )

        self.assertIsNone(route)
        self.assertEqual(clock.calls, 1)

    def test_allocator_enforces_attack_phase_range_controller_and_count(self):
        # Break caught: domain mistakes escape the shared action gate.
        payload = defense_payload()
        turn = Turn.load(payload)
        base = {
            "action": "attack",
            "controllerId": "10010",
            "targetPos": [{"x": 6, "y": 9}, {"x": 6, "y": 9}],
        }
        self.assertTrue(ActionAllocator(turn).try_add(ActionProposal(
            10020, 10010, base,
        )))

        wrong_count = dict(base, targetPos=[{"x": 6, "y": 9}])
        self.assertFalse(ActionAllocator(turn).try_add(ActionProposal(
            10020, 10010, wrong_count,
        )))
        daylight = defense_payload(round_no=70)
        self.assertFalse(ActionAllocator(Turn.load(daylight)).try_add(
            ActionProposal(10020, 10010, base)
        ))
        far_controller = defense_payload()
        far_controller["teamOur"]["roles"][0]["pos"] = {"x": 0, "y": 0}
        self.assertFalse(ActionAllocator(Turn.load(far_controller)).try_add(
            ActionProposal(10020, 10010, base)
        ))

    def test_weapon_level_target_count_and_cooldown(self):
        # Break caught: upgraded target count or rocket cooldown is ignored.
        candidates = proposals(defense_payload())
        attacks = {
            item.proposal.command_owner_id: item.proposal.command
            for item in candidates
            if item.proposal.command["action"] == "attack"
        }

        self.assertEqual(len(attacks[10020]["targetPos"]), 2)
        self.assertEqual(len(attacks[10030]["targetPos"]), 1)
        self.assertNotIn(10040, attacks)

    def test_night_defense_ignores_stale_daytime_funding_reservation(self):
        # Break caught: a retained daytime fund plan suppresses a live night gun.
        payload = defense_payload()
        engine = DecisionEngine()
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.set_plan(
            10010,
            Pos(6, 6),
            "fund:WeaponUpgradeVoucher1:10020:10020",
            80,
        )

        response = engine.decide(payload)

        self.assertIn("10020", response["roleCommandMap"])
        self.assertEqual(
            response["roleCommandMap"]["10020"]["action"], "attack",
        )

    def test_invalid_daytime_funding_post_does_not_block_gunner_return(self):
        payload = self._limited_route_payload()
        engine = DecisionEngine()
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.set_plan(
            10010,
            Pos(6, 6),
            "fund:WeaponUpgradeVoucher1:99999:99999",
            70,
        )

        response = engine.decide(payload)

        self.assertEqual(
            response["roleCommandMap"]["10010"]["action"], "move",
        )
        self.assertEqual(engine.state.state.plans[10010].reason, "gunner:10020")

    def test_daylight_and_out_of_range_targets_are_not_attacked(self):
        # Break caught: a structurally valid attack violates phase or range.
        daylight = defense_payload(round_no=70)
        self.assertFalse(any(
            item.proposal.command["action"] == "attack"
            for item in proposals(daylight)
        ))

        distant = defense_payload()
        distant["robot"]["roles"] = [
            robot(30001, "smallRobot", 19, 19, 40),
        ]
        self.assertFalse(any(
            item.proposal.command["action"] == "attack"
            for item in proposals(distant)
        ))

    def test_multi_tower_coordination_avoids_already_covered_kill(self):
        # Break caught: both towers waste their full volley on a 10 HP target.
        payload = defense_payload()
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 6, 5),
            unit(10012, "worker", 9, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 9, 6, health=1000),
        ]
        payload["robot"]["roles"] = [
            robot(30001, "smallRobot", 7, 8, 10),
            robot(30002, "middleRobot", 9, 10, 60, target="defender"),
        ]

        attacks = [
            item.proposal.command for item in proposals(payload)
            if item.proposal.command["action"] == "attack"
        ]

        self.assertEqual(len(attacks), 2)
        self.assertNotEqual(attacks[0]["targetPos"], attacks[1]["targetPos"])

    def test_overlapping_adjacent_controllers_staff_both_towers(self):
        # Break caught: tower-order greedy consumes the only controller for tower two.
        payload = defense_payload()
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 7, 6),
            unit(10012, "worker", 5, 6),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 8, 6, health=1000),
        ]
        payload["robot"]["roles"] = [
            robot(30001, "smallRobot", 6, 9, 40),
            robot(30002, "middleRobot", 8, 9, 60),
        ]

        response = DecisionEngine().decide(payload)
        attacks = {
            owner_id: command
            for owner_id, command in response["roleCommandMap"].items()
            if command["action"] == "attack"
        }

        self.assertEqual(set(attacks), {"10020", "10030"})
        self.assertEqual(attacks["10020"]["controllerId"], "10012")
        self.assertEqual(attacks["10030"]["controllerId"], "10010")

    def test_daytime_positioning_uses_same_maximum_matching_as_projection(self):
        # Break caught: projected two-post coverage and issued gunner plans disagree.
        payload = defense_payload(round_no=68)
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 7, 8),
            unit(10012, "worker", 3, 6),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 10, 6, health=1000),
        ]
        payload["robot"]["roles"] = []

        engine = DecisionEngine()
        response = engine.decide(payload)

        self.assertEqual(
            {
                role_id: engine.state.state.plans[int(role_id)].reason
                for role_id, command in response["roleCommandMap"].items()
                if command["action"] == "move"
            },
            {"10010": "gunner:10030", "10012": "gunner:10020"},
        )

    def test_existing_gunner_pairing_is_stable_when_coverage_is_unchanged(self):
        payload = defense_payload(round_no=65)
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 7, 8),
            unit(10012, "worker", 4, 7),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 10, 6, health=1000),
        ]
        payload["robot"]["roles"] = []
        engine = DecisionEngine()
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.set_plan(10010, Pos(6, 6), "gunner:10020", 71)
        engine.state.set_plan(10012, Pos(10, 6), "gunner:10030", 71)

        current = json.loads(json.dumps(payload))
        current["roundNo"] = 66
        response = engine.decide(current)

        self.assertEqual(
            {
                role_id: engine.state.state.plans[int(role_id)].reason
                for role_id, command in response["roleCommandMap"].items()
                if command["action"] == "move"
            },
            {"10010": "gunner:10020", "10012": "gunner:10030"},
        )

    def test_two_staffed_towers_do_not_recall_pioneer_without_third_job(self):
        # Break caught: pioneer returns even though both available guns are staffed.
        payload = defense_payload(round_no=65)
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 6, 5),
            unit(10012, "worker", 9, 5),
            unit(10011, "pioneer", 18, 18),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 9, 6, health=1000),
        ]

        candidates = proposals(payload)

        self.assertFalse(any(
            item.proposal.actor_id == 10011 for item in candidates
        ))

    def test_third_weapon_can_create_a_bounded_pioneer_gunner_plan(self):
        # Break caught: the third tower exists but no third controller is assigned.
        payload = defense_payload(round_no=65)
        payload["teamOur"]["roles"][2]["pos"] = {"x": 17, "y": 6}

        candidates = proposals(payload)
        pioneer = [
            item for item in candidates if item.proposal.actor_id == 10011
        ]

        self.assertEqual(len(pioneer), 1)
        self.assertEqual(pioneer[0].proposal.command["action"], "move")
        self.assertEqual(pioneer[0].plan_reason, "gunner:10040")

    def test_enemy_occupied_gunner_stand_is_avoided(self):
        # Break caught: an occupied gunner route is still scheduled.
        occupied = defense_payload(round_no=65)
        occupied["teamOur"]["roles"][2]["pos"] = {"x": 17, "y": 6}
        occupied["teamEnemy"]["roles"] = [
            unit(20010, "worker", 13, 6),
        ]
        pioneer = next(
            item for item in proposals(occupied)
            if item.proposal.actor_id == 10011
        )
        self.assertNotEqual(pioneer.plan_target, Turn.load(occupied).enemies[0].pos)
        self.assertNotEqual(
            pioneer.proposal.command["targetPos"], [{"x": 13, "y": 6}]
        )

    def test_missing_or_zero_health_station_does_not_disable_remaining_weapon(self):
        # Break caught: loss of base crashes or suppresses a valid remaining gun.
        for keep_zero_station in (False, True):
            payload = defense_payload()
            roles = payload["teamOur"]["roles"]
            if keep_zero_station:
                for entry in roles:
                    if entry["roleType"] == "station":
                        entry["health"] = 0
            else:
                payload["teamOur"]["roles"] = [
                    entry for entry in roles if entry["roleType"] != "station"
                ]

            attacks = [
                item for item in proposals(payload)
                if item.proposal.command["action"] == "attack"
            ]
            self.assertTrue(attacks)

    def test_distant_gunners_start_returning_even_if_they_miss_dusk(self):
        # Break caught: every unstaffed tower retries one late worker then gives up.
        payload = defense_payload(round_no=59)
        payload["mapInfo"].update({"width": 41, "height": 32})
        payload["teamOur"]["roles"][0]["pos"] = {"x": 35, "y": 30}
        payload["teamOur"]["roles"][1]["pos"] = {"x": 34, "y": 30}
        payload["teamOur"]["roles"][2]["pos"] = {"x": 7, "y": 5}

        moves = [
            item for item in proposals(payload)
            if item.proposal.command["action"] == "move"
        ]

        self.assertEqual({item.proposal.actor_id for item in moves}, {10010, 10012})
        self.assertTrue(all(item.deadline_round > 70 for item in moves))

        response = DecisionEngine().decide(payload)
        self.assertEqual(
            {
                int(role_id)
                for role_id, command in response["roleCommandMap"].items()
                if command["action"] == "move"
            },
            {10010, 10012},
        )

    def test_route_cost_can_start_recall_before_fixed_dusk_window(self):
        # Break caught: the fixed 12-round gate hides a 28-step return route.
        payload = defense_payload(round_no=45)
        payload["mapInfo"].update({"width": 41, "height": 32})
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 35, 30),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
        ]

        engine = DecisionEngine()
        response = engine.decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")
        self.assertEqual(engine.state.state.plans[10010].reason, "gunner:10020")

        shorter = defense_payload(round_no=45)
        shorter["teamOur"]["roles"] = [
            unit(10010, "worker", 16, 6),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
        ]
        self.assertFalse(any(
            item.plan_reason == "gunner:10020" for item in proposals(shorter)
        ))

    def test_railgun_conservatively_spends_energy_on_current_health(self):
        # Break caught: another tower's forecast makes railgun energy pass through.
        defense = importlib.import_module("agent.defense")
        payload = defense_payload()
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 5, 5),
            unit(10030, "railgun", 6, 6, health=1000),
        ]
        payload["robot"]["roles"] = [
            robot(30001, "smallRobot", 7, 7, 10),
            robot(30002, "smallRobot", 8, 8, 10),
        ]
        turn = Turn.load(payload)
        projected = {30001: 10}

        defense._apply_projected_damage(
            turn,
            turn.weapons()[0],
            turn.robots[1],
            1,
            projected,
        )

        self.assertNotIn(30002, projected)


if __name__ == "__main__":
    unittest.main()
