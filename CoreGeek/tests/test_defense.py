import importlib
import inspect
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.actions import ActionAllocator, ActionProposal, PlannedAction
from agent.brain import DecisionEngine
from agent.defense import daytime_post_assignments, daytime_work_can_return
from agent.economy import propose_economy
from agent.grid import next_step
from agent.protocol import move_command, sell_command
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


def shared_rocket_payload(*, workers=((10010, 6, 8),),
                          cooldowns=(2, 0, 0), round_no=71):
    payload = defense_payload(round_no=round_no)
    payload["teamOur"]["roles"] = [
        *(unit(role_id, "worker", x, y) for role_id, x, y in workers),
        unit(10013, "station", 8, 9, health=1500),
        unit(10020, "rocket", 7, 8, health=1000, cooldown=cooldowns[0]),
        unit(10030, "rocket", 7, 9, health=1000, cooldown=cooldowns[1]),
        unit(10040, "rocket", 7, 7, health=1000, cooldown=cooldowns[2]),
    ]
    payload["robot"]["roles"] = [
        robot(30001, "smallRobot", 7, 5, 50),
        robot(30002, "smallRobot", 8, 5, 50),
    ]
    return payload


def long_day_work_payload(*, round_no=10, vendor=(34, 14)):
    payload = defense_payload(round_no=round_no)
    payload["mapInfo"].update({
        "width": 41, "height": 32,
        "zones": [{"pos": {"x": vendor[0], "y": vendor[1]},
                   "neutralType": "vendor"}],
    })
    worker = unit(10010, "worker", 33, 14)
    worker["backpack"] = ["copper"]
    payload["teamOur"]["roles"] = [
        worker, unit(10013, "station", 9, 22, health=1500),
        unit(10020, "rocket", 7, 20, health=1000),
        unit(10021, "rocket", 8, 21, health=1000),
        unit(10022, "rocket", 9, 21, health=1000),
    ]
    payload["teamEnemy"]["roles"] = []
    payload["robot"]["roles"] = []
    payload["vendorShopList"] = [{"name": "copper", "price": 5}]
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
    def test_shared_gunners_never_double_use_a_role(self):
        for workers, expected in (
            ((), set()),
            (((10010, 6, 8),), {"10020"}),
            (((10010, 6, 8), (10012, 6, 9)), {"10020", "10030"}),
            (((10010, 6, 8), (10012, 6, 9), (10011, 6, 7)),
             {"10020", "10030", "10040"}),
        ):
            with self.subTest(workers=workers):
                response = DecisionEngine(clock=lambda: 0.0).decide(
                    shared_rocket_payload(workers=workers, cooldowns=(0, 0, 0)),
                )
                attacks = {
                    weapon_id: command for weapon_id, command in
                    response["roleCommandMap"].items()
                    if command["action"] == "attack"
                }
                self.assertEqual(set(attacks), expected)
                self.assertEqual(len({
                    command["controllerId"] for command in attacks.values()
                }), len(attacks))

    def test_shared_gunner_cannot_fire_all_cooling_or_targetless_towers(self):
        for cooldowns, no_targets in (((2, 2, 2), False), ((0, 0, 0), True)):
            with self.subTest(cooldowns=cooldowns, no_targets=no_targets):
                payload = shared_rocket_payload(cooldowns=cooldowns)
                if no_targets:
                    payload["robot"]["roles"] = []
                response = DecisionEngine(clock=lambda: 0.0).decide(payload)
                self.assertFalse(any(
                    command["action"] == "attack"
                    for command in response["roleCommandMap"].values()
                ))

    def test_shared_gunner_prefers_in_range_tower_even_with_higher_id(self):
        payload = shared_rocket_payload(cooldowns=(0, 0, 0))
        for role in payload["teamOur"]["roles"]:
            if role["id"] == 10020:
                role["attackRange"] = 1
        response = DecisionEngine(clock=lambda: 0.0).decide(payload)
        attacks = {
            weapon_id for weapon_id, command in response["roleCommandMap"].items()
            if command["action"] == "attack"
        }
        self.assertEqual(attacks, {"10030"})

    def test_cooling_old_plan_and_role_order_do_not_lock_shared_gunner(self):
        payload = shared_rocket_payload()
        payload["teamOur"]["roles"].reverse()
        engine = DecisionEngine(clock=lambda: 0.0)
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.set_plan(10010, Pos(6, 8), "gunner:10020", 80)

        response = engine.decide(payload)

        self.assertEqual(
            {key for key, value in response["roleCommandMap"].items()
             if value["action"] == "attack"},
            {"10030"},
        )

    def test_shared_gunner_returns_without_attack_at_deadline(self):
        self.assertFalse(any(
            item.proposal.command["action"] == "attack"
            for item in importlib.import_module("agent.defense").propose_defense(
                Turn.load(shared_rocket_payload(cooldowns=(0, 0, 0))),
                state_for(shared_rocket_payload(cooldowns=(0, 0, 0))),
                clock=lambda: 1.0,
                deadline=1.0,
                max_expansions=64,
            )
        ))

    def test_target_selection_failure_reassigns_shared_gunner(self):
        payload = shared_rocket_payload(cooldowns=(0, 0, 0))
        defense = importlib.import_module("agent.defense")
        select = defense._select_rocket_targets

        def fail_first(turn, weapon, count, projected_damage, **kwargs):
            if weapon.unit_id == 10020:
                return ()
            return select(turn, weapon, count, projected_damage, **kwargs)

        with patch.object(defense, "_select_rocket_targets", side_effect=fail_first):
            response = DecisionEngine(clock=lambda: 0.0).decide(payload)
        attacks = {
            weapon_id for weapon_id, command in response["roleCommandMap"].items()
            if command["action"] == "attack"
        }
        self.assertEqual(attacks, {"10030"})

    def test_shared_gunner_rotates_after_cooldown_feedback(self):
        engine = DecisionEngine(clock=lambda: 0.0)
        first = shared_rocket_payload(cooldowns=(0, 2, 2))
        response = engine.decide(first)
        self.assertEqual(
            {key for key, value in response["roleCommandMap"].items()
             if value["action"] == "attack"},
            {"10020"},
        )
        next_round = shared_rocket_payload(cooldowns=(2, 0, 2), round_no=72)
        next_round["lastRoundRoleActionResults"] = {"10020": True}
        response = engine.decide(next_round)
        self.assertEqual(
            {key for key, value in response["roleCommandMap"].items()
             if value["action"] == "attack"},
            {"10030"},
        )

    def test_one_shared_gunner_fires_ready_rocket_not_cooling_rocket(self):
        # Break caught: a cooling tower wins the one-role adjacency match.
        response = DecisionEngine(clock=lambda: 0.0).decide(
            shared_rocket_payload(),
        )
        attacks = {
            weapon_id: command for weapon_id, command in
            response["roleCommandMap"].items()
            if command["action"] == "attack"
        }
        self.assertEqual(set(attacks), {"10030"})
        self.assertEqual(attacks["10030"]["controllerId"], "10010")

    def test_two_shared_gunners_cover_both_ready_rockets(self):
        # Break caught: maximum staffing counts a cooling rocket as useful.
        response = DecisionEngine(clock=lambda: 0.0).decide(
            shared_rocket_payload(workers=((10010, 6, 8), (10012, 6, 9))),
        )
        attacks = {
            weapon_id: command for weapon_id, command in
            response["roleCommandMap"].items()
            if command["action"] == "attack"
        }
        self.assertEqual(set(attacks), {"10030", "10040"})
        self.assertEqual(
            {command["controllerId"] for command in attacks.values()},
            {"10010", "10012"},
        )

    def test_long_legal_sale_keeps_exact_post_return_route(self):
        payload = long_day_work_payload()
        turn = Turn.load(payload)
        worker = turn.unit(10010)
        candidate = PlannedAction(ActionProposal(
            10010, 10010, sell_command("copper", 1),
        ))
        route = (Pos(7, 19), None, 26)
        assignments = {10020: (worker, route)}
        kwargs = dict(clock=lambda: 0.0, deadline=1.0, max_expansions=256)
        self.assertTrue(daytime_work_can_return(
            turn, worker, candidate, route, assignments, **kwargs,
        ))
        economy_candidates = propose_economy(
            turn, state_for(payload), **kwargs,
        )
        self.assertTrue(any(item.proposal.command["action"] == "sell"
                            for item in economy_candidates))
        self.assertEqual(
            DecisionEngine(clock=lambda: 0.0).decide(payload)["roleCommandMap"]["10010"],
            sell_command("copper", 1),
        )
        late = Turn.load(long_day_work_payload(round_no=43))
        self.assertFalse(daytime_work_can_return(
            late, late.unit(10010), candidate, route,
            {10020: (late.unit(10010), route)}, **kwargs,
        ))
        blocked = long_day_work_payload()
        blocked["teamOur"]["roles"].append(
            unit(10023, "wall", 7, 19, health=1000),
        )
        blocked_turn = Turn.load(blocked)
        self.assertFalse(daytime_work_can_return(
            blocked_turn, blocked_turn.unit(10010), candidate, route,
            {10020: (blocked_turn.unit(10010), route)}, **kwargs,
        ))

    def test_long_work_route_uses_same_first_step_as_deep_economy_path(self):
        defense = importlib.import_module("agent.defense")
        turn = Turn.load(long_day_work_payload(vendor=(7, 19)))
        worker = turn.unit(10010)
        target = Pos(7, 19)
        kwargs = dict(clock=lambda: 0.0, deadline=1.0, max_expansions=256)
        route = defense._day_work_route(turn, worker, target, **kwargs)
        self.assertIsNotNone(route)
        stand, cost, first_step = route
        economic_path = next_step(
            turn, worker, stand, prefer_deep_ties=True, **kwargs,
        )
        self.assertEqual((cost, first_step),
                         (economic_path.cost, economic_path.step))
        candidate = PlannedAction(ActionProposal(
            10010, 10010, move_command(first_step), destination=first_step,
        ), plan_target=target, plan_reason="vendor")
        post = (Pos(6, 19), None, 27)
        self.assertTrue(daytime_work_can_return(
            turn, worker, candidate, post,
            {10020: (worker, post)}, **kwargs,
        ))
        unreachable = long_day_work_payload(vendor=(7, 19))
        unreachable["teamOur"]["roles"].extend(
            unit(20000 + y, "wall", 20, y, health=1000)
            for y in range(32)
        )
        cut_turn = Turn.load(unreachable)
        cut_worker = cut_turn.unit(10010)
        self.assertIsNone(defense._day_work_route(
            cut_turn, cut_worker, target, **kwargs,
        ))
        self.assertFalse(daytime_work_can_return(
            cut_turn, cut_worker, candidate, post,
            {10020: (cut_worker, post)}, **kwargs,
        ))

    def test_post_assignment_diagnostic_distinguishes_budget_partial_and_deadline(self):
        self.assertIn(
            "diagnostic_sink", inspect.signature(daytime_post_assignments).parameters,
        )
        payload = defense_payload(round_no=60)
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 5, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
            unit(10030, "railgun", 12, 6, health=1000),
        ]
        payload["robot"]["roles"] = []
        turn = Turn.load(payload)

        empty_payload = json.loads(json.dumps(payload))
        empty_payload["teamOur"]["roles"][0]["pos"] = {"x": 1, "y": 1}
        empty_trace = []
        empty = daytime_post_assignments(
            Turn.load(empty_payload), state_for(empty_payload),
            clock=lambda: 0, deadline=1,
            max_expansions=0, diagnostic_sink=empty_trace.append,
        )
        self.assertEqual(empty, {})
        self.assertEqual(empty_trace[0]["status"], "empty")
        self.assertGreater(empty_trace[0]["routeResults"]["expansion_limit"], 0)

        partial_trace = []
        partial = daytime_post_assignments(
            turn, state_for(payload), clock=lambda: 0, deadline=1,
            max_expansions=256, diagnostic_sink=partial_trace.append,
        )
        self.assertEqual(len(partial), 1)
        self.assertEqual(partial_trace[0]["status"], "partial")

        deadline_trace = []
        self.assertIsNone(daytime_post_assignments(
            turn, state_for(payload), clock=lambda: 10, deadline=1,
            max_expansions=256, diagnostic_sink=deadline_trace.append,
        ))
        self.assertEqual(deadline_trace[0]["status"], "deadline")
        self.assertEqual(deadline_trace[0]["required"], 2)

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

    def test_idle_pioneer_buys_and_uses_cash_upgrade_voucher(self):
        payload = defense_payload(round_no=10)
        payload["teamOur"]["teamId"] = "idle-pioneer-upgrade"
        payload["teamOur"]["goldNum"] = 150
        payload["robot"]["roles"] = []
        payload["mapInfo"]["zones"] = [{
            "pos": {"x": 12, "y": 4}, "neutralType": "weaponShop",
        }]
        payload["weaponShopList"] = [{
            "name": "WeaponUpgradeVoucher2", "price": 150,
        }]
        for role in payload["teamOur"]["roles"]:
            if role["roleType"] == "worker":
                role["backPackCapability"] = 1
                role["backpack"] = ["stone"]
        engine = DecisionEngine()

        first = engine.decide(payload)["roleCommandMap"]

        self.assertEqual(first["10011"], {
            "action": "buy", "name": "WeaponUpgradeVoucher2", "num": 1,
        })
        self.assertFalse(any(
            command.get("action") in ("collect", "build", "sell")
            for role_id, command in first.items() if role_id == "10011"
        ))

        payload["roundNo"] = 11
        payload["teamOur"]["goldNum"] = 0
        payload["teamOur"]["roles"][2]["backpack"] = [
            "WeaponUpgradeVoucher2"
        ]
        payload["lastRoundRoleActionResults"] = {"10011": True}
        second = engine.decide(payload)["roleCommandMap"]

        self.assertEqual(second["10011"], {
            "action": "use", "name": "WeaponUpgradeVoucher2",
            "targetPos": [{"x": 12, "y": 6}],
        })

    def test_pioneer_reuses_protected_cash_while_moving_to_shop(self):
        payload = defense_payload(round_no=10)
        payload["teamOur"]["teamId"] = "pioneer-protected-shop-route"
        payload["teamOur"]["goldNum"] = 150
        payload["robot"]["roles"] = []
        payload["teamOur"]["roles"][2]["pos"] = {"x": 12, "y": 2}
        payload["mapInfo"]["zones"] = [{
            "pos": {"x": 16, "y": 2}, "neutralType": "weaponShop",
        }]
        payload["weaponShopList"] = [{
            "name": "WeaponUpgradeVoucher2", "price": 150,
        }]
        for role in payload["teamOur"]["roles"]:
            if role["roleType"] == "worker":
                role["backPackCapability"] = 1
                role["backpack"] = ["stone"]
        engine = DecisionEngine()
        bought = False
        saw_move = False

        for round_no in range(10, 18):
            payload["roundNo"] = round_no
            command = engine.decide(payload)["roleCommandMap"]["10011"]
            self.assertNotIn(command["action"], ("collect", "build", "sell"))
            if command["action"] == "buy":
                bought = True
                break
            self.assertEqual(command["action"], "move")
            saw_move = True
            payload["teamOur"]["roles"][2]["pos"] = command["targetPos"][0]
            payload["lastRoundRoleActionResults"] = {"10011": True}

        self.assertTrue(saw_move)
        self.assertTrue(bought)

    def test_pioneer_does_not_purchase_when_task_is_available(self):
        payload = defense_payload(round_no=10)
        payload["teamOur"]["teamId"] = "task-before-pioneer-upgrade"
        payload["teamOur"]["goldNum"] = 150
        payload["robot"]["roles"] = []
        payload["mapInfo"]["zones"] = [{
            "pos": {"x": 12, "y": 4}, "neutralType": "weaponShop",
        }]
        payload["weaponShopList"] = [{
            "name": "WeaponUpgradeVoucher2", "price": 150,
        }]
        payload["teamOur"]["playerTasks"] = [{
            "taskType": "synthetic",
            "taskPosition": {"x": 13, "y": 5},
            "coldDownRounds": 0,
            "scoreReward": 10,
            "goldReward": 10,
            "isValid": True,
            "timeoutRounds": 20,
        }]

        commands = DecisionEngine().decide(payload)["roleCommandMap"]

        self.assertNotEqual(commands.get("10011", {}).get("action"), "buy")

    def test_pioneer_may_purchase_when_task_solver_window_is_too_short(self):
        payload = defense_payload(round_no=10)
        payload["teamOur"]["teamId"] = "late-task-allows-pioneer-upgrade"
        payload["teamOur"]["goldNum"] = 150
        payload["robot"]["roles"] = []
        payload["mapInfo"]["zones"] = [{
            "pos": {"x": 12, "y": 4}, "neutralType": "weaponShop",
        }]
        payload["weaponShopList"] = [{
            "name": "WeaponUpgradeVoucher2", "price": 150,
        }]
        payload["teamOur"]["playerTasks"] = [{
            "taskType": "synthetic",
            "taskPosition": {"x": 13, "y": 5},
            "coldDownRounds": 0,
            "scoreReward": 10,
            "goldReward": 10,
            "isValid": True,
            "timeoutRounds": 1,
        }]
        for role in payload["teamOur"]["roles"]:
            if role["roleType"] == "worker":
                role["backPackCapability"] = 1
                role["backpack"] = ["stone"]

        commands = DecisionEngine().decide(payload)["roleCommandMap"]

        self.assertEqual(commands["10011"], {
            "action": "buy", "name": "WeaponUpgradeVoucher2", "num": 1,
        })

    def test_pioneer_does_not_start_upgrade_without_return_window(self):
        payload = defense_payload(round_no=70)
        payload["teamOur"]["teamId"] = "late-pioneer-upgrade"
        payload["teamOur"]["goldNum"] = 150
        payload["robot"]["roles"] = []
        payload["mapInfo"]["zones"] = [{
            "pos": {"x": 12, "y": 4}, "neutralType": "weaponShop",
        }]
        payload["weaponShopList"] = [{
            "name": "WeaponUpgradeVoucher2", "price": 150,
        }]
        for role in payload["teamOur"]["roles"]:
            if role["roleType"] == "worker":
                role["backPackCapability"] = 1
                role["backpack"] = ["stone"]

        commands = DecisionEngine().decide(payload)["roleCommandMap"]

        self.assertNotEqual(commands.get("10011", {}).get("action"), "buy")

    def test_new_task_pauses_held_voucher_then_voucher_is_used(self):
        payload = defense_payload(round_no=10)
        payload["teamOur"]["teamId"] = "task-pauses-pioneer-voucher"
        payload["teamOur"]["goldNum"] = 150
        payload["robot"]["roles"] = []
        payload["teamOur"]["roles"][2]["pos"] = {"x": 12, "y": 2}
        payload["mapInfo"]["zones"] = [{
            "pos": {"x": 12, "y": 3}, "neutralType": "weaponShop",
        }]
        payload["weaponShopList"] = [{
            "name": "WeaponUpgradeVoucher2", "price": 150,
        }]
        for role in payload["teamOur"]["roles"]:
            if role["roleType"] == "worker":
                role["backPackCapability"] = 1
                role["backpack"] = ["stone"]
        engine = DecisionEngine()
        self.assertEqual(
            engine.decide(payload)["roleCommandMap"]["10011"]["action"],
            "buy",
        )

        payload["roundNo"] = 11
        payload["teamOur"]["goldNum"] = 0
        payload["teamOur"]["roles"][2]["backpack"] = [
            "WeaponUpgradeVoucher2"
        ]
        payload["lastRoundRoleActionResults"] = {"10011": True}
        payload["teamOur"]["playerTasks"] = [{
            "taskType": "synthetic",
            "taskPosition": {"x": 13, "y": 5},
            "coldDownRounds": 0,
            "scoreReward": 10,
            "goldReward": 10,
            "isValid": True,
            "timeoutRounds": 20,
        }]
        task_command = engine.decide(payload)["roleCommandMap"]["10011"]
        self.assertIn(task_command["action"], ("move", "acceptTask"))
        self.assertNotIn(task_command["action"], ("buy", "use"))

        payload["roundNo"] = 12
        payload["teamOur"]["playerTasks"] = []
        payload["lastRoundRoleActionResults"] = {"10011": True}
        used = False
        saw_move = False
        for round_no in range(12, 22):
            payload["roundNo"] = round_no
            resumed = engine.decide(payload)["roleCommandMap"]["10011"]
            if resumed["action"] == "use":
                self.assertEqual(resumed["name"], "WeaponUpgradeVoucher2")
                used = True
                break
            self.assertEqual(resumed["action"], "move")
            saw_move = True
            payload["teamOur"]["roles"][2]["pos"] = resumed["targetPos"][0]
            payload["lastRoundRoleActionResults"] = {"10011": True}
        self.assertTrue(saw_move)
        self.assertTrue(used)

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

    def test_unactionable_funding_plan_does_not_hold_worker_through_dusk(self):
        # A stale fund plan cannot reserve the only worker and tower for 11 idle rounds.
        payload = defense_payload(round_no=60)
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 18, 18),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000),
        ]
        payload["robot"]["roles"] = []
        payload["weaponShopList"] = []
        payload["mapInfo"]["zones"] = []
        engine = DecisionEngine()
        engine.state.observe(
            Turn.load(payload), payload, request_fingerprint(payload),
        )
        engine.state.set_plan(
            10010, Pos(8, 9),
            "fund:StationUpgradeVoucher1:10020:10013", 70,
        )

        rounds_with_no_return = []
        first_trace = []
        for round_no in range(60, 73):
            payload["roundNo"] = round_no
            if round_no >= 71:
                payload["robot"]["roles"] = [
                    robot(30001, "smallRobot", 2, 10, 30),
                ]
            if round_no > 60:
                payload["lastRoundRoleActionResults"] = {"10010": True}
            trace = []
            command = engine.decide(
                payload, trace_sink=trace.append,
            )["roleCommandMap"].get("10010")
            if round_no == 60:
                first_trace = trace
            position = payload["teamOur"]["roles"][0]["pos"]
            if max(abs(position["x"] - 6), abs(position["y"] - 6)) > 1:
                if command is None or command.get("action") != "move":
                    rounds_with_no_return.append(round_no)
            if command is not None and command.get("action") == "move":
                payload["teamOur"]["roles"][0]["pos"] = command["targetPos"][0]
        self.assertEqual(rounds_with_no_return, [])
        self.assertIn("defensePlanning", first_trace[0])
        self.assertEqual(first_trace[0]["defensePlanning"]["fundingReservedRoles"], 0)
        self.assertEqual(first_trace[0]["defensePlanning"]["postStatus"], "complete")
        self.assertGreaterEqual(
            first_trace[0]["defensePlanning"]["returnCandidates"], 1,
        )

    def test_long_open_return_starts_before_night_and_arrives(self):
        payload = defense_payload(round_no=281)
        payload["mapInfo"].update({
            "width": 41, "height": 32,
            "zones": [{"pos": {"x": 34, "y": 14}, "neutralType": "copper"}],
        })
        payload["teamOur"]["roles"] = [
            unit(10010, "worker", 33, 14),
            unit(10013, "station", 9, 22, health=1500),
            unit(10020, "gatling", 8, 19, health=1000),
        ]
        payload["robot"]["roles"] = []
        payload["weaponShopList"] = []
        payload["vendorShopList"] = [{"name": "copper", "price": 5}]
        engine = DecisionEngine()
        first_return_round = None
        for round_no in range(281, 332):
            payload["roundNo"] = round_no
            if round_no > 281:
                payload["lastRoundRoleActionResults"] = {"10010": True}
            command = engine.decide(payload)["roleCommandMap"].get("10010")
            if command is not None and command.get("action") == "move":
                if first_return_round is None:
                    first_return_round = round_no
                payload["teamOur"]["roles"][0]["pos"] = command["targetPos"][0]
        position = payload["teamOur"]["roles"][0]["pos"]
        self.assertIsNotNone(first_return_round)
        self.assertLess(first_return_round, 331)
        self.assertLessEqual(max(abs(position["x"] - 8),
                                 abs(position["y"] - 19)), 1)

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
