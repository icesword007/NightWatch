import unittest

from agent.actions import ActionProposal, PlannedAction
from agent.brain import DecisionEngine
from agent.economy import (
    prepare_base_reserve, propose_base_upgrade, propose_economy,
    reserve_purchase_candidate,
)
from agent.protocol import Pos, Turn
from agent.state import BaseReserve, PlanState, request_fingerprint
from test_economy import economy_payload, role, state_for


def base_payload(*, round_no=1, worker_pos=(6, 1), gold=100, health=1500,
                 items=(), robots=()):
    payload = economy_payload(
        round_no=round_no, worker_pos=worker_pos, gold=gold, items=items,
    )
    payload["weaponShopList"] = [
        {"name": "StationUpgradeVoucher1", "price": 100},
        {"name": "Medicine", "price": 10},
    ]
    payload["teamOur"]["roles"][1]["health"] = health
    payload["robot"]["roles"] = list(robots)
    return payload


def seed_reserve(engine, payload):
    turn = Turn.load(payload)
    engine.state.observe(turn, payload, request_fingerprint(payload))
    engine.state.state.base_reserve = BaseReserve(
        10010, 10013, 1, "StationUpgradeVoucher1",
    )


def balanced_payload(*, health=1500, gold=100, items=(), round_no=1):
    payload = base_payload(
        round_no=round_no, worker_pos=(5, 2), gold=gold,
        health=health, items=items,
    )
    payload["teamOur"]["teamId"] = "balanced-cash-investment"
    payload["teamOur"]["roles"][2]["level"] = 2
    payload["weaponShopList"].insert(
        0, {"name": "WeaponUpgradeVoucher1", "price": 100},
    )
    return payload


class BaseUpgradeTests(unittest.TestCase):
    def test_new_cash_investment_balances_base_after_first_weapon_level(self):
        # Break caught: the first eligible weapon voucher always precedes a
        # fully funded base voucher after one weapon has advanced a level.
        payload = balanced_payload()
        response = DecisionEngine(clock=lambda: 0.0).decide(payload)
        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "buy", "name": "StationUpgradeVoucher1", "num": 1,
        })

    def test_balanced_purchase_keeps_night_reserve_until_damage(self):
        engine = DecisionEngine(clock=lambda: 0.0)
        payload = balanced_payload()
        self.assertEqual(
            engine.decide(payload)["roleCommandMap"]["10010"]["name"],
            "StationUpgradeVoucher1",
        )
        self.assertIsNotNone(engine.state.state.base_reserve)
        payload["roundNo"] = 71
        payload["teamOur"]["goldNum"] = 0
        payload["teamOur"]["roles"][0]["backpack"] = ["StationUpgradeVoucher1"]
        payload["teamOur"]["roles"][0]["pos"] = {"x": 8, "y": 9}
        payload["lastRoundRoleActionResults"] = {"10010": True}
        held = engine.decide(payload)["roleCommandMap"].get("10010", {})
        self.assertNotEqual(held.get("action"), "use")
        payload["roundNo"] = 72
        payload["teamOur"]["roles"][1]["health"] = 300
        self.assertEqual(engine.decide(payload)["roleCommandMap"]["10010"], {
            "action": "use", "name": "StationUpgradeVoucher1",
            "targetPos": [{"x": 9, "y": 9}],
        })

    def test_balanced_purchase_uses_damaged_base_during_day(self):
        engine = DecisionEngine(clock=lambda: 0.0)
        payload = balanced_payload(health=1000)
        self.assertEqual(
            engine.decide(payload)["roleCommandMap"]["10010"]["name"],
            "StationUpgradeVoucher1",
        )
        self.assertIsNone(engine.state.state.base_reserve)
        payload["roundNo"] = 2
        payload["teamOur"]["goldNum"] = 0
        payload["teamOur"]["roles"][0]["backpack"] = ["StationUpgradeVoucher1"]
        payload["teamOur"]["roles"][0]["pos"] = {"x": 8, "y": 9}
        payload["lastRoundRoleActionResults"] = {"10010": True}
        self.assertEqual(engine.decide(payload)["roleCommandMap"]["10010"], {
            "action": "use", "name": "StationUpgradeVoucher1",
            "targetPos": [{"x": 9, "y": 9}],
        })

    def test_balanced_purchase_keeps_weapon_priority_until_weapon_leads(self):
        payload = balanced_payload()
        payload["teamOur"]["roles"][2]["level"] = 1
        self.assertEqual(
            DecisionEngine(clock=lambda: 0.0).decide(payload)
            ["roleCommandMap"]["10010"]["name"],
            "WeaponUpgradeVoucher1",
        )
        payload = balanced_payload(gold=150)
        payload["teamOur"]["roles"][1].update(level=2, health=3000)
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher2", "price": 150},
            {"name": "StationUpgradeVoucher2", "price": 150},
        ]
        self.assertEqual(
            DecisionEngine(clock=lambda: 0.0).decide(payload)
            ["roleCommandMap"]["10010"]["name"],
            "WeaponUpgradeVoucher2",
        )
        payload["teamOur"]["roles"][1].update(level=3, health=4500)
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher2", "price": 150},
        ]
        self.assertEqual(
            DecisionEngine(clock=lambda: 0.0).decide(payload)
            ["roleCommandMap"]["10010"]["name"],
            "WeaponUpgradeVoucher2",
        )

    def test_balanced_purchase_falls_back_on_price_and_cash(self):
        payload = balanced_payload()
        payload["weaponShopList"][1]["price"] = 101
        self.assertEqual(
            DecisionEngine(clock=lambda: 0.0).decide(payload)
            ["roleCommandMap"]["10010"]["name"],
            "WeaponUpgradeVoucher1",
        )
        payload = balanced_payload(gold=0, items=("copper",))
        payload["vendorShopList"] = [{"name": "copper", "price": 100}]
        engine = DecisionEngine(clock=lambda: 0.0)
        response = engine.decide(payload)["roleCommandMap"].get("10010", {})
        self.assertNotEqual(response.get("name"), "StationUpgradeVoucher1")
        self.assertTrue(engine.state.state.plans[10010].reason.startswith(
            "fund:WeaponUpgradeVoucher1:"
        ))

    def test_balanced_purchase_does_not_replace_cross_role_commitment(self):
        payload = balanced_payload()
        payload["teamOur"]["roles"].insert(
            1, role(10011, "worker", 5, 1),
        )
        engine = DecisionEngine(clock=lambda: 0.0)
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.set_plan(
            10011, Pos(8, 8), "fund:WeaponUpgradeVoucher1:10020:10030", 70,
        )
        commands = engine.decide(payload)["roleCommandMap"].values()
        self.assertFalse(any(
            command.get("name") == "StationUpgradeVoucher1"
            for command in commands
        ))

    def test_balanced_purchase_does_not_replace_held_voucher_or_medicine(self):
        payload = balanced_payload()
        payload["teamOur"]["roles"].insert(
            1, role(10011, "worker", 8, 9, items=("WeaponUpgradeVoucher1",)),
        )
        commands = DecisionEngine(clock=lambda: 0.0).decide(payload)[
            "roleCommandMap"
        ].values()
        self.assertFalse(any(
            command.get("name") == "StationUpgradeVoucher1"
            for command in commands
        ))
        payload = balanced_payload()
        payload["teamOur"]["roles"][0]["health"] = 100
        self.assertEqual(
            DecisionEngine(clock=lambda: 0.0).decide(payload)
            ["roleCommandMap"]["10010"]["name"],
            "Medicine",
        )

    def test_balanced_purchase_leaves_cash_for_other_roles_urgent_medicine(self):
        payload = balanced_payload()
        payload["teamOur"]["roles"].insert(
            1, role(10011, "worker", 5, 1, health=30),
        )

        commands = DecisionEngine(clock=lambda: 0.0).decide(payload)[
            "roleCommandMap"
        ]

        self.assertFalse(any(
            command.get("name") == "StationUpgradeVoucher1"
            for command in commands.values()
        ))

    def test_balanced_purchase_needs_a_complete_route_and_return_margin(self):
        payload = balanced_payload()
        payload["teamOur"]["roles"].extend(
            role(30000 + index, "wall", x, y, health=1000)
            for index, (x, y) in enumerate((
                (8, 9), (8, 10), (9, 8), (9, 10),
                (10, 8), (10, 9), (10, 10),
            ))
        )
        self.assertEqual(
            DecisionEngine(clock=lambda: 0.0).decide(payload)
            ["roleCommandMap"]["10010"]["name"],
            "WeaponUpgradeVoucher1",
        )
        at_boundary = balanced_payload(round_no=58)
        self.assertEqual(
            DecisionEngine(clock=lambda: 0.0).decide(at_boundary)
            ["roleCommandMap"]["10010"]["name"],
            "StationUpgradeVoucher1",
        )
        after_boundary = balanced_payload(round_no=59)
        self.assertEqual(
            DecisionEngine(clock=lambda: 0.0).decide(after_boundary)
            ["roleCommandMap"]["10010"]["name"],
            "WeaponUpgradeVoucher1",
        )
        budget_unknown = balanced_payload()
        command = DecisionEngine(
            clock=lambda: 0.0, max_search_expansions=8,
        ).decide(budget_unknown)["roleCommandMap"].get("10010", {})
        self.assertNotEqual(command.get("name"), "StationUpgradeVoucher1")

    def test_balanced_purchase_reserves_cash_during_route_for_other_worker(self):
        payload = balanced_payload()
        payload["teamOur"]["roles"][0]["pos"] = {"x": 3, "y": 2}
        payload["teamOur"]["roles"].insert(
            1, role(10011, "worker", 5, 1),
        )
        engine = DecisionEngine(clock=lambda: 0.0)
        commands = engine.decide(payload)["roleCommandMap"]
        self.assertTrue(engine.state.state.plans[10010].reason.startswith(
            "fund:StationUpgradeVoucher1:"
        ))
        self.assertFalse(any(
            command.get("action") == "buy" for command in commands.values()
        ))

    def test_balanced_base_reserve_is_not_displaced_by_other_new_funding(self):
        # Break caught: another worker's sale-funded weapon plan can prevent
        # the newly bought full-health base voucher from being reserved.
        payload = balanced_payload()
        payload["teamOur"]["roles"].insert(
            1, role(10011, "worker", 3, 2, items=("copper",)),
        )
        engine = DecisionEngine(clock=lambda: 0.0)

        commands = engine.decide(payload)["roleCommandMap"]

        self.assertEqual(commands["10010"].get("name"),
                         "StationUpgradeVoucher1")
        self.assertIsNotNone(engine.state.state.base_reserve)

    def test_balanced_purchase_uses_current_price_after_shop_change(self):
        engine = DecisionEngine(clock=lambda: 0.0)
        payload = balanced_payload()
        payload["teamOur"]["roles"][0]["pos"] = {"x": 3, "y": 2}
        self.assertEqual(
            engine.decide(payload)["roleCommandMap"]["10010"]["action"],
            "move",
        )
        payload["roundNo"] = 2
        payload["teamOur"]["roles"][0]["pos"] = {"x": 4, "y": 1}
        payload["weaponShopList"][1]["price"] = 101
        payload["lastRoundRoleActionResults"] = {"10010": True}
        command = engine.decide(payload)["roleCommandMap"].get("10010", {})
        self.assertNotEqual(command.get("name"), "StationUpgradeVoucher1")

    def test_balanced_purchase_keeps_capacity_towers_and_urgent_wall(self):
        full = balanced_payload(items=("copper",) * 100)
        commands = DecisionEngine(clock=lambda: 0.0).decide(full)[
            "roleCommandMap"
        ].values()
        self.assertFalse(any(
            command.get("name") == "StationUpgradeVoucher1"
            for command in commands
        ))
        incomplete = balanced_payload()
        incomplete["teamOur"]["roles"].pop()
        commands = DecisionEngine(clock=lambda: 0.0).decide(incomplete)[
            "roleCommandMap"
        ].values()
        self.assertFalse(any(
            command.get("name") == "StationUpgradeVoucher1"
            for command in commands
        ))
        urgent = balanced_payload()
        urgent["teamOur"]["roles"].append(
            role(30000, "wall", 7, 9, health=5),
        )
        urgent["robot"]["roles"] = [{
            "id": 50000, "roleType": "smallRobot",
            "pos": {"x": 7, "y": 8}, "health": 30,
            "targetTeam": "challenger",
        }]
        commands = DecisionEngine(clock=lambda: 0.0).decide(urgent)[
            "roleCommandMap"
        ].values()
        self.assertFalse(any(
            command.get("name") == "StationUpgradeVoucher1"
            for command in commands
        ))

    def test_day_damage_releases_only_the_station_reserve_plan(self):
        payload = base_payload(
            round_no=131, worker_pos=(8, 9), gold=0, health=300,
            items=("StationUpgradeVoucher1",),
        )
        turn = Turn.load(payload)
        for reason, should_keep in (
            ("fund:StationUpgradeVoucher1:10020:10013", False),
            ("mine:copper", True),
            ("gunner:10020", True),
        ):
            with self.subTest(reason=reason):
                state = state_for(payload)
                state.base_reserve = BaseReserve(
                    10010, 10013, 1, "StationUpgradeVoucher1",
                )
                state.plans[10010] = PlanState(
                    10010, Pos(9, 9), reason, None, state.session_index,
                )
                prepare_base_reserve(
                    turn, state, frozenset(), lambda: 0.0, 1.0, 256,
                )
                self.assertIsNone(state.base_reserve)
                self.assertEqual(state.base_reserve_event, "day_damage_released")
                self.assertEqual(10010 in state.plans, should_keep)

        state = state_for(payload)
        state.base_reserve = BaseReserve(
            10010, 10013, 1, "StationUpgradeVoucher1",
        )
        prepare_base_reserve(
            turn, state, frozenset((10010,)), lambda: 0.0, 1.0, 256,
        )
        self.assertIsNone(state.base_reserve)
        self.assertEqual(state.base_reserve_event, "holder_task_reserved")

    def test_held_reserve_crosses_night_then_releases_only_if_day_base_damaged(self):
        for health, expect_use in ((1500, False), (300, True)):
            with self.subTest(health=health):
                engine = DecisionEngine(clock=lambda: 0.0)
                payload = base_payload(
                    round_no=71, worker_pos=(8, 9), gold=0,
                    items=("StationUpgradeVoucher1",),
                )
                payload["teamOur"]["teamId"] = f"cross-night-base-{health}"
                seed_reserve(engine, payload)
                engine.decide(payload)
                self.assertIsNotNone(engine.state.state.base_reserve)
                payload["roundNo"] = 130
                engine.decide(payload)
                self.assertIsNotNone(engine.state.state.base_reserve)
                payload["roundNo"] = 131
                payload["teamOur"]["roles"][1]["health"] = health
                command = engine.decide(payload)["roleCommandMap"].get("10010", {})
                self.assertEqual(command.get("action") == "use", expect_use)
                self.assertEqual(engine.state.state.base_reserve is None, expect_use)

    def test_damaged_base_buys_then_uses_via_ordinary_day_path(self):
        engine = DecisionEngine()
        payload = base_payload(health=1000)
        payload["teamOur"]["teamId"] = "damaged-ordinary-base-upgrade"
        first = engine.decide(payload)
        self.assertEqual(first["roleCommandMap"]["10010"]["action"], "buy")
        self.assertIsNone(engine.state.state.base_reserve)
        payload["roundNo"] = 2
        payload["teamOur"]["goldNum"] = 0
        payload["teamOur"]["roles"][0]["backpack"] = [
            "StationUpgradeVoucher1"
        ]
        payload["teamOur"]["roles"][0]["pos"] = {"x": 8, "y": 9}
        payload["lastRoundRoleActionResults"] = {"10010": True}
        second = engine.decide(payload)
        self.assertEqual(second["roleCommandMap"]["10010"]["action"], "use")
        self.assertIsNone(engine.state.state.base_reserve)

    def test_damaged_reserve_release_does_not_invent_blocked_route(self):
        payload = base_payload(
            round_no=131, worker_pos=(0, 0), gold=0, health=300,
            items=("StationUpgradeVoucher1",),
        )
        payload["teamOur"]["teamId"] = "blocked-damaged-reserve"
        payload["teamOur"]["roles"].extend(
            role(20100 + index, "wall", x, y, health=1000)
            for index, (x, y) in enumerate(((0, 1), (1, 0), (1, 1)))
        )
        engine = DecisionEngine(clock=lambda: 0.0)
        seed_reserve(engine, payload)
        command = engine.decide(payload)["roleCommandMap"].get("10010")
        self.assertIsNone(command)
        self.assertIsNone(engine.state.state.base_reserve)

    def test_reserve_skip_reason_names_existing_gates(self):
        other_funding = PlannedAction(
            ActionProposal(10010, 10010, {
                "action": "move", "targetPos": [{"x": 5, "y": 1}],
            }),
            plan_reason="fund:WeaponUpgradeVoucher1:10020:10020",
        )
        for label, change, candidates, expected in (
            ("night", {"roundNo": 71}, (), "night"),
            ("dusk", {"roundNo": 70}, (), "dusk"),
            ("damaged", {"health": 1000}, (), "station_damaged"),
            ("towers", {"removeTower": True}, (), "towers_incomplete"),
            ("low_cash", {"goldNum": 0}, (), "cash_insufficient"),
            ("no_route", {}, (), "route_or_priority_unavailable"),
            ("other", {}, (other_funding,), "other_funding_priority"),
        ):
            with self.subTest(label=label):
                payload = base_payload()
                if "roundNo" in change:
                    payload["roundNo"] = change["roundNo"]
                if "health" in change:
                    payload["teamOur"]["roles"][1]["health"] = change["health"]
                if "goldNum" in change:
                    payload["teamOur"]["goldNum"] = change["goldNum"]
                if change.get("removeTower"):
                    payload["teamOur"]["roles"].pop()
                state = state_for(payload)
                result = reserve_purchase_candidate(
                    Turn.load(payload), state, candidates, frozenset(),
                )
                self.assertEqual(result, candidates)
                self.assertEqual(state.base_reserve_reason, expected)

    def test_day_damage_releases_bought_reserve_for_immediate_use(self):
        # Break caught: an earlier night reserve hides a usable daytime voucher.
        engine = DecisionEngine()
        payload = base_payload()
        payload["teamOur"]["teamId"] = "damaged-held-base-reserve"
        first = engine.decide(payload)
        self.assertEqual(first["roleCommandMap"]["10010"], {
            "action": "buy", "name": "StationUpgradeVoucher1", "num": 1,
        })
        self.assertIsNotNone(engine.state.state.base_reserve)

        payload["roundNo"] = 2
        payload["teamOur"]["goldNum"] = 0
        payload["teamOur"]["roles"][0]["backpack"] = [
            "StationUpgradeVoucher1"
        ]
        payload["teamOur"]["roles"][0]["pos"] = {"x": 8, "y": 9}
        payload["teamOur"]["roles"][1]["health"] = 1000
        payload["lastRoundRoleActionResults"] = {"10010": True}
        second = engine.decide(payload)
        self.assertEqual(second["roleCommandMap"]["10010"], {
            "action": "use", "name": "StationUpgradeVoucher1",
            "targetPos": [{"x": 9, "y": 9}],
        })
        self.assertIsNone(engine.state.state.base_reserve)

    def test_reserve_does_not_stall_day_work_after_leaving_base(self):
        payload = base_payload(
            worker_pos=(8, 9), gold=0,
            items=("StationUpgradeVoucher1",),
        )
        engine = DecisionEngine()
        seed_reserve(engine, payload)
        moves = []
        for round_no in range(1, 5):
            payload["roundNo"] = round_no
            if round_no > 1:
                payload["lastRoundRoleActionResults"] = {"10010": True}
            command = engine.decide(payload)["roleCommandMap"].get("10010")
            self.assertIsNotNone(command, f"round {round_no} stalled")
            self.assertEqual(command["action"], "move")
            moves.append(command["targetPos"][0])
            payload["teamOur"]["roles"][0]["pos"] = command["targetPos"][0]
        self.assertEqual(moves, [
            {"x": 7, "y": 8}, {"x": 6, "y": 7},
            {"x": 5, "y": 6}, {"x": 4, "y": 5},
        ])

    def test_reserved_holder_can_do_other_day_work(self):
        payload = base_payload(
            worker_pos=(8, 9), gold=0,
            items=("StationUpgradeVoucher1",),
        )
        turn = Turn.load(payload)
        state = state_for(payload)
        state.base_reserve = BaseReserve(
            10010, 10013, 1, "StationUpgradeVoucher1",
        )
        actions = propose_economy(
            turn, state, clock=lambda: 0, deadline=1,
            max_expansions=256,
        )
        self.assertEqual(actions[0].plan_reason, "mine:copper")

    def test_medicine_precedes_reserved_base_upgrade(self):
        engine = DecisionEngine()
        payload = base_payload(
            round_no=71, worker_pos=(8, 9), gold=0, health=300,
            items=("StationUpgradeVoucher1", "Medicine"),
        )
        payload["teamOur"]["roles"][0]["health"] = 100
        seed_reserve(engine, payload)
        result = engine.decide(payload)
        self.assertEqual(result["roleCommandMap"]["10010"]["name"], "Medicine")

    def test_level_feedback_and_session_rewind_clear_reserve(self):
        engine = DecisionEngine()
        payload = base_payload()
        engine.decide(payload)
        self.assertIsNotNone(engine.state.state.base_reserve)
        payload["roundNo"] = 2
        payload["teamOur"]["roles"][0]["backpack"] = [
            "StationUpgradeVoucher1"
        ]
        payload["teamOur"]["roles"][1]["level"] = 2
        payload["teamOur"]["roles"][1]["health"] = 3000
        payload["lastRoundRoleActionResults"] = {"10010": True}
        engine.decide(payload)
        self.assertIsNone(engine.state.state.base_reserve)
        self.assertEqual(engine.state.state.base_reserve_event, "upgrade_observed")
        payload["roundNo"] = 1
        payload["lastRoundRoleActionResults"] = {}
        engine.decide(payload)
        self.assertIsNone(engine.state.state.base_reserve)
        self.assertEqual(engine.state.state.session_index, 2)

    def test_unmarked_night_voucher_keeps_existing_immediate_use(self):
        payload = base_payload(
            round_no=71, worker_pos=(8, 9), gold=0,
            items=("StationUpgradeVoucher1",),
        )
        result = DecisionEngine().decide(payload)
        self.assertEqual(result["roleCommandMap"]["10010"]["action"], "use")

    def test_existing_reserve_prevents_second_purchase(self):
        engine = DecisionEngine()
        payload = base_payload(
            worker_pos=(8, 9), gold=100,
            items=("StationUpgradeVoucher1",),
        )
        payload["teamOur"]["roles"].insert(1, role(10011, "worker", 6, 1))
        seed_reserve(engine, payload)
        result = engine.decide(payload)
        self.assertFalse(any(
            command.get("action") == "buy"
            and command.get("name") == "StationUpgradeVoucher1"
            for command in result["roleCommandMap"].values()
        ))

    def test_failed_purchase_clears_reserve_without_retry_same_day(self):
        engine = DecisionEngine()
        payload = base_payload()
        self.assertEqual(engine.decide(payload)["roleCommandMap"]["10010"]["action"], "buy")
        payload["roundNo"] = 2
        payload["lastRoundRoleActionResults"] = {"10010": False}
        result = engine.decide(payload)
        self.assertIsNone(engine.state.state.base_reserve)
        self.assertFalse(any(
            command.get("action") == "buy"
            and command.get("name") == "StationUpgradeVoucher1"
            for command in result["roleCommandMap"].values()
        ))

    def test_full_health_weak_visible_threat_does_not_spend_reserve(self):
        robot = {
            "id": 30001, "pos": {"x": 10, "y": 7},
            "roleType": "smallRobot", "health": 30,
            "targetTeam": "defender", "abnormalState": "",
        }
        payload = base_payload(
            round_no=71, worker_pos=(8, 9), gold=0,
            items=("StationUpgradeVoucher1",), robots=(robot,),
        )
        turn = Turn.load(payload)
        state = state_for(payload)
        state.base_reserve = BaseReserve(
            10010, 10013, 1, "StationUpgradeVoucher1",
        )
        self.assertIsNone(propose_base_upgrade(
            turn, state, unavailable_role_ids=frozenset(),
            protected_role_ids=frozenset(), clock=lambda: 0,
            deadline=1, max_expansions=256,
        ))

    def test_damaged_base_without_rate_uses_adjacent_reserve(self):
        payload = base_payload(
            round_no=71, worker_pos=(8, 9), gold=0, health=1000,
            items=("StationUpgradeVoucher1",),
        )
        turn = Turn.load(payload)
        state = state_for(payload)
        state.base_reserve = BaseReserve(
            10010, 10013, 1, "StationUpgradeVoucher1",
        )
        candidate = propose_base_upgrade(
            turn, state, unavailable_role_ids=frozenset(),
            protected_role_ids=frozenset((10010,)), clock=lambda: 0,
            deadline=1, max_expansions=256,
        )
        self.assertEqual(candidate.proposal.command["action"], "use")
        self.assertEqual(state.base_reserve_reason, "use_under_uncertainty")

    def test_night_move_requires_route_margin_and_cannot_pull_only_gunner(self):
        payload = base_payload(
            round_no=71, worker_pos=(7, 9), gold=0, health=1000,
            items=("StationUpgradeVoucher1",),
        )
        turn = Turn.load(payload)
        state = state_for(payload)
        state.base_reserve = BaseReserve(
            10010, 10013, 1, "StationUpgradeVoucher1",
        )
        state.base_recent_drops = (400,)
        kwargs = dict(
            unavailable_role_ids=frozenset(), clock=lambda: 0,
            deadline=1, max_expansions=256,
        )
        candidate = propose_base_upgrade(
            turn, state, protected_role_ids=frozenset(), **kwargs,
        )
        self.assertEqual(candidate.proposal.command["action"], "move")
        self.assertIsNone(propose_base_upgrade(
            turn, state, protected_role_ids=frozenset((10010,)), **kwargs,
        ))
        state.base_recent_drops = (600,)
        self.assertIsNone(propose_base_upgrade(
            turn, state, protected_role_ids=frozenset(), **kwargs,
        ))

    def test_day_purchase_reserves_one_voucher_and_does_not_use_at_full_health(self):
        engine = DecisionEngine()
        payload = base_payload()
        first = engine.decide(payload)
        self.assertEqual(first["roleCommandMap"]["10010"]["action"], "buy")
        self.assertIsNotNone(engine.state.state.base_reserve)

        payload["roundNo"] = 2
        payload["teamOur"]["goldNum"] = 0
        payload["teamOur"]["roles"][0]["backpack"] = [
            "StationUpgradeVoucher1"
        ]
        payload["teamOur"]["roles"][0]["pos"] = {"x": 8, "y": 9}
        payload["lastRoundRoleActionResults"] = {"10010": True}
        second = engine.decide(payload)
        self.assertNotEqual(
            second["roleCommandMap"].get("10010", {}).get("action"), "use",
        )

    def test_night_upgrade_waits_for_weak_threat_then_uses_under_large_damage(self):
        engine = DecisionEngine()
        payload = base_payload(round_no=71, worker_pos=(8, 9), gold=0,
                               items=("StationUpgradeVoucher1",))
        seed_reserve(engine, payload)
        engine.decide(payload)
        self.assertIsNotNone(engine.state.state.base_reserve)

        payload["roundNo"] = 72
        payload["teamOur"]["roles"][1]["health"] = 350
        result = engine.decide(payload)
        self.assertEqual(result["roleCommandMap"]["10010"]["action"], "use")

    def test_wrong_level_and_failed_use_release_reserve(self):
        engine = DecisionEngine()
        payload = base_payload(round_no=71, worker_pos=(8, 9), gold=0,
                               health=350,
                               items=("StationUpgradeVoucher1",))
        seed_reserve(engine, payload)
        engine.decide(payload)
        self.assertIsNotNone(engine.state.state.base_reserve)
        payload["roundNo"] = 72
        payload["teamOur"]["roles"][1]["health"] = 100
        self.assertEqual(
            engine.decide(payload)["roleCommandMap"]["10010"]["action"],
            "use",
        )
        payload["roundNo"] = 73
        payload["lastRoundRoleActionResults"] = {"10010": False}
        engine.decide(payload)
        self.assertIsNone(engine.state.state.base_reserve)

        payload["roundNo"] = 74
        payload["teamOur"]["roles"][1]["level"] = 3
        engine.decide(payload)
        self.assertIsNone(engine.state.state.base_reserve)


if __name__ == "__main__":
    unittest.main()
