import unittest

from agent.brain import DecisionEngine
from agent.economy import propose_base_upgrade, propose_economy
from agent.protocol import Turn
from agent.state import BaseReserve, request_fingerprint
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


class BaseUpgradeTests(unittest.TestCase):
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
