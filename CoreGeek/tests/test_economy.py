import copy
import importlib
import json
import unittest
from pathlib import Path

from agent.actions import ActionAllocator, ActionProposal
from agent.brain import DecisionEngine
from agent.protocol import Pos, Turn
from agent.state import (
    CompletedAction,
    PendingAction,
    PlanState,
    StateStore,
    request_fingerprint,
)


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def role(unit_id, kind, x, y, *, health=220, level=1, items=(), cooldown=0):
    return {
        "id": unit_id,
        "pos": {"x": x, "y": y},
        "roleType": kind,
        "health": health,
        "attackPower": 10 if kind in ("gatling", "railgun") else 20,
        "attackRange": 0,
        "backPackCapability": 100 if kind == "worker" else 0,
        "backpack": list(items),
        "level": level,
        "cooldown": cooldown,
    }


def economy_payload(*, round_no=1, worker_pos=(2, 1), items=(), gold=0):
    payload = load_fixture()
    payload["roundNo"] = round_no
    payload["mapInfo"].update({
        "width": 12,
        "height": 12,
        "zones": [
            {"pos": {"x": 2, "y": 2}, "neutralType": "copper"},
            {"pos": {"x": 4, "y": 2}, "neutralType": "vendor"},
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ],
    })
    payload["teamOur"].update({
        "teamId": "economy-chain",
        "goldNum": gold,
        "roles": [
            role(10010, "worker", *worker_pos, items=items),
            role(10013, "station", 9, 9, health=1500),
            role(10020, "gatling", 8, 2, health=1000),
            role(10030, "railgun", 8, 8, health=1000),
            role(10040, "rocket", 11, 8, health=1000),
        ],
    })
    payload["teamEnemy"]["roles"] = []
    payload["robot"]["roles"] = []
    payload["vendorShopList"] = [
        {"name": "copper", "price": 100},
    ]
    payload["weaponShopList"] = [
        {"name": "WeaponUpgradeVoucher1", "price": 100},
        {"name": "Medicine", "price": 10},
        {"name": "WallFixer", "price": 10},
    ]
    return payload


def state_for(payload):
    store = StateStore()
    turn = Turn.load(payload)
    store.observe(turn, payload, request_fingerprint(payload))
    return store.state


class EconomyTests(unittest.TestCase):
    def test_full_backpack_and_insufficient_gold_do_not_advance_chain(self):
        # Break caught: collection/purchase is issued despite current hard limits.
        economy = importlib.import_module("agent.economy")
        full = economy_payload(worker_pos=(2, 1), items=("junk",) * 100)
        full_turn = Turn.load(full)
        full_candidates = economy.propose_economy(
            full_turn, state_for(full), clock=lambda: 0.0, deadline=1.0,
            max_expansions=64,
        )
        self.assertFalse(any(
            item.proposal.command["action"] == "collect"
            for item in full_candidates
        ))

        poor = economy_payload(worker_pos=(5, 2), gold=99)
        poor["mapInfo"]["zones"] = [
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        poor_turn = Turn.load(poor)
        poor_candidates = economy.propose_economy(
            poor_turn, state_for(poor), clock=lambda: 0.0, deadline=1.0,
            max_expansions=64,
        )
        self.assertFalse(any(
            item.proposal.command["action"] == "buy"
            for item in poor_candidates
        ))

    def test_medicine_wall_fixer_and_station_upgrade_use_current_inventory(self):
        # Break caught: maintenance effects are assumed without held items/targets.
        economy = importlib.import_module("agent.economy")

        medicine = economy_payload(items=("Medicine",))
        medicine["teamOur"]["roles"][0]["health"] = 100
        medicine_actions = economy.propose_economy(
            Turn.load(medicine), state_for(medicine),
            clock=lambda: 0.0, deadline=1.0, max_expansions=64,
        )
        self.assertEqual(medicine_actions[0].proposal.command, {
            "action": "use", "name": "Medicine",
        })
        self.assertTrue(ActionAllocator(Turn.load(medicine)).try_add(
            medicine_actions[0].proposal
        ))

        fixer = economy_payload(worker_pos=(7, 8), items=("WallFixer",))
        fixer["teamOur"]["roles"].append(
            role(40000, "wall", 7, 9, health=500),
        )
        fixer_actions = economy.propose_economy(
            Turn.load(fixer), state_for(fixer),
            clock=lambda: 0.0, deadline=1.0, max_expansions=64,
        )
        self.assertEqual(fixer_actions[0].proposal.command, {
            "action": "use",
            "name": "WallFixer",
            "targetPos": [{"x": 7, "y": 9}],
        })
        self.assertTrue(ActionAllocator(Turn.load(fixer)).try_add(
            fixer_actions[0].proposal
        ))

        upgrade = economy_payload(
            worker_pos=(8, 10), items=("StationUpgradeVoucher1",),
        )
        upgrade_actions = economy.propose_economy(
            Turn.load(upgrade), state_for(upgrade),
            clock=lambda: 0.0, deadline=1.0, max_expansions=64,
        )
        self.assertEqual(upgrade_actions[0].proposal.command, {
            "action": "use",
            "name": "StationUpgradeVoucher1",
            "targetPos": [{"x": 9, "y": 9}],
        })
        self.assertTrue(ActionAllocator(Turn.load(upgrade)).try_add(
            upgrade_actions[0].proposal
        ))

    def test_buy_cost_comes_from_current_shop_not_caller_hint(self):
        # Break caught: omitted gold_cost allows two purchases with only 10 gold.
        payload = economy_payload(worker_pos=(5, 2), gold=10)
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 5, 1),
        )
        turn = Turn.load(payload)
        allocator = ActionAllocator(turn)

        first = allocator.try_add(ActionProposal(
            command_owner_id=10010,
            actor_id=10010,
            command={"action": "buy", "name": "Medicine", "num": 1},
        ))
        second = allocator.try_add(ActionProposal(
            command_owner_id=10012,
            actor_id=10012,
            command={"action": "buy", "name": "Medicine", "num": 1},
        ))

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(allocator.gold_remaining, 0)

    def test_continuous_collect_sell_buy_arrive_and_use_chain(self):
        # Break caught: expected income or purchases advance before observation.
        engine = DecisionEngine()

        first = economy_payload()
        response = engine.decide(first)
        self.assertEqual(
            response["roleCommandMap"]["10010"],
            {"action": "collect", "targetPos": [{"x": 2, "y": 2}]},
        )

        second = economy_payload(round_no=2, worker_pos=(3, 2), items=("copper",))
        second["lastRoundRoleActionResults"] = {"10010": True}
        response = engine.decide(second)
        self.assertEqual(
            response["roleCommandMap"]["10010"],
            {"action": "sell", "name": "copper", "num": 1},
        )

        third = economy_payload(round_no=3, worker_pos=(5, 2), gold=100)
        third["lastRoundRoleActionResults"] = {"10010": True}
        response = engine.decide(third)
        self.assertEqual(
            response["roleCommandMap"]["10010"],
            {"action": "buy", "name": "WeaponUpgradeVoucher1", "num": 1},
        )

        fourth = economy_payload(
            round_no=4,
            worker_pos=(5, 2),
            items=("WeaponUpgradeVoucher1",),
        )
        fourth["lastRoundRoleActionResults"] = {"10010": True}
        response = engine.decide(fourth)
        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")
        self.assertEqual(
            engine.state.state.plans[10010].target, Pos(8, 2)
        )
        self.assertEqual(
            engine.state.state.plans[10010].reason,
            "use:WeaponUpgradeVoucher1:10020",
        )

        fifth = economy_payload(
            round_no=5,
            worker_pos=(7, 2),
            items=("WeaponUpgradeVoucher1",),
        )
        fifth["lastRoundRoleActionResults"] = {"10010": True}
        response = engine.decide(fifth)
        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "use",
            "name": "WeaponUpgradeVoucher1",
            "targetPos": [{"x": 8, "y": 2}],
        })

    def test_wrong_upgrade_voucher_level_is_not_used(self):
        # Break caught: a level-1 voucher is wasted on a level-2 weapon.
        economy = importlib.import_module("agent.economy")
        payload = economy_payload(
            worker_pos=(7, 2), items=("WeaponUpgradeVoucher1",),
        )
        for entry in payload["teamOur"]["roles"]:
            if entry["roleType"] in ("gatling", "railgun", "rocket"):
                entry["level"] = 2
        turn = Turn.load(payload)

        candidates = economy.propose_economy(
            turn,
            state_for(payload),
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=64,
        )

        self.assertFalse(any(
            candidate.proposal.command.get("action") == "use"
            and candidate.proposal.command.get("name")
            == "WeaponUpgradeVoucher1"
            for candidate in candidates
        ))

    def test_dead_voucher_holder_cannot_spend_preserved_backpack(self):
        # Break caught: a dead worker's retained backpack is treated as usable.
        economy = importlib.import_module("agent.economy")
        payload = economy_payload(
            worker_pos=(7, 2), items=("WeaponUpgradeVoucher1",),
        )
        payload["teamOur"]["roles"][0]["health"] = 0

        candidates = economy.propose_economy(
            Turn.load(payload), state_for(payload),
            clock=lambda: 0.0, deadline=1.0, max_expansions=64,
        )

        self.assertFalse(any(
            item.proposal.actor_id == 10010 for item in candidates
        ))

    def test_failed_build_target_is_not_repeated(self):
        # Break caught: an invalid experimental build cell is retried forever.
        economy = importlib.import_module("agent.economy")
        payload = economy_payload(worker_pos=(7, 8), gold=75)
        payload["teamOur"]["roles"] = [
            role(10010, "worker", 7, 8),
            role(10013, "station", 9, 9, health=1500),
        ]
        turn = Turn.load(payload)
        state = state_for(payload)
        failed = Pos(8, 8)
        state.action_history.append(CompletedAction(
            PendingAction(0, 10010, 10010, "build", failed), False,
        ))

        candidates = economy.propose_economy(
            turn,
            state,
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=64,
        )

        build_targets = [
            candidate.proposal.destination
            for candidate in candidates
            if candidate.proposal.command["action"] in ("build", "move")
            and candidate.plan_reason
            and candidate.plan_reason.startswith("build:")
        ]
        self.assertTrue(build_targets)
        self.assertNotIn(failed, build_targets)

    def test_disappeared_mine_invalidates_long_range_goal(self):
        # Break caught: a worker keeps walking toward a mine absent this round.
        economy = importlib.import_module("agent.economy")
        first = economy_payload(worker_pos=(0, 0))
        state = state_for(first)
        state.plans[10010] = PlanState(
            10010, Pos(2, 2), "mine:copper", None, state.session_index,
        )
        changed = copy.deepcopy(first)
        changed["mapInfo"]["zones"] = []

        candidates = economy.propose_economy(
            Turn.load(changed), state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=64,
        )

        self.assertFalse(any(
            item.plan_target == Pos(2, 2) for item in candidates
        ))

    def test_wall_is_only_proposed_when_explicitly_needed(self):
        # Break caught: unproven wall value blocks the three-tower economy.
        economy = importlib.import_module("agent.economy")
        payload = economy_payload(worker_pos=(6, 9), items=("stone",))
        payload["teamOur"]["roles"] = [
            role(10010, "worker", 6, 9, items=("stone",)),
            role(10013, "station", 9, 9, health=1500),
            role(10020, "gatling", 8, 8, health=1000),
            role(10030, "railgun", 9, 8, health=1000),
            role(10040, "rocket", 10, 8, health=1000),
        ]
        turn = Turn.load(payload)

        normal = economy.propose_economy(
            turn, state_for(payload), clock=lambda: 0.0, deadline=1.0,
            max_expansions=64,
        )
        needed = economy.propose_economy(
            turn, state_for(payload), clock=lambda: 0.0, deadline=1.0,
            max_expansions=64, need_wall=True,
        )

        self.assertFalse(any(
            item.proposal.command.get("name") == "wall" for item in normal
        ))
        self.assertTrue(any(
            item.proposal.command.get("name") == "wall" for item in needed
        ))

    def test_preserved_mine_plan_rechecks_capacity_and_sells_instead(self):
        # Break caught: the old target outranks a newly full backpack.
        engine = DecisionEngine()
        first = economy_payload(worker_pos=(0, 0))
        response = engine.decide(first)
        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")
        self.assertTrue(
            engine.state.state.plans[10010].reason.startswith("mine:")
        )

        second = economy_payload(
            round_no=2,
            worker_pos=(1, 1),
            items=("copper",) * 100,
        )
        second["lastRoundRoleActionResults"] = {"10010": True}
        response = engine.decide(second)

        self.assertNotEqual(
            response["roleCommandMap"]["10010"]["action"], "collect"
        )
        self.assertEqual(engine.state.state.plans[10010].reason, "vendor")

    def test_shop_plan_rechecks_gold_lost_between_rounds(self):
        # Break caught: a stale purchase is retried after shared gold disappears.
        engine = DecisionEngine()
        first = economy_payload(worker_pos=(0, 0), gold=100)
        first["mapInfo"]["zones"] = [
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        response = engine.decide(first)
        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")
        self.assertEqual(
            engine.state.state.plans[10010].reason,
            "shop:WeaponUpgradeVoucher1",
        )

        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["teamOur"]["goldNum"] = 0
        second["teamOur"]["roles"][0]["pos"] = {"x": 5, "y": 2}
        second["lastRoundRoleActionResults"] = {"10010": True}
        response = engine.decide(second)

        command = response["roleCommandMap"].get("10010", {})
        self.assertNotEqual(command.get("action"), "buy")
        plan = engine.state.state.plans.get(10010)
        self.assertTrue(plan is None or plan.reason != "shop:WeaponUpgradeVoucher1")

    def test_build_plan_stops_when_third_tower_appears(self):
        # Break caught: an en-route worker attempts a fourth tower next round.
        engine = DecisionEngine()
        first = economy_payload(worker_pos=(0, 0), gold=25)
        first["teamOur"]["roles"] = [
            role(10010, "worker", 0, 0),
            role(10013, "station", 9, 9, health=1500),
            role(10020, "gatling", 8, 2, health=1000),
            role(10030, "railgun", 8, 8, health=1000),
        ]
        response = engine.decide(first)
        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")
        self.assertTrue(
            engine.state.state.plans[10010].reason.startswith("build:")
        )

        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["teamOur"]["roles"][0]["pos"] = {"x": 1, "y": 1}
        second["teamOur"]["roles"].append(
            role(10040, "rocket", 10, 8, health=1000)
        )
        second["lastRoundRoleActionResults"] = {"10010": True}
        response = engine.decide(second)

        self.assertFalse(any(
            command.get("action") == "build"
            and command.get("name") in ("gatling", "railgun", "rocket")
            for command in response["roleCommandMap"].values()
        ))
        plan = engine.state.state.plans.get(10010)
        self.assertTrue(plan is None or not plan.reason.startswith("build:"))


if __name__ == "__main__":
    unittest.main()
