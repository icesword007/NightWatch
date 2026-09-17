import copy
import importlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import agent.economy as economy
from agent.actions import ActionAllocator, ActionProposal
from agent.brain import DecisionEngine
from agent.grid import PathResult
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


def with_completed_wall_line(payload):
    from agent.layout import plan_defense_layout

    targets = plan_defense_layout(Turn.load(payload)).wall_targets
    payload["teamOur"]["roles"].extend(
        role(10100 + offset, "wall", target.x, target.y, health=1000)
        for offset, target in enumerate(targets)
    )
    return payload


class EconomyTests(unittest.TestCase):
    def test_wall_upgrade_purchase_use_and_level_feedback_chain(self):
        payload = economy_payload(worker_pos=(11, 7), gold=20)
        payload["teamOur"]["teamId"] = "wall-upgrade-chain"
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 10, "y": 7}, "neutralType": "weaponShop"},
        ]
        with_completed_wall_line(payload)
        payload["weaponShopList"] = [
            {"name": "WallUpgradeVoucher1", "price": 20},
        ]
        engine = DecisionEngine()

        first = engine.decide(payload)
        self.assertEqual(first["roleCommandMap"]["10010"], {
            "action": "buy", "name": "WallUpgradeVoucher1", "num": 1,
        })
        target_id = economy._plan_use_target_id(
            engine.state.state.plans[10010]
        )
        target = next(
            entry for entry in payload["teamOur"]["roles"]
            if entry["id"] == target_id
        )

        payload["roundNo"] = 2
        payload["teamOur"]["goldNum"] = 0
        payload["teamOur"]["roles"][0]["backpack"] = [
            "WallUpgradeVoucher1"
        ]
        payload["teamOur"]["roles"][0]["pos"] = {
            "x": target["pos"]["x"] - 1, "y": target["pos"]["y"],
        }
        payload["lastRoundRoleActionResults"] = {"10010": True}
        second = engine.decide(payload)
        self.assertEqual(second["roleCommandMap"]["10010"], {
            "action": "use", "name": "WallUpgradeVoucher1",
            "targetPos": [target["pos"]],
        })

        payload["roundNo"] = 3
        payload["teamOur"]["roles"][0]["backpack"] = []
        target["level"] = 2
        payload["lastRoundRoleActionResults"] = {"10010": True}
        third = engine.decide(payload)
        self.assertFalse(any(
            command.get("name") == "WallUpgradeVoucher1"
            for command in third["roleCommandMap"].values()
        ))

    def test_wall_upgrade_candidates_follow_existing_investments(self):
        payload = economy_payload()
        with_completed_wall_line(payload)
        payload["weaponShopList"].append(
            {"name": "WallUpgradeVoucher1", "price": 20},
        )
        turn = Turn.load(payload)
        state = state_for(payload)
        from agent.layout import ensure_defense_layout
        ensure_defense_layout(turn, state)
        context = economy.RouteSearchContext(
            {}, wall_upgrade_targets=state.layout_wall_targets,
        )
        token = economy._ROUTE_SEARCH_CONTEXT.set(context)
        try:
            candidates = economy._purchase_candidates(turn, turn.workers()[0])
        finally:
            economy._ROUTE_SEARCH_CONTEXT.reset(token)

        self.assertLess(
            candidates.index("WeaponUpgradeVoucher1"),
            candidates.index("WallUpgradeVoucher1"),
        )

    def test_wall_upgrade_purchase_waits_for_towers_and_blueprint_walls(self):
        payload = economy_payload(gold=20)
        payload["teamOur"]["roles"].append(
            role(10050, "wall", 5, 3, health=1000),
        )
        payload["weaponShopList"] = [
            {"name": "WallUpgradeVoucher1", "price": 20},
        ]

        response = DecisionEngine().decide(payload)

        self.assertFalse(any(
            command.get("name") == "WallUpgradeVoucher1"
            for command in response["roleCommandMap"].values()
        ))

    def test_wall_upgrade_level_two_moves_buys_uses_then_stops_at_level_three(self):
        payload = economy_payload(worker_pos=(5, 2), gold=30)
        payload["teamOur"]["teamId"] = "wall-upgrade-level-two-chain"
        with_completed_wall_line(payload)
        for entry in payload["teamOur"]["roles"]:
            if entry["roleType"] == "wall":
                entry["level"] = 2
                entry["health"] = 1500
        payload["weaponShopList"] = [
            {"name": "WallUpgradeVoucher2", "price": 30},
        ]
        engine = DecisionEngine()
        actions = []
        upgraded = None
        for _ in range(30):
            response = engine.decide(payload)
            command = response["roleCommandMap"]["10010"]
            actions.append(command["action"])
            worker = payload["teamOur"]["roles"][0]
            if command["action"] == "move":
                worker["pos"] = copy.deepcopy(command["targetPos"][0])
            elif command["action"] == "buy":
                payload["teamOur"]["goldNum"] = 0
                worker["backpack"] = ["WallUpgradeVoucher2"]
            elif command["action"] == "use":
                upgraded = next(
                    entry for entry in payload["teamOur"]["roles"]
                    if entry["pos"] == command["targetPos"][0]
                )
                upgraded["level"] = 3
                upgraded["backpack"] = []
                worker["backpack"] = []
                break
            payload["roundNo"] += 1
            payload["lastRoundRoleActionResults"] = {"10010": True}
        self.assertIn("move", actions)
        self.assertIn("buy", actions)
        self.assertEqual(actions[-1], "use")
        payload["roundNo"] += 1
        payload["teamOur"]["goldNum"] = 30
        payload["lastRoundRoleActionResults"] = {"10010": True}
        response = engine.decide(payload)
        self.assertFalse(any(
            command.get("name") == "WallUpgradeVoucher2"
            for command in response["roleCommandMap"].values()
        ))

    def test_two_workers_buy_wall_vouchers_for_distinct_targets(self):
        payload = economy_payload(worker_pos=(5, 1), gold=40)
        with_completed_wall_line(payload)
        payload["teamOur"]["roles"].insert(1, role(10012, "worker", 5, 3))
        payload["weaponShopList"] = [
            {"name": "WallUpgradeVoucher1", "price": 20},
        ]
        engine = DecisionEngine()
        response = engine.decide(payload)
        self.assertEqual(sum(
            command.get("action") == "buy"
            for command in response["roleCommandMap"].values()
        ), 2)
        self.assertEqual(len({
            economy._plan_use_target_id(engine.state.state.plans[role_id])
            for role_id in (10010, 10012)
        }), 2)

    def test_held_wall_voucher_uses_despite_new_purchase_gate(self):
        for round_no in (1, 71):
            with self.subTest(round_no=round_no):
                payload = economy_payload(
                    round_no=round_no, worker_pos=(7, 8),
                    items=("WallUpgradeVoucher1",),
                )
                payload["teamOur"]["roles"].append(
                    role(10050, "wall", 7, 9, health=1000),
                )
                payload["weaponShopList"] = [
                    {"name": "WallUpgradeVoucher1", "price": 20},
                ]
                response = DecisionEngine().decide(payload)
                self.assertEqual(response["roleCommandMap"]["10010"]["action"], "use")

    def test_wall_upgrade_gate_keeps_session_layout_when_wall_is_occupied(self):
        first = economy_payload(gold=0)
        first["teamOur"]["teamId"] = "wall-upgrade-stable-layout"
        first["teamEnemy"]["roles"] = [
            role(90013, "station", 1, 9, health=1500),
        ]
        with_completed_wall_line(first)
        engine = DecisionEngine()
        engine.decide(first)
        fixed_targets = engine.state.state.layout_wall_targets

        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["teamOur"]["goldNum"] = 20
        second["weaponShopList"] = [
            {"name": "WallUpgradeVoucher1", "price": 20},
        ]
        missing = fixed_targets[0]
        second["teamOur"]["roles"] = [
            entry for entry in second["teamOur"]["roles"]
            if not (
                entry["roleType"] == "wall"
                and entry["pos"] == {"x": missing.x, "y": missing.y}
            )
        ]
        second["teamEnemy"]["roles"] = [
            role(90040, "rocket", missing.x, missing.y, health=1000),
        ]

        response = engine.decide(second)

        self.assertEqual(engine.state.state.layout_wall_targets, fixed_targets)
        self.assertFalse(any(
            command.get("name") == "WallUpgradeVoucher1"
            for command in response["roleCommandMap"].values()
        ))
    def test_two_workers_reserve_distinct_upgrade_targets(self):
        payload = economy_payload(worker_pos=(5, 1), gold=200)
        payload["teamOur"]["roles"].insert(1, role(10012, "worker", 5, 3))
        engine = DecisionEngine()

        response = engine.decide(payload)

        self.assertEqual(sum(
            command.get("action") == "buy"
            for command in response["roleCommandMap"].values()
        ), 2)
        targets = {
            economy._plan_use_target_id(engine.state.state.plans[role_id])
            for role_id in (10010, 10012)
        }
        self.assertEqual(len(targets), 2)

    def test_two_held_vouchers_do_not_use_same_building(self):
        payload = economy_payload(
            worker_pos=(7, 1), items=("WeaponUpgradeVoucher1",),
        )
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 7, 3,
                    items=("WeaponUpgradeVoucher1",)),
        )

        response = DecisionEngine().decide(payload)

        uses = [
            command["targetPos"][0]
            for command in response["roleCommandMap"].values()
            if command.get("action") == "use"
        ]
        self.assertEqual(len(uses), len({(p["x"], p["y"]) for p in uses}))

    def test_duplicate_existing_plans_choose_stable_owner(self):
        payload = economy_payload(
            worker_pos=(7, 1), items=("WeaponUpgradeVoucher1",),
        )
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 7, 3,
                    items=("WeaponUpgradeVoucher1",)),
        )
        state = state_for(payload)
        for role_id in (10010, 10012):
            state.plans[role_id] = PlanState(
                role_id, Pos(8, 2),
                "use:WeaponUpgradeVoucher1:10020", 70,
                state.session_index,
            )

        actions = economy.propose_economy(
            Turn.load(payload), state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=64,
        )
        targets = {
            action.proposal.actor_id: economy._plan_use_target_id(action)
            for action in actions if action.proposal.actor_id in (10010, 10012)
        }
        self.assertEqual(targets[10010], 10020)
        self.assertNotEqual(targets[10012], 10020)

    def test_night_two_held_vouchers_use_target_once(self):
        payload = economy_payload(
            round_no=71, worker_pos=(7, 1),
            items=("WeaponUpgradeVoucher1",),
        )
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 7, 3,
                    items=("WeaponUpgradeVoucher1",)),
        )
        response = DecisionEngine().decide(payload)
        targets = [
            tuple(command["targetPos"][0].values())
            for command in response["roleCommandMap"].values()
            if command.get("action") == "use"
        ]
        self.assertEqual(len(targets), len(set(targets)))

    def test_existing_joint_buyer_remains_owner(self):
        payload = economy_payload(
            round_no=30, worker_pos=(3, 1), items=("copper",) * 6,
        )
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 3, 3, items=("copper",) * 6),
        )
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        state = state_for(payload)
        state.plans[10010] = PlanState(
            10010, Pos(4, 2),
            "fund:WeaponUpgradeVoucher1:10030:10020:joint:10012",
            70, state.session_index,
        )
        state.plans[10012] = PlanState(
            10012, Pos(4, 2),
            "fund:WeaponUpgradeVoucher1:10020:10020:joint:10012",
            70, state.session_index,
        )

        actions = economy.propose_economy(
            Turn.load(payload), state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=64,
        )

        self.assertTrue(all(
            ":joint:10012" in (action.plan_reason or "")
            for action in actions if action.proposal.actor_id in (10010, 10012)
        ))

    def test_only_one_target_starts_only_one_purchase(self):
        payload = economy_payload(worker_pos=(5, 1), gold=200)
        payload["teamOur"]["roles"].insert(1, role(10012, "worker", 5, 3))
        for entry in payload["teamOur"]["roles"]:
            if entry["id"] in (10030, 10040):
                entry["level"] = 2

        response = DecisionEngine().decide(payload)

        self.assertEqual(sum(
            command.get("action") == "buy"
            for command in response["roleCommandMap"].values()
        ), 1)

    def test_two_wall_fixers_do_not_repair_same_wall(self):
        payload = economy_payload(worker_pos=(7, 8), items=("WallFixer",))
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 7, 7, items=("WallFixer",)),
        )
        payload["teamOur"]["roles"].append(
            role(10050, "wall", 7, 9, health=100),
        )

        response = DecisionEngine().decide(payload)

        self.assertEqual(sum(
            command.get("action") == "use"
            and command.get("name") == "WallFixer"
            for command in response["roleCommandMap"].values()
        ), 1)

    @staticmethod
    def _block_weapon(payload, weapon_id):
        weapon = next(
            entry for entry in payload["teamOur"]["roles"]
            if entry["id"] == weapon_id
        )
        x, y = weapon["pos"]["x"], weapon["pos"]["y"]
        next_id = 49000
        occupied = {
            (entry["pos"]["x"], entry["pos"]["y"])
            for entry in payload["teamOur"]["roles"]
        }
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                pos = (x + dx, y + dy)
                if (dx or dy) and pos not in occupied:
                    payload["teamOur"]["roles"].append(
                        role(next_id, "wall", *pos, health=1000)
                    )
                    next_id += 1

    def test_held_voucher_skips_blocked_first_matching_weapon(self):
        payload = economy_payload(
            worker_pos=(7, 8), items=("WeaponUpgradeVoucher1",),
        )
        payload["teamOur"]["teamId"] = "investment-held-second-target"
        self._block_weapon(payload, 10020)

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "use",
            "name": "WeaponUpgradeVoucher1",
            "targetPos": [{"x": 8, "y": 8}],
        })

    def test_affordable_voucher_skips_blocked_first_matching_weapon(self):
        payload = economy_payload(worker_pos=(5, 2), gold=100)
        payload["teamOur"]["teamId"] = "investment-buy-second-target"
        self._block_weapon(payload, 10020)

        engine = DecisionEngine()
        response = engine.decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "buy",
            "name": "WeaponUpgradeVoucher1",
            "num": 1,
        })
        self.assertIn(":10030", engine.state.state.plans[10010].reason)

    def test_active_voucher_plan_keeps_target_when_unit_order_changes(self):
        payload = economy_payload(
            worker_pos=(7, 8), items=("WeaponUpgradeVoucher1",),
        )
        payload["teamOur"]["teamId"] = "investment-stable-target"
        payload["teamOur"]["roles"] = [
            payload["teamOur"]["roles"][0],
            payload["teamOur"]["roles"][1],
            payload["teamOur"]["roles"][3],
            payload["teamOur"]["roles"][2],
            payload["teamOur"]["roles"][4],
        ]
        state = state_for(payload)
        state.plans[10010] = PlanState(
            10010, Pos(8, 8),
            "fund:WeaponUpgradeVoucher1:10030:10030",
            70, state.session_index,
        )
        payload["teamOur"]["roles"][2:4] = reversed(
            payload["teamOur"]["roles"][2:4]
        )

        candidates = economy.propose_economy(
            Turn.load(payload), state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=64,
        )
        action = next(
            candidate for candidate in candidates
            if candidate.proposal.actor_id == 10010
        )

        self.assertEqual(action.plan_reason, "use:WeaponUpgradeVoucher1:10030")
        self.assertEqual(action.proposal.command["targetPos"], [{"x": 8, "y": 8}])

    def test_held_voucher_prefers_immediate_complete_second_target(self):
        payload = economy_payload(
            worker_pos=(7, 8), items=("WeaponUpgradeVoucher1",),
        )
        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "use", "name": "WeaponUpgradeVoucher1",
            "targetPos": [{"x": 8, "y": 8}],
        })

    def test_confirmed_blocked_committed_target_reselects(self):
        payload = economy_payload(
            worker_pos=(7, 8), items=("WeaponUpgradeVoucher1",),
        )
        self._block_weapon(payload, 10020)
        state = state_for(payload)
        state.plans[10010] = PlanState(
            10010, Pos(8, 2),
            "fund:WeaponUpgradeVoucher1:10020:10020",
            70, state.session_index,
        )

        candidates = economy.propose_economy(
            Turn.load(payload), state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=64,
        )
        action = next(
            candidate for candidate in candidates
            if candidate.proposal.actor_id == 10010
        )

        self.assertEqual(action.proposal.command["action"], "use")
        self.assertEqual(action.proposal.command["targetPos"], [{"x": 8, "y": 8}])

    def test_second_target_purchase_use_and_return_chain(self):
        payload = economy_payload(worker_pos=(5, 2), gold=100)
        payload["teamOur"]["teamId"] = "investment-second-target-chain"
        self._block_weapon(payload, 10020)
        engine = DecisionEngine()
        actions = []

        for _ in range(16):
            response = engine.decide(payload)
            command = response["roleCommandMap"]["10010"]
            actions.append(command["action"])
            worker = payload["teamOur"]["roles"][0]
            if command["action"] == "move":
                worker["pos"] = copy.deepcopy(command["targetPos"][0])
            elif command["action"] == "buy":
                payload["teamOur"]["goldNum"] = 0
                worker["backpack"] = ["WeaponUpgradeVoucher1"]
            elif command["action"] == "use":
                self.assertEqual(command["targetPos"], [{"x": 8, "y": 8}])
                worker["backpack"] = []
                next(
                    entry for entry in payload["teamOur"]["roles"]
                    if entry["id"] == 10030
                )["level"] = 2
                break
            payload["roundNo"] += 1
            payload["lastRoundRoleActionResults"] = {"10010": True}

        self.assertIn("buy", actions)
        self.assertEqual(actions[-1], "use")
        worker = payload["teamOur"]["roles"][0]
        self.assertLessEqual(max(
            abs(worker["pos"]["x"] - 8), abs(worker["pos"]["y"] - 8),
        ), 1)

    def test_destroyed_committed_wall_reselects_remaining_repair(self):
        payload = economy_payload(worker_pos=(7, 8), items=("WallFixer",))
        payload["teamOur"]["roles"].extend([
            role(10050, "wall", 2, 2, health=100),
            role(10060, "wall", 7, 9, health=100),
        ])
        state = state_for(payload)
        state.plans[10010] = PlanState(
            10010, Pos(2, 2), "fund:WallFixer:10030:10050",
            70, state.session_index,
        )
        payload["teamOur"]["roles"] = [
            entry for entry in payload["teamOur"]["roles"]
            if entry["id"] != 10050
        ]

        candidates = economy.propose_economy(
            Turn.load(payload), state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=64,
        )
        action = next(
            candidate for candidate in candidates
            if candidate.proposal.actor_id == 10010
        )

        self.assertEqual(action.proposal.command["action"], "use")
        self.assertEqual(action.proposal.command["targetPos"], [{"x": 7, "y": 9}])

    def test_repaired_committed_wall_reselects_remaining_repair(self):
        payload = economy_payload(worker_pos=(7, 8), items=("WallFixer",))
        payload["teamOur"]["roles"].extend([
            role(10050, "wall", 2, 2, health=1000),
            role(10060, "wall", 7, 9, health=100),
        ])
        state = state_for(payload)
        state.plans[10010] = PlanState(
            10010, Pos(2, 2), "use:WallFixer:10050",
            70, state.session_index,
        )

        candidates = economy.propose_economy(
            Turn.load(payload), state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=64,
        )
        action = next(
            candidate for candidate in candidates
            if candidate.proposal.actor_id == 10010
        )
        self.assertEqual(action.proposal.command["action"], "use")
        self.assertEqual(action.proposal.command["targetPos"], [{"x": 7, "y": 9}])

    def test_late_committed_target_reselects_timely_second_target(self):
        payload = economy_payload(
            round_no=69, worker_pos=(7, 8),
            items=("WeaponUpgradeVoucher1",),
        )
        engine = DecisionEngine()
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.set_plan(
            10010, Pos(8, 2), "use:WeaponUpgradeVoucher1:10020", 70,
        )

        response = engine.decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "use", "name": "WeaponUpgradeVoucher1",
            "targetPos": [{"x": 8, "y": 8}],
        })
        self.assertEqual(
            engine.state.state.plans[10010].reason,
            "use:WeaponUpgradeVoucher1:10030",
        )

    def test_local_limit_keeps_committed_target_across_engine_rounds(self):
        payload = economy_payload(
            worker_pos=(7, 2), items=("WeaponUpgradeVoucher1",),
        )
        payload["teamOur"]["teamId"] = "investment-limit-stable-engine"
        engine = DecisionEngine()
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.set_plan(
            10010, Pos(8, 8), "use:WeaponUpgradeVoucher1:10030", 70,
        )

        with patch.object(
            economy, "next_step",
            return_value=PathResult("expansion_limit", None, 64, None),
        ):
            engine.decide(payload)
            payload["roundNo"] = 2
            engine.decide(payload)

        self.assertEqual(
            engine.state.state.plans[10010].reason,
            "use:WeaponUpgradeVoucher1:10030",
        )

    def test_invalid_committed_upgrade_target_safely_reselects(self):
        payload = economy_payload(
            worker_pos=(7, 2), items=("WeaponUpgradeVoucher1",),
        )
        payload["teamOur"]["teamId"] = "investment-invalid-target"
        next(
            entry for entry in payload["teamOur"]["roles"]
            if entry["id"] == 10030
        )["level"] = 2
        state = state_for(payload)
        state.plans[10010] = PlanState(
            10010, Pos(8, 8),
            "fund:WeaponUpgradeVoucher1:10030:10030",
            70, state.session_index,
        )

        candidates = economy.propose_economy(
            Turn.load(payload), state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=64,
        )
        action = next(
            candidate for candidate in candidates
            if candidate.proposal.actor_id == 10010
        )

        self.assertEqual(action.proposal.command["action"], "use")
        self.assertEqual(action.proposal.command["targetPos"], [{"x": 8, "y": 2}])

    def test_local_search_limit_does_not_switch_committed_target(self):
        payload = economy_payload(
            worker_pos=(7, 2), items=("WeaponUpgradeVoucher1",),
        )
        state = state_for(payload)
        state.plans[10010] = PlanState(
            10010, Pos(8, 8),
            "fund:WeaponUpgradeVoucher1:10030:10030",
            70, state.session_index,
        )

        with patch.object(
            economy, "next_step",
            return_value=PathResult("expansion_limit", None, 64, None),
        ):
            candidates = economy.propose_economy(
                Turn.load(payload), state, clock=lambda: 0.0, deadline=1.0,
                max_expansions=64,
            )

        self.assertFalse(any(
            candidate.proposal.command.get("action") == "use"
            and candidate.proposal.command.get("targetPos")
            == [{"x": 8, "y": 2}]
            for candidate in candidates
        ))

    def test_later_target_truncation_keeps_completed_first_route(self):
        payload = economy_payload(
            worker_pos=(0, 0), items=("WeaponUpgradeVoucher1",),
        )
        turn = Turn.load(payload)
        worker = turn.workers()[0]
        context = economy.RouteSearchContext({})
        token = economy._ROUTE_SEARCH_CONTEXT.set(context)

        def routes(_turn, moving, target, *_args):
            if target == Pos(8, 8) and moving.pos == Pos(0, 0):
                context.truncated_reason = "search_limit"
                return ()
            if context.truncated_reason is not None:
                return ()
            if target == Pos(8, 2):
                return ((Pos(7, 2), 7),)
            return ()

        try:
            with patch.object(economy, "_routes_to_adjacent", side_effect=routes):
                route = economy._held_item_route(
                    turn, worker, "WeaponUpgradeVoucher1",
                    lambda: 0.0, 1.0, 64,
                )
        finally:
            economy._ROUTE_SEARCH_CONTEXT.reset(token)

        self.assertIsNotNone(route)
        self.assertEqual(route.use_target_id, 10020)
        self.assertEqual(context.truncated_reason, "search_limit")

    def test_wall_fixer_skips_blocked_first_damaged_wall(self):
        payload = economy_payload(worker_pos=(7, 8), items=("WallFixer",))
        payload["teamOur"]["teamId"] = "investment-wall-second-target"
        payload["teamOur"]["roles"].extend([
            role(10050, "wall", 2, 2, health=100),
            role(10060, "wall", 7, 9, health=100),
        ])
        occupied = {
            (entry["pos"]["x"], entry["pos"]["y"])
            for entry in payload["teamOur"]["roles"]
        }
        next_id = 49100
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                pos = (2 + dx, 2 + dy)
                if (dx or dy) and pos not in occupied:
                    payload["teamOur"]["roles"].append(
                        role(next_id, "wall", *pos, health=1000)
                    )
                    next_id += 1

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "use", "name": "WallFixer",
            "targetPos": [{"x": 7, "y": 9}],
        })

    def test_investment_diagnostic_marks_stable_target_continuation(self):
        payload = economy_payload(
            worker_pos=(7, 8), items=("WeaponUpgradeVoucher1",),
        )
        state = state_for(payload)
        state.plans[10010] = PlanState(
            10010, Pos(8, 8),
            "fund:WeaponUpgradeVoucher1:10030:10030",
            70, state.session_index,
        )
        diagnostics = []

        economy.propose_economy(
            Turn.load(payload), state, clock=lambda: 0.0, deadline=1.0,
            max_expansions=64, diagnostic_sink=diagnostics.append,
        )

        self.assertEqual(diagnostics[0]["investmentTarget"], {
            "item": "WeaponUpgradeVoucher1",
            "targetId": "10030",
            "selection": "continued",
        })

    def test_existing_mine_plan_yields_to_current_single_funding_route(self):
        # Break caught: a valid old mine plan bypasses a newly executable sale chain.
        payload = economy_payload(
            round_no=15,
            worker_pos=(2, 1),
            items=("copper",),
        )
        state = state_for(payload)
        state.plans[10010] = PlanState(
            10010,
            Pos(2, 2),
            "mine:copper",
            None,
            state.session_index,
        )

        candidates = economy.propose_economy(
            Turn.load(payload),
            state,
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=64,
        )
        worker = next(
            candidate for candidate in candidates
            if candidate.proposal.actor_id == 10010
        )

        self.assertEqual(worker.proposal.command, {
            "action": "move", "targetPos": [{"x": 3, "y": 1}],
        })
        self.assertEqual(
            worker.plan_reason,
            "fund:WeaponUpgradeVoucher1:10020:10020",
        )

    def test_new_joint_funding_does_not_claim_reserved_wall_builder(self):
        # Break caught: funding arbitration takes the builder selected before economy.
        payload = economy_payload(round_no=30, worker_pos=(3, 1))
        payload["teamOur"]["teamId"] = "economy-r8-builder-reservation"
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 3, 3, items=("copper",) * 6),
        )
        payload["teamOur"]["roles"][0]["backpack"] = ["copper"] * 6
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 4, "y": 2}, "neutralType": "vendor"},
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        engine = DecisionEngine()

        response = engine.decide(payload)
        builder_id = engine.state.state.fortification_builder_id

        self.assertIsNotNone(builder_id)
        self.assertNotIn(
            ":joint:", engine.state.state.plans[builder_id].reason,
        )
        self.assertIn(
            response["roleCommandMap"][str(builder_id)]["action"],
            ("move", "collect", "build"),
        )

    def test_wall_builder_replaces_old_ordinary_copper_route(self):
        # Break caught: a preexisting ordinary mine plan starves construction forever.
        payload = economy_payload(round_no=4, worker_pos=(0, 0))
        payload["teamOur"]["teamId"] = "economy-r8-old-copper"
        payload["teamOur"]["roles"] = [
            payload["teamOur"]["roles"][0],
            payload["teamOur"]["roles"][1],
            payload["teamOur"]["roles"][2],
            payload["teamOur"]["roles"][3],
        ]
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 4, "y": 0}, "neutralType": "copper"},
            {"pos": {"x": 6, "y": 0}, "neutralType": "vendor"},
        ]
        payload["vendorShopList"] = [
            {"name": "stone", "price": 1},
            {"name": "copper", "price": 20},
        ]
        engine = DecisionEngine()
        first = engine.decide(payload)["roleCommandMap"]["10010"]
        self.assertEqual(engine.state.state.plans[10010].target, Pos(4, 0))

        following = copy.deepcopy(payload)
        following["roundNo"] = 5
        following["lastRoundRoleActionResults"] = {"10010": True}
        following["teamOur"]["roles"][0]["pos"] = first["targetPos"][0]
        following["teamOur"]["roles"].append(
            role(10040, "rocket", 11, 8, health=1000),
        )
        second = engine.decide(following)["roleCommandMap"]["10010"]

        self.assertEqual(second, {
            "action": "collect", "targetPos": [{"x": 1, "y": 1}],
        })

    def test_real_economy_entry_uses_reachable_funding_route_after_local_limit(self):
        # Break caught: the real planner mines stone when one hard vendor stand
        # hides other reachable stands for a fully funded upgrade route.
        payload = economy_payload(
            worker_pos=(0, 0),
            items=("copper",) * 20,
        )
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 6, "y": 6}, "neutralType": "vendor"},
            {"pos": {"x": 6, "y": 8}, "neutralType": "weaponShop"},
        ]
        turn = Turn.load(payload)

        candidates = economy.propose_economy(
            turn,
            state_for(payload),
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=7,
        )

        funding = [
            candidate for candidate in candidates
            if candidate.plan_reason
            == "fund:WeaponUpgradeVoucher1:10030:10030"
        ]
        self.assertEqual(len(funding), 1)
        self.assertEqual(funding[0].proposal.command_owner_id, 10010)
        self.assertEqual(funding[0].proposal.actor_id, 10010)
        self.assertEqual(funding[0].proposal.command, {
            "action": "move", "targetPos": [{"x": 0, "y": 1}],
        })
        self.assertEqual(funding[0].proposal.destination, Pos(0, 1))
        self.assertTrue(ActionAllocator(turn).try_add(funding[0].proposal))

    def test_real_economy_entry_moves_to_vendor_after_local_limit(self):
        # Break caught: the execution move gives up after one hard vendor stand.
        payload = economy_payload(
            worker_pos=(0, 0),
            items=("copper",) * 100,
        )
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 6, "y": 6}, "neutralType": "vendor"},
        ]
        payload["weaponShopList"] = []
        turn = Turn.load(payload)

        candidates = economy.propose_economy(
            turn,
            state_for(payload),
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=7,
        )

        vendor_moves = [
            candidate for candidate in candidates
            if candidate.plan_reason == "vendor"
        ]
        self.assertEqual(len(vendor_moves), 1)
        self.assertEqual(vendor_moves[0].proposal.command, {
            "action": "move", "targetPos": [{"x": 0, "y": 1}],
        })
        self.assertEqual(vendor_moves[0].proposal.destination, Pos(0, 1))
        self.assertEqual(vendor_moves[0].plan_target, Pos(6, 6))
        self.assertTrue(ActionAllocator(turn).try_add(vendor_moves[0].proposal))

    def test_economy_move_adjacent_stops_after_global_deadline(self):
        payload = economy_payload(worker_pos=(0, 0))
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
        ]
        turn = Turn.load(payload)
        worker = turn.workers()[0]

        class ExpiredClock:
            calls = 0

            def __call__(self):
                self.calls += 1
                return 10.0

        clock = ExpiredClock()
        candidate = economy._move_adjacent(
            turn,
            worker,
            Pos(6, 6),
            "vendor",
            None,
            clock,
            5.0,
            7,
        )

        self.assertIsNone(candidate)
        self.assertEqual(clock.calls, 1)

    def test_real_economy_entry_checks_other_timely_shop_stands(self):
        # Break caught: purchase timing rejects the shop after one hard stand.
        payload = economy_payload(worker_pos=(0, 0), gold=100)
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 6, "y": 6}, "neutralType": "weaponShop"},
        ]
        payload["vendorShopList"] = []
        turn = Turn.load(payload)

        candidates = economy.propose_economy(
            turn,
            state_for(payload),
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=7,
        )

        shop_moves = [
            candidate for candidate in candidates
            if candidate.plan_reason
            == "fund:WeaponUpgradeVoucher1:10030:10030"
        ]
        self.assertEqual(len(shop_moves), 1)
        self.assertEqual(shop_moves[0].proposal.command, {
            "action": "move", "targetPos": [{"x": 0, "y": 1}],
        })
        self.assertEqual(shop_moves[0].proposal.destination, Pos(0, 1))
        self.assertEqual(shop_moves[0].plan_target, Pos(6, 6))
        self.assertTrue(ActionAllocator(turn).try_add(shop_moves[0].proposal))

    def test_purchase_timeliness_stops_after_global_deadline(self):
        payload = economy_payload(worker_pos=(0, 0), gold=100)
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 6, "y": 6}, "neutralType": "weaponShop"},
        ]
        turn = Turn.load(payload)
        worker = turn.workers()[0]

        class ExpiredClock:
            calls = 0

            def __call__(self):
                self.calls += 1
                return 10.0

        clock = ExpiredClock()
        timely = economy._purchase_is_timely(
            turn,
            worker,
            Pos(6, 6),
            "WeaponUpgradeVoucher1",
            clock,
            5.0,
            7,
        )

        self.assertFalse(timely)
        self.assertEqual(clock.calls, 1)

    def test_adjacent_route_keeps_found_stands_after_local_expansion_limit(self):
        # Break caught: one hard stand discards other reachable target stands.
        payload = economy_payload(worker_pos=(0, 0))
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
        ]
        payload["teamOur"]["roles"] = [payload["teamOur"]["roles"][0]]
        turn = Turn.load(payload)
        worker = turn.workers()[0]
        context = economy.RouteSearchContext({})
        token = economy._ROUTE_SEARCH_CONTEXT.set(context)
        try:
            first = economy._routes_to_adjacent(
                turn,
                worker,
                Pos(6, 6),
                lambda: 0.0,
                1.0,
                7,
            )
            searches_after_first = context.path_searches
            second = economy._routes_to_adjacent(
                turn,
                worker,
                Pos(6, 6),
                lambda: 0.0,
                1.0,
                7,
            )
        finally:
            economy._ROUTE_SEARCH_CONTEXT.reset(token)

        self.assertEqual(first, ((Pos(5, 6), 6), (Pos(6, 5), 6)))
        self.assertEqual(second, first)
        self.assertEqual(context.truncated_reason, "expansion_limit")
        self.assertEqual(context.cache_hits, 0)
        self.assertGreater(context.path_searches, searches_after_first)

    def test_joint_planning_reuses_paths_and_keeps_real_actions(self):
        # Break caught: equivalent joint subroutes repeat thousands of searches.
        payload = economy_payload(
            round_no=30,
            worker_pos=(2, 1),
            items=("copper",) * 10,
        )
        payload["teamOur"]["teamId"] = "economy-joint-search-budget"
        payload["mapInfo"]["width"] = 41
        payload["mapInfo"]["height"] = 32
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 2, 3, items=("copper",) * 10),
        )
        payload["vendorShopList"] = [{"name": "copper", "price": 5}]
        payload["weaponShopList"] = [
            {"name": "StationUpgradeVoucher1", "price": 100},
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        real_next_step = economy.next_step
        search_count = 0

        def counted_next_step(*args, **kwargs):
            nonlocal search_count
            search_count += 1
            return real_next_step(*args, **kwargs)

        with patch.object(economy, "next_step", counted_next_step):
            response = DecisionEngine().decide(payload)

        self.assertEqual(set(response["roleCommandMap"]), {"10010", "10012"})
        self.assertTrue(all(
            command["action"] == "move"
            for command in response["roleCommandMap"].values()
        ))
        self.assertLessEqual(search_count, 1894)

    def test_route_cache_is_not_reused_across_changed_rounds(self):
        payload = economy_payload(
            round_no=30,
            worker_pos=(1, 1),
            items=("copper",) * 10,
        )
        payload["teamOur"]["teamId"] = "economy-request-cache-boundary"
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        engine = DecisionEngine()

        first = engine.decide(payload)
        first_step = first["roleCommandMap"]["10010"]["targetPos"][0]
        changed = copy.deepcopy(payload)
        changed["roundNo"] = 31
        changed["lastRoundRoleActionResults"] = {"10010": False}
        changed["teamOur"]["roles"].append(
            role(10050, "wall", first_step["x"], first_step["y"], health=1000),
        )

        response = engine.decide(changed)

        self.assertNotEqual(
            response["roleCommandMap"].get("10010", {}).get("targetPos"),
            [first_step],
        )

    def test_many_joint_routes_leave_a_bounded_search_delivery_reserve(self):
        # Break caught: many shops/vendors consume the whole response budget.
        payload = economy_payload(
            round_no=30,
            worker_pos=(2, 1),
            items=("copper",) * 10,
        )
        payload["teamOur"]["teamId"] = "economy-joint-search-cap"
        payload["mapInfo"]["width"] = 41
        payload["mapInfo"]["height"] = 32
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 2, 3, items=("copper",) * 10),
        )
        payload["mapInfo"]["zones"] = [
            *(
                {"pos": {"x": x, "y": 5}, "neutralType": "vendor"}
                for x in range(10, 18)
            ),
            *(
                {"pos": {"x": x, "y": 15}, "neutralType": "weaponShop"}
                for x in range(20, 28)
            ),
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 5}]
        payload["weaponShopList"] = [
            {"name": "StationUpgradeVoucher1", "price": 100},
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        payload["worldNews"] = {}
        real_next_step = economy.next_step
        search_count = 0

        def counted_next_step(*args, **kwargs):
            nonlocal search_count
            search_count += 1
            return real_next_step(*args, **kwargs)

        traces = []
        with patch.object(economy, "next_step", counted_next_step):
            response = DecisionEngine(max_search_expansions=64).decide(
                payload, trace_sink=traces.append,
            )

        self.assertLessEqual(search_count, 1608)
        self.assertEqual(response["prompt"], "")
        self.assertEqual(response["executeCmd"], "")
        self.assertTrue(all(
            command["action"] in {"move", "sell", "buy", "use", "collect"}
            for command in response["roleCommandMap"].values()
        ))
        self.assertEqual(set(response["roleCommandMap"]), {"10010", "10012"})
        self.assertTrue(all(
            command["action"] == "move"
            for command in response["roleCommandMap"].values()
        ))
        self.assertEqual(traces[0]["economyPlanning"]["pathSearches"], 1600)
        self.assertEqual(
            traces[0]["economyPlanning"]["truncatedReason"], "search_limit",
        )
        self.assertEqual(
            traces[0]["economyPlanning"]["jointStatus"], "accepted",
        )

    def test_three_towers_fourteen_wall_targets_stay_within_search_cap(self):
        from agent.layout import plan_defense_layout

        payload = economy_payload(round_no=30, worker_pos=(0, 0), gold=10)
        payload["teamOur"]["teamId"] = "investment-14-wall-pressure"
        payload["mapInfo"].update({"width": 41, "height": 32})
        blueprint = plan_defense_layout(Turn.load(payload))
        payload["teamOur"]["roles"].extend(
            role(20000 + index, "wall", target.x, target.y, health=100)
            for index, target in enumerate(blueprint.wall_targets)
        )
        diagnostics = []
        candidates = economy.propose_economy(
            Turn.load(payload), state_for(payload), clock=lambda: 0.0,
            deadline=1.0, max_expansions=64,
            diagnostic_sink=diagnostics.append,
        )

        command = next(
            candidate.proposal.command for candidate in candidates
            if candidate.proposal.actor_id == 10010
        )
        self.assertIn(command["action"], {"move", "use"})
        self.assertLessEqual(
            diagnostics[0]["pathSearches"], 1600,
        )
        self.assertEqual(
            diagnostics[0]["investmentTarget"]["item"],
            "WallFixer",
        )

    def test_economy_uses_only_its_reserved_share_of_request_budget(self):
        payload = economy_payload(round_no=30)
        payload["worldNews"] = {}
        observed = {}

        def capture_economy(*args, **kwargs):
            observed["deadline"] = kwargs["deadline"]
            if kwargs.get("diagnostic_sink") is not None:
                kwargs["diagnostic_sink"]({
                    "pathSearches": 0,
                    "cacheHits": 0,
                    "truncatedReason": None,
                    "jointStatus": None,
                    "heldInvestment": None,
                })
            return ()

        with patch("agent.brain.propose_economy", side_effect=capture_economy):
            response = DecisionEngine(
                clock=lambda: 10.0, budget_seconds=4.0,
            ).decide(payload)

        self.assertEqual(observed["deadline"], 13.0)
        self.assertEqual(response["prompt"], "")
        self.assertEqual(response["executeCmd"], "")

    def test_unfunded_joint_route_reports_actual_funds_blocker(self):
        payload = economy_payload(
            round_no=30,
            worker_pos=(3, 1),
            items=("copper",),
        )
        payload["teamOur"]["teamId"] = "economy-joint-funds-diagnostic"
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 3, 3, items=("copper",)),
        )
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        traces = []

        DecisionEngine().decide(payload, trace_sink=traces.append)

        self.assertEqual(
            traces[0]["economyPlanning"]["jointStatus"],
            "actual_funds_insufficient",
        )

    def test_underfilled_worker_sells_early_to_fund_reachable_upgrade(self):
        engine = DecisionEngine()
        payload = economy_payload(
            round_no=30,
            worker_pos=(2, 1),
            items=("copper",) * 10,
            gold=0,
        )
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]

        response = engine.decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")
        self.assertTrue(
            engine.state.state.plans[10010].reason.startswith("fund:")
        )

    def test_funding_executes_vendor_from_the_verified_complete_route(self):
        payload = economy_payload(
            round_no=30,
            worker_pos=(2, 2),
            items=("copper",) * 10,
            gold=0,
        )
        payload["teamOur"]["teamId"] = "economy-bound-vendor"
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 6, "y": 2}, "neutralType": "vendor"},
            {"pos": {"x": 5, "y": 8}, "neutralType": "vendor"},
            {"pos": {"x": 1, "y": 9}, "neutralType": "weaponShop"},
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]

        engine = DecisionEngine()
        response = engine.decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")
        self.assertEqual(engine.state.state.plans[10010].target, Pos(5, 8))

    def test_funding_executes_shop_from_the_verified_complete_route(self):
        payload = economy_payload(
            round_no=30,
            worker_pos=(3, 2),
            items=("copper",) * 10,
            gold=0,
        )
        payload["teamOur"]["teamId"] = "economy-bound-shop"
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 4, "y": 2}, "neutralType": "vendor"},
            {"pos": {"x": 1, "y": 2}, "neutralType": "weaponShop"},
            {"pos": {"x": 6, "y": 6}, "neutralType": "weaponShop"},
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        engine = DecisionEngine()
        self.assertEqual(
            engine.decide(payload)["roleCommandMap"]["10010"]["action"],
            "sell",
        )

        funded = copy.deepcopy(payload)
        funded["roundNo"] = 31
        funded["teamOur"]["roles"][0]["backpack"] = []
        funded["teamOur"]["goldNum"] = 100
        funded["lastRoundRoleActionResults"] = {"10010": True}
        engine.decide(funded)

        self.assertEqual(engine.state.state.plans[10010].target, Pos(6, 6))

    def test_underfilled_worker_sells_early_to_fund_required_third_tower(self):
        payload = economy_payload(
            round_no=30,
            worker_pos=(3, 2),
            items=("copper",) * 5,
            gold=0,
        )
        payload["teamOur"]["teamId"] = "economy-fund-third-tower"
        payload["teamOur"]["roles"] = payload["teamOur"]["roles"][:-1]
        payload["vendorShopList"] = [{"name": "copper", "price": 5}]

        engine = DecisionEngine()
        response = engine.decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "sell", "name": "copper", "num": 5,
        })
        self.assertTrue(
            engine.state.state.plans[10010].reason.startswith("fund:build:")
        )
        payload["teamOur"]["roles"][0]["backpack"] = []
        payload["teamOur"]["goldNum"] = 25
        payload["lastRoundRoleActionResults"] = {
            "10010": True,
            "10012": False,
        }
        built = None
        for round_no in range(31, 43):
            payload["roundNo"] = round_no
            command = engine.decide(payload)["roleCommandMap"]["10010"]
            if command["action"] == "build":
                built = command
                break
            self.assertEqual(command["action"], "move")
            payload["teamOur"]["roles"][0]["pos"] = copy.deepcopy(
                command["targetPos"][0]
            )
        self.assertIsNotNone(built)
        self.assertEqual(built["name"], "rocket")

    def test_third_tower_funding_uses_vendor_from_verified_build_route(self):
        payload = economy_payload(
            round_no=30,
            worker_pos=(2, 2),
            items=("copper",) * 5,
            gold=0,
        )
        payload["teamOur"]["teamId"] = "economy-bound-build-vendor"
        payload["teamOur"]["roles"] = payload["teamOur"]["roles"][:-1]
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 2, "y": 4}, "neutralType": "vendor"},
            {"pos": {"x": 6, "y": 6}, "neutralType": "vendor"},
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 5}]

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "move", "targetPos": [{"x": 3, "y": 3}],
        })

    def test_underfilled_worker_keeps_mining_when_inventory_cannot_fund_need(self):
        engine = DecisionEngine()
        payload = economy_payload(
            round_no=30,
            worker_pos=(2, 1),
            items=("copper",),
            gold=0,
        )
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]

        response = engine.decide(payload)

        self.assertEqual(
            response["roleCommandMap"]["10010"],
            {"action": "collect", "targetPos": [{"x": 2, "y": 2}]},
        )

    def test_mining_prefers_higher_realizable_value_per_round_over_nearest(self):
        # Break caught: nearest-first spends the day on low-value stone.
        payload = economy_payload(round_no=30, worker_pos=(0, 0))
        payload["teamOur"]["teamId"] = "economy-mine-yield"
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 4, "y": 0}, "neutralType": "copper"},
            {"pos": {"x": 6, "y": 0}, "neutralType": "vendor"},
            {"pos": {"x": 8, "y": 0}, "neutralType": "weaponShop"},
        ]
        payload["vendorShopList"] = [
            {"name": "stone", "price": 1},
            {"name": "copper", "price": 20},
        ]
        with_completed_wall_line(payload)

        engine = DecisionEngine()
        response = engine.decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")
        self.assertEqual(engine.state.state.plans[10010].target, Pos(4, 0))

    def test_mining_reselects_when_current_quote_loses_realizable_value(self):
        # Break caught: an in-transit mine plan ignores a changed official quote.
        payload = economy_payload(round_no=30, worker_pos=(0, 0))
        payload["teamOur"]["teamId"] = "economy-mine-reprice"
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 4, "y": 0}, "neutralType": "copper"},
            {"pos": {"x": 6, "y": 0}, "neutralType": "vendor"},
        ]
        payload["vendorShopList"] = [
            {"name": "stone", "price": 20},
            {"name": "copper", "price": 1},
        ]
        engine = DecisionEngine()
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.set_plan(10010, Pos(4, 0), "mine:copper", None)

        response = engine.decide(payload)

        self.assertEqual(
            response["roleCommandMap"]["10010"],
            {"action": "collect", "targetPos": [{"x": 1, "y": 1}]},
        )
        self.assertEqual(engine.state.state.plans[10010].target, Pos(1, 1))

    def test_mining_keeps_current_target_for_only_marginal_improvement(self):
        # Break caught: tiny quote changes make the worker oscillate between mines.
        payload = economy_payload(round_no=30, worker_pos=(0, 0))
        payload["teamOur"]["teamId"] = "economy-mine-stable"
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 4, "y": 0}, "neutralType": "copper"},
            {"pos": {"x": 6, "y": 0}, "neutralType": "vendor"},
        ]
        payload["vendorShopList"] = [
            {"name": "stone", "price": 10},
            {"name": "copper", "price": 11},
        ]
        engine = DecisionEngine()
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.set_plan(10010, Pos(1, 1), "mine:stone", None)

        response = engine.decide(payload)

        self.assertEqual(
            response["roleCommandMap"]["10010"],
            {"action": "collect", "targetPos": [{"x": 1, "y": 1}]},
        )
        self.assertEqual(engine.state.state.plans[10010].target, Pos(1, 1))

    def test_mining_ignores_unreachable_high_value_mine(self):
        # Break caught: quoted value is ranked without a legal collection stand.
        payload = economy_payload(round_no=30, worker_pos=(0, 0))
        payload["teamOur"]["teamId"] = "economy-mine-blocked"
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 4, "y": 4}, "neutralType": "copper"},
            {"pos": {"x": 6, "y": 0}, "neutralType": "vendor"},
        ]
        payload["teamOur"]["roles"].extend(
            role(10100 + index, "wall", x, y, health=1000)
            for index, (x, y) in enumerate(
                (x, y)
                for x in range(3, 6)
                for y in range(3, 6)
                if (x, y) != (4, 4)
            )
        )
        payload["vendorShopList"] = [
            {"name": "stone", "price": 1},
            {"name": "copper", "price": 100},
        ]

        engine = DecisionEngine()
        response = engine.decide(payload)

        self.assertEqual(
            response["roleCommandMap"]["10010"],
            {"action": "collect", "targetPos": [{"x": 1, "y": 1}]},
        )
        self.assertEqual(engine.state.state.plans[10010].target, Pos(1, 1))

    def test_mining_without_market_data_keeps_legal_nearest_behavior(self):
        # Break caught: incomplete market data suppresses all basic collection.
        payload = economy_payload(round_no=30, worker_pos=(0, 0))
        payload["teamOur"]["teamId"] = "economy-mine-no-market"
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 4, "y": 0}, "neutralType": "copper"},
        ]
        payload["vendorShopList"] = []
        payload["weaponShopList"] = []

        response = DecisionEngine().decide(payload)

        self.assertEqual(
            response["roleCommandMap"]["10010"],
            {"action": "collect", "targetPos": [{"x": 1, "y": 1}]},
        )

    def test_failed_collection_reselects_another_visible_mine(self):
        # Break caught: a failed mine remains the best target and is retried forever.
        payload = economy_payload(round_no=30, worker_pos=(0, 0))
        payload["teamOur"]["teamId"] = "economy-mine-failed"
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "copper"},
            {"pos": {"x": 2, "y": 0}, "neutralType": "stone"},
            {"pos": {"x": 4, "y": 0}, "neutralType": "vendor"},
        ]
        payload["vendorShopList"] = [
            {"name": "stone", "price": 1},
            {"name": "copper", "price": 20},
        ]
        with_completed_wall_line(payload)
        engine = DecisionEngine()
        first = engine.decide(payload)
        self.assertEqual(
            first["roleCommandMap"]["10010"],
            {"action": "collect", "targetPos": [{"x": 1, "y": 1}]},
        )

        failed = copy.deepcopy(payload)
        failed["roundNo"] = 31
        failed["lastRoundRoleActionResults"] = {"10010": False}
        response = engine.decide(failed)

        self.assertEqual(engine.state.state.plans[10010].target, Pos(2, 0))
        self.assertNotEqual(
            response["roleCommandMap"]["10010"].get("targetPos"),
            [{"x": 1, "y": 1}],
        )

    def test_far_high_value_mine_is_rejected_when_liquidation_misses_dusk(self):
        # Break caught: raw mineral price outranks a realizable near-dusk route.
        payload = economy_payload(round_no=65, worker_pos=(0, 0))
        payload["teamOur"]["teamId"] = "economy-mine-dusk"
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 8, "y": 0}, "neutralType": "copper"},
            {"pos": {"x": 3, "y": 0}, "neutralType": "vendor"},
        ]
        payload["vendorShopList"] = [
            {"name": "stone", "price": 1},
            {"name": "copper", "price": 100},
        ]

        engine = DecisionEngine()
        engine.decide(payload)

        plan = engine.state.state.plans.get(10010)
        self.assertTrue(plan is None or plan.target != Pos(8, 0))

    def test_funding_sell_uses_only_quantity_needed_for_current_deficit(self):
        engine = DecisionEngine()
        payload = economy_payload(
            round_no=40,
            worker_pos=(3, 2),
            items=("copper",) * 10,
            gold=70,
        )
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]

        response = engine.decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "sell", "name": "copper", "num": 3,
        })

    def test_feasible_funding_chain_beats_ordinary_recall_until_hard_cutoff(self):
        feasible = economy_payload(
            round_no=60,
            worker_pos=(3, 2),
            items=("copper",) * 10,
            gold=0,
        )
        feasible["vendorShopList"] = [{"name": "copper", "price": 10}]
        engine = DecisionEngine()

        response = engine.decide(feasible)

        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "sell")
        self.assertTrue(engine.state.state.plans[10010].reason.startswith("fund:"))

        late = copy.deepcopy(feasible)
        late["teamOur"]["teamId"] = "economy-hard-cutoff"
        late["roundNo"] = 66
        late_engine = DecisionEngine()
        response = late_engine.decide(late)

        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")
        self.assertTrue(
            late_engine.state.state.plans[10010].reason.startswith("gunner:")
        )

    def test_temporary_gunner_can_leave_only_for_full_timely_funding_route(self):
        payload = economy_payload(
            round_no=60,
            worker_pos=(7, 2),
            items=("copper",) * 10,
            gold=0,
        )
        payload["teamOur"]["teamId"] = "economy-temporary-gunner-fund"
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]

        engine = DecisionEngine()
        response = engine.decide(payload)

        self.assertIn(
            response["roleCommandMap"]["10010"]["action"], {"move", "sell"},
        )
        self.assertTrue(engine.state.state.plans[10010].reason.startswith("fund:"))

    def test_shared_gold_allows_only_one_real_engine_purchase(self):
        payload = economy_payload(round_no=30, worker_pos=(5, 2), gold=10)
        payload["teamOur"]["teamId"] = "economy-shared-gold"
        payload["teamOur"]["roles"][0]["health"] = 100
        payload["teamOur"]["roles"].insert(
            1, role(10011, "worker", 5, 3, health=100),
        )
        payload["weaponShopList"] = [{"name": "Medicine", "price": 10}]

        response = DecisionEngine().decide(payload)

        buys = [
            command for command in response["roleCommandMap"].values()
            if command.get("action") == "buy"
        ]
        self.assertEqual(len(buys), 1)
        self.assertEqual(buys[0]["name"], "Medicine")

    def test_two_workers_jointly_sell_buy_use_and_return_to_distinct_posts(self):
        # Break caught: two real inventories are never coordinated when each is short.
        payload = economy_payload(round_no=30, worker_pos=(1, 1))
        payload["teamOur"]["teamId"] = "economy-joint-full-chain"
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 1, 3, items=("copper",) * 6),
        )
        payload["teamOur"]["roles"][0]["backpack"] = ["copper"] * 6
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 4, "y": 2}, "neutralType": "vendor"},
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        engine = DecisionEngine()
        actions = []

        for _ in range(35):
            response = engine.decide(payload)
            commands = response["roleCommandMap"]
            actions.extend(command["action"] for command in commands.values())
            self.assertLessEqual(sum(
                command["action"] == "buy" for command in commands.values()
            ), 1)
            for raw_id, command in commands.items():
                role_id = int(raw_id)
                actor = next(
                    entry for entry in payload["teamOur"]["roles"]
                    if entry["id"] == role_id
                )
                action = command["action"]
                if action == "move":
                    actor["pos"] = copy.deepcopy(command["targetPos"][0])
                elif action == "sell":
                    quantity = command["num"]
                    for _ in range(quantity):
                        actor["backpack"].remove(command["name"])
                    payload["teamOur"]["goldNum"] += 10 * quantity
                elif action == "buy":
                    payload["teamOur"]["goldNum"] -= 100
                    actor["backpack"].append(command["name"])
                elif action == "use":
                    actor["backpack"].remove(command["name"])
                    target_pos = command["targetPos"][0]
                    target = next(
                        entry for entry in payload["teamOur"]["roles"]
                        if entry["pos"] == target_pos
                    )
                    target["level"] += 1
            payload["lastRoundRoleActionResults"] = {
                role_id: True for role_id in commands
            }
            payload["roundNo"] += 1
            workers = payload["teamOur"]["roles"][:2]
            staffed = {
                tower["id"]
                for tower in payload["teamOur"]["roles"]
                if tower["roleType"] in ("gatling", "railgun", "rocket")
                and any(max(
                    abs(worker["pos"]["x"] - tower["pos"]["x"]),
                    abs(worker["pos"]["y"] - tower["pos"]["y"]),
                ) == 1 for worker in workers)
            }
            if "use" in actions and len(staffed) == 2:
                break

        self.assertIn("sell", actions)
        self.assertIn("buy", actions)
        self.assertIn("use", actions)
        self.assertEqual(len(staffed), 2)

    def test_joint_funding_full_chain_is_faction_equivalent(self):
        for faction in ("challenger", "defender"):
            with self.subTest(faction=faction):
                payload = economy_payload(round_no=30, worker_pos=(1, 1))
                payload["teamOur"]["type"] = faction
                payload["teamOur"]["teamId"] = f"economy-joint-{faction}"
                payload["teamOur"]["roles"].insert(
                    1, role(10012, "worker", 1, 3, items=("copper",) * 10),
                )
                payload["teamOur"]["roles"][0]["backpack"] = ["copper"] * 10
                payload["mapInfo"]["zones"] = [
                    {"pos": {"x": 4, "y": 2}, "neutralType": "vendor"},
                    {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
                ]
                payload["vendorShopList"] = [{"name": "copper", "price": 5}]
                payload["weaponShopList"] = [
                    {"name": "WeaponUpgradeVoucher1", "price": 100},
                ]
                engine = DecisionEngine()
                actions = []

                for _ in range(35):
                    response = engine.decide(payload)
                    commands = response["roleCommandMap"]
                    actions.extend(command["action"] for command in commands.values())
                    self.assertLessEqual(sum(
                        command["action"] == "buy"
                        for command in commands.values()
                    ), 1)
                    for raw_id, command in commands.items():
                        actor = next(
                            entry for entry in payload["teamOur"]["roles"]
                            if entry["id"] == int(raw_id)
                        )
                        action = command["action"]
                        if action == "move":
                            actor["pos"] = copy.deepcopy(command["targetPos"][0])
                        elif action == "sell":
                            for _ in range(command["num"]):
                                actor["backpack"].remove(command["name"])
                            payload["teamOur"]["goldNum"] += 5 * command["num"]
                        elif action == "buy":
                            payload["teamOur"]["goldNum"] -= 100
                            actor["backpack"].append(command["name"])
                        elif action == "use":
                            actor["backpack"].remove(command["name"])
                            target_pos = command["targetPos"][0]
                            target = next(
                                entry for entry in payload["teamOur"]["roles"]
                                if entry["pos"] == target_pos
                            )
                            target["level"] += 1
                    payload["lastRoundRoleActionResults"] = {
                        role_id: True for role_id in commands
                    }
                    payload["roundNo"] += 1
                    workers = payload["teamOur"]["roles"][:2]
                    staffed = {
                        tower["id"]
                        for tower in payload["teamOur"]["roles"]
                        if tower["roleType"] in ("gatling", "railgun", "rocket")
                        and any(max(
                            abs(worker["pos"]["x"] - tower["pos"]["x"]),
                            abs(worker["pos"]["y"] - tower["pos"]["y"]),
                        ) == 1 for worker in workers)
                    }
                    if "use" in actions and len(staffed) == 2:
                        break

                self.assertIn("sell", actions)
                self.assertIn("buy", actions)
                self.assertIn("use", actions)
                self.assertEqual(len(staffed), 2)

    def test_joint_funding_waits_for_confirmed_shared_gold(self):
        # Break caught: the buyer spends a collaborator's expected sale proceeds.
        payload = economy_payload(round_no=30, worker_pos=(3, 1))
        payload["teamOur"]["teamId"] = "economy-joint-confirmed-gold"
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 3, 3, items=("copper",) * 6),
        )
        payload["teamOur"]["roles"][0]["backpack"] = ["copper"] * 6
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 4, "y": 2}, "neutralType": "vendor"},
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        engine = DecisionEngine()

        first = engine.decide(payload)
        self.assertEqual(
            {command["action"] for command in first["roleCommandMap"].values()},
            {"sell"},
        )
        joint_plans = [
            plan for role_id, plan in engine.state.state.plans.items()
            if role_id in (10010, 10012) and ":joint:" in plan.reason
        ]
        self.assertEqual(len(joint_plans), 2)

        buyer_command = first["roleCommandMap"]["10010"]
        buyer = payload["teamOur"]["roles"][0]
        for _ in range(buyer_command["num"]):
            buyer["backpack"].remove("copper")
        payload["teamOur"]["goldNum"] = buyer_command["num"] * 10
        payload["roundNo"] = 31
        payload["lastRoundRoleActionResults"] = {"10010": True}

        second = engine.decide(payload)

        self.assertFalse(any(
            command["action"] == "buy"
            for command in second["roleCommandMap"].values()
        ))
        self.assertEqual(
            second["roleCommandMap"]["10012"]["action"], "sell",
        )
        self.assertLess(payload["teamOur"]["goldNum"], 100)

    def test_joint_funding_skips_blocked_first_matching_weapon(self):
        payload = economy_payload(round_no=30, worker_pos=(1, 1))
        payload["teamOur"]["teamId"] = "investment-joint-second-target"
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 1, 3, items=("copper",) * 10),
        )
        payload["teamOur"]["roles"][0]["backpack"] = ["copper"] * 10
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 4, "y": 2}, "neutralType": "vendor"},
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 5}]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        self._block_weapon(payload, 10020)

        engine = DecisionEngine()
        engine.decide(payload)

        joint = [
            plan.reason for role_id, plan in engine.state.state.plans.items()
            if role_id in (10010, 10012) and ":joint:" in plan.reason
        ]
        self.assertEqual(len(joint), 2)
        self.assertTrue(all(":10030:joint:" in reason for reason in joint))

    def test_joint_funding_rejects_insufficient_or_late_combined_value(self):
        # Break caught: an incomplete or untimely pair is protected as critical funding.
        base = economy_payload(round_no=30, worker_pos=(1, 1))
        base["teamOur"]["teamId"] = "economy-joint-reject"
        base["teamOur"]["roles"].insert(
            1, role(10012, "worker", 1, 3, items=("copper",) * 4),
        )
        base["teamOur"]["roles"][0]["backpack"] = ["copper"] * 4
        base["mapInfo"]["zones"] = [
            {"pos": {"x": 4, "y": 2}, "neutralType": "vendor"},
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        base["vendorShopList"] = [{"name": "copper", "price": 10}]
        base["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]

        insufficient_engine = DecisionEngine()
        insufficient_engine.decide(base)
        self.assertFalse(any(
            ":joint:" in plan.reason
            for plan in insufficient_engine.state.state.plans.values()
        ))

        late = copy.deepcopy(base)
        late["teamOur"]["teamId"] = "economy-joint-late"
        late["roundNo"] = 65
        late["teamOur"]["roles"][0]["backpack"] = ["copper"] * 6
        late["teamOur"]["roles"][1]["backpack"] = ["copper"] * 6
        late["teamOur"]["roles"][1]["pos"] = {"x": 11, "y": 11}
        late_engine = DecisionEngine()
        traces = []
        late_engine.decide(late, trace_sink=traces.append)
        self.assertFalse(any(
            ":joint:" in plan.reason
            for plan in late_engine.state.state.plans.values()
        ))
        self.assertEqual(
            traces[0]["economyPlanning"]["jointStatus"], "return_deadline",
        )

    def test_joint_upgrade_does_not_preempt_missing_third_tower(self):
        # Break caught: a combined upgrade commitment outranks the core tower line.
        payload = economy_payload(round_no=30, worker_pos=(1, 1))
        payload["teamOur"]["teamId"] = "economy-joint-after-towers"
        payload["teamOur"]["roles"] = payload["teamOur"]["roles"][:-1]
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 1, 3, items=("copper",)),
        )
        payload["teamOur"]["roles"][0]["backpack"] = ["copper"]
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 4, "y": 2}, "neutralType": "vendor"},
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 15}]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 20},
        ]

        engine = DecisionEngine()
        engine.decide(payload)

        self.assertFalse(any(
            ":joint:" in plan.reason
            for plan in engine.state.state.plans.values()
        ))

    def test_accepted_mining_and_joint_actions_emit_bounded_diagnostics(self):
        # Break caught: an accepted economic choice cannot be reconstructed from trace.
        mining = economy_payload(round_no=30, worker_pos=(0, 0))
        mining["teamOur"]["teamId"] = "economy-mining-trace"
        mining["mapInfo"]["zones"] = [
            {"pos": {"x": 1, "y": 1}, "neutralType": "stone"},
            {"pos": {"x": 4, "y": 0}, "neutralType": "copper"},
            {"pos": {"x": 6, "y": 0}, "neutralType": "vendor"},
        ]
        mining["vendorShopList"] = [
            {"name": "stone", "price": 1},
            {"name": "copper", "price": 20},
        ]
        with_completed_wall_line(mining)
        mining_trace = []
        DecisionEngine().decide(mining, trace_sink=mining_trace.append)
        mining_action = next(
            action for action in mining_trace[0]["actions"]
            if action["roleId"] == "10010"
        )
        self.assertEqual(mining_action["economy"], {
            "kind": "mining",
            "mineral": "copper",
            "target": {"x": 4, "y": 0},
            "estimatedValue": 20,
            "estimatedRounds": 7,
        })

        joint = economy_payload(round_no=30, worker_pos=(3, 1))
        joint["teamOur"]["teamId"] = "economy-joint-trace"
        joint["teamOur"]["roles"].insert(
            1, role(10012, "worker", 3, 3, items=("copper",) * 6),
        )
        joint["teamOur"]["roles"][0]["backpack"] = ["copper"] * 6
        joint["mapInfo"]["zones"] = [
            {"pos": {"x": 4, "y": 2}, "neutralType": "vendor"},
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        joint["vendorShopList"] = [{"name": "copper", "price": 10}]
        joint["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        joint_trace = []
        DecisionEngine().decide(joint, trace_sink=joint_trace.append)
        details = [action["economy"] for action in joint_trace[0]["actions"]]
        self.assertEqual(len(details), 2)
        self.assertTrue(all(detail["kind"] == "jointFunding" for detail in details))
        self.assertTrue(all(detail["purchase"] == "WeaponUpgradeVoucher1" for detail in details))
        self.assertTrue(all(detail["participants"] == ["10010", "10012"] for detail in details))
        self.assertTrue(all(detail["currentGold"] == 0 for detail in details))
        self.assertEqual(
            {tuple(sorted(detail["expectedContribution"].items())) for detail in details},
            {(('10010', 60), ('10012', 40))},
        )

    def test_cancelled_joint_plan_reports_reason_on_accepted_fallback(self):
        # Break caught: a stale joint commitment disappears without a bounded reason.
        payload = economy_payload(round_no=30, worker_pos=(3, 1))
        payload["teamOur"]["teamId"] = "economy-joint-cancel-trace"
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 3, 3, items=("copper",) * 6),
        )
        payload["teamOur"]["roles"][0]["backpack"] = ["copper"] * 6
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 4, "y": 2}, "neutralType": "vendor"},
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
            {"pos": {"x": 1, "y": 1}, "neutralType": "copper"},
        ]
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        engine = DecisionEngine()
        engine.decide(payload)

        changed = copy.deepcopy(payload)
        changed["roundNo"] = 31
        changed["weaponShopList"][0]["price"] = 130
        changed["lastRoundRoleActionResults"] = {}
        traces = []
        engine.decide(changed, trace_sink=traces.append)

        reasons = {
            action.get("economy", {}).get("jointCancelReason")
            for action in traces[0]["actions"]
        }
        self.assertIn("combined_value_insufficient", reasons)

    def test_successful_joint_use_finishes_with_distinct_bound_posts(self):
        # Break caught: completed joint use is mistaken for target invalidation.
        payload = economy_payload(round_no=41, worker_pos=(1, 1))
        payload["teamOur"]["teamId"] = "economy-joint-return"
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 10, 8),
        )
        next(
            entry for entry in payload["teamOur"]["roles"]
            if entry["id"] == 10020
        )["level"] = 2
        engine = DecisionEngine()
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.set_plan(
            10010,
            Pos(8, 2),
            "fund:WeaponUpgradeVoucher1:10030:10020:joint:10010",
            69,
        )
        engine.state.set_plan(
            10012,
            Pos(8, 8),
            "fund:WeaponUpgradeVoucher1:10040:10020:joint:10010",
            69,
        )
        engine.state.state.action_history.append(CompletedAction(
            PendingAction(
                40,
                10010,
                10010,
                "use",
                Pos(8, 2),
                source_session=engine.state.state.session_index,
                name="WeaponUpgradeVoucher1",
            ),
            True,
        ))

        bound_posts = {10010: 10030, 10012: 10040}
        for _ in range(20):
            response = engine.decide(payload)
            commands = response["roleCommandMap"]
            if _ == 0:
                self.assertNotIn("10012", commands)
            for worker_id, post_id in bound_posts.items():
                worker = next(
                    entry for entry in payload["teamOur"]["roles"]
                    if entry["id"] == worker_id
                )
                post = next(
                    entry for entry in payload["teamOur"]["roles"]
                    if entry["id"] == post_id
                )
                at_post = max(
                    abs(worker["pos"]["x"] - post["pos"]["x"]),
                    abs(worker["pos"]["y"] - post["pos"]["y"]),
                ) == 1
                if not at_post:
                    self.assertEqual(commands[str(worker_id)]["action"], "move")
                    self.assertIn(
                        ":joint:", engine.state.state.plans[worker_id].reason,
                    )
            for raw_id, command in commands.items():
                self.assertEqual(command["action"], "move")
                worker = next(
                    entry for entry in payload["teamOur"]["roles"]
                    if entry["id"] == int(raw_id)
                )
                worker["pos"] = copy.deepcopy(command["targetPos"][0])
            payload["lastRoundRoleActionResults"] = {
                role_id: True for role_id in commands
            }
            payload["roundNo"] += 1
            if all(
                max(
                    abs(next(
                        entry for entry in payload["teamOur"]["roles"]
                        if entry["id"] == worker_id
                    )["pos"][axis] - next(
                        entry for entry in payload["teamOur"]["roles"]
                        if entry["id"] == post_id
                    )["pos"][axis])
                    for axis in ("x", "y")
                ) == 1
                for worker_id, post_id in bound_posts.items()
            ):
                break

        self.assertGreater(_, 0)
        self.assertLess(_, 19)

    def test_joint_route_with_colliding_first_steps_is_not_partly_committed(self):
        # Break caught: only one side of a coordinated route reaches the response.
        payload = economy_payload(round_no=30, worker_pos=(0, 0))
        payload["teamOur"]["teamId"] = "economy-joint-collision"
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 0, 2, items=("copper",) * 6),
        )
        payload["teamOur"]["roles"][0]["backpack"] = ["copper"] * 6
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 2, "y": 1}, "neutralType": "vendor"},
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        payload["teamOur"]["roles"].extend(
            role(11000 + index, "wall", x, y, health=1000)
            for index, (x, y) in enumerate(
                ((1, 0), (2, 0), (3, 0), (3, 1),
                 (3, 2), (2, 2), (1, 2))
            )
        )
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        engine = DecisionEngine()

        response = engine.decide(payload)

        destinations = [
            tuple(command["targetPos"][0].values())
            for command in response["roleCommandMap"].values()
            if command["action"] == "move"
        ]
        self.assertEqual(len(destinations), len(set(destinations)))
        self.assertFalse(any(
            ":joint:" in plan.reason
            for plan in engine.state.state.plans.values()
        ))

    def test_missing_joint_feedback_allows_self_care_without_prepayment(self):
        payload = economy_payload(round_no=30, worker_pos=(3, 1))
        payload["teamOur"]["teamId"] = "economy-joint-self-care"
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 3, 3, items=("copper",) * 6),
        )
        payload["teamOur"]["roles"][0]["backpack"] = ["copper"] * 6
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        engine = DecisionEngine()
        engine.decide(payload)

        changed = copy.deepcopy(payload)
        changed["roundNo"] = 31
        changed["teamOur"]["roles"][0]["health"] = 100
        changed["teamOur"]["roles"][0]["backpack"].append("Medicine")
        changed["lastRoundRoleActionResults"] = {}
        response = engine.decide(changed)

        self.assertEqual(
            response["roleCommandMap"]["10010"],
            {"action": "use", "name": "Medicine"},
        )
        self.assertFalse(any(
            command["action"] == "buy"
            for command in response["roleCommandMap"].values()
        ))
        self.assertFalse(any(
            ":joint:" in plan.reason
            for plan in engine.state.state.plans.values()
        ))

    def test_cancelled_joint_self_care_is_not_duplicated_or_mislabeled(self):
        payload = economy_payload(
            round_no=31,
            worker_pos=(3, 1),
            items=("Medicine",),
        )
        payload["teamOur"]["roles"][0]["health"] = 100
        turn = Turn.load(payload)
        state = state_for(payload)
        state.plans[10010] = PlanState(
            10010,
            Pos(8, 2),
            "fund:WeaponUpgradeVoucher1:10020:10020:joint:10010",
            69,
            state.session_index,
        )

        candidates = economy.propose_economy(
            turn,
            state,
            clock=lambda: 0.0,
            deadline=1.0,
            max_expansions=64,
        )
        self_care = [
            candidate for candidate in candidates
            if candidate.proposal.actor_id == 10010
            and candidate.proposal.command.get("action") == "use"
            and candidate.proposal.command.get("name") == "Medicine"
        ]

        self.assertEqual(len(self_care), 1)
        self.assertNotEqual(
            (self_care[0].diagnostic or {}).get("kind"), "heldInvestment",
        )

    def test_dead_joint_contributor_releases_survivor_without_prepayment(self):
        payload = economy_payload(round_no=30, worker_pos=(3, 1))
        payload["teamOur"]["teamId"] = "economy-joint-death"
        payload["teamOur"]["roles"].insert(
            1, role(10012, "worker", 3, 3, items=("copper",) * 6),
        )
        payload["teamOur"]["roles"][0]["backpack"] = ["copper"] * 6
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        engine = DecisionEngine()
        engine.decide(payload)

        changed = copy.deepcopy(payload)
        changed["roundNo"] = 31
        changed["teamOur"]["roles"][1]["health"] = 0
        changed["lastRoundRoleActionResults"] = {}
        response = engine.decide(changed)

        self.assertFalse(any(
            command["action"] == "buy"
            for command in response["roleCommandMap"].values()
        ))
        self.assertFalse(any(
            ":joint:" in plan.reason
            for plan in engine.state.state.plans.values()
        ))

    def test_joint_buyer_uses_held_voucher_after_contributor_disappears(self):
        # Break caught: losing the contributor hides an already-owned investment.
        payload = economy_payload(
            round_no=40,
            worker_pos=(8, 10),
            items=("StationUpgradeVoucher1",),
        )
        payload["teamOur"]["teamId"] = "economy-held-after-joint"
        engine = DecisionEngine()
        turn = Turn.load(payload)
        engine.state.observe(turn, payload, request_fingerprint(payload))
        engine.state.set_plan(
            10010,
            Pos(9, 9),
            "fund:StationUpgradeVoucher1:10020:10013:joint:10010",
            69,
        )

        response = engine.decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "use",
            "name": "StationUpgradeVoucher1",
            "targetPos": [{"x": 9, "y": 9}],
        })

    def test_failed_sell_feedback_never_assumes_shared_gold_arrived(self):
        engine = DecisionEngine()
        first = economy_payload(
            round_no=40,
            worker_pos=(3, 2),
            items=("copper",) * 10,
            gold=0,
        )
        first["vendorShopList"] = [{"name": "copper", "price": 10}]
        self.assertEqual(
            engine.decide(first)["roleCommandMap"]["10010"]["action"],
            "sell",
        )

        failed = copy.deepcopy(first)
        failed["roundNo"] = 41
        failed["lastRoundRoleActionResults"] = {"10010": False}
        response = engine.decide(failed)

        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "sell")
        self.assertNotEqual(response["roleCommandMap"]["10010"]["action"], "buy")

    def test_disappeared_upgrade_target_cancels_funding_plan(self):
        engine = DecisionEngine()
        first = economy_payload(
            round_no=30,
            worker_pos=(2, 1),
            items=("copper",) * 10,
            gold=0,
        )
        first["vendorShopList"] = [{"name": "copper", "price": 10}]
        engine.decide(first)
        self.assertTrue(engine.state.state.plans[10010].reason.startswith("fund:"))

        disappeared = copy.deepcopy(first)
        disappeared["roundNo"] = 31
        disappeared["teamOur"]["roles"] = disappeared["teamOur"]["roles"][:2]
        response = engine.decide(disappeared)

        self.assertNotEqual(
            engine.state.state.plans.get(10010).reason,
            "fund:WeaponUpgradeVoucher1",
        )
        self.assertNotEqual(response["roleCommandMap"]["10010"]["action"], "buy")

    def test_dead_funding_worker_releases_plan(self):
        engine = DecisionEngine()
        first = economy_payload(
            round_no=30,
            worker_pos=(2, 1),
            items=("copper",) * 10,
            gold=0,
        )
        first["vendorShopList"] = [{"name": "copper", "price": 10}]
        engine.decide(first)

        dead = copy.deepcopy(first)
        dead["roundNo"] = 31
        dead["teamOur"]["roles"][0]["health"] = 0
        response = engine.decide(dead)

        self.assertNotIn(10010, engine.state.state.plans)
        self.assertNotIn("10010", response["roleCommandMap"])

    def test_funding_plan_observes_sell_then_buy_then_use_feedback(self):
        engine = DecisionEngine()
        payload = economy_payload(
            round_no=40,
            worker_pos=(3, 2),
            items=("copper",) * 10,
            gold=0,
        )
        payload["teamOur"]["teamId"] = "economy-full-funding-chain"
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        actions = []

        for _ in range(12):
            response = engine.decide(payload)
            command = response["roleCommandMap"].get("10010")
            self.assertIsNotNone(command)
            action = command["action"]
            actions.append(action)
            if "buy" in actions and action != "use":
                self.assertTrue(
                    engine.state.state.plans[10010].reason.startswith("fund:")
                )
            worker = payload["teamOur"]["roles"][0]
            if action == "move":
                worker["pos"] = copy.deepcopy(command["targetPos"][0])
            elif action == "sell":
                worker["backpack"] = []
                payload["teamOur"]["goldNum"] = 100
            elif action == "buy":
                self.assertEqual(command["name"], "WeaponUpgradeVoucher1")
                worker["backpack"] = ["WeaponUpgradeVoucher1"]
                payload["teamOur"]["goldNum"] = 0
            elif action == "use":
                self.assertEqual(command["name"], "WeaponUpgradeVoucher1")
                worker["backpack"] = []
                payload["teamOur"]["roles"][2]["level"] = 2
                break
            payload["roundNo"] += 1
            payload["lastRoundRoleActionResults"] = {"10010": True}

        self.assertIn("sell", actions)
        self.assertIn("buy", actions)
        self.assertEqual(actions[-1], "use")

        payload["roundNo"] = 71
        payload["lastRoundRoleActionResults"] = {"10010": True}
        payload["teamOur"]["roles"][2]["attackRange"] = 3
        payload["robot"]["roles"] = [{
            "id": 30001,
            "pos": {"x": 9, "y": 3},
            "roleType": "smallRobot",
            "health": 40,
            "attackPower": 5,
            "abnormalState": "",
            "targetTeam": "challenger",
        }]
        response = engine.decide(payload)
        self.assertEqual(
            response["roleCommandMap"]["10020"]["controllerId"], "10010",
        )

    def test_funding_that_cannot_return_from_wall_repair_to_post_is_rejected(self):
        payload = economy_payload(
            round_no=65,
            worker_pos=(3, 2),
            items=("copper",),
            gold=0,
        )
        payload["teamOur"]["teamId"] = "economy-repair-misses-post"
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        payload["weaponShopList"] = [{"name": "WallFixer", "price": 10}]
        payload["teamOur"]["roles"].append(
            role(10050, "wall", 3, 3, health=100)
        )

        engine = DecisionEngine()
        engine.decide(payload)

        plan = engine.state.state.plans.get(10010)
        self.assertTrue(plan is None or not plan.reason.startswith("fund:"))

    def test_active_funding_post_is_not_claimed_by_another_gunner_plan(self):
        # Break caught: defense ignores the post bound into an accepted funding route.
        previous = economy_payload(
            round_no=59,
            worker_pos=(5, 8),
            items=("WeaponUpgradeVoucher1",),
        )
        previous["teamOur"]["teamId"] = "economy-reserved-funding-post"
        previous["teamOur"]["roles"] = [
            role(10010, "worker", 5, 8, items=("WeaponUpgradeVoucher1",)),
            role(10012, "worker", 7, 4),
            role(10013, "station", 9, 9, health=1500),
            role(10020, "gatling", 8, 2, health=1000),
            role(10030, "railgun", 8, 8, health=1000),
        ]
        engine = DecisionEngine()
        initial_turn = Turn.load(previous)
        engine.state.observe(
            initial_turn, previous, request_fingerprint(previous),
        )
        engine.state.set_plan(
            10010,
            Pos(8, 8),
            "fund:WeaponUpgradeVoucher1:10020:10030",
            70,
        )

        current = copy.deepcopy(previous)
        current["roundNo"] = 60
        response = engine.decide(current)

        self.assertTrue(
            engine.state.state.plans[10010].reason.startswith("fund:")
        )
        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")
        self.assertEqual(engine.state.state.plans[10012].reason, "gunner:10030")

    def test_blocked_vendor_route_cannot_start_funding_plan(self):
        payload = economy_payload(
            round_no=30,
            worker_pos=(2, 1),
            items=("copper",) * 10,
            gold=0,
        )
        payload["teamOur"]["teamId"] = "economy-blocked-funding"
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        blockers = (
            (3, 1), (3, 2), (3, 3), (4, 1),
            (4, 3), (5, 1), (5, 2), (5, 3),
        )
        payload["teamOur"]["roles"].extend(
            role(10100 + index, "wall", x, y, health=1000)
            for index, (x, y) in enumerate(blockers)
        )

        engine = DecisionEngine()
        response = engine.decide(payload)

        plan = engine.state.state.plans.get(10010)
        self.assertTrue(plan is None or not plan.reason.startswith("fund:"))
        self.assertNotIn(
            response["roleCommandMap"]["10010"]["action"], {"sell", "buy"},
        )

    def test_night_worker_adjacent_to_shop_can_buy_needed_medicine(self):
        # Break caught: rounds_until_night=0 suppresses every nighttime purchase.
        payload = economy_payload(round_no=71, worker_pos=(5, 2), gold=10)
        payload["teamOur"]["teamId"] = "economy-night-medicine"
        payload["teamOur"]["roles"][0]["health"] = 100
        payload["teamOur"]["roles"] = payload["teamOur"]["roles"][:2]
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        payload["weaponShopList"] = [{"name": "Medicine", "price": 10}]

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "buy",
            "name": "Medicine",
            "num": 1,
        })

    def test_unreachable_nearest_shop_does_not_hide_reachable_shop(self):
        # Break caught: selecting by distance alone stops at an enclosed shop.
        payload = economy_payload(round_no=1, worker_pos=(1, 1), gold=10)
        payload["teamOur"]["teamId"] = "economy-reachable-shop"
        payload["teamOur"]["roles"][0]["health"] = 100
        blocked = [
            {"pos": {"x": x, "y": y}, "neutralType": "vendor"}
            for x in range(2, 5)
            for y in range(0, 3)
            if (x, y) != (3, 1)
        ]
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 3, "y": 1}, "neutralType": "weaponShop"},
            {"pos": {"x": 8, "y": 8}, "neutralType": "weaponShop"},
            *blocked,
        ]
        payload["weaponShopList"] = [{"name": "Medicine", "price": 10}]

        engine = DecisionEngine()
        response = engine.decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")
        self.assertEqual(engine.state.state.plans[10010].target, Pos(8, 8))

    def test_affordable_immediate_repair_is_not_blocked_by_costly_upgrade(self):
        # Break caught: the first logical need is unaffordable and hides WallFixer.
        payload = economy_payload(worker_pos=(5, 2), gold=20)
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        payload["teamOur"]["roles"].append(
            role(10050, "wall", 5, 3, health=100)
        )
        payload["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
            {"name": "WallFixer", "price": 10},
        ]

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "buy",
            "name": "WallFixer",
            "num": 1,
        })

        without_repair = copy.deepcopy(payload)
        without_repair["teamOur"]["teamId"] = "economy-no-forced-buy"
        without_repair["weaponShopList"] = [
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]
        response = DecisionEngine().decide(without_repair)
        self.assertFalse(any(
            command.get("action") == "buy"
            for command in response["roleCommandMap"].values()
        ))

    def test_micro_medicine_purchase_yields_to_next_economic_candidate(self):
        payload = economy_payload(round_no=10, worker_pos=(2, 1), gold=100)
        payload["teamOur"]["roles"][0]["health"] = 219

        actions = economy.propose_economy(
            Turn.load(payload), state_for(payload),
            clock=lambda: 0.0, deadline=1.0, max_expansions=64,
        )
        worker_action = next(
            action for action in actions
            if action.proposal.actor_id == 10010
        )

        self.assertIn("WeaponUpgradeVoucher1", worker_action.plan_reason)
        self.assertNotEqual(
            worker_action.proposal.command.get("name"), "Medicine",
        )

    def test_micro_wall_repair_binds_heavier_worthwhile_wall_instead(self):
        payload = economy_payload(round_no=10, worker_pos=(5, 2), gold=10)
        payload["teamOur"]["roles"].extend((
            role(10050, "wall", 5, 3, health=999),
            role(10051, "wall", 7, 3, health=500),
        ))
        payload["weaponShopList"] = [
            {"name": "WallFixer", "price": 10},
        ]

        actions = economy.propose_economy(
            Turn.load(payload), state_for(payload),
            clock=lambda: 0.0, deadline=1.0, max_expansions=64,
        )
        worker_action = next(
            action for action in actions
            if action.proposal.actor_id == 10010
        )

        self.assertTrue(
            worker_action.plan_reason.startswith("fund:WallFixer:"),
        )
        self.assertEqual(economy._plan_use_target_id(worker_action), 10051)

    def test_full_purchase_cost_does_not_hide_far_heavy_wall(self):
        payload = economy_payload(round_no=10, worker_pos=(1, 1), gold=10)
        payload["teamOur"]["roles"].extend((
            role(10050, "wall", 7, 3, health=997),
            role(10051, "wall", 7, 5, health=500),
        ))
        payload["weaponShopList"] = [
            {"name": "WallFixer", "price": 10},
        ]

        route = economy._purchase_route(
            Turn.load(payload), Turn.load(payload).workers()[0],
            "WallFixer", lambda: 0.0, 1.0, 64,
        )

        self.assertIsNotNone(route)
        self.assertEqual(route.use_target_id, 10051)
        self.assertEqual(route.rounds, 10)

    def test_sale_and_joint_routes_enumerate_wall_value_at_full_cost(self):
        payload = economy_payload(
            round_no=10, worker_pos=(1, 1), items=("copper",), gold=0,
        )
        payload["vendorShopList"] = [{"name": "copper", "price": 10}]
        payload["weaponShopList"] = [
            {"name": "WallFixer", "price": 10},
        ]
        payload["teamOur"]["roles"].extend((
            role(10050, "wall", 7, 3, health=997),
            role(10051, "wall", 7, 5, health=500),
        ))
        turn = Turn.load(payload)
        sale_route = economy._funding_chain(
            turn, turn.workers()[0], "WallFixer", 10,
            lambda: 0.0, 1.0, 64,
        )
        self.assertEqual(sale_route.use_target_id, 10051)
        self.assertEqual(sale_route.rounds, 11)

        payload["teamOur"]["roles"][0]["backpack"] = ["copper"] * 6
        payload["teamOur"]["roles"].insert(
            1, role(10011, "worker", 1, 3, items=("copper",) * 6),
        )
        payload["weaponShopList"] = [
            {"name": "WallFixer", "price": 100},
        ]
        turn = Turn.load(payload)
        joint_route = economy._joint_funding_route(
            turn, turn.workers()[0], turn.workers()[1], "WallFixer", None,
            lambda: 0.0, 1.0, 64,
        )
        self.assertEqual(joint_route.use_target_id, 10051)
        self.assertEqual(
            joint_route.buyer_rounds + joint_route.contributor_rounds, 17,
        )

    def test_only_micro_wall_repair_yields_to_upgrade_candidate(self):
        payload = economy_payload(round_no=10, worker_pos=(5, 2), gold=100)
        payload["teamOur"]["roles"].append(
            role(10050, "wall", 5, 3, health=999),
        )
        payload["weaponShopList"] = [
            {"name": "WallFixer", "price": 10},
            {"name": "WeaponUpgradeVoucher1", "price": 100},
        ]

        actions = economy.propose_economy(
            Turn.load(payload), state_for(payload),
            clock=lambda: 0.0, deadline=1.0, max_expansions=64,
        )
        worker_action = next(
            action for action in actions
            if action.proposal.actor_id == 10010
        )

        self.assertIn("WeaponUpgradeVoucher1", worker_action.plan_reason)

    def test_held_micro_maintenance_items_bypass_new_purchase_gate(self):
        medicine = economy_payload(items=("Medicine",))
        medicine["teamOur"]["roles"][0]["health"] = 219
        medicine_action = economy.propose_economy(
            Turn.load(medicine), state_for(medicine),
            clock=lambda: 0.0, deadline=1.0, max_expansions=64,
        )[0]
        self.assertEqual(medicine_action.proposal.command, {
            "action": "use", "name": "Medicine",
        })

        fixer = economy_payload(worker_pos=(7, 8), items=("WallFixer",))
        fixer["teamOur"]["roles"].append(
            role(10050, "wall", 7, 9, health=999),
        )
        fixer_action = economy.propose_economy(
            Turn.load(fixer), state_for(fixer),
            clock=lambda: 0.0, deadline=1.0, max_expansions=64,
        )[0]
        self.assertEqual(fixer_action.proposal.command["name"], "WallFixer")

    def test_emergency_medicine_purchase_remains_eligible(self):
        payload = economy_payload(round_no=10, worker_pos=(1, 1), gold=10)
        payload["teamOur"]["roles"][0]["health"] = 40
        payload["weaponShopList"] = [
            {"name": "Medicine", "price": 10},
        ]

        actions = economy.propose_economy(
            Turn.load(payload), state_for(payload),
            clock=lambda: 0.0, deadline=1.0, max_expansions=64,
        )
        worker_action = next(
            action for action in actions
            if action.proposal.actor_id == 10010
        )

        self.assertTrue(worker_action.plan_reason.startswith("fund:Medicine:"))

    def test_heavy_medicine_need_still_starts_complete_funding_route(self):
        payload = economy_payload(round_no=10, worker_pos=(2, 1), gold=10)
        payload["teamOur"]["roles"][0]["health"] = 100
        payload["weaponShopList"] = [
            {"name": "Medicine", "price": 10},
        ]

        actions = economy.propose_economy(
            Turn.load(payload), state_for(payload),
            clock=lambda: 0.0, deadline=1.0, max_expansions=64,
        )
        worker_action = next(
            action for action in actions
            if action.proposal.actor_id == 10010
        )

        self.assertTrue(worker_action.plan_reason.startswith("fund:Medicine:"))

    def test_existing_micro_medicine_funding_commitment_is_not_regated(self):
        payload = economy_payload(round_no=10, worker_pos=(2, 1), gold=10)
        payload["teamOur"]["roles"][0]["health"] = 219
        payload["weaponShopList"] = [
            {"name": "Medicine", "price": 10},
        ]
        state = state_for(payload)
        state.plans[10010] = PlanState(
            10010, Pos(6, 2), "fund:Medicine:10020:0", 69,
            source_session=state.session_index,
        )

        actions = economy.propose_economy(
            Turn.load(payload), state,
            clock=lambda: 0.0, deadline=1.0, max_expansions=64,
        )
        worker_action = next(
            action for action in actions
            if action.proposal.actor_id == 10010
        )

        self.assertTrue(worker_action.plan_reason.startswith("fund:Medicine:"))

    def test_two_workers_cannot_bypass_micro_medicine_gate(self):
        payload = economy_payload(round_no=10, worker_pos=(5, 2), gold=20)
        payload["teamOur"]["roles"][0]["health"] = 219
        payload["teamOur"]["roles"].insert(
            1, role(10011, "worker", 5, 3, health=219),
        )
        payload["weaponShopList"] = [
            {"name": "Medicine", "price": 10},
        ]

        response = DecisionEngine().decide(payload)

        self.assertFalse(any(
            command.get("action") == "buy"
            and command.get("name") == "Medicine"
            for command in response["roleCommandMap"].values()
        ))

    def test_night_micro_medicine_purchase_uses_same_value_gate(self):
        payload = economy_payload(round_no=71, worker_pos=(5, 2), gold=10)
        payload["teamOur"]["teamId"] = "night-micro-medicine-gate"
        payload["teamOur"]["roles"][0]["health"] = 219
        payload["teamOur"]["roles"] = payload["teamOur"]["roles"][:2]
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 6, "y": 2}, "neutralType": "weaponShop"},
        ]
        payload["weaponShopList"] = [
            {"name": "Medicine", "price": 10},
        ]

        response = DecisionEngine().decide(payload)

        self.assertFalse(any(
            command.get("action") == "buy"
            and command.get("name") == "Medicine"
            for command in response["roleCommandMap"].values()
        ))

        emergency = copy.deepcopy(payload)
        emergency["teamOur"]["teamId"] = "night-emergency-medicine"
        emergency["teamOur"]["roles"][0]["health"] = 40
        response = DecisionEngine().decide(emergency)
        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "buy", "name": "Medicine", "num": 1,
        })

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
            "fund:WeaponUpgradeVoucher1:10020:10020",
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
            "fund:WeaponUpgradeVoucher1:10020:10020",
        )

        second = copy.deepcopy(first)
        second["roundNo"] = 2
        second["teamOur"]["goldNum"] = 0
        second["teamOur"]["roles"][0]["pos"] = {"x": 5, "y": 2}
        second["lastRoundRoleActionResults"] = {"10010": True}
        response = engine.decide(second)

        command = response["roleCommandMap"].get("10010", {})
        self.assertNotEqual(command.get("action"), "buy")

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
