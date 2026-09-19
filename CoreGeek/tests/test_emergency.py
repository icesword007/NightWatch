import copy
import json
import time
import unittest
from pathlib import Path

from agent.brain import DecisionEngine
from agent.emergency import propose_held_emergency
from agent.protocol import Turn
from agent.state import StateStore, request_fingerprint


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def unit(unit_id, kind, x, y, *, health=220, cooldown=0, items=()):
    return {
        "id": unit_id,
        "pos": {"x": x, "y": y},
        "roleType": kind,
        "health": health,
        "attackPower": 20 if kind == "rocket" else 10,
        "attackRange": 0,
        "backPackCapability": 100 if kind == "worker" else 40,
        "backpack": list(items),
        "level": 1,
        "cooldown": cooldown,
    }


def robot(robot_id, x, y, health, *, target="challenger", abnormal=""):
    return {
        "id": robot_id,
        "pos": {"x": x, "y": y},
        "roleType": "middleRobot",
        "health": health,
        "abnormalState": abnormal,
        "targetTeam": target,
    }


def emergency_payload(*, round_no=71, item="Bomb", holder_pos=(6, 5)):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["roundNo"] = round_no
    payload["mapInfo"].update({"width": 30, "height": 30, "zones": []})
    payload["teamOur"].update({
        "teamId": "emergency-tests",
        "type": "challenger",
        "goldNum": 0,
        "playerTasks": [],
        "roles": [
            unit(10010, "worker", *holder_pos, items=(item,)),
            unit(10012, "worker", 9, 5),
            unit(10011, "pioneer", 12, 5),
            unit(10013, "station", 8, 9, health=1500),
            unit(10020, "gatling", 6, 6, health=1000, cooldown=3),
            unit(10030, "railgun", 9, 6, health=1000, cooldown=3),
            unit(10040, "rocket", 12, 6, health=1000, cooldown=3),
        ],
    })
    payload["teamEnemy"]["roles"] = []
    payload["robot"]["roles"] = [robot(30001, 8, 7, 80)]
    payload["phaseTask"] = ""
    payload["llmResp"] = ""
    payload["lastCmdResult"] = ""
    payload["errors"] = []
    payload["weaponShopList"] = []
    payload["vendorShopList"] = []
    return payload


def state_for(payload):
    store = StateStore()
    turn = Turn.load(payload)
    store.observe(turn, payload, request_fingerprint(payload))
    return store.state


def purchase_payload(*, round_no=71, gold=200):
    payload = emergency_payload(round_no=round_no, item="Bomb", holder_pos=(1, 1))
    payload["teamOur"]["roles"][0]["backpack"] = []
    payload["teamOur"]["goldNum"] = gold
    payload["mapInfo"]["zones"] = [{
        "pos": {"x": 1, "y": 2}, "neutralType": "weaponShop",
    }] + [
        {"pos": {"x": x, "y": y}, "neutralType": "blocked"}
        for x in (5, 6, 7) for y in (5, 6, 7) if (x, y) != (6, 6)
    ]
    payload["weaponShopList"] = [
        {"name": "Bomb", "price": 100},
        {"name": "DizzyWeapon", "price": 100},
    ]
    return payload


class HeldEmergencyTests(unittest.TestCase):
    def test_real_engine_buys_one_bomb_for_urgent_kill(self):
        payload = purchase_payload()
        traces = []

        response = DecisionEngine().decide(payload, trace_sink=traces.append)

        purchases = [
            command for command in response["roleCommandMap"].values()
            if command["action"] == "buy"
            and command.get("name") in ("Bomb", "DizzyWeapon")
        ]
        self.assertEqual(purchases, [{"action": "buy", "name": "Bomb", "num": 1}])
        self.assertEqual(response["roleCommandMap"]["10010"], purchases[0])
        self.assertFalse(any(
            command["action"] == "use"
            and command.get("name") in ("Bomb", "DizzyWeapon")
            for command in response["roleCommandMap"].values()
        ))
        purchase_trace = next(
            action for action in traces[-1]["actions"]
            if action["reason"] == "emergency"
        )
        self.assertEqual(purchase_trace["emergency"], {
            "kind": "emergencyPurchase", "item": "Bomb", "price": 100,
            "affectedUrgent": 1, "killedUrgent": 1,
        })

    def test_purchase_uses_only_arrived_inventory_and_never_rebuys_that_night(self):
        engine = DecisionEngine()
        first = purchase_payload()
        self.assertEqual(
            engine.decide(first)["roleCommandMap"]["10010"],
            {"action": "buy", "name": "Bomb", "num": 1},
        )
        arrived = purchase_payload(round_no=72)
        arrived["teamOur"]["roles"][0]["backpack"] = ["Bomb"]
        arrived["teamOur"]["goldNum"] = 100
        arrived["lastRoundRoleActionResults"] = {"10010": True}
        self.assertEqual(
            engine.decide(arrived)["roleCommandMap"]["10010"],
            {"action": "use", "name": "Bomb", "targetPos": [{"x": 8, "y": 7}]},
        )
        consumed = purchase_payload(round_no=73)
        consumed["teamOur"]["goldNum"] = 100
        consumed["lastRoundRoleActionResults"] = {"10010": True}
        response = engine.decide(consumed)
        self.assertFalse(any(
            command.get("name") in ("Bomb", "DizzyWeapon")
            for command in response["roleCommandMap"].values()
        ))

    def test_failed_or_missing_delivery_does_not_use_or_repeat_purchase(self):
        for feedback in (False, True):
            with self.subTest(feedback=feedback):
                engine = DecisionEngine()
                engine.decide(purchase_payload())
                missing = purchase_payload(round_no=72)
                missing["lastRoundRoleActionResults"] = {"10010": feedback}
                response = engine.decide(missing)
                self.assertFalse(any(
                    command.get("name") in ("Bomb", "DizzyWeapon")
                    for command in response["roleCommandMap"].values()
                ))

    def test_purchase_falls_back_to_dizzy_only_when_bomb_cannot_kill(self):
        payload = purchase_payload()
        payload["robot"]["roles"][0]["health"] = 101

        self.assertEqual(
            DecisionEngine().decide(payload)["roleCommandMap"]["10010"],
            {"action": "buy", "name": "DizzyWeapon", "num": 1},
        )

    def test_purchase_requires_current_threat_shop_capacity_and_price(self):
        cases = []
        day = purchase_payload(round_no=1)
        cases.append(("day", day))
        last_night_round = purchase_payload(round_no=130)
        cases.append(("last_night_round", last_night_round))
        far = purchase_payload()
        far["robot"]["roles"][0]["pos"] = {"x": 1, "y": 20}
        cases.append(("far", far))
        enemy = purchase_payload()
        enemy["robot"]["roles"][0]["targetTeam"] = "defender"
        cases.append(("enemy", enemy))
        dizzy = purchase_payload()
        dizzy["robot"]["roles"][0]["abnormalState"] = "dizzy"
        cases.append(("dizzy", dizzy))
        no_base = purchase_payload()
        no_base["teamOur"]["roles"] = [
            role for role in no_base["teamOur"]["roles"]
            if role["roleType"] != "station"
        ]
        cases.append(("no_base", no_base))
        missing_shop = purchase_payload()
        missing_shop["mapInfo"]["zones"] = [
            zone for zone in missing_shop["mapInfo"]["zones"]
            if zone["neutralType"] != "weaponShop"
        ]
        cases.append(("missing_shop", missing_shop))
        distant_shop = purchase_payload()
        distant_shop["mapInfo"]["zones"][0]["pos"] = {"x": 1, "y": 3}
        cases.append(("distant_shop", distant_shop))
        full = purchase_payload()
        full["teamOur"]["roles"][0]["backPackCapability"] = 0
        cases.append(("full", full))
        no_listing = purchase_payload()
        no_listing["weaponShopList"] = []
        cases.append(("no_listing", no_listing))
        no_funds = purchase_payload(gold=99)
        cases.append(("no_funds", no_funds))
        held = purchase_payload()
        held["teamOur"]["roles"][1]["backpack"] = ["Bomb"]
        cases.append(("held", held))
        for label, payload in cases:
            with self.subTest(label=label):
                response = DecisionEngine().decide(payload)
                self.assertFalse(any(
                    command["action"] == "buy"
                    and command.get("name") in ("Bomb", "DizzyWeapon")
                    for command in response["roleCommandMap"].values()
                ))

    def test_purchase_does_not_displace_gunner_return_or_task_owner(self):
        returning = purchase_payload()
        returning["mapInfo"]["zones"] = returning["mapInfo"]["zones"][:1]
        response = DecisionEngine().decide(returning)
        self.assertEqual(response["roleCommandMap"]["10010"]["action"], "move")

        task = purchase_payload()
        task["teamOur"]["roles"][0]["pos"] = {"x": 12, "y": 5}
        task["teamOur"]["roles"][2]["pos"] = {"x": 1, "y": 1}
        task["phaseTask"] = "solve active task"
        response = DecisionEngine().decide(task)
        self.assertFalse(any(
            command["action"] == "buy"
            and command.get("name") in ("Bomb", "DizzyWeapon")
            for command in response["roleCommandMap"].values()
        ))

    def test_purchase_preserves_missing_tower_funds(self):
        payload = purchase_payload(gold=124)
        payload["teamOur"]["roles"] = [
            role for role in payload["teamOur"]["roles"]
            if role["roleType"] != "gatling"
        ]
        self.assertFalse(any(
            command["action"] == "buy"
            and command.get("name") in ("Bomb", "DizzyWeapon")
            for command in DecisionEngine().decide(payload)["roleCommandMap"].values()
        ))
        payload["teamOur"]["goldNum"] = 125
        self.assertEqual(
            DecisionEngine().decide(payload)["roleCommandMap"]["10010"],
            {"action": "buy", "name": "Bomb", "num": 1},
        )

    def test_attack_self_care_and_deadline_prevent_procurement(self):
        attack = purchase_payload()
        next(
            role for role in attack["teamOur"]["roles"]
            if role["roleType"] == "railgun"
        )["cooldown"] = 0
        response = DecisionEngine().decide(attack)
        self.assertTrue(any(
            command["action"] == "attack"
            for command in response["roleCommandMap"].values()
        ))
        self.assertFalse(any(
            command["action"] == "buy"
            and command.get("name") in ("Bomb", "DizzyWeapon")
            for command in response["roleCommandMap"].values()
        ))

        self_care = purchase_payload()
        self_care["teamOur"]["roles"][0]["health"] = 20
        self_care["teamOur"]["roles"][0]["backpack"] = ["Medicine"]
        response = DecisionEngine().decide(self_care)
        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "use", "name": "Medicine",
        })

        expired = DecisionEngine(clock=lambda: 10.0, budget_seconds=0.0)
        self.assertFalse(any(
            command["action"] == "buy"
            and command.get("name") in ("Bomb", "DizzyWeapon")
            for command in expired.decide(purchase_payload())["roleCommandMap"].values()
        ))

    def test_two_eligible_buyers_choose_one_stably_and_replay_is_cached(self):
        choices = []
        for reverse in (False, True):
            payload = purchase_payload()
            payload["teamOur"]["roles"][1]["pos"] = {"x": 2, "y": 1}
            payload["teamOur"]["roles"][2]["pos"] = {"x": 20, "y": 20}
            payload["mapInfo"]["zones"].extend(
                {"pos": {"x": x, "y": y}, "neutralType": "blocked"}
                for cx, cy in ((9, 6), (12, 6))
                for x in (cx - 1, cx, cx + 1)
                for y in (cy - 1, cy, cy + 1)
                if (x, y) != (cx, cy)
            )
            if reverse:
                payload["teamOur"]["roles"].reverse()
            engine = DecisionEngine()
            response = engine.decide(payload)
            self.assertEqual(engine.decide(copy.deepcopy(payload)), response)
            purchases = [
                (role_id, command)
                for role_id, command in response["roleCommandMap"].items()
                if command["action"] == "buy"
                and command.get("name") in ("Bomb", "DizzyWeapon")
            ]
            self.assertEqual(len(purchases), 1)
            choices.append(purchases[0])
        self.assertEqual(choices[0], choices[1])
        self.assertEqual(choices[0][0], "10010")

    def test_purchase_limit_resets_only_for_a_new_night_or_session(self):
        engine = DecisionEngine()
        engine.decide(purchase_payload())
        later = purchase_payload(round_no=73)
        self.assertFalse(any(
            command["action"] == "buy"
            and command.get("name") in ("Bomb", "DizzyWeapon")
            for command in engine.decide(later)["roleCommandMap"].values()
        ))
        new_night = purchase_payload(round_no=201)
        self.assertEqual(
            engine.decide(new_night)["roleCommandMap"]["10010"]["name"],
            "Bomb",
        )
        new_session = purchase_payload()
        new_session["teamOur"]["teamId"] = "new-match"
        self.assertEqual(
            engine.decide(new_session)["roleCommandMap"]["10010"]["name"],
            "Bomb",
        )

    def test_prior_medicine_purchase_preserves_gold_and_does_not_mark_emergency(self):
        def with_medicine_buyer(payload):
            buyer = payload["teamOur"]["roles"][1]
            buyer["pos"] = {"x": 2, "y": 1}
            buyer["health"] = 40
            payload["weaponShopList"].append({"name": "Medicine", "price": 10})
            payload["mapInfo"]["zones"].extend(
                {"pos": {"x": x, "y": y}, "neutralType": "blocked"}
                for x in (8, 9, 10) for y in (5, 6, 7)
                if (x, y) != (9, 6)
            )
            return payload

        engine = DecisionEngine()
        first = with_medicine_buyer(purchase_payload(gold=100))
        response = engine.decide(first)
        self.assertEqual(response["roleCommandMap"]["10012"], {
            "action": "buy", "name": "Medicine", "num": 1,
        })
        self.assertFalse(any(
            command.get("name") in ("Bomb", "DizzyWeapon")
            for command in response["roleCommandMap"].values()
        ))
        self.assertIsNone(engine.state.state.emergency_purchase_night)

        funded = with_medicine_buyer(purchase_payload(round_no=72, gold=100))
        funded["teamOur"]["roles"][1]["backpack"] = ["Medicine"]
        funded["lastRoundRoleActionResults"] = {"10012": True}
        response = engine.decide(funded)
        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "buy", "name": "Bomb", "num": 1,
        })
        self.assertEqual(response["roleCommandMap"]["10012"], {
            "action": "use", "name": "Medicine",
        })

    def test_arrived_bomb_is_not_forced_when_urgent_threat_has_disappeared(self):
        engine = DecisionEngine()
        engine.decide(purchase_payload())
        cleared = purchase_payload(round_no=72, gold=100)
        cleared["teamOur"]["roles"][0]["backpack"] = ["Bomb"]
        cleared["robot"]["roles"] = []
        cleared["lastRoundRoleActionResults"] = {"10010": True}
        response = engine.decide(cleared)
        self.assertFalse(any(
            command.get("name") == "Bomb"
            for command in response["roleCommandMap"].values()
        ))

    def test_real_engine_uses_held_bomb_during_cooling_attack_gap(self):
        payload = emergency_payload()
        traces = []

        response = DecisionEngine().decide(payload, trace_sink=traces.append)

        uses = [
            command for command in response["roleCommandMap"].values()
            if command["action"] == "use" and command.get("name") == "Bomb"
        ]
        self.assertEqual(uses, [{
            "action": "use", "name": "Bomb", "targetPos": [{"x": 8, "y": 7}],
        }])
        self.assertEqual(response["roleCommandMap"]["10010"], uses[0])
        emergency_trace = next(
            action for action in traces[-1]["actions"]
            if action["reason"] == "emergency"
        )
        self.assertEqual(emergency_trace["emergency"], {
            "kind": "heldEmergency",
            "item": "Bomb",
            "affectedUrgent": 1,
            "killedUrgent": 1,
        })

    def test_bomb_feedback_inventory_and_robot_state_prevent_repeat(self):
        engine = DecisionEngine()
        first = emergency_payload()
        self.assertEqual(
            engine.decide(first)["roleCommandMap"]["10010"]["name"], "Bomb",
        )
        second = emergency_payload(round_no=72)
        second["teamOur"]["roles"][0]["backpack"] = []
        second["robot"]["roles"] = []
        second["lastRoundRoleActionResults"] = {"10010": True}

        response = engine.decide(second)

        self.assertFalse(any(
            command.get("name") == "Bomb"
            for command in response["roleCommandMap"].values()
        ))

        failed_engine = DecisionEngine()
        failed_first = emergency_payload()
        self.assertEqual(
            failed_engine.decide(failed_first)["roleCommandMap"]["10010"]["name"],
            "Bomb",
        )
        failed_second = emergency_payload(round_no=72)
        failed_second["lastRoundRoleActionResults"] = {"10010": False}
        self.assertFalse(any(
            command.get("name") == "Bomb"
            for command in failed_engine.decide(failed_second)["roleCommandMap"].values()
        ))

    def test_dizzy_uses_real_state_and_can_be_reconsidered_after_recovery(self):
        engine = DecisionEngine()
        first = emergency_payload(item="DizzyWeapon", holder_pos=(1, 1))
        first["teamOur"]["roles"][0]["backpack"] = [
            "DizzyWeapon", "DizzyWeapon",
        ]
        first["teamOur"]["roles"][1]["pos"] = {"x": 1, "y": 4}
        response = engine.decide(first)
        self.assertEqual(response["roleCommandMap"]["10010"], {
            "action": "use",
            "name": "DizzyWeapon",
            "targetPos": [{"x": 8, "y": 7}],
        })
        self.assertEqual(
            response["roleCommandMap"]["10012"]["action"], "move",
        )

        dizzy = emergency_payload(
            round_no=72, item="DizzyWeapon", holder_pos=(1, 1),
        )
        dizzy["robot"]["roles"][0]["abnormalState"] = "dizzy"
        self.assertFalse(any(
            command.get("name") == "DizzyWeapon"
            for command in engine.decide(dizzy)["roleCommandMap"].values()
        ))

        recovered = emergency_payload(
            round_no=73, item="DizzyWeapon", holder_pos=(1, 1),
        )
        self.assertEqual(
            engine.decide(recovered)["roleCommandMap"]["10010"]["name"],
            "DizzyWeapon",
        )

    def test_normal_weapon_attack_prevents_every_emergency_item(self):
        payload = emergency_payload()
        gatling = next(
            role for role in payload["teamOur"]["roles"] if role["id"] == 10020
        )
        gatling["cooldown"] = 0
        payload["robot"]["roles"][0]["pos"] = {"x": 6, "y": 7}

        response = DecisionEngine().decide(payload)

        self.assertTrue(any(
            command["action"] == "attack"
            for command in response["roleCommandMap"].values()
        ))
        self.assertFalse(any(
            command.get("name") in ("Bomb", "DizzyWeapon")
            for command in response["roleCommandMap"].values()
        ))

    def test_task_owner_and_medicine_holder_are_not_taken(self):
        task = emergency_payload(item="Bomb")
        pioneer = next(
            role for role in task["teamOur"]["roles"] if role["id"] == 10011
        )
        pioneer["backpack"] = ["Bomb"]
        task["teamOur"]["roles"][0]["backpack"] = []
        task["phaseTask"] = "active task"
        task_response = DecisionEngine().decide(task)
        self.assertFalse(any(
            command.get("name") == "Bomb"
            for command in task_response["roleCommandMap"].values()
        ))

        medicine = emergency_payload(item="Bomb")
        holder = medicine["teamOur"]["roles"][0]
        holder["health"] = 40
        holder["backpack"] = ["Medicine", "Bomb"]
        medicine_response = DecisionEngine().decide(medicine)
        self.assertEqual(medicine_response["roleCommandMap"]["10010"], {
            "action": "use", "name": "Medicine",
        })

    def test_two_holders_and_input_reordering_choose_one_stable_action(self):
        choices = []
        for reverse in (False, True):
            payload = emergency_payload()
            payload["teamOur"]["roles"][1]["backpack"] = ["Bomb"]
            if reverse:
                payload["teamOur"]["roles"].reverse()
                payload["robot"]["roles"].reverse()
            response = DecisionEngine().decide(payload)
            emergency = [
                (owner, command) for owner, command in response["roleCommandMap"].items()
                if command.get("name") in ("Bomb", "DizzyWeapon")
            ]
            self.assertEqual(len(emergency), 1)
            choices.append(emergency[0])
        self.assertEqual(choices[0], choices[1])
        self.assertEqual(choices[0][0], "10010")

    def test_bomb_is_preferred_when_it_can_kill_an_urgent_robot(self):
        payload = emergency_payload()
        payload["teamOur"]["roles"][0]["backpack"] = [
            "DizzyWeapon", "Bomb",
        ]
        payload["robot"]["roles"].append(robot(30002, 9, 7, 500))

        response = DecisionEngine().decide(payload)

        self.assertEqual(response["roleCommandMap"]["10010"]["name"], "Bomb")

        no_kill = emergency_payload()
        no_kill["teamOur"]["roles"][0]["backpack"] = [
            "DizzyWeapon", "Bomb",
        ]
        no_kill["robot"]["roles"][0]["health"] = 101
        self.assertEqual(
            DecisionEngine().decide(no_kill)["roleCommandMap"]["10010"]["name"],
            "DizzyWeapon",
        )

    def test_nearest_urgent_layer_precedes_larger_farther_cluster(self):
        payload = emergency_payload(item="DizzyWeapon")
        payload["robot"]["roles"] = [
            robot(30001, 8, 7, 500),
            robot(30002, 11, 7, 500),
            robot(30003, 12, 7, 500),
        ]

        response = DecisionEngine().decide(payload)

        self.assertEqual(
            response["roleCommandMap"]["10010"]["targetPos"],
            [{"x": 8, "y": 7}],
        )

    def test_held_voucher_and_shop_listing_do_not_start_emergency_procurement(self):
        held = emergency_payload()
        holder = held["teamOur"]["roles"][0]
        holder["backpack"] = ["Bomb", "WeaponUpgradeVoucher1"]
        response = DecisionEngine().decide(held)
        self.assertEqual(
            response["roleCommandMap"]["10010"]["name"],
            "WeaponUpgradeVoucher1",
        )

        shop_only = emergency_payload()
        shop_only["teamOur"]["roles"][0]["backpack"] = []
        shop_only["teamOur"]["goldNum"] = 1_000
        shop_only["mapInfo"]["zones"] = [{
            "pos": {"x": 6, "y": 4}, "neutralType": "weaponShop",
        }]
        shop_only["weaponShopList"] = [
            {"name": "Bomb", "price": 100},
            {"name": "DizzyWeapon", "price": 100},
        ]
        shop_response = DecisionEngine().decide(shop_only)
        self.assertFalse(any(
            command.get("name") in ("Bomb", "DizzyWeapon")
            for command in shop_response["roleCommandMap"].values()
        ))

    def test_trigger_boundaries_do_not_create_emergency_action(self):
        cases = []
        day = emergency_payload(round_no=1)
        cases.append(("day", day))
        far = emergency_payload()
        far["robot"]["roles"][0]["pos"] = {"x": 1, "y": 1}
        cases.append(("far", far))
        enemy = emergency_payload()
        enemy["robot"]["roles"][0]["targetTeam"] = "defender"
        cases.append(("enemy", enemy))
        dizzy = emergency_payload(item="DizzyWeapon")
        dizzy["robot"]["roles"][0]["abnormalState"] = "dizzy"
        cases.append(("dizzy", dizzy))
        no_base = emergency_payload()
        no_base["teamOur"]["roles"] = [
            role for role in no_base["teamOur"]["roles"]
            if role["roleType"] != "station"
        ]
        cases.append(("no_base", no_base))
        destroyed_base = emergency_payload()
        next(
            role for role in destroyed_base["teamOur"]["roles"]
            if role["roleType"] == "station"
        )["health"] = 0
        cases.append(("destroyed_base", destroyed_base))
        no_inventory = emergency_payload()
        no_inventory["teamOur"]["roles"][0]["backpack"] = []
        cases.append(("no_inventory", no_inventory))
        dead_holder = emergency_payload()
        dead_holder["teamOur"]["roles"][0]["health"] = 0
        cases.append(("dead_holder", dead_holder))
        out_of_bounds_robot = emergency_payload()
        out_of_bounds_robot["robot"]["roles"][0]["pos"] = {"x": 30, "y": 7}
        cases.append(("out_of_bounds_robot", out_of_bounds_robot))
        too_healthy_for_bomb = emergency_payload()
        too_healthy_for_bomb["robot"]["roles"][0]["health"] = 101
        cases.append(("bomb_cannot_kill", too_healthy_for_bomb))

        for label, payload in cases:
            with self.subTest(label=label):
                response = DecisionEngine().decide(payload)
                self.assertFalse(any(
                    command.get("name") in ("Bomb", "DizzyWeapon")
                    for command in response["roleCommandMap"].values()
                ))

    def test_reserved_weapon_or_expired_deadline_returns_no_candidate(self):
        payload = emergency_payload()
        turn = Turn.load(payload)
        state = state_for(payload)
        self.assertIsNone(propose_held_emergency(
            turn,
            state,
            defense_actions=(),
            reserved_weapon_ids=frozenset((10020,)),
            clock=lambda: 0.0,
            deadline=1.0,
        ))
        self.assertIsNone(propose_held_emergency(
            turn,
            state,
            defense_actions=(),
            clock=lambda: 2.0,
            deadline=1.0,
        ))

    def test_512_robot_scan_is_bounded_and_returns_legal_single_target(self):
        payload = emergency_payload()
        payload["robot"]["roles"] = [
            robot(30_000 + index, 8 + index % 2, 7 + index % 3, 80)
            for index in range(512)
        ]
        turn = Turn.load(payload)
        started = time.monotonic()

        proposal = propose_held_emergency(
            turn,
            state_for(payload),
            defense_actions=(),
            clock=time.monotonic,
            deadline=started + 1.0,
        )

        self.assertIsNotNone(proposal)
        self.assertEqual(len(proposal.proposal.command["targetPos"]), 1)
        self.assertLess(time.monotonic() - started, 1.0)


if __name__ == "__main__":
    unittest.main()
