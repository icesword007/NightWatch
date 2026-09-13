import copy
import importlib
import json
import unittest
from pathlib import Path

from agent.protocol import Pos, Turn


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def load_actions_module(test_case):
    try:
        return importlib.import_module("agent.actions")
    except ModuleNotFoundError:
        test_case.fail("agent.actions is missing")


def two_worker_turn(*, gold=25, first_items=(), second_items=(), zones=()):
    payload = load_fixture()
    payload["teamOur"]["goldNum"] = gold
    first = payload["teamOur"]["roles"][0]
    first["backpack"] = list(first_items)
    second = copy.deepcopy(first)
    second["id"] = 10012
    second["pos"] = {"x": 3, "y": 2}
    second["backpack"] = list(second_items)
    payload["teamOur"]["roles"].append(second)
    payload["mapInfo"]["zones"].extend(zones)
    payload["weaponShopList"] = [
        {"name": "Medicine", "price": 10},
    ]
    payload["vendorShopList"] = [
        {"name": "stone", "price": 1},
    ]
    return Turn.load(payload)


def turn_with_weapon():
    payload = load_fixture()
    payload["roundNo"] = 71
    controller = payload["teamOur"]["roles"][0]
    weapon = copy.deepcopy(controller)
    weapon.update({
        "id": 10020,
        "pos": {"x": 1, "y": 2},
        "roleType": "gatling",
        "health": 300,
        "level": 1,
        "backPackCapability": 0,
        "backpack": [],
    })
    payload["teamOur"]["roles"].append(weapon)
    return Turn.load(payload)


class ActionTests(unittest.TestCase):
    def test_owner_actor_and_action_shape_are_validated(self):
        # Break caught: a valid actor launders a command for a nonexistent owner.
        actions = load_actions_module(self)
        allocator = actions.ActionAllocator(two_worker_turn())

        self.assertFalse(allocator.try_add(actions.ActionProposal(
            command_owner_id=99999,
            actor_id=10010,
            command={"action": "move", "targetPos": [{"x": 2, "y": 1}]},
            destination=Pos(2, 1),
        )))
        self.assertFalse(allocator.try_add(actions.ActionProposal(
            command_owner_id=10010,
            actor_id=10012,
            command={"action": "move", "targetPos": [{"x": 2, "y": 1}]},
            destination=Pos(2, 1),
        )))
        self.assertFalse(allocator.try_add(actions.ActionProposal(
            command_owner_id=10010,
            actor_id=10010,
            command={"action": "not-an-action"},
        )))
        self.assertFalse(allocator.try_add(actions.ActionProposal(
            command_owner_id=10010,
            actor_id=10010,
            command={"action": "move"},
        )))

    def test_target_reservation_must_match_command_target(self):
        # Break caught: allocator reserves one cell but emits a different target.
        actions = load_actions_module(self)
        allocator = actions.ActionAllocator(two_worker_turn())

        self.assertFalse(allocator.try_add(actions.ActionProposal(
            command_owner_id=10010,
            actor_id=10010,
            command={"action": "move", "targetPos": [{"x": 3, "y": 1}]},
            destination=Pos(2, 1),
        )))

    def test_attack_requires_owned_weapon_and_matching_controller(self):
        # Break caught: an arbitrary command owner is accepted as a weapon.
        actions = load_actions_module(self)
        command = {
            "action": "attack",
            "controllerId": "10010",
            "targetPos": [{"x": 4, "y": 4}],
        }
        invalid = actions.ActionAllocator(turn_with_weapon())
        self.assertFalse(invalid.try_add(actions.ActionProposal(
            command_owner_id=99999,
            actor_id=10010,
            command=command,
        )))

        valid = actions.ActionAllocator(turn_with_weapon())
        self.assertTrue(valid.try_add(actions.ActionProposal(
            command_owner_id=10020,
            actor_id=10010,
            command=command,
        )))

    def test_accepted_command_is_isolated_from_caller_mutation(self):
        # Break caught: caller mutates a frozen proposal's nested command later.
        actions = load_actions_module(self)
        allocator = actions.ActionAllocator(two_worker_turn())
        command = {"action": "move", "targetPos": [{"x": 2, "y": 1}]}
        self.assertTrue(allocator.try_add(actions.ActionProposal(
            command_owner_id=10010,
            actor_id=10010,
            command=command,
            destination=Pos(2, 1),
        )))

        command["action"] = "not-an-action"
        command["targetPos"][0]["x"] = 4

        self.assertEqual(
            allocator.complete_response()["roleCommandMap"]["10010"],
            {"action": "move", "targetPos": [{"x": 2, "y": 1}]},
        )

    def test_shared_gold_can_fund_only_one_build(self):
        # Break caught: two workers both spend the same observed 25 gold.
        actions = load_actions_module(self)
        allocator = actions.ActionAllocator(two_worker_turn(gold=25))

        first = allocator.try_add(actions.ActionProposal(
            command_owner_id=10010,
            actor_id=10010,
            command={
                "action": "build",
                "name": "gatling",
                "targetPos": [{"x": 2, "y": 1}],
            },
            gold_cost=25,
            destination=Pos(2, 1),
        ))
        second = allocator.try_add(actions.ActionProposal(
            command_owner_id=10012,
            actor_id=10012,
            command={
                "action": "build",
                "name": "railgun",
                "targetPos": [{"x": 3, "y": 1}],
            },
            gold_cost=25,
            destination=Pos(3, 1),
        ))

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(allocator.gold_remaining, 0)

    def test_existing_and_same_turn_weapon_builds_respect_team_limit(self):
        # Break caught: each worker independently sees room below the three-tower cap.
        actions = load_actions_module(self)
        payload = load_fixture()
        payload["teamOur"]["goldNum"] = 50
        first = payload["teamOur"]["roles"][0]
        second = copy.deepcopy(first)
        second.update({"id": 10012, "pos": {"x": 3, "y": 2}})
        weapon_a = copy.deepcopy(first)
        weapon_a.update({
            "id": 10020, "roleType": "gatling", "pos": {"x": 0, "y": 4},
            "health": 1000, "level": 1, "backPackCapability": 0,
        })
        weapon_b = copy.deepcopy(weapon_a)
        weapon_b.update({
            "id": 10030, "roleType": "railgun", "pos": {"x": 1, "y": 4},
        })
        payload["teamOur"]["roles"] = [first, second, weapon_a, weapon_b]
        allocator = actions.ActionAllocator(Turn.load(payload))

        self.assertTrue(allocator.try_add(actions.ActionProposal(
            10010,
            10010,
            {
                "action": "build", "name": "rocket",
                "targetPos": [{"x": 2, "y": 1}],
            },
            destination=Pos(2, 1),
        )))
        self.assertFalse(allocator.try_add(actions.ActionProposal(
            10012,
            10012,
            {
                "action": "build", "name": "gatling",
                "targetPos": [{"x": 3, "y": 1}],
            },
            destination=Pos(3, 1),
        )))

    def test_item_reservation_cannot_use_another_roles_backpack(self):
        # Break caught: a worker consumes Medicine held by a different worker.
        actions = load_actions_module(self)
        allocator = actions.ActionAllocator(two_worker_turn(
            first_items=("Medicine",), second_items=(),
        ))

        wrong_owner = allocator.try_add(actions.ActionProposal(
            command_owner_id=10012,
            actor_id=10012,
            command={"action": "use", "name": "Medicine"},
            item_costs=("Medicine",),
        ))

        self.assertFalse(wrong_owner)
        self.assertEqual(allocator.reserved_items, {})

    def test_weapon_controller_conflicts_with_personal_action(self):
        # Break caught: one role moves and controls a weapon in the same turn.
        actions = load_actions_module(self)
        allocator = actions.ActionAllocator(two_worker_turn())
        self.assertTrue(allocator.try_add(actions.ActionProposal(
            command_owner_id=10010,
            actor_id=10010,
            command={"action": "move", "targetPos": [{"x": 2, "y": 1}]},
            destination=Pos(2, 1),
        )))

        attack = allocator.try_add(actions.ActionProposal(
            command_owner_id=10020,
            actor_id=10010,
            command={
                "action": "attack",
                "controllerId": "10010",
                "targetPos": [{"x": 4, "y": 4}],
            },
        ))

        self.assertFalse(attack)

    def test_cancel_releases_resources_and_dependent_actions(self):
        # Break caught: cancelling a proposal leaks gold, actor, target, or dependency.
        actions = load_actions_module(self)
        allocator = actions.ActionAllocator(two_worker_turn(
            gold=25, first_items=("Medicine",),
        ))
        self.assertTrue(allocator.try_add(actions.ActionProposal(
            command_owner_id=10010,
            actor_id=10010,
            command={
                "action": "build",
                "name": "gatling",
                "targetPos": [{"x": 2, "y": 1}],
            },
            gold_cost=25,
            item_costs=("Medicine",),
            destination=Pos(2, 1),
        )))
        self.assertTrue(allocator.try_add(actions.ActionProposal(
            command_owner_id=10012,
            actor_id=10012,
            command={"action": "move", "targetPos": [{"x": 3, "y": 1}]},
            destination=Pos(3, 1),
            depends_on=(10010,),
        )))

        allocator.cancel(10010)

        self.assertEqual(allocator.gold_remaining, 25)
        self.assertEqual(allocator.reserved_items, {})
        self.assertEqual(
            allocator.complete_response()["roleCommandMap"], {}
        )
        self.assertTrue(allocator.try_add(actions.ActionProposal(
            command_owner_id=10012,
            actor_id=10012,
            command={
                "action": "build",
                "name": "railgun",
                "targetPos": [{"x": 2, "y": 1}],
            },
            gold_cost=25,
            destination=Pos(2, 1),
        )))

    def test_two_roles_cannot_reserve_the_same_destination(self):
        # Break caught: two friendly moves are planned into the same cell.
        actions = load_actions_module(self)
        allocator = actions.ActionAllocator(two_worker_turn())
        target = Pos(2, 1)
        self.assertTrue(allocator.try_add(actions.ActionProposal(
            command_owner_id=10010,
            actor_id=10010,
            command={"action": "move", "targetPos": [target.dump()]},
            destination=target,
        )))

        self.assertFalse(allocator.try_add(actions.ActionProposal(
            command_owner_id=10012,
            actor_id=10012,
            command={"action": "move", "targetPos": [target.dump()]},
            destination=target,
        )))

    def test_expected_sell_income_is_not_available_for_buy(self):
        # Break caught: same-turn sell proceeds are spent before feedback confirms them.
        actions = load_actions_module(self)
        allocator = actions.ActionAllocator(two_worker_turn(
            gold=0, first_items=("stone",) * 10,
            zones=(
                {"pos": {"x": 2, "y": 3}, "neutralType": "vendor"},
                {"pos": {"x": 3, "y": 3}, "neutralType": "weaponShop"},
            ),
        ))
        self.assertTrue(allocator.try_add(actions.ActionProposal(
            command_owner_id=10010,
            actor_id=10010,
            command={"action": "sell", "name": "stone", "num": 10},
        )))

        self.assertFalse(allocator.try_add(actions.ActionProposal(
            command_owner_id=10012,
            actor_id=10012,
            command={"action": "buy", "name": "Medicine"},
            gold_cost=10,
        )))

    def test_execute_cmd_requires_an_active_task(self):
        # Break caught: a task-only sandbox command is emitted outside a task.
        actions = load_actions_module(self)
        allocator = actions.ActionAllocator(two_worker_turn())

        with self.assertRaises(ValueError):
            allocator.complete_response(execute_cmd="pwd", task_active=False)

    def test_collect_rechecks_current_backpack_capacity(self):
        # Break caught: a preserved mine plan collects after the backpack fills.
        actions = load_actions_module(self)
        payload = load_fixture()
        worker = payload["teamOur"]["roles"][0]
        worker["backPackCapability"] = 1
        worker["backpack"] = ["stone"]

        self.assertFalse(actions.ActionAllocator(Turn.load(payload)).try_add(
            actions.ActionProposal(
                command_owner_id=10010,
                actor_id=10010,
                command={
                    "action": "collect",
                    "targetPos": [{"x": 1, "y": 1}],
                },
            )
        ))


if __name__ == "__main__":
    unittest.main()
