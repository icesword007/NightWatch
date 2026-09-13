import copy
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

from .protocol import (
    PIONEER,
    STATION,
    TOWER_TYPES,
    WALL,
    WORKER,
    Pos,
    Turn,
    distance,
)

_ACTIONS = frozenset({
    "move", "attack", "sell", "buy", "build", "remove", "acceptTask",
    "submitAnswer", "summonTreasure", "use", "drop", "collect",
})
_TARGET_ACTIONS = frozenset({
    "move", "attack", "build", "remove", "summonTreasure", "collect",
})
_SINGLE_TARGET_ACTIONS = _TARGET_ACTIONS - {"attack"}
_NAME_ACTIONS = frozenset({"sell", "buy", "build", "use", "drop"})
_DESTINATION_ACTIONS = frozenset({"move", "build"})
MAX_TASK_ANSWER_CHARS = 16_384


@dataclass(frozen=True, slots=True)
class ActionProposal:
    command_owner_id: int
    actor_id: int
    command: dict[str, Any]
    gold_cost: int = 0
    item_costs: tuple[str, ...] = ()
    destination: Pos | None = None
    depends_on: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PlannedAction:
    proposal: ActionProposal
    plan_target: Pos | None = None
    plan_reason: str | None = None
    deadline_round: int | None = None


class ActionAllocator:
    def __init__(self, turn: Turn) -> None:
        self._turn = turn
        self._gold = turn.gold
        self._owners = {
            unit.unit_id: unit for unit in turn.ours if unit.health > 0
        }
        self._inventories = {
            unit.unit_id: Counter(unit.backpack) for unit in turn.controllable()
        }
        self._weapons = {unit.unit_id for unit in turn.weapons()}
        self._proposals: dict[int, ActionProposal] = {}
        self._actors: set[int] = set()
        self._destinations: set[Pos] = set()
        self._gold_reserved = 0
        self._items: dict[int, Counter[str]] = {}

    @property
    def gold_remaining(self) -> int:
        return self._gold - self._gold_reserved

    @property
    def reserved_items(self) -> dict[int, dict[str, int]]:
        return {
            role_id: dict(items)
            for role_id, items in self._items.items()
            if items
        }

    def try_add(self, proposal: ActionProposal) -> bool:
        requirements = self._required_resources(proposal)
        if requirements is None:
            return False
        gold_cost, item_costs = requirements
        accepted = replace(
            copy.deepcopy(proposal),
            gold_cost=gold_cost,
            item_costs=item_costs,
        )
        if not self._can_add(accepted):
            return False
        self._proposals[accepted.command_owner_id] = accepted
        self._actors.add(accepted.actor_id)
        if accepted.destination is not None:
            self._destinations.add(accepted.destination)
        self._gold_reserved += accepted.gold_cost
        if accepted.item_costs:
            self._items.setdefault(accepted.actor_id, Counter()).update(
                accepted.item_costs
            )
        return True

    def cancel(self, command_owner_id: int) -> None:
        cancelled = {command_owner_id}
        changed = True
        while changed:
            changed = False
            for owner_id, proposal in self._proposals.items():
                if owner_id in cancelled:
                    continue
                if any(dependency in cancelled for dependency in proposal.depends_on):
                    cancelled.add(owner_id)
                    changed = True
        for owner_id in cancelled:
            self._proposals.pop(owner_id, None)
        self._rebuild_reservations()

    def complete_response(
        self,
        *,
        prompt: str = "",
        execute_cmd: str = "",
        task_active: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(prompt, str) or not isinstance(execute_cmd, str):
            raise ValueError("prompt and executeCmd must be strings")
        if execute_cmd and not task_active:
            raise ValueError("executeCmd requires an active task")
        return {
            "roleCommandMap": {
                str(owner_id): copy.deepcopy(proposal.command)
                for owner_id, proposal in sorted(self._proposals.items())
            },
            "prompt": prompt,
            "executeCmd": execute_cmd,
        }

    def _can_add(self, proposal: ActionProposal) -> bool:
        if proposal.gold_cost < 0:
            return False
        if not isinstance(proposal.command, dict):
            return False
        action = proposal.command.get("action")
        if action not in _ACTIONS:
            return False
        if proposal.command_owner_id not in self._owners:
            return False
        if proposal.command_owner_id in self._proposals:
            return False
        if proposal.actor_id in self._actors:
            return False
        if proposal.actor_id not in self._inventories:
            return False
        if proposal.gold_cost > self.gold_remaining:
            return False
        if proposal.destination is not None and proposal.destination in self._destinations:
            return False
        if any(dependency not in self._proposals for dependency in proposal.depends_on):
            return False
        targets = self._command_targets(proposal.command)
        if action in _TARGET_ACTIONS and targets is None:
            return False
        if action in _SINGLE_TARGET_ACTIONS and len(targets) != 1:
            return False
        if action not in _TARGET_ACTIONS and "targetPos" in proposal.command:
            if targets is None:
                return False
        if action in _NAME_ACTIONS:
            name = proposal.command.get("name")
            if not isinstance(name, str) or not name:
                return False
        if action in ("sell", "buy") and "num" in proposal.command:
            number = proposal.command["num"]
            if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
                return False
        if action == "submitAnswer" and not isinstance(
            proposal.command.get("taskAnswer"), str,
        ):
            return False
        if action == "summonTreasure":
            items = proposal.command.get("item")
            if (
                not isinstance(items, list)
                or not items
                or not all(isinstance(item, str) and item for item in items)
            ):
                return False
        if action in _DESTINATION_ACTIONS and proposal.destination is None:
            return False
        if proposal.destination is not None:
            if targets is None or targets[0] != proposal.destination:
                return False
        if action == "attack":
            if proposal.command_owner_id not in self._weapons:
                return False
            try:
                if int(proposal.command.get("controllerId")) != proposal.actor_id:
                    return False
            except (TypeError, ValueError):
                return False
        elif proposal.command_owner_id != proposal.actor_id:
            return False
        if not self._c_action_is_legal(proposal, targets):
            return False
        requested = Counter(proposal.item_costs)
        available = self._inventories[proposal.actor_id]
        already_reserved = self._items.get(proposal.actor_id, Counter())
        for item, count in requested.items():
            if count <= 0 or count + already_reserved[item] > available[item]:
                return False
        return True

    def _required_resources(
        self,
        proposal: ActionProposal,
    ) -> tuple[int, tuple[str, ...]] | None:
        command = proposal.command
        if not isinstance(command, dict):
            return None
        action = command.get("action")
        name = command.get("name")
        number = command.get("num", 1)
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            return None
        if action == "buy":
            price = self._turn.weapon_prices.get(name)
            if price is None:
                return None
            return price * number, ()
        if action == "build" and name in TOWER_TYPES:
            return 25, ()
        if action == "build" and name == WALL:
            return 0, ("stone",)
        if action == "sell":
            return 0, tuple(str(name) for _ in range(number))
        if action in ("use", "drop") and isinstance(name, str):
            return 0, (name,)
        if action == "summonTreasure":
            items = command.get("item")
            if isinstance(items, list):
                return 0, tuple(str(item) for item in items)
        return 0, ()

    def _c_action_is_legal(
        self,
        proposal: ActionProposal,
        targets: tuple[Pos, ...] | None,
    ) -> bool:
        action = proposal.command["action"]
        actor = self._owners[proposal.actor_id]
        if action == "move":
            target = targets[0]
            return (
                distance(actor.pos, target) == 1
                and self._turn.land(target)
                and target not in self._turn.blocked(actor)
            )
        if action == "attack":
            weapon = self._owners[proposal.command_owner_id]
            expected_targets = 1 if weapon.kind == "railgun" else weapon.level
            return (
                not self._turn.is_day
                and weapon.cooldown == 0
                and distance(actor.pos, weapon.pos) == 1
                and len(targets) == expected_targets
                and all(
                    distance(weapon.pos, target) <= weapon.range_of_attack()
                    for target in targets
                )
                and self._gatling_cone_is_valid(weapon, targets)
            )
        if action == "build":
            target = targets[0]
            if proposal.command["name"] in TOWER_TYPES:
                planned_weapons = sum(
                    item.command.get("action") == "build"
                    and item.command.get("name") in TOWER_TYPES
                    for item in self._proposals.values()
                )
                if len(self._turn.weapons()) + planned_weapons >= 3:
                    return False
            return (
                actor.kind == WORKER
                and self._turn.is_day
                and proposal.command["name"] in (*TOWER_TYPES, WALL)
                and distance(actor.pos, target) == 1
                and self._turn.land(target)
                and target not in self._turn.occupied_cells()
            )
        if action == "collect":
            return (
                actor.kind == WORKER
                and actor.capacity is not None
                and not actor.backpack_full
                and distance(actor.pos, targets[0]) == 1
                and self._turn.zones.get(targets[0]) in ("stone", "iron", "copper")
            )
        if action == "sell":
            return (
                proposal.command["name"] in ("stone", "iron", "copper")
                and proposal.command["name"] in self._turn.vendor_prices
                and self._adjacent_to_zone(actor.pos, "vendor")
            )
        if action == "buy":
            count = proposal.command.get("num", 1)
            if actor.capacity is None:
                return False
            free = actor.capacity - len(actor.backpack)
            return (
                self._adjacent_to_zone(actor.pos, "weaponShop")
                and free >= count
            )
        if action == "remove":
            target_unit = self._unit_at(targets[0])
            return (
                actor.kind == WORKER
                and distance(actor.pos, targets[0]) == 1
                and target_unit is not None
                and target_unit.kind == WALL
            )
        if action == "use":
            return self._use_is_legal(actor, proposal.command, targets)
        if action == "acceptTask":
            return (
                actor.kind == PIONEER
                and not self._turn.phase_task
                and any(
                    task.is_valid
                    and task.cooldown_rounds == 0
                    and any(
                        distance(actor.pos, cell) == 1
                        for cell in self._turn.task_cells(task)
                    )
                    for task in self._turn.player_tasks
                )
            )
        if action == "submitAnswer":
            answer = proposal.command.get("taskAnswer")
            return (
                actor.kind == PIONEER
                and bool(self._turn.phase_task)
                and isinstance(answer, str)
                and 0 < len(answer) <= MAX_TASK_ANSWER_CHARS
            )
        return True

    def _use_is_legal(
        self,
        actor: Any,
        command: dict[str, Any],
        targets: tuple[Pos, ...] | None,
    ) -> bool:
        name = command["name"]
        if name == "Medicine":
            return "targetPos" not in command
        if name == "WallFixer":
            if targets is None or len(targets) != 1:
                return False
            target = self._unit_at(targets[0])
            return (
                target is not None
                and target.kind == WALL
                and distance(actor.pos, target.pos) == 1
            )
        voucher = self._voucher_target(name)
        if voucher is None:
            return True
        target_kind, required_level = voucher
        if targets is None or len(targets) != 1:
            return False
        target = self._unit_at(targets[0])
        if target is None or distance(actor.pos, target.pos) != 1:
            return False
        if target_kind == "weapon":
            return target.kind in TOWER_TYPES and target.level == required_level
        return target.kind == target_kind and target.level == required_level

    def _adjacent_to_zone(self, pos: Pos, kind: str) -> bool:
        return any(distance(pos, zone) == 1 for zone in self._turn.zones_of(kind))

    def _unit_at(self, pos: Pos) -> Any:
        for unit in self._turn.ours:
            if unit.health > 0 and pos in self._turn.footprint(unit):
                return unit
        return None

    @staticmethod
    def _voucher_target(name: str) -> tuple[str, int] | None:
        prefixes = {
            "WeaponUpgradeVoucher": "weapon",
            "StationUpgradeVoucher": STATION,
            "WallUpgradeVoucher": WALL,
        }
        for prefix, target_kind in prefixes.items():
            if name in (f"{prefix}1", f"{prefix}2"):
                return target_kind, int(name[-1])
        return None

    @staticmethod
    def _gatling_cone_is_valid(
        weapon: Any,
        targets: tuple[Pos, ...],
    ) -> bool:
        if weapon.kind != "gatling":
            return True
        vectors = [
            (target.x - weapon.pos.x, target.y - weapon.pos.y)
            for target in targets
        ]
        return all(
            ax * bx + ay * by >= 0
            for index, (ax, ay) in enumerate(vectors)
            for bx, by in vectors[index + 1:]
        )

    @staticmethod
    def _command_targets(command: dict[str, Any]) -> tuple[Pos, ...] | None:
        raw_targets = command.get("targetPos")
        if not isinstance(raw_targets, list) or not raw_targets:
            return None
        try:
            return tuple(Pos.load(target) for target in raw_targets)
        except ValueError:
            return None

    def _rebuild_reservations(self) -> None:
        self._actors = set()
        self._destinations = set()
        self._gold_reserved = 0
        self._items = {}
        for proposal in self._proposals.values():
            self._actors.add(proposal.actor_id)
            if proposal.destination is not None:
                self._destinations.add(proposal.destination)
            self._gold_reserved += proposal.gold_cost
            if proposal.item_costs:
                self._items.setdefault(proposal.actor_id, Counter()).update(
                    proposal.item_costs
                )
