from dataclasses import dataclass
from typing import Any

DAY_ROUNDS = 70
NIGHT_ROUNDS = 60
ROUNDS_PER_DAY = DAY_ROUNDS + NIGHT_ROUNDS

WEAPON_BUILD_COST = 25
WALL_MATERIAL = "stone"
LAND = "land"
STATION = "station"
WALL = "wall"
WORKER = "worker"
PIONEER = "pioneer"
TOWER_TYPES = ("gatling", "railgun", "rocket")
CONTROLLABLE_TYPES = (WORKER, PIONEER)
TOWER_RANGE_BY_LEVEL = {
    "gatling": (3, 5, 7),
    "railgun": (6, 8, 10),
    "rocket": (10, 15, 10**9),
}
ROBOT_ATTACK_POWER = {
    "smallRobot": 5,
    "middleRobot": 10,
    "largeRobot": 20,
    "bossRobot": 40,
}


@dataclass(frozen=True, slots=True)
class Pos:
    x: int
    y: int

    @classmethod
    def load(cls, raw: Any) -> "Pos":
        if not isinstance(raw, dict):
            raise ValueError("position must be an object")
        try:
            return cls(int(raw["x"]), int(raw["y"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("position requires integer x and y") from error

    def dump(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


def distance(first: Pos, second: Pos) -> int:
    return max(abs(first.x - second.x), abs(first.y - second.y))


def station_footprint(pos: Pos) -> tuple[Pos, ...]:
    return (
        pos,
        Pos(pos.x + 1, pos.y),
        Pos(pos.x, pos.y - 1),
        Pos(pos.x + 1, pos.y - 1),
    )


@dataclass(frozen=True, slots=True)
class Unit:
    unit_id: int
    pos: Pos
    kind: str
    health: int
    attack_power: int
    level: int
    cooldown: int
    attack_range: int
    capacity: int | None
    backpack: tuple[str, ...]

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Unit":
        if not isinstance(raw, dict):
            raise ValueError("unit must be an object")
        try:
            raw_capacity = raw.get("backPackCapability")
            return cls(
                int(raw["id"]),
                Pos.load(raw["pos"]),
                str(raw["roleType"]),
                int(raw["health"]),
                int(raw.get("attackPower") or 0),
                int(raw.get("level") or 0),
                int(raw.get("cooldown") or 0),
                int(raw.get("attackRange") or 0),
                int(raw_capacity) if raw_capacity is not None else None,
                tuple(str(item) for item in raw.get("backpack") or ()),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("unit requires id, pos, roleType, and health") from error

    @property
    def backpack_full(self) -> bool:
        if self.capacity is None:
            return False
        return len(self.backpack) >= self.capacity

    def range_of_attack(self) -> int:
        if self.attack_range > 0:
            return self.attack_range
        table = TOWER_RANGE_BY_LEVEL.get(self.kind)
        if table is None:
            return 0
        level = min(max(self.level, 1), len(table))
        return table[level - 1]


@dataclass(frozen=True, slots=True)
class Robot:
    robot_id: int
    pos: Pos
    kind: str
    health: int
    abnormal_state: str
    target_team: str

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Robot":
        if not isinstance(raw, dict):
            raise ValueError("robot must be an object")
        try:
            return cls(
                int(raw["id"]),
                Pos.load(raw["pos"]),
                str(raw["roleType"]),
                int(raw["health"]),
                str(raw.get("abnormalState") or ""),
                str(raw.get("targetTeam") or ""),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "robot requires id, pos, roleType, and health"
            ) from error

    @property
    def attack_power(self) -> int:
        return ROBOT_ATTACK_POWER.get(self.kind, 0)


@dataclass(frozen=True, slots=True)
class PlayerTask:
    task_type: str
    pos: Pos
    cooldown_rounds: int
    score_reward: int
    gold_reward: int
    is_valid: bool
    timeout_rounds: int | None

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "PlayerTask":
        if not isinstance(raw, dict):
            raise ValueError("player task must be an object")
        try:
            raw_timeout = raw.get("timeoutRounds")
            raw_valid = raw["isValid"]
            if not isinstance(raw_valid, bool):
                raise ValueError("isValid must be a boolean")
            return cls(
                str(raw["taskType"]),
                Pos.load(raw["taskPosition"]),
                int(raw["coldDownRounds"]),
                int(raw["scoreReward"]),
                int(raw["goldReward"]),
                raw_valid,
                int(raw_timeout) if raw_timeout is not None else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("player task fields are invalid") from error


@dataclass(frozen=True, slots=True)
class ProtocolError:
    code: int
    description: str


@dataclass(frozen=True, slots=True)
class Turn:
    round_no: int
    is_day: bool
    team_id: str
    team_type: str
    gold: int
    width: int
    height: int
    zones: dict[Pos, str]
    ours: tuple[Unit, ...]
    enemies: tuple[Unit, ...]
    robots: tuple[Robot, ...]
    vendor_prices: dict[str, int]
    weapon_prices: dict[str, int]
    total_score: int
    player_tasks: tuple[PlayerTask, ...]
    phase_task: str
    llm_response: str
    command_result: str
    errors: tuple[ProtocolError, ...]

    @classmethod
    def load(cls, payload: dict[str, Any]) -> "Turn":
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        try:
            round_no = int(payload["roundNo"])
            info = payload["mapInfo"]
            team = payload["teamOur"]
            enemy = payload["teamEnemy"]
            robots = payload["robot"]
            if not all(
                isinstance(value, dict)
                for value in (info, team, enemy, robots)
            ):
                raise ValueError("request sections must be objects")
            return cls(
                round_no,
                (round_no - 1) % ROUNDS_PER_DAY < DAY_ROUNDS,
                str(team["teamId"]),
                str(team["type"]),
                int(team.get("goldNum") or 0),
                int(info["width"]),
                int(info["height"]),
                {
                    Pos.load(zone["pos"]): str(zone["neutralType"])
                    for zone in info.get("zones") or ()
                },
                tuple(Unit.load(role) for role in team.get("roles") or ()),
                tuple(Unit.load(role) for role in enemy.get("roles") or ()),
                tuple(
                    Robot.load(robot) for robot in robots.get("roles") or ()
                ),
                _price_map(payload.get("vendorShopList")),
                _price_map(payload.get("weaponShopList")),
                int(team.get("totalScore") or 0),
                tuple(
                    PlayerTask.load(task)
                    for task in team.get("playerTasks") or ()
                ),
                str(payload.get("phaseTask") or ""),
                str(payload.get("llmResp") or ""),
                str(payload.get("lastCmdResult") or ""),
                _errors(payload.get("errors")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "request requires roundNo, mapInfo, teamOur, teamEnemy, and robot"
            ) from error

    def station(self) -> Unit | None:
        for unit in self.ours:
            if unit.kind == STATION and unit.health > 0:
                return unit
        return None

    def alive(self, kinds: tuple[str, ...]) -> tuple[Unit, ...]:
        return tuple(
            unit for unit in self.ours
            if unit.kind in kinds and unit.health > 0
        )

    def controllable(self) -> tuple[Unit, ...]:
        return tuple(sorted(
            self.alive(CONTROLLABLE_TYPES), key=lambda unit: unit.unit_id,
        ))

    def workers(self) -> tuple[Unit, ...]:
        return tuple(sorted(
            self.alive((WORKER,)), key=lambda unit: unit.unit_id,
        ))

    def pioneers(self) -> tuple[Unit, ...]:
        return tuple(sorted(
            self.alive((PIONEER,)), key=lambda unit: unit.unit_id,
        ))

    def task_cells(self, task: PlayerTask) -> tuple[Pos, ...]:
        suffix = task.task_type[-1:] if task.task_type[-1:] in ("1", "2") else ""
        zone_kind = f"{self.team_type}TaskPoint{suffix}"
        cells = self.zones_of(zone_kind) if suffix else ()
        return cells or (task.pos,)

    def weapons(self) -> tuple[Unit, ...]:
        return tuple(sorted(
            self.alive(TOWER_TYPES),
            key=lambda unit: (unit.pos.x, unit.pos.y),
        ))

    def walls(self) -> tuple[Unit, ...]:
        return self.alive((WALL,))

    def stone_mines(self) -> tuple[Pos, ...]:
        return tuple(
            pos for pos, kind in self.zones.items() if kind == WALL_MATERIAL
        )

    def zones_of(self, *kinds: str) -> tuple[Pos, ...]:
        wanted = frozenset(kinds)
        return tuple(sorted(
            (pos for pos, kind in self.zones.items() if kind in wanted),
            key=lambda pos: (pos.x, pos.y),
        ))

    def unit(self, unit_id: int) -> Unit | None:
        for unit in self.ours:
            if unit.unit_id == unit_id and unit.health > 0:
                return unit
        return None

    @property
    def round_in_day(self) -> int:
        return (self.round_no - 1) % ROUNDS_PER_DAY + 1

    @property
    def rounds_until_night(self) -> int:
        if not self.is_day:
            return 0
        return DAY_ROUNDS - self.round_in_day + 1

    def footprint(self, unit: Unit) -> tuple[Pos, ...]:
        if unit.kind == STATION:
            return station_footprint(unit.pos)
        return (unit.pos,)

    def land(self, pos: Pos) -> bool:
        if not 0 <= pos.x < self.width or not 0 <= pos.y < self.height:
            return False
        return self.zones.get(pos, LAND) == LAND

    def occupied_cells(self) -> frozenset[Pos]:
        cells: set[Pos] = set()
        for unit in (*self.ours, *self.enemies):
            if unit.health > 0:
                cells.update(self.footprint(unit))
        return frozenset(cells)

    def blocked(self, moving: Unit) -> frozenset[Pos]:
        cells = {pos for pos, kind in self.zones.items() if kind != LAND}
        cells.update(self.occupied_cells())
        cells.discard(moving.pos)
        for robot in self.robots:
            cells.add(robot.pos)
        return frozenset(cells)


def move_command(pos: Pos) -> dict[str, Any]:
    return {"action": "move", "targetPos": [pos.dump()]}


def collect_command(pos: Pos) -> dict[str, Any]:
    return {"action": "collect", "targetPos": [pos.dump()]}


def build_command(pos: Pos, name: str) -> dict[str, Any]:
    return {"action": "build", "targetPos": [pos.dump()], "name": name}


def attack_command(controller_id: int, pos: Pos) -> dict[str, Any]:
    return {
        "action": "attack",
        "targetPos": [pos.dump()],
        "controllerId": str(controller_id),
    }


def sell_command(name: str, number: int = 1) -> dict[str, Any]:
    return {"action": "sell", "name": name, "num": number}


def buy_command(name: str, number: int = 1) -> dict[str, Any]:
    return {"action": "buy", "name": name, "num": number}


def use_command(name: str, pos: Pos | None = None) -> dict[str, Any]:
    command: dict[str, Any] = {"action": "use", "name": name}
    if pos is not None:
        command["targetPos"] = [pos.dump()]
    return command


def accept_task_command() -> dict[str, Any]:
    return {"action": "acceptTask"}


def submit_answer_command(answer: str) -> dict[str, Any]:
    return {"action": "submitAnswer", "taskAnswer": answer}


def _price_map(raw: Any) -> dict[str, int]:
    if not isinstance(raw, list):
        return {}
    prices = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            name = str(item["name"])
            price = int(item["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if name and price >= 0:
            prices[name] = price
    return prices


def _errors(raw: Any) -> tuple[ProtocolError, ...]:
    if not isinstance(raw, list):
        return ()
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            result.append(ProtocolError(
                int(item["errorCode"]),
                str(item.get("description") or ""),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(result)
