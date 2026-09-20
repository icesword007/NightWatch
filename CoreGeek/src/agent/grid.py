import time
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Callable

from .protocol import Pos, Turn, Unit, distance

_STEPS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


@dataclass(frozen=True, slots=True)
class PathResult:
    status: str
    step: Pos | None
    expansions: int
    cost: int | None


def next_step(
    turn: Turn,
    moving: Unit,
    goal: Pos,
    *,
    reserved: frozenset[Pos] = frozenset(),
    clock: Callable[[], float] = time.monotonic,
    deadline: float | None = None,
    max_expansions: int = 256,
    prefer_deep_ties: bool = False,
) -> PathResult:
    if moving.pos == goal:
        return PathResult("already_there", None, 0, 0)
    blocked = set(turn.blocked(moving))
    blocked.update(reserved)
    blocked.discard(moving.pos)
    order = count()
    frontier: list[tuple[int, int, int, Pos]] = [
        (distance(moving.pos, goal), 0, next(order), moving.pos)
    ]
    came_from: dict[Pos, Pos] = {}
    best = {moving.pos: 0}
    seen: set[Pos] = set()
    expansions = 0

    while frontier:
        if deadline is not None and clock() >= deadline:
            return PathResult("deadline", None, expansions, None)
        if expansions >= max_expansions:
            return PathResult("expansion_limit", None, expansions, None)
        _, priority_cost, _, current = heappop(frontier)
        cost = -priority_cost if prefer_deep_ties else priority_cost
        if current in seen:
            continue
        if current == goal:
            return PathResult(
                "found",
                _first_step(came_from, moving.pos, goal),
                expansions,
                cost,
            )
        seen.add(current)
        expansions += 1
        for dx, dy in _STEPS:
            step = Pos(current.x + dx, current.y + dy)
            if step in blocked or not turn.land(step):
                continue
            new_cost = cost + 1
            if new_cost >= best.get(step, new_cost + 1):
                continue
            best[step] = new_cost
            came_from[step] = current
            heappush(
                frontier,
                (
                    new_cost + distance(step, goal),
                    -new_cost if prefer_deep_ties else new_cost,
                    next(order),
                    step,
                ),
            )
    return PathResult("unreachable", None, expansions, None)


def _first_step(came_from: dict[Pos, Pos], start: Pos, goal: Pos) -> Pos:
    current = goal
    while came_from[current] != start:
        current = came_from[current]
    return current
