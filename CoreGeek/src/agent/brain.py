from typing import Any

from .protocol import Pos, Turn, move_command

_NEIGHBOUR_STEPS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    """Return one deterministic S0 move without enabling demo strategy."""
    turn = Turn.load(payload)
    role_commands: dict[str, dict[str, Any]] = {}

    for role in turn.controllable():
        blocked = turn.blocked(role)
        for dx, dy in _NEIGHBOUR_STEPS:
            target = Pos(role.pos.x + dx, role.pos.y + dy)
            if turn.land(target) and target not in blocked:
                role_commands[str(role.unit_id)] = move_command(target)
                break
        if role_commands:
            break

    return {
        "roleCommandMap": role_commands,
        "prompt": "",
        "executeCmd": "",
    }
