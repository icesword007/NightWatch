import copy
import json
import unittest
from pathlib import Path

from agent import grid
from agent.protocol import Pos, Turn, station_footprint


FIXTURE = Path(__file__).parent / "fixtures" / "s0_request.json"


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def turn_and_worker(payload=None):
    turn = Turn.load(payload or load_fixture())
    return turn, turn.controllable()[0]


class GridTests(unittest.TestCase):
    def test_current_position_is_already_there_not_path_failure(self):
        # Break caught: start==goal enters backtracking without a predecessor.
        turn, worker = turn_and_worker()

        try:
            result = grid.next_step(turn, worker, worker.pos)
        except KeyError:
            self.fail("start==goal entered predecessor backtracking")

        self.assertEqual(getattr(result, "status", None), "already_there")
        self.assertIsNone(result.step)

    def test_surrounded_role_reports_unreachable(self):
        # Break caught: no-path searches retry forever or fabricate a step.
        payload = load_fixture()
        payload["mapInfo"]["zones"] = [
            {"pos": {"x": 2 + dx, "y": 2 + dy}, "neutralType": "stone"}
            for dx, dy in (
                (-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1),
            )
        ]
        payload["teamEnemy"]["roles"] = []
        payload["robot"]["roles"] = []
        turn, worker = turn_and_worker(payload)

        result = grid.next_step(turn, worker, Pos(4, 4))

        self.assertEqual(result.status, "unreachable")
        self.assertIsNone(result.step)

    def test_visible_enemy_and_reserved_destination_are_not_used(self):
        # Break caught: pathfinder steps onto an enemy or another role's claimed cell.
        turn, worker = turn_and_worker()

        result = grid.next_step(
            turn,
            worker,
            Pos(4, 2),
            reserved=frozenset({Pos(3, 2)}),
        )

        self.assertEqual(result.status, "found")
        self.assertIsNotNone(result.cost)
        self.assertNotEqual(result.step, turn.enemies[0].pos)
        self.assertNotEqual(result.step, Pos(3, 2))

    def test_station_footprint_blocks_all_four_cells(self):
        # Break caught: pathfinder treats three quarters of a 2x2 base as empty.
        payload = load_fixture()
        station = copy.deepcopy(payload["teamEnemy"]["roles"][0])
        station.update({
            "id": 20013,
            "roleType": "station",
            "pos": {"x": 3, "y": 3},
            "health": 1500,
        })
        payload["teamEnemy"]["roles"] = [station]
        turn, worker = turn_and_worker(payload)

        self.assertTrue(
            set(station_footprint(Pos(3, 3))).issubset(turn.blocked(worker))
        )

    def test_deadline_stops_before_expanding(self):
        # Break caught: wall-clock deadline is checked only after search finishes.
        turn, worker = turn_and_worker()

        result = grid.next_step(
            turn,
            worker,
            Pos(4, 4),
            clock=lambda: 10.0,
            deadline=5.0,
        )

        self.assertEqual(result.status, "deadline")
        self.assertEqual(result.expansions, 0)

    def test_expansion_limit_stops_search(self):
        # Break caught: a large map can consume unbounded search work.
        turn, worker = turn_and_worker()

        result = grid.next_step(
            turn, worker, Pos(4, 4), max_expansions=0,
        )

        self.assertEqual(result.status, "expansion_limit")
        self.assertEqual(result.expansions, 0)


if __name__ == "__main__":
    unittest.main()
