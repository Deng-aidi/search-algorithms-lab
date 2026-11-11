"""公开用例断言：题目一、题目二、题目三、题目四（标准库 unittest）。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q1_maze import solve_q1
from src.q2_weighted import solve_q2
from src.q3_alphabeta import AlphaBetaStats, TreeNode, solve_q3
from src.q4_treasure import Q4SearchStats, build_distance_matrix, solve_q4

Q1_START = (0, 0)
Q1_GOAL = (3, 4)
Q1_GRID = [
    ['S', '.', '.', 'X', '.'],
    ['.', 'X', '.', '.', '.'],
    ['.', 'X', '.', 'X', '.'],
    ['.', '.', '.', 'X', 'E'],
    ['X', 'X', '.', '.', '.'],
]
Q1_EXPECTED_PATH = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]

Q2_START = (0, 0)
Q2_GOAL = (4, 4)
Q2_GRID = [
    ['S', 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, 'E'],
]


class Q1PublicCases(unittest.TestCase):
    def test_q1_public(self):
        path = solve_q1(Q1_START, Q1_GOAL, Q1_GRID)
        self.assertEqual(len(path), 8)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (3, 4))
        for prev, cur in zip(path, path[1:]):
            self.assertEqual(abs(prev[0] - cur[0]) + abs(prev[1] - cur[1]), 1)
        for row, col in path:
            self.assertNotEqual(Q1_GRID[row][col], 'X')
        self.assertEqual(path, Q1_EXPECTED_PATH)

    def test_q1_no_path(self):
        grid = [
            ['S', '.', '.'],
            ['X', 'X', 'X'],
            ['.', '.', 'E'],
        ]
        self.assertEqual(solve_q1((0, 0), (2, 2), grid), [])


class Q2PublicCases(unittest.TestCase):
    def test_q2_public(self):
        self.assertEqual(solve_q2(Q2_START, Q2_GOAL, Q2_GRID), 7)

    def test_q2_start_equals_goal(self):
        grid = [
            ['S', 1],
            [1, 'E'],
        ]
        self.assertEqual(solve_q2((0, 0), (0, 0), grid), 0)

    def test_q2_no_path(self):
        grid = [
            ['S', 1, 1],
            ['X', 'X', 'X'],
            [1, 'X', 'E'],
        ]
        self.assertEqual(solve_q2((0, 0), (2, 2), grid), -1)


Q3_TREE_DICT = {
    "tree_structure": {
        "is_maximizing_player": True,
        "children": [
            {
                "is_maximizing_player": False,
                "children": [
                    {"is_maximizing_player": True, "children": [{"value": 3}, {"value": 5}]},
                    {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
                ],
            },
            {
                "is_maximizing_player": False,
                "children": [
                    {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
                    {"is_maximizing_player": True, "children": [{"value": 1}, {"value": 2}]},
                ],
            },
        ],
    }
}


def _build_q3_dag():
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])


class Q3PublicCases(unittest.TestCase):
    def test_q3_public_tree_node(self):
        stats = AlphaBetaStats()
        self.assertEqual(solve_q3(_build_q3_dag(), stats), 5)
        stats_dict = AlphaBetaStats()
        self.assertEqual(solve_q3(Q3_TREE_DICT, stats_dict), 5)
        self.assertEqual(stats.nodes, stats_dict.nodes)

    def test_q3_public_tree_dict(self):
        stats = AlphaBetaStats()
        self.assertEqual(solve_q3(Q3_TREE_DICT, stats), 5)
        stats_dag = AlphaBetaStats()
        self.assertEqual(solve_q3(_build_q3_dag(), stats_dag), 5)
        self.assertEqual(stats.nodes, stats_dag.nodes)

    def test_q3_root_is_leaf(self):
        stats = AlphaBetaStats()
        self.assertEqual(solve_q3(TreeNode(value=7), stats), 7)
        self.assertEqual(stats.nodes, 1)
        self.assertEqual(stats.leaves, 1)
        self.assertEqual(stats.cutoffs, 0)

    def test_q3_pruning_known_case(self):
        min1 = TreeNode(children=[TreeNode(value=3), TreeNode(value=100)])
        min2 = TreeNode(children=[TreeNode(value=2), TreeNode(value=999)])
        root = TreeNode(children=[min1, min2])
        stats = AlphaBetaStats()
        self.assertEqual(solve_q3(root, stats), 3)
        self.assertEqual(stats.cutoffs, 1)
        self.assertEqual(stats.leaves, 3)
        self.assertEqual(stats.nodes, 6)


Q4_GRID = [
    ['A', 1, 1, 'X', 2, 1, 'T1'],
    [1, 'X', 6, 'X', 1, 'X', 1],
    [1, 1, 'T2', 1, 1, 1, 1],
    ['X', 1, 'X', 8, 'X', 1, 'X'],
    [1, 1, 1, 1, 'T3', 1, 1],
    [1, 'X', 9, 'X', 1, 'X', 1],
    ['T4', 1, 1, 'X', 1, 1, 'B'],
]
Q4_VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
Q4_NAMES = ["A", "B", "T1", "T2", "T3", "T4"]


class Q4PublicCases(unittest.TestCase):
    def test_q4_h2(self):
        self.assertEqual(solve_q4(Q4_GRID, Q4_VALUES, 2), {"best_target": "T2", "best_value": -1})

    def test_q4_h4(self):
        self.assertEqual(solve_q4(Q4_GRID, Q4_VALUES, 4), {"best_target": "T3", "best_value": 3})

    def test_q4_distance_matrix_shape(self):
        matrix = build_distance_matrix(Q4_GRID)
        self.assertEqual(list(matrix.keys()), Q4_NAMES)
        for src in Q4_NAMES:
            self.assertEqual(list(matrix[src].keys()), Q4_NAMES)
            for dst in Q4_NAMES:
                self.assertEqual(matrix[src][dst], matrix[dst][src])
        for name in Q4_NAMES:
            self.assertEqual(matrix[name][name], 0)
        self.assertEqual(matrix["A"]["T2"], 4)
        self.assertEqual(matrix["B"]["T4"], 10)
        self.assertEqual(matrix["T1"]["T3"], 6)

    def test_q4_minimax_matches_alpha_beta(self):
        for depth in (1, 2, 3, 4):
            stats_ab = Q4SearchStats()
            stats_mm = Q4SearchStats()
            with_ab = solve_q4(Q4_GRID, Q4_VALUES, depth, True, out_stats=stats_ab)
            without_ab = solve_q4(Q4_GRID, Q4_VALUES, depth, False, out_stats=stats_mm)
            self.assertEqual(with_ab["best_target"], without_ab["best_target"])
            self.assertEqual(with_ab["best_value"], without_ab["best_value"])
            self.assertEqual(stats_mm.cutoffs, 0)

    def test_q4_empty_treasures(self):
        self.assertEqual(solve_q4(Q4_GRID, {}, 4), {"best_target": None, "best_value": 0})

    def test_q4_enter_cost_overrides(self):
        base = build_distance_matrix(Q4_GRID)
        overrides = build_distance_matrix(Q4_GRID, enter_cost_overrides={(4, 4): 3})
        self.assertNotEqual(overrides["A"]["T3"], base["A"]["T3"])
        self.assertEqual(overrides["A"]["T3"] - base["A"]["T3"], 2)
        self.assertNotEqual(overrides["T2"]["T3"], overrides["T3"]["T2"])
        result = solve_q4(Q4_GRID, Q4_VALUES, 4, enter_cost_overrides={(4, 4): 3})
        self.assertEqual(set(result.keys()), {"best_target", "best_value"})
        self.assertIn(result["best_target"], Q4_VALUES)
        self.assertIsInstance(result["best_value"], int)


if __name__ == '__main__':
    unittest.main()