"""题目一：迷宫寻路入口；搜索复用 src.astar.astar。"""

from typing import List, Tuple

from src.astar import astar

Node = Tuple[int, int]


def _passable_for(grid: List[List[str]]):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    def passable(node: Node) -> bool:
        r, c = node
        return 0 <= r < rows and 0 <= c < cols and grid[r][c] != "X"

    return passable


def solve_q1(start: Node, goal: Node, grid: List[List[str]]) -> List[Node]:
    """返回从 start 到 goal 的路径（含两端）；无解返回 []。"""
    return astar(
        start,
        goal,
        _passable_for(grid),
        lambda _node: 1,
        goal_enter_cost=1,
    ).path
