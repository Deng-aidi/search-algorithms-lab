"""题目二：加权地形入口；搜索复用 src.astar.astar。"""

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


def _enter_cost_for(grid: List[List[str]]):
    def enter_cost(node: Node) -> int:
        value = grid[node[0]][node[1]]
        if value == "S":
            return 1
        if value == "E":
            return 0
        return int(value)

    return enter_cost


def solve_q2(start: Node, goal: Node, grid: List[List[str]]) -> int:
    """返回最小总代价；无解返回 -1（题级约定，astar 的 None 在此转换）。"""
    result = astar(
        start,
        goal,
        _passable_for(grid),
        _enter_cost_for(grid),
        goal_enter_cost=0,
    )
    return -1 if result.cost is None else int(result.cost)
