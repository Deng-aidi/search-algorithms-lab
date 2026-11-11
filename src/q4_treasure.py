
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from src.astar import astar

Number = Union[int, float]
Node = Tuple[int, int]

# 距离矩阵行列顺序固定。
Q4_NAMES: Tuple[str, ...] = ("A", "B", "T1", "T2", "T3", "T4")


@dataclass
class Q4SearchStats:
    nodes: int = 0
    leaves: int = 0
    cutoffs: int = 0


@dataclass(frozen=True)
class Q4State:
    a_pos: Node
    b_pos: Node
    a_name: str
    b_name: str
    remaining: Tuple[str, ...]
    score_diff: int
    turn: str
    positions: Mapping[str, Node] = field(default_factory=dict)


def _scan_positions(grid: List[List[Any]]) -> Dict[str, Node]:
    positions: Dict[str, Node] = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if isinstance(cell, str) and cell in Q4_NAMES:
                positions[cell] = (r, c)
    return positions


def _passable_for(grid: List[List[Any]]):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    def passable(node: Node) -> bool:
        r, c = node
        return 0 <= r < rows and 0 <= c < cols and grid[r][c] != "X"

    return passable


def _enter_cost_for(grid: List[List[Any]], enter_cost_overrides: Optional[Dict[Node, int]] = None):
    overrides = enter_cost_overrides or {}

    def enter_cost(node: Node) -> int:
        if node in overrides:
            return int(overrides[node])
        value = grid[node[0]][node[1]]
        if isinstance(value, str) and not value.isdigit():
            return 1  # 'A'、'B'、'T1'..'T4'
        return int(value)

    return enter_cost


def build_distance_matrix(
    grid: List[List[Any]],
    enter_cost_overrides: Optional[Dict[Node, int]] = None,
) -> Dict[str, Dict[str, Optional[int]]]:
    """六点两两最小进入代价；不可达为 None，d(x, x) = 0（不计入 astar）。

    enter_cost_overrides 的键是坐标 (row, col) 而非名字，因此"把 T3 的进入代价改为 3"
    不需要改地图字符，也不会影响 _scan_positions 的定位结果。
    """
    positions = _scan_positions(grid)
    passable = _passable_for(grid)
    enter_cost = _enter_cost_for(grid, enter_cost_overrides)

    matrix: Dict[str, Dict[str, Optional[int]]] = {}
    for src in Q4_NAMES:
        row: Dict[str, Optional[int]] = {}
        for dst in Q4_NAMES:
            if src == dst:
                row[dst] = 0
                continue
            # astar 的 goal_enter_cost 会覆盖 enter_cost(goal)，终点格的坐标级覆盖需在此同步生效。
            override = None if enter_cost_overrides is None else enter_cost_overrides.get(positions[dst])
            result = astar(
                positions[src],
                positions[dst],
                passable,
                enter_cost,
                goal_enter_cost=1 if override is None else int(override),
            )
            row[dst] = None if result.cost is None else int(result.cost)
        matrix[src] = row
    return matrix


def _advance(state: Q4State, treasure: str, dist, values) -> Q4State:
    """A 取宝藏则 score_diff += Gain，B 取则 -= Gain；remaining 有序移除，回合交替。"""
    if state.turn == "A":
        gain = values[treasure] - dist[state.a_name][treasure]
        remaining = tuple(x for x in state.remaining if x != treasure)
        return Q4State(
            a_pos=state.positions[treasure],
            b_pos=state.b_pos,
            a_name=treasure,
            b_name=state.b_name,
            remaining=remaining,
            score_diff=state.score_diff + gain,
            turn="B",
            positions=state.positions,
        )
    gain = values[treasure] - dist[state.b_name][treasure]
    remaining = tuple(x for x in state.remaining if x != treasure)
    return Q4State(
        a_pos=state.a_pos,
        b_pos=state.positions[treasure],
        a_name=state.a_name,
        b_name=treasure,
        remaining=remaining,
        score_diff=state.score_diff - gain,
        turn="A",
        positions=state.positions,
    )


def minimax(state: Q4State, depth: int, dist, values, stats: Q4SearchStats) -> int:
    """无剪枝；cutoffs 恒为 0。"""
    stats.nodes += 1
    if depth <= 0 or not state.remaining:
        stats.leaves += 1
        return state.score_diff

    is_max = state.turn == "A"
    best: Optional[int] = None
    for name in state.remaining:
        value = minimax(_advance(state, name, dist, values), depth - 1, dist, values, stats)
        best = value if best is None else (max(best, value) if is_max else min(best, value))
    return best


def alpha_beta(
    state: Q4State,
    depth: int,
    dist,
    values,
    alpha: Number,
    beta: Number,
    stats: Q4SearchStats,
) -> int:
    """MAX=A、MIN=B；alpha >= beta 时 break 并记一次 cutoff。"""
    stats.nodes += 1
    if depth <= 0 or not state.remaining:
        stats.leaves += 1
        return state.score_diff

    if state.turn == "A":
        value: Optional[int] = None
        for name in state.remaining:
            child = alpha_beta(_advance(state, name, dist, values), depth - 1, dist, values,
                               alpha, beta, stats)
            value = child if value is None else max(value, child)
            alpha = value if value > alpha else alpha
            if alpha >= beta:
                stats.cutoffs += 1
                break
        return value

    value = None
    for name in state.remaining:
        child = alpha_beta(_advance(state, name, dist, values), depth - 1, dist, values,
                           alpha, beta, stats)
        value = child if value is None else min(value, child)
        beta = value if value < beta else beta
        if alpha >= beta:
            stats.cutoffs += 1
            break
    return value


def solve_q4(
    grid: List[List[Any]],
    treasures: Dict[str, int],
    H: int,
    use_alpha_beta: bool = True,
    out_stats: Optional[Q4SearchStats] = None,
    enter_cost_overrides: Optional[Dict[Node, int]] = None,
) -> Dict[str, Any]:
    """根为 A（MAX），遍历 sorted(treasures)；并列保留先出现者。

    enter_cost_overrides 覆盖的是坐标级进入代价（见 build_distance_matrix）。
    """
    if not treasures:
        return {"best_target": None, "best_value": 0}

    positions = _scan_positions(grid)
    names = sorted(treasures)
    dist = build_distance_matrix(grid, enter_cost_overrides)
    state = Q4State(
        a_pos=positions["A"],
        b_pos=positions["B"],
        a_name="A",
        b_name="B",
        remaining=tuple(names),
        score_diff=0,
        turn="A",
        positions=positions,
    )

    stats = Q4SearchStats() if out_stats is None else out_stats
    stats.nodes += 1  # 根层也计入统计

    best_target: Optional[str] = None
    best_value: Optional[int] = None
    alpha: Number = float("-inf")
    beta: Number = float("inf")
    for name in names:
        child = _advance(state, name, dist, treasures)
        if use_alpha_beta:
            value = alpha_beta(child, H - 1, dist, treasures, alpha, beta, stats)
        else:
            value = minimax(child, H - 1, dist, treasures, stats)
        if best_value is None or value > best_value:
            best_target, best_value = name, value
        if use_alpha_beta:
            alpha = value if value > alpha else alpha
            if alpha >= beta:
                stats.cutoffs += 1
                break

    return {"best_target": best_target, "best_value": int(best_value)}
