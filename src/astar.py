"""通用数值代价 A*，本文件不解析符号。"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Final, List, Optional, Set, Tuple, Union

Number = Union[int, float]
Node = Tuple[int, int]

# S3：四方向邻居枚举顺序固定为 上、下、左、右。
NEIGHBOR_OFFSETS: Final[Tuple[Node, ...]] = ((-1, 0), (1, 0), (0, -1), (0, 1))

_INF = float("inf")


@dataclass
class AStarResult:
    """A* 统一返回结果；cost 与返回 path 沿边累加值一致，无解为 None（S4）。"""

    path: List[Node]
    cost: Optional[Number]
    pops: int
    expanded_unique: int
    closed_skips: int
    reopens: int
    search_ms: Optional[float] = None


def _manhattan(node: Node, goal: Node) -> int:
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def _num(value: Number) -> Number:
    """S1：数值原样使用，int() 兜底只作用于 enter_cost 的返回值。"""
    if isinstance(value, (int, float)):
        return value
    return int(value)


def _check_h(value: Number) -> Number:
    """S5 / T5：非数值、NaN、负值抛 ValueError；inf 放行。"""
    if not isinstance(value, (int, float)):
        raise ValueError(f"heuristic 必须返回数值，实际为 {type(value).__name__}")
    if math.isnan(value):
        raise ValueError("heuristic 返回 NaN")
    if value < 0:
        raise ValueError(f"heuristic 返回值不得为负：{value}")
    return value


def astar(
    start: Node,
    goal: Node,
    passable: Callable[[Node], bool],
    enter_cost: Callable[[Node], Number],
    goal_enter_cost: Optional[Number] = None,
    allow_reopen: bool = False,
    heuristic: Optional[Callable[[Node, Node], Number]] = None,
    measure_time: bool = False,
) -> AStarResult:
    """在隐式图上搜索最短路；图由 passable / enter_cost 定义，本函数不接触地图结构。"""
    h_fn = _manhattan if heuristic is None else heuristic  # T5：None 回落曼哈顿

    g: Dict[Node, Number] = {start: 0}
    parent: Dict[Node, Node] = {}
    closed: Dict[Node, Number] = {}
    heap: List[Tuple[Number, Number, int, int, Number]] = []
    expanded: Set[Node] = set()
    pops = 0
    closed_skips = 0
    reopens = 0
    goal_g: Optional[Number] = None

    h0 = _check_h(h_fn(start, goal))
    heapq.heappush(heap, (h0, h0, start[0], start[1], 0))  # S6：(f, h, r, c, g) 升序

    t0 = time.perf_counter() if measure_time else 0.0  # S2：堆就绪后、主循环前

    while heap:
        _, _, r, c, g_pop = heapq.heappop(heap)
        pops += 1
        cur = (r, c)

        # U3：弹出未进入扩展处理即计入 closed_skips（过期堆项）。
        if g_pop > g.get(cur, _INF):
            closed_skips += 1
            continue
        # U3：closed 检查先于计数与终点判定；allow_reopen=False 时永久关闭。
        if cur in closed and (not allow_reopen or g_pop >= closed[cur]):
            closed_skips += 1
            continue

        expanded.add(cur)  # T3：进入扩展处理即计入，含终点
        closed[cur] = g_pop
        if cur == goal:
            goal_g = g_pop
            break

        for dr, dc in NEIGHBOR_OFFSETS:  # S3：固定枚举顺序，保证可复现
            nb = (r + dr, c + dc)
            if not passable(nb):  # U2：先 passable（越界与障碍都在调用方）
                continue
            if nb in closed and not allow_reopen:
                continue

            if nb == goal and goal_enter_cost is not None:  # U1：非 None 时完全不调用 enter_cost(goal)
                step: Number = goal_enter_cost
            else:
                step = _num(enter_cost(nb))  # S1

            ng = g_pop + step
            if ng < g.get(nb, _INF):
                g[nb] = ng
                parent[nb] = cur
                hn = _check_h(h_fn(nb, goal))
                heapq.heappush(heap, (ng + hn, hn, nb[0], nb[1], ng))
                if nb in closed:
                    reopens += 1  # S3：入堆侧计数

    t1 = time.perf_counter()  # U5：无解分支也在此统一收尾
    search_ms = (t1 - t0) * 1000.0 if measure_time else None  # T4：不含路径重建与结果构造

    if goal_g is None:
        return AStarResult([], None, pops, len(expanded), closed_skips, reopens, search_ms)

    path: List[Node] = [goal]
    node = goal
    while node != start:
        node = parent[node]
        path.append(node)
    path.reverse()
    return AStarResult(path, goal_g, pops, len(expanded), closed_skips, reopens, search_ms)
