
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

Number = Union[int, float]


@dataclass
class TreeNode:
    value: Optional[Number] = None
    children: List["TreeNode"] = field(default_factory=list)


@dataclass
class AlphaBetaStats:
    nodes: int = 0      # 进入递归的节点总数（含根、内部节点、叶子）
    leaves: int = 0     # 实际访问到的叶子数
    cutoffs: int = 0    # 触发 alpha >= beta 而提前 break 的次数


def alpha_beta(
    node: TreeNode,
    alpha: Number,
    beta: Number,
    is_max: bool,
    stats: AlphaBetaStats,
) -> Number:
    stats.nodes += 1
    if not node.children:
        stats.leaves += 1
        return node.value

    if is_max:
        value = float("-inf")
        for child in node.children:
            value = max(value, alpha_beta(child, alpha, beta, False, stats))
            alpha = max(alpha, value)
            if alpha >= beta:
                stats.cutoffs += 1
                break
        return value

    value = float("inf")
    for child in node.children:
        value = min(value, alpha_beta(child, alpha, beta, True, stats))
        beta = min(beta, value)
        if alpha >= beta:
            stats.cutoffs += 1
            break
    return value


def tree_from_dict(data: Dict[str, Any]) -> TreeNode:
    """接受 {"tree_structure": {...}} 包装或裸节点 dict；"value" 为叶子，"children" 为内部节点。"""
    root = data["tree_structure"] if "tree_structure" in data else data

    def build(node: Dict[str, Any]) -> TreeNode:
        if "value" in node:
            return TreeNode(value=node["value"])
        return TreeNode(children=[build(child) for child in node.get("children", [])])

    return build(root)


def solve_q3(
    tree: Union[TreeNode, Dict[str, Any]],
    stats: Optional[AlphaBetaStats] = None,
) -> Number:
    """根为 MAX，逐层交替；stats 为 None 时内部新建，仅用于统计。"""
    stats = AlphaBetaStats() if stats is None else stats
    root = tree_from_dict(tree) if isinstance(tree, dict) else tree
    return alpha_beta(root, float("-inf"), float("inf"), True, stats)
