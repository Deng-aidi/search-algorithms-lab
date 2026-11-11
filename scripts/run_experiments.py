"""Required experiments for Q1-Q4; writes report_assets/metrics.csv and figures."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import extensions
from reporting import (COLORS, MetricsWriter, apply_style, caption_block, get_pyplot,
                       record_caption, save_fig)
from src.astar import astar
from src.q3_alphabeta import AlphaBetaStats, TreeNode, alpha_beta
from src.q4_treasure import (Q4_NAMES, Q4SearchStats, Q4State, alpha_beta as q4_alpha_beta,
                             build_distance_matrix, solve_q4)

NOTE = "public;deterministic"

Q1_GRID = [
    ['S', '.', '.', 'X', '.'],
    ['.', 'X', '.', '.', '.'],
    ['.', 'X', '.', 'X', '.'],
    ['.', '.', '.', 'X', 'E'],
    ['X', 'X', '.', '.', '.'],
]

Q2_GRID = [
    ['S', 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, 'E'],
]

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


def _manhattan(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def _passable_for(grid):
    rows, cols = len(grid), len(grid[0])

    def passable(node):
        r, c = node
        return 0 <= r < rows and 0 <= c < cols and grid[r][c] != "X"

    return passable


def _q1_run(heuristic):
    """Run Q1 A* and derive the expansion order from groups of 4 passable() calls."""
    seen = []
    rows, cols = len(Q1_GRID), len(Q1_GRID[0])

    def passable(node):
        seen.append(node)
        r, c = node
        return 0 <= r < rows and 0 <= c < cols and Q1_GRID[r][c] != "X"

    result = astar((0, 0), (3, 4), passable, lambda _node: 1, goal_enter_cost=1,
                   heuristic=heuristic)
    order = []
    for i in range(0, len(seen), 4):
        group = seen[i:i + 4]
        order.append(((group[0][0] + group[1][0]) // 2, group[0][1]))
    if result.cost is not None:
        order.append((3, 4))
    return result, order


def _plot_q1_orders(orders):
    name = "p2_q1_expansion_order.png"
    plt = get_pyplot()
    if plt is None:
        print("matplotlib unavailable, skipped: %s" % name)
        return None
    apply_style()
    import matplotlib.patches as mpatches

    vmax = max(len(order) for order in orders.values()) - 1
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.9), layout="constrained")
    image = None
    for ax, (label, order) in zip(axes, orders.items()):
        values = [[float("nan")] * 5 for _ in range(5)]
        for index, (r, c) in enumerate(order):
            values[r][c] = index
        image = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_facecolor("0.97")
        for r in range(5):
            for c in range(5):
                if Q1_GRID[r][c] == "X":
                    ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                                    facecolor="0.75", edgecolor="none"))
                    ax.text(c, r, "X", ha="center", va="center", fontsize=9, color="0.2")
                elif (r, c) in order:
                    index = order.index((r, c))
                    ax.text(c, r, str(index), ha="center", va="center", fontsize=8,
                            color="white" if index >= vmax * 0.55 else "black")
        ax.text(0.0, -0.40, "S", ha="center", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="black",
                          linewidth=0.8))
        ax.text(4.0, 2.60, "E", ha="center", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="black",
                          linewidth=0.8))
        ax.set_title(label)
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
    fig.colorbar(image, ax=axes.tolist(), label="expansion index")
    save_fig(fig, name)
    record_caption(name,
                   "Q1: h=0 expands all %d cells; manhattan expands only %d, all inside the "
                   "shortest-path band." % (len(orders["h=0"]), len(orders["h=manhattan"])))
    return None


def experiment_q1(writer):
    runs = [("h=0", lambda _node, _goal: 0), ("h=manhattan", _manhattan)]
    lines = []
    orders = {}
    expanded = {}
    for label, heuristic in runs:
        result, order = _q1_run(heuristic)
        orders[label] = order
        expanded[label] = result.expanded_unique
        writer.add("q1", "heuristic", label, "expanded_unique", result.expanded_unique,
                   "nodes", 0, NOTE)
        writer.add("q1", "heuristic", label, "pops", result.pops, "nodes", 0, NOTE)
        writer.add("q1", "heuristic", label, "cost", result.cost, "cost", 0, NOTE)
        writer.add("q1", "heuristic", label, "path_len", len(result.path), "count", 0, NOTE)
        lines.append("q1 %s expanded_unique=%d pops=%d cost=%s path_len=%d"
                     % (label, result.expanded_unique, result.pops, result.cost,
                        len(result.path)))
    _plot_q1_orders(orders)
    return "\n".join(lines), {"q1_expanded": {"h0": expanded["h=0"],
                                              "manhattan": expanded["h=manhattan"]}}


def _q2_enter_cost(node):
    value = Q2_GRID[node[0]][node[1]]
    if value == "S":
        return 1
    if value == "E":
        return 0
    return int(value)


def experiment_q2(writer):
    runs = [
        ("h=0", lambda _node, _goal: 0),
        ("h=manhattan", _manhattan),
        ("h=h_adm", lambda a, b: max(0, _manhattan(a, b) - 1)),
    ]
    passable = _passable_for(Q2_GRID)
    labels = []
    expanded = []
    costs = []
    lines = []
    for label, heuristic in runs:
        result = astar((0, 0), (4, 4), passable, _q2_enter_cost, goal_enter_cost=0,
                       heuristic=heuristic)
        labels.append(label)
        expanded.append(result.expanded_unique)
        costs.append(result.cost)
        writer.add("q2", "heuristic", label, "cost", result.cost, "cost", 0, NOTE)
        writer.add("q2", "heuristic", label, "expanded_unique", result.expanded_unique,
                   "nodes", 0, NOTE)
        lines.append("q2 %s cost=%s expanded_unique=%d"
                     % (label, result.cost, result.expanded_unique))

    name = "p2_q2_heuristics.png"
    plt = get_pyplot()
    if plt is None:
        print("matplotlib unavailable, skipped: %s" % name)
    else:
        apply_style()
        fig, ax = plt.subplots(figsize=(7.4, 4.6), layout="constrained")
        bar_colors = [COLORS["secondary"] if label == "h=0" else COLORS["primary"]
                      for label in labels]
        bars = ax.bar(labels, expanded, color=bar_colors, edgecolor="white", linewidth=0.8)
        for bar, value in zip(bars, expanded):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), str(value),
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("expanded_unique")
        ax.set_ylim(0, max(expanded) * 1.2)
        ax.set_title("Q2 expanded_unique by heuristic (all three return cost = %d)" % costs[0])
        ax.grid(True, axis="y", color="0.9")
        ax.set_axisbelow(True)
        save_fig(fig, name)
        record_caption(name,
                       "Q2: all three heuristics return cost=%d; h=0 expands %d nodes while "
                       "manhattan and h_adm each expand %d, so h_adm does not change the "
                       "expansion set." % (costs[0], expanded[0], expanded[1]))
    return "\n".join(lines), {"q2_expanded": {"h0": expanded[0], "manhattan": expanded[1],
                                              "hadm": expanded[2]},
                              "q2_baseline_cost": costs[0]}


def _q3_tree():
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


def _subtree_value(node, is_max):
    return alpha_beta(node, float("-inf"), float("inf"), is_max, AlphaBetaStats())


def _plot_q3_tree():
    name = "p3_q3_tree.png"
    plt = get_pyplot()
    if plt is None:
        print("matplotlib unavailable, skipped: %s" % name)
        return None
    apply_style()
    import matplotlib.patches as mpatches

    nodes = {
        "root": (4.0, 3.0, "MAX", "plain"),
        "min1": (2.2, 2.0, "MIN", "plain"),
        "min2": (5.8, 2.0, "MIN", "plain"),
        "max1": (1.2, 1.0, "MAX", "plain"),
        "max2": (3.6, 1.0, "MAX", "plain"),
        "max3": (6.6, 1.0, "MAX", "plain"),
        "l3": (0.7, 0.0, "3", "leaf"),
        "l5": (1.7, 0.0, "5", "leaf"),
        "l6": (3.1, 0.0, "6", "leaf"),
        "l9": (4.1, 0.0, "9", "pruned"),
        "l1": (6.1, 0.0, "1", "leaf"),
        "l2": (7.1, 0.0, "2", "leaf"),
    }
    edges = [("root", "min1"), ("root", "min2"), ("min1", "max1"), ("min1", "max2"),
             ("min2", "max3"), ("max1", "l3"), ("max1", "l5"), ("max2", "l6"),
             ("max2", "l9"), ("max3", "l1"), ("max3", "l2")]

    fig, ax = plt.subplots(figsize=(8.8, 5.0), layout="constrained")
    ax.set_xlim(-0.4, 8.2)
    ax.set_ylim(-0.75, 3.6)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    for src, dst in edges:
        x1, y1 = nodes[src][:2]
        x2, y2 = nodes[dst][:2]
        pruned = dst == "l9"
        ax.plot([x1, x2], [y1 - 0.22, y2 + 0.22], color="0.55" if pruned else "0.35",
                linestyle="--" if pruned else "-", linewidth=1.0)
    ax.plot([nodes["min2"][0], nodes["max2"][0]], [nodes["min2"][1] - 0.22,
                                                  nodes["max2"][1] + 0.22],
            color=COLORS["accent"], linestyle="--", linewidth=1.1)
    ax.text(4.75, 1.66, "shared (visited twice)", fontsize=7.5, color=COLORS["accent"],
            ha="center", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    for _key, (x, y, label, kind) in nodes.items():
        face = "0.9" if kind == "pruned" else "#EAF1F8"
        edge = "0.55" if kind == "pruned" else "0.25"
        ax.add_patch(mpatches.FancyBboxPatch((x - 0.33, y - 0.2), 0.66, 0.4,
                                             boxstyle="square,pad=0.02", facecolor=face,
                                             edgecolor=edge,
                                             linestyle="--" if kind == "pruned" else "-",
                                             linewidth=0.9))
        ax.text(x, y, label, ha="center", va="center", fontsize=8)
    ax.text(4.1, -0.6, "pruned (first visit)", ha="center", fontsize=7.5, color="0.35")
    ax.set_title("Q3 game tree: shared max2 and one pruned leaf (visited nodes = 14)")
    save_fig(fig, name)
    record_caption(name,
                   "Q3: the shared max2 node is visited twice; on its first visit leaf 9 is cut "
                   "off by the alpha bound, while max3's two leaves are both visited, giving "
                   "7 leaves and 14 visited nodes.")
    return None


def experiment_q3(writer):
    variants = [("order=original", _q3_tree())]
    ascending = sorted(_q3_tree().children, key=lambda child: _subtree_value(child, False))
    descending = sorted(_q3_tree().children, key=lambda child: _subtree_value(child, False),
                        reverse=True)
    variants.append(("order=asc", TreeNode(children=ascending)))
    variants.append(("order=desc", TreeNode(children=descending)))

    counts = []
    labels = []
    lines = []
    for label, tree in variants:
        stats = AlphaBetaStats()
        value = alpha_beta(tree, float("-inf"), float("inf"), True, stats)
        writer.add("q3", "child_order", label, "nodes", stats.nodes, "count", 0, NOTE)
        writer.add("q3", "child_order", label, "leaves", stats.leaves, "count", 0, NOTE)
        writer.add("q3", "child_order", label, "cutoffs", stats.cutoffs, "count", 0, NOTE)
        writer.add("q3", "child_order", label, "value", value, "cost", 0, NOTE)
        labels.append(label)
        counts.append((stats.cutoffs, stats.leaves, stats.nodes))
        lines.append("q3 %s nodes=%d leaves=%d cutoffs=%d value=%s"
                     % (label, stats.nodes, stats.leaves, stats.cutoffs, value))

    name = "p3_q3_order.png"
    plt = get_pyplot()
    if plt is None:
        print("matplotlib unavailable, skipped: %s" % name)
    else:
        apply_style()
        fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
        positions = list(range(len(labels)))
        width = 0.36
        cut_bars = ax.bar([p - width / 2 for p in positions], [item[0] for item in counts],
                          width=width, color=COLORS["primary"], edgecolor="white",
                          linewidth=0.8, label="cutoffs")
        leaf_bars = ax.bar([p + width / 2 for p in positions], [item[1] for item in counts],
                           width=width, color=COLORS["secondary"], edgecolor="white",
                           linewidth=0.8, label="leaves")
        for bar, item in zip(cut_bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), str(item[0]),
                    ha="center", va="bottom", fontsize=9)
        for bar, item in zip(leaf_bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), str(item[1]),
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel("count")
        ax.set_ylim(0, max(max(item[0], item[1]) for item in counts) * 1.2)
        ax.set_title("Q3 cutoffs and leaves by child order (visited nodes = %d for all three)"
                     % counts[0][2])
        ax.legend()
        ax.grid(True, axis="y", color="0.9")
        ax.set_axisbelow(True)
        save_fig(fig, name)
        record_caption(name,
                       "Q3: ascending order cuts cutoffs from %d to %d while leaves (%d) and "
                       "nodes (%d) stay unchanged, so cutoff count alone does not measure "
                       "pruning efficiency."
                       % (counts[0][0], counts[1][0], counts[0][1], counts[0][2]))
    _plot_q3_tree()
    return "\n".join(lines), {
        "q3_cutoffs": {"original": counts[0][0], "asc": counts[1][0], "desc": counts[2][0]},
        "q3_leaves": counts[0][1],
        "q3_visited": counts[0][2],
    }


def _q4_positions(grid):
    positions = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if isinstance(cell, str) and cell in Q4_NAMES:
                positions[cell] = (r, c)
    return positions


def _q4_first_move_table(writer):
    """U of each root move; Q4State has no public advance, so it is built field by field."""
    dist = build_distance_matrix(Q4_GRID)
    positions = _q4_positions(Q4_GRID)
    names = sorted(Q4_VALUES)
    table = {}
    for depth in (2, 4):
        values = {}
        for name in names:
            child = Q4State(
                a_pos=positions[name],
                b_pos=positions["B"],
                a_name=name,
                b_name="B",
                remaining=tuple(other for other in names if other != name),
                score_diff=Q4_VALUES[name] - dist["A"][name],
                turn="B",
                positions=positions,
            )
            stats = Q4SearchStats()
            values[name] = q4_alpha_beta(child, depth - 1, dist, Q4_VALUES,
                                         float("-inf"), float("inf"), stats)
            writer.add("q4", "q4_first_move", "H=%d;first=%s" % (depth, name), "U",
                       values[name], "cost", 0, NOTE)
        best = max(values.values())
        expected = solve_q4(Q4_GRID, Q4_VALUES, depth)["best_value"]
        if best != expected:
            raise AssertionError("first-move max U %r != solve_q4 best_value %r at H=%d"
                                 % (best, expected, depth))
        table[depth] = values

    winners = {}
    for depth in (2, 4):
        best = max(table[depth].values())
        winners[depth] = [name for name in names if table[depth][name] == best]
    lines = ["first_move  H=2_U  H=4_U"]
    for name in names:
        lines.append("%-10s %5d  %5d" % (name, table[2][name], table[4][name]))
    lines.append("note: compare ranks within one H only; H=2 leaves are static U cut-offs, "
                 "H=4 leaves are terminal outcomes, so absolute U is not comparable across H.")
    lines.append("note: argmax at H=2 -> %s, at H=4 -> %s (tie set: H=2 {%s}, H=4 {%s})"
                 % (winners[2][0], winners[4][0], ",".join(winners[2]),
                    ",".join(winners[4])))
    return lines, {"q4_first": {"H2": (winners[2][0], table[2][winners[2][0]]),
                                "H4": (winners[4][0], table[4][winners[4][0]])}}


def experiment_q4(writer):
    matrix = build_distance_matrix(Q4_GRID)
    lines = []
    for src in Q4_NAMES:
        for dst in Q4_NAMES:
            writer.add("q4", "distance_matrix", "", "d.%s.%s" % (src, dst),
                       matrix[src][dst], "cost", 0, NOTE)
        lines.append("%s %s" % (src, " ".join(str(matrix[src][dst]) for dst in Q4_NAMES)))

    rows = []
    for depth in (1, 2, 3, 4):
        for algorithm in ("minimax", "alpha_beta"):
            stats = Q4SearchStats()
            result = solve_q4(Q4_GRID, Q4_VALUES, depth,
                              use_alpha_beta=(algorithm == "alpha_beta"), out_stats=stats)
            config = "H=%d;algo=%s" % (depth, algorithm)
            writer.add("q4", "game_tree", config, "nodes", stats.nodes, "count", 0, NOTE)
            writer.add("q4", "game_tree", config, "leaves", stats.leaves, "count", 0, NOTE)
            writer.add("q4", "game_tree", config, "cutoffs", stats.cutoffs, "count", 0, NOTE)
            writer.add("q4", "game_tree", config, "best_value", result["best_value"], "cost",
                       0, NOTE)
            rows.append((depth, algorithm, result["best_target"], result["best_value"],
                         stats.nodes, stats.leaves, stats.cutoffs))
            lines.append("H=%d %-10s best_target=%-3s best_value=%d nodes=%d leaves=%d cutoffs=%d"
                         % (depth, algorithm, result["best_target"], result["best_value"],
                            stats.nodes, stats.leaves, stats.cutoffs))

    minimax_nodes = [row[4] for row in rows if row[1] == "minimax"]
    alphabet_nodes = [row[4] for row in rows if row[1] == "alpha_beta"]
    cutoffs = [row[6] for row in rows if row[1] == "alpha_beta"]
    first_lines, first_facts = _q4_first_move_table(writer)
    lines.extend(first_lines)
    facts = {
        "q4_nodes": {"minimax": minimax_nodes, "alpha_beta": alphabet_nodes,
                     "cutoffs": cutoffs},
    }
    facts.update(first_facts)

    name = "p4_q4_matrix.png"
    plt = get_pyplot()
    if plt is None:
        print("matplotlib unavailable, skipped: %s" % name)
        print("matplotlib unavailable, skipped: p4_q4_pruning.png")
        return "\n".join(lines), facts

    apply_style()
    grid_values = [[matrix[src][dst] for dst in Q4_NAMES] for src in Q4_NAMES]
    flat = [value for row in grid_values for value in row]
    mid = (min(flat) + max(flat)) / 2.0
    fig, ax = plt.subplots(figsize=(6.6, 5.6), layout="constrained")
    image = ax.imshow(grid_values, cmap="YlGnBu")
    ax.set_xticks(range(len(Q4_NAMES)))
    ax.set_yticks(range(len(Q4_NAMES)))
    ax.set_xticklabels(Q4_NAMES)
    ax.set_yticklabels(Q4_NAMES)
    ax.set_xlabel("to")
    ax.set_ylabel("from")
    ax.set_title("Q4 distance matrix (symmetric because A/B/T all cost 1)")
    for r in range(len(Q4_NAMES)):
        for c in range(len(Q4_NAMES)):
            ax.text(c, r, str(grid_values[r][c]), ha="center", va="center", fontsize=9,
                    color="white" if grid_values[r][c] > mid else "black")
    fig.colorbar(image, ax=ax, label="cost")
    save_fig(fig, name)
    record_caption(name,
                   "Q4: the six-point distance matrix is symmetric because A/B/T entry cost "
                   "is uniformly 1 (range %d..%d)." % (min(flat), max(flat)))

    name = "p4_q4_pruning.png"
    depths = [1, 2, 3, 4]
    percents = [(mm - ab) / mm * 100.0 for mm, ab in zip(minimax_nodes, alphabet_nodes)]
    top = max(minimax_nodes)
    fig, ax = plt.subplots(figsize=(7.8, 4.8), layout="constrained")
    positions = list(range(len(depths)))
    ax.bar([p - 0.18 for p in positions], minimax_nodes, width=0.36, label="minimax",
           color=COLORS["secondary"], edgecolor="white", linewidth=0.8)
    bars_ab = ax.bar([p + 0.18 for p in positions], alphabet_nodes, width=0.36,
                     label="alpha_beta", color=COLORS["primary"], edgecolor="white",
                     linewidth=0.8)
    for bar, cutoff in zip(bars_ab, cutoffs):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), "cutoffs=%d" % cutoff,
                ha="center", va="bottom", fontsize=8)
    for index, percent in enumerate(percents):
        ax.text(index, max(minimax_nodes[index], alphabet_nodes[index]) + top * 0.10,
                "saved=%.1f%%" % percent, ha="center", va="bottom", fontsize=8,
                color="black")
    ax.set_xticks(positions)
    ax.set_xticklabels(["H=%d" % depth for depth in depths])
    ax.set_ylabel("nodes")
    ax.set_ylim(0, top * 1.28)
    ax.set_title("Q4 nodes visited by depth (labels = alpha_beta cutoffs)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", color="0.9")
    ax.set_axisbelow(True)
    save_fig(fig, name)
    record_caption(name,
                   "Q4: alpha-beta saves %.1f%% nodes at H=2, %.1f%% at H=3 and %.1f%% at H=4, "
                   "so the payoff is not monotone in depth."
                   % (percents[1], percents[2], percents[3]))
    return "\n".join(lines), facts


EXPERIMENTS = [
    ("q1", experiment_q1),
    ("q2", experiment_q2),
    ("q3", experiment_q3),
    ("q4", experiment_q4),
]


def _snapshot_block(facts, only):
    if only != "all":
        return "--- report snapshot ---\n(partial run: snapshot omitted)\nblock_end"
    q1 = facts["q1_expanded"]
    q2 = facts["q2_expanded"]
    q3 = facts["q3_cutoffs"]
    q4 = facts["q4_nodes"]
    first = facts["q4_first"]
    timing = facts["timing"]
    a = facts["a_seeds"]
    k_min = facts["k_min"]
    return "\n".join([
        "--- report snapshot ---",
        "q1_expanded h0=%d manhattan=%d" % (q1["h0"], q1["manhattan"]),
        "q2_expanded h0=%d manhattan=%d hadm=%d" % (q2["h0"], q2["manhattan"], q2["hadm"]),
        "q2_baseline_cost=%d k_min=%s k_min_cost=%s"
        % (facts["q2_baseline_cost"], "none" if k_min is None else k_min,
           "none" if facts["k_min_cost"] is None else facts["k_min_cost"]),
        "q3_cutoffs original=%d asc=%d desc=%d leaves=%d visited_nodes=%d"
        % (q3["original"], q3["asc"], q3["desc"], facts["q3_leaves"], facts["q3_visited"]),
        "q4_nodes minimax=%s alpha_beta=%s cutoffs=%s"
        % ("/".join(str(v) for v in q4["minimax"]),
           "/".join(str(v) for v in q4["alpha_beta"]),
           "/".join(str(v) for v in q4["cutoffs"])),
        "q4_first_move H2 argmax=%s value=%d H4 argmax=%s value=%d"
        % (first["H2"][0], first["H2"][1], first["H4"][0], first["H4"][1]),
        "timing median h0=%.5f manhattan=%.5f baseline=%.5f overlap=%s"
        % (timing["h0"], timing["manhattan"], timing["baseline"], timing["overlap"]),
        "a_seeds tried=%d diverged=%d rate=%.4f" % (a["tried"], a["diverged"], a["rate"]),
        "block_end",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the required Q1-Q4 experiments.")
    parser.add_argument("--only", choices=["q1", "q2", "q3", "q4", "ext", "all"], default="all")
    args = parser.parse_args(argv)

    writer = MetricsWriter()
    chunks = []
    facts = {}
    for key, func in EXPERIMENTS:
        if args.only in ("all", key):
            text, part = func(writer)
            chunks.append(text)
            facts.update(part)
    if args.only in ("all", "ext"):
        text, part = extensions.run(writer)
        chunks.append(text)
        facts.update(part)
    if args.only == "all":
        writer.flush()
    else:
        chunks.append("partial run (--only %s): metrics.csv not written" % args.only)
    print("\n".join(chunks))
    print("")
    print(caption_block())
    print("")
    print(_snapshot_block(facts, args.only))


if __name__ == "__main__":
    main()
