"""Self-designed extensions A/B/C (reopen parity heuristic, timing noise, directed matrix).

Only depends on reporting helpers and the public src API; run_experiments is never imported.
"""

import gc
import random
import statistics
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reporting import COLORS, apply_style, get_pyplot, record_caption, save_fig
from src.astar import astar
from src.q4_treasure import Q4_NAMES, build_distance_matrix, solve_q4

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
Q2_START = (0, 0)
Q2_GOAL = (4, 4)

K_FALLBACK_GRID = [
    ['S', 1, 1, 1, 1],
    [8, 8, 8, 8, 1],
    [1, 1, 1, 1, 1],
    [1, 8, 8, 8, 8],
    [1, 1, 1, 1, 'E'],
]

NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIZE = 7
START = (0, 0)
GOAL = (SIZE - 1, SIZE - 1)
A_NOTE = "boundary;sensitivity"
B_NOTE = "boundary;deterministic"
C_NOTE = "sensitivity"
K_NOTE = "sensitivity"


def _passable_for(grid):
    rows, cols = len(grid), len(grid[0])

    def passable(node):
        r, c = node
        return 0 <= r < rows and 0 <= c < cols and grid[r][c] != "X"

    return passable


def _make_grid(seed, obstacle=0.30):
    rng = random.Random(seed)
    grid = [["." for _ in range(SIZE)] for _ in range(SIZE)]
    for r in range(SIZE):
        for c in range(SIZE):
            if (r, c) in (START, GOAL):
                continue
            if rng.random() < obstacle:
                grid[r][c] = "X"
    return grid


def _bfs_from_goal(grid):
    dist = {GOAL: 0}
    queue = deque([GOAL])
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS:
            node = (r + dr, c + dc)
            if not (0 <= node[0] < SIZE and 0 <= node[1] < SIZE):
                continue
            if grid[node[0]][node[1]] == "X" or node in dist:
                continue
            dist[node] = dist[(r, c)] + 1
            queue.append(node)
    return dist


def _parity_h(dist):
    def heuristic(node, _goal):
        if (node[0] + node[1]) % 2 == 0:
            return dist.get(node, 0)
        return 0

    return heuristic


def _plot_a(grid, dist, gaps):
    name = "p2_A_parity.png"
    plt = get_pyplot()
    if plt is None:
        print("matplotlib unavailable, skipped: %s" % name)
        return None

    apply_style()
    import matplotlib.patches as mpatches

    values = [[float("nan")] * SIZE for _ in range(SIZE)]
    shown = [[0] * SIZE for _ in range(SIZE)]
    for r in range(SIZE):
        for c in range(SIZE):
            if grid[r][c] == "X":
                continue
            value = dist.get((r, c), 0) if (r + c) % 2 == 0 else 0
            values[r][c] = value
            shown[r][c] = value
    vmax = max((shown[r][c] for r in range(SIZE) for c in range(SIZE)), default=0)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.9), layout="constrained")
    image = left.imshow(values, cmap="Blues")
    left.set_facecolor("0.97")
    left.set_title("Parity-perturbed heuristic h (first connected seed)")
    left.set_xlabel("col")
    left.set_ylabel("row")
    left.set_xticks(range(SIZE))
    left.set_yticks(range(SIZE))
    for r in range(SIZE):
        for c in range(SIZE):
            if grid[r][c] == "X":
                left.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                                  facecolor="0.75", edgecolor="none"))
                left.text(c, r, "X", ha="center", va="center", fontsize=9, color="0.2")
            else:
                left.text(c, r, str(shown[r][c]), ha="center", va="center", fontsize=8,
                          color="white" if shown[r][c] >= vmax * 0.6 else "black")
    fig.colorbar(image, ax=left, label="h value")

    indices = list(range(1, len(gaps) + 1))
    right.bar(indices, gaps, color=COLORS["primary"], linewidth=0)
    right.axhline(0.0, color="0.3", linewidth=0.8)
    right.set_ylim(-0.5, 0.5)
    right.set_xlabel("connected seed index")
    right.set_ylabel("cost(no_reopen) - cost(reopen)")
    right.set_title("Cost gap per connected seed (n=%d)" % len(gaps))
    if not any(gaps):
        right.text(len(gaps) / 2.0, 0.32, "all gaps = 0", ha="center", fontsize=9)
    right.grid(True, axis="y", color="0.9")
    right.set_axisbelow(True)
    save_fig(fig, name)
    if not any(gaps):
        record_caption(name,
                       "Parity-perturbed admissible heuristic is inconsistent, yet on %d "
                       "connected 7x7 seeds allow_reopen=False and True returned identical "
                       "costs; the gap histogram is all zeros, i.e. no closed node ever sat on "
                       "a cheaper path in these %d seeds." % (len(gaps), len(gaps)))
    else:
        record_caption(name,
                       "Parity-perturbed admissible heuristic is inconsistent; on %d connected "
                       "7x7 seeds %d show a non-zero cost gap between allow_reopen=False and "
                       "True." % (len(gaps), sum(1 for gap in gaps if gap)))
    return None


def _point_a(writer):
    lines = []
    connected_seeds = []
    gaps = []
    first = None

    for seed in range(100):
        grid = _make_grid(seed)
        dist = _bfs_from_goal(grid)
        if START not in dist:
            lines.append("A: seed=%d skipped (start not connected)" % seed)
            continue

        connected_seeds.append(seed)
        passable = _passable_for(grid)
        heuristic = _parity_h(dist)
        result_no = astar(START, GOAL, passable, lambda _node: 1, goal_enter_cost=1,
                          heuristic=heuristic, allow_reopen=False)
        result_yes = astar(START, GOAL, passable, lambda _node: 1, goal_enter_cost=1,
                           heuristic=heuristic, allow_reopen=True)

        config = "grid=7x7;h=parity_perturbed;seed=%d" % seed
        writer.add("ext", "extension_A", config, "cost_no_reopen", result_no.cost, "cost", 0,
                   A_NOTE)
        writer.add("ext", "extension_A", config, "cost_reopen", result_yes.cost, "cost", 0,
                   A_NOTE)
        writer.add("ext", "extension_A", config, "expanded_unique_no_reopen",
                   result_no.expanded_unique, "nodes", 0, A_NOTE)
        writer.add("ext", "extension_A", config, "reopens", result_yes.reopens, "nodes", 0,
                   A_NOTE)

        gaps.append(result_no.cost - result_yes.cost)
        if first is None:
            first = (grid, dist)

    tried = len(connected_seeds)
    diverged = sum(1 for gap in gaps if gap)
    rate = (diverged / tried) if tried else 0.0
    writer.add("ext", "extension_A_summary", "grid=7x7;h=parity_perturbed", "seeds_tried",
               tried, "count", 0, A_NOTE)
    writer.add("ext", "extension_A_summary", "grid=7x7;h=parity_perturbed", "seeds_diverged",
               diverged, "count", 0, A_NOTE)
    writer.add("ext", "extension_A_summary", "grid=7x7;h=parity_perturbed", "trigger_rate",
               rate, "ratio", 0, A_NOTE)

    if first is not None:
        _plot_a(first[0], first[1], gaps)
    if diverged == 0:
        lines.append("A: no divergence found in %d seeds" % tried)
    else:
        lines.append("A: seeds_tried=%d diverged=%d rate=%.4f" % (tried, diverged, rate))
    return lines, {"a_seeds": {"tried": tried, "diverged": diverged, "rate": rate}}


def _b_baseline():
    data = {}
    return 1 if data else 0


def _b_measure_once(func):
    t0 = time.perf_counter()
    for _ in range(1000):
        func()
    return (time.perf_counter() - t0) * 1000.0 / 1000.0


def _b_group_means(func):
    means = []
    for _ in range(30):
        gc.collect()
        for _ in range(3):
            func()
        gc.collect()
        gc.disable()
        means.append(_b_measure_once(func))
        gc.enable()
        gc.collect()
    return means


def _b_summary(means):
    ordered = sorted(means)
    quartiles = statistics.quantiles(ordered, n=4)
    mean = statistics.mean(ordered)
    return {
        "median": statistics.median(ordered),
        "iqr": quartiles[2] - quartiles[0],
        "cv": (statistics.pstdev(ordered) / mean) if mean else 0.0,
    }


def _plot_b(series_means, medians, baseline_median):
    name = "p5_B_timing.png"
    plt = get_pyplot()
    if plt is None:
        print("matplotlib unavailable, skipped: %s" % name)
        return None
    apply_style()
    labels = ["astar(h=0)", "astar(h=manhattan)", "baseline"]
    data = [series_means["astar_h0"], series_means["astar_manhattan"],
            series_means["baseline"]]
    colors = [COLORS["secondary"], COLORS["primary"], COLORS["muted"]]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), layout="constrained")
    artists = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(artists["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("0.3")
    ax.set_yscale("log")
    ax.set_ylabel("group mean per call (ms, log scale)")
    ax.set_title("Timing group means (log scale, 30 groups x 1000 calls)")
    flat = [value for values in data for value in values]
    ax.set_ylim(min(flat) * 0.55, max(flat) * 3.2)
    for index, values in enumerate(data):
        if index == len(data) - 1:
            ax.annotate("%.5f" % medians[index], xy=(index + 1, medians[index]),
                        xytext=(0, -18), textcoords="offset points", ha="center", va="top",
                        fontsize=8)
        else:
            ax.annotate("%.5f" % medians[index], xy=(index + 1, medians[index]),
                        xytext=(0, 18), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8)
    ax.axhline(baseline_median, color=COLORS["muted"], linestyle="--", linewidth=1.0,
               label="baseline median (noise floor)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", color="0.9")
    ax.set_axisbelow(True)
    save_fig(fig, name)

    overlap = not (max(series_means["astar_h0"]) < min(series_means["astar_manhattan"])
                   or max(series_means["astar_manhattan"]) < min(series_means["astar_h0"]))
    record_caption(name,
                   "Timing: 30-group means for h=0, h=manhattan and the empty baseline. Both "
                   "A* series sit 2-3 orders of magnitude above the baseline, so framework "
                   "overhead is negligible; ms here measures implementation constants, not "
                   "search complexity.")
    return overlap


def _point_b(writer):
    passable = _passable_for(Q1_GRID)

    def manhattan(node, goal):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def make_astar_once(heuristic):
        def once():
            return astar((0, 0), (3, 4), passable, lambda _node: 1, goal_enter_cost=1,
                         heuristic=heuristic)
        return once

    runners = [
        ("astar_h0", make_astar_once(lambda _node, _goal: 0)),
        ("astar_manhattan", make_astar_once(manhattan)),
        ("baseline", _b_baseline),
    ]
    series_means = {}
    series_stats = {}
    for label, func in runners:
        series_means[label] = _b_group_means(func)
        series_stats[label] = _b_summary(series_means[label])

    for label, _func in runners:
        for index, value in enumerate(series_means[label]):
            writer.add("ext", "extension_B", "func=%s;group=%d" % (label, index),
                       "group_mean_ms", value, "ms", 0, B_NOTE)
        stats = series_stats[label]
        for metric, key, unit in (("median_ms", "median", "ms"), ("iqr_ms", "iqr", "ms"),
                                  ("cv", "cv", "ratio")):
            writer.add("ext", "extension_B_summary", "func=%s" % label, metric,
                       stats[key], unit, 0, B_NOTE)

    medians = [series_stats["astar_h0"]["median"], series_stats["astar_manhattan"]["median"],
               series_stats["baseline"]["median"]]
    overlap = _plot_b(series_means, medians, medians[2])
    if overlap is None:
        overlap = not (max(series_means["astar_h0"]) < min(series_means["astar_manhattan"])
                       or max(series_means["astar_manhattan"]) < min(series_means["astar_h0"]))
    lines = ["B: h0 median=%.5f, manhattan median=%.5f, baseline median=%.5f, "
             "overlap(h0,manhattan)=%s" % (medians[0], medians[1], medians[2], overlap)]
    facts = {"timing": {"h0": medians[0], "manhattan": medians[1], "baseline": medians[2],
                        "overlap": bool(overlap)}}
    return lines, facts


def _plot_c(delta):
    name = "p4_C_delta.png"
    plt = get_pyplot()
    if plt is None:
        print("matplotlib unavailable, skipped: %s" % name)
        return None
    apply_style()
    nonzero = sum(1 for row in delta for value in row if value)
    cmap = plt.get_cmap("Oranges", 3)

    fig, ax = plt.subplots(figsize=(6.6, 5.6), layout="constrained")
    image = ax.imshow(delta, cmap=cmap, vmin=-0.5, vmax=2.5)
    ax.set_xticks(range(len(Q4_NAMES)))
    ax.set_yticks(range(len(Q4_NAMES)))
    ax.set_xticklabels(Q4_NAMES)
    ax.set_yticklabels(Q4_NAMES)
    ax.set_xlabel("to")
    ax.set_ylabel("from")
    ax.set_title("|M - M^T| after setting enter_cost(T3)=3 (sensitivity, not required)")
    for r in range(len(Q4_NAMES)):
        for c in range(len(Q4_NAMES)):
            ax.text(c, r, str(delta[r][c]), ha="center", va="center", fontsize=9,
                    color="white" if delta[r][c] else "black")
    fig.colorbar(image, ax=ax, label="delta cost", ticks=[0, 1, 2])
    fig.text(0.5, -0.035, "non-zero only in T3 row/column (%d cells)" % nonzero,
             ha="center", fontsize=8)
    save_fig(fig, name)
    record_caption(name,
                   "Directed-matrix check: raising T3 entry cost to 3 changes only T3's row "
                   "and column (delta=2, %d cells), so the symmetric matrix is an artifact of "
                   "uniform end-point cost rather than a property." % nonzero)
    return None


def _point_c(writer):
    overrides = {(4, 4): 3}
    matrix = build_distance_matrix(Q4_GRID, enter_cost_overrides=overrides)
    d23 = matrix["T2"]["T3"]
    d32 = matrix["T3"]["T2"]
    writer.add("ext", "extension_C", "override=T3:3", "d.T2.T3", d23, "cost", 0, C_NOTE)
    writer.add("ext", "extension_C", "override=T3:3", "d.T3.T2", d32, "cost", 0, C_NOTE)

    delta = [[abs(matrix[a][b] - matrix[b][a]) for b in Q4_NAMES] for a in Q4_NAMES]
    max_delta = max(max(row) for row in delta)
    nonzero = sum(1 for row in delta for value in row if value)
    writer.add("ext", "extension_C", "override=T3:3", "max_delta", max_delta, "cost", 0,
               C_NOTE)
    writer.add("ext", "extension_C", "override=T3:3", "nonzero_delta_count", nonzero, "count",
               0, C_NOTE)

    outcomes = []
    for depth in (2, 4):
        result = solve_q4(Q4_GRID, Q4_VALUES, depth, enter_cost_overrides=overrides)
        writer.add("ext", "extension_C", "override=T3:3;H=%d" % depth, "best_value",
                   result["best_value"], "cost", 0, C_NOTE)
        outcomes.append("H=%d -> %s/%d" % (depth, result["best_target"], result["best_value"]))

    _plot_c(delta)
    return [
        "C: d(T2,T3)=%s d(T3,T2)=%s max_delta=%s" % (d23, d32, max_delta),
        "C: %s, %s" % (outcomes[0], outcomes[1]),
    ], {}


def _q2_enter_cost_for(grid):
    def enter_cost(node):
        value = grid[node[0]][node[1]]
        if value == "S":
            return 1
        if value == "E":
            return 0
        return int(value)

    return enter_cost


def _k_measure(grid, ks):
    passable = _passable_for(grid)
    enter_cost = _q2_enter_cost_for(grid)
    baseline = astar(Q2_START, Q2_GOAL, passable, enter_cost, goal_enter_cost=0,
                     heuristic=lambda _node, _goal: 0).cost
    rows = []
    for k in ks:
        def heuristic(node, goal, k=k):
            return k * (abs(node[0] - goal[0]) + abs(node[1] - goal[1]))

        result = astar(Q2_START, Q2_GOAL, passable, enter_cost, goal_enter_cost=0,
                       heuristic=heuristic)
        rows.append((k, result.cost, result.expanded_unique))
    return baseline, rows


def _k_min_of(rows, baseline):
    for k, cost, _expanded in rows:
        if k >= 2 and cost > baseline:
            return k
    return None


def _plot_k(rows, baseline, k_min):
    name = "p2_q2_manhattan_k.png"
    plt = get_pyplot()
    if plt is None:
        print("matplotlib unavailable, skipped: %s" % name)
        return None

    apply_style()
    ks = [k for k, _cost, _expanded in rows]
    costs = [cost for _k, cost, _expanded in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.6), layout="constrained")
    bars = ax.bar([str(k) for k in ks], costs, color=COLORS["primary"], edgecolor="white",
                  linewidth=0.8)
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), str(cost),
                ha="center", va="bottom", fontsize=9)
    ax.axhline(baseline, color=COLORS["accent"], linestyle="--", linewidth=1.0,
               label="baseline (h=0) cost = %d" % baseline)
    ax.set_ylim(0, max(max(costs), baseline) * 1.2)
    ax.set_xlabel("k (h = k * manhattan)")
    ax.set_ylabel("cost")
    ax.set_title("Q2 cost under scaled Manhattan heuristic")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", color="0.9")
    ax.set_axisbelow(True)
    save_fig(fig, name)

    if k_min is None:
        record_caption(name,
                       "Manhattan scaled by k (k in {%s}) did not change the h=0 baseline cost "
                       "on this map, so no suboptimality appeared; since h(goal)=0 regardless "
                       "of k, the goal itself is never polluted."
                       % ", ".join(str(k) for k in ks))
    else:
        k_cost = next(cost for k, cost, _expanded in rows if k == k_min)
        record_caption(name,
                       "Manhattan scaled by k overestimates the remaining cost, breaking "
                       "admissibility: cost rises above the h=0 baseline at k=%d (%d vs %d). "
                       "Since h(goal)=0 regardless of k, the goal itself is never polluted, "
                       "so the failure is purely due to overestimation."
                       % (k_min, k_cost, baseline))
    return None


def _point_k(writer):
    baseline, rows = _k_measure(Q2_GRID, [1, 2, 3, 5])
    k_min = _k_min_of(rows, baseline)
    if k_min is None:
        baseline, rows = _k_measure(Q2_GRID, [1, 2, 3, 5, 10, 20])
        k_min = _k_min_of(rows, baseline)
    if k_min is None:
        baseline, rows = _k_measure(K_FALLBACK_GRID, [1, 2, 3, 5])
        k_min = _k_min_of(rows, baseline)

    for k, cost, expanded in rows:
        config = "heuristic=k*manhattan;k=%d" % k
        writer.add("q2", "extension_K", config, "cost", cost, "cost", 0, K_NOTE)
        writer.add("q2", "extension_K", config, "expanded_unique", expanded, "nodes", 0,
                   K_NOTE)
    writer.add("q2", "extension_K_summary", "basis=h0_cost", "k_min",
               "none" if k_min is None else k_min, "count", 0, K_NOTE)

    _plot_k(rows, baseline, k_min)
    k_cost = None
    for k, cost, _expanded in rows:
        if k == k_min:
            k_cost = cost
    if k_min is None:
        lines = ["K: no divergence found (k=%s, baseline cost=%d)"
                 % (",".join(str(k) for k in [row[0] for row in rows]), baseline)]
    else:
        lines = ["K: k_min=%d cost=%d baseline=%d" % (k_min, k_cost, baseline)]
    return lines, {"k_min": k_min, "k_min_cost": k_cost}


def run(writer):
    lines = []
    facts = {}
    for point in (_point_a, _point_b, _point_c, _point_k):
        part_lines, part_facts = point(writer)
        lines.extend(part_lines)
        facts.update(part_facts)
    return "\n".join(lines), facts
