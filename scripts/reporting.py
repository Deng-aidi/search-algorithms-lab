"""Metrics CSV writing and figure saving helpers for the experiment scripts.

matplotlib is never imported at module import time; it is loaded lazily inside
get_pyplot()/save_fig() so that a missing matplotlib only degrades plotting.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "report_assets" / "metrics.csv"
FIGURES_DIR = ROOT / "report_assets" / "figures"

COLUMNS = ["problem", "experiment", "config", "metric", "value", "unit", "seed", "note"]
UNITS = ("nodes", "ms", "count", "ratio", "cost")
NOTE_TAGS = ("public", "boundary", "sensitivity", "deterministic")

COLORS = {
    "primary": "#4C78A8",
    "secondary": "#9ECAE1",
    "accent": "#F58518",
    "muted": "#B0B0B0",
}

_CAPTIONS = []


def apply_style():
    """Apply the shared rcParams; returns pyplot or None when matplotlib is missing."""
    plt = get_pyplot()
    if plt is None:
        return None
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "grid.color": "0.9",
        "axes.edgecolor": "0.3",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
    })
    return plt


def record_caption(name, text):
    """Print one caption candidate and keep it for the final block."""
    line = "caption %s: %s" % (name, text)
    _CAPTIONS.append(line)
    print(line)
    return line


def caption_block():
    if not _CAPTIONS:
        return "--- captions ---\n(no figures)"
    return "--- captions ---\n" + "\n".join(_CAPTIONS)


class MetricsWriter:
    """The only writer of metrics.csv; flush() writes the file exactly once."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else METRICS_PATH
        self._rows = []
        self._flushed = False

    def add(self, problem, experiment, config, metric, value, unit, seed=0,
            note="public;deterministic"):
        if unit not in UNITS:
            raise ValueError("unit not allowed: %r" % (unit,))
        if "," in str(config):
            raise ValueError("config must not contain a comma: %r" % (config,))
        tags = [tag for tag in str(note).split(";") if tag]
        for tag in tags:
            if tag not in NOTE_TAGS:
                raise ValueError("note tag not allowed: %r" % (tag,))
        self._rows.append([problem, experiment, str(config), metric, value, unit,
                           int(seed), ";".join(tags)])

    def flush(self):
        if self._flushed:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            writer.writerows(self._rows)
        self._flushed = True


def get_pyplot():
    """Return matplotlib.pyplot (Agg backend) or None when matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def save_fig(fig, name):
    plt = get_pyplot()
    if plt is None:
        print("matplotlib unavailable, skipped: %s" % name)
        return None
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
