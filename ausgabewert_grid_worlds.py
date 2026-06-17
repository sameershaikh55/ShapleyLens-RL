"""
Ausgabewert-Auswertung: gwa, gwb, gwc, gwd.

Berechnet und vergleicht Ausgabewerte (Shapley, Banzhaf, …) pro Grid World.
Ergebnisse → Ordner ``Ausgabewert/`` (nicht Teil des RL-Trainings).

    python ausgabewert_grid_worlds.py
    python ausgabewert_grid_worlds.py gwa gwb
"""

from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from characteristics import Characteristics
from shapley import Shapley
from banzhaf import Banzhaf
from nucleolus import Nucleolus
from tau import TauValue
from utopia_payoff import UtopiaPayoff
from gately import Gately
from q_agent_1 import Agent
from utils import F_not_i, get_state_dist, train, tqdm_label

import matplotlib.pyplot as plt

OUT = ROOT / "Ausgabewert"
CHAR = "local_sverl"
FEATURE_NAMES = ["x", "y"]
METHODS = ("shapley", "banzhaf", "nucleolus", "tau", "utopia", "gately")
METHOD_LABELS = {
    "shapley": "Shapley",
    "banzhaf": "Banzhaf",
    "nucleolus": "Nucleolus",
    "tau": "Tau",
    "utopia": "Utopia",
    "gately": "Gately",
}

TRAIN_STEPS = 300_000
STATE_SAMPLES = 50_000
NUM_ROLLS = 5_000

CONFIGS = {
    "gwa": {
        "grid": ("gwa.gwa", "Grid", []),
        "states": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "epsilon": 1.0,
        "desc": "2×3 Grid, Ziel oben, 4 Zustände.",
    },
    "gwb": {
        "grid": ("gwb.gwb", "Grid", []),
        "states": np.array([[0, 0], [1, 0], [1, 1], [1, 2]]),
        "epsilon": 1.0,
        "desc": "2×4 Grid, Hindernis bei (0,1), 4 Zustände.",
    },
    "gwc": {
        "grid": ("gwc.gwc", "Grid", []),
        "states": np.array([[0, 0], [1, 0], [1, 1], [1, 2], [0, 2]]),
        "epsilon": 1.0,
        "desc": "2×4 Grid, zwei Hindernisse, 5 Zustände.",
    },
    "gwd": {
        "grid": ("gwd.gwd", "Grid", [10, 10, 20]),
        "states": None,
        "epsilon": 0.1,
        "desc": "10×10 Grid, 20 Blöcke.",
        "seed": 42,
    },
}


def load_grid(name: str, cfg: dict):
    mod_name, cls_name, args = cfg["grid"]
    mod = importlib.import_module(mod_name)
    Grid = getattr(mod, cls_name)
    if name == "gwd":
        np.random.seed(cfg.get("seed", 42))
    return Grid(*args) if args else Grid()


def scalar(v) -> float:
    arr = np.asarray(v, dtype=float)
    return float(arr.item() if arr.ndim == 0 else np.mean(arr))


def _state_key(data: dict, state) -> tuple | None:
    st = tuple(int(x) for x in state)
    if st in data:
        return st
    return next((k for k in data if tuple(k) == st), None)


def format_state(state) -> str:
    return "(" + ", ".join(str(int(x)) for x in state) + ")"


def comparison_table(results: dict, states, feature_names: list[str] | None = None) -> str:
    names = feature_names or FEATURE_NAMES
    headers = [METHOD_LABELS[m] for m in METHODS]
    align = "|".join(["--------:"] * len(METHODS))
    lines = [
        "| Zustand | Feature | " + " | ".join(headers) + " |",
        "|:--------|:-------:|" + align + "|",
    ]
    for state in states:
        st = tuple(int(x) for x in state)
        for fi, fn in enumerate(names):
            row = [f"`{format_state(st)}`", fn]
            for m in METHODS:
                d = results.get(m, {}).get(CHAR, {})
                key = _state_key(d, st)
                row.append(f"{scalar(d[key][fi]):.3f}" if key is not None else "—")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def mean_comparison_table(results: dict, states, feature_names: list[str] | None = None) -> str:
    """Mittelwert über alle Zustände — kompakte Gesamtübersicht."""
    names = feature_names or FEATURE_NAMES
    headers = [METHOD_LABELS[m] for m in METHODS]
    align = "|".join(["--------:"] * len(METHODS))
    lines = [
        "| Feature | " + " | ".join(headers) + " |",
        "|:-------:|" + align + "|",
    ]
    for fi, fn in enumerate(names):
        row = [fn]
        for m in METHODS:
            d = results.get(m, {}).get(CHAR, {})
            vals = []
            for state in states:
                key = _state_key(d, state)
                if key is not None:
                    vals.append(scalar(d[key][fi]))
            row.append(f"{np.mean(vals):.3f}" if vals else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _mean_matrix(results: dict, states, feature_names: list[str] | None = None) -> np.ndarray:
    """Shape: (n_features, n_methods) — Mittel über alle Zustände."""
    names = feature_names or FEATURE_NAMES
    mat = np.zeros((len(names), len(METHODS)))
    for mi, m in enumerate(METHODS):
        d = results.get(m, {}).get(CHAR, {})
        for fi in range(len(names)):
            vals = []
            for state in states:
                key = _state_key(d, state)
                if key is not None:
                    vals.append(scalar(d[key][fi]))
            mat[fi, mi] = np.mean(vals) if vals else 0.0
    return mat


def save_summary_plot(
    results: dict, states, out_dir: Path, game_name: str,
    feature_names: list[str] | None = None,
) -> None:
    """Ein Gesamt-Diagramm pro Spiel: Mittelwert über alle Zustände."""
    names = feature_names or FEATURE_NAMES
    out_dir.mkdir(parents=True, exist_ok=True)
    mat = _mean_matrix(results, states, names)
    n_feat, n_methods = mat.shape

    if n_feat > 4:
        fig, ax = plt.subplots(figsize=(10, max(4, n_feat * 0.45 + 1)))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=30, ha="right")
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(names)
        ax.set_title(f"{game_name.upper()} — Mittelwert pro Feature")
        for r in range(n_feat):
            for c in range(n_methods):
                ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "vergleich_gesamt.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    x = np.arange(n_methods)
    width = 0.35
    for fi, fn in enumerate(names):
        offset = (fi - 0.5) * width
        ax.bar(x + offset, mat[fi], width, label=fn)
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=30, ha="right")
    ax.set_ylabel("Mittlerer Ausgabewert")
    ax.set_title(f"{game_name.upper()} — Mittel über alle Zustände")
    ax.legend()

    # Heatmap (kompakt)
    ax = axes[1]
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=30, ha="right")
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(names)
    ax.set_title("Übersicht (Mittelwert)")
    for r in range(n_feat):
        for c in range(n_methods):
            ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    path = out_dir / "vergleich_gesamt.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_comparison_plots(
    results: dict, states, out_dir: Path, game_name: str,
    feature_names: list[str] | None = None,
) -> None:
    save_summary_plot(results, states, out_dir, game_name, feature_names)


def run_game(name: str, cfg: dict) -> dict:
    print(f"\n{'='*60}\n  {name.upper()}\n{'='*60}")
    env = load_grid(name, cfg)
    agent = Agent(env.state_dim, env.num_actions, epsilon=cfg["epsilon"], gamma=1, alpha=0.2)

    train(agent, env, TRAIN_STEPS)
    agent.get_policy()
    agent.get_value_table()

    states = cfg["states"]
    state_dist = get_state_dist(agent, env, STATE_SAMPLES)
    if states is None:
        states = np.unique(np.array(list(state_dist)), axis=0)

    F = np.arange(env.state_dim)
    ch = Characteristics(env, states)
    ch.pi_Cs = {
        tuple(C): dict(agent.get_pi_C(C, state_dist, states))
        for C in tqdm_label(F_not_i(F), f"{name} pi_C")
    }
    ch.v_Cs = {
        tuple(C): agent.get_v_C(C, state_dist, states)
        for C in tqdm_label(F_not_i(F), f"{name} v_C")
    }
    char_data = ch.local_sverl_C_values(NUM_ROLLS, ch.pi_Cs, multi_process=False, num_p=1)

    calculators = {
        "shapley": Shapley(states),
        "banzhaf": Banzhaf(states, normalized=True),
        "nucleolus": Nucleolus(states),
        "tau": TauValue(states, strict=False),
        "utopia": UtopiaPayoff(states, normalized=True),
        "gately": Gately(states),
    }
    results = {}
    for m in METHODS:
        try:
            results[m] = {CHAR: calculators[m].run(char_data)}
        except Exception as exc:
            print(f"  WARN {m}: {exc}")

    return {"results": results, "states": states}


def write_game_report(name: str, cfg: dict, data: dict) -> None:
    game_dir = OUT / name
    md = [
        f"# {name.upper()} — Ausgabewert-Vergleich\n\n",
        f"{cfg['desc']}\n\n",
        f"Charakteristik: **{CHAR}** | Train: {TRAIN_STEPS:,} | Rolls: {NUM_ROLLS:,}\n\n",
        "## Gesamtvergleich (Mittel über alle Zustände)\n\n",
        mean_comparison_table(data["results"], data["states"]),
        "\n\n",
        f"![{name} gesamt](vergleich_gesamt.png)\n\n",
        "## Detailtabelle (pro Zustand)\n\n",
        comparison_table(data["results"], data["states"]),
        "\n",
    ]
    (game_dir / "AUSGABEWERTE.md").write_text("".join(md), encoding="utf-8")


def write_summary(all_data: dict) -> None:
    parts = [
        "# Ausgabewert-Vergleich — gwa, gwb, gwc, gwd\n\n",
        "Pro Spiel: Ausgabewerte berechnen und **innerhalb des Spiels** vergleichen.\n\n",
    ]
    for name, data in all_data.items():
        parts.append(f"## {name.upper()}\n\n")
        parts.append(f"{CONFIGS[name]['desc']}\n\n")
        parts.append("### Gesamtvergleich\n\n")
        parts.append(mean_comparison_table(data["results"], data["states"]))
        parts.append("\n\n")
        parts.append(f"![{name}]({name}/vergleich_gesamt.png)\n\n")
        parts.append("<details><summary>Detailtabelle (alle Zustände)</summary>\n\n")
        parts.append(comparison_table(data["results"], data["states"]))
        parts.append("\n\n</details>\n\n---\n\n")
    (OUT / "GRID_WORLDS_AUSGABEWERTE.md").write_text("".join(parts), encoding="utf-8")
    print(f"Bericht: {OUT / 'GRID_WORLDS_AUSGABEWERTE.md'}")


def main():
    names = [a for a in sys.argv[1:] if a in CONFIGS] or list(CONFIGS)
    OUT.mkdir(parents=True, exist_ok=True)
    all_data = {}

    for name in names:
        cfg = CONFIGS[name]
        game_dir = OUT / name
        game_dir.mkdir(parents=True, exist_ok=True)

        data = run_game(name, cfg)
        all_data[name] = data

        with open(game_dir / "results.pkl", "wb") as f:
            pickle.dump(data["results"], f)

        save_comparison_plots(data["results"], data["states"], game_dir, name)
        write_game_report(name, cfg, data)
        print(f"  -> {game_dir}")

    write_summary(all_data)
    print(f"\nFertig: {OUT.resolve()}")


if __name__ == "__main__":
    main()
