"""Load and plot aggregated Snakemake experiment results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RESULTS = Path("checkpoints/rmd17_aspirin/experiment_results.json")
DEFAULT_OUTPUT_DIR = Path("checkpoints/rmd17_aspirin/plots")

RUN_RANDOM_RE = re.compile(
    r"^(?P<molecule>.+)_split(?P<split>\d+)_(?P<selection>random)_seed(?P<seed>\d+)$"
)
RUN_METRIC_RE = re.compile(
    r"^(?P<molecule>.+)_split(?P<split>\d+)_(?P<selection>.+)_ws(?P<ws>\d+)_st(?P<st>\d+)_seed(?P<seed>\d+)$"
)


def load_results(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


def parse_run_name(run_name: str) -> dict[str, Any]:
    teacher_noise = run_name.endswith("_teacher_noise")
    base = run_name[: -len("_teacher_noise")] if teacher_noise else run_name

    match = RUN_RANDOM_RE.match(base)
    if match:
        groups = match.groupdict()
        return {
            "run_name": run_name,
            "molecule": groups["molecule"],
            "split_id": int(groups["split"]),
            "selection": groups["selection"],
            "window_size": np.nan,
            "stride": np.nan,
            "seed": int(groups["seed"]),
            "teacher_noise": teacher_noise,
        }

    match = RUN_METRIC_RE.match(base)
    if match:
        groups = match.groupdict()
        return {
            "run_name": run_name,
            "molecule": groups["molecule"],
            "split_id": int(groups["split"]),
            "selection": groups["selection"],
            "window_size": int(groups["ws"]),
            "stride": int(groups["st"]),
            "seed": int(groups["seed"]),
            "teacher_noise": teacher_noise,
        }

    return {
        "run_name": run_name,
        "molecule": None,
        "split_id": np.nan,
        "selection": run_name,
        "window_size": np.nan,
        "stride": np.nan,
        "seed": np.nan,
        "teacher_noise": teacher_noise,
    }


def _metric_block(entry: dict[str, Any], *path: str) -> dict[str, Any]:
    node: Any = entry
    for key in path:
        if not isinstance(node, dict):
            return {}
        node = node.get(key, {})
    return node if isinstance(node, dict) else {}


def results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in results:
        meta = parse_run_name(str(entry.get("run_name", "")))
        teacher_forces = _metric_block(entry, "teacher", "test_metrics", "forces") or _metric_block(
            entry, "forces"
        )
        teacher_energy = _metric_block(entry, "teacher", "test_metrics", "energy") or _metric_block(
            entry, "energy"
        )
        student_forces = _metric_block(entry, "student", "test_metrics", "forces")
        student_energy = _metric_block(entry, "student", "test_metrics", "energy")
        distill = entry.get("distillation_metrics", {}) or {}

        rows.append(
            {
                **meta,
                "teacher_best_loss": entry.get("best_loss"),
                "student_best_loss": _metric_block(entry, "student").get("best_loss"),
                "teacher_force_rmse_ev_A": teacher_forces.get("rmse_ev_A"),
                "student_force_rmse_ev_A": student_forces.get("rmse_ev_A"),
                "teacher_force_mae_ev_A": teacher_forces.get("mae_ev_A"),
                "student_force_mae_ev_A": student_forces.get("mae_ev_A"),
                "teacher_energy_rmse_ev": teacher_energy.get("rmse_ev"),
                "student_energy_rmse_ev": student_energy.get("rmse_ev"),
                "kl_teacher_to_student": distill.get("kl_teacher_to_student_test_force_hist"),
                "kl_student_to_teacher": distill.get("kl_student_to_teacher_test_force_hist"),
                "student_auc_excess_valid_loss": _metric_block(entry, "student").get(
                    "auc_excess_valid_loss"
                ),
            }
        )
    return pd.DataFrame(rows)


def _selection_label(selection: str) -> str:
    labels = {
        "random": "Random",
        "gzip_bytes_cart": "gzip(cart)",
        "gzip_bytes_zmat": "gzip(zmat)",
        "gzip_bytes_soap": "gzip(soap)",
    }
    return labels.get(selection, selection)


def _aggregate_by_selection(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["selection", "teacher_noise"], dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["label"] = grouped["selection"].map(_selection_label)
    noise = grouped["teacher_noise"].map({False: "baseline", True: "teacher noise"})
    grouped["variant"] = grouped["label"] + " (" + noise + ")"
    return grouped


def plot_force_rmse(df: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar chart: teacher vs student test force RMSE by selection."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    variants = [False, True]
    titles = ["Baseline", "Teacher noise"]

    for ax, teacher_noise, title in zip(axes, variants, titles):
        sub = df[df["teacher_noise"] == teacher_noise]
        if sub.empty:
            ax.set_visible(False)
            continue

        order = ["random", "gzip_bytes_cart", "gzip_bytes_zmat", "gzip_bytes_soap"]
        present = [s for s in order if s in sub["selection"].unique()]
        extra = [s for s in sorted(sub["selection"].unique()) if s not in present]
        order = present + extra

        x = np.arange(len(order))
        width = 0.35
        teacher_vals = []
        student_vals = []
        teacher_err = []
        student_err = []
        for sel in order:
            chunk = sub[sub["selection"] == sel]
            teacher_vals.append(chunk["teacher_force_rmse_ev_A"].mean())
            student_vals.append(chunk["student_force_rmse_ev_A"].mean())
            teacher_err.append(chunk["teacher_force_rmse_ev_A"].std(ddof=0))
            student_err.append(chunk["student_force_rmse_ev_A"].std(ddof=0))

        ax.bar(x - width / 2, teacher_vals, width, yerr=teacher_err, capsize=3, label="Teacher")
        ax.bar(x + width / 2, student_vals, width, yerr=student_err, capsize=3, label="Student")
        ax.set_xticks(x)
        ax.set_xticklabels([_selection_label(s) for s in order], rotation=20, ha="right")
        ax.set_ylabel("Test force RMSE (eV/Å)")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Test force RMSE by data-selection method", y=1.02)
    fig.tight_layout()
    out = output_dir / "force_rmse_teacher_vs_student.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_kl_panel(ax: plt.Axes, sub: pd.DataFrame, title: str) -> None:
    agg = _aggregate_by_selection(sub, "kl_teacher_to_student")
    x = np.arange(len(agg))
    agg2 = _aggregate_by_selection(sub, "kl_student_to_teacher")
    ax.bar(x - 0.2, agg["mean"], 0.4, yerr=agg["std"], capsize=3, label="KL(T→S)")
    ax.bar(x + 0.2, agg2["mean"], 0.4, yerr=agg2["std"], capsize=3, label="KL(S→T)")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["label"], rotation=20, ha="right")
    ax.set_ylabel("KL divergence")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def plot_kl_and_auc(df: pd.DataFrame, output_dir: Path) -> Path:
    """KL divergences and student AUC excess valid loss."""
    has_baseline = not df[~df["teacher_noise"]].empty
    has_noise = not df[df["teacher_noise"]].empty
    n_kl = int(has_baseline) + int(has_noise)
    fig, axes = plt.subplots(1, max(2, n_kl + 1), figsize=(6 * max(2, n_kl + 1), 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    kl_axes = axes[:n_kl] if n_kl else [axes[0]]
    idx = 0
    if has_baseline:
        _plot_kl_panel(kl_axes[idx], df[~df["teacher_noise"]], "Baseline — force histogram KL")
        idx += 1
    if has_noise:
        _plot_kl_panel(kl_axes[idx], df[df["teacher_noise"]], "Teacher noise — force histogram KL")
        idx += 1
    if n_kl == 0:
        kl_axes[0].text(0.5, 0.5, "No KL data", ha="center", va="center")
        kl_axes[0].set_axis_off()

    ax_auc = axes[-1]
    sub = df[~df["student_auc_excess_valid_loss"].isna()]
    if sub.empty:
        ax_auc.text(0.5, 0.5, "No student AUC data", ha="center", va="center")
        ax_auc.set_axis_off()
    else:
        agg = (
            sub.groupby(["selection", "teacher_noise"], dropna=False)["student_auc_excess_valid_loss"]
            .agg(["mean", "std"])
            .reset_index()
        )
        agg["label"] = agg.apply(
            lambda r: f"{_selection_label(r['selection'])} ({'noise' if r['teacher_noise'] else 'base'})",
            axis=1,
        )
        x = np.arange(len(agg))
        ax_auc.bar(x, agg["mean"], yerr=agg["std"], capsize=3, color="tab:green")
        ax_auc.set_xticks(x)
        ax_auc.set_xticklabels(agg["label"], rotation=25, ha="right")
        ax_auc.set_ylabel("AUC excess valid loss")
        ax_auc.set_title("Student validation curve summary")
        ax_auc.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = output_dir / "kl_and_student_auc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_force_histograms(results: list[dict[str, Any]], output_dir: Path, max_runs: int = 4) -> Path | None:
    """Overlay normalized test-force histograms for a few runs."""
    candidates = [
        r
        for r in results
        if isinstance(r.get("distillation_metrics"), dict)
        and r["distillation_metrics"].get("teacher_hist")
    ]
    if not candidates:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for entry in candidates[:max_runs]:
        distill = entry["distillation_metrics"]
        edges = np.asarray(distill["hist_bin_edges"], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        teacher = np.asarray(distill["teacher_hist"], dtype=float)
        student = np.asarray(distill["student_hist"], dtype=float)
        label = str(entry.get("run_name", ""))[:48]
        ax.plot(centers, teacher, alpha=0.8, label=f"{label} (T)")
        ax.plot(centers, student, alpha=0.6, linestyle="--", label=f"{label} (S)")

    ax.set_xlabel("Test force component (model units)")
    ax.set_ylabel("Normalized count")
    ax.set_title("Teacher vs student test-force histograms (sample runs)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = output_dir / "force_histograms_sample.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_all(results: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = results_to_dataframe(results)
    df.to_csv(output_dir / "experiment_results_summary.csv", index=False)

    paths = [
        plot_force_rmse(df, output_dir),
        plot_kl_and_auc(df, output_dir),
    ]
    hist_path = plot_force_histograms(results, output_dir)
    if hist_path is not None:
        paths.append(hist_path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-json",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Path to aggregated results (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG/CSV outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = args.results_json.expanduser().resolve()
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Results file not found: {results_path}\n"
            "Run Snakemake first, e.g.:\n"
            "  snakemake checkpoints/rmd17_aspirin/experiment_results.json -j 1"
        )

    results = load_results(results_path)
    print(f"Loaded {len(results)} runs from {results_path}")
    df = results_to_dataframe(results)
    print(df[["run_name", "selection", "seed", "teacher_noise", "teacher_force_rmse_ev_A", "student_force_rmse_ev_A"]].to_string(index=False))

    paths = plot_all(results, args.output_dir.resolve())
    for path in paths:
        print(f"Wrote {path}")

    if args.show:
        for path in paths:
            if path.suffix == ".png":
                img = plt.imread(path)
                plt.figure(figsize=(10, 6))
                plt.imshow(img)
                plt.axis("off")
                plt.title(path.name)
        plt.show()


if __name__ == "__main__":
    main()
