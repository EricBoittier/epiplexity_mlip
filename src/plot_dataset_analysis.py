"""Plot official RMD17 split distributions, KL divergences, pair distances, and gzip metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmml.data.rmd17 import load_rmd17_npz, resolve_rmd17_splits_dir
from mmml.interfaces.chemcoordInterface.interface import patch_chemcoord_for_pandas3

from src.config import SoapConfig
from src.dataset_distributions import (
    GZIP_METRIC_COLUMNS,
    build_subset_catalog,
    collect_pair_types,
    histograms_with_shared_bins,
    kl_matrix,
    kl_matrix_pair_types,
    mean_force_per_structure,
    pair_distance_histograms,
    scalar_histogram,
    subset_label,
)
from src.experiment import scan_information_content

patch_chemcoord_for_pandas3()


def _bool_arg(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def plot_kl_heatmap(df: pd.DataFrame, title: str, output_path: Path) -> None:
    labels = list(df.index)
    fig_w = max(8.0, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
    data = df.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="KL(P || Q)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overview_split(
    *,
    split_id: int,
    train_ds: dict[str, Any],
    test_ds: dict[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    e_train = np.asarray(train_ds["E"], dtype=float).ravel()
    e_test = np.asarray(test_ds["E"], dtype=float).ravel()
    axes[0].hist(e_train, bins=80, alpha=0.6, density=True, label="train pool")
    axes[0].hist(e_test, bins=80, alpha=0.6, density=True, label="test")
    axes[0].set_xlabel("Energy (dataset units)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Split {split_id} — energy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    f_train = np.abs(np.asarray(train_ds["F"], dtype=float)).ravel()
    f_test = np.abs(np.asarray(test_ds["F"], dtype=float)).ravel()
    axes[1].hist(f_train, bins=80, alpha=0.6, density=True, label="train pool")
    axes[1].hist(f_test, bins=80, alpha=0.6, density=True, label="test")
    axes[1].set_xlabel("|F| (dataset units)")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Split {split_id} — |forces|")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    m_train = mean_force_per_structure(train_ds)
    m_test = mean_force_per_structure(test_ds)
    axes[2].scatter(e_train, m_train, s=8, alpha=0.35, label="train pool")
    axes[2].scatter(e_test, m_test, s=8, alpha=0.35, label="test")
    axes[2].set_xlabel("Energy")
    axes[2].set_ylabel("Mean |F| per structure")
    axes[2].set_title(f"Split {split_id} — E vs mean |F|")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_gzip_split(info_df: pd.DataFrame, split_id: int, output_path: Path) -> None:
    cols = [c for c in GZIP_METRIC_COLUMNS if c in info_df.columns]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        values = np.asarray(info_df[col], dtype=float)
        ax.hist(values, bins=40, color="tab:blue", alpha=0.8)
        ax.set_xlabel(col)
        ax.set_ylabel("Window count")
        ax.set_title(f"Split {split_id}")
        ax.grid(alpha=0.3)
    if "start" in info_df.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        for col in cols:
            ax2.plot(info_df["start"], info_df[col], alpha=0.7, label=col)
        ax2.set_xlabel("Window start index")
        ax2.set_ylabel("gzip bytes")
        ax2.set_title(f"Split {split_id} — gzip vs window")
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        ts_path = output_path.parent / f"split{split_id:02d}_gzip_vs_window.png"
        fig2.savefig(ts_path, dpi=200, bbox_inches="tight")
        plt.close(fig2)
    fig.suptitle(f"Split {split_id} — gzip compression scores", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pair_kl_summary(mean_kl: pd.DataFrame, output_path: Path) -> None:
    plot_kl_heatmap(mean_kl, "Mean KL across element-pair distance types", output_path)


def run_analysis(args: argparse.Namespace) -> list[Path]:
    output_dir = Path(args.output_dir).resolve()
    for sub in ("overview", "kl", "gzip", "tables"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path).resolve()
    splits_dir = resolve_rmd17_splits_dir(
        data_path,
        Path(args.splits_dir) if args.splits_dir else None,
    )
    split_ids = [int(s) for s in args.split_ids]

    print(f"Loading {data_path} ...")
    data = load_rmd17_npz(data_path, convert_to_ev=_bool_arg(args.convert_to_ev))
    catalog = build_subset_catalog(data, splits_dir, split_ids)
    labels = sorted(catalog.keys())
    written: list[Path] = []

    for split_id in split_ids:
        train_key = subset_label(split_id, "train")
        test_key = subset_label(split_id, "test")
        out = output_dir / "overview" / f"split{split_id:02d}_energy_forces.png"
        plot_overview_split(
            split_id=split_id,
            train_ds=catalog[train_key],
            test_ds=catalog[test_key],
            output_path=out,
        )
        written.append(out)

    print("Computing energy / force histograms for KL ...")
    energy_hists = histograms_with_shared_bins(labels, catalog, kind="energy", bins=args.bins)
    force_hists = histograms_with_shared_bins(labels, catalog, kind="forces", bins=args.bins)

    kl_energy = kl_matrix(labels, energy_hists)
    kl_forces = kl_matrix(labels, force_hists)
    kl_energy.to_csv(output_dir / "tables" / "kl_energy.csv")
    kl_forces.to_csv(output_dir / "tables" / "kl_forces.csv")
    energy_path = output_dir / "kl" / "kl_energy_heatmap.png"
    forces_path = output_dir / "kl" / "kl_forces_heatmap.png"
    plot_kl_heatmap(kl_energy, "KL — energy distributions", energy_path)
    plot_kl_heatmap(kl_forces, "KL — |force| distributions", forces_path)
    written.extend([energy_path, forces_path])

    print("Computing element-pair distance histograms ...")
    pair_hists_by_label: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for label in labels:
        ds = catalog[label]
        pair_hists_by_label[label] = pair_distance_histograms(
            np.asarray(ds["R"]),
            np.asarray(ds["Z"]),
            r_max=float(args.r_max),
            bins=args.bins,
        )

    pair_types = collect_pair_types(pair_hists_by_label)
    pair_kl_mats = kl_matrix_pair_types(labels, pair_hists_by_label, pair_types)

    pair_rows: list[dict[str, Any]] = []
    for pair, mat in pair_kl_mats.items():
        mat.to_csv(output_dir / "tables" / f"kl_pair_{pair.replace('-', '_')}.csv")
        pair_path = output_dir / "kl" / f"kl_pair_distances_{pair.replace('-', '_')}.png"
        plot_kl_heatmap(mat, f"KL — {pair} interatomic distances", pair_path)
        written.append(pair_path)
        for i, li in enumerate(mat.index):
            for j, lj in enumerate(mat.columns):
                pair_rows.append(
                    {"pair_type": pair, "from": li, "to": lj, "kl": float(mat.iloc[i, j])}
                )

    pd.DataFrame(pair_rows).to_csv(output_dir / "tables" / "kl_pairs.csv", index=False)

    if pair_kl_mats:
        stack = np.zeros((len(labels), len(labels)), dtype=float)
        for mat in pair_kl_mats.values():
            aligned = mat.reindex(index=labels, columns=labels).fillna(0.0)
            stack += aligned.to_numpy(dtype=float)
        mean_kl = pd.DataFrame(stack / len(pair_kl_mats), index=labels, columns=labels)
        mean_kl.to_csv(output_dir / "tables" / "kl_pairs_mean.csv")
        summary_path = output_dir / "kl" / "kl_pair_distances_all_pairs.png"
        plot_pair_kl_summary(mean_kl, summary_path)
        written.append(summary_path)

    soap = SoapConfig()
    gzip_info_by_split: dict[int, pd.DataFrame] = {}
    print("Scanning gzip / information metrics on train pools ...")
    for split_id in split_ids:
        train_ds = catalog[subset_label(split_id, "train")]
        info_df = scan_information_content(
            train_ds,
            window_size=int(args.window_size),
            stride=int(args.stride),
            stop=args.scan_stop,
            soap=soap,
        )
        gzip_info_by_split[split_id] = info_df
        pkl_path = output_dir / "tables" / f"split{split_id:02d}_info_df.pkl"
        info_df.to_pickle(pkl_path)
        gzip_fig = output_dir / "gzip" / f"split{split_id:02d}_gzip_metrics.png"
        plot_gzip_split(info_df, split_id, gzip_fig)
        written.append(gzip_fig)

    gzip_hists: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_id in split_ids:
        info_df = gzip_info_by_split[split_id]
        for col in GZIP_METRIC_COLUMNS:
            if col not in info_df.columns:
                continue
            label = f"split{split_id:02d}_{col}"
            values = np.asarray(info_df[col], dtype=float)
            pad = max(1e-6, 0.05 * (float(values.max()) - float(values.min())))
            gzip_hists[label] = scalar_histogram(
                values,
                bins=args.bins,
                vmin=float(values.min()) - pad,
                vmax=float(values.max()) + pad,
            )

    rep_labels = sorted(gzip_hists.keys())
    if len(rep_labels) >= 2:
        kl_gzip = kl_matrix(rep_labels, gzip_hists)
        kl_gzip.to_csv(output_dir / "tables" / "kl_gzip_representations.csv")
        gzip_kl_path = output_dir / "gzip" / "gzip_kl_representations.png"
        plot_kl_heatmap(kl_gzip, "KL — gzip score distributions (per split × representation)", gzip_kl_path)
        written.append(gzip_kl_path)

    print(f"Wrote {len(written)} figures under {output_dir}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default="/scicore/home/meuwly/boitti0000/data/rmd17/npz_data/rmd17_aspirin.npz",
    )
    parser.add_argument("--splits-dir", type=str, default="")
    parser.add_argument("--split-ids", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--convert-to-ev", type=_bool_arg, default=True)
    parser.add_argument("--r-max", type=float, default=12.0, help="Max pair distance (Å)")
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--scan-stop", type=int, default=None, help="Limit frames for gzip scan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
