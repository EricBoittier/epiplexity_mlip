"""Distribution and KL helpers for official RMD17 split analysis (no training)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase.data import chemical_symbols

from src.histogram_metrics import compute_normalized_histogram, kl_divergence

GZIP_METRIC_COLUMNS = ("gzip_bytes_cart", "gzip_bytes_zmat", "gzip_bytes_soap")


def element_pair_key(z_i: int, z_j: int) -> str:
    """Canonical element-pair label with sorted atomic numbers (e.g. H-C, C-O)."""
    a, b = int(z_i), int(z_j)
    if a > b:
        a, b = b, a
    return f"{chemical_symbols[a]}-{chemical_symbols[b]}"


def _subset_arrays(data: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
    n_total = len(data["E"])
    out: dict[str, Any] = {}
    for key, value in data.items():
        value_arr = np.asarray(value) if isinstance(value, np.ndarray) else value
        if isinstance(value_arr, np.ndarray) and value_arr.shape[:1] == (n_total,):
            out[key] = value_arr[idx]
        else:
            out[key] = value
    return out


def subset_label(split_id: int, subset: str) -> str:
    return f"split{split_id:02d}_{subset}"


def build_subset_catalog(
    data: dict[str, Any],
    splits_dir: Path,
    split_ids: list[int],
) -> dict[str, dict[str, Any]]:
    """Official train pool and test subsets per split id."""
    from mmml.data.rmd17 import load_rmd17_official_splits

    catalog: dict[str, dict[str, Any]] = {}
    n_total = len(data["E"])
    for split_id in split_ids:
        train_pool, test_idx = load_rmd17_official_splits(splits_dir, split_id)
        if train_pool.max() >= n_total or test_idx.max() >= n_total:
            raise ValueError(f"Split {split_id} indices out of range for dataset length {n_total}")
        catalog[subset_label(split_id, "train")] = _subset_arrays(data, train_pool)
        catalog[subset_label(split_id, "test")] = _subset_arrays(data, test_idx)
    return catalog


def _atomic_numbers_per_frame(Z: np.ndarray, n_frames: int) -> np.ndarray:
    z = np.asarray(Z)
    if z.ndim == 1:
        return np.broadcast_to(z.reshape(1, -1), (n_frames, z.shape[0]))
    return z


def pair_distance_histograms(
    R: np.ndarray,
    Z: np.ndarray,
    *,
    r_max: float = 12.0,
    bins: int = 64,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Accumulate normalized distance histograms per element-pair type."""
    r = np.asarray(R, dtype=float)
    n_frames = r.shape[0]
    z_per_frame = _atomic_numbers_per_frame(Z, n_frames)

    accum: dict[str, np.ndarray] = {}
    edges = np.linspace(0.0, float(r_max), int(bins) + 1)

    for frame in range(n_frames):
        pos = r[frame]
        z = z_per_frame[frame]
        n_atoms = pos.shape[0]
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = float(np.linalg.norm(pos[i] - pos[j]))
                if dist > r_max:
                    continue
                key = element_pair_key(int(z[i]), int(z[j]))
                hist, _ = np.histogram(np.array([dist]), bins=edges)
                if key not in accum:
                    accum[key] = hist.astype(float)
                else:
                    accum[key] += hist.astype(float)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, counts in accum.items():
        counts = counts.astype(float)
        denom = float(counts.sum())
        if denom <= 0.0:
            prob = np.full_like(counts, 1.0 / len(counts))
        else:
            prob = counts / denom
        out[key] = (prob, edges.copy())
    return out


def scalar_histogram(
    values: np.ndarray,
    *,
    bins: int,
    vmin: float,
    vmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    return compute_normalized_histogram(np.asarray(values, dtype=float).ravel(), bins=bins, vmin=vmin, vmax=vmax)


def energy_histogram(dataset: dict[str, Any], *, bins: int = 128) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(dataset["E"], dtype=float).ravel()
    pad = max(1e-6, 0.05 * (float(e.max()) - float(e.min())))
    return scalar_histogram(e, bins=bins, vmin=float(e.min()) - pad, vmax=float(e.max()) + pad)


def force_magnitude_histogram(dataset: dict[str, Any], *, bins: int = 128) -> tuple[np.ndarray, np.ndarray]:
    f = np.abs(np.asarray(dataset["F"], dtype=float)).ravel()
    pad = max(1e-6, 0.05 * (float(f.max()) - float(f.min())))
    return scalar_histogram(f, bins=bins, vmin=0.0, vmax=float(f.max()) + pad)


def align_histograms(
    h1: np.ndarray,
    h2: np.ndarray,
    edges1: np.ndarray,
    edges2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebin two histograms onto a shared edge grid (union of ranges)."""
    e1 = np.asarray(edges1, dtype=float)
    e2 = np.asarray(edges2, dtype=float)
    vmin = float(min(e1[0], e2[0]))
    vmax = float(max(e1[-1], e2[-1]))
    n_bins = max(len(h1), len(h2))
    edges = np.linspace(vmin, vmax, n_bins + 1)

    def _rebin(counts: np.ndarray, old_edges: np.ndarray) -> np.ndarray:
        centers = 0.5 * (old_edges[:-1] + old_edges[1:])
        new_centers = 0.5 * (edges[:-1] + edges[1:])
        rebinned = np.interp(new_centers, centers, counts, left=0.0, right=0.0)
        rebinned = np.maximum(rebinned, 0.0)
        s = float(rebinned.sum())
        if s <= 0.0:
            return np.full_like(rebinned, 1.0 / len(rebinned))
        return rebinned / s

    return _rebin(np.asarray(h1, dtype=float), e1), _rebin(np.asarray(h2, dtype=float), e2)


def kl_between_histograms(
    h1: np.ndarray,
    h2: np.ndarray,
    edges1: np.ndarray,
    edges2: np.ndarray,
) -> float:
    p, q = align_histograms(h1, h2, edges1, edges2)
    return kl_divergence(p, q)


def kl_matrix(
    labels: list[str],
    hist_by_label: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Asymmetric KL(P||Q) for all ordered label pairs."""
    n = len(labels)
    mat = np.zeros((n, n), dtype=float)
    for i, li in enumerate(labels):
        hi, ei = hist_by_label[li]
        for j, lj in enumerate(labels):
            hj, ej = hist_by_label[lj]
            mat[i, j] = kl_between_histograms(hi, hj, ei, ej)
    return pd.DataFrame(mat, index=labels, columns=labels)


def kl_matrix_pair_types(
    labels: list[str],
    pair_hists_by_label: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    pair_types: list[str],
) -> dict[str, pd.DataFrame]:
    """KL matrices per element-pair distance distribution."""
    out: dict[str, pd.DataFrame] = {}
    for pair in pair_types:
        hist_by_label = {
            label: pair_hists_by_label[label][pair]
            for label in labels
            if pair in pair_hists_by_label[label]
        }
        if len(hist_by_label) < 2:
            continue
        pair_labels = list(hist_by_label.keys())
        out[pair] = kl_matrix(pair_labels, hist_by_label)
    return out


def collect_pair_types(pair_hists_by_label: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]) -> list[str]:
    types: set[str] = set()
    for per_label in pair_hists_by_label.values():
        types.update(per_label.keys())
    return sorted(types)


def global_hist_range(
    labels: list[str],
    hist_by_label: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    vmins: list[float] = []
    vmaxs: list[float] = []
    for label in labels:
        _, edges = hist_by_label[label]
        vmins.append(float(edges[0]))
        vmaxs.append(float(edges[-1]))
    return min(vmins), max(vmaxs)


def histograms_with_shared_bins(
    labels: list[str],
    datasets: dict[str, dict[str, Any]],
    *,
    kind: str,
    bins: int = 128,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build per-label histograms using shared bin edges across all subsets."""
    raw_values: dict[str, np.ndarray] = {}
    for label in labels:
        ds = datasets[label]
        if kind == "energy":
            raw_values[label] = np.asarray(ds["E"], dtype=float).ravel()
        elif kind == "forces":
            raw_values[label] = np.abs(np.asarray(ds["F"], dtype=float)).ravel()
        else:
            raise ValueError(f"Unknown kind {kind!r}")

    vmin = min(float(v.min()) for v in raw_values.values())
    vmax = max(float(v.max()) for v in raw_values.values())
    if kind == "forces":
        vmin = 0.0
    pad = max(1e-6, 0.05 * (vmax - vmin))
    vmin -= pad
    vmax += pad

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, values in raw_values.items():
        out[label] = scalar_histogram(values, bins=bins, vmin=vmin, vmax=vmax)
    return out


def mean_force_per_structure(dataset: dict[str, Any]) -> np.ndarray:
    f = np.asarray(dataset["F"], dtype=float)
    return np.abs(f).reshape(f.shape[0], -1).mean(axis=1)


def gzip_metric_histograms(
    info_df: pd.DataFrame,
    *,
    bins: int = 64,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for col in GZIP_METRIC_COLUMNS:
        if col not in info_df.columns:
            continue
        values = np.asarray(info_df[col], dtype=float)
        pad = max(1e-6, 0.05 * (float(values.max()) - float(values.min())))
        out[col] = scalar_histogram(
            values,
            bins=bins,
            vmin=float(values.min()) - pad,
            vmax=float(values.max()) + pad,
        )
    return out
