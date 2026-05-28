"""Histogram and KL helpers for distillation metrics (no heavy ML imports)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import EV_PER_KCAL

# Keys returned by mmml.physnetjax plot_stats (physnetjax/analysis/analysis.py).
PLOT_STATS_ENERGY_KEYS = ("predEs", "E_pred", "E_preds", "E_hat", "E_model", "pred_E")
PLOT_STATS_FORCE_KEYS = ("predFs", "F_pred", "F_preds", "F_hat", "F_model", "pred_F")


def extract_array_with_fallback(
    stats: dict[str, Any],
    preferred_keys: tuple[str, ...],
    fallback: np.ndarray,
    expected_shape: tuple[int, ...],
) -> tuple[np.ndarray, str]:
    for key in preferred_keys:
        if key in stats:
            arr = np.asarray(stats[key])
            if arr.size == int(np.prod(expected_shape)):
                return arr.reshape(expected_shape), key
            if arr.shape == expected_shape:
                return arr, key
    return np.asarray(fallback).reshape(expected_shape), "fallback_ground_truth"


def compute_normalized_histogram(
    values: np.ndarray,
    *,
    bins: int,
    vmin: float,
    vmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, range=(vmin, vmax))
    hist = hist.astype(float)
    denom = float(hist.sum())
    if denom <= 0.0:
        return np.full_like(hist, 1.0 / len(hist), dtype=float), edges
    return hist / denom, edges


def plot_stats_arrays_to_dataset_units(
    e_pred: np.ndarray,
    f_pred: np.ndarray,
    *,
    e_source: str,
    f_source: str,
    convert_to_ev: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """plot_stats returns kcal/mol; RMD17 loaders use eV when convert_to_ev is set."""
    e_out = np.asarray(e_pred, dtype=float)
    f_out = np.asarray(f_pred, dtype=float)
    if convert_to_ev:
        if e_source != "fallback_ground_truth":
            e_out = e_out * EV_PER_KCAL
        if f_source != "fallback_ground_truth":
            f_out = f_out * EV_PER_KCAL
    return e_out, f_out


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(P || Q) for discrete distributions P, Q with additive smoothing."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))
