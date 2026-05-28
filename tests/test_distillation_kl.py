"""Tests for force-histogram KL helpers."""

from __future__ import annotations

import numpy as np

from src.config import EV_PER_KCAL
from src.histogram_metrics import (
    PLOT_STATS_FORCE_KEYS,
    extract_array_with_fallback,
    kl_divergence,
    plot_stats_arrays_to_dataset_units,
)


def test_extract_uses_predfs_from_plot_stats() -> None:
    f_ref = np.arange(12, dtype=float).reshape(2, 2, 3)
    stats = {"predFs": np.arange(12, dtype=float) + 0.5}
    out, source = extract_array_with_fallback(
        stats,
        preferred_keys=PLOT_STATS_FORCE_KEYS,
        fallback=f_ref,
        expected_shape=f_ref.shape,
    )
    assert source == "predFs"
    assert np.allclose(out, stats["predFs"].reshape(f_ref.shape))


def test_kl_zero_only_for_identical_distributions() -> None:
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.6, 0.3, 0.1])
    assert kl_divergence(p, p) == 0.0
    assert kl_divergence(p, q) > 0.0


def test_plot_stats_kcal_to_ev_conversion() -> None:
    e_kcal = np.array([23.06])
    f_kcal = np.array([100.0])
    e_ev, f_ev = plot_stats_arrays_to_dataset_units(
        e_kcal,
        f_kcal,
        e_source="predEs",
        f_source="predFs",
        convert_to_ev=True,
    )
    assert np.isclose(e_ev[0], 1.0, rtol=0.02)
    assert np.isclose(f_ev[0], 100.0 * EV_PER_KCAL, rtol=0.02)


def test_kl_sparse_histograms_not_masked_to_zero() -> None:
    p = np.zeros(128)
    q = np.zeros(128)
    p[10] = 0.5
    p[20] = 0.5
    q[10] = 0.4
    q[30] = 0.6
    assert kl_divergence(p, q) > 0.0


if __name__ == "__main__":
    test_extract_uses_predfs_from_plot_stats()
    test_plot_stats_kcal_to_ev_conversion()
    test_kl_zero_only_for_identical_distributions()
    test_kl_sparse_histograms_not_masked_to_zero()
    print("ok")
