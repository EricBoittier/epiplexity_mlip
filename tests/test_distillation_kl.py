"""Tests for force-histogram KL helpers."""

from __future__ import annotations

import numpy as np

from src.histogram_metrics import (
    PLOT_STATS_FORCE_KEYS,
    extract_array_with_fallback,
    kl_divergence,
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
    test_kl_zero_only_for_identical_distributions()
    test_kl_sparse_histograms_not_masked_to_zero()
    print("ok")
