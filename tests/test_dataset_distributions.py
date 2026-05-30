"""Tests for dataset distribution / KL helpers (no mmml)."""

from __future__ import annotations

import numpy as np

from src.dataset_distributions import (
    align_histograms,
    element_pair_key,
    kl_between_histograms,
    kl_matrix,
    pair_distance_histograms,
    subset_label,
)
from src.histogram_metrics import kl_divergence


def test_element_pair_key_sorted() -> None:
    assert element_pair_key(6, 1) == element_pair_key(1, 6) == "H-C"
    assert element_pair_key(8, 6) == "C-O"


def test_subset_label_format() -> None:
    assert subset_label(3, "train") == "split03_train"


def test_pair_distance_histogram_single_frame() -> None:
    # H at origin, C at 1.0 Å along x
    r = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=float)
    z = np.array([[1, 6]])
    hists = pair_distance_histograms(r, z, r_max=12.0, bins=32)
    assert "H-C" in hists
    prob, edges = hists["H-C"]
    assert prob.shape == (32,)
    assert np.isclose(prob.sum(), 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_bin = int(np.argmax(prob))
    assert 0.9 < centers[peak_bin] < 1.1


def test_kl_matrix_diagonal_near_zero() -> None:
    values = np.linspace(0.0, 1.0, 50)
    h, e = np.histogram(values, bins=32, range=(0.0, 1.0), density=True)
    h = h.astype(float)
    h /= h.sum()
    labels = ["a", "b"]
    hist_by_label = {"a": (h, e), "b": (h, e)}
    mat = kl_matrix(labels, hist_by_label)
    assert mat.shape == (2, 2)
    assert mat.loc["a", "a"] < 1e-6
    assert mat.loc["b", "b"] < 1e-6


def test_kl_divergence_identical() -> None:
    p = np.array([0.5, 0.3, 0.2])
    assert kl_divergence(p, p) == 0.0


def test_align_histograms_same_support() -> None:
    h1 = np.array([0.25, 0.25, 0.25, 0.25])
    e1 = np.linspace(0.0, 1.0, 5)
    p, q = align_histograms(h1, h1, e1, e1)
    assert np.allclose(p, q)
    assert np.isclose(kl_between_histograms(h1, h1, e1, e1), 0.0, atol=1e-9)


if __name__ == "__main__":
    test_element_pair_key_sorted()
    test_subset_label_format()
    test_pair_distance_histogram_single_frame()
    test_kl_matrix_diagonal_near_zero()
    test_kl_divergence_identical()
    test_align_histograms_same_support()
    print("ok")
