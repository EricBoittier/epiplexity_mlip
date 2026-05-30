"""Tests for information window selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiment import make_information_split


def test_random_metric_uses_same_windows_different_order() -> None:
    info_df = pd.DataFrame(
        {
            "start": [0, 10, 20],
            "gzip_bytes_cart": [3.0, 1.0, 2.0],
            "indices": [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])],
        }
    )
    by_cart, _, _, _ = make_information_split(info_df, metric="gzip_bytes_cart", train_fraction=0.5, seed=42)
    by_random_a, _, _, _ = make_information_split(info_df, metric="random", train_fraction=0.5, seed=42)
    by_random_b, _, _, _ = make_information_split(info_df, metric="random", train_fraction=0.5, seed=42)
    by_random_other, _, _, _ = make_information_split(info_df, metric="random", train_fraction=0.5, seed=99)

    assert set(by_cart) != set(by_random_a) or np.array_equal(by_cart, by_random_a) is False
    assert np.array_equal(by_random_a, by_random_b)
    assert not np.array_equal(by_random_a, by_random_other)


if __name__ == "__main__":
    test_random_metric_uses_same_windows_different_order()
    print("ok")
