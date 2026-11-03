"""Revision C12: the noise field read off the two SNAP stacks."""

import numpy as np

from iceberg_sar.benchmark.tnr_ablation import noise_field


def test_noise_field_exact_where_unclipped_and_filled_where_clipped() -> None:
    n_true = np.full((40, 40), 1e-3, np.float32)
    signal = np.full((40, 40), 5e-3, np.float32)
    signal[10:14, 10:14] = 5e-4           # below the noise floor: SNAP clips these
    no_tnr = signal + n_true
    tnr = np.where(signal > n_true, signal, 3.7e-12).astype(np.float32)
    valid = np.ones((40, 40), bool)
    n, filled = noise_field(no_tnr, tnr, valid, 1e-11, 3)
    assert np.allclose(n, 1e-3, rtol=1e-4)
    assert filled == 16 / 1600
