from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from iceberg_sar.thor.encoder import pixel_m_from_name  # noqa: E402
from iceberg_sar.thor.heads import (  # noqa: E402
    HeadParams,
    build_head,
    extract_peaks,
    focal_loss,
)
from iceberg_sar.thor.infer import kept_ranges, token_argmax, window_starts  # noqa: E402
from iceberg_sar.thor.targets import pool_tokens, render_heatmap, token_weights  # noqa: E402
from iceberg_sar.thor.train import ChipSet, pr_table  # noqa: E402


def test_pixel_m_from_name():
    assert pixel_m_from_name("S1A_IW_20190511T094108_DHP_RTC10_G_spuned_B52C") == 10.0
    with pytest.raises(ValueError):
        pixel_m_from_name("not_a_product")


def test_pool_tokens_and_weights_ignore_and_keep_targets():
    valid = np.ones((32, 32), bool)
    valid[:, :8] = False                   # first token column is land
    assert pool_tokens(valid, 8)[:, 0].max() == 0 and pool_tokens(valid, 8)[:, 1:].min() == 1
    ignore = np.array([[20.0, 20.0]])      # token (2, 2) and its 3x3 neighbourhood
    target = np.array([[12.0, 12.0]])      # token (1, 1): inside the ignored block, kept
    w = token_weights(valid, ignore, target, 8, 0.5, 1)
    assert w[:, 0].sum() == 0
    assert w[1, 1] == 1 and w[2, 2] == 0 and w[3, 3] == 0 and w[1, 2] == 0
    assert w[0, 3] == 1


def test_focal_loss_rewards_right_peak_and_ignores_weight_zero():
    target = torch.from_numpy(render_heatmap(np.array([[20.0, 20.0]]), (8, 8), 8, 6.0))[None]
    weight = torch.ones_like(target)
    good = torch.full_like(target, -8.0)
    good[0, 2, 2] = 8.0
    bad = torch.full_like(target, -8.0)
    bad[0, 6, 6] = 8.0
    assert focal_loss(good, target, weight) < 0.01 < focal_loss(bad, target, weight)
    weight[0, 6, 6] = 0                    # an ignored token can't cost anything
    weight[0, 2, 2] = 0
    assert focal_loss(bad, target, weight) < 0.01


def test_heads_shape_and_prior():
    p = HeadParams(hidden=16, dropout=0.0, prior=0.01)
    x = torch.randn(2, 12, 8, 8)
    for kind in ("linear", "centernet"):
        out = build_head(kind, 12, p)(x)
        assert out.shape == (2, 1, 8, 8)
    lin = build_head("linear", 12, p)
    torch.nn.init.zeros_(lin.out.weight)
    np.testing.assert_allclose(torch.sigmoid(lin(x)).detach().numpy(), 0.01, rtol=1e-5)
    with pytest.raises(ValueError):
        build_head("unet", 12, p)


def test_extract_peaks_nms_threshold_and_plateau():
    prob = np.zeros((6, 6))
    prob[1, 1], prob[1, 2] = 0.9, 0.5      # neighbour of a stronger peak: suppressed
    prob[4, 4] = prob[4, 5] = 0.7          # plateau: one peak
    prob[0, 5] = 0.05                      # below threshold
    ij, s = extract_peaks(prob, 0.1)
    assert ij.tolist() == [[1, 1], [4, 4]] and s.tolist() == [0.9, 0.7]
    valid = np.ones_like(prob, bool)
    valid[1, 1] = False
    ij, _ = extract_peaks(prob, 0.1, valid)
    assert [1, 2] in ij.tolist()           # an invalid peak can't suppress its neighbour


@pytest.mark.parametrize("n_px", [512, 900, 1000, 2047])
def test_windows_cover_each_token_once(n_px):
    crop, overlap, patch = 512, 128, 8
    starts = window_starts(n_px, crop, overlap, patch)
    assert starts[0] == 0 and all(s % patch == 0 for s in starts)
    assert n_px - patch < starts[-1] + crop <= n_px
    g = crop // patch
    ranges = kept_ranges([s // patch for s in starts], g)
    assert ranges[0][0] == 0 and ranges[-1][1] == starts[-1] // patch + g
    for (_, hi), (lo2, _) in zip(ranges, ranges[1:], strict=False):
        assert hi == lo2                   # contiguous, no double counting
    for s, (lo, hi) in zip(starts, ranges, strict=True):
        t0 = s // patch
        assert t0 <= lo < hi <= t0 + g     # each window only writes tokens it has
    if len(starts) > 2:                    # a full stride after the first: keeps its centre
        t0 = starts[1] // patch
        assert ranges[1][0] == t0 + overlap // patch // 2


def test_window_starts_rejects_bad_overlap():
    with pytest.raises(ValueError):
        window_starts(2000, 512, 100, 8)
    with pytest.raises(ValueError):
        window_starts(400, 512, 128, 8)


def test_token_argmax():
    img = np.zeros((16, 16), np.float32)
    img[3, 5] = 9
    img[12, 9] = 4
    img[0, 8:] = np.nan
    rows, cols = token_argmax(img, 8)
    assert (rows[0, 0], cols[0, 0]) == (3, 5)
    assert (rows[1, 1], cols[1, 1]) == (12, 9)


def test_pr_table_counts_matches_and_skips_ignored_tokens():
    g, patch = 8, 8
    berg = np.array([[20.0, 20.0]])
    prob = np.zeros((1, g, g), np.float32)
    prob[0, 2, 2] = 0.9                    # on the berg
    prob[0, 6, 6] = 0.6                    # false alarm
    prob[0, 6, 1] = 0.8                    # on an ignored token: neither TP nor FP
    weight = np.ones((1, g, g), np.float32)
    weight[0, 6, 1] = 0
    data = ChipSet(pd.DataFrame({"chip_id": ["a"]}), np.zeros((1, 1, g, g), np.float16),
                   np.zeros((1, g, g)), weight, [berg])
    pr = pr_table(prob, data, patch, 10.0, 100.0, np.array([0.5, 0.7]))
    assert pr["detections"].tolist() == [2, 1]
    assert pr["true_positives"].tolist() == [1, 1]
    assert pr["precision"].tolist() == [0.5, 1.0] and pr["recall"].tolist() == [1.0, 1.0]


def test_channel_stats_uses_weighted_tokens_only():
    from iceberg_sar.thor.train import channel_stats

    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, size=(5, 4, 8, 8)).astype(np.float16)
    weight = np.ones((5, 8, 8), np.float32)
    weight[:, 0] = 0
    x[:, :, 0] = 1000                      # ignored tokens must not move the stats
    mean, std = channel_stats(x, weight)
    ref = x[:, :, 1:].astype(np.float64).transpose(1, 0, 2, 3).reshape(4, -1)
    np.testing.assert_allclose(mean, ref.mean(axis=1), rtol=1e-6)
    np.testing.assert_allclose(std, ref.std(axis=1), rtol=1e-6)


def test_ensemble_averages_member_probabilities():
    from iceberg_sar.thor.heads import Ensemble

    p = HeadParams(hidden=8, dropout=0.0, prior=0.01)
    a, b = build_head("linear", 4, p), build_head("linear", 4, p)
    for m, bias in ((a, 2.0), (b, -2.0)):
        torch.nn.init.zeros_(m.out.weight)
        torch.nn.init.constant_(m.out.bias, bias)
    out = torch.sigmoid(Ensemble([a, b])(torch.randn(1, 4, 3, 3)))
    np.testing.assert_allclose(out.detach().numpy(), 0.5, atol=1e-5)  # mean(0.88, 0.12)


def test_ensemble_checkpoint_round_trip(tmp_path):
    from iceberg_sar.thor.train import load_head, make_ensemble

    p = HeadParams(hidden=8, dropout=0.0, prior=0.01)
    paths = []
    for i in range(2):
        m = build_head("centernet", 4, p)
        path = tmp_path / f"m{i}.pt"
        torch.save({"kind": "centernet", "channels": 4, "head": asdict(p),
                    "state_dict": m.state_dict()}, path)
        paths.append(path)
    ens = load_head(make_ensemble(paths, tmp_path / "ens.pt"))
    x = torch.randn(2, 4, 6, 6)
    members = [load_head(q) for q in paths]
    want = torch.stack([torch.sigmoid(m(x)) for m in members]).mean(0)
    np.testing.assert_allclose(torch.sigmoid(ens(x)).detach().numpy(), want.detach().numpy(),
                               rtol=1e-4, atol=1e-6)


def test_fold_selects_val_dates():
    from iceberg_sar.thor.train import TrainParams, split_index

    class Cfg:
        def section(self, name):
            return {"train": {
                "focal_alpha": 2, "focal_beta": 4, "ignore_tokens": 1, "min_valid": 0.5,
                "epochs": 10, "patience": 3, "batch": 4, "lr": 1e-3, "weight_decay": 0.0,
                "match_m": 200, "seed": 0,
                "folds": {"a": [["2020-01-01", "2020-12-31"]]}}}

    p = TrainParams.from_config(Cfg(), "a")
    index = pd.DataFrame({
        "time": ["2019-05-11", "2020-05-05", "2019-03-06", "2019-04-29"],
        "split": ["train", "train", "val", "test"]})
    parts = split_index(index, p)
    assert parts["train"]["time"].tolist() == ["2019-05-11"]
    assert parts["val"]["time"].tolist() == ["2020-05-05"]
    assert parts["pack_ice"]["time"].tolist() == ["2019-03-06"]   # test never appears
    with pytest.raises(KeyError):
        TrainParams.from_config(Cfg(), "z")
