"""Train a head on cached THOR features; score it on the val chips.

Targets are rendered here from the chip points: a Gaussian per labelled berg on the token grid
(peaks exactly 1), with tokens on land / nodata and around unlabelled CFAR detections or
masked labels ignored. Early stopping on val loss; val precision / recall per epoch from
3x3 peaks matched one-to-one to the labels within `match_m`.

Two stages, both from `thor.train` in the config:
- folds (`folds`): train on one year, early-stop and score on the other. These choose the
  head and how long to train;
- final (`final`): the 5d model from both years, with no val set left. Either a refit for a
  fixed number of epochs, or the mean of the early-stopped fold models (an ensemble).
The April 2019 test holdout is used by neither.

Val precision is a lower bound: an unlabelled berg the CFAR also missed counts against it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from iceberg_sar.config import Config
from iceberg_sar.matching import match_points
from iceberg_sar.thor.chips import load_index
from iceberg_sar.thor.encoder import ThorParams, pixel_m_from_name
from iceberg_sar.thor.features import FeatureParams, cache_meta, feature_path
from iceberg_sar.thor.heads import Ensemble, HeadParams, build_head, extract_peaks, focal_loss
from iceberg_sar.thor.targets import TargetParams, render_heatmap, token_weights

THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)


@dataclass(frozen=True)
class TrainParams:
    focal_alpha: float
    focal_beta: float
    ignore_tokens: int
    min_valid: float
    epochs: int
    patience: int
    batch: int
    lr: float
    weight_decay: float
    match_m: float
    seed: int
    val_dates: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = ()  # empty: the chips' own split

    @classmethod
    def from_config(cls, cfg: Config, fold: str | None = None) -> TrainParams:
        """`fold` picks `val_dates` from `thor.train.folds`; None = the chips' own split."""
        t = cfg.section("thor")["train"]
        folds = t.get("folds") or {}
        if fold is not None and fold not in folds:
            raise KeyError(f"Fold {fold!r} not in thor.train.folds ({sorted(folds)})")
        return cls(
            focal_alpha=float(t["focal_alpha"]), focal_beta=float(t["focal_beta"]),
            ignore_tokens=int(t["ignore_tokens"]), min_valid=float(t["min_valid"]),
            epochs=int(t["epochs"]), patience=int(t["patience"]), batch=int(t["batch"]),
            lr=float(t["lr"]), weight_decay=float(t["weight_decay"]),
            match_m=float(t["match_m"]), seed=int(t["seed"]),
            val_dates=tuple((pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1))
                            for a, b in (folds[fold] if fold is not None else [])),
        )


def split_index(index: pd.DataFrame, p: TrainParams) -> dict[str, pd.DataFrame]:
    """train / val chips for the head, and the chips' own val split as `pack_ice` if unused.

    With `val_dates`, training-split chips in those dates become val; the chips' own val
    split (Feb-Mar 2019, bergs mostly inside pack ice) is then kept out of training and
    scored separately. The 2019 April holdout ("test") is never touched.
    """
    if not p.val_dates:
        return {"train": index[index["split"] == "train"], "val": index[index["split"] == "val"]}
    t = pd.to_datetime(index["time"])
    in_val = np.zeros(len(index), bool)
    for a, b in p.val_dates:
        in_val |= ((t >= a) & (t < b)).to_numpy()
    own = index["split"] == "train"
    return {"train": index[own & ~in_val], "val": index[own & in_val],
            "pack_ice": index[index["split"] == "val"]}


@dataclass
class ChipSet:
    """Features, targets and per-chip points for one split, in memory."""

    index: pd.DataFrame
    x: np.ndarray            # (N, C, h, w) float16
    target: np.ndarray       # (N, h, w) float32 Gaussian heatmaps
    weight: np.ndarray       # (N, h, w) float32 0/1
    berg_rc: list[np.ndarray]  # crop pixel coords of the targets, per chip


def chip_targets(
    chip: dict[str, np.ndarray], tp: ThorParams, sigma_px: float, p: TrainParams
) -> tuple[np.ndarray, np.ndarray]:
    """(target heatmap, loss weight) on the token grid for one chip .npz."""
    g = tp.grid_px()
    berg = chip["berg_rc"].reshape(-1, 2)
    ignore = np.concatenate([chip["unlabelled_rc"].reshape(-1, 2),
                             chip["masked_rc"].reshape(-1, 2)])
    target = render_heatmap(berg, (g, g), tp.patch_px, sigma_px)
    weight = token_weights(chip["valid"], ignore, berg, tp.patch_px, p.min_valid,
                           p.ignore_tokens)
    return target, weight


def load_split(
    index: pd.DataFrame, chips_dir: Path, features_dir: Path, cfg: Config, p: TrainParams
) -> ChipSet:
    tp = ThorParams.from_config(cfg)
    sigma_m = TargetParams.from_config(cfg).sigma_m
    xs, targets, weights, bergs = [], [], [], []
    for _, r in index.iterrows():
        chip = np.load(chips_dir / r["path"])
        t, w = chip_targets(chip, tp, sigma_m / pixel_m_from_name(r["scene"]), p)
        xs.append(np.load(feature_path(features_dir, r)))
        targets.append(t)
        weights.append(w)
        bergs.append(chip["berg_rc"].reshape(-1, 2))
    return ChipSet(index.reset_index(drop=True), np.stack(xs), np.stack(targets),
                   np.stack(weights), bergs)


def channel_stats(x: np.ndarray, weight: np.ndarray, n_max: int = 200,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean / std over the weighted tokens of up to `n_max` chips."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(x), size=min(n_max, len(x)), replace=False)
    c = x.shape[1]
    s, s2, n = np.zeros(c), np.zeros(c), 0
    for i in pick:  # running sums: all tokens at once would be GBs in float64
        t = x[i].reshape(c, -1)[:, weight[i].reshape(-1) > 0].astype(np.float64)
        s += t.sum(axis=1)
        s2 += (t**2).sum(axis=1)
        n += t.shape[1]
    mean = s / max(n, 1)
    return mean, np.sqrt(np.maximum(s2 / max(n, 1) - mean**2, 0.0))


def predict(model, x: np.ndarray, device: str, batch: int = 32) -> np.ndarray:
    """(N, C, h, w) features -> (N, h, w) berg probability."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).to(device)
            out.append(torch.sigmoid(model(xb))[:, 0].float().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, *x.shape[2:]), np.float32)


def chip_peaks_px(prob: np.ndarray, weight: np.ndarray, patch_px: int,
                  score_min: float) -> tuple[np.ndarray, np.ndarray]:
    """Peaks on scored tokens -> (crop pixel (row, col) at token centres, scores)."""
    ij, s = extract_peaks(prob, score_min, valid=weight > 0)
    return (ij + 0.5) * patch_px, s


def pr_table(
    probs: np.ndarray, data: ChipSet, patch_px: int, pixel_m: float, match_m: float,
    thresholds: np.ndarray = THRESHOLDS,
) -> pd.DataFrame:
    """Matched precision / recall over all chips at each threshold.

    Peaks on ignored tokens are neither true nor false positives. Overlapping chips of a scene
    are scored separately (a berg in two chips counts twice, as in the 5b summary).
    """
    peaks = [chip_peaks_px(probs[i], data.weight[i], patch_px, float(thresholds.min()))
             for i in range(len(probs))]
    n_truth = sum(len(b) for b in data.berg_rc)
    rows = []
    for t in thresholds:
        tp = n_det = 0
        for (rc, s), berg in zip(peaks, data.berg_rc, strict=True):
            det = rc[s >= t] * pixel_m
            m = match_points(det, berg * pixel_m, match_m)
            tp += m.true_positives
            n_det += len(det)
        prec = tp / n_det if n_det else float("nan")
        rec = tp / n_truth if n_truth else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if n_det and tp else 0.0
        rows.append({"threshold": float(t), "detections": n_det, "true_positives": tp,
                     "truth": n_truth, "precision": prec, "recall": rec, "f1": f1})
    return pd.DataFrame(rows)


def train_head(
    kind: str, cfg: Config, out_dir: Path, device: str = "cuda", log=print,
    fold: str | None = None, refit_epochs: int | None = None,
) -> dict[str, Path]:
    """Train one head; writes weights, curves and a summary to `out_dir`.

    With `fold`, early-stopped on that fold's val chips. With `refit_epochs`, trained on all
    training-split chips for exactly that many epochs (cosine schedule over them, so the
    learning rate is ~0 at the end) and no val set; the pack-ice set is still scored.
    """
    import torch

    tp, fp = ThorParams.from_config(cfg), FeatureParams.from_config(cfg)
    hp, p = HeadParams.from_config(cfg), TrainParams.from_config(cfg, fold)
    if refit_epochs is not None:
        p = replace(p, epochs=int(refit_epochs), val_dates=())
    thor_dir = cfg.root / cfg.section("thor")["training"]["dir"]
    chips_dir, features_dir = thor_dir / "chips", thor_dir / "features"
    meta_path = features_dir / "meta.json"
    if not meta_path.is_file() or json.loads(meta_path.read_text(encoding="utf-8")) != \
            cache_meta(tp, fp):
        raise FileNotFoundError("No feature cache for the current config; run `thor-features`")

    index = load_index(chips_dir)
    t0 = time.time()
    parts = split_index(index, p)
    if refit_epochs is not None:  # everything trainable; the chips' own val is the pack-ice set
        parts = {"train": parts["train"], "pack_ice": parts["val"]}
    train = load_split(parts["train"], chips_dir, features_dir, cfg, p)
    val = load_split(parts["val"], chips_dir, features_dir, cfg, p) if "val" in parts else None
    pixel_m = pixel_m_from_name(str(index["scene"].iloc[0]))
    log(f"loaded {len(train.x)} train / {len(val.x) if val else 0} val chips "
        f"in {time.time() - t0:.0f} s")

    torch.manual_seed(p.seed)
    rng = np.random.default_rng(p.seed)
    channels = train.x.shape[1]
    model = build_head(kind, channels, hp)
    model.norm.set_stats(*channel_stats(train.x, train.weight, seed=p.seed))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p.epochs)

    def to_dev(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a).to(device)

    def loss_on(data: ChipSet, idx: np.ndarray) -> torch.Tensor:
        logits = model(to_dev(data.x[idx]))[:, 0]
        return focal_loss(logits, to_dev(data.target[idx]), to_dev(data.weight[idx]),
                          p.focal_alpha, p.focal_beta)

    def val_loss() -> float:
        assert val is not None
        model.eval()
        total, peaks = 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(val.x), p.batch):
                idx = np.arange(i, min(i + p.batch, len(val.x)))
                n_pos = float((val.target[idx] >= 1).sum() * 1.0)
                total += float(loss_on(val, idx)) * max(n_pos, 1.0)
                peaks += max(n_pos, 1.0)
        return total / peaks  # same normalization as one big batch

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("val_chips.csv", "val_probs.npy", "val_pr.csv", "val_overlays.png"):
        (out_dir / stale).unlink(missing_ok=True)
    if val is not None:
        val.index[["chip_id", "scene", "path"]].to_csv(out_dir / "val_chips.csv", index=False)
    weights_path = out_dir / "head.pt"
    history, best, since_best = [], float("inf"), 0

    def save(epoch: int) -> None:
        torch.save({"kind": kind, "channels": channels, "head": asdict(hp),
                    "features": cache_meta(tp, fp), "epoch": epoch, "fold": fold,
                    "state_dict": model.state_dict()}, weights_path)

    for epoch in range(1, p.epochs + 1):
        model.train()
        order = rng.permutation(len(train.x))
        tr_loss, n = 0.0, 0
        for i in range(0, len(order), p.batch):
            idx = np.sort(order[i:i + p.batch])
            loss = loss_on(train, idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss, n = tr_loss + float(loss), n + 1
        sched.step()
        if val is None:
            history.append({"epoch": epoch, "train_loss": tr_loss / n})
            if epoch == 1 or epoch % 10 == 0:
                log(f"epoch {epoch:3d}  train {tr_loss / n:.3f}")
            continue
        vl = val_loss()
        pr = pr_table(predict(model, val.x, device), val, tp.patch_px, pixel_m, p.match_m)
        bestf = pr.loc[pr["f1"].idxmax()]
        at5 = pr.loc[pr["threshold"] == 0.5].iloc[0]
        history.append({"epoch": epoch, "train_loss": tr_loss / n, "val_loss": vl,
                        "val_best_f1": bestf["f1"], "val_best_threshold": bestf["threshold"],
                        "val_precision_05": at5["precision"], "val_recall_05": at5["recall"]})
        improved = vl < best
        if improved:
            best, since_best = vl, 0
            save(epoch)
        else:
            since_best += 1
        if epoch == 1 or improved or epoch % 10 == 0:
            log(f"epoch {epoch:3d}  train {tr_loss / n:.3f}  val {vl:.3f}  "
                f"F1 {bestf['f1']:.2f}@{bestf['threshold']:.2f}  "
                f"P/R@0.5 {at5['precision']:.2f}/{at5['recall']:.2f}{'  *' if improved else ''}")
        if since_best >= p.patience:
            log(f"early stop at epoch {epoch} (best val loss {best:.3f})")
            break

    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / "history.csv", index=False)
    if val is None:
        save(len(hist))
        model = load_head(weights_path, device)
        summary = {"head": kind, "mode": "refit", "epochs_run": len(hist),
                   "final_train_loss": round(float(hist["train_loss"].iloc[-1]), 4),
                   "train_chips": len(train.x),
                   "train_scenes": sorted(train.index["scene"].str[7:15].unique().tolist()),
                   "parameters": sum(t.numel() for t in model.parameters())}
        summary["pack_ice"] = score_pack_ice(model, parts["pack_ice"], out_dir, cfg, p, device)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"weights": weights_path, "history": out_dir / "history.csv",
                "curves": plot_curves(hist, None, out_dir / "curves.png", kind),
                "summary": out_dir / "summary.json"}
    model = load_head(weights_path, device)
    probs = predict(model, val.x, device)
    pr = pr_table(probs, val, tp.patch_px, pixel_m, p.match_m)
    pr.to_csv(out_dir / "val_pr.csv", index=False)
    best_row = pr.loc[pr["f1"].idxmax()]
    summary = {"head": kind, "mode": "fold", "fold": fold,
               "best_epoch": int(hist.loc[hist["val_loss"].idxmin(), "epoch"]),
               "epochs_run": len(hist), "val_loss": round(float(hist["val_loss"].min()), 4),
               "val_best_f1": round(float(best_row["f1"]), 3),
               "val_best_threshold": float(best_row["threshold"]),
               "val_precision": round(float(best_row["precision"]), 3),
               "val_recall": round(float(best_row["recall"]), 3),
               "val_truth": int(best_row["truth"]), "train_chips": len(train.x),
               "val_chips": len(val.x), "parameters": sum(t.numel() for t in model.parameters())}
    summary["val_scenes"] = sorted(val.index["scene"].str[7:15].unique().tolist())
    if "pack_ice" in parts:  # the chips' own val split, scored at the val-chosen threshold
        summary["pack_ice"] = score_pack_ice(model, parts["pack_ice"], out_dir, cfg, p, device,
                                             float(best_row["threshold"]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(out_dir / "val_probs.npy", probs.astype(np.float16))
    return {"weights": weights_path, "history": out_dir / "history.csv",
            "curves": plot_curves(hist, pr, out_dir / "curves.png", kind),
            "summary": out_dir / "summary.json"}


def score_pack_ice(
    model, index: pd.DataFrame, out_dir: Path, cfg: Config, p: TrainParams, device: str,
    threshold: float | None = None,
) -> dict[str, float | int | str]:
    """Matched P / R on the pack-ice chips at `threshold` (None: the best-F1 one on this set,
    which is optimistic; the result says which). Writes pack_ice_pr.csv."""
    tp = ThorParams.from_config(cfg)
    thor_dir = cfg.root / cfg.section("thor")["training"]["dir"]
    ice = load_split(index, thor_dir / "chips", thor_dir / "features", cfg, p)
    pixel_m = pixel_m_from_name(str(index["scene"].iloc[0]))
    pr = pr_table(predict(model, ice.x, device), ice, tp.patch_px, pixel_m, p.match_m)
    pr.to_csv(out_dir / "pack_ice_pr.csv", index=False)
    r = (pr.loc[pr["f1"].idxmax()] if threshold is None
         else pr.loc[pr["threshold"] == threshold].iloc[0])
    return {"chips": len(ice.x), "truth": int(r["truth"]), "threshold": float(r["threshold"]),
            "threshold_from": "best F1 on this set" if threshold is None else "val",
            "precision": round(float(r["precision"]), 3),
            "recall": round(float(r["recall"]), 3), "f1": round(float(r["f1"]), 3)}


def make_ensemble(members: list[Path], out: Path) -> Path:
    """One checkpoint holding several trained heads; `load_head` averages their probabilities."""
    import torch

    cks = [torch.load(m, map_location="cpu", weights_only=False) for m in members]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "ensemble", "members": cks,
                "member_paths": [str(m) for m in members]}, out)
    return out


def load_head(path: Path, device: str = "cpu"):
    """Rebuild a trained head (or an ensemble of heads) from its checkpoint."""
    import torch

    ck = torch.load(path, map_location=device, weights_only=False)
    return _from_checkpoint(ck).to(device).eval()


def _from_checkpoint(ck: dict):
    if ck["kind"] == "ensemble":
        return Ensemble([_from_checkpoint(m) for m in ck["members"]])
    model = build_head(ck["kind"], ck["channels"], HeadParams(**ck["head"]))
    model.load_state_dict(ck["state_dict"])
    return model


def train_final(
    kind: str, cfg: Config, head_dir: Path, device: str = "cuda", log=print,
) -> dict[str, Path]:
    """The 5d model in `head_dir/final`, as `thor.train.final.<kind>` says: a fixed-epoch refit
    on both years (`epochs`) or the mean of fold models already trained (`ensemble`)."""
    spec = cfg.section("thor")["train"]["final"][kind]
    out = head_dir / "final"
    if "epochs" in spec:
        return train_head(kind, cfg, out, device, log, refit_epochs=int(spec["epochs"]))
    members = [head_dir / f"fold_{f}" / "head.pt" for f in spec["ensemble"]]
    missing = [m for m in members if not m.is_file()]
    if missing:
        raise FileNotFoundError(f"Train the folds first: {', '.join(map(str, missing))}")
    out.mkdir(parents=True, exist_ok=True)
    weights = make_ensemble(members, out / "head.pt")
    p = TrainParams.from_config(cfg)
    index = load_index(cfg.root / cfg.section("thor")["training"]["dir"] / "chips")
    model = load_head(weights, device)
    summary = {"head": kind, "mode": "ensemble", "members": [str(m) for m in members],
               "parameters": sum(t.numel() for t in model.parameters()),
               "pack_ice": score_pack_ice(model, index[index["split"] == "val"], out, cfg, p,
                                          device)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"weights": weights, "summary": out / "summary.json"}


def plot_curves(hist: pd.DataFrame, pr: pd.DataFrame | None, out: Path, kind: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if pr is None:  # refit: no val set, train loss only
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.plot(hist["epoch"], hist["train_loss"], label="train")
        ax.set(title=f"THOR {kind} head, refit on all train chips", xlabel="epoch",
               ylabel="focal loss", yscale="log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return out

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
    ax[0].plot(hist["epoch"], hist["val_loss"], label="val")
    best = hist.loc[hist["val_loss"].idxmin()]
    ax[0].axvline(best["epoch"], color="grey", ls=":", lw=1)
    ax[0].set(title="Focal loss", xlabel="epoch", yscale="log")
    ax[0].legend()
    ax[1].plot(hist["epoch"], hist["val_best_f1"], label="best F1")
    ax[1].plot(hist["epoch"], hist["val_precision_05"], label="precision @0.5")
    ax[1].plot(hist["epoch"], hist["val_recall_05"], label="recall @0.5")
    ax[1].axvline(best["epoch"], color="grey", ls=":", lw=1)
    ax[1].set(title="Val chips (matched points)", xlabel="epoch", ylim=(0, 1))
    ax[1].legend()
    ax[2].plot(pr["recall"], pr["precision"], marker=".")
    for _, r in pr.iloc[::4].iterrows():
        ax[2].annotate(f"{r['threshold']:.2f}", (r["recall"], r["precision"]), fontsize=7)
    ax[2].set(title=f"Val precision-recall (epoch {int(best['epoch'])})", xlabel="recall",
              ylabel="precision (lower bound)", xlim=(0, 1), ylim=(0, 1))
    fig.suptitle(f"THOR {kind} head")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def overlay_sheet(
    head_dir: Path, cfg: Config, png: Path, n: int = 20, seed: int = 0,
    threshold: float | None = None,
) -> Path:
    """Val chips (HH dB) with the predicted heatmap, peaks and labels overlaid.

    Uses `val_probs.npy` from training. Red circles = labels, lime x = peaks at or above
    `threshold` (default: the best-F1 threshold), cyan squares = unlabelled CFAR detections,
    blue = pack ice (no buffer). Greyed tokens are ignored by the loss and the scoring.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from iceberg_sar.thor.chips import overview_selection

    tp, p = ThorParams.from_config(cfg), TrainParams.from_config(cfg)
    summary = json.loads((head_dir / "summary.json").read_text(encoding="utf-8"))
    thr = summary["val_best_threshold"] if threshold is None else threshold
    thor_dir = cfg.root / cfg.section("thor")["training"]["dir"]
    chips_dir = thor_dir / "chips"
    index = load_index(chips_dir)
    val_ids = pd.read_csv(head_dir / "val_chips.csv")["chip_id"]
    val_index = index.set_index("chip_id").loc[val_ids].reset_index()
    probs = np.load(head_dir / "val_probs.npy").astype(np.float32)
    if len(probs) != len(val_index):
        raise ValueError("val_probs.npy does not match the val chips; retrain the head")
    sel = overview_selection(val_index, n, seed)
    sigma_m = TargetParams.from_config(cfg).sigma_m

    cols = 5
    rows_n = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2 * cols, 3.4 * rows_n),
                             constrained_layout=True, squeeze=False)
    hh_mean, hh_std = tp.norm["HH"]
    extent = (-0.5, tp.crop_px - 0.5, tp.crop_px - 0.5, -0.5)
    for ax, (i, r) in zip(axes.flat, sel.iterrows(), strict=False):
        z = np.load(chips_dir / r["path"])
        _, weight = chip_targets(z, tp, sigma_m / pixel_m_from_name(r["scene"]), p)
        hh = z["x"][tp.bands.index("HH")].astype(np.float32) * hh_std + hh_mean
        ax.imshow(np.where(z["valid"], hh, np.nan), cmap="gray", vmin=-30, vmax=-5)
        ax.imshow(np.ma.masked_less(probs[i], 0.05), cmap="autumn_r", alpha=0.6, vmin=0,
                  vmax=1, extent=extent, interpolation="nearest")
        ax.imshow(np.ma.masked_where(weight > 0, np.ones_like(weight)), cmap="gray",
                  alpha=0.35, vmin=0, vmax=1, extent=extent, interpolation="nearest")
        rc, s = chip_peaks_px(probs[i], weight, tp.patch_px, thr)
        b, u = z["berg_rc"].reshape(-1, 2), z["unlabelled_rc"].reshape(-1, 2)
        ax.scatter(b[:, 1], b[:, 0], s=60, facecolors="none", edgecolors="red", linewidths=0.9)
        ax.scatter(u[:, 1], u[:, 0], s=50, marker="s", facecolors="none", edgecolors="cyan",
                   linewidths=0.8)
        ax.scatter(rc[:, 1], rc[:, 0], s=40, marker="x", c="lime", linewidths=1.0)
        if z["ice"].any():
            ax.contour(z["ice"].astype(np.float32), levels=[0.5], colors="deepskyblue",
                       linewidths=0.6)
        ax.set_title(f"{r['kind']}  {str(r['time'])[:10]}  max p {probs[i].max():.2f}\n"
                     f"{len(b)} labels, {len(rc)} peaks, {len(u)} unlabelled", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"THOR {summary['head']} head on val chips: heatmap (orange), peaks >= "
                 f"{thr:.2f} (lime x), labels (red), unlabelled CFAR (cyan), pack ice (blue), "
                 f"ignored tokens (grey)", fontsize=10)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=100)
    plt.close(fig)
    return png
