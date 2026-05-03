"""Standalone backtest harness: evaluate trained models on one held-out season."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml.features import POST_QUALI_FEATURES, PRE_QUALI_FEATURES
from ml.monte_carlo import run_simulation

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_NORMAL_Q90_Q10_Z = 2.5631


def _evaluate(bundle: dict, df: pd.DataFrame, features: list[str]) -> dict:
    X = df[features]
    q10 = bundle["models"]["q10"].predict(X)
    q50 = bundle["models"]["q50"].predict(X)
    q90 = bundle["models"]["q90"].predict(X)
    mu = np.asarray(q50)
    sigma = np.clip((q90 - q10) / _NORMAL_Q90_Q10_Z, 0.5, None)

    rhos, win_ok, podium_hits, briers, maes = [], 0, 0.0, [], []
    n = 0
    df = df.reset_index(drop=True)
    for (season, rnd), grp in df.groupby(["season", "round"]):
        if len(grp) < 5:
            continue
        n += 1
        idx = grp.index.to_numpy()
        race_mu, race_sigma = mu[idx], sigma[idx]
        actual = grp["finish_position"].to_numpy()

        rho, _ = spearmanr(race_mu, actual)
        if not np.isnan(rho):
            rhos.append(rho)
        maes.append(float(np.mean(np.abs(race_mu - actual))))

        prob = run_simulation(race_mu, race_sigma, n_sims=5_000)
        win_ok += int(int(np.argmax(prob[:, 0])) == int(np.argmin(actual)))
        pred_pod = set(np.argsort(-prob[:, :3].sum(axis=1))[:3].tolist())
        actual_pod = set(np.argsort(actual)[:3].tolist())
        podium_hits += len(pred_pod & actual_pod) / 3.0

        win_prob = prob[:, 0]
        actual_winner_onehot = (actual == actual.min()).astype(float)
        briers.append(float(np.mean((win_prob - actual_winner_onehot) ** 2)))

    return {
        "n_races": n,
        "spearman_rho_mean": float(np.mean(rhos)) if rhos else float("nan"),
        "mae_mean": float(np.mean(maes)) if maes else float("nan"),
        "win_pick_accuracy": (win_ok / n) if n else float("nan"),
        "podium_hit_rate": (podium_hits / n) if n else float("nan"),
        "brier_winner": float(np.mean(briers)) if briers else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--training", type=Path, default=Path("ml/data/training.parquet"))
    ap.add_argument("--pre", type=Path, default=Path("ml/artifacts/pre_quali_finish.pkl"))
    ap.add_argument("--post", type=Path, default=Path("ml/artifacts/post_quali_finish.pkl"))
    args = ap.parse_args()

    df = pd.read_parquet(args.training)
    df = df[df["season"] == args.season].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"No rows for season {args.season} in {args.training}")

    if args.pre.exists():
        pre_bundle = joblib.load(args.pre)
        pre_df = df[df["mode"] == "pre_quali"].reset_index(drop=True)
        log.info("PRE-QUALI on %d rows: %s", len(pre_df), _evaluate(pre_bundle, pre_df, PRE_QUALI_FEATURES))
    else:
        log.warning("missing %s", args.pre)

    if args.post.exists():
        post_bundle = joblib.load(args.post)
        post_df = df[df["mode"] == "post_quali"].reset_index(drop=True)
        if not post_df.empty:
            log.info("POST-QUALI on %d rows: %s", len(post_df), _evaluate(post_bundle, post_df, POST_QUALI_FEATURES))
    else:
        log.warning("missing %s", args.post)


if __name__ == "__main__":
    main()
