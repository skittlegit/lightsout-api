"""Train the three quantile bundles.

Validation = held-out most-recent full season. Reports per-race Spearman ρ,
MAE, win-pick accuracy, podium hit rate, Brier score on win prob.

Each saved bundle is a dict::

    {
        "version": "preq-v1.<n>",
        "features": [...],
        "models":  {"q10": booster, "q50": booster, "q90": booster},
        "trained_at": "...",
        "metrics": {...},
    }
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from ml.features import POLE_FEATURES, POST_QUALI_FEATURES, PRE_QUALI_FEATURES
from ml.monte_carlo import run_simulation

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


_QUANTILES = (0.10, 0.50, 0.90)
_NORMAL_Q90_Q10_Z = 2.5631


def _fit_quantile(X: pd.DataFrame, y: np.ndarray, q: float) -> lgb.Booster:
    train = lgb.Dataset(X, label=y)
    params = {
        "objective": "quantile",
        "alpha": q,
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbose": -1,
    }
    return lgb.train(params, train, num_boost_round=600)


def _fit_bundle(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    features: list[str],
    target: str,
    version: str,
) -> tuple[dict, dict]:
    X_train = df_train[features]
    y_train = df_train[target].to_numpy(dtype=float)
    X_val = df_val[features]
    y_val = df_val[target].to_numpy(dtype=float)

    models: dict[str, lgb.Booster] = {}
    for q in _QUANTILES:
        log.info("Fitting q=%.2f on %d rows", q, len(df_train))
        models[f"q{int(q * 100):02d}"] = _fit_quantile(X_train, y_train, q)

    metrics = _evaluate_finish(models, X_val, y_val, df_val)
    bundle = {
        "version": version,
        "features": features,
        "models": models,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    return bundle, metrics


def _evaluate_finish(
    models: dict[str, lgb.Booster],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    df_val: pd.DataFrame,
) -> dict:
    q10 = models["q10"].predict(X_val)
    q50 = models["q50"].predict(X_val)
    q90 = models["q90"].predict(X_val)
    mu = q50
    sigma = np.clip((q90 - q10) / _NORMAL_Q90_Q10_Z, 0.5, None)

    mae = float(mean_absolute_error(y_val, q50))

    rhos, win_correct, podium_hits, briers = [], 0, 0, []
    n_races = 0
    for (season, rnd), grp in df_val.groupby(["season", "round"]):
        idx = grp.index
        if len(idx) < 5:
            continue
        n_races += 1
        race_mu = mu[idx - df_val.index[0]] if df_val.index.is_monotonic_increasing else mu[grp.index.argsort()]
        race_sigma = sigma[idx - df_val.index[0]] if df_val.index.is_monotonic_increasing else sigma[grp.index.argsort()]
        # Safer: positional lookup
        positional = np.searchsorted(df_val.index.to_numpy(), grp.index.to_numpy())
        race_mu = mu[positional]
        race_sigma = sigma[positional]
        actual = grp["finish_position"].to_numpy()

        rho, _ = spearmanr(race_mu, actual)
        if not np.isnan(rho):
            rhos.append(rho)

        prob = run_simulation(race_mu, race_sigma, n_sims=4_000)
        win_idx_pred = int(np.argmax(prob[:, 0]))
        win_idx_actual = int(np.argmin(actual))
        win_correct += int(win_idx_pred == win_idx_actual)

        podium_pred_set = set(np.argsort(-prob[:, :3].sum(axis=1))[:3].tolist())
        podium_actual_set = set(np.argsort(actual)[:3].tolist())
        podium_hits += len(podium_pred_set & podium_actual_set) / 3.0

        # Brier on winner
        win_prob = prob[:, 0]
        actual_winner_onehot = (actual == actual.min()).astype(float)
        briers.append(float(np.mean((win_prob - actual_winner_onehot) ** 2)))

    return {
        "n_val_races": n_races,
        "spearman_rho_mean": float(np.mean(rhos)) if rhos else float("nan"),
        "mae": mae,
        "win_pick_accuracy": (win_correct / n_races) if n_races else float("nan"),
        "podium_hit_rate": (podium_hits / n_races) if n_races else float("nan"),
        "brier_winner": float(np.mean(briers)) if briers else float("nan"),
    }


# ---------------------------------------------------------------------------
def train(out_dir: Path) -> None:
    training_path = Path("ml/data/training.parquet")
    quali_path = Path("ml/data/quali_training.parquet")
    if not training_path.exists():
        raise FileNotFoundError(f"{training_path} not found — run `python -m ml.build_dataset` first")

    df = pd.read_parquet(training_path)
    val_season = int(df["season"].max())
    log.info("Validation season = %s", val_season)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- pre-quali finish ----
    sub = df[df["mode"] == "pre_quali"].reset_index(drop=True)
    train_df = sub[sub["season"] < val_season].reset_index(drop=True)
    val_df = sub[sub["season"] == val_season].reset_index(drop=True)
    bundle, metrics = _fit_bundle(
        train_df, val_df, PRE_QUALI_FEATURES, "finish_position", version="preq-v1.0",
    )
    log.info("pre-quali metrics: %s", metrics)
    joblib.dump(bundle, out_dir / "pre_quali_finish.pkl")

    # ---- post-quali finish ----
    sub = df[df["mode"] == "post_quali"].reset_index(drop=True)
    if not sub.empty:
        train_df = sub[sub["season"] < val_season].reset_index(drop=True)
        val_df = sub[sub["season"] == val_season].reset_index(drop=True)
        bundle, metrics = _fit_bundle(
            train_df, val_df, POST_QUALI_FEATURES, "finish_position", version="postq-v1.0",
        )
        log.info("post-quali metrics: %s", metrics)
        joblib.dump(bundle, out_dir / "post_quali_finish.pkl")
    else:
        log.warning("no post_quali rows — skipping post_quali_finish")

    # ---- pole (quali gap regression) ----
    if quali_path.exists():
        qdf = pd.read_parquet(quali_path)
        train_df = qdf[qdf["season"] < val_season].reset_index(drop=True)
        val_df = qdf[qdf["season"] == val_season].reset_index(drop=True)
        models: dict[str, lgb.Booster] = {}
        for q in _QUANTILES:
            log.info("Fitting pole q=%.2f", q)
            models[f"q{int(q * 100):02d}"] = _fit_quantile(
                train_df[POLE_FEATURES], train_df["gap_to_pole_s"].to_numpy(float), q,
            )
        # quick pole MAE on q50
        pole_mae = float(mean_absolute_error(
            val_df["gap_to_pole_s"].to_numpy(float),
            models["q50"].predict(val_df[POLE_FEATURES]),
        )) if not val_df.empty else float("nan")
        bundle = {
            "version": "pole-v1.0",
            "features": POLE_FEATURES,
            "models": models,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {"mae_seconds": pole_mae},
        }
        log.info("pole metrics: mae=%.3fs", pole_mae)
        joblib.dump(bundle, out_dir / "pole.pkl")
    else:
        log.warning("no quali_training.parquet — skipping pole model")

    log.info("Done. Artifacts written to %s", out_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("ml/artifacts"))
    args = p.parse_args()
    train(args.out_dir)


if __name__ == "__main__":
    main()
