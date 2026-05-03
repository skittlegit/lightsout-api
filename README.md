# lightsout-api

FastAPI backend powering the F1 2026 race-result and pole-sitter prediction dashboard.

## Architecture

Three LightGBM quantile-regression bundles (q=0.10, 0.50, 0.90):

- `pre_quali_finish.pkl` — predicts finishing position from features known before qualifying
- `post_quali_finish.pkl` — predicts finishing position with grid + quali pace included
- `pole.pkl` — predicts qualifying gap-to-pole; lowest predicted gap = predicted pole sitter

At inference, μ = q50 and σ ≈ (q90 − q10) / 2.5631 per driver. Monte Carlo samples
~10 000 race orderings (`s_i ~ N(μ_i, σ_i)`, sort, tally positions) to produce the
full P(driver, position) joint distribution. Win prob, podium prob, points prob, and
expected position are all derived from the same simulation — internally consistent.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env

# 1. Build training set (slow — pulls FastF1 data 2018-2025)
python -m ml.build_dataset

# 2. Train all three model bundles
python -m ml.train

# 3. (Optional) Backtest on a held-out season
python -m ml.evaluate --season 2025

# 4. Serve
uvicorn app.main:app --reload
```

## Endpoints

| Method | Path | Notes |
|--------|------|-------|
| GET    | `/health` | Reports which models loaded |
| GET    | `/api/standings/drivers?season=2026` | Cached 1h |
| GET    | `/api/standings/constructors?season=2026` | Cached 1h |
| GET    | `/api/calendar?season=2026` | Cached 24h |
| GET    | `/api/predictions/next` | Resolves next round, returns prediction |
| GET    | `/api/predictions/{round}?season=2026` | Pre + (if available) post-quali |
| POST   | `/api/predictions/{round}/refresh` | `X-API-Key` required |
| POST   | `/api/retrain` | `X-API-Key`; runs in background, returns 202 |

## Deployment (Railway)

- Builder: NIXPACKS (see `railway.toml`)
- Mount `fastf1_cache` as a persistent volume
- Set `CORS_ORIGINS`, `RETRAIN_API_KEY` env vars
- Model artifacts are gitignored — fetch them at deploy time from S3/R2 or a `models` branch
- If artifacts are missing the API still serves standings/calendar; predictions return `model_unavailable`
