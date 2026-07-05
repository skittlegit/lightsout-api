FROM python:3.11-slim

WORKDIR /app

# lightgbm links against the OpenMP runtime, which the slim image does not
# ship. Without libgomp1, `import lightgbm` raises and joblib.load of the model
# bundles fails silently — the API then reports models_loaded: [] and every
# prediction returns model_unavailable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p ml/artifacts ml/data fastf1_cache

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
