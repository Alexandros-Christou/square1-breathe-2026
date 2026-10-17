# Breathe: chronological air-quality estimation

Original autonomous competition pipeline for Square1 Breathe2026. Uses only the organiser's shipped data. No external source retrieval or target lookup.

## Run

Use Python3.12.14 and the exact package versions in requirements.txt:

```sh
python -m pip install -r requirements.txt
python forecast.py --data data --out output
```

Place `train.csv`, `test.csv` and `sample_submission.csv` from the official Breathe competition in `data/` first. The script writes validation.json, manifest.json and submission.csv. Do not publish the data or submission file.

For the independent reproduction check:

```sh
python forecast.py --data data --out repro --predict-only
```

The two submission CSVs must have the same SHA256 hash. The fixed seed is1701 and numeric kernels use4threads. The final model uses absolute-error histogram gradient boosting, and predictions are clipped at zero.

Chronological development: train before April2024, validate April–December2024. Confirmation: train before January2025, validate January–March2025. Internal scikit-learn early stopping uses only data inside each outer training period. The historical comparisons are not official public/private leaderboard scores.

Data attribution: US EPA Air Quality System daily PM2.5 parameter88101, with NASA Langley POWER meteorology. Use is governed by the organiser's data card and source licences. Only source code, requirements and these instructions are intended for publication.

The model substantially underestimates pollution extremes. It is a competition estimator, not an operational health-alert service.
