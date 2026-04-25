# Fitro-Fyp (Fitaro) — AI Garment Size Recommender

Final Year Project — garment size recommendation with explanations.

Fitaro predicts the best-fit garment size (S / M / L / XL / XXL) from body
measurements and fit preference, then explains *why* that size was chosen using
SHAP values converted to plain English.

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python main.py
```

This will:
- Load and validate the dataset.
- Train an XGBoost classifier with 5-fold cross-validated hyperparameter search.
- Evaluate accuracy, macro-F1, and the **adjacent-size error rate**.
- Save SHAP global importance plots to `outputs/plots/`.
- Print 3 sample recommendations with full justifications.

## Make a prediction

```bash
# Use the hardcoded default example
python predict.py

# Override individual measurements
python predict.py --height 1.68 --weight 60 --age 25 \
    --chest 36 --length 27 --sleeve 23 --shoulder 15.5 --fit Slimfit
```

## Frontend (web UI)

Create and run the frontend from the `Frontend/` folder (this does **not** replace `predict.py`).

```bash
pip install -r requirements.txt
python Frontend/app.py
```

Then open the UI at `http://127.0.0.1:8000/`.

## Hosting (so users can use the model via the frontend)

This project ships a single FastAPI app (`Frontend/app.py`) that:
- serves the web UI (templates + static assets)
- exposes the model endpoint at `POST /api/predict`
- can retrain via the UI (`/model` → Train again)

### Option A (recommended): Render (Docker)

1. Push this repo to GitHub (steps below).
2. In Render, create a **New Web Service** → connect your GitHub repo.
3. Choose **Docker** as the environment (Render will use `Dockerfile`).
4. Set **Health Check Path** to `/`.
5. Deploy.

After deploy, open your Render service URL and you’ll be able to use the model directly from the frontend UI.

### Option B: Any host that can run a web command

Use the command from `Procfile`:

```bash
uvicorn Frontend.app:app --host 0.0.0.0 --port $PORT
```

## Project layout

```
fitaro/
├── dataset/                        # Raw CSV — do not modify
├── src/
│   ├── config.py                   # All constants and paths
│   ├── data_loader.py              # CSV loading + validation
│   ├── preprocessor.py             # Feature scaling / encoding
│   ├── model.py                    # XGBoost training, saving, loading
│   ├── evaluator.py                # Metrics + confusion matrix
│   ├── explainer.py                # SHAP plots + text justification
│   └── predictor.py                # End-to-end prediction class
├── outputs/
│   ├── models/                     # Saved model + preprocessor (.joblib)
│   ├── plots/                      # PNG plots (confusion matrix, SHAP)
│   └── reports/                    # Text evaluation report
├── main.py                         # Full pipeline runner
├── predict.py                      # Standalone inference script
└── requirements.txt
```

## Key metric: Adjacent-Size Error Rate

Misclassifying a customer as M when they are L is far less harmful than
predicting S for an XXL. The evaluation report therefore breaks errors into:

- **Adjacent errors** (S↔M, M↔L, L↔XL, XL↔XXL) — off-by-one mistakes.
- **Non-adjacent errors** — larger misses that would result in poor fit.

A high adjacent-size error rate and low non-adjacent rate indicates the model
is making sensible near-miss predictions even when it is wrong.

## Fit preference values

| Value | Meaning |
|-------|---------|
| `Regular` | Standard fit |
| `Slimfit` | Closer-to-body cut |
| `Oversize` | Loose, relaxed cut |

## Example justification output

```
Recommended size: L (confidence: 71%)

Why this size?
  → Chest measurement (42.0 in) pushed toward size L
  → Weight (82 kg) pushed toward size L
  → Oversize fit preference pushed toward size L
  → Height, body length, sleeve length had minimal influence

Next closest size: XL (18%)
```
