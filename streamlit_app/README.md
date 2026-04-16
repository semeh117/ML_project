# ⚽ ML Football Predictor — Streamlit App

A full machine learning app to predict football match outcomes and simulate season results using Monte Carlo methods.

---

## 📦 Features

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview and feature list |
| 📊 Train Model | Upload Matches.csv, train 3 ML models, view metrics & feature importance |
| 🎯 Match Predictor | Predict any single match outcome with probability bars + MC simulation |
| 🏆 Season Simulator | Run Monte Carlo PL season simulation with editable standings |

---

## 🚀 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this folder to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — done! 🎉

> No server, no Docker, no cost.

---

## 📁 CSV Format Required

Your `Matches.csv` must include these columns:

```
HomeElo, AwayElo, Form3Home, Form5Home, Form3Away, Form5Away,
OddHome, OddDraw, OddAway, FTResult
```

`FTResult` values: `H` (home win), `A` (away win), `D` (draw)

---

## 🧠 Models

| Model | Why |
|-------|-----|
| Logistic Regression | Interpretable baseline, works well with linear odds signals |
| Random Forest | Captures non-linear interactions, robust to outliers |
| XGBoost | State-of-the-art tabular model, best on imbalanced classes |

Best model selected by **Log Loss** (not accuracy), ensuring well-calibrated probabilities for the Monte Carlo simulator.
