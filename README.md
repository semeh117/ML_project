⚽ Football Match Outcome Prediction
Machine Learning Project — Premier League & Champions League 2025-26

Overview
An end-to-end machine learning pipeline that predicts football match outcomes and uses those predictions to simulate the Premier League title race and the UEFA Champions League winner via Monte Carlo simulation.

Can Arsenal win the double?
The model says: 99.1% PL title · 31.8% UCL winner

Project Structure
ml/
├── ML.ipynb                  # Main notebook (all steps)
├── Matches.csv               # Primary dataset — 230k historical matches
├── EloRatings.csv            # Club Elo ratings (2000–2025)
├── fbref_2024_25.csv         # Player stats 2024-25 season (auxiliary)
├── best_model.pkl            # Saved trained model
├── pl_predictions.png        # PL title probability chart
├── final_predictions.png     # UCL probability chart
└── README.md

# Methodology

# Phase 1 — ML Classifier
Trained on 129,661 historical matches to predict full-time result (H / D / A)

Step	    Detail
Features	HomeElo, AwayElo, Form3, Form5, Odds, Derived features
Label	    FTResult — Home Win / Draw / Away Win
Split	    Time-based 80/20 — no data leakage
Models	    Logistic Regression, Random Forest, XGBoost
Selection	Best log loss → Logistic Regression


Why time-based split? A random split would let the model see future matches during training — artificially inflating accuracy. We split chronologically to mirror real deployment.
Why log loss over accuracy? The simulator uses predict_proba() — we need well-calibrated probabilities, not just correct labels.

 # Phase 2 — Monte Carlo Simulator

 The trained model's predict_proba() feeds directly into a 50,000-iteration simulation.

PL Simulator — generates remaining fixtures, simulates match outcomes, accumulates points, counts title wins
UCL Simulator — two-legged ties with home advantage, Poisson goal model, 50/50 penalty resolution

Performance optimisation: All fixture probabilities are pre-computed in a single batched predict_proba() call before the loop — reducing runtime from 5+ minutes → under 10 seconds.

# Results
Premier League 2025-26 (as of MW31)

Team	Current Pts	Title %	Avg Final Pts
Arsenal	70	99.1%	83.8
Man City	61	0.9%	72.1
Man United	55	0.0%	61.2
Aston Villa	54	0.0%	58.6
Liverpool	49	0.0%	59.7

EFA Champions League 2025-26 (QF onwards)

Club	    QF Advance	Reach Final	 Win UCL

Arsenal	         78.2%	     48.5%	   31.8%
Bayern Munich    70.7%   	 31.9%	   21.4%
PSG	             48.7%	     30.8%	   12.6%
Barcelona	     53.9%	     25.6%	   12.3%
Liverpool	     51.3%	     28.1%	   8.6%
Real Madrid      29.3%	     10.0%	   5.2%
Atletico Madrid	 46.1%	     15.5%	   4.4%
Sporting CP      21.8%	     9.7%	   3.8%

Model Performance
ModelAccuracyLog LossLogistic Regression51.37%0.9895 ✓Random Forest51.08%0.9907XGBoost51.37%0.9931


Model	                Accuracy	Log Loss
Logistic Regression     51.37%	    0.9895 
Random Forest	        51.08%	    0.9907
XGBoost	                51.37%	    0.9931

# Data Sources

Dataset      	      Source	                  Description
Matches.csv	          kaggle.com        	   230k matches, 20+ leagues, 2000–2025
EloRatings.csv	      eloratings.net	       Club Elo ratings, global coverage
fbref_2024_25.csv	  fbref.com     	       Player-level stats, 2024-25 PL season

Note: fbref_2024_25.csv was evaluated but not integrated into the final model — it contains no squad identifier column and covers PL players only, making it unsuitable for UCL predictions. Flagged for future work.


Limitations

- Training data covers up to June 2025 — no 2025-26 season match results
- Current Elo values were manually sourced from eloratings.net (March 2026)
- UCL penalty shootouts modelled as 50/50 — no club-specific conversion rates
- fbref xG data not integrated (no squad column, PL-only coverage)




