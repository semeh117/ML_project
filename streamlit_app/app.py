import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import itertools
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import xgboost as xgb

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ ML Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .main { background: #0D1117; }

  .hero {
    background: linear-gradient(135deg, #0D1117 0%, #1a0a0a 100%);
    border: 1px solid #2a2a3a;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    text-align: center;
  }
  .hero h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 52px;
    color: #F5C842;
    letter-spacing: 2px;
    margin: 0;
  }
  .hero p { color: #8892a4; font-size: 16px; margin-top: 8px; }

  .metric-card {
    background: #161b27;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-card .val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 42px;
    color: #F5C842;
    line-height: 1;
  }
  .metric-card .lbl { font-size: 12px; color: #8892a4; text-transform: uppercase; letter-spacing: .08em; margin-top: 4px; }

  .section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    color: #E8EAF0;
    letter-spacing: 1px;
    border-bottom: 2px solid #EF0107;
    padding-bottom: 6px;
    margin-bottom: 20px;
  }

  .winner-box {
    background: linear-gradient(135deg, rgba(239,1,7,.12), rgba(245,200,66,.07));
    border: 1px solid rgba(245,200,66,.4);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .winner-box .wval {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 56px;
    line-height: 1;
  }
  .winner-box .wlbl { color: #8892a4; font-size: 14px; }

  .prob-bar-wrap { background: #1e2433; border-radius: 6px; height: 10px; overflow: hidden; margin-top: 4px; }
  .stButton > button {
    background: #EF0107;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
  }
  .stButton > button:hover { background: #c00; }

  div[data-testid="stSidebar"] { background: #0D1117; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FEATURES = [
    'HomeElo', 'AwayElo',
    'Form3Home', 'Form5Home',
    'Form3Away', 'Form5Away',
    'OddHome', 'OddDraw', 'OddAway',
]
LABEL = 'FTResult'
FEATURES_V2 = FEATURES + ['EloDiff', 'FormDiff3', 'FormDiff5',
                           'ImpliedHome', 'ImpliedDraw', 'ImpliedAway']

MODEL_PATH = 'best_model.pkl'
N_SIMS = 20_000  # Keep fast for web


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def elo_to_odds(elo_home, elo_away, home_adv=65):
    diff = (elo_home + home_adv) - elo_away
    p_h = 1 / (1 + 10 ** (-diff / 400))
    p_a = 1 - p_h
    p_d = 0.27
    p_h *= (1 - p_d)
    p_a *= (1 - p_d)
    margin = 1.05
    return (margin / p_h, margin / p_d, margin / p_a)


def add_derived(df):
    df = df.copy()
    df['OddHome'] = df['OddHome'].clip(1.01, 20.0)
    df['OddDraw'] = df['OddDraw'].clip(1.01, 15.0)
    df['OddAway'] = df['OddAway'].clip(1.01, 30.0)
    df['EloDiff']    = df['HomeElo'] - df['AwayElo']
    df['FormDiff3']  = df['Form3Home'] - df['Form3Away']
    df['FormDiff5']  = df['Form5Home'] - df['Form5Away']
    df['ImpliedHome'] = 1 / df['OddHome']
    df['ImpliedDraw'] = 1 / df['OddDraw']
    df['ImpliedAway'] = 1 / df['OddAway']
    return df


def train_models(df):
    X = df[FEATURES_V2]
    le = LabelEncoder()
    y  = le.fit_transform(df[LABEL])

    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=0.5, solver='lbfgs'),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                                 min_samples_leaf=20, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8,
                                      eval_metric='mlogloss', random_state=42, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_test_sc)
            proba = model.predict_proba(X_test_sc)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)

        results[name] = {
            'model': model,
            'preds': preds,
            'proba': proba,
            'acc':   accuracy_score(y_test, preds),
            'log_loss': log_loss(y_test, proba),
            'report': classification_report(y_test, preds, target_names=le.classes_, output_dict=True),
            'cm': confusion_matrix(y_test, preds),
        }

    best_name = min(results, key=lambda k: results[k]['log_loss'])

    # Save
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'model':    results[best_name]['model'],
            'scaler':   scaler,
            'encoder':  le,
            'name':     best_name,
            'features': FEATURES_V2,
            'results':  {k: {kk: vv for kk, vv in v.items() if kk != 'model'} for k, v in results.items()},
            'y_test':   y_test,
        }, f)

    return results, best_name, scaler, le, X_test, y_test


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def build_row(home_elo, away_elo, f3h, f5h, f3a, f5a, odd_h, odd_d, odd_a):
    odd_h = np.clip(odd_h, 1.01, 20.0)
    odd_d = np.clip(odd_d, 1.01, 15.0)
    odd_a = np.clip(odd_a, 1.01, 30.0)
    row = {
        'HomeElo': home_elo, 'AwayElo': away_elo,
        'Form3Home': f3h, 'Form5Home': f5h,
        'Form3Away': f3a, 'Form5Away': f5a,
        'OddHome': odd_h, 'OddDraw': odd_d, 'OddAway': odd_a,
        'EloDiff': home_elo - away_elo,
        'FormDiff3': f3h - f3a, 'FormDiff5': f5h - f5a,
        'ImpliedHome': 1 / odd_h, 'ImpliedDraw': 1 / odd_d, 'ImpliedAway': 1 / odd_a,
    }
    return pd.DataFrame([row])[FEATURES_V2]


def predict_match(saved, home_elo, away_elo, f3h, f5h, f3a, f5a, odd_h, odd_d, odd_a):
    X_input = build_row(home_elo, away_elo, f3h, f5h, f3a, f5a, odd_h, odd_d, odd_a)
    X_sc = saved['scaler'].transform(X_input)
    le   = saved['encoder']
    if saved['name'] == 'Logistic Regression':
        proba = saved['model'].predict_proba(X_sc)[0]
    else:
        proba = saved['model'].predict_proba(X_input)[0]
    p_away = proba[le.transform(['A'])[0]]
    p_draw = proba[le.transform(['D'])[0]]
    p_home = proba[le.transform(['H'])[0]]
    return p_home, p_draw, p_away


# ─── Monte Carlo ────────────────────────────────────────────
def simulate_pl(saved, standings, remaining=7, n_sims=N_SIMS):
    teams = list(standings.keys())
    all_fixtures = list(itertools.permutations(teams, 2))
    np.random.shuffle(all_fixtures)
    fixtures, game_count = [], {t: 0 for t in teams}
    for home, away in all_fixtures:
        if game_count[home] < remaining and game_count[away] < remaining:
            fixtures.append((home, away))
            game_count[home] += 1
            game_count[away] += 1

    rows = []
    for home, away in fixtures:
        s = standings
        odd_h, odd_d, odd_a = elo_to_odds(s[home]['elo'], s[away]['elo'])
        rows.append({
            'HomeElo': s[home]['elo'], 'AwayElo': s[away]['elo'],
            'Form3Home': s[home]['form3'], 'Form5Home': s[home]['form5'],
            'Form3Away': s[away]['form3'], 'Form5Away': s[away]['form5'],
            'OddHome': np.clip(odd_h, 1.01, 20), 'OddDraw': np.clip(odd_d, 1.01, 15),
            'OddAway': np.clip(odd_a, 1.01, 30),
            'EloDiff': s[home]['elo'] - s[away]['elo'],
            'FormDiff3': s[home]['form3'] - s[away]['form3'],
            'FormDiff5': s[home]['form5'] - s[away]['form5'],
            'ImpliedHome': 1 / np.clip(odd_h, 1.01, 20),
            'ImpliedDraw': 1 / np.clip(odd_d, 1.01, 15),
            'ImpliedAway': 1 / np.clip(odd_a, 1.01, 30),
        })

    X_batch = pd.DataFrame(rows)[FEATURES_V2]
    X_sc    = saved['scaler'].transform(X_batch)
    le      = saved['encoder']
    if saved['name'] == 'Logistic Regression':
        proba_batch = saved['model'].predict_proba(X_sc)
    else:
        proba_batch = saved['model'].predict_proba(X_batch)

    idx_h, idx_d, idx_a = le.transform(['H'])[0], le.transform(['D'])[0], le.transform(['A'])[0]
    probs = [(proba_batch[i][idx_h], proba_batch[i][idx_d], proba_batch[i][idx_a])
             for i in range(len(fixtures))]

    team_idx = {t: i for i, t in enumerate(teams)}
    base_pts = np.array([standings[t]['pts'] for t in teams])
    title_count = np.zeros(len(teams), dtype=int)
    top4_count  = np.zeros(len(teams), dtype=int)
    final_pts   = np.zeros((n_sims, len(teams)), dtype=int)
    rand_matrix = np.random.random((n_sims, len(fixtures)))

    for sim in range(n_sims):
        sim_pts = base_pts.copy()
        for j, (home, away) in enumerate(fixtures):
            ph, pd_, pa = probs[j]
            r = rand_matrix[sim, j]
            if r < ph:               sim_pts[team_idx[home]] += 3
            elif r < ph + pd_:       sim_pts[team_idx[home]] += 1; sim_pts[team_idx[away]] += 1
            else:                    sim_pts[team_idx[away]] += 3
        final_pts[sim] = sim_pts
        title_count[np.argmax(sim_pts)] += 1
        for idx in np.argsort(sim_pts)[-4:]:
            top4_count[idx] += 1

    return pd.DataFrame({
        'Team':           teams,
        'Current Pts':    [standings[t]['pts'] for t in teams],
        'Title %':        title_count / n_sims * 100,
        'Top 4 %':        top4_count  / n_sims * 100,
        'Avg Final Pts':  final_pts.mean(axis=0).round(1),
        'Min Pts':        final_pts.min(axis=0),
        'Max Pts':        final_pts.max(axis=0),
    }).sort_values('Title %', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Navigation")
    page = st.radio("", ["🏠 Home", "📊 Train Model", "🎯 Match Predictor", "🏆 Season Simulator"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("ML model trained on 129,661 historical football matches using Elo ratings, recent form, and bookmaker odds.")


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
      <h1>⚽ ML FOOTBALL PREDICTOR</h1>
      <p>Logistic Regression · Random Forest · XGBoost · Monte Carlo Simulation</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1,c2,c3,c4],
        ["129,661","3","50K","15"],
        ["Matches in Dataset","ML Models","MC Simulations","Input Features"]
    ):
        col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔄 Workflow")
        st.markdown("""
        1. **Load & Clean** — 129k matches, drop rows with missing features  
        2. **Feature Engineering** — EloDiff, FormDiff, Implied probabilities  
        3. **Time-based split** — 80% train (old) / 20% test (recent), no leakage  
        4. **Train 3 models** — LR baseline → RF ensemble → XGBoost boosting  
        5. **Evaluate** — Accuracy + Log Loss (for probability calibration)  
        6. **Monte Carlo** — 50k simulated seasons using model probabilities  
        """)
    with col2:
        st.markdown("### 📌 Features Used")
        feat_df = pd.DataFrame({
            "Feature": ["HomeElo / AwayElo", "Form3/5 Home & Away", "OddHome / Draw / Away",
                        "EloDiff", "FormDiff3 / FormDiff5", "ImpliedHome/Draw/Away"],
            "Type": ["Raw","Raw","Raw","Engineered","Engineered","Engineered"],
            "Signal": ["Team strength","Recent momentum","Market consensus","Strength gap","Form gap","True probability"],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.info("👈 **Upload your Matches.csv** in the *Train Model* page to get started, or jump straight to *Match Predictor* if you already have a trained model.")


# ─────────────────────────────────────────────
# PAGE: TRAIN MODEL
# ─────────────────────────────────────────────
elif page == "📊 Train Model":
    st.markdown('<div class="section-header">📊 Train ML Models</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your **Matches.csv**", type="csv")

    if uploaded:
        with st.spinner("Loading data…"):
            matches = pd.read_csv(uploaded, low_memory=False)
            if 'MatchDate' in matches.columns:
                matches['MatchDate'] = pd.to_datetime(matches['MatchDate'], errors='coerce')
            df_raw = matches[FEATURES + [LABEL]].dropna()
            df     = add_derived(df_raw)

        st.success(f"✅ Loaded **{len(df):,}** complete matches")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Matches", f"{len(df):,}")
        col2.metric("Home Wins (H)", f"{(df[LABEL]=='H').sum():,}  ({(df[LABEL]=='H').mean():.1%})")
        col3.metric("Away Wins (A)", f"{(df[LABEL]=='A').sum():,}  ({(df[LABEL]=='A').mean():.1%})")

        with st.expander("Feature Statistics"):
            st.dataframe(df[FEATURES_V2].describe().round(2), use_container_width=True)

        if st.button("🚀 Train All 3 Models"):
            with st.spinner("Training… this may take 1–2 minutes"):
                results, best_name, scaler, le, X_test, y_test = train_models(df)

            st.success(f"✅ Training complete! Best model: **{best_name}**")

            st.markdown("### Model Comparison")
            comp = pd.DataFrame({
                name: {
                    'Accuracy': f"{r['acc']*100:.2f}%",
                    'Log Loss': f"{r['log_loss']:.4f}",
                    'H F1': f"{r['report']['H']['f1-score']:.3f}",
                    'A F1': f"{r['report']['A']['f1-score']:.3f}",
                    'D F1': f"{r['report']['D']['f1-score']:.3f}",
                } for name, r in results.items()
            }).T
            st.dataframe(comp.style.highlight_min(subset=['Log Loss'], color='#1a3a1a'), use_container_width=True)

            st.markdown("### Confusion Matrices")
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            fig.patch.set_facecolor('#0D1117')
            for ax, (name, r) in zip(axes, results.items()):
                sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Reds', ax=ax,
                            xticklabels=le.classes_, yticklabels=le.classes_)
                ax.set_title(f"{name}\nAcc: {r['acc']*100:.1f}%", color='white')
                ax.set_facecolor('#161b27')
                ax.tick_params(colors='white')
                ax.yaxis.label.set_color('white')
                ax.xaxis.label.set_color('white')
            plt.tight_layout()
            st.pyplot(fig)

            # Feature importance
            xgb_model = results['XGBoost']['model']
            fi = pd.Series(xgb_model.feature_importances_, index=FEATURES_V2).sort_values()
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            fig2.patch.set_facecolor('#0D1117')
            ax2.set_facecolor('#161b27')
            colors = ['#EF0107' if v > 0.1 else '#F5C842' if v > 0.05 else '#4a5568' for v in fi.values]
            fi.plot(kind='barh', ax=ax2, color=colors)
            ax2.axvline(0.05, color='white', linestyle='--', alpha=0.4)
            ax2.set_title('XGBoost — Feature Importance', color='white', fontsize=13)
            ax2.tick_params(colors='white')
            ax2.set_xlabel('Importance', color='white')
            plt.tight_layout()
            st.pyplot(fig2)
    else:
        st.info("👆 Upload your Matches.csv to train the model. The file needs these columns:\n\n`HomeElo, AwayElo, Form3Home, Form5Home, Form3Away, Form5Away, OddHome, OddDraw, OddAway, FTResult`")


# ─────────────────────────────────────────────
# PAGE: MATCH PREDICTOR
# ─────────────────────────────────────────────
elif page == "🎯 Match Predictor":
    st.markdown('<div class="section-header">🎯 Single Match Predictor</div>', unsafe_allow_html=True)

    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ No trained model found. Please go to **Train Model** first and upload your CSV.")
        st.stop()

    saved = load_model()
    st.info(f"Using model: **{saved['name']}**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏠 Home Team")
        home_name  = st.text_input("Team Name", "Arsenal", key="h_name")
        home_elo   = st.number_input("Elo Rating", 1000, 2500, 2372, key="h_elo")
        h_form3    = st.slider("Form (last 3 games, max pts=9)", 0, 9, 7, key="h3")
        h_form5    = st.slider("Form (last 5 games, max pts=15)", 0, 15, 11, key="h5")

    with col2:
        st.markdown("#### ✈️ Away Team")
        away_name  = st.text_input("Team Name", "Sporting CP", key="a_name")
        away_elo   = st.number_input("Elo Rating", 1000, 2500, 2178, key="a_elo")
        a_form3    = st.slider("Form (last 3 games)", 0, 9, 5, key="a3")
        a_form5    = st.slider("Form (last 5 games)", 0, 15, 7, key="a5")

    st.markdown("#### 💰 Bookmaker Odds")
    oc1, oc2, oc3 = st.columns(3)
    odd_h = oc1.number_input(f"{home_name} Win", 1.01, 30.0, 1.55, step=0.05)
    odd_d = oc2.number_input("Draw", 1.01, 20.0, 4.10, step=0.05)
    odd_a = oc3.number_input(f"{away_name} Win", 1.01, 50.0, 5.80, step=0.05)

    if st.button("🔮 Predict Match Outcome"):
        ph, pd_, pa = predict_match(saved, home_elo, away_elo, h_form3, h_form5, a_form3, a_form5, odd_h, odd_d, odd_a)

        st.markdown("---")
        st.markdown(f"### {home_name} vs {away_name}")

        r1, r2, r3 = st.columns(3)
        for col, label, prob, color in zip(
            [r1, r2, r3],
            [f"🏠 {home_name} Win", "🤝 Draw", f"✈️ {away_name} Win"],
            [ph, pd_, pa],
            ["#EF0107", "#F5C842", "#1A8FE3"]
        ):
            col.markdown(f"""
            <div style="background:#161b27;border:1px solid {color}40;border-radius:12px;padding:20px;text-align:center;">
              <div style="font-size:14px;color:#8892a4;margin-bottom:8px;">{label}</div>
              <div style="font-family:'Bebas Neue',sans-serif;font-size:52px;color:{color};line-height:1;">{prob:.1%}</div>
              <div class="prob-bar-wrap" style="margin-top:12px;">
                <div style="height:10px;width:{prob*100:.1f}%;background:{color};border-radius:6px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        winner = max([("Home Win", ph), ("Draw", pd_), ("Away Win", pa)], key=lambda x: x[1])
        st.markdown(f"""
        <br><div style="background:rgba(245,200,66,.08);border:1px solid rgba(245,200,66,.3);border-radius:10px;
        padding:16px 24px;text-align:center;margin-top:12px;">
          <span style="color:#8892a4;">Most likely outcome: </span>
          <span style="color:#F5C842;font-weight:600;font-size:18px;">{winner[0]} ({winner[1]:.1%})</span>
        </div>
        """, unsafe_allow_html=True)

        # Monte Carlo for this single match
        st.markdown("#### 🎲 Monte Carlo Simulation (10,000 runs)")
        outcomes = np.random.choice(['H', 'D', 'A'], size=10_000, p=[ph, pd_, pa])
        counts = {k: (outcomes == k).sum() for k in ['H', 'D', 'A']}

        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor('#0D1117')
        ax.set_facecolor('#161b27')
        bars = ax.bar(['Home Win', 'Draw', 'Away Win'],
                      [counts['H'], counts['D'], counts['A']],
                      color=['#EF0107', '#F5C842', '#1A8FE3'], edgecolor='none', width=0.5)
        for bar, val in zip(bars, counts.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{val/100:.1f}%', ha='center', color='white', fontsize=11, fontweight='bold')
        ax.set_ylabel('Simulations', color='white')
        ax.tick_params(colors='white')
        ax.spines[:].set_visible(False)
        ax.set_ylim(0, max(counts.values()) * 1.15)
        plt.tight_layout()
        st.pyplot(fig)


# ─────────────────────────────────────────────
# PAGE: SEASON SIMULATOR
# ─────────────────────────────────────────────
elif page == "🏆 Season Simulator":
    st.markdown('<div class="section-header">🏆 Season Monte Carlo Simulator</div>', unsafe_allow_html=True)

    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ No trained model found. Please go to **Train Model** first.")
        st.stop()

    saved = load_model()

    st.markdown("### 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League Standings")
    st.caption("Edit the table below with current standings, then run the simulation.")

    default_standings = {
        'Arsenal':      {'pts': 70, 'played': 31, 'elo': 2372, 'form3': 7, 'form5': 11},
        'Man City':     {'pts': 61, 'played': 30, 'elo': 2255, 'form3': 6, 'form5': 10},
        'Man United':   {'pts': 55, 'played': 31, 'elo': 2050, 'form3': 5, 'form5':  8},
        'Aston Villa':  {'pts': 54, 'played': 31, 'elo': 2020, 'form3': 4, 'form5':  7},
        'Liverpool':    {'pts': 49, 'played': 31, 'elo': 2206, 'form3': 4, 'form5':  7},
    }

    standings_df = pd.DataFrame(default_standings).T.reset_index()
    standings_df.columns = ['Team', 'pts', 'played', 'elo', 'form3', 'form5']

    edited = st.data_editor(standings_df, use_container_width=True, hide_index=True, num_rows="dynamic")
    remaining = st.slider("Remaining matches per team", 1, 20, 7)
    n_sims_pl  = st.select_slider("Simulations", [5000, 10000, 20000, 50000], value=20000)

    if st.button("▶️ Run Premier League Simulation"):
        standings = {
            row['Team']: {'pts': int(row['pts']), 'played': int(row['played']),
                          'elo': float(row['elo']), 'form3': int(row['form3']), 'form5': int(row['form5'])}
            for _, row in edited.iterrows()
        }

        with st.spinner(f"Running {n_sims_pl:,} simulations…"):
            pl_res = simulate_pl(saved, standings, remaining, n_sims_pl)

        st.markdown("### Results")
        st.dataframe(
            pl_res.style
                  .background_gradient(subset=['Title %'], cmap='Reds')
                  .background_gradient(subset=['Top 4 %'], cmap='Blues')
                  .format({'Title %': '{:.1f}%', 'Top 4 %': '{:.1f}%', 'Avg Final Pts': '{:.1f}'}),
            use_container_width=True, hide_index=True
        )

        # Bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0D1117')
        colors = ['#EF0107', '#6CABDD', '#DA291C', '#95BFE5', '#C8102E',
                  '#FFCD00', '#7A2182', '#003087'][:len(pl_res)]

        for ax, col, title in zip(axes, ['Title %', 'Top 4 %'], ['Title Probability', 'Top 4 Probability']):
            ax.set_facecolor('#161b27')
            bars = ax.bar(pl_res['Team'], pl_res[col], color=colors, edgecolor='none', width=0.6)
            for bar, val in zip(bars, pl_res[col]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{val:.1f}%', ha='center', color='white', fontsize=10, fontweight='bold')
            ax.set_title(title, color='white', fontsize=13, fontweight='bold')
            ax.tick_params(colors='white', axis='x', rotation=20)
            ax.tick_params(colors='white', axis='y')
            ax.set_ylabel('Probability (%)', color='white')
            ax.spines[:].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

        # Points distribution for leader
        leader = pl_res.iloc[0]['Team']
        st.markdown(f"#### 📊 Final Points Distribution — {leader}")
        st.caption(f"Avg: **{pl_res.iloc[0]['Avg Final Pts']}** pts | Range: [{pl_res.iloc[0]['Min Pts']} — {pl_res.iloc[0]['Max Pts']}]")
