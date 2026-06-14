"""Streamlit dashboard for FIFA prediction results."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import FIFAPredictor
from src.simulation.simulator import world_cup_2026_groups


st.set_page_config(
    page_title="FIFA World Cup Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ FIFA World Cup Group Stage Predictor")
st.markdown("Hybrid ML + Statistical ensemble model for match outcome prediction.")


@st.cache_resource
def load_predictor():
    predictor = FIFAPredictor()
    ensemble_path = Path('models/hybrid_ensemble.joblib')
    if ensemble_path.exists():
        from src.models.ensemble import HybridEnsemble
        predictor.ensemble = HybridEnsemble.load(ensemble_path)
        predictor.is_trained_ = True
        predictor.X_ = pd.DataFrame()
        return predictor
    return None


predictor = load_predictor()

tab1, tab2, tab3, tab4 = st.tabs(["Match Predictor", "Group Simulation", "Model Performance", "About"])

with tab1:
    st.header("Predict a Match")
    col1, col2 = st.columns([1, 1])

    all_teams = sorted([
        'Brazil', 'Argentina', 'Uruguay', 'Colombia', 'Chile', 'Peru', 'Ecuador', 'Paraguay', 'Bolivia', 'Venezuela',
        'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 'England', 'Portugal', 'Belgium', 'Croatia', 'Denmark',
        'Switzerland', 'Poland', 'Sweden', 'Austria', 'Czech Republic', 'Serbia', 'Ukraine', 'Turkey', 'Russia', 'Greece',
        'Mexico', 'United States', 'Costa Rica', 'Honduras', 'Canada', 'Panama', 'Jamaica',
        'Japan', 'South Korea', 'Australia', 'Saudi Arabia', 'Iran', 'Qatar', 'United Arab Emirates',
        'Nigeria', 'Ghana', 'Senegal', 'Cameroon', 'Ivory Coast', 'Egypt', 'Morocco', 'Algeria', 'Tunisia', 'South Africa',
        'New Zealand', 'Scotland', 'Wales',
    ])

    with col1:
        team1 = st.selectbox("Home Team", all_teams, index=all_teams.index('Brazil'))
    with col2:
        team2 = st.selectbox("Away Team", all_teams, index=all_teams.index('Argentina'))

    if st.button("Predict", type="primary", use_container_width=True):
        if predictor and predictor.is_trained_:
            result = predictor.predict_match(team1, team2)
            probs = result['probabilities']

            fig, ax = plt.subplots(figsize=(8, 3))
            colors = ['#27ae60', '#f39c12', '#e74c3c']
            labels = [f'{team1} Win', 'Draw', f'{team2} Win']
            values = [probs['home_win'], probs['draw'], probs['away_win']]
            bars = ax.barh(labels, values, color=colors, height=0.6)
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1%}', va='center', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title(f'{team1} vs {team2} — {result["prediction"]}', fontsize=16, fontweight='bold')
            st.pyplot(fig)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric(f"{team1} Win", f"{probs['home_win']:.1%}")
            col_b.metric("Draw", f"{probs['draw']:.1%}")
            col_c.metric(f"{team2} Win", f"{probs['away_win']:.1%}")
        else:
            st.warning("No trained model found. Please run `python cli.py train` first.")

with tab2:
    st.header("2026 World Cup Group Stage Simulation")
    st.caption(f"Based on 10,000 Monte Carlo simulations")

    groups = world_cup_2026_groups()

    if st.button("Run Simulation", type="primary", use_container_width=True):
        if predictor and predictor.is_trained_:
            results, qualifiers = predictor.simulate_tournament(groups)

            for group_name, data in results.items():
                st.subheader(f"Group {group_name}")
                sorted_teams = sorted(data['qualification_probs'].keys(),
                                      key=lambda t: data['qualification_probs'][t], reverse=True)

                table_data = []
                for team in sorted_teams:
                    qp = data['qualification_probs'][team] * 100
                    tp = data['top_team_probs'][team] * 100
                    pd_dist = data['points_distribution'][team]
                    avg_pts = sum(i * pd_dist[i] for i in range(10))
                    table_data.append({
                        'Team': team,
                        'Qualify %': f"{qp:.1f}%",
                        'Win Group %': f"{tp:.1f}%",
                        'Avg Points': f"{avg_pts:.2f}",
                    })

                st.dataframe(
                    pd.DataFrame(table_data),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Team': st.column_config.TextColumn('Team', width='medium'),
                        'Qualify %': st.column_config.TextColumn('Qualify %', width='small'),
                        'Win Group %': st.column_config.TextColumn('Win Group %', width='small'),
                        'Avg Points': st.column_config.TextColumn('Avg Pts', width='small'),
                    }
                )

            st.subheader("🏆 Predicted Round of 16 Qualifiers")
            qual_cols = st.columns(4)
            for i, (group_name, teams) in enumerate(qualifiers.items()):
                with qual_cols[i % 4]:
                    st.markdown(f"**Group {group_name}**")
                    st.markdown(f"1. {teams[0]}")
                    st.markdown(f"2. {teams[1]}")
        else:
            st.warning("No trained model found. Please run `python cli.py train` first.")

with tab3:
    st.header("Model Performance")

    results_csv = Path('results/comparison.csv')
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        st.dataframe(df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            models = df['model']
            accs = df['accuracy']
            colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
            bars = ax.barh(models, accs, color=colors_bar[:len(models)])
            for bar, acc in zip(bars, accs):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f'{acc:.4f}', va='center', fontsize=10)
            ax.set_xlabel('Accuracy')
            ax.set_title('Model Accuracy Comparison')
            ax.set_xlim(0, 1)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            loglosses = df['log_loss']
            bars = ax.barh(models, loglosses, color=colors_bar[:len(models)])
            for bar, ll in zip(bars, loglosses):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f'{ll:.4f}', va='center', fontsize=10)
            ax.set_xlabel('Log Loss')
            ax.set_title('Model Log Loss (lower is better)')
            st.pyplot(fig)
    else:
        st.info("No evaluation results found. Run `python cli.py evaluate` first.")

with tab4:
    st.header("About the Model")
    st.markdown("""
    ### Architecture
    This predictor uses a **hybrid ensemble** combining:

    | Model | Type | Role |
    |-------|------|------|
    | **XGBoost** | Gradient Boosted Trees | Primary classifier |
    | **LightGBM** | Gradient Boosted Trees | Ensemble diversity |
    | **Random Forest** | Bagging Ensemble | Robust baseline |
    | **Neural Network** | 3-layer MLP | Pattern detection |
    | **Poisson** | Statistical | Score-based prediction |
    | **Elo** | Rating System | Head-to-head dynamics |

    The outputs are stacked via a **logistic regression meta-learner** with Platt scaling calibration.

    ### Features
    - Rolling form (weighted by recency)
    - Goal scoring/conceding averages
    - FIFA / Elo ranking differentials
    - Head-to-head history
    - Tournament experience
    - Confederation alignment
    - Host nation advantage

    ### Accuracy
    The model is evaluated on 2018 and 2022 World Cup group stage results.
    """)
