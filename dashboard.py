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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Match Predictor", "Group Simulation", "All Fixtures", "Live Tournament", "Model Performance"])

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
    st.header("All Group Stage Fixtures")
    if predictor and predictor.is_trained_:
        groups = world_cup_2026_groups()
        if st.button("Show All Fixtures", type="primary", use_container_width=True):
            fixtures_df = predictor.predict_all_group_matches(groups)
            for group in sorted(fixtures_df['Group'].unique()):
                st.subheader(f"Group {group}")
                gdf = fixtures_df[fixtures_df['Group'] == group].reset_index(drop=True)
                gdf['Match'] = gdf['Team1'] + ' vs ' + gdf['Team2']
                display = gdf[['Match', 'H Win %', 'Draw %', 'A Win %', 'Prediction']].copy()

                def color_pred(val):
                    if val == 'Home Win':
                        return 'background-color: #27ae6033'
                    elif val == 'Away Win':
                        return 'background-color: #e74c3c33'
                    return 'background-color: #f39c1233'

                styled = display.style.applymap(color_pred, subset=['Prediction'])
                for _, row in display.iterrows():
                    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1.2])
                    c1.markdown(f"**{row['Team1']}** vs **{row['Team2']}**")
                    c2.markdown(f"{row['H Win %']}%")
                    c3.markdown(f"{row['Draw %']}%")
                    c4.markdown(f"{row['A Win %']}%")
                    pred_color = {'Home Win': '#27ae60', 'Draw': '#f39c12', 'Away Win': '#e74c3c'}
                    c5.markdown(f":{pred_color.get(row['Prediction'], '#333')}[**{row['Prediction']}**]")
                st.divider()
    else:
        st.warning("No trained model found. Please run `python cli.py train` first.")

with tab4:
    st.header("Live Tournament")
    st.caption("Enter actual results for played matches to see updated qualification probabilities.")

    groups = world_cup_2026_groups()

    if 'fixed_results' not in st.session_state:
        st.session_state.fixed_results = {g: {} for g in groups}
    if 'live_run' not in st.session_state:
        st.session_state.live_run = False

    for group_name in groups:
        with st.expander(f"Group {group_name} — {', '.join(groups[group_name])}", expanded=False):
            st.subheader(f"Group {group_name}")
            teams = groups[group_name]
            fixtures = [(teams[i], teams[j]) for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]]
            cols = st.columns(6)
            for idx, (t1, t2) in enumerate(fixtures):
                key_t1 = f"g_{group_name}_{t1}_{t2}_t1"
                key_t2 = f"g_{group_name}_{t1}_{t2}_t2"
                key_played = f"g_{group_name}_{t1}_{t2}_played"
                with cols[idx % 3]:
                    st.markdown(f"**{t1}** vs **{t2}**")
                    c1, c2, c3 = st.columns([1, 1, 1])
                    g1_key = f"live_{group_name}_{t1}_{t2}_g1"
                    g2_key = f"live_{group_name}_{t1}_{t2}_g2"
                    with c1:
                        g1 = st.number_input(f"{t1}", min_value=0, max_value=20, value=0, key=g1_key, label_visibility="collapsed")
                    with c2:
                        st.markdown("—")
                    with c3:
                        g2 = st.number_input(f"{t2}", min_value=0, max_value=20, value=0, key=g2_key, label_visibility="collapsed")
                    if g1 > 0 or g2 > 0:
                        st.session_state.fixed_results[group_name][(t1, t2)] = (g1, g2)
                        st.markdown(f"✓ *{g1}-{g2}*", unsafe_allow_html=True)

    if st.button("Update Predictions", type="primary", use_container_width=True):
        if predictor and predictor.is_trained_:
            fixed = {g: v for g, v in st.session_state.fixed_results.items() if v}
            results, standings, qualifiers = predictor.live_simulate(fixed, groups)
            st.session_state.live_results = results
            st.session_state.live_standings = standings
            st.session_state.live_qualifiers = qualifiers
            st.session_state.live_run = True
        else:
            st.warning("No trained model. Run `python cli.py train` first.")

    if st.session_state.get('live_run'):
        results = st.session_state.live_results
        standings = st.session_state.live_standings
        qualifiers = st.session_state.live_qualifiers

        st.subheader("Current Group Standings")
        for group_name in groups:
            st.markdown(f"**Group {group_name}**")
            sdf = standings.get(group_name, pd.DataFrame())
            if not sdf.empty:
                cols_to_show = ['Team', 'Pld', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']
                sdf_display = sdf[cols_to_show] if all(c in sdf.columns for c in cols_to_show) else sdf
                st.dataframe(sdf_display, use_container_width=True, hide_index=True)

        st.subheader("Updated Qualification Probabilities")
        for group_name, data in results.items():
            st.markdown(f"**Group {group_name}**")
            sorted_teams = sorted(data['qualification_probs'].keys(),
                                  key=lambda t: data['qualification_probs'][t], reverse=True)
            table_data = []
            for team in sorted_teams:
                qp = data['qualification_probs'][team] * 100
                tp = data['top_team_probs'][team] * 100
                table_data.append({'Team': team, 'Qualify %': f"{qp:.1f}%", 'Win Group %': f"{tp:.1f}%"})
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        st.subheader("Predicted Qualifiers")
        qual_cols = st.columns(4)
        for i, (group_name, teams) in enumerate(qualifiers.items()):
            with qual_cols[i % 4]:
                st.markdown(f"**Group {group_name}**")
                st.markdown(f"1. {teams[0]}")
                st.markdown(f"2. {teams[1]}")

with tab5:
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
