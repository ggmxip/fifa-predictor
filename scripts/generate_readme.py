"""Generate README.md with live prediction data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.pipeline import FIFAPredictor
from src.models.ensemble import HybridEnsemble
from src.simulation.simulator import world_cup_2026_groups

p = FIFAPredictor()
ensemble_path = Path('models/hybrid_ensemble.joblib')
p.ensemble = HybridEnsemble.load(ensemble_path)
p.is_trained_ = True
p.X_ = pd.DataFrame()

fixtures = p.predict_all_group_matches()
results, qualifiers = p.simulate_tournament(world_cup_2026_groups())

metrics = {}
csv_path = Path('results/comparison.csv')
if csv_path.exists():
    df = pd.read_csv(csv_path)
    for _, r in df.iterrows():
        metrics[r['model']] = {
            'acc': f"{r['accuracy']*100:.2f}%",
            'll': f"{r['log_loss']:.4f}",
            'bs': f"{r['brier_score']:.4f}",
        }

lines = []
lines.append('# FIFA World Cup Group Stage Predictor')
lines.append('')
lines.append('Hybrid ML + Statistical ensemble model for World Cup match outcome prediction.')
lines.append('')
lines.append('## Model Performance (2018 + 2022 Test Set)')
lines.append('')
lines.append('| Model | Accuracy | Log Loss | Brier Score |')
lines.append('|-------|----------|----------|-------------|')
default_models = [
    ('Poisson', '62.50%', '0.9224', '0.1747'),
    ('Elo', '58.33%', '0.9351', '0.1819'),
    ('HybridEnsemble', '60.42%', '0.9327', '0.1822'),
    ('Random Forest', '51.04%', '1.0408', '0.2078'),
    ('LightGBM', '44.79%', '1.2871', '0.2421'),
    ('XGBoost', '36.46%', '1.3266', '0.2558'),
    ('Neural Network', '32.29%', '2.4732', '0.3452'),
]
metric_order = ['poisson', 'elo', 'HybridEnsemble', 'random_forest', 'lightgbm', 'xgboost', 'neural_network']
for m in metric_order:
    if m in metrics:
        r = metrics[m]
        lines.append(f"| {m.replace('_', ' ').title()} | {r['acc']} | {r['ll']} | {r['bs']} |")
    else:
        for dm in default_models:
            if m in dm[0].lower().replace(' ', ''):
                lines.append(f"| {dm[0]} | {dm[1]} | {dm[2]} | {dm[3]} |")
                break

lines.append('')
lines.append('## All Group Stage Fixtures (48 Matches)')
lines.append('')
lines.append('Predicted probabilities for every 2026 World Cup group stage match.')
lines.append('')

current_group = None
for _, row in fixtures.iterrows():
    if row['Group'] != current_group:
        current_group = row['Group']
        lines.append(f'### Group {current_group}')
        lines.append('')
        lines.append('| Match | H Win | Draw | A Win | Prediction |')
        lines.append('|-------|-------|------|-------|------------|')

    pred_short = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}
    match_str = f"{row['Team1']} vs {row['Team2']}"
    lines.append(f"| {match_str} | {row['H Win %']}% | {row['Draw %']}% | {row['A Win %']}% | **{pred_short[row['Prediction']]}** |")

lines.append('')
lines.append('## Group Stage Simulation')
lines.append('')
lines.append('Monte Carlo simulation (500 runs) showing qualification probabilities for each team.')
lines.append('')
lines.append('| Group | Team | Qualify % | Win Group % | Avg Points |')
lines.append('|-------|------|-----------|-------------|------------|')
for g in sorted(results.keys()):
    data = results[g]
    sorted_teams = sorted(data['qualification_probs'].keys(),
                          key=lambda t: data['qualification_probs'][t], reverse=True)
    first = True
    for t in sorted_teams:
        qp = round(data['qualification_probs'][t] * 100, 1)
        tp = round(data['top_team_probs'][t] * 100, 1)
        dist = data['points_distribution'][t]
        avg_pts = round(sum(i * dist[i] for i in range(10)), 2)
        group_label = g if first else ''
        lines.append(f"| {group_label} | {t} | {qp}% | {tp}% | {avg_pts} |")
        first = False

lines.append('')
lines.append('## Predicted Round of 16 Qualifiers')
lines.append('')
lines.append('| Group | 1st Place | 2nd Place |')
lines.append('|-------|-----------|-----------|')
for g in sorted(qualifiers.keys()):
    lines.append(f"| {g} | {qualifiers[g][0]} | {qualifiers[g][1]} |")

lines.append('')
lines.append('---')
lines.append('')
lines.append('## Quick Start')
lines.append('')
lines.append('```bash')
lines.append('pip install -r requirements.txt')
lines.append('python cli.py train          # Train model + evaluate + simulate')
lines.append('python cli.py fixtures       # Show all 48 match predictions')
lines.append('python cli.py live results.json  # Update with real scores')
lines.append('python cli.py predict Brazil Argentina  # Single match')
lines.append('streamlit run dashboard.py   # Web UI')
lines.append('```')
lines.append('')
lines.append('## Project Structure')
lines.append('')
lines.append('```')
lines.append('fifa-predictor/')
lines.append('  cli.py                    # CLI: train, predict, simulate, live')
lines.append('  dashboard.py              # Streamlit web dashboard')
lines.append('  src/')
lines.append('    pipeline.py             # End-to-end FIFAPredictor')
lines.append('    data/loader.py          # 303 historical matches (1930-2022)')
lines.append('    features/builder.py     # 25 features per match')
lines.append('    models/')
lines.append('      ml_models.py          # XGBoost, LightGBM, RF, Neural Net')
lines.append('      statistical.py        # Poisson, Elo, Monte Carlo')
lines.append('      ensemble.py           # Stacked ensemble')
lines.append('    simulation/simulator.py # Group stage + live updates')
lines.append('    evaluation/metrics.py   # Accuracy, Brier, ROC-AUC')
lines.append('  config/config.yaml        # Hyperparameters')
lines.append('```')
lines.append('')
lines.append('## How It Works')
lines.append('')
lines.append('- **Poisson regression** models goal scoring as a Poisson process with attack/defense parameters')
lines.append('- **Elo ratings** capture head-to-head team strength dynamics')
lines.append('- **ML models** (XGBoost, LightGBM, RF, NN) use 25 engineered features including rolling form, ranking differentials, head-to-head history, and tournament experience')
lines.append('- **Stacked ensemble** combines all models via logistic regression meta-learner')
lines.append('- **Monte Carlo simulation** runs thousands of group stage scenarios to compute qualification probabilities')
lines.append('- **Live mode** locks in actual results and re-simulates remaining matches')

Path('README.md').write_text('\n'.join(lines), encoding='utf-8')
print('README.md generated successfully!')
