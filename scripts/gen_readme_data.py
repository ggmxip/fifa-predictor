"""Generate prediction data for README."""

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
out = {}
out['fixtures'] = []
for _, row in fixtures.iterrows():
    pred_short = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}
    out['fixtures'].append({
        'group': row['Group'], 't1': row['Team1'], 't2': row['Team2'],
        'h': row['H Win %'], 'd': row['Draw %'], 'a': row['A Win %'],
        'pred': pred_short[row['Prediction']],
    })

results, qualifiers = p.simulate_tournament(world_cup_2026_groups())
out['simulation'] = {}
for g, data in results.items():
    sorted_teams = sorted(data['qualification_probs'].keys(),
                          key=lambda t: data['qualification_probs'][t], reverse=True)
    out['simulation'][g] = []
    for t in sorted_teams:
        qp = round(data['qualification_probs'][t] * 100, 1)
        tp = round(data['top_team_probs'][t] * 100, 1)
        dist = data['points_distribution'][t]
        avg_pts = round(sum(i * dist[i] for i in range(10)), 2)
        out['simulation'][g].append({'team': t, 'qual_pct': qp, 'top_pct': tp, 'avg_pts': avg_pts})

out['qualifiers'] = {g: ts for g, ts in qualifiers.items()}

csv_path = Path('results/comparison.csv')
if csv_path.exists():
    df = pd.read_csv(csv_path)
    out['metrics'] = []
    for _, r in df.iterrows():
        out['metrics'].append({
            'model': r['model'], 'accuracy': round(r['accuracy'] * 100, 2),
            'log_loss': round(r['log_loss'], 4), 'brier': round(r['brier_score'], 4),
        })

import json
print(json.dumps(out, indent=2))
