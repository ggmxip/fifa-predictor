"""CLI tool for FIFA match prediction."""

import click
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.pipeline import FIFAPredictor
from src.simulation.simulator import world_cup_2026_groups


@click.group()
def cli():
    """FIFA World Cup Predictor — Hybrid ML + Statistical Model."""


@cli.command()
@click.option('--config', default='config/config.yaml', help='Path to config file')
def train(config):
    """Train the full prediction pipeline."""
    predictor = FIFAPredictor(config)
    predictor.run()
    click.echo("\n✓ Training complete! Model saved to models/hybrid_ensemble.joblib")


@cli.command()
@click.argument('team1')
@click.argument('team2')
@click.option('--config', default='config/config.yaml')
@click.option('--neutral', is_flag=True, help='Neutral venue')
def predict(team1, team2, config, neutral):
    """Predict outcome of a single match."""
    predictor = FIFAPredictor(config)

    ensemble_path = Path('models/hybrid_ensemble.joblib')
    if ensemble_path.exists():
        from src.models.ensemble import HybridEnsemble
        predictor.ensemble = HybridEnsemble.load(ensemble_path)
        predictor.is_trained_ = True
        predictor.X_ = pd.DataFrame()
    else:
        click.echo("No trained model found. Running training pipeline first...")
        matches, rankings, teams = predictor.load_data()
        X, y = predictor.build_features(matches, rankings, teams)
        X_train, _, y_train, _ = predictor.train_test_split(X, y)
        predictor.train(X_train, y_train)

    result = predictor.predict_match(team1, team2, neutral=neutral)

    click.echo(f"\n  {team1} vs {team2}")
    click.echo(f"  {'─' * 30}")
    click.echo(f"  Prediction: {result['prediction']}")
    click.echo(f"  {'─' * 30}")
    click.echo(f"  {team1} win:  {result['probabilities']['home_win']:.1%}")
    click.echo(f"  Draw:        {result['probabilities']['draw']:.1%}")
    click.echo(f"  {team2} win:  {result['probabilities']['away_win']:.1%}")


@cli.command()
@click.option('--config', default='config/config.yaml')
@click.option('--sims', default=10000, help='Number of Monte Carlo simulations')
def simulate(config, sims):
    """Simulate the 2026 World Cup group stage."""
    predictor = FIFAPredictor(config)
    predictor.config['simulation']['n_monte_carlo'] = sims

    ensemble_path = Path('models/hybrid_ensemble.joblib')
    if ensemble_path.exists():
        from src.models.ensemble import HybridEnsemble
        predictor.ensemble = HybridEnsemble.load(ensemble_path)
        predictor.is_trained_ = True
        predictor.X_ = pd.DataFrame()
    else:
        click.echo("No trained model found. Running training pipeline first...")
        matches, rankings, teams = predictor.load_data()
        X, y = predictor.build_features(matches, rankings, teams)
        X_train, _, y_train, _ = predictor.train_test_split(X, y)
        predictor.train(X_train, y_train)

    groups = world_cup_2026_groups()
    results, qualifiers = predictor.simulate_tournament(groups)

    click.echo(f"\n  {'='*50}")
    click.echo(f"  PREDICTED ROUND OF 16 QUALIFIERS")
    click.echo(f"  {'='*50}")
    for group_name, teams in qualifiers.items():
        click.echo(f"  Group {group_name}: {teams[0]} & {teams[1]}")


@cli.command()
@click.option('--config', default='config/config.yaml')
def evaluate(config):
    """Evaluate model on historical tournaments."""
    predictor = FIFAPredictor(config)

    ensemble_path = Path('models/hybrid_ensemble.joblib')
    if ensemble_path.exists():
        from src.models.ensemble import HybridEnsemble
        predictor.ensemble = HybridEnsemble.load(ensemble_path)
        predictor.is_trained_ = True
        matches, rankings, teams = predictor.load_data()
        X, y = predictor.build_features(matches, rankings, teams)
        _, X_test, _, y_test = predictor.train_test_split(X, y)
        predictor.evaluate(X_test, y_test, '2018+2022')
    else:
        click.echo("No trained model found. Running full training pipeline...")
        predictor.run()


@cli.command()
@click.option('--config', default='config/config.yaml')
def fixtures(config):
    """Display all 48 group stage match predictions."""
    predictor = FIFAPredictor(config)

    ensemble_path = Path('models/hybrid_ensemble.joblib')
    if ensemble_path.exists():
        from src.models.ensemble import HybridEnsemble
        predictor.ensemble = HybridEnsemble.load(ensemble_path)
        predictor.is_trained_ = True
        predictor.X_ = pd.DataFrame()
    else:
        click.echo("No trained model found. Run `python cli.py train` first.")
        return

    groups = world_cup_2026_groups()
    fixtures_df = predictor.predict_all_group_matches(groups)

    for group in sorted(fixtures_df['Group'].unique()):
        click.echo(f"\n{'='*55}")
        click.echo(f"  Group {group}")
        click.echo(f"{'='*55}")
        gdf = fixtures_df[fixtures_df['Group'] == group]
        for _, row in gdf.iterrows():
            pred_color = {'Home Win': 'H', 'Draw': 'D', 'Away Win': 'A'}
            click.echo(f"  {row['Team1']:<20} vs {row['Team2']:<20}  "
                       f"{row['H Win %']:>5.1f}% / {row['Draw %']:>5.1f}% / {row['A Win %']:>5.1f}%  "
                       f"→ {pred_color.get(row['Prediction'], '?'):>1}")


@cli.command()
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--config', default='config/config.yaml')
@click.option('--sims', default=500, help='Number of Monte Carlo simulations')
def live(results_file, config, sims):
    """Simulate remaining matches with fixed results from a JSON file.

    JSON format: { "Group": { "TeamA-TeamB": "3-1", ... }, ... }
    """
    import json
    predictor = FIFAPredictor(config)
    predictor.config['simulation']['n_monte_carlo'] = sims

    ensemble_path = Path('models/hybrid_ensemble.joblib')
    if ensemble_path.exists():
        from src.models.ensemble import HybridEnsemble
        predictor.ensemble = HybridEnsemble.load(ensemble_path)
        predictor.is_trained_ = True
        predictor.X_ = pd.DataFrame()
    else:
        click.echo("No trained model found. Run `python cli.py train` first.")
        return

    with open(results_file) as f:
        raw = json.load(f)

    groups = world_cup_2026_groups()
    fixed_results = {}
    for group_name, matches in raw.items():
        fixed_results[group_name] = {}
        for match_key, score_str in matches.items():
            teams = match_key.split('-')
            if len(teams) != 2:
                click.echo(f"  ⚠ Invalid match key: {match_key} (use TeamA-TeamB)")
                continue
            parts = score_str.split('-')
            if len(parts) != 2:
                click.echo(f"  ⚠ Invalid score: {score_str} (use 3-1)")
                continue
            fixed_results[group_name][(teams[0].strip(), teams[1].strip())] = (int(parts[0]), int(parts[1]))

    results, standings, qualifiers = predictor.live_simulate(fixed_results, groups)

    click.echo(f"\n{'='*55}")
    click.echo(f"  LIVE STANDINGS")
    click.echo(f"{'='*55}")
    for group_name, sdf in standings.items():
        click.echo(f"\n  Group {group_name}:")
        if not sdf.empty:
            cols = ['Team', 'Pld', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']
            for _, row in sdf.iterrows():
                vals = {c: row[c] for c in cols if c in sdf.columns}
                click.echo(f"    {row['Team']:<20}  P:{vals.get('Pld','-')}  "
                           f"W:{vals.get('W','-')}  D:{vals.get('D','-')}  L:{vals.get('L','-')}  "
                           f"GF:{vals.get('GF','-')}  GA:{vals.get('GA','-')}  GD:{vals.get('GD','-')}  "
                           f"Pts:{vals.get('Pts','-')}")

    click.echo(f"\n{'='*55}")
    click.echo(f"  UPDATED QUALIFICATION PROBABILITIES")
    click.echo(f"{'='*55}")
    for group_name, data in results.items():
        click.echo(f"\n  Group {group_name}")
        sorted_teams = sorted(data['qualification_probs'].keys(),
                              key=lambda t: data['qualification_probs'][t], reverse=True)
        for team in sorted_teams:
            qp = data['qualification_probs'][team] * 100
            tp = data['top_team_probs'][team] * 100
            click.echo(f"    {team:<20}  Qualify: {qp:>5.1f}%  Win Group: {tp:>5.1f}%")

    click.echo(f"\n{'='*55}")
    click.echo(f"  PREDICTED QUALIFIERS")
    click.echo(f"{'='*55}")
    for group_name, teams in qualifiers.items():
        click.echo(f"  Group {group_name}: {teams[0]} & {teams[1]}")


if __name__ == '__main__':
    cli()
