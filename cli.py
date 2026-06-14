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


if __name__ == '__main__':
    cli()
