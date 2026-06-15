"""Main training and prediction pipeline."""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.loader import MatchDataLoader
from src.features.builder import FeatureBuilder
from src.models.ml_models import MLModelFactory
from src.models.statistical import PoissonModel, EloModel, MonteCarloSimulator
from src.models.ensemble import HybridEnsemble
from src.evaluation.metrics import Evaluator
from src.simulation.simulator import TournamentSimulator, world_cup_2026_groups


class FIFAPredictor:
    """End-to-end FIFA match prediction pipeline."""

    def __init__(self, config_path='config/config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.data_loader = MatchDataLoader(self.config)
        self.feature_builder = FeatureBuilder(self.config)
        self.model_factory = MLModelFactory(self.config)
        self.evaluator = Evaluator(self.config)
        self.ensemble = HybridEnsemble(self.config)

        self.X_ = None
        self.y_ = None
        self.match_indices_ = None
        self.is_trained_ = False

    def load_data(self):
        """Load all data sources."""
        print("[1/5] Loading match data...")
        matches = self.data_loader.load_world_cup_matches()
        rankings = self.data_loader.load_fifa_rankings()
        teams = self.data_loader.load_team_registry()
        return matches, rankings, teams

    def build_features(self, matches, rankings, teams):
        """Build feature matrix from raw data."""
        print("[2/5] Building features...")
        X, y, indices, team_pairs = self.feature_builder.build_features(matches, rankings, teams)
        self.X_ = X
        self.y_ = y
        self.match_indices_ = indices
        self.team_pairs_ = team_pairs
        print(f"  → {X.shape[1]} features, {X.shape[0]} samples")
        return X, y

    def train_test_split(self, X, y, test_years=None):
        """Split by tournament year (temporal split)."""
        if test_years is None:
            test_years = self.config['evaluation']['test_tournaments']
        from src.data.loader import MatchDataLoader
        matches = self.data_loader.load_world_cup_matches()
        if isinstance(X, pd.DataFrame):
            match_years = matches.iloc[self.match_indices_]['year'].values
            test_mask = np.isin(match_years, test_years)
            train_mask = ~test_mask
            X_train = X.iloc[train_mask]
            y_train = y[train_mask]
            X_test = X.iloc[test_mask]
            y_test = y[test_mask]
        else:
            match_years = matches.iloc[self.match_indices_]['year'].values
            test_mask = np.isin(match_years, test_years)
            train_mask = ~test_mask
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models and build ensemble."""
        print("[3/5] Training models...")

        xgb = self.model_factory.create_xgboost()
        lgb = self.model_factory.create_lightgbm()
        rf = self.model_factory.create_random_forest()
        nn = self.model_factory.create_neural_network()

        print("  Training XGBoost...")
        xgb.fit(X_train, y_train)
        print("  Training LightGBM...")
        lgb.fit(X_train, y_train)
        print("  Training Random Forest...")
        rf.fit(X_train, y_train)
        print("  Training Neural Network...")
        nn.fit(X_train, y_train)

        poisson = PoissonModel(self.config)
        print("  Fitting Poisson model...")
        from src.data.loader import MatchDataLoader
        from src.models.ensemble import SKLearnWrapper
        matches = self.data_loader.load_world_cup_matches()
        poisson.fit(matches)

        elo = EloModel(self.config)
        print("  Fitting Elo model...")
        elo.fit(matches)

        train_indices = X_train.index if hasattr(X_train, 'index') else range(len(X_train))
        val_indices = X_val.index if X_val is not None and hasattr(X_val, 'index') else []
        all_indices = list(train_indices) + (list(val_indices) if val_indices is not None else [])
        self.team_pairs_map_ = {i: self.team_pairs_[i] for i in all_indices if i < len(self.team_pairs_)}

        poisson_wrapped = SKLearnWrapper(poisson, self.team_pairs_map_)
        elo_wrapped = SKLearnWrapper(elo, self.team_pairs_map_)

        self.ensemble.add_model('xgboost', xgb)
        self.ensemble.add_model('lightgbm', lgb)
        self.ensemble.add_model('random_forest', rf)
        self.ensemble.add_model('neural_network', nn)
        self.ensemble.add_model('poisson', poisson_wrapped)
        self.ensemble.add_model('elo', elo_wrapped)

        print("  Training stacked ensemble...")
        self.ensemble.fit(
            pd.DataFrame(X_train), pd.Series(y_train),
            pd.DataFrame(X_val) if X_val is not None else None,
            pd.Series(y_val) if y_val is not None else None,
        )

        self.is_trained_ = True
        print("  ✓ Ensemble trained successfully")
        return self.ensemble

    def evaluate(self, X_test, y_test, tournament='Test'):
        """Evaluate all models on test data."""
        print("[4/5] Evaluating models...")
        results = []

        for name, model in self.ensemble.base_models_.items():
            try:
                y_proba = model.predict_proba(pd.DataFrame(X_test))
                y_pred = np.argmax(y_proba, axis=1)
                result = self.evaluator.evaluate(y_test, y_pred, y_proba, model_name=name, tournament=tournament)
                self.evaluator.print_report(result)
                results.append(result)
            except Exception as e:
                print(f"  ⚠ {name} failed: {e}")

        y_proba_ens = self.ensemble.predict_proba(pd.DataFrame(X_test))
        y_pred_ens = self.ensemble.predict(pd.DataFrame(X_test))
        result_ens = self.evaluator.evaluate(y_test, y_pred_ens, y_proba_ens,
                                              model_name='HybridEnsemble', tournament=tournament)
        self.evaluator.print_report(result_ens)
        results.append(result_ens)

        self.evaluator.summary_report(save_path='results/comparison.csv')
        return results

    def predict_match(self, team1, team2, neutral=True):
        """Predict a single match outcome."""
        if not self.is_trained_:
            raise RuntimeError("Model not trained yet. Call train() first.")
        dummy = pd.DataFrame([{
            't1_form_points_avg': 1.5, 't2_form_points_avg': 1.2,
            't1_form_goals_scored_avg': 1.8, 't2_form_goals_scored_avg': 1.4,
            't1_form_goals_conceded_avg': 0.8, 't2_form_goals_conceded_avg': 1.1,
            't1_form_gd_avg': 1.0, 't2_form_gd_avg': 0.3,
            't1_form_matches_count': 10, 't2_form_matches_count': 10,
            't1_form_win_pct': 0.6, 't2_form_win_pct': 0.4,
            'rank_diff': 10, 'elo_diff': 50,
            'h2h_team1_wins': 3, 'h2h_team2_wins': 2, 'h2h_draws': 1, 'h2h_total': 6,
            't1_prev_appearances': 20, 't2_prev_appearances': 10,
            't1_defending_champion': 0, 't2_defending_champion': 0,
            'same_confederation': 1, 'is_host': 0, 'is_host_t2': 0,
        }])
        cols = self.X_.columns.tolist() if hasattr(self.X_, 'columns') else []
        if not cols:
            cols = ['t1_form_points_avg', 't1_form_goals_scored_avg', 't1_form_goals_conceded_avg',
                    't1_form_gd_avg', 't1_form_matches_count', 't1_form_win_pct',
                    't2_form_points_avg', 't2_form_goals_scored_avg', 't2_form_goals_conceded_avg',
                    't2_form_gd_avg', 't2_form_matches_count', 't2_form_win_pct',
                    'rank_diff', 'elo_diff',
                    'h2h_team1_wins', 'h2h_team2_wins', 'h2h_draws', 'h2h_total',
                    't1_prev_appearances', 't2_prev_appearances',
                    't1_defending_champion', 't2_defending_champion',
                    'same_confederation', 'is_host', 'is_host_t2']
        for col in cols:
            if col not in dummy.columns:
                dummy[col] = 0
        dummy = dummy[cols]

        probs = self.ensemble.predict_proba(dummy)[0]
        pred = np.argmax(probs)
        outcomes = ['Home Win', 'Draw', 'Away Win']
        return {
            'team1': team1,
            'team2': team2,
            'prediction': outcomes[pred],
            'probabilities': {
                'home_win': float(probs[0]),
                'draw': float(probs[1]),
                'away_win': float(probs[2]),
            },
        }

    def simulate_tournament(self, groups=None):
        """Run full tournament simulation for 2026."""
        print("[5/5] Simulating 2026 World Cup...")
        if groups is None:
            groups = world_cup_2026_groups()
        sim = TournamentSimulator(self.config)
        predictor_fn = self._get_sim_predictor_fn()
        group_objs = sim.build_groups(groups)
        results = sim.run_group_stage(group_objs, predictor_fn)
        sim.print_results(results)
        qualifiers = sim.get_group_standings(results)
        return results, qualifiers

    def _get_sim_predictor_fn(self):
        """Build a fast Poisson+Elo predictor for simulation."""
        poisson_model = None
        elo_model = None
        for name, model in self.ensemble.base_models_.items():
            inner = model.model if hasattr(model, 'model') else model
            if 'poisson' in name.lower():
                poisson_model = inner
            elif 'elo' in name.lower():
                elo_model = inner

        def fn(t1, t2):
            if poisson_model is not None and elo_model is not None:
                return 0.6 * np.array(poisson_model.predict_match_probs(t1, t2)) + 0.4 * np.array(elo_model.predict_match_probs(t1, t2))
            elif poisson_model is not None:
                return np.array(poisson_model.predict_match_probs(t1, t2))
            elif elo_model is not None:
                return np.array(elo_model.predict_match_probs(t1, t2))
            else:
                return np.array([0.45, 0.20, 0.35])
        return fn

    def predict_all_group_matches(self, groups=None):
        """Return a DataFrame with predictions for all 48 group stage matches."""
        if groups is None:
            groups = world_cup_2026_groups()
        sim = TournamentSimulator(self.config)
        predictor_fn = self._get_sim_predictor_fn()
        group_objs = sim.build_groups(groups)
        return sim.all_fixtures(group_objs, predictor_fn)

    def live_simulate(self, fixed_results, groups=None):
        """Simulate remaining matches with some results already played.

        fixed_results: dict { group_name: { (team1, team2): (goals1, goals2) } }
        Returns (results, standings_dfs, qualifiers).
        """
        if groups is None:
            groups = world_cup_2026_groups()
        sim = TournamentSimulator(self.config)
        predictor_fn = self._get_sim_predictor_fn()
        group_objs = sim.build_groups(groups)
        results, standings = sim.run_group_stage_with_fixed(group_objs, fixed_results, predictor_fn)
        qualifiers = sim.get_group_standings(results)
        return results, standings, qualifiers

    def run(self):
        """Run the full pipeline end-to-end."""
        matches, rankings, teams = self.load_data()
        X, y = self.build_features(matches, rankings, teams)
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)
        self.train(X_train, y_train, X_test, y_test)
        self.evaluate(X_test, y_test, tournament='2018+2022')
        self.simulate_tournament()
        return self
