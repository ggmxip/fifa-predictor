"""Statistical models: Poisson, Elo, Monte Carlo simulation."""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from itertools import combinations


class PoissonModel:
    """Bivariate Poisson model for predicting match scores."""

    def __init__(self, config):
        self.config = config
        self.team_attack_ = {}
        self.team_defense_ = {}
        self.home_advantage_ = 0
        self.target_mean_ = None

    def fit(self, matches_df):
        """Fit attack/defense parameters using MLE."""
        teams = list(set(matches_df['team1'].unique()) | set(matches_df['team2'].unique()))
        n_teams = len(teams)
        team_to_idx = {t: i for i, t in enumerate(teams)}

        goals1 = matches_df['goals1'].values
        goals2 = matches_df['goals2'].values
        home_idx = np.array([team_to_idx[t] for t in matches_df['team1']])
        away_idx = np.array([team_to_idx[t] for t in matches_df['team2']])

        self.target_mean_ = np.mean([goals1.mean(), goals2.mean()])

        def neg_log_likelihood(params):
            mu = params[0]
            attack = np.exp(params[1:1+n_teams])
            defense = np.exp(params[1+n_teams:1+2*n_teams])
            home = np.exp(params[-1])

            pred_home = mu * attack[home_idx] * defense[away_idx] * home
            pred_away = mu * attack[away_idx] * defense[home_idx]

            pred_home = np.clip(pred_home, 0.01, None)
            pred_away = np.clip(pred_away, 0.01, None)

            ll = poisson.logpmf(goals1, pred_home).sum() + poisson.logpmf(goals2, pred_away).sum()
            return -ll

        init_params = np.zeros(1 + 2 * n_teams + 1)
        init_params[0] = np.log(self.target_mean_)
        init_params[-1] = np.log(1.3)

        result = minimize(neg_log_likelihood, init_params, method='L-BFGS-B', options={'maxiter': 1000})
        opt = result.x

        self.home_advantage_ = np.exp(opt[-1])
        for i, team in enumerate(teams):
            self.team_attack_[team] = np.exp(opt[1+i])
            self.team_defense_[team] = np.exp(opt[1+n_teams+i])

        return self

    def predict_match_probs(self, team1, team2, neutral=True):
        """Predict win/draw/loss probabilities."""
        mu = self.target_mean_
        att1 = self.team_attack_.get(team1, 1.0)
        def1 = self.team_defense_.get(team1, 1.0)
        att2 = self.team_attack_.get(team2, 1.0)
        def2 = self.team_defense_.get(team2, 1.0)
        home = 1.0 if neutral else self.home_advantage_

        lambda_home = mu * att1 * def2 * home
        lambda_away = mu * att2 * def1

        max_goals = 10
        prob_matrix = np.outer(
            poisson.pmf(np.arange(max_goals + 1), lambda_home),
            poisson.pmf(np.arange(max_goals + 1), lambda_away),
        )

        p_team1 = np.tril(prob_matrix, -1).sum()
        p_draw = np.trace(prob_matrix)
        p_team2 = np.triu(prob_matrix, 1).sum()
        total = p_team1 + p_draw + p_team2

        return np.array([p_team1 / total, p_draw / total, p_team2 / total])


class EloModel:
    """Elo rating system for international football."""

    def __init__(self, config):
        self.config = config
        self.ratings_ = {}
        self.k_factor = config['models']['elo']['k_factor']
        self.initial_rating = config['models']['elo']['initial_rating']
        self.home_advantage = config['models']['elo']['home_advantage']
        self.regression_factor = config['models']['elo']['regression_factor']

    def fit(self, matches_df):
        """Fit Elo ratings by iterating through matches chronologically."""
        matches = matches_df.sort_values('date')

        for _, match in matches.iterrows():
            t1, t2 = match['team1'], match['team2']
            g1, g2 = match['goals1'], match['goals2']

            if t1 not in self.ratings_:
                self.ratings_[t1] = self.initial_rating
            if t2 not in self.ratings_:
                self.ratings_[t2] = self.initial_rating

            expected1 = 1 / (1 + 10 ** ((self.ratings_[t2] - self.ratings_[t1] - self.home_advantage) / 400))
            expected2 = 1 - expected1

            goal_diff = abs(g1 - g2)
            if g1 > g2:
                actual1, actual2 = 1, 0
                diff = max(abs(self.ratings_[t1] - self.ratings_[t2]), 1)
                margin = np.log(goal_diff + 1) * (2.2 / diff * 0.001 + 2.2)
            elif g2 > g1:
                actual1, actual2 = 0, 1
                diff = max(abs(self.ratings_[t2] - self.ratings_[t1]), 1)
                margin = np.log(goal_diff + 1) * (2.2 / diff * 0.001 + 2.2)
            else:
                actual1, actual2 = 0.5, 0.5
                margin = 1

            k = self.k_factor * margin
            self.ratings_[t1] += k * (actual1 - expected1)
            self.ratings_[t2] += k * (actual2 - expected2)

        return self

    def predict_match_probs(self, team1, team2, neutral=True):
        """Predict win/draw/loss probabilities from Elo ratings."""
        r1 = self.ratings_.get(team1, self.initial_rating)
        r2 = self.ratings_.get(team2, self.initial_rating)
        ha = 0 if neutral else self.home_advantage

        expected1 = 1 / (1 + 10 ** ((r2 - r1 - ha) / 400))
        expected2 = 1 - expected1

        draw_prob = 0.25 * (1 - abs(expected1 - expected2))
        p1 = expected1 * (1 - draw_prob)
        p2 = expected2 * (1 - draw_prob)

        total = p1 + draw_prob + p2
        return np.array([p1 / total, draw_prob / total, p2 / total])


class MonteCarloSimulator:
    """Monte Carlo simulation for tournament outcomes."""

    def __init__(self, config, base_predictor):
        self.config = config
        self.n_sims = config['simulation']['n_monte_carlo']
        self.base_predictor = base_predictor

    def simulate_match(self, team1, team2, neutral=True):
        """Simulate a single match outcome from probabilities."""
        probs = self.base_predictor(team1, team2, neutral)
        return np.random.choice([0, 1, 2], p=probs)

    def simulate_group(self, teams, fixtures):
        """Simulate a full group stage and return standings."""
        points = {t: 0 for t in teams}
        gd = {t: 0 for t in teams}
        gs = {t: 0 for t in teams}

        for t1, t2 in fixtures:
            result = self.simulate_match(t1, t2, neutral=True)
            if result == 0:
                points[t1] += 3
                gd[t1] += 1
                gd[t2] -= 1
                gs[t1] += 1
            elif result == 1:
                points[t1] += 1
                points[t2] += 1
            else:
                points[t2] += 3
                gd[t2] += 1
                gd[t1] -= 1
                gs[t2] += 1

        standings = sorted(teams, key=lambda t: (points[t], gd[t], gs[t]), reverse=True)
        return standings, points, gd

    def simulate_group_stage(self, groups):
        """Simulate all groups and return qualifiers."""
        qualifiers = {}
        for group_name, (teams, fixtures) in groups.items():
            standings, _, _ = self.simulate_group(teams, fixtures)
            qualifiers[group_name] = standings[:2]
        return qualifiers
