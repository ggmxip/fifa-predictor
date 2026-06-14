"""Feature engineering for match prediction."""

import pandas as pd
import numpy as np
from pathlib import Path


class FeatureBuilder:
    """Builds feature matrices from raw match data."""

    def __init__(self, config):
        self.config = config
        self.window = config['features']['rolling_window']
        self.decay = config['features']['recency_weight_decay']

    def build_features(self, matches_df, rankings_df, team_registry):
        """Build complete feature matrix for all matches."""
        matches = matches_df.sort_values('date').copy()
        all_teams = set(matches['team1'].unique()) | set(matches['team2'].unique())

        team_form = self._compute_form_features(matches, all_teams)
        ranking_features = self._compute_ranking_features(matches, rankings_df)
        h2h_features = self._compute_head_to_head(matches)
        tournament_features = self._compute_tournament_history(matches, team_registry)

        X_rows = []
        y_rows = []
        team_pairs = []

        for _, match in matches.iterrows():
            features = {}

            features.update(self._get_form_features(team_form, match['team1'], match['date']))
            features.update(self._get_form_features(team_form, match['team2'], match['date']))

            for prefix in ['team1_', 'team2_']:
                for key, val in features.copy().items():
                    if key.startswith(prefix):
                        pass
            t1_form = {f't1_{k}': v for k, v in self._get_form_features(team_form, match['team1'], match['date']).items()}
            t2_form = {f't2_{k}': v for k, v in self._get_form_features(team_form, match['team2'], match['date']).items()}
            features = {**t1_form, **t2_form}

            rank_feat = self._get_ranking_features(ranking_features, match['team1'], match['team2'], match['date'])
            features.update(rank_feat)

            h2h = self._get_h2h_features(h2h_features, match['team1'], match['team2'], match['date'])
            features.update(h2h)

            tourn = self._get_tournament_features(tournament_features, match['team1'], match['team2'], match['year'])
            features.update(tourn)

            conf = team_registry.set_index('team').to_dict('index')
            f1_conf = conf.get(match['team1'], {})
            f2_conf = conf.get(match['team2'], {})
            features['same_confederation'] = 1 if f1_conf.get('confederation') == f2_conf.get('confederation') else 0
            features['is_host'] = 1 if match['team1'] == match['host'] else 0
            features['is_host_t2'] = 1 if match['team2'] == match['host'] else 0

            X_rows.append(features)
            y_rows.append(match['label'])
            team_pairs.append((match['team1'], match['team2']))

        X = pd.DataFrame(X_rows)
        X = X.fillna(0)
        y = np.array(y_rows)

        return X, y, matches.index.tolist(), team_pairs

    def _compute_form_features(self, matches, all_teams):
        """Compute rolling form for each team over time."""
        form = {}
        for team in all_teams:
            team_matches = []
            for _, row in matches.iterrows():
                if row['team1'] == team:
                    goals_for = row['goals1']
                    goals_against = row['goals2']
                    result = 3 if goals_for > goals_against else (1 if goals_for == goals_against else 0)
                    team_matches.append({'date': row['date'], 'goals_for': goals_for, 'goals_against': goals_against, 'points': result})
                elif row['team2'] == team:
                    goals_for = row['goals2']
                    goals_against = row['goals1']
                    result = 3 if goals_for > goals_against else (1 if goals_for == goals_against else 0)
                    team_matches.append({'date': row['date'], 'goals_for': goals_for, 'goals_against': goals_against, 'points': result})
            form[team] = pd.DataFrame(team_matches).sort_values('date') if team_matches else pd.DataFrame()
        return form

    def _get_form_features(self, form, team, match_date):
        """Get rolling form features for a team before a given date."""
        features = {
            'form_points_avg': 0, 'form_goals_scored_avg': 0, 'form_goals_conceded_avg': 0,
            'form_gd_avg': 0, 'form_matches_count': 0, 'form_win_pct': 0,
        }
        if team not in form or form[team].empty:
            return features

        tm = form[team][form[team]['date'] < match_date].tail(self.window)
        if tm.empty:
            return features

        weights = np.array([self.decay ** i for i in range(len(tm))][::-1])
        features['form_points_avg'] = float(np.average(tm['points'], weights=weights))
        features['form_goals_scored_avg'] = float(np.average(tm['goals_for'], weights=weights))
        features['form_goals_conceded_avg'] = float(np.average(tm['goals_against'], weights=weights))
        features['form_gd_avg'] = features['form_goals_scored_avg'] - features['form_goals_conceded_avg']
        features['form_matches_count'] = len(tm)
        features['form_win_pct'] = (tm['points'] == 3).mean()
        return features

    def _compute_ranking_features(self, matches, rankings):
        return rankings

    def _get_ranking_features(self, rankings, team1, team2, match_date):
        features = {'rank_diff': 0, 'elo_diff': 0}
        if rankings is None or rankings.empty:
            return features
        r1 = rankings[(rankings['team'] == team1) & (rankings['date'] <= match_date)]
        r2 = rankings[(rankings['team'] == team2) & (rankings['date'] <= match_date)]
        if not r1.empty and not r2.empty:
            r1 = r1.iloc[-1]
            r2 = r2.iloc[-1]
            features['rank_diff'] = float(r2['rank'] - r1['rank'])
            features['elo_diff'] = float(r1.get('points', 0) - r2.get('points', 0))
        return features

    def _compute_head_to_head(self, matches):
        h2h = {}
        for _, row in matches.iterrows():
            key = tuple(sorted([row['team1'], row['team2']]))
            if key not in h2h:
                h2h[key] = []
            result = 0 if row['goals1'] > row['goals2'] else (1 if row['goals1'] == row['goals2'] else 2)
            if key == (row['team1'], row['team2']):
                h2h[key].append({'date': row['date'], 'result': result, 'team1': row['team1'], 'goals1': row['goals1'], 'goals2': row['goals2']})
            else:
                flipped = 0 if result == 2 else (2 if result == 0 else 1)
                h2h[key].append({'date': row['date'], 'result': flipped, 'team1': row['team2'], 'goals1': row['goals2'], 'goals2': row['goals1']})
        return {k: pd.DataFrame(v).sort_values('date') for k, v in h2h.items()}

    def _get_h2h_features(self, h2h, team1, team2, match_date):
        features = {'h2h_team1_wins': 0, 'h2h_team2_wins': 0, 'h2h_draws': 0, 'h2h_total': 0}
        key = tuple(sorted([team1, team2]))
        if key not in h2h or h2h[key].empty:
            return features
        past = h2h[key][h2h[key]['date'] < match_date]
        if past.empty:
            return features
        features['h2h_total'] = len(past)
        features['h2h_team1_wins'] = int((past['result'] == 0).sum())
        features['h2h_team2_wins'] = int((past['result'] == 2).sum())
        features['h2h_draws'] = int((past['result'] == 1).sum())
        return features

    def _compute_tournament_history(self, matches, team_registry):
        history = {}
        for _, row in matches.iterrows():
            for team in [row['team1'], row['team2']]:
                if team not in history:
                    history[team] = []
                history[team].append(row['year'])
        return {team: sorted(set(years)) for team, years in history.items()}

    def _get_tournament_features(self, history, team1, team2, year):
        features = {
            't1_prev_appearances': 0, 't2_prev_appearances': 0,
            't1_defending_champion': 0, 't2_defending_champion': 0,
        }
        t1_years = [y for y in history.get(team1, []) if y < year]
        t2_years = [y for y in history.get(team2, []) if y < year]
        features['t1_prev_appearances'] = len(t1_years)
        features['t2_prev_appearances'] = len(t2_years)
        return features
