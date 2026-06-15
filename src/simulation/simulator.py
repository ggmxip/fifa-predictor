"""Group stage simulator for FIFA World Cup."""

import numpy as np
import pandas as pd
from itertools import combinations


GROUP_FIXTURES = list(combinations(range(4), 2))


class GroupStage:
    """Represents a World Cup group with 4 teams."""

    def __init__(self, name, teams):
        self.name = name
        self.teams = teams
        self.fixtures = [(teams[i], teams[j]) for i, j in GROUP_FIXTURES]

    def simulate(self, predictor_fn, n_simulations=10000):
        qual_probs = {t: 0.0 for t in self.teams}
        top_team_probs = {t: 0.0 for t in self.teams}
        points_dist = {t: np.zeros(10) for t in self.teams}

        for _ in range(n_simulations):
            points, gd, gs = self._run_simulation(predictor_fn)
            standings = sorted(self.teams, key=lambda t: (points[t], gd[t], gs[t]), reverse=True)
            qual_probs[standings[0]] += 1
            qual_probs[standings[1]] += 1
            top_team_probs[standings[0]] += 1
            for t in self.teams:
                if points[t] <= 9:
                    points_dist[t][points[t]] += 1

        for t in self.teams:
            qual_probs[t] /= n_simulations
            top_team_probs[t] /= n_simulations
            points_dist[t] /= n_simulations

        return qual_probs, top_team_probs, points_dist

    def simulate_with_fixed_results(self, fixed_results, predictor_fn, n_simulations=10000):
        """Simulate remaining matches with some results already fixed.

        fixed_results: dict { (team1, team2): (goals1, goals2) } for played matches.
        """
        fixed_set = set(fixed_results.keys())
        remaining = [(t1, t2) for t1, t2 in self.fixtures if (t1, t2) not in fixed_set and (t2, t1) not in fixed_set]

        base_points = {t: 0 for t in self.teams}
        base_gd = {t: 0 for t in self.teams}
        base_gs = {t: 0 for t in self.teams}

        for (t1, t2), (g1, g2) in fixed_results.items():
            base_gs[t1] += g1; base_gs[t2] += g2
            base_gd[t1] += g1 - g2; base_gd[t2] += g2 - g1
            if g1 > g2:
                base_points[t1] += 3
            elif g2 > g1:
                base_points[t2] += 3
            else:
                base_points[t1] += 1; base_points[t2] += 1

        qual_probs = {t: 0.0 for t in self.teams}
        top_team_probs = {t: 0.0 for t in self.teams}
        points_dist = {t: np.zeros(10) for t in self.teams}

        for _ in range(n_simulations):
            points = base_points.copy()
            gd = base_gd.copy()
            gs = base_gs.copy()

            for t1, t2 in remaining:
                probs = predictor_fn(t1, t2)
                result = np.random.choice([0, 1, 2], p=probs)
                if result == 0:
                    points[t1] += 3; gd[t1] += 1; gd[t2] -= 1; gs[t1] += 1
                elif result == 1:
                    points[t1] += 1; points[t2] += 1
                else:
                    points[t2] += 3; gd[t2] += 1; gd[t1] -= 1; gs[t2] += 1

            standings = sorted(self.teams, key=lambda t: (points[t], gd[t], gs[t]), reverse=True)
            qual_probs[standings[0]] += 1
            qual_probs[standings[1]] += 1
            top_team_probs[standings[0]] += 1
            for t in self.teams:
                if points[t] <= 9:
                    points_dist[t][points[t]] += 1

        for t in self.teams:
            qual_probs[t] /= n_simulations
            top_team_probs[t] /= n_simulations
            points_dist[t] /= n_simulations

        return qual_probs, top_team_probs, points_dist

    def compute_standings(self, fixed_results):
        """Compute actual group standings from fixed results.

        fixed_results: dict { (team1, team2): (goals1, goals2) }
        Returns DataFrame with Team, Pld, W, D, L, GF, GA, GD, Pts.
        """
        stats = {t: {'Pld': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0}
                 for t in self.teams}
        for (t1, t2), (g1, g2) in fixed_results.items():
            stats[t1]['Pld'] += 1; stats[t2]['Pld'] += 1
            stats[t1]['GF'] += g1; stats[t1]['GA'] += g2
            stats[t2]['GF'] += g2; stats[t2]['GA'] += g1
            stats[t1]['GD'] += g1 - g2; stats[t2]['GD'] += g2 - g1
            if g1 > g2:
                stats[t1]['W'] += 1; stats[t2]['L'] += 1
                stats[t1]['Pts'] += 3
            elif g2 > g1:
                stats[t2]['W'] += 1; stats[t1]['L'] += 1
                stats[t2]['Pts'] += 3
            else:
                stats[t1]['D'] += 1; stats[t2]['D'] += 1
                stats[t1]['Pts'] += 1; stats[t2]['Pts'] += 1

        df = pd.DataFrame.from_dict(stats, orient='index')
        df.index.name = 'Team'
        df = df.reset_index().sort_values(['Pts', 'GD', 'GF'], ascending=False)
        return df

    def _run_simulation(self, predictor_fn):
        points = {t: 0 for t in self.teams}
        gd = {t: 0 for t in self.teams}
        gs = {t: 0 for t in self.teams}
        for t1, t2 in self.fixtures:
            probs = predictor_fn(t1, t2)
            result = np.random.choice([0, 1, 2], p=probs)
            if result == 0:
                points[t1] += 3; gd[t1] += 1; gd[t2] -= 1; gs[t1] += 1
            elif result == 1:
                points[t1] += 1; points[t2] += 1
            else:
                points[t2] += 3; gd[t2] += 1; gd[t1] -= 1; gs[t2] += 1
        return points, gd, gs


class TournamentSimulator:
    """Full tournament simulator for World Cup."""

    def __init__(self, config):
        self.config = config
        self.n_simulations = config['simulation']['n_monte_carlo']

    def build_groups(self, teams_dict):
        return {name: GroupStage(name, teams) for name, teams in teams_dict.items()}

    def run_group_stage(self, groups, predictor_fn):
        results = {}
        for name, group in groups.items():
            qual, top, points = group.simulate(predictor_fn, self.n_simulations)
            results[name] = {
                'qualification_probs': qual,
                'top_team_probs': top,
                'points_distribution': points,
            }
        return results

    def run_group_stage_with_fixed(self, groups, all_fixed_results, predictor_fn):
        """Run simulation with fixed results across all groups.

        all_fixed_results: dict { group_name: { (team1, team2): (g1, g2) } }
        """
        results = {}
        standings = {}
        for name, group in groups.items():
            fixed = all_fixed_results.get(name, {})
            qual, top, points = group.simulate_with_fixed_results(fixed, predictor_fn, self.n_simulations)
            results[name] = {
                'qualification_probs': qual,
                'top_team_probs': top,
                'points_distribution': points,
            }
            standings[name] = group.compute_standings(fixed)
        return results, standings

    def all_fixtures(self, groups, predictor_fn):
        """Return a list of all group stage match predictions."""
        rows = []
        for name, group in groups.items():
            for t1, t2 in group.fixtures:
                probs = predictor_fn(t1, t2)
                pred = ['Home Win', 'Draw', 'Away Win'][np.argmax(probs)]
                rows.append({
                    'Group': name,
                    'Team1': t1,
                    'Team2': t2,
                    'H Win %': round(probs[0] * 100, 1),
                    'Draw %': round(probs[1] * 100, 1),
                    'A Win %': round(probs[2] * 100, 1),
                    'Prediction': pred,
                })
        return pd.DataFrame(rows)

    def print_results(self, results):
        for group_name, data in results.items():
            print(f"\n{'='*50}")
            print(f"  Group {group_name}")
            print(f"{'='*50}")
            print(f"{'Team':<20} {'Qualify %':<12} {'Win Group %':<14} {'Avg Pts':<10}")
            print("-" * 56)
            sorted_teams = sorted(data['qualification_probs'].keys(),
                                  key=lambda t: data['qualification_probs'][t], reverse=True)
            for team in sorted_teams:
                qp = data['qualification_probs'][team] * 100
                tp = data['top_team_probs'][team] * 100
                pd_dist = data['points_distribution'][team]
                avg_pts = sum(i * pd_dist[i] for i in range(10))
                print(f"{team:<20} {qp:<12.1f} {tp:<14.1f} {avg_pts:<10.2f}")

    def get_group_standings(self, results):
        qualifiers = {}
        for group_name, data in results.items():
            sorted_teams = sorted(data['qualification_probs'].keys(),
                                  key=lambda t: data['qualification_probs'][t], reverse=True)
            qualifiers[group_name] = sorted_teams[:2]
        return qualifiers


def world_cup_2026_groups():
    return {
        'A': ['Mexico', 'South Korea', 'Czech Republic', 'South Africa'],
        'B': ['Switzerland', 'Canada', 'Qatar', 'Bosnia and Herzegovina'],
        'C': ['Scotland', 'Morocco', 'Brazil', 'Haiti'],
        'D': ['United States', 'Australia', 'Turkey', 'Paraguay'],
        'E': ['Germany', 'Ecuador', 'Ivory Coast', 'Curacao'],
        'F': ['Japan', 'Netherlands', 'Sweden', 'Tunisia'],
        'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
        'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
        'I': ['France', 'Senegal', 'Iraq', 'Norway'],
        'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
        'K': ['Portugal', 'Congo DR', 'Uzbekistan', 'Colombia'],
        'L': ['England', 'Croatia', 'Ghana', 'Panama'],
    }
