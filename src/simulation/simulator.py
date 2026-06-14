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
        """Simulate group thousands of times and return qualification probabilities."""
        qual_probs = {t: 0.0 for t in self.teams}
        top_team_probs = {t: 0.0 for t in self.teams}
        points_dist = {t: np.zeros(10) for t in self.teams}

        for _ in range(n_simulations):
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


class TournamentSimulator:
    """Full tournament simulator for World Cup."""

    def __init__(self, config):
        self.config = config
        self.n_simulations = config['simulation']['n_monte_carlo']

    def build_groups(self, teams_dict):
        """Build GroupStage objects from a dict of {group_name: [team_names]}."""
        return {name: GroupStage(name, teams) for name, teams in teams_dict.items()}

    def run_group_stage(self, groups, predictor_fn):
        """Run group stage simulation for all groups."""
        results = {}
        for name, group in groups.items():
            qual, top, points = group.simulate(predictor_fn, self.n_simulations)
            results[name] = {
                'qualification_probs': qual,
                'top_team_probs': top,
                'points_distribution': points,
            }
        return results

    def print_results(self, results):
        """Pretty-print simulation results."""
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
        """Return predicted qualifying teams based on simulation."""
        qualifiers = {}
        for group_name, data in results.items():
            sorted_teams = sorted(data['qualification_probs'].keys(),
                                  key=lambda t: data['qualification_probs'][t], reverse=True)
            qualifiers[group_name] = sorted_teams[:2]
        return qualifiers


def world_cup_2026_groups():
    """Return projected 2026 World Cup groups."""
    return {
        'A': ['Mexico', 'United States', 'Canada', 'New Zealand'],
        'B': ['England', 'Iran', 'Scotland', 'Wales'],
        'C': ['Argentina', 'Chile', 'Nigeria', 'South Korea'],
        'D': ['France', 'Netherlands', 'Senegal', 'Ecuador'],
        'E': ['Spain', 'Croatia', 'Japan', 'Costa Rica'],
        'F': ['Brazil', 'Switzerland', 'Serbia', 'Cameroon'],
        'G': ['Germany', 'Denmark', 'Uruguay', 'Saudi Arabia'],
        'H': ['Portugal', 'Belgium', 'Ghana', 'Australia'],
        'I': ['Italy', 'Poland', 'Morocco', 'Qatar'],
        'J': ['Colombia', 'Sweden', 'Ivory Coast', 'United Arab Emirates'],
        'K': ['Czech Republic', 'Peru', 'Algeria', 'Panama'],
        'L': ['Turkey', 'Ukraine', 'Tunisia', 'Paraguay'],
    }
