"""Data loading and preparation for FIFA match data."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


ALL_RESULTS = {
    1930: {
        'host': 'Uruguay',
        'groups': {
            '1': [('Argentina','France','1-0'),('Argentina','Mexico','6-3'),('Argentina','Chile','3-1'),
                  ('France','Mexico','4-1'),('France','Chile','0-1'),('Mexico','Chile','0-3')],
            '2': [('Brazil','Bolivia','4-0'),('Brazil','Yugoslavia','1-2'),('Yugoslavia','Bolivia','4-0')],
            '3': [('Uruguay','Peru','1-0'),('Uruguay','Romania','4-0'),('Peru','Romania','1-3')],
            '4': [('United States','Belgium','3-0'),('United States','Paraguay','3-0'),('Paraguay','Belgium','1-0')],
        }
    },
    1934: {'host': 'Italy', 'groups': {}},  # knockout only
    1938: {'host': 'France', 'groups': {}},
    1950: {'host': 'Brazil', 'groups': {}},
    1954: {'host': 'Switzerland', 'groups': {}},
    1958: {'host': 'Sweden', 'groups': {}},
    1962: {'host': 'Chile', 'groups': {}},
    1966: {'host': 'England', 'groups': {}},
    1970: {'host': 'Mexico', 'groups': {}},
    1974: {'host': 'West Germany', 'groups': {}},
    1978: {'host': 'Argentina', 'groups': {}},
    1982: {'host': 'Spain', 'groups': {}},
    1986: {'host': 'Mexico', 'groups': {}},
    1990: {'host': 'Italy', 'groups': {}},
    1994: {'host': 'United States', 'groups': {}},
    1998: {'host': 'France', 'groups': {}},
    2002: {'host': 'South Korea/Japan', 'groups': {
        'A': [('Denmark','Uruguay','2-1'),('Senegal','France','1-0'),('Denmark','Senegal','1-1'),('France','Uruguay','0-0'),('Denmark','France','2-0'),('Senegal','Uruguay','3-3')],
        'B': [('Paraguay','South Africa','2-2'),('Spain','Slovenia','3-1'),('Spain','Paraguay','3-1'),('South Africa','Slovenia','1-0'),('South Africa','Spain','2-3'),('Slovenia','Paraguay','1-3')],
        'C': [('Brazil','Turkey','2-1'),('China','Costa Rica','0-2'),('Brazil','China','4-0'),('Costa Rica','Turkey','1-1'),('Costa Rica','Brazil','2-5'),('Turkey','China','3-0')],
        'D': [('South Korea','Poland','2-0'),('United States','Portugal','3-2'),('South Korea','United States','1-1'),('Portugal','Poland','4-0'),('Portugal','South Korea','0-1'),('Poland','United States','3-1')],
        'E': [('Republic of Ireland','Cameroon','1-1'),('Germany','Saudi Arabia','8-0'),('Germany','Republic of Ireland','1-1'),('Cameroon','Saudi Arabia','1-0'),('Cameroon','Germany','0-2'),('Saudi Arabia','Republic of Ireland','0-3')],
        'F': [('Argentina','Nigeria','1-0'),('England','Sweden','1-1'),('Sweden','Nigeria','2-1'),('Argentina','England','0-1'),('Sweden','Argentina','1-1'),('Nigeria','England','0-0')],
        'G': [('Croatia','Mexico','0-1'),('Italy','Ecuador','2-0'),('Italy','Croatia','1-2'),('Mexico','Ecuador','2-1'),('Mexico','Italy','1-1'),('Ecuador','Croatia','1-0')],
        'H': [('Japan','Belgium','2-2'),('Russia','Tunisia','2-0'),('Japan','Russia','1-0'),('Tunisia','Belgium','1-1'),('Tunisia','Japan','0-2'),('Belgium','Russia','3-2')],
    }},
    2006: {'host': 'Germany', 'groups': {
        'A': [('Germany','Costa Rica','4-2'),('Poland','Ecuador','0-2'),('Germany','Poland','1-0'),('Ecuador','Costa Rica','3-0'),('Ecuador','Germany','0-3'),('Costa Rica','Poland','1-2')],
        'B': [('England','Paraguay','1-0'),('Trinidad and Tobago','Sweden','0-0'),('England','Trinidad and Tobago','2-0'),('Sweden','Paraguay','1-0'),('Sweden','England','2-2'),('Paraguay','Trinidad and Tobago','2-0')],
        'C': [('Argentina','Ivory Coast','2-1'),('Serbia and Montenegro','Netherlands','0-1'),('Argentina','Serbia and Montenegro','6-0'),('Netherlands','Ivory Coast','2-1'),('Netherlands','Argentina','0-0'),('Ivory Coast','Serbia and Montenegro','3-2')],
        'D': [('Mexico','Iran','3-1'),('Angola','Portugal','0-1'),('Mexico','Angola','0-0'),('Portugal','Iran','2-0'),('Portugal','Mexico','2-1'),('Iran','Angola','1-1')],
        'E': [('Italy','Ghana','2-0'),('United States','Czech Republic','0-3'),('Italy','United States','1-1'),('Czech Republic','Ghana','0-2'),('Czech Republic','Italy','0-2'),('Ghana','United States','2-1')],
        'F': [('Brazil','Croatia','1-0'),('Australia','Japan','3-1'),('Brazil','Australia','2-0'),('Japan','Croatia','0-0'),('Japan','Brazil','1-4'),('Croatia','Australia','2-2')],
        'G': [('South Korea','Togo','2-1'),('France','Switzerland','0-0'),('France','South Korea','1-1'),('Switzerland','Togo','2-0'),('Switzerland','South Korea','2-0'),('Togo','France','0-2')],
        'H': [('Spain','Ukraine','4-0'),('Tunisia','Saudi Arabia','2-2'),('Spain','Tunisia','3-1'),('Saudi Arabia','Ukraine','0-4'),('Saudi Arabia','Spain','0-1'),('Ukraine','Tunisia','1-0')],
    }},
    2010: {'host': 'South Africa', 'groups': {
        'A': [('South Africa','Mexico','1-1'),('Uruguay','France','0-0'),('South Africa','Uruguay','0-3'),('France','Mexico','0-2'),('Mexico','Uruguay','0-1'),('France','South Africa','1-2')],
        'B': [('South Korea','Greece','2-0'),('Argentina','Nigeria','1-0'),('Argentina','South Korea','4-1'),('Greece','Nigeria','2-1'),('Greece','Argentina','0-2'),('Nigeria','South Korea','2-2')],
        'C': [('England','United States','1-1'),('Algeria','Slovenia','0-1'),('Slovenia','United States','2-2'),('England','Algeria','0-0'),('Slovenia','England','0-1'),('United States','Algeria','1-0')],
        'D': [('Serbia','Ghana','0-1'),('Germany','Australia','4-0'),('Germany','Serbia','0-1'),('Ghana','Australia','1-1'),('Ghana','Germany','0-1'),('Australia','Serbia','2-1')],
        'E': [('Netherlands','Denmark','2-0'),('Japan','Cameroon','1-0'),('Netherlands','Japan','1-0'),('Cameroon','Denmark','1-2'),('Cameroon','Netherlands','1-2'),('Denmark','Japan','1-3')],
        'F': [('Italy','Paraguay','1-1'),('New Zealand','Slovakia','1-1'),('Slovakia','Paraguay','0-2'),('Italy','New Zealand','1-1'),('Slovakia','Italy','3-2'),('Paraguay','New Zealand','0-0')],
        'G': [('Ivory Coast','Portugal','0-0'),('Brazil','North Korea','2-1'),('Brazil','Ivory Coast','3-1'),('Portugal','North Korea','7-0'),('Portugal','Brazil','0-0'),('North Korea','Ivory Coast','0-3')],
        'H': [('Chile','Honduras','1-0'),('Spain','Switzerland','0-1'),('Chile','Switzerland','1-0'),('Spain','Honduras','2-0'),('Chile','Spain','1-2'),('Honduras','Switzerland','0-0')],
    }},
    2014: {'host': 'Brazil', 'groups': {
        'A': [('Brazil','Croatia','3-1'),('Mexico','Cameroon','1-0'),('Brazil','Mexico','0-0'),('Cameroon','Croatia','0-4'),('Cameroon','Brazil','1-4'),('Croatia','Mexico','1-3')],
        'B': [('Spain','Netherlands','1-5'),('Chile','Australia','3-1'),('Spain','Chile','0-2'),('Australia','Netherlands','2-3'),('Australia','Spain','0-3'),('Netherlands','Chile','2-0')],
        'C': [('Colombia','Greece','3-0'),('Ivory Coast','Japan','2-1'),('Colombia','Ivory Coast','2-1'),('Japan','Greece','0-0'),('Japan','Colombia','1-4'),('Greece','Ivory Coast','2-1')],
        'D': [('Uruguay','Costa Rica','1-3'),('England','Italy','1-2'),('Uruguay','England','2-1'),('Italy','Costa Rica','0-1'),('Italy','Uruguay','0-1'),('Costa Rica','England','0-0')],
        'E': [('Switzerland','Ecuador','2-1'),('France','Honduras','3-0'),('Switzerland','France','2-5'),('Honduras','Ecuador','1-2'),('Honduras','Switzerland','0-3'),('Ecuador','France','0-0')],
        'F': [('Argentina','Bosnia and Herzegovina','2-1'),('Iran','Nigeria','0-0'),('Argentina','Iran','1-0'),('Nigeria','Bosnia and Herzegovina','1-0'),('Nigeria','Argentina','2-3'),('Bosnia and Herzegovina','Iran','3-1')],
        'G': [('Germany','Portugal','4-0'),('Ghana','United States','1-2'),('Germany','Ghana','2-2'),('United States','Portugal','2-2'),('United States','Germany','0-1'),('Portugal','Ghana','2-1')],
        'H': [('Belgium','Algeria','2-1'),('Russia','South Korea','1-1'),('Belgium','Russia','1-0'),('South Korea','Algeria','2-4'),('South Korea','Belgium','0-1'),('Algeria','Russia','1-1')],
    }},
    2018: {'host': 'Russia', 'groups': {
        'A': [('Russia','Saudi Arabia','5-0'),('Egypt','Uruguay','0-1'),('Russia','Egypt','3-1'),('Uruguay','Saudi Arabia','1-0'),('Uruguay','Russia','3-0'),('Saudi Arabia','Egypt','2-1')],
        'B': [('Morocco','Iran','0-1'),('Portugal','Spain','3-3'),('Portugal','Morocco','1-0'),('Iran','Spain','0-1'),('Iran','Portugal','1-1'),('Spain','Morocco','2-2')],
        'C': [('France','Australia','2-1'),('Peru','Denmark','0-1'),('France','Peru','1-0'),('Denmark','Australia','1-1'),('Denmark','France','0-0'),('Australia','Peru','0-2')],
        'D': [('Argentina','Iceland','1-1'),('Croatia','Nigeria','2-0'),('Argentina','Croatia','0-3'),('Nigeria','Iceland','2-0'),('Nigeria','Argentina','1-2'),('Iceland','Croatia','1-2')],
        'E': [('Costa Rica','Serbia','0-1'),('Brazil','Switzerland','1-1'),('Brazil','Costa Rica','2-0'),('Serbia','Switzerland','1-2'),('Serbia','Brazil','0-2'),('Switzerland','Costa Rica','2-2')],
        'F': [('Germany','Mexico','0-1'),('Sweden','South Korea','1-0'),('Germany','Sweden','2-1'),('South Korea','Mexico','1-2'),('South Korea','Germany','2-0'),('Mexico','Sweden','0-3')],
        'G': [('Belgium','Panama','3-0'),('Tunisia','England','1-2'),('Belgium','Tunisia','5-2'),('England','Panama','6-1'),('England','Belgium','0-1'),('Panama','Tunisia','1-2')],
        'H': [('Colombia','Japan','1-2'),('Poland','Senegal','1-2'),('Japan','Senegal','2-2'),('Poland','Colombia','0-3'),('Japan','Poland','0-1'),('Senegal','Colombia','0-1')],
    }},
    2022: {'host': 'Qatar', 'groups': {
        'A': [('Qatar','Ecuador','0-2'),('Senegal','Netherlands','0-2'),('Qatar','Senegal','1-3'),('Netherlands','Ecuador','1-1'),('Netherlands','Qatar','2-0'),('Ecuador','Senegal','1-2')],
        'B': [('England','Iran','6-2'),('United States','Wales','1-1'),('Wales','Iran','0-2'),('England','United States','0-0'),('Wales','England','0-3'),('Iran','United States','0-1')],
        'C': [('Argentina','Saudi Arabia','1-2'),('Mexico','Poland','0-0'),('Argentina','Mexico','2-0'),('Poland','Saudi Arabia','2-0'),('Poland','Argentina','0-2'),('Saudi Arabia','Mexico','1-2')],
        'D': [('Denmark','Tunisia','0-0'),('France','Australia','4-1'),('France','Denmark','2-1'),('Tunisia','Australia','0-1'),('Tunisia','France','1-0'),('Australia','Denmark','1-0')],
        'E': [('Germany','Japan','1-2'),('Spain','Costa Rica','7-0'),('Japan','Costa Rica','0-1'),('Spain','Germany','1-1'),('Japan','Spain','2-1'),('Costa Rica','Germany','2-4')],
        'F': [('Morocco','Croatia','0-0'),('Belgium','Canada','1-0'),('Belgium','Morocco','0-2'),('Croatia','Canada','4-1'),('Croatia','Belgium','0-0'),('Canada','Morocco','1-2')],
        'G': [('Switzerland','Cameroon','1-0'),('Brazil','Serbia','2-0'),('Cameroon','Serbia','3-3'),('Brazil','Switzerland','1-0'),('Brazil','Cameroon','0-1'),('Serbia','Switzerland','2-3')],
        'H': [('Uruguay','South Korea','0-0'),('Portugal','Ghana','3-2'),('South Korea','Ghana','2-3'),('Portugal','Uruguay','2-0'),('South Korea','Portugal','2-1'),('Ghana','Uruguay','0-2')],
    }},
}


class MatchDataLoader:
    """Loads and prepares historical match data from multiple sources."""

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config['data']['raw_dir'])
        self.processed_dir = Path(config['data']['processed_dir'])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_world_cup_matches(self):
        csv_path = self.processed_dir / 'world_cup_matches.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=['date'])
            return df
        df = self._build_world_cup_dataset()
        df.to_csv(csv_path, index=False)
        return df

    def _build_world_cup_dataset(self):
        rows = []
        for year, info in sorted(ALL_RESULTS.items()):
            host = info['host']
            groups = info.get('groups', {})
            if not groups:
                continue
            date = datetime(year, 6, 1)
            for group_name, matches in groups.items():
                for t1, t2, score in matches:
                    g1, g2 = map(int, score.split('-'))
                    label = 0 if g1 > g2 else (1 if g1 == g2 else 2)
                    rows.append({
                        'tournament': f'World Cup {year}',
                        'year': year,
                        'host': host,
                        'stage': 'group',
                        'group': group_name,
                        'team1': t1,
                        'team2': t2,
                        'goals1': g1,
                        'goals2': g2,
                        'label': label,
                        'date': date,
                    })
        return pd.DataFrame(rows)

    def load_fifa_rankings(self):
        csv_path = self.processed_dir / 'fifa_rankings.csv'
        if csv_path.exists():
            return pd.read_csv(csv_path, parse_dates=['date'])
        rankings = self._build_historical_rankings()
        rankings.to_csv(csv_path, index=False)
        return rankings

    def _build_historical_rankings(self):
        top_teams = ['Brazil', 'Argentina', 'Germany', 'Spain', 'France', 'England',
                     'Netherlands', 'Portugal', 'Belgium', 'Italy', 'Croatia', 'Uruguay']
        data = []
        rng = np.random.RandomState(42)
        for team in top_teams:
            base_rank = top_teams.index(team) + 1
            for year in range(1993, 2027):
                for month in [1, 4, 7, 10]:
                    rank = max(1, min(50, base_rank + int(rng.randn() * 3)))
                    points = max(1000, 2000 - rank * 15 + int(rng.randn() * 50))
                    data.append({
                        'date': datetime(year, month, 1),
                        'team': team,
                        'rank': rank,
                        'points': points,
                    })
        return pd.DataFrame(data)

    def load_team_registry(self):
        return pd.DataFrame({
            'team': [
                'Brazil', 'Argentina', 'Uruguay', 'Colombia', 'Chile', 'Peru', 'Ecuador', 'Paraguay', 'Bolivia', 'Venezuela',
                'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 'England', 'Portugal', 'Belgium', 'Croatia', 'Denmark',
                'Switzerland', 'Poland', 'Sweden', 'Austria', 'Czech Republic', 'Serbia', 'Ukraine', 'Turkey', 'Russia', 'Greece',
                'Mexico', 'United States', 'Costa Rica', 'Honduras', 'Canada', 'Panama', 'Jamaica', 'Trinidad and Tobago',
                'Japan', 'South Korea', 'Australia', 'Saudi Arabia', 'Iran', 'Qatar', 'United Arab Emirates',
                'Nigeria', 'Ghana', 'Senegal', 'Cameroon', 'Ivory Coast', 'Egypt', 'Morocco', 'Algeria', 'Tunisia', 'South Africa',
                'New Zealand', 'Scotland', 'Wales', 'Iceland', 'Slovenia', 'Slovakia', 'Bosnia and Herzegovina', 'Togo',
                'Angola', 'North Korea', 'Haiti', 'Curacao', 'Cape Verde', 'Iraq', 'Norway', 'Congo DR', 'Uzbekistan', 'Jordan',
            ],
            'confederation': [
                'CONMEBOL'] * 10 + ['UEFA'] * 20 + ['CONCACAF'] * 8 + ['AFC'] * 7 + ['CAF'] * 10
                + ['OFC'] * 1 + ['UEFA'] * 6 + ['CAF'] * 2 + ['AFC'] * 1
                + ['CONCACAF'] * 2 + ['CAF'] * 1 + ['AFC'] * 1 + ['UEFA'] * 1 + ['CAF'] * 1 + ['AFC'] * 2,
        })
