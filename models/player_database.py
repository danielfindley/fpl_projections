"""
Player Database Builder

Creates a comprehensive player database for each season and generates
full player-gameweek combinations for predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class PlayerDatabase:
    """Builds and manages player databases for each season."""
    
    def __init__(self):
        self.player_db = {}  # {season: DataFrame with all players}
    
    def build_season_database(self, df: pd.DataFrame, season: str) -> pd.DataFrame:
        """
        Build player database for a season.
        
        Returns all unique players who appeared in any match that season,
        with their team associations.
        """
        season_df = df[df['season'] == season].copy()
        
        # Get all unique players who appeared (even with 0 minutes)
        # Group by player and get their most common team, position, etc.
        player_info = season_df.groupby('player_name').agg({
            'team': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],  # Most common team
            'position': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],  # Most common position
            'player_id': 'first',  # Keep player_id if available
            'minutes': ['sum', 'max', 'count'],  # Total minutes, max in one game, appearances
        }).reset_index()
        
        # Flatten column names
        player_info.columns = ['player_name', 'team', 'position', 'player_id', 
                              'total_minutes', 'max_minutes', 'appearances']
        
        # Add season
        player_info['season'] = season
        
        # Create player_id if missing (use name + team)
        if 'player_id' in player_info.columns:
            player_info['player_id'] = player_info.apply(
                lambda row: row['player_id'] if pd.notna(row['player_id']) 
                else f"{row['player_name']}_{row['team']}",
                axis=1
            )
        else:
            player_info['player_id'] = player_info.apply(
                lambda row: f"{row['player_name']}_{row['team']}",
                axis=1
            )
        
        return player_info
    
    def get_all_players_for_season(self, season: str) -> pd.DataFrame:
        """Get all players who appeared in a season."""
        if season in self.player_db:
            return self.player_db[season]
        return pd.DataFrame()
    
    def create_full_gameweek_combinations(
        self, 
        df: pd.DataFrame, 
        season: str, 
        gameweek: int
    ) -> pd.DataFrame:
        """
        Create full player-gameweek combinations.
        
        For a given gameweek, creates rows for ALL players in the season
        matched with their teams (players stay with their teams).
        """
        # Get all players in the season
        season_players = self.build_season_database(df, season)
        
        # Get all teams playing in this gameweek and their opponents
        gw_matches = df[(df['season'] == season) & (df['gameweek'] == gameweek)]
        
        # Create match lookup: team -> (opponent, is_home)
        match_lookup = {}
        for _, row in gw_matches.iterrows():
            match_lookup[row['team']] = {
                'opponent': row['opponent'],
                'is_home': row.get('is_home', True)
            }
        
        # Create combinations: each player with their team
        # Include opponent and is_home from match lookup
        combinations = []
        for _, player in season_players.iterrows():
            team = player['team']
            if team in match_lookup:
                match_info = match_lookup[team]
                combinations.append({
                    'player_name': player['player_name'],
                    'player_id': player['player_id'],
                    'team': team,
                    'opponent': match_info['opponent'],
                    'is_home': match_info['is_home'],
                    'season': season,
                    'gameweek': gameweek,
                    'position': player['position'],
                })
        
        return pd.DataFrame(combinations)
    
    def merge_with_actual_data(
        self, 
        full_combinations: pd.DataFrame, 
        actual_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge full combinations with actual match data.
        
        If a player actually played, use their actual data.
        If not, create a row with 0 minutes and fill features.
        """
        # Merge on player_name, team, season, gameweek
        merge_cols = ['player_name', 'team', 'season', 'gameweek']
        
        # Get columns that exist in actual_data
        available_cols = ['minutes', 'goals', 'assists', 'xg', 'xag', 'opponent', 'is_home']
        actual_cols = [c for c in available_cols if c in actual_data.columns]
        actual_merge = actual_data[merge_cols + actual_cols].copy()
        
        # Merge
        merged = full_combinations.merge(
            actual_merge,
            on=merge_cols,
            how='left',
            suffixes=('', '_actual')
        )
        
        # Fill missing values (players who didn't play)
        if 'minutes' in merged.columns:
            merged['minutes'] = merged['minutes'].fillna(0)
        else:
            merged['minutes'] = 0
            
        for col in ['goals', 'assists', 'xg', 'xag']:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
            else:
                merged[col] = 0
        
        # Handle opponent and is_home (should already be in full_combinations)
        if 'opponent_actual' in merged.columns:
            merged['opponent'] = merged['opponent'].fillna(merged['opponent_actual'])
        if 'is_home_actual' in merged.columns:
            merged['is_home'] = merged['is_home'].fillna(merged['is_home_actual'])
        
        return merged

