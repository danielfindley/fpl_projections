"""
FPL API Player Status Fetcher

Fetches player availability status from the FPL API including:
- Status flags (available, injured, unavailable)
- Chance of playing next round (0-100)
- News/injury information

Usage:
    from scripts.fpl_api_status import get_fpl_player_status, merge_fpl_status
    
    status_df = get_fpl_player_status()
    df_with_status = merge_fpl_status(predictions_df, status_df, data_dir='data')
"""

import json
import requests
import pandas as pd
import numpy as np
from typing import Optional, Union
from pathlib import Path


# FPL API endpoint
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def get_fpl_player_status(verbose: bool = True) -> pd.DataFrame:
    """
    Fetch player status data from FPL API.
    
    Returns DataFrame with columns:
    - fpl_id: FPL player ID
    - fpl_web_name: Short display name (e.g., "Salah")
    - fpl_full_name: Full name (first + second name)
    - fpl_team_id: FPL team ID
    - fpl_team_name: Team name
    - fpl_status: Status code ('a'=available, 'i'=injured, 'u'=unavailable, 'd'=doubtful, 's'=suspended)
    - fpl_news: Injury/availability news text
    - fpl_chance_of_playing: Chance of playing next round (0-100, None if unknown)
    - fpl_status_available: 1 if status='a', else 0
    - fpl_status_injured: 1 if status='i', else 0
    - fpl_status_unavailable: 1 if status='u', else 0
    - fpl_status_doubtful: 1 if status='d', else 0
    - fpl_status_suspended: 1 if status='s', else 0
    """
    if verbose:
        print("[*] Fetching player status from FPL API...")
    
    try:
        response = requests.get(FPL_BOOTSTRAP_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        if verbose:
            print(f"  [WARN] Failed to fetch FPL API: {e}")
        return pd.DataFrame()
    
    # Get team ID to name mapping
    teams = {t['id']: t['name'] for t in data['teams']}
    
    # Extract player data
    players_data = []
    for player in data['elements']:
        status = player.get('status', 'a')
        chance = player.get('chance_of_playing_next_round')
        
        # Handle None chance - if status is 'a' (available), assume 100%
        if chance is None:
            if status == 'a':
                chance = 100
            else:
                chance = 0  # Conservative default for unknown injured/unavailable
        
        players_data.append({
            'fpl_id': player['id'],
            'fpl_web_name': player.get('web_name', ''),
            'fpl_first_name': player.get('first_name', ''),
            'fpl_second_name': player.get('second_name', ''),
            'fpl_full_name': f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
            'fpl_team_id': player.get('team'),
            'fpl_team_name': teams.get(player.get('team'), 'Unknown'),
            'fpl_status': status,
            'fpl_news': player.get('news', ''),
            'fpl_chance_of_playing': chance,
            # One-hot encoded status flags
            'fpl_status_available': 1 if status == 'a' else 0,
            'fpl_status_injured': 1 if status == 'i' else 0,
            'fpl_status_unavailable': 1 if status == 'u' else 0,
            'fpl_status_doubtful': 1 if status == 'd' else 0,
            'fpl_status_suspended': 1 if status == 's' else 0,
        })
    
    df = pd.DataFrame(players_data)
    
    if verbose:
        print(f"  [OK] Fetched {len(df)} players from FPL API")
        print(f"    Available: {(df['fpl_status'] == 'a').sum()}")
        print(f"    Injured: {(df['fpl_status'] == 'i').sum()}")
        print(f"    Doubtful: {(df['fpl_status'] == 'd').sum()}")
        print(f"    Unavailable: {(df['fpl_status'] == 'u').sum()}")
        print(f"    Suspended: {(df['fpl_status'] == 's').sum()}")
    
    return df


def merge_fpl_status(df: pd.DataFrame, 
                      fpl_status_df: pd.DataFrame = None,
                      player_name_col: str = 'player_name',
                      team_col: str = 'team',
                      data_dir: Union[str, Path] = 'data',
                      verbose: bool = True) -> pd.DataFrame:
    """
    Merge FPL status data into predictions dataframe.
    
    Uses name_matches.json (same as bonus matching) for consistent player matching.
    
    Args:
        df: DataFrame with player predictions
        fpl_status_df: FPL status DataFrame (if None, will fetch from API)
        player_name_col: Column name for player names in df
        team_col: Column name for team in df
        data_dir: Path to data directory containing name_matching/name_matches.json
        verbose: Whether to print progress
    
    Returns:
        DataFrame with FPL status columns merged in
    """
    df = df.copy()
    data_dir = Path(data_dir)
    
    # Fetch status if not provided
    if fpl_status_df is None or len(fpl_status_df) == 0:
        fpl_status_df = get_fpl_player_status(verbose=verbose)
    
    if len(fpl_status_df) == 0:
        if verbose:
            print("  [WARN] No FPL status data available, using defaults")
        # Add default columns
        df['fpl_status'] = 'a'
        df['fpl_chance_of_playing'] = 100
        df['fpl_status_available'] = 1
        df['fpl_status_injured'] = 0
        df['fpl_status_unavailable'] = 0
        df['fpl_status_doubtful'] = 0
        df['fpl_status_suspended'] = 0
        df['fpl_news'] = ''
        return df
    
    # Load name_matches.json (same matching as bonus data)
    name_matches_path = data_dir / 'name_matching' / 'name_matches.json'
    name_matches = {}
    if name_matches_path.exists():
        with open(name_matches_path, encoding='utf-8') as f:
            name_matches = json.load(f)
        if verbose:
            print(f"  [*] Loaded {len(name_matches)} name matches from {name_matches_path}")
    else:
        if verbose:
            print(f"  [WARN] name_matches.json not found at {name_matches_path}")
    
    # Create FPL ID lookup from status data
    fpl_id_to_status = {}
    for _, row in fpl_status_df.iterrows():
        fpl_id_to_status[row['fpl_id']] = {
            'fpl_status': row['fpl_status'],
            'fpl_chance_of_playing': row['fpl_chance_of_playing'],
            'fpl_status_available': row['fpl_status_available'],
            'fpl_status_injured': row['fpl_status_injured'],
            'fpl_status_unavailable': row['fpl_status_unavailable'],
            'fpl_status_doubtful': row['fpl_status_doubtful'],
            'fpl_status_suspended': row['fpl_status_suspended'],
            'fpl_news': row['fpl_news'],
        }
    
    # Initialize status columns with defaults (available)
    df['fpl_status'] = 'a'
    df['fpl_chance_of_playing'] = 100
    df['fpl_status_available'] = 1
    df['fpl_status_injured'] = 0
    df['fpl_status_unavailable'] = 0
    df['fpl_status_doubtful'] = 0
    df['fpl_status_suspended'] = 0
    df['fpl_news'] = ''
    
    matched = 0
    unmatched = []
    
    for idx, row in df.iterrows():
        player_name = row[player_name_col]
        
        # Look up in name_matches.json (same as bonus matching)
        if player_name in name_matches:
            match_data = name_matches[player_name]
            if isinstance(match_data, dict) and 'fpl_id' in match_data:
                fpl_id = match_data['fpl_id']
                
                # Look up status by FPL ID
                if fpl_id in fpl_id_to_status:
                    status = fpl_id_to_status[fpl_id]
                    df.at[idx, 'fpl_status'] = status['fpl_status']
                    df.at[idx, 'fpl_chance_of_playing'] = status['fpl_chance_of_playing']
                    df.at[idx, 'fpl_status_available'] = status['fpl_status_available']
                    df.at[idx, 'fpl_status_injured'] = status['fpl_status_injured']
                    df.at[idx, 'fpl_status_unavailable'] = status['fpl_status_unavailable']
                    df.at[idx, 'fpl_status_doubtful'] = status['fpl_status_doubtful']
                    df.at[idx, 'fpl_status_suspended'] = status['fpl_status_suspended']
                    df.at[idx, 'fpl_news'] = status['fpl_news']
                    matched += 1
                    continue
        
        # Not found in name_matches.json
        unmatched.append(f"{player_name} ({row[team_col]})")
    
    if verbose:
        print(f"  [OK] Matched {matched}/{len(df)} players with FPL status")
        if unmatched and len(unmatched) <= 10:
            print(f"    Unmatched: {unmatched}")
        elif unmatched:
            print(f"    Unmatched: {len(unmatched)} players (showing first 10: {unmatched[:10]})")
    
    return df


def apply_availability_filter(df: pd.DataFrame, 
                               pred_minutes_col: str = 'pred_minutes',
                               verbose: bool = True) -> pd.DataFrame:
    """
    Apply hard filters based on FPL availability status.
    
    Rules:
    - fpl_chance_of_playing == 0: Set pred_minutes to 0
    - fpl_status in ('i', 'u', 's'): Set pred_minutes to 0
    - fpl_chance_of_playing < 25: Set pred_minutes to 0
    - fpl_chance_of_playing < 75: Scale pred_minutes by chance/100
    
    Args:
        df: DataFrame with FPL status and predictions
        pred_minutes_col: Column name for predicted minutes
        verbose: Whether to print changes
    
    Returns:
        DataFrame with adjusted predictions
    """
    df = df.copy()
    
    if pred_minutes_col not in df.columns:
        if verbose:
            print(f"  [WARN] Column '{pred_minutes_col}' not found, skipping availability filter")
        return df
    
    if 'fpl_chance_of_playing' not in df.columns:
        if verbose:
            print(f"  [WARN] FPL status not merged, skipping availability filter")
        return df
    
    original_minutes = df[pred_minutes_col].copy()
    
    # Hard filter: 0% chance or injured/unavailable/suspended
    zero_mask = (
        (df['fpl_chance_of_playing'] == 0) |
        (df['fpl_status'].isin(['i', 'u', 's']))
    )
    
    # Low chance filter: <25% -> 0
    low_mask = (df['fpl_chance_of_playing'] < 25) & ~zero_mask
    
    # Medium chance filter: 25-74% -> scale down
    medium_mask = (df['fpl_chance_of_playing'] >= 25) & (df['fpl_chance_of_playing'] < 75)
    
    # Apply filters
    df.loc[zero_mask, pred_minutes_col] = 0
    df.loc[low_mask, pred_minutes_col] = 0
    df.loc[medium_mask, pred_minutes_col] = (
        df.loc[medium_mask, pred_minutes_col] * 
        df.loc[medium_mask, 'fpl_chance_of_playing'] / 100
    )
    
    if verbose:
        zeroed = zero_mask.sum() + low_mask.sum()
        scaled = medium_mask.sum()
        print(f"  [*] Availability filter applied:")
        print(f"      Zeroed minutes: {zeroed} players")
        print(f"      Scaled minutes: {scaled} players")
        
        # Show affected players
        affected = df[zero_mask | low_mask].copy()
        if len(affected) > 0:
            affected['_orig_mins'] = original_minutes[zero_mask | low_mask]
            print(f"      Players set to 0 minutes:")
            for _, row in affected.head(15).iterrows():
                news = row.get('fpl_news', '')[:40] + '...' if len(row.get('fpl_news', '')) > 40 else row.get('fpl_news', '')
                print(f"        {row.get('player_name', 'Unknown'):25s} "
                      f"({row.get('fpl_status', '?')}, {row.get('fpl_chance_of_playing', 0):.0f}%) "
                      f"[{news}]")
            if len(affected) > 15:
                print(f"        ... and {len(affected) - 15} more")
    
    return df


if __name__ == "__main__":
    # Test the functions
    print("Testing FPL API Status Fetcher")
    print("=" * 60)
    
    status_df = get_fpl_player_status(verbose=True)
    
    print("\nSample of injured/doubtful players:")
    injured = status_df[status_df['fpl_status'].isin(['i', 'd', 'u', 's'])]
    print(injured[['fpl_web_name', 'fpl_team_name', 'fpl_status', 'fpl_chance_of_playing', 'fpl_news']].head(20).to_string())

