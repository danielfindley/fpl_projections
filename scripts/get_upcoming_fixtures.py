"""
Fetch upcoming Premier League fixtures from FBRef for predictions.
Uses undetected_chromedriver to bypass bot detection.
"""

import undetected_chromedriver as uc
import pandas as pd
import time
from io import StringIO
from pathlib import Path
import json
from datetime import datetime

# Configuration
FIXTURES_URL = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
OUTPUT_DIR = Path("data/upcoming")


def get_upcoming_fixtures(target_gameweek: int = None, save: bool = True):
    """
    Fetch upcoming fixtures from FBRef.
    
    Args:
        target_gameweek: Specific gameweek to get (None = all upcoming)
        save: Whether to save to CSV
        
    Returns:
        DataFrame with upcoming fixtures
    """
    print("=" * 60)
    print("FETCHING UPCOMING FIXTURES FROM FBREF")
    print("=" * 60)
    
    # Start browser
    print("\nStarting browser...")
    driver = uc.Chrome(version_main=142)
    
    try:
        print(f"Loading {FIXTURES_URL}")
        driver.get(FIXTURES_URL)
        time.sleep(5)  # Wait for page load
        
        html = driver.page_source
        df_list = pd.read_html(StringIO(html))
        
        # Find the fixtures table
        fixtures_df = None
        for table in df_list:
            cols = [str(c) for c in table.columns]
            if any('Wk' in c for c in cols) and len(table) > 100:
                fixtures_df = table.copy()
                print(f"Found fixtures table with {len(fixtures_df)} rows")
                break
        
        if fixtures_df is None:
            print("ERROR: Could not find fixtures table!")
            return None
        
        # Clean up the dataframe
        # Handle multi-level columns if present
        if isinstance(fixtures_df.columns, pd.MultiIndex):
            fixtures_df.columns = [col[-1] if isinstance(col, tuple) else col for col in fixtures_df.columns]
        
        # Remove header rows that got mixed in
        fixtures_df = fixtures_df[fixtures_df['Wk'] != 'Wk']
        fixtures_df['Wk'] = pd.to_numeric(fixtures_df['Wk'], errors='coerce')
        fixtures_df = fixtures_df.dropna(subset=['Wk'])
        fixtures_df['Wk'] = fixtures_df['Wk'].astype(int)
        
        # Parse date
        if 'Date' in fixtures_df.columns:
            fixtures_df['Date'] = pd.to_datetime(fixtures_df['Date'], errors='coerce')
        
        # Identify upcoming fixtures (no score yet)
        if 'Score' in fixtures_df.columns:
            upcoming = fixtures_df[fixtures_df['Score'].isna() | (fixtures_df['Score'] == '')]
        else:
            # If no Score column, use date
            upcoming = fixtures_df[fixtures_df['Date'] >= datetime.now()]
        
        print(f"\nFound {len(upcoming)} upcoming fixtures")
        
        # Filter to target gameweek if specified
        if target_gameweek is not None:
            upcoming = upcoming[upcoming['Wk'] == target_gameweek]
            print(f"Filtered to GW{target_gameweek}: {len(upcoming)} fixtures")
        
        # Clean up columns
        cols_to_keep = ['Wk', 'Date', 'Time', 'Home', 'Away']
        if 'xG' in upcoming.columns:
            cols_to_keep.extend(['xG', 'xG.1'])  # xG columns if present
        
        upcoming = upcoming[[c for c in cols_to_keep if c in upcoming.columns]]
        upcoming = upcoming.rename(columns={'Wk': 'gameweek', 'xG.1': 'xG_away'})
        
        # Add useful columns
        upcoming['season'] = '2025-26'
        
        print("\nUpcoming fixtures:")
        print(upcoming.to_string(index=False))
        
        if save:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            if target_gameweek:
                output_path = OUTPUT_DIR / f"gw{target_gameweek}_fixtures.csv"
            else:
                output_path = OUTPUT_DIR / "upcoming_fixtures.csv"
            
            upcoming.to_csv(output_path, index=False)
            print(f"\nSaved to {output_path}")
        
        return upcoming
        
    finally:
        driver.quit()
        print("\nBrowser closed.")


def prepare_prediction_data(fixtures_df: pd.DataFrame, player_data_path: str = "data/features/player_goal_prediction.parquet"):
    """
    Prepare prediction data by joining fixtures with player data.
    
    Args:
        fixtures_df: DataFrame with upcoming fixtures
        player_data_path: Path to player features parquet
        
    Returns:
        DataFrame ready for prediction
    """
    print("\n" + "=" * 60)
    print("PREPARING PREDICTION DATA")
    print("=" * 60)
    
    # Load player data
    player_df = pd.read_parquet(player_data_path)
    player_df = player_df[~player_df['player_name'].str.contains('Players', na=False)]
    
    print(f"Loaded {len(player_df)} player records")
    
    # Get unique players from most recent data
    latest_season = player_df['season'].max()
    latest_gw = player_df[player_df['season'] == latest_season]['gameweek'].max()
    
    # Get players who played in recent gameweeks
    recent_players = player_df[
        (player_df['season'] == latest_season) & 
        (player_df['gameweek'] >= latest_gw - 5)
    ][['player_name', 'team', 'position']].drop_duplicates()
    
    print(f"Found {len(recent_players)} active players")
    
    # Normalize team names for matching
    def normalize_team(name):
        if pd.isna(name):
            return ''
        name = str(name).lower().strip()
        name = name.replace(' ', '_')
        return name
    
    # Create prediction rows for each fixture
    prediction_rows = []
    
    for _, fixture in fixtures_df.iterrows():
        home_team = normalize_team(fixture.get('Home', ''))
        away_team = normalize_team(fixture.get('Away', ''))
        gameweek = fixture.get('gameweek', 0)
        
        # Get home team players
        home_players = recent_players[recent_players['team'].apply(normalize_team) == home_team]
        for _, player in home_players.iterrows():
            prediction_rows.append({
                'player_name': player['player_name'],
                'team': player['team'],
                'opponent': away_team,
                'position': player['position'],
                'gameweek': gameweek,
                'season': '2025-26',
                'is_home': 1
            })
        
        # Get away team players
        away_players = recent_players[recent_players['team'].apply(normalize_team) == away_team]
        for _, player in away_players.iterrows():
            prediction_rows.append({
                'player_name': player['player_name'],
                'team': player['team'],
                'opponent': home_team,
                'position': player['position'],
                'gameweek': gameweek,
                'season': '2025-26',
                'is_home': 0
            })
    
    pred_df = pd.DataFrame(prediction_rows)
    print(f"\nCreated {len(pred_df)} prediction rows")
    
    if len(pred_df) > 0:
        print(f"Teams: {pred_df['team'].nunique()}")
        print(f"Players: {pred_df['player_name'].nunique()}")
    
    return pred_df


def get_fixtures_from_fpl_api(gameweek: int = None):
    """
    Alternative: Get fixtures from FPL API (faster, no scraping needed).
    
    Args:
        gameweek: Target gameweek (None = next upcoming)
        
    Returns:
        DataFrame with fixtures
    """
    import requests
    
    print("=" * 60)
    print("FETCHING FIXTURES FROM FPL API")
    print("=" * 60)
    
    # Get bootstrap data for team names
    bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    teams = {t['id']: t['name'] for t in bootstrap['teams']}
    
    # Get current gameweek
    current_gw = None
    for event in bootstrap['events']:
        if event['is_current']:
            current_gw = event['id']
            break
        if event['is_next']:
            current_gw = event['id']
            break
    
    if gameweek is None:
        gameweek = current_gw + 1 if current_gw else 16
    
    print(f"Target gameweek: {gameweek}")
    
    # Get fixtures
    fixtures = requests.get("https://fantasy.premierleague.com/api/fixtures/").json()
    
    # Filter to target gameweek
    gw_fixtures = [f for f in fixtures if f.get('event') == gameweek]
    
    fixture_rows = []
    for f in gw_fixtures:
        fixture_rows.append({
            'gameweek': gameweek,
            'home_team': teams.get(f['team_h'], 'Unknown'),
            'away_team': teams.get(f['team_a'], 'Unknown'),
            'kickoff_time': f.get('kickoff_time', ''),
            'finished': f.get('finished', False)
        })
    
    fixtures_df = pd.DataFrame(fixture_rows)
    
    print(f"\nGW{gameweek} Fixtures:")
    print(fixtures_df[['home_team', 'away_team', 'kickoff_time']].to_string(index=False))
    
    return fixtures_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Get upcoming fixtures for FPL predictions")
    parser.add_argument("--gameweek", "-g", type=int, default=None, help="Target gameweek")
    parser.add_argument("--source", "-s", choices=["fbref", "fpl"], default="fpl", help="Data source")
    parser.add_argument("--prepare", "-p", action="store_true", help="Also prepare prediction data")
    
    args = parser.parse_args()
    
    if args.source == "fpl":
        fixtures = get_fixtures_from_fpl_api(args.gameweek)
    else:
        fixtures = get_upcoming_fixtures(args.gameweek)
    
    if args.prepare and fixtures is not None:
        pred_data = prepare_prediction_data(fixtures)
        
        # Save prediction data
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        gw = args.gameweek or 16
        output_path = OUTPUT_DIR / f"gw{gw}_prediction_data.csv"
        pred_data.to_csv(output_path, index=False)
        print(f"\nSaved prediction data to {output_path}")























