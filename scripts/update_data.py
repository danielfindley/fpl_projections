"""
Update Data Script

1. Rescrape the latest completed gameweek to ensure data is fresh
2. Get fixtures for the next gameweek (teams playing)
3. Generate list of players needing projections

Usage:
    python scripts/update_data.py
    python scripts/update_data.py --current-gw 15 --next-gw 16
"""

import pandas as pd
import requests
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Configuration - use path relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
SEASON = "2025-26"


def get_current_and_next_gw() -> Tuple[int, int]:
    """Get current and next gameweek from FPL API."""
    print("Checking FPL API for current gameweek...")
    
    bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    
    current_gw = None
    next_gw = None
    
    for event in bootstrap['events']:
        if event['is_current']:
            current_gw = event['id']
        if event['is_next']:
            next_gw = event['id']
    
    # If no current (between seasons), use the last finished
    if current_gw is None:
        for event in reversed(bootstrap['events']):
            if event['finished']:
                current_gw = event['id']
                next_gw = current_gw + 1
                break
    
    print(f"  Current GW: {current_gw}, Next GW: {next_gw}")
    return current_gw, next_gw


def get_next_gw_fixtures(gameweek: int) -> pd.DataFrame:
    """
    Get fixtures for the next gameweek from FPL API.
    Returns teams that are playing (no stats, just who's playing who).
    """
    print(f"\n[1] Fetching GW{gameweek} fixtures from FPL API...")
    
    # Get team names
    bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    teams = {t['id']: t['name'] for t in bootstrap['teams']}
    team_short = {t['id']: t['short_name'] for t in bootstrap['teams']}
    
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
            'home_team_short': team_short.get(f['team_h'], ''),
            'away_team_short': team_short.get(f['team_a'], ''),
            'kickoff_time': f.get('kickoff_time', ''),
            'finished': f.get('finished', False)
        })
    
    fixtures_df = pd.DataFrame(fixture_rows)
    
    print(f"  Found {len(fixtures_df)} fixtures for GW{gameweek}:")
    for _, row in fixtures_df.iterrows():
        print(f"    {row['home_team']} vs {row['away_team']}")
    
    return fixtures_df


def get_players_for_teams(teams: List[str]) -> pd.DataFrame:
    """
    Get list of players from teams that are playing.
    Uses our existing data to find active players.
    """
    print(f"\n[2] Finding players for {len(teams)} teams...")
    
    # Load all player data from our existing CSVs
    all_players = []
    
    # Normalize team name for matching
    def normalize(name):
        return name.lower().replace(' ', '_').replace("'", "")
    
    team_normalized = {normalize(t): t for t in teams}
    
    # Scan through recent gameweeks to find active players
    season_dir = DATA_DIR / SEASON
    if not season_dir.exists():
        # Try 2024-25 as fallback
        season_dir = DATA_DIR / "2024-25"
    
    # Get the last few gameweeks
    gw_dirs = sorted([d for d in season_dir.iterdir() if d.is_dir() and d.name.startswith('gw')],
                     key=lambda x: int(x.name.replace('gw', '')), reverse=True)
    
    seen_players = set()
    
    for gw_dir in gw_dirs[:5]:  # Last 5 gameweeks
        for match_dir in gw_dir.iterdir():
            if not match_dir.is_dir():
                continue
            
            for team_dir in match_dir.iterdir():
                if not team_dir.is_dir():
                    continue
                
                team_name_normalized = normalize(team_dir.name)
                
                # Check if this team is playing next week
                matching_team = None
                for norm_name, orig_name in team_normalized.items():
                    if norm_name in team_name_normalized or team_name_normalized in norm_name:
                        matching_team = orig_name
                        break
                
                if not matching_team:
                    continue
                
                # Load summary.csv for player list
                summary_path = team_dir / 'summary.csv'
                if not summary_path.exists():
                    continue
                
                try:
                    summary = pd.read_csv(summary_path)
                    if 'Player' not in summary.columns:
                        continue
                    
                    for _, row in summary.iterrows():
                        player_name = row.get('Player', '')
                        if pd.isna(player_name) or not player_name:
                            continue
                        if 'Players' in str(player_name):  # Skip totals row
                            continue
                        
                        player_key = f"{player_name}_{team_dir.name}"
                        if player_key in seen_players:
                            continue
                        seen_players.add(player_key)
                        
                        minutes = pd.to_numeric(row.get('Min', 0), errors='coerce') or 0
                        
                        all_players.append({
                            'player_name': player_name,
                            'team': team_dir.name,
                            'team_display': matching_team,
                            'position': row.get('Pos', 'MID'),
                            'recent_minutes': minutes
                        })
                except Exception as e:
                    continue
    
    players_df = pd.DataFrame(all_players)
    
    # Remove duplicates, keep the one with most recent minutes
    if len(players_df) > 0:
        players_df = players_df.sort_values('recent_minutes', ascending=False)
        players_df = players_df.drop_duplicates(subset=['player_name', 'team'], keep='first')
    
    print(f"  Found {len(players_df)} unique players")
    
    return players_df


def get_latest_gw_data_status(gameweek: int) -> Dict:
    """Check if we have data for the latest gameweek."""
    print(f"\n[3] Checking data status for GW{gameweek}...")
    
    gw_dir = DATA_DIR / SEASON / f"gw{gameweek}"
    
    if not gw_dir.exists():
        print(f"  ⚠️ No data directory found for GW{gameweek}")
        return {'exists': False, 'matches': 0, 'teams': []}
    
    matches = list(gw_dir.iterdir())
    match_count = len([m for m in matches if m.is_dir()])
    
    teams = []
    for match_dir in matches:
        if match_dir.is_dir():
            for team_dir in match_dir.iterdir():
                if team_dir.is_dir():
                    teams.append(team_dir.name)
    
    print(f"  ✓ Found {match_count} matches, {len(teams)} team datasets")
    
    return {
        'exists': True,
        'matches': match_count,
        'teams': teams,
        'path': gw_dir
    }


def create_prediction_list(fixtures_df: pd.DataFrame, players_df: pd.DataFrame, next_gw: int) -> pd.DataFrame:
    """Create the final list of players needing predictions."""
    print(f"\n[4] Creating prediction list for GW{next_gw}...")
    
    prediction_rows = []
    
    # Normalize for matching
    def normalize(name):
        return name.lower().replace(' ', '_').replace("'", "")
    
    for _, fixture in fixtures_df.iterrows():
        home_team = fixture['home_team']
        away_team = fixture['away_team']
        
        # Find home team players
        home_players = players_df[
            players_df['team_display'].str.lower() == home_team.lower()
        ]
        
        for _, player in home_players.iterrows():
            prediction_rows.append({
                'player_name': player['player_name'],
                'team': player['team'],
                'team_display': home_team,
                'opponent': away_team,
                'position': player['position'],
                'is_home': 1,
                'gameweek': next_gw,
                'season': SEASON
            })
        
        # Find away team players
        away_players = players_df[
            players_df['team_display'].str.lower() == away_team.lower()
        ]
        
        for _, player in away_players.iterrows():
            prediction_rows.append({
                'player_name': player['player_name'],
                'team': player['team'],
                'team_display': away_team,
                'opponent': home_team,
                'position': player['position'],
                'is_home': 0,
                'gameweek': next_gw,
                'season': SEASON
            })
    
    pred_df = pd.DataFrame(prediction_rows)
    
    print(f"  ✓ Created {len(pred_df)} prediction rows")
    print(f"    Teams: {pred_df['team_display'].nunique()}")
    print(f"    Players: {pred_df['player_name'].nunique()}")
    
    return pred_df


def main(current_gw: int = None, next_gw: int = None):
    """Main update function."""
    print("=" * 70)
    print("FPL DATA UPDATE SCRIPT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get gameweeks if not specified
    if current_gw is None or next_gw is None:
        current_gw, next_gw = get_current_and_next_gw()
    
    print(f"\nCurrent GW: {current_gw} (latest completed)")
    print(f"Next GW: {next_gw} (to predict)")
    
    # Step 1: Get next week's fixtures
    fixtures_df = get_next_gw_fixtures(next_gw)
    
    # Get list of teams playing
    teams_playing = list(fixtures_df['home_team'].unique()) + list(fixtures_df['away_team'].unique())
    teams_playing = list(set(teams_playing))
    print(f"\n  Teams playing in GW{next_gw}: {len(teams_playing)}")
    
    # Step 2: Get players for those teams
    players_df = get_players_for_teams(teams_playing)
    
    # Step 3: Check current GW data status
    data_status = get_latest_gw_data_status(current_gw)
    
    # Step 4: Create prediction list
    prediction_df = create_prediction_list(fixtures_df, players_df, next_gw)
    
    # Save outputs
    output_dir = DATA_DIR / "upcoming"
    output_dir.mkdir(exist_ok=True)
    
    # Save fixtures
    fixtures_path = output_dir / f"gw{next_gw}_fixtures.csv"
    fixtures_df.to_csv(fixtures_path, index=False)
    print(f"\n✓ Saved fixtures to: {fixtures_path}")
    
    # Save prediction list
    prediction_path = output_dir / f"gw{next_gw}_players_to_predict.csv"
    prediction_df.to_csv(prediction_path, index=False)
    print(f"✓ Saved prediction list to: {prediction_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Next gameweek: GW{next_gw}")
    print(f"  Fixtures: {len(fixtures_df)}")
    print(f"  Teams playing: {len(teams_playing)}")
    print(f"  Players to predict: {len(prediction_df)}")
    print(f"\n  Current GW{current_gw} data: {'✓ Available' if data_status['exists'] else '✗ Missing'}")
    
    if not data_status['exists']:
        print(f"\n  ⚠️ Run scraper to get GW{current_gw} data:")
        print(f"     python scripts/scrape_fbref_simple.py")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"  1. Ensure GW{current_gw} data is scraped and parsed")
    print(f"  2. Run prediction pipeline for GW{next_gw}:")
    print(f"     from pipelines.fpl_prediction_pipeline import run_pipeline")
    print(f"     predictions = run_pipeline(target_gw={next_gw})")
    print("=" * 70)
    
    return {
        'current_gw': current_gw,
        'next_gw': next_gw,
        'fixtures': fixtures_df,
        'players': prediction_df,
        'data_status': data_status
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update FPL prediction data")
    parser.add_argument("--current-gw", type=int, default=None, help="Current completed gameweek")
    parser.add_argument("--next-gw", type=int, default=None, help="Next gameweek to predict")
    
    args = parser.parse_args()
    
    main(current_gw=args.current_gw, next_gw=args.next_gw)

