"""
Re-parse all saved HTML files to extract the CORRECT player stats tables.

The issue: pd.read_html() grabs tables by index, but FBRef pages have variable
numbers of sidebar tables, so the indices are wrong.

Solution: Parse HTML properly and find player stats tables by looking for:
- Tables with "Player" column
- Tables with many rows (10+ players)
- Tables with stats columns like "Min", "Gls", "xG"
"""

import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from io import StringIO
import re
import json
from datetime import datetime

RAW_HTML_DIR = Path("data/raw_html")
DATA_DIR = Path("data")
REPARSE_PROGRESS_FILE = DATA_DIR / "reparse_progress.json"


def load_reparse_progress():
    """Load list of already re-parsed files."""
    if REPARSE_PROGRESS_FILE.exists():
        with open(REPARSE_PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def save_reparse_progress(reparsed_files):
    """Save progress."""
    REPARSE_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPARSE_PROGRESS_FILE, "w") as f:
        json.dump(list(reparsed_files), f)


def clean_columns(df):
    """Clean column names from FBRef multi-level headers."""
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            col = '_'.join(str(c) for c in col)
        parts = str(col).split('_')
        new_cols.append(parts[-1])
    df.columns = new_cols
    return df


def clean_team_name(name):
    """Clean team name for folder naming."""
    name = re.sub(r'[^\w\s-]', '', name)
    name = name.strip().replace(' ', '_')
    return name.lower()


def find_player_stats_tables(html):
    """
    Find the actual player stats tables by looking at table structure.
    
    Returns dict with:
    - 'home': {'summary': df, 'passing': df, ...}
    - 'away': {'summary': df, 'passing': df, ...}
    """
    soup = BeautifulSoup(html, 'lxml')
    
    # Find all tables with IDs containing 'stats_'
    # These are the player stats tables
    stats_tables = soup.find_all('table', id=lambda x: x and 'stats_' in x)
    
    results = {'home': {}, 'away': {}}
    
    # Categorize tables by their ID suffix
    table_types = {
        'summary': [],
        'passing': [],
        'passing_types': [],
        'defense': [],
        'possession': [],
        'misc': [],
    }
    
    for table in stats_tables:
        table_id = table.get('id', '')
        
        # Skip goalkeeper tables
        if 'keeper' in table_id.lower():
            continue
            
        # Determine table type from ID
        if 'summary' in table_id:
            table_types['summary'].append(table)
        elif 'passing_types' in table_id:
            table_types['passing_types'].append(table)
        elif 'passing' in table_id:
            table_types['passing'].append(table)
        elif 'defense' in table_id:
            table_types['defense'].append(table)
        elif 'possession' in table_id:
            table_types['possession'].append(table)
        elif 'misc' in table_id:
            table_types['misc'].append(table)
    
    # For each table type, first table is home, second is away
    for stat_type, tables in table_types.items():
        if len(tables) >= 1:
            try:
                df = pd.read_html(StringIO(str(tables[0])))[0]
                df = clean_columns(df)
                results['home'][stat_type] = df
            except:
                pass
                
        if len(tables) >= 2:
            try:
                df = pd.read_html(StringIO(str(tables[1])))[0]
                df = clean_columns(df)
                results['away'][stat_type] = df
            except:
                pass
    
    return results


def get_teams_from_html(html):
    """Extract home and away team names from the HTML."""
    soup = BeautifulSoup(html, 'lxml')
    
    # Find scorebox which has team names
    scorebox = soup.find('div', class_='scorebox')
    if not scorebox:
        return None, None
    
    team_links = scorebox.find_all('a', href=lambda x: x and '/squads/' in x)
    if len(team_links) >= 2:
        home_team = team_links[0].text.strip()
        away_team = team_links[1].text.strip()
        return home_team, away_team
    
    return None, None


def reparse_match(html_path):
    """Re-parse a single match HTML file."""
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Get teams
    home_team, away_team = get_teams_from_html(html)
    if not home_team or not away_team:
        print(f"  [WARN] Could not find teams in {html_path.name}")
        return False
    
    # Find player stats tables
    tables = find_player_stats_tables(html)
    
    # Check if we got any tables
    if not tables['home'] and not tables['away']:
        print(f"  [WARN] No player stats tables found in {html_path.name}")
        return False
    
    # Extract season and match info from path
    # Path format: data/raw_html/{season}/{match_id}.html
    season = html_path.parent.name
    match_id = html_path.stem
    
    # Try to get gameweek from match_id (format: season_gwX_team1_vs_team2)
    gw_match = re.search(r'_gw(\d+)_', match_id)
    if gw_match:
        gameweek = gw_match.group(1)
    else:
        gameweek = "unknown"
    
    # Create output directory
    match_dir = DATA_DIR / season / f"gw{gameweek}" / f"{clean_team_name(home_team)}_vs_{clean_team_name(away_team)}"
    
    # Save home team tables
    home_dir = match_dir / clean_team_name(home_team)
    home_dir.mkdir(parents=True, exist_ok=True)
    
    for stat_type, df in tables['home'].items():
        stat_name = 'pass_types' if stat_type == 'passing_types' else stat_type
        df.to_csv(home_dir / f"{stat_name}.csv", index=False)
    
    # Save away team tables
    away_dir = match_dir / clean_team_name(away_team)
    away_dir.mkdir(parents=True, exist_ok=True)
    
    for stat_type, df in tables['away'].items():
        stat_name = 'pass_types' if stat_type == 'passing_types' else stat_type
        df.to_csv(away_dir / f"{stat_name}.csv", index=False)
    
    print(f"  [OK] {home_team} vs {away_team} - Home: {len(tables['home'])} tables, Away: {len(tables['away'])} tables")
    return True


def reparse_all(force=False):
    """Re-parse all saved HTML files."""
    html_files = list(RAW_HTML_DIR.glob("**/*.html"))
    print(f"Found {len(html_files)} HTML files total")
    
    # Load progress
    reparsed = load_reparse_progress()
    print(f"Already re-parsed: {len(reparsed)} files")
    
    if not force:
        html_files = [f for f in html_files if str(f) not in reparsed]
        print(f"Remaining to re-parse: {len(html_files)} files\n")
    else:
        print("Force mode: re-parsing ALL files\n")
    
    if not html_files:
        print("Nothing to re-parse!")
        return
    
    success = 0
    failed = 0
    start_time = datetime.now()
    
    for i, html_path in enumerate(html_files):
        season = html_path.parent.name
        print(f"[{i+1}/{len(html_files)}] {season}/{html_path.name}")
        
        try:
            if reparse_match(html_path):
                success += 1
                reparsed.add(str(html_path))
                # Save progress every 10 files
                if success % 10 == 0:
                    save_reparse_progress(reparsed)
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1
        
        if (i + 1) % 50 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(html_files) - i - 1) / rate if rate > 0 else 0
            print(f"\n--- Progress: {i+1}/{len(html_files)} | Success: {success} | Failed: {failed} | ETA: {remaining/60:.1f} min ---\n")
    
    # Final save
    save_reparse_progress(reparsed)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}")
    print(f"DONE! Success: {success} | Failed: {failed} | Time: {elapsed/60:.1f} min")
    print(f"Progress saved to: {REPARSE_PROGRESS_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Re-parse FBRef HTML files")
    parser.add_argument("--force", action="store_true", help="Force re-parse all files, ignoring progress")
    args = parser.parse_args()
    
    reparse_all(force=args.force)

