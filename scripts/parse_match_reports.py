"""
Parse FBRef match report HTML files into structured DataFrames.

Run this AFTER scraping to convert raw HTML → clean parquet files.
Parsing is fast and can be re-run to fix issues without re-scraping.
"""

import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "fbref"
OUTPUT_DIR = DATA_DIR / "intermediate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_match_report(html: str, match_id: str) -> List[Dict]:
    """
    Parse a single match report HTML into player-level records.
    
    Each match report contains:
    - Summary stats table (both teams)
    - Passing stats table
    - Defensive stats table
    - etc.
    
    We primarily use the summary tables which have the core stats.
    """
    soup = BeautifulSoup(html, "lxml")
    players = []
    
    # -------------------------------------------------------------------------
    # Extract match metadata
    # -------------------------------------------------------------------------
    
    # Get teams from scorebox
    scorebox = soup.find("div", class_="scorebox")
    if not scorebox:
        print(f"  ⚠️ No scorebox found for {match_id}")
        return players
    
    # Teams
    team_links = scorebox.find_all("a", href=lambda x: x and "/squads/" in x)
    teams = [link.text.strip() for link in team_links[:2]]
    if len(teams) != 2:
        print(f"  ⚠️ Could not find both teams for {match_id}")
        return players
    home_team, away_team = teams
    
    # Date
    date_elem = soup.find("span", class_="venuetime")
    match_date = date_elem.get("data-venue-date") if date_elem else None
    
    # Score
    scores = scorebox.find_all("div", class_="score")
    home_goals = int(scores[0].text.strip()) if scores else None
    away_goals = int(scores[1].text.strip()) if len(scores) > 1 else None
    
    # Gameweek (from breadcrumb or URL)
    gameweek = None
    breadcrumb = soup.find("div", id="info")
    if breadcrumb:
        gw_match = re.search(r"Matchweek (\d+)", breadcrumb.text)
        if gw_match:
            gameweek = int(gw_match.group(1))
    
    # -------------------------------------------------------------------------
    # Parse player stats tables
    # -------------------------------------------------------------------------
    
    # FBRef uses table IDs like:
    # - stats_[team_id]_summary for main stats
    # - stats_[team_id]_passing for passing
    # - etc.
    
    # Find all summary tables (one per team)
    summary_tables = soup.find_all("table", id=lambda x: x and "_summary" in str(x))
    
    for table in summary_tables:
        table_id = table.get("id", "")
        
        # Determine which team this table is for
        # The table comes AFTER the team header
        prev_header = table.find_previous(["h2", "h3"])
        if prev_header:
            header_text = prev_header.text.lower()
            is_home = home_team.lower() in header_text
        else:
            # Fallback: first table is usually home
            is_home = summary_tables.index(table) == 0
        
        team = home_team if is_home else away_team
        opponent = away_team if is_home else home_team
        team_goals = home_goals if is_home else away_goals
        opp_goals = away_goals if is_home else home_goals
        
        # Parse player rows
        tbody = table.find("tbody")
        if not tbody:
            continue
            
        for row in tbody.find_all("tr"):
            # Skip header rows
            if "thead" in str(row.get("class", "")):
                continue
            
            player_data = parse_player_row(row)
            if player_data:
                # Add match context
                player_data.update({
                    "match_id": match_id,
                    "match_date": match_date,
                    "gameweek": gameweek,
                    "team": team,
                    "opponent": opponent,
                    "is_home": is_home,
                    "team_goals": team_goals,
                    "opponent_goals": opp_goals,
                })
                players.append(player_data)
    
    # -------------------------------------------------------------------------
    # BONUS: Parse additional stat tables and merge
    # -------------------------------------------------------------------------
    
    # Get passing stats
    passing_tables = soup.find_all("table", id=lambda x: x and "_passing" in str(x))
    passing_stats = {}
    for table in passing_tables:
        tbody = table.find("tbody")
        if not tbody:
            continue
        for row in tbody.find_all("tr"):
            player_cell = row.find("th", {"data-stat": "player"})
            if player_cell:
                player_link = player_cell.find("a")
                if player_link and "/players/" in player_link.get("href", ""):
                    pid = player_link["href"].split("/players/")[1].split("/")[0]
                    passing_stats[pid] = {
                        "passes_completed": get_stat(row, "passes_completed"),
                        "passes_attempted": get_stat(row, "passes"),
                        "pass_pct": get_stat(row, "passes_pct"),
                        "key_passes": get_stat(row, "assisted_shots"),
                        "passes_into_final_third": get_stat(row, "passes_into_final_third"),
                        "passes_into_penalty_area": get_stat(row, "passes_into_penalty_area"),
                        "crosses_into_penalty_area": get_stat(row, "crosses_into_penalty_area"),
                    }
    
    # Merge passing stats
    for player in players:
        pid = player.get("player_id")
        if pid and pid in passing_stats:
            player.update(passing_stats[pid])
    
    return players


def parse_player_row(row) -> Optional[Dict]:
    """Parse a single player row from the summary table."""
    
    # Get player name and ID
    player_cell = row.find("th", {"data-stat": "player"})
    if not player_cell:
        return None
    
    player_link = player_cell.find("a")
    if not player_link:
        return None
    
    player_name = player_link.text.strip()
    player_url = player_link.get("href", "")
    
    # Extract player ID from URL
    if "/players/" in player_url:
        player_id = player_url.split("/players/")[1].split("/")[0]
    else:
        player_id = None
    
    # Get position
    pos_cell = row.find("td", {"data-stat": "position"})
    position = pos_cell.text.strip() if pos_cell else None
    
    return {
        "player_id": player_id,
        "player_name": player_name,
        "position": position,
        "minutes": get_stat(row, "minutes"),
        "goals": get_stat(row, "goals"),
        "assists": get_stat(row, "assists"),
        "pens_made": get_stat(row, "pens_made"),
        "pens_att": get_stat(row, "pens_att"),
        "shots": get_stat(row, "shots"),
        "shots_on_target": get_stat(row, "shots_on_target"),
        "xg": get_stat(row, "xg"),
        "npxg": get_stat(row, "npxg"),
        "xa": get_stat(row, "xg_assist"),
        "sca": get_stat(row, "sca"),  # shot-creating actions
        "gca": get_stat(row, "gca"),  # goal-creating actions
        "touches": get_stat(row, "touches"),
        "progressive_carries": get_stat(row, "progressive_carries"),
        "progressive_passes": get_stat(row, "progressive_passes"),
        "tackles": get_stat(row, "tackles"),
        "interceptions": get_stat(row, "interceptions"),
        "blocks": get_stat(row, "blocks"),
        "clearances": get_stat(row, "clearances"),
        "aerials_won": get_stat(row, "aerials_won"),
        "aerials_lost": get_stat(row, "aerials_lost"),
        "yellow_cards": get_stat(row, "cards_yellow"),
        "red_cards": get_stat(row, "cards_red"),
    }


def get_stat(row, stat_name, default=0) -> float:
    """Extract a stat value from a row by data-stat attribute."""
    cell = row.find(["td", "th"], {"data-stat": stat_name})
    if cell:
        text = cell.text.strip()
        if text:
            try:
                return float(text)
            except ValueError:
                return default
    return default


# =============================================================================
# MAIN PARSING PIPELINE
# =============================================================================

def parse_all_matches():
    """Parse all downloaded match HTML files."""
    
    matches_dir = RAW_DIR / "matches"
    html_files = sorted(matches_dir.glob("*.html"))
    
    print(f"\n{'='*60}")
    print(f"Parsing {len(html_files)} match reports...")
    print(f"{'='*60}\n")
    
    all_players = []
    successful = 0
    failed = 0
    
    for i, html_file in enumerate(html_files):
        match_id = html_file.stem
        
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                html = f.read()
            
            players = parse_match_report(html, match_id)
            
            if players:
                all_players.extend(players)
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"  ✗ Error parsing {match_id}: {e}")
            failed += 1
        
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(html_files)} | Players found: {len(all_players)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_players)
    
    # Clean up data types
    if not df.empty:
        # Convert date
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        
        # Ensure numeric columns are numeric
        numeric_cols = ["minutes", "goals", "assists", "shots", "xg", "xa", 
                        "npxg", "tackles", "interceptions", "yellow_cards", "red_cards"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Save
    output_path = OUTPUT_DIR / "player_match_stats.parquet"
    df.to_parquet(output_path, index=False)
    
    # Also save CSV for easy inspection
    df.to_csv(OUTPUT_DIR / "player_match_stats.csv", index=False)
    
    print(f"\n{'='*60}")
    print(f"PARSING COMPLETE!")
    print(f"{'='*60}")
    print(f"  Matches parsed successfully: {successful}")
    print(f"  Matches failed: {failed}")
    print(f"  Total player-match records: {len(df)}")
    print(f"  Unique players: {df['player_id'].nunique()}")
    print(f"  Date range: {df['match_date'].min()} to {df['match_date'].max()}")
    print(f"\n  Saved to: {output_path}")
    print(f"{'='*60}\n")
    
    return df


def create_team_match_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player stats to team-match level.
    
    This gives us team xG, goals, etc. for computing team form.
    """
    
    # Group by match and team
    team_stats = player_df.groupby(["match_id", "match_date", "gameweek", "team", "opponent", "is_home"]).agg({
        "goals": "sum",
        "xg": "sum",
        "xa": "sum",
        "shots": "sum",
        "shots_on_target": "sum",
        "team_goals": "first",
        "opponent_goals": "first",
    }).reset_index()
    
    # Rename for clarity
    team_stats = team_stats.rename(columns={
        "goals": "team_goals_from_players",  # Should match team_goals
        "xg": "team_xg",
        "xa": "team_xa",
    })
    
    # Add xGA (opponent's xG = our xGA)
    # Need to look up opponent's xG for this match
    xg_lookup = team_stats.set_index(["match_id", "team"])["team_xg"].to_dict()
    team_stats["team_xga"] = team_stats.apply(
        lambda r: xg_lookup.get((r["match_id"], r["opponent"]), 0),
        axis=1
    )
    
    # Clean sheet flag
    team_stats["clean_sheet"] = team_stats["opponent_goals"] == 0
    
    # Save
    team_stats.to_parquet(OUTPUT_DIR / "team_match_stats.parquet", index=False)
    team_stats.to_csv(OUTPUT_DIR / "team_match_stats.csv", index=False)
    
    print(f"Created team stats: {len(team_stats)} team-match records")
    
    return team_stats


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    # Parse all matches
    player_df = parse_all_matches()
    
    if not player_df.empty:
        # Create team-level aggregates
        team_df = create_team_match_stats(player_df)
        
        # Show sample
        print("\n📊 SAMPLE DATA:")
        print("\nPlayer Match Stats (first 5 rows):")
        print(player_df[["player_name", "team", "opponent", "minutes", "goals", "xg", "xa"]].head())
        
        print("\nTeam Match Stats (first 5 rows):")
        print(team_df[["team", "opponent", "team_xg", "team_xga", "clean_sheet"]].head())


