"""
Quick script to rescrape GW15 for 2025-26 season.
Uses existing scraping infrastructure with undetected_chromedriver.

Follows patterns from scrape_all_seasons.py and scrape_fbref_simple.py
"""
import sys
sys.path.insert(0, '..')

from pathlib import Path
import pandas as pd
import shutil
import re
import time
from io import StringIO
from bs4 import BeautifulSoup

# Use undetected chromedriver like your existing scraper
import undetected_chromedriver as uc

# Configuration
DATA_DIR = Path("../data")
RAW_HTML_DIR = DATA_DIR / "raw_html" / "2025-26"
SEASON = "2025-26"
TARGET_GW = 15

# Timing - match existing scripts
INITIAL_PAGE_DELAY = 5  # Wait for CloudFlare verification
REQUEST_DELAY = 4  # seconds between match report requests

# FBRef URLs
FIXTURES_URL = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"


def clean_team_name(name):
    """Clean team name for folder naming."""
    name = re.sub(r'[^\w\s-]', '', name)
    name = name.strip().replace(' ', '_')
    return name.lower()


def get_driver():
    """Initialize undetected Chrome driver - matches scrape_all_seasons.py"""
    print("[*] Starting Chrome driver...")
    return uc.Chrome(version_main=143)  # Match your existing scripts


def scrape_fixtures(driver):
    """Scrape fixtures to get match report URLs - follows scrape_fbref_simple.py pattern."""
    print(f"\n{'='*60}")
    print(f"Loading fixtures for {SEASON}...")
    print(f"{'='*60}")
    
    driver.get(FIXTURES_URL)
    time.sleep(INITIAL_PAGE_DELAY)  # Wait for CloudFlare verification
    
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')
    
    # Find fixtures table - same pattern as scrape_fbref_simple.py
    table = soup.find("table", id=lambda x: x and "sched" in str(x).lower())
    
    if not table:
        print("⚠️ Could not find schedule table")
        return []
    
    fixtures = []
    rows = table.find_all("tr")
    
    for row in rows:
        # Find match report link - same pattern as scrape_fbref_simple.py
        match_link = row.find("a", string="Match Report")
        if not match_link:
            continue
        
        try:
            wk = row.find("th", {"data-stat": "gameweek"})
            date = row.find("td", {"data-stat": "date"})
            home = row.find("td", {"data-stat": "home_team"})
            score = row.find("td", {"data-stat": "score"})
            away = row.find("td", {"data-stat": "away_team"})
            
            if not all([wk, date, home, score, away]):
                continue
            
            # Only target gameweek
            if wk.text.strip() != str(TARGET_GW):
                continue
            
            # Only completed matches have scores
            score_text = score.text.strip()
            if not score_text or "–" not in score_text:
                continue
            
            fixtures.append({
                "season": SEASON,
                "gameweek": wk.text.strip(),
                "date": date.text.strip(),
                "home_team": home.text.strip(),
                "away_team": away.text.strip(),
                "score": score_text,
                "match_url": "https://fbref.com" + match_link["href"],
                "match_id": match_link["href"].split("/matches/")[1].split("/")[0],
            })
            
        except Exception as e:
            continue
    
    print(f"✓ Found {len(fixtures)} completed GW{TARGET_GW} matches:")
    for f in fixtures:
        print(f"    {f['home_team']} vs {f['away_team']} ({f['score']})")
    
    return fixtures


def scrape_match_reports(driver, fixtures):
    """Scrape match report HTML files - follows scrape_all_seasons.py pattern."""
    print(f"\n{'='*60}")
    print(f"Scraping {len(fixtures)} match reports...")
    print(f"{'='*60}")
    
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    
    for i, fixture in enumerate(fixtures):
        url = fixture["match_url"]
        home = fixture["home_team"]
        away = fixture["away_team"]
        
        # Create match ID - same pattern as scrape_all_seasons.py
        match_id = f"{SEASON}_gw{TARGET_GW}_{clean_team_name(home)}_vs_{clean_team_name(away)}"
        save_path = RAW_HTML_DIR / f"{match_id}.html"
        
        print(f"\n  Scraping: {home} vs {away} (GW{TARGET_GW})")
        
        try:
            driver.get(url)
            time.sleep(REQUEST_DELAY)  # Same delay as scrape_all_seasons.py
            
            html = driver.page_source
            
            # Verify we got actual content (not CloudFlare page)
            if "Just a moment" in html or len(html) < 10000:
                print(f"    ⚠️ Got CloudFlare page, waiting longer...")
                time.sleep(5)
                html = driver.page_source
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html)
            
            print(f"    ✓ Saved: {save_path.name}")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")


def reparse_gw15():
    """Parse the scraped HTML into structured folders."""
    print(f"\n[3] Parsing match reports...")
    
    # Import the reparse functions
    sys.path.insert(0, str(Path(__file__).parent))
    from reparse_matches import find_player_stats_tables, get_teams_from_html, clean_team_name as reparse_clean
    
    html_files = list(RAW_HTML_DIR.glob(f"*_gw{TARGET_GW}_*.html"))
    print(f"    Found {len(html_files)} GW{TARGET_GW} HTML files")
    
    for html_path in html_files:
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        
        home_team, away_team = get_teams_from_html(html)
        if not home_team or not away_team:
            print(f"    ✗ Could not find teams in {html_path.name}")
            continue
        
        tables = find_player_stats_tables(html)
        
        # Create output directory
        match_dir = DATA_DIR / SEASON / f"gw{TARGET_GW}" / f"{reparse_clean(home_team)}_vs_{reparse_clean(away_team)}"
        
        # Save home team tables
        home_dir = match_dir / reparse_clean(home_team)
        home_dir.mkdir(parents=True, exist_ok=True)
        for stat_type, df in tables['home'].items():
            stat_name = 'pass_types' if stat_type == 'passing_types' else stat_type
            df.to_csv(home_dir / f"{stat_name}.csv", index=False)
        
        # Save away team tables
        away_dir = match_dir / reparse_clean(away_team)
        away_dir.mkdir(parents=True, exist_ok=True)
        for stat_type, df in tables['away'].items():
            stat_name = 'pass_types' if stat_type == 'passing_types' else stat_type
            df.to_csv(away_dir / f"{stat_name}.csv", index=False)
        
        print(f"    ✓ {home_team} vs {away_team}")


def main():
    print("=" * 60)
    print(f"RESCRAPING GW{TARGET_GW} FOR {SEASON}")
    print("=" * 60)
    
    # Delete old GW15 data first
    gw15_dir = DATA_DIR / SEASON / f"gw{TARGET_GW}"
    if gw15_dir.exists():
        print(f"\n[0] Removing old GW{TARGET_GW} data...")
        shutil.rmtree(gw15_dir)
    
    driver = None
    try:
        driver = get_driver()
        
        # Scrape fixtures
        fixtures = scrape_fixtures(driver)
        if not fixtures:
            print("No fixtures found!")
            return
        
        # Scrape match reports
        scrape_match_reports(driver, fixtures)
        
    finally:
        if driver:
            driver.quit()
    
    # Parse into structured format
    reparse_gw15()
    
    # Show results
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    if gw15_dir.exists():
        matches = [m for m in gw15_dir.iterdir() if m.is_dir()]
        print(f"\nGW{TARGET_GW} matches: {len(matches)}")
        for m in matches:
            teams = [t.name for t in m.iterdir() if t.is_dir()]
            print(f"  {m.name}: {teams}")


if __name__ == "__main__":
    main()

