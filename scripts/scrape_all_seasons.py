"""
FBRef Full Season Scraper

Scrapes ALL match reports for multiple seasons and organizes data into:
    data/{season}/gw{gameweek}/{home_team}_vs_{away_team}/{team}/{stat_type}.csv

Stat types:
    - summary: Goals, assists, xG, xA, shots, etc.
    - passing: Pass completion, progressive passes, etc.
    - pass_types: Live/dead ball, crosses, etc.
    - defense: Tackles, interceptions, blocks, etc.
    - possession: Touches, carries, take-ons, etc.
    - misc: Cards, fouls, aerials, etc.

Run with: python scripts/scrape_all_seasons.py
"""

import undetected_chromedriver as uc
import pandas as pd
import time
import re
import json
from pathlib import Path
from io import StringIO
from datetime import datetime
from selenium.webdriver.common.by import By

# =============================================================================
# CONFIGURATION
# =============================================================================

SEASON_URLS = {
    # Current season (2025-26) - partial, ongoing
    "2025-26": "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures",
    # Historical seasons - complete
    "2024-25": "https://fbref.com/en/comps/9/2024-2025/schedule/2024-2025-Premier-League-Scores-and-Fixtures",
    "2023-24": "https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures",
    "2022-23": "https://fbref.com/en/comps/9/2022-2023/schedule/2022-2023-Premier-League-Scores-and-Fixtures",
    "2021-22": "https://fbref.com/en/comps/9/2021-2022/schedule/2021-2022-Premier-League-Scores-and-Fixtures",
    
    "2020-21": "https://fbref.com/en/comps/9/2020-2021/schedule/2020-2021-Premier-League-Scores-and-Fixtures",
    "2019-20": "https://fbref.com/en/comps/9/2019-2020/schedule/2019-2020-Premier-League-Scores-and-Fixtures",
    "2018-19": "https://fbref.com/en/comps/9/2018-2019/schedule/2018-2019-Premier-League-Scores-and-Fixtures",
    "2017-18": "https://fbref.com/en/comps/9/2017-2018/schedule/2017-2018-Premier-League-Scores-and-Fixtures",
}

DATA_DIR = Path("data")
RAW_HTML_DIR = DATA_DIR / "raw_html"  # Store raw HTML for re-parsing

# Table indices for each stat type (home team first, then away team)
TABLE_MAPPING = {
    # Home team tables
    "home": {
        "summary": 0,
        "passing": 1,
        "pass_types": 2,
        "defense": 3,
        "possession": 4,
        "misc": 5,
    },
    # Away team tables (offset by 6)
    "away": {
        "summary": 6,
        "passing": 7,
        "pass_types": 8,
        "defense": 9,
        "possession": 10,
        "misc": 11,
    }
}

REQUEST_DELAY = 4  # Seconds between requests


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_team_name(name):
    """Clean team name for folder naming."""
    # Remove special characters, replace spaces with underscores
    name = re.sub(r'[^\w\s-]', '', name)
    name = name.strip().replace(' ', '_')
    return name.lower()


def clean_columns(df):
    """Clean column names from FBRef multi-level headers."""
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            col = '_'.join(str(c) for c in col)
        # Take the last part after underscore
        parts = str(col).split('_')
        new_cols.append(parts[-1])
    df.columns = new_cols
    return df


def parse_gameweek_from_url_or_page(driver, match_url):
    """Try to extract gameweek from the page."""
    try:
        # Try to find gameweek in page content
        page_text = driver.page_source
        gw_match = re.search(r'Matchweek (\d+)', page_text)
        if gw_match:
            return int(gw_match.group(1))
    except:
        pass
    return None


def get_match_metadata(row):
    """Extract match metadata from a fixture row."""
    try:
        # Get gameweek
        gw_cell = row.find_element(By.CSS_SELECTOR, '[data-stat="gameweek"]')
        gameweek = gw_cell.text.strip()
        
        # Get home team
        home_cell = row.find_element(By.CSS_SELECTOR, '[data-stat="home_team"]')
        home_team = home_cell.text.strip()
        
        # Get away team
        away_cell = row.find_element(By.CSS_SELECTOR, '[data-stat="away_team"]')
        away_team = away_cell.text.strip()
        
        # Get score (to verify match is complete)
        score_cell = row.find_element(By.CSS_SELECTOR, '[data-stat="score"]')
        score = score_cell.text.strip()
        
        # Get match report link
        try:
            match_link = row.find_element(By.LINK_TEXT, "Match Report")
            match_url = match_link.get_attribute("href")
        except:
            match_url = None
            
        return {
            "gameweek": gameweek,
            "home_team": home_team,
            "away_team": away_team,
            "score": score,
            "match_url": match_url,
        }
    except Exception as e:
        return None


# =============================================================================
# MAIN SCRAPER CLASS
# =============================================================================

class FBRefSeasonScraper:
    def __init__(self):
        self.driver = None
        self.scraped_matches = self._load_progress()
        
    def _load_progress(self):
        """Load list of already scraped matches to enable resume."""
        progress_file = DATA_DIR / "scrape_progress.json"
        if progress_file.exists():
            with open(progress_file) as f:
                return set(json.load(f))
        return set()
    
    def _save_progress(self):
        """Save progress for resume capability."""
        progress_file = DATA_DIR / "scrape_progress.json"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_file, "w") as f:
            json.dump(list(self.scraped_matches), f)
    
    def _init_driver(self):
        if self.driver is None:
            print("[*] Starting Chrome driver...")
            self.driver = uc.Chrome(version_main=142)
    
    def _close_driver(self):
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def get_season_fixtures(self, season, url):
        """Get all fixture data for a season."""
        print(f"\n{'='*60}")
        print(f"Getting fixtures for {season}")
        print(f"{'='*60}")
        
        self._init_driver()
        self.driver.get(url)
        time.sleep(5)
        
        fixtures = []
        
        # Find all rows in the fixtures table
        try:
            table = self.driver.find_element(By.CSS_SELECTOR, 'table[id*="sched"]')
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            for row in rows:
                meta = get_match_metadata(row)
                if meta and meta["match_url"] and meta["score"]:
                    # Only include completed matches with scores
                    if "–" in meta["score"] or "-" in meta["score"]:
                        meta["season"] = season
                        fixtures.append(meta)
            
            print(f"[OK] Found {len(fixtures)} completed matches")
            
        except Exception as e:
            print(f"[ERROR] Failed to get fixtures: {e}")
        
        return fixtures
    
    def scrape_match(self, match_info):
        """Scrape a single match and save all tables."""
        season = match_info["season"]
        gw = match_info["gameweek"]
        home = match_info["home_team"]
        away = match_info["away_team"]
        url = match_info["match_url"]
        
        # Create unique match ID
        match_id = f"{season}_gw{gw}_{clean_team_name(home)}_vs_{clean_team_name(away)}"
        
        # Skip if already scraped
        if match_id in self.scraped_matches:
            return True
        
        print(f"\n  Scraping: {home} vs {away} (GW{gw})")
        
        try:
            self._init_driver()
            self.driver.get(url)
            time.sleep(REQUEST_DELAY)
            
            html = self.driver.page_source
            
            # Save raw HTML for potential re-parsing
            html_dir = RAW_HTML_DIR / season
            html_dir.mkdir(parents=True, exist_ok=True)
            with open(html_dir / f"{match_id}.html", "w", encoding="utf-8") as f:
                f.write(html)
            
            # Parse tables
            tables = pd.read_html(StringIO(html))
            
            if len(tables) < 12:
                print(f"    [WARN] Only found {len(tables)} tables, expected 12+")
                return False
            
            # Create output directory structure
            match_dir = DATA_DIR / season / f"gw{gw}" / f"{clean_team_name(home)}_vs_{clean_team_name(away)}"
            
            # Save home team tables
            home_dir = match_dir / clean_team_name(home)
            home_dir.mkdir(parents=True, exist_ok=True)
            
            for stat_type, idx in TABLE_MAPPING["home"].items():
                if idx < len(tables):
                    df = clean_columns(tables[idx].copy())
                    df.to_csv(home_dir / f"{stat_type}.csv", index=False)
            
            # Save away team tables
            away_dir = match_dir / clean_team_name(away)
            away_dir.mkdir(parents=True, exist_ok=True)
            
            for stat_type, idx in TABLE_MAPPING["away"].items():
                if idx < len(tables):
                    df = clean_columns(tables[idx].copy())
                    df.to_csv(away_dir / f"{stat_type}.csv", index=False)
            
            # Mark as scraped
            self.scraped_matches.add(match_id)
            self._save_progress()
            
            print(f"    [OK] Saved to {match_dir}")
            return True
            
        except Exception as e:
            print(f"    [ERROR] {e}")
            return False
    
    def scrape_season(self, season, url):
        """Scrape all matches for a season."""
        fixtures = self.get_season_fixtures(season, url)
        
        total = len(fixtures)
        success = 0
        failed = 0
        skipped = 0
        
        print(f"\nScraping {total} matches for {season}...")
        
        for i, match in enumerate(fixtures):
            match_id = f"{season}_gw{match['gameweek']}_{clean_team_name(match['home_team'])}_vs_{clean_team_name(match['away_team'])}"
            
            if match_id in self.scraped_matches:
                skipped += 1
                continue
            
            result = self.scrape_match(match)
            if result:
                success += 1
            else:
                failed += 1
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"\n  --- Progress: {i+1}/{total} | Success: {success} | Skipped: {skipped} | Failed: {failed} ---\n")
        
        print(f"\n{'='*60}")
        print(f"Season {season} complete!")
        print(f"  Success: {success} | Skipped: {skipped} | Failed: {failed}")
        print(f"{'='*60}")
    
    def scrape_all_seasons(self, seasons=None):
        """Scrape all specified seasons."""
        if seasons is None:
            seasons = list(SEASON_URLS.keys())
        
        print("""
        ============================================================
                    FBRef Full Season Scraper
        ============================================================
        
        This will scrape player stats from ALL Premier League matches.
        
        Data structure:
            data/{season}/gw{X}/{home}_vs_{away}/{team}/{stat_type}.csv
        
        Stat types: summary, passing, pass_types, defense, possession, misc
        
        Progress is saved, so you can resume if interrupted.
        ============================================================
        """)
        
        try:
            for season in seasons:
                if season not in SEASON_URLS:
                    print(f"[WARN] Unknown season: {season}")
                    continue
                
                self.scrape_season(season, SEASON_URLS[season])
                
        finally:
            self._close_driver()
            self._save_progress()
            print("\n[DONE] Scraping complete!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape FBRef match data")
    parser.add_argument("--seasons", nargs="+", help="Seasons to scrape (e.g., 2023-24 2022-23)")
    parser.add_argument("--test", action="store_true", help="Test mode: scrape only 3 matches")
    args = parser.parse_args()
    
    scraper = FBRefSeasonScraper()
    
    if args.test:
        # Test mode: just scrape first 3 matches from current season
        print("\n[TEST MODE] Scraping only 3 matches...")
        fixtures = scraper.get_season_fixtures("2024-25", SEASON_URLS["2024-25"])
        for match in fixtures[:3]:
            scraper.scrape_match(match)
        scraper._close_driver()
    else:
        # Full scrape
        scraper.scrape_all_seasons(args.seasons)

