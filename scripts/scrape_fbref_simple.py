"""
Simple FBRef scraper - Run this to collect data!

This script:
1. Gets all fixtures for specified seasons (including match report URLs)
2. Scrapes each match report (all player stats in one page!)
3. Saves raw HTML for later parsing

Estimated time:
- 380 matches/season × 4 seconds = ~25 min per season
- 5 seasons = ~2 hours total (with resume capability)
"""

import undetected_chromedriver as uc
import pandas as pd
import time
import re
from pathlib import Path
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("data/raw/fbref")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = {
    "2024-25": "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures",
    "2023-24": "https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures",
    "2022-23": "https://fbref.com/en/comps/9/2022-2023/schedule/2022-2023-Premier-League-Scores-and-Fixtures",
    "2021-22": "https://fbref.com/en/comps/9/2021-2022/schedule/2021-2022-Premier-League-Scores-and-Fixtures",
    "2020-21": "https://fbref.com/en/comps/9/2020-2021/schedule/2020-2021-Premier-League-Scores-and-Fixtures",
}

# How long to wait between requests (be nice to FBRef!)
DELAY = 4  # seconds

# =============================================================================
# STEP 1: SCRAPE FIXTURES
# =============================================================================

def scrape_fixtures(driver, seasons_to_scrape=None):
    """
    Scrape fixtures and extract match report URLs.
    
    This is YOUR existing script, enhanced to grab match report links.
    """
    if seasons_to_scrape is None:
        seasons_to_scrape = list(SEASONS.keys())
    
    all_fixtures = []
    
    for season in seasons_to_scrape:
        url = SEASONS[season]
        print(f"\n{'='*60}")
        print(f"Loading fixtures for {season}...")
        print(f"{'='*60}")
        
        driver.get(url)
        time.sleep(5)
        
        html = driver.page_source
        soup = BeautifulSoup(html, "lxml")
        
        # Save raw HTML
        fixtures_dir = DATA_DIR / "fixtures"
        fixtures_dir.mkdir(exist_ok=True)
        with open(fixtures_dir / f"{season}.html", "w", encoding="utf-8") as f:
            f.write(html)
        
        # Find fixtures table and extract match report URLs
        table = soup.find("table", id=lambda x: x and "sched" in str(x).lower())
        
        if not table:
            print(f"⚠️ Could not find schedule table for {season}")
            continue
            
        rows = table.find_all("tr")
        season_fixtures = []
        
        for row in rows:
            # Find match report link
            match_link = row.find("a", string="Match Report")
            if not match_link:
                continue
                
            # Extract data
            try:
                wk = row.find("th", {"data-stat": "gameweek"})
                date = row.find("td", {"data-stat": "date"})
                home = row.find("td", {"data-stat": "home_team"})
                score = row.find("td", {"data-stat": "score"})
                away = row.find("td", {"data-stat": "away_team"})
                
                if not all([wk, date, home, score, away]):
                    continue
                
                # Only completed matches have scores
                score_text = score.text.strip()
                if not score_text or "–" not in score_text:
                    continue
                
                fixture = {
                    "season": season,
                    "gameweek": wk.text.strip(),
                    "date": date.text.strip(),
                    "home_team": home.text.strip(),
                    "score": score_text,
                    "away_team": away.text.strip(),
                    "match_url": "https://fbref.com" + match_link["href"],
                    "match_id": match_link["href"].split("/matches/")[1].split("/")[0],
                }
                season_fixtures.append(fixture)
                
            except Exception as e:
                continue
        
        print(f"✓ Found {len(season_fixtures)} completed matches for {season}")
        all_fixtures.extend(season_fixtures)
    
    # Save fixtures
    df = pd.DataFrame(all_fixtures)
    df.to_csv(DATA_DIR / "all_fixtures.csv", index=False)
    print(f"\n✓ Saved {len(df)} total fixtures to {DATA_DIR / 'all_fixtures.csv'}")
    
    return df


# =============================================================================
# STEP 2: SCRAPE MATCH REPORTS
# =============================================================================

def scrape_match_reports(driver, fixtures_df, resume=True, limit=None):
    """
    Scrape individual match reports.
    
    Each match report contains ALL player stats for both teams.
    """
    matches_dir = DATA_DIR / "matches"
    matches_dir.mkdir(exist_ok=True)
    
    urls = fixtures_df[["match_id", "match_url"]].values.tolist()
    
    if limit:
        urls = urls[:limit]
    
    total = len(urls)
    scraped = 0
    skipped = 0
    failed = 0
    
    print(f"\n{'='*60}")
    print(f"Scraping {total} match reports...")
    print(f"Estimated time: {total * DELAY / 60:.1f} minutes")
    print(f"{'='*60}\n")
    
    for i, (match_id, url) in enumerate(urls):
        save_path = matches_dir / f"{match_id}.html"
        
        # Skip if already scraped
        if resume and save_path.exists():
            skipped += 1
            continue
        
        try:
            print(f"[{i+1}/{total}] {match_id}...", end=" ", flush=True)
            driver.get(url)
            time.sleep(DELAY)
            
            html = driver.page_source
            
            # Verify we got actual content
            if len(html) < 10000:
                print("⚠️ Page too small, might be blocked")
                failed += 1
                continue
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html)
            
            print("✓")
            scraped += 1
            
        except Exception as e:
            print(f"✗ {e}")
            failed += 1
            continue
        
        # Progress summary every 25 matches
        if (i + 1) % 25 == 0:
            print(f"\n--- Progress: {i+1}/{total} | Scraped: {scraped} | Skipped: {skipped} | Failed: {failed} ---\n")
    
    print(f"\n{'='*60}")
    print(f"DONE! Scraped: {scraped} | Skipped (already had): {skipped} | Failed: {failed}")
    print(f"HTML files saved to: {matches_dir}")
    print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         FBRef Premier League Data Scraper                ║
    ╠══════════════════════════════════════════════════════════╣
    ║  This will scrape match reports containing:              ║
    ║  - All player stats (minutes, goals, assists, etc.)      ║
    ║  - xG and xA for every player                            ║
    ║  - Passing, shooting, defensive stats                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize driver
    print("Initializing Chrome driver...")
    driver = uc.Chrome(version_main=131)  # Adjust version if needed
    
    try:
        # STEP 1: Get fixtures (fast - just 5 pages)
        print("\n" + "="*60)
        print("STEP 1: Fetching fixtures for all seasons...")
        print("="*60)
        
        fixtures = scrape_fixtures(
            driver,
            seasons_to_scrape=["2024-25", "2023-24", "2022-23"]  # Add more as needed
        )
        
        # STEP 2: Scrape match reports
        print("\n" + "="*60)
        print("STEP 2: Scraping match reports...")
        print("="*60)
        
        scrape_match_reports(
            driver,
            fixtures,
            resume=True,  # Skip already downloaded
            limit=None,   # Set to 10 for testing
        )
        
    finally:
        driver.quit()
        print("\nDriver closed.")


