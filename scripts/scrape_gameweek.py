"""
Scrape a specific gameweek from FBRef.

This script:
1. Fetches fixtures for the target gameweek from FBRef
2. Scrapes each match report HTML
3. Parses the HTML into structured CSV files

Usage:
    python scripts/scrape_gameweek.py --gameweek 16
    python scripts/scrape_gameweek.py --gameweek 16 --season 2025-26
    python scripts/scrape_gameweek.py -g 16 -s 2025-26 --no-delete

Can also be imported and used programmatically:
    from scripts.scrape_gameweek import scrape_gameweek
    scrape_gameweek(gameweek=16, season='2025-26')
"""

import sys
import argparse
from pathlib import Path
import shutil
import re
import time
from bs4 import BeautifulSoup

# Use undetected chromedriver to bypass bot detection
import undetected_chromedriver as uc

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - relative to project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

# Timing - be nice to FBRef
INITIAL_PAGE_DELAY = 10  # Initial wait after page load
CAPTCHA_TIMEOUT = 120  # Maximum seconds to wait for captcha/CloudFlare
CAPTCHA_CHECK_INTERVAL = 3  # Seconds between checks for page load
REQUEST_DELAY = 4  # Seconds between match report requests

# FBRef URLs
FIXTURES_URL = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_team_name(name: str) -> str:
    """Clean team name for folder naming."""
    name = re.sub(r'[^\w\s-]', '', name)
    name = name.strip().replace(' ', '_')
    return name.lower()


def get_driver():
    """Initialize undetected Chrome driver."""
    print("[*] Starting Chrome driver...")
    return uc.Chrome(version_main=143)


# =============================================================================
# SCRAPING FUNCTIONS
# =============================================================================

def wait_for_page_load(driver, check_text: str = None, check_element: str = None, 
                        timeout: int = CAPTCHA_TIMEOUT) -> bool:
    """
    Wait for page to fully load, handling CloudFlare captcha/verification.
    
    Args:
        driver: Selenium WebDriver instance
        check_text: Text that should be present when page is loaded (optional)
        check_element: Element ID/class that should be present (optional)
        timeout: Maximum seconds to wait
    
    Returns:
        True if page loaded successfully, False if timeout
    """
    start_time = time.time()
    attempt = 0
    
    print(f"  [*] Waiting for page to load (captcha timeout: {timeout}s)...")
    
    while time.time() - start_time < timeout:
        attempt += 1
        html = driver.page_source
        
        # Check for CloudFlare/captcha indicators
        cloudflare_indicators = [
            "Just a moment",
            "Checking your browser",
            "Enable JavaScript and cookies",
            "Verifying you are human",
            "cf-browser-verification",
            "challenge-running"
        ]
        
        is_cloudflare = any(indicator in html for indicator in cloudflare_indicators)
        
        if is_cloudflare:
            elapsed = int(time.time() - start_time)
            print(f"    [Attempt {attempt}] CloudFlare captcha detected, waiting... ({elapsed}s elapsed)")
            time.sleep(CAPTCHA_CHECK_INTERVAL)
            continue
        
        # Check if page has actual content
        if len(html) < 10000:
            elapsed = int(time.time() - start_time)
            print(f"    [Attempt {attempt}] Page too small ({len(html)} bytes), waiting... ({elapsed}s elapsed)")
            time.sleep(CAPTCHA_CHECK_INTERVAL)
            continue
        
        # Optional: check for specific text
        if check_text and check_text not in html:
            elapsed = int(time.time() - start_time)
            print(f"    [Attempt {attempt}] Expected content not found, waiting... ({elapsed}s elapsed)")
            time.sleep(CAPTCHA_CHECK_INTERVAL)
            continue
        
        # Success - page loaded!
        elapsed = int(time.time() - start_time)
        print(f"  [✓] Page loaded successfully after {elapsed}s")
        return True
    
    print(f"  [✗] Timeout after {timeout}s - page did not load")
    return False


def scrape_fixtures(driver, gameweek: int, season: str) -> list:
    """
    Scrape fixtures to get match report URLs for a specific gameweek.
    
    Args:
        driver: Selenium WebDriver instance
        gameweek: Target gameweek number
        season: Season string (e.g., '2025-26')
    
    Returns:
        List of fixture dictionaries with match URLs
    """
    print(f"\n{'='*60}")
    print(f"Loading fixtures for {season} GW{gameweek}...")
    print(f"{'='*60}")
    
    driver.get(FIXTURES_URL)
    time.sleep(INITIAL_PAGE_DELAY)
    
    # Wait for captcha/CloudFlare to complete - look for "Premier League" text
    if not wait_for_page_load(driver, check_text="Premier League"):
        print("⚠️ Failed to load fixtures page - captcha may have timed out")
        print("   Please try running again - you may need to solve the captcha manually")
        return []
    
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')
    
    # Find fixtures table
    table = soup.find("table", id=lambda x: x and "sched" in str(x).lower())
    
    if not table:
        print("⚠️ Could not find schedule table")
        return []
    
    fixtures = []
    rows = table.find_all("tr")
    
    for row in rows:
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
            if wk.text.strip() != str(gameweek):
                continue
            
            # Only completed matches have scores
            score_text = score.text.strip()
            if not score_text or "–" not in score_text:
                continue
            
            fixtures.append({
                "season": season,
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
    
    print(f"✓ Found {len(fixtures)} completed GW{gameweek} matches:")
    for f in fixtures:
        print(f"    {f['home_team']} vs {f['away_team']} ({f['score']})")
    
    return fixtures


def scrape_match_reports(driver, fixtures: list, gameweek: int, season: str) -> None:
    """
    Scrape match report HTML files for each fixture.
    
    Args:
        driver: Selenium WebDriver instance
        fixtures: List of fixture dictionaries
        gameweek: Target gameweek number
        season: Season string
    """
    print(f"\n{'='*60}")
    print(f"Scraping {len(fixtures)} match reports...")
    print(f"{'='*60}")
    
    raw_html_dir = DATA_DIR / "raw_html" / season
    raw_html_dir.mkdir(parents=True, exist_ok=True)
    
    for i, fixture in enumerate(fixtures):
        url = fixture["match_url"]
        home = fixture["home_team"]
        away = fixture["away_team"]
        
        match_id = f"{season}_gw{gameweek}_{clean_team_name(home)}_vs_{clean_team_name(away)}"
        save_path = raw_html_dir / f"{match_id}.html"
        
        print(f"\n  [{i+1}/{len(fixtures)}] Scraping: {home} vs {away}")
        
        try:
            driver.get(url)
            time.sleep(REQUEST_DELAY)
            
            # Wait for page to load with proper captcha handling
            # Use shorter timeout for individual matches (they usually load faster after first page)
            if not wait_for_page_load(driver, check_text="Match Report", timeout=60):
                print(f"    ⚠️ Failed to load match page after waiting - trying one more time...")
                driver.refresh()
                time.sleep(REQUEST_DELAY)
                if not wait_for_page_load(driver, timeout=60):
                    print(f"    ✗ Could not load match page, skipping")
                    continue
            
            html = driver.page_source
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html)
            
            print(f"    ✓ Saved: {save_path.name}")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")


def parse_match_htmls(gameweek: int, season: str) -> None:
    """
    Parse scraped HTML files into structured CSV folders.
    
    Args:
        gameweek: Target gameweek number
        season: Season string
    """
    print(f"\n{'='*60}")
    print(f"Parsing match reports into structured data...")
    print(f"{'='*60}")
    
    # Import the reparse functions
    sys.path.insert(0, str(SCRIPT_DIR))
    from reparse_matches import find_player_stats_tables, get_teams_from_html, clean_team_name as reparse_clean
    
    raw_html_dir = DATA_DIR / "raw_html" / season
    html_files = list(raw_html_dir.glob(f"*_gw{gameweek}_*.html"))
    print(f"  Found {len(html_files)} GW{gameweek} HTML files")
    
    parsed_count = 0
    for html_path in html_files:
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        
        home_team, away_team = get_teams_from_html(html)
        if not home_team or not away_team:
            print(f"    ✗ Could not find teams in {html_path.name}")
            continue
        
        tables = find_player_stats_tables(html)
        
        # Create output directory
        match_dir = DATA_DIR / season / f"gw{gameweek}" / f"{reparse_clean(home_team)}_vs_{reparse_clean(away_team)}"
        
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
        parsed_count += 1
    
    print(f"\n  Parsed {parsed_count} matches successfully")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def scrape_gameweek(gameweek: int, season: str = "2025-26", delete_existing: bool = True, 
                     verbose: bool = True) -> dict:
    """
    Scrape and parse a specific gameweek from FBRef.
    
    Args:
        gameweek: Target gameweek number (1-38)
        season: Season string (e.g., '2025-26')
        delete_existing: Whether to delete existing data for this gameweek
        verbose: Whether to print progress
    
    Returns:
        Dictionary with results:
        - 'gameweek': Target gameweek
        - 'season': Season
        - 'matches_found': Number of completed matches found
        - 'matches_scraped': Number of matches successfully scraped
        - 'output_dir': Path to output directory
    """
    if verbose:
        print("=" * 60)
        print(f"SCRAPING GW{gameweek} FOR {season}")
        print("=" * 60)
    
    gw_dir = DATA_DIR / season / f"gw{gameweek}"
    
    # Delete old data if requested
    if delete_existing and gw_dir.exists():
        if verbose:
            print(f"\n[0] Removing old GW{gameweek} data...")
        shutil.rmtree(gw_dir)
    
    driver = None
    fixtures = []
    
    try:
        driver = get_driver()
        
        # Scrape fixtures
        fixtures = scrape_fixtures(driver, gameweek, season)
        if not fixtures:
            if verbose:
                print(f"\n⚠️ No completed matches found for GW{gameweek}")
            return {
                'gameweek': gameweek,
                'season': season,
                'matches_found': 0,
                'matches_scraped': 0,
                'output_dir': None
            }
        
        # Scrape match reports
        scrape_match_reports(driver, fixtures, gameweek, season)
        
    finally:
        if driver:
            driver.quit()
            if verbose:
                print("\n[*] Browser closed")
    
    # Parse into structured format
    parse_match_htmls(gameweek, season)
    
    # Count results
    matches_scraped = 0
    if gw_dir.exists():
        matches_scraped = len([m for m in gw_dir.iterdir() if m.is_dir()])
    
    if verbose:
        print("\n" + "=" * 60)
        print("SCRAPING COMPLETE!")
        print("=" * 60)
        print(f"  Gameweek: {gameweek}")
        print(f"  Season: {season}")
        print(f"  Matches found: {len(fixtures)}")
        print(f"  Matches scraped: {matches_scraped}")
        print(f"  Output directory: {gw_dir}")
        
        if gw_dir.exists():
            print(f"\n  Match folders:")
            for m in sorted(gw_dir.iterdir()):
                if m.is_dir():
                    teams = [t.name for t in m.iterdir() if t.is_dir()]
                    print(f"    {m.name}: {teams}")
    
    return {
        'gameweek': gameweek,
        'season': season,
        'matches_found': len(fixtures),
        'matches_scraped': matches_scraped,
        'output_dir': gw_dir
    }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Scrape a specific Premier League gameweek from FBRef",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/scrape_gameweek.py --gameweek 16
    python scripts/scrape_gameweek.py -g 16 -s 2025-26
    python scripts/scrape_gameweek.py -g 16 --no-delete
        """
    )
    
    parser.add_argument(
        "-g", "--gameweek",
        type=int,
        required=True,
        help="Gameweek number to scrape (1-38)"
    )
    
    parser.add_argument(
        "-s", "--season",
        type=str,
        default="2025-26",
        help="Season to scrape (default: 2025-26)"
    )
    
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Don't delete existing data before scraping"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate gameweek
    if args.gameweek < 1 or args.gameweek > 38:
        print(f"Error: Gameweek must be between 1 and 38, got {args.gameweek}")
        sys.exit(1)
    
    # Run scraper
    result = scrape_gameweek(
        gameweek=args.gameweek,
        season=args.season,
        delete_existing=not args.no_delete,
        verbose=not args.quiet
    )
    
    # Exit with appropriate code
    if result['matches_scraped'] > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

