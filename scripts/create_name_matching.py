"""
Create name matching files between FPL API and FBRef data.
"""

import requests
import pandas as pd
import json
from pathlib import Path
from difflib import SequenceMatcher

# Create output directory
output_dir = Path('data/name_matching')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1: WRITE FPL PLAYER NAMES
# ============================================================
print('=' * 60)
print('STEP 1: Fetching FPL API player names...')
print('=' * 60)

bootstrap = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/').json()
teams = {t['id']: t['name'] for t in bootstrap['teams']}

with open(output_dir / 'fpl_names.txt', 'w', encoding='utf-8') as f:
    for p in bootstrap['elements']:
        team_name = teams.get(p['team'], 'Unknown')
        # Format: id|first_name|second_name|web_name|team_name
        f.write(f"{p['id']}|{p['first_name']}|{p['second_name']}|{p['web_name']}|{team_name}\n")

print(f"Wrote {len(bootstrap['elements'])} FPL players to {output_dir / 'fpl_names.txt'}")

# ============================================================
# STEP 2: WRITE FBREF PLAYER NAMES
# ============================================================
print('\n' + '=' * 60)
print('STEP 2: Extracting FBRef player names...')
print('=' * 60)

# Load from parquet
parquet_path = Path('data/features/player_goal_prediction.parquet')
if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    df = df[~df['player_name'].str.contains('Players', na=False)]
    
    # Get unique player-team combinations
    fbref_players = df[['player_name', 'team']].drop_duplicates()
    
    with open(output_dir / 'fbref_names.txt', 'w', encoding='utf-8') as f:
        for _, row in fbref_players.iterrows():
            f.write(f"{row['player_name']}|{row['team']}\n")
    
    print(f"Wrote {len(fbref_players)} FBRef players to {output_dir / 'fbref_names.txt'}")
else:
    print(f"ERROR: {parquet_path} not found!")
    exit(1)

# ============================================================
# STEP 3: CREATE MATCHING DICTIONARY
# ============================================================
print('\n' + '=' * 60)
print('STEP 3: Creating name matches...')
print('=' * 60)

def normalize_name(name):
    """Normalize name for matching."""
    if pd.isna(name):
        return ''
    name = str(name).lower().strip()
    # Remove accents
    replacements = {
        'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ã': 'a',
        'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e',
        'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i',
        'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'õ': 'o', 'ø': 'o',
        'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u',
        'ñ': 'n', 'ç': 'c', 'ß': 'ss',
        '-': ' ', "'": '', '\u2019': ''
    }
    for k, v in replacements.items():
        name = name.replace(k, v)
    return name

def similarity(a, b):
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()

# Read FPL names
fpl_players = []
with open(output_dir / 'fpl_names.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 5:
            fpl_players.append({
                'fpl_id': int(parts[0]),
                'first_name': parts[1],
                'second_name': parts[2],
                'web_name': parts[3],
                'team': parts[4],
                'full_name': f"{parts[1]} {parts[2]}"
            })

print(f"Read {len(fpl_players)} FPL players")

# Read FBRef names
fbref_players = []
with open(output_dir / 'fbref_names.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            fbref_players.append({
                'name': parts[0],
                'team': parts[1]
            })

print(f"Read {len(fbref_players)} FBRef players")

# Match names
print("\nMatching names...")
matches = {}
unmatched = []

for fb in fbref_players:
    fb_name = fb['name']
    best_match = None
    best_score = 0
    
    for fpl in fpl_players:
        # Try matching against different name fields
        for fpl_name in [fpl['full_name'], fpl['web_name'], fpl['second_name']]:
            score = similarity(fb_name, fpl_name)
            if score > best_score:
                best_score = score
                best_match = fpl
    
    if best_score >= 0.75:  # Match threshold
        matches[fb_name] = {
            'fpl_id': best_match['fpl_id'],
            'fpl_web_name': best_match['web_name'],
            'fpl_full_name': best_match['full_name'],
            'fpl_team': best_match['team'],
            'match_score': round(best_score, 3)
        }
    else:
        unmatched.append({
            'fbref_name': fb_name,
            'fbref_team': fb['team'],
            'best_match': best_match['full_name'] if best_match else None,
            'best_score': round(best_score, 3)
        })

# Write matches to JSON
with open(output_dir / 'name_matches.json', 'w', encoding='utf-8') as f:
    json.dump(matches, f, indent=2, ensure_ascii=False)

# Write matches to readable txt
with open(output_dir / 'name_matches.txt', 'w', encoding='utf-8') as f:
    f.write("FBREF_NAME | FPL_ID | FPL_WEB_NAME | FPL_FULL_NAME | SCORE\n")
    f.write("=" * 80 + "\n")
    for fb_name, match in sorted(matches.items()):
        f.write(f"{fb_name} | {match['fpl_id']} | {match['fpl_web_name']} | {match['fpl_full_name']} | {match['match_score']}\n")

# Write unmatched to txt
with open(output_dir / 'unmatched.txt', 'w', encoding='utf-8') as f:
    f.write("FBREF_NAME | FBREF_TEAM | BEST_FPL_MATCH | SCORE\n")
    f.write("=" * 80 + "\n")
    for u in sorted(unmatched, key=lambda x: x['best_score'], reverse=True):
        f.write(f"{u['fbref_name']} | {u['fbref_team']} | {u['best_match']} | {u['best_score']}\n")

print(f"\nMatched: {len(matches)} players")
print(f"Unmatched: {len(unmatched)} players")
print(f"\nOutput files:")
print(f"  - {output_dir / 'fpl_names.txt'}")
print(f"  - {output_dir / 'fbref_names.txt'}")
print(f"  - {output_dir / 'name_matches.txt'}")
print(f"  - {output_dir / 'name_matches.json'}")
print(f"  - {output_dir / 'unmatched.txt'}")

# Show sample matches
print("\n" + "=" * 60)
print("SAMPLE MATCHES (first 15)")
print("=" * 60)
for i, (fb_name, match) in enumerate(list(matches.items())[:15]):
    print(f"  {fb_name:30s} -> {match['fpl_full_name']:30s} ({match['match_score']:.0%})")

# Show unmatched
print("\n" + "=" * 60)
print(f"UNMATCHED (top 15 by score)")
print("=" * 60)
for u in unmatched[:15]:
    print(f"  {u['fbref_name']:30s} -> {u['best_match'] or 'None':30s} ({u['best_score']:.0%})")

