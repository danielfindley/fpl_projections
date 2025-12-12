"""Debug defcon data loading."""
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, '.')

from models.defcon_model import DefconModel

# Load defcon data
print("Loading defcon data...")
model = DefconModel()
data = model.load_defcon_data('data')

print(f"Total defcon records: {len(data)}")

if len(data) > 0:
    print(f"\nSeasons: {sorted(data['season'].unique())}")
    print(f"Sample teams: {list(data['team'].unique()[:10])}")
    print(f"Sample players: {list(data['player_name'].unique()[:5])}")
    
    # Check GW15 2025-26
    gw15 = data[(data['season'] == '2025-26') & (data['gameweek'] == 15)]
    print(f"\nGW15 2025-26 records: {len(gw15)}")
    
    if len(gw15) > 0:
        print(f"GW15 teams: {list(gw15['team'].unique())}")
        print(f"\nSample GW15 data:")
        print(gw15[['player_name', 'team', 'defcon', 'is_defender']].head(10))
    else:
        # Check what gameweeks exist for 2025-26
        s25 = data[data['season'] == '2025-26']
        print(f"\n2025-26 gameweeks available: {sorted(s25['gameweek'].unique())}")
        print(f"Max gameweek: {s25['gameweek'].max()}")
    
    # Compare with player_goal_prediction.parquet team names
    print("\n--- Comparing team names ---")
    parquet_path = Path('data/features/player_goal_prediction.parquet')
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        parquet_teams = set(df['team'].unique())
        defcon_teams = set(data['team'].unique())
        
        print(f"Parquet teams sample: {list(parquet_teams)[:5]}")
        print(f"Defcon teams sample: {list(defcon_teams)[:5]}")
        
        # Check overlap
        overlap = parquet_teams & defcon_teams
        print(f"\nOverlapping teams: {len(overlap)}")
        if len(overlap) < 5:
            print("Teams that DON'T match:")
            print(f"  In parquet only: {list(parquet_teams - defcon_teams)[:5]}")
            print(f"  In defcon only: {list(defcon_teams - parquet_teams)[:5]}")
else:
    print("No defcon data loaded!")
    
    # Check if directories exist
    data_path = Path('data')
    print(f"\nData path exists: {data_path.exists()}")
    if data_path.exists():
        print(f"Contents: {[d.name for d in data_path.iterdir() if d.is_dir()]}")

