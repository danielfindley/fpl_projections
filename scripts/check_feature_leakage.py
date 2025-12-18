"""
Check if features in parquet file use shift(1) correctly (no data leakage).
"""

import pandas as pd

df = pd.read_parquet('data/features/player_goal_prediction.parquet')
df = df[~df['player_name'].str.contains('Players', na=False)]

# Check Salah across gameweeks
salah = df[(df['player_name'] == 'Mohamed Salah') & (df['season'] == '2025-26')].sort_values('gameweek')

if len(salah) >= 15:
    print("=" * 80)
    print("CHECKING FEATURE PREPARATION FOR DATA LEAKAGE")
    print("=" * 80)
    
    print("\nMohamed Salah - Sample gameweeks:")
    print(salah[['gameweek', 'minutes', 'goals', 'xg', 'player_roll5_xg', 'team_xg_roll5']].head(10).to_string())
    
    # Check GW15 specifically
    gw15 = salah[salah['gameweek'] == 15]
    gw14 = salah[salah['gameweek'] == 14]
    
    if len(gw15) > 0 and len(gw14) > 0:
        print("\n" + "=" * 80)
        print("KEY TEST: GW15 features should NOT include GW15 data")
        print("=" * 80)
        
        if 'player_roll5_xg' in gw15.columns:
            gw15_roll5 = gw15['player_roll5_xg'].iloc[0]
            
            # Calculate what it SHOULD be (GW10-14 average)
            gw10_14 = salah[(salah['gameweek'] >= 10) & (salah['gameweek'] <= 14)]
            if 'xg' in gw10_14.columns:
                expected_roll5 = gw10_14['xg'].mean()
                
                print(f"\nGW15 player_roll5_xg value: {gw15_roll5:.3f}")
                print(f"Expected (GW10-14 xg avg): {expected_roll5:.3f}")
                print(f"Difference: {abs(gw15_roll5 - expected_roll5):.3f}")
                
                if abs(gw15_roll5 - expected_roll5) < 0.1:
                    print("✓ CORRECT: GW15 roll5_xg does NOT include GW15 data")
                else:
                    print("✗ LEAKAGE: GW15 roll5_xg appears to include GW15 data!")
        
        # Check if GW15 has actual game stats (it shouldn't for predictions)
        if 'goals' in gw15.columns:
            gw15_goals = gw15['goals'].iloc[0]
            if pd.notna(gw15_goals) and gw15_goals > 0:
                print(f"\n⚠️ WARNING: GW15 has actual goals={gw15_goals} - this is OK if it's historical data")
            else:
                print(f"\n✓ GW15 goals is {gw15_goals} (expected for prediction)")









