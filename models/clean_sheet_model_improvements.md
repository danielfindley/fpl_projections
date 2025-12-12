# Clean Sheet Model Improvements

## Current Issue
Arsenal (elite defense, xGA 0.38) getting only 18% CS probability at home vs Brentford seems too low.

## Root Causes Identified

1. **Over-weighting opponent recent form**: Brentford's 2.16 xG roll5 and 3 goals last game are heavily weighted
2. **No home/away separation**: Arsenal's defensive stats combine home + away (they're much better at home)
3. **Missing relative strength**: Model doesn't account for Arsenal being elite vs Brentford being mid-table
4. **No interaction features**: `is_home` is just a binary, doesn't interact with defensive strength
5. **Recent form bias**: `opp_goals_scored_last1 = 3.0` might be weighted too heavily

## Recommended Improvements

### 1. Add Home/Away Separated Stats (HIGH PRIORITY)
```python
# Add to FEATURES:
'team_xga_roll5_home',
'team_xga_roll5_away',
'team_clean_sheets_roll5_home',
'team_clean_sheets_roll5_away',
'opp_xg_roll5_home',  # Opponent's xG when playing at home
'opp_xg_roll5_away',  # Opponent's xG when playing away
```

### 2. Add Relative Strength Features (HIGH PRIORITY)
```python
# Add to FEATURES:
'xga_xg_ratio',  # team_xga_roll5 / opp_xg_roll5 (lower = better for CS)
'defensive_advantage',  # opp_xg_roll5 - team_xga_roll5 (negative = advantage)
'cs_rate_vs_opp_xg',  # team_clean_sheets_roll10 / (opp_xg_roll5 + 1)
```

### 3. Add Interaction Features (MEDIUM PRIORITY)
```python
# Add to FEATURES:
'is_home_x_xga',  # is_home * team_xga_roll5 (home teams with low xGA = high CS)
'is_home_x_cs_rate',  # is_home * (team_clean_sheets_roll10 / 10)
```

### 4. Add Form Momentum Features (MEDIUM PRIORITY)
```python
# Add to FEATURES:
'team_xga_trend',  # team_xga_roll3 - team_xga_roll10 (improving = negative)
'opp_xg_trend',  # opp_xg_roll3 - opp_xg_roll10 (improving = positive, worse for CS)
'recent_vs_longterm',  # team_goals_conceded_roll3 / team_goals_conceded_roll10
```

### 5. Add Strength-Adjusted Features (LOW PRIORITY)
```python
# Add to FEATURES:
'team_strength_rank',  # League position or strength rating
'opp_strength_rank',
'strength_differential',  # team_strength - opp_strength
```

### 6. Improve Feature Engineering
- Weight recent games more (exponential decay instead of simple average)
- Use opponent's away xG when predicting home team CS (more relevant)
- Add "expected goals against" based on Poisson: exp(-opp_xg_roll5)

## Implementation Priority

1. **Immediate**: Add relative strength features (xga_xg_ratio, defensive_advantage)
2. **Short-term**: Add home/away separated stats
3. **Medium-term**: Add interaction features and form momentum
4. **Long-term**: Add strength rankings and advanced feature engineering

## Expected Impact

- **Relative strength features**: Should increase CS probability for elite defenses vs weaker attacks
- **Home/away separation**: Should better capture home advantage for strong teams
- **Interaction features**: Should properly weight home advantage for elite defenses
- **Form momentum**: Should reduce over-reaction to single-game spikes

## Code Changes Needed

See `clean_sheet_model_v2.py` for implementation.

