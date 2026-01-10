"""
Monte Carlo Simulation-Based Bonus Points Model

Instead of predicting bonus directly, this model:
1. Predicts baseline BPS (raw score from "boring" stats)
2. Uses existing model predictions (goals, assists, CS probability)
3. Runs Monte Carlo simulation to determine bonus from BPS rankings

This is more accurate because bonus points are a ranking-based competition.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict, List, Tuple
from pathlib import Path


# BPS scoring rules (2025-26 season)
BPS_RULES = {
    # Major events (simulated)
    'goal': {
        'GK': 12, 'DEF': 12, 'MID': 18, 'FWD': 24
    },
    'assist': 9,
    'clean_sheet': {
        'GK': 12, 'DEF': 12, 'MID': 0, 'FWD': 0
    },
    # Penalties
    'goal_conceded': {
        'GK': -4, 'DEF': -4, 'MID': 0, 'FWD': 0
    },
    'yellow_card': -3,
    'red_card': -9,
    'own_goal': -6,
    'penalty_miss': -6,
    'penalty_save': 15,
}


def normalize_team_name(name: str) -> str:
    """
    Normalize team name for consistent match grouping.
    
    Handles variations like:
    - 'brighton__hove_albion' vs 'Brighton'
    - 'manchester_city' vs 'Man City'
    - 'manchester_united' vs 'Man Utd'
    """
    if pd.isna(name):
        return ''
    
    name = str(name).lower().strip()
    # Replace underscores with spaces and normalize whitespace
    name = name.replace('_', ' ').replace('  ', ' ').strip()
    
    # Canonical mappings - map all variations to a standard short name
    mappings = {
        # Full names to short
        'brighton hove albion': 'brighton',
        'brighton  hove albion': 'brighton',
        'manchester city': 'man city',
        'manchester united': 'man utd',
        'tottenham hotspur': 'spurs',
        'tottenham': 'spurs',
        'wolverhampton wanderers': 'wolves',
        'wolverhampton': 'wolves',
        'nottingham forest': 'forest',
        'nottham forest': 'forest',
        "nott'm forest": 'forest',
        'newcastle united': 'newcastle',
        'west ham united': 'west ham',
        'crystal palace': 'palace',
        'aston villa': 'villa',
        'leeds united': 'leeds',
        'leicester city': 'leicester',
        # Short variations
        'man city': 'man city',
        'man utd': 'man utd',
    }
    
    # Check exact matches first
    if name in mappings:
        return mappings[name]
    
    # Check partial matches
    for full_name, short_name in mappings.items():
        if full_name in name:
            return short_name
    
    # Default: return first word (handles most cases like 'arsenal', 'chelsea', etc.)
    # But keep two-word names that aren't in mappings
    words = name.split()
    if len(words) == 1:
        return words[0]
    elif len(words) == 2 and words[0] in ['west', 'man', 'aston', 'crystal', 'leeds', 'leicester']:
        return name  # Keep compound names
    else:
        return words[0]  # Just first word for long names


class BaselineBPSModel:
    """
    Predicts baseline BPS score from "boring" stats.
    
    This excludes goals, assists, and clean sheets - those are simulated separately.
    Baseline BPS comes from: passes, tackles, recoveries, saves, etc.
    
    Also includes yellow/red card history since cards reduce BPS (-3 for yellow, -9 for red).
    """
    
    FEATURES = [
        # Passing/possession stats (primary contributors to baseline BPS)
        'touches_per90_roll5',
        'passes_per90_roll5', 
        'passes_completed_per90_roll5',
        'progressive_passes_per90_roll5',
        'key_passes_per90_roll5',
        'key_passes_per90_roll3',
        'key_passes_per90_roll1',
        
        # Defensive stats (contribute to BPS)
        'tackles_per90_roll5',
        'tackles_per90_roll3',
        'interceptions_per90_roll5',
        'interceptions_per90_roll3',
        'clearances_per90_roll5',
        'blocks_per90_roll5',
        'recoveries_per90_roll5',
        
        # SCA/GCA (shot creating actions)
        'sca_per90_roll5',
        'sca_per90_roll3',
        'sca_per90_roll1',
        'gca_per90_roll5',
        'gca_per90_roll3',
        
        # Shots (contribute to BPS even if not scored)
        'shots_per90_roll5',
        'shots_per90_roll3',
        'shots_on_target_per90_roll5',
        
        # Recent scoring form (indicates attacking involvement, correlates with baseline BPS)
        'goals_per90_roll5',
        'goals_per90_roll3',
        'assists_per90_roll5',
        'assists_per90_roll3',
        'xg_per90_roll5',
        'xa_per90_roll5',
        
        # Yellow/Red card history (negative BPS contributors: -3 yellow, -9 red)
        'yellow_cards_roll5',       # Yellow cards in last 5 games
        'yellow_cards_roll10',      # Yellow cards in last 10 games
        'yellow_card_rate',         # Cards per game rate
        'red_cards_roll20',         # Red cards in last 20 games (rare)
        
        # LIFETIME PLAYER PROFILE (career-long stats)
        'lifetime_goals_per90',             # Career goals per 90 mins
        'lifetime_assists_per90',           # Career assists per 90 mins
        'lifetime_xg_per90',                # Career xG per 90 mins
        'lifetime_xag_per90',               # Career xAG per 90 mins
        'lifetime_key_passes_per90',        # Career key passes per 90 mins
        'lifetime_sca_per90',               # Career SCA per 90 mins
        'lifetime_gca_per90',               # Career GCA per 90 mins
        'lifetime_defcon_per90',            # Career defensive contribution per 90
        'lifetime_tackles_per90',           # Career tackles per 90 mins
        'lifetime_interceptions_per90',     # Career interceptions per 90 mins
        'lifetime_minutes',                 # Total career minutes (experience indicator)
        
        # Position indicators (different baseline BPS by position)
        'is_forward',
        'is_midfielder',
        'is_defender',
        'is_goalkeeper',
        
        # Match context
        'is_home',
        
        # Team/opponent context
        'team_xg_roll5',
        'team_goals_roll5',
        'opp_xg_against_roll5',
        'opp_conceded_roll5',
        
        # Minutes (baseline BPS scales with time played)
        'roll5_minutes_avg',
        'roll3_minutes_avg',
        
        # Season indicators (BPS rules change slightly each season)
        'season_2021_22',
        'season_2022_23',
        'season_2023_24',
        'season_2024_25',
        'season_2025_26',
    ]
    
    TARGET = 'baseline_bps'  # We'll compute this from historical data
    
    def __init__(self, **xgb_params):
        default_params = {
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBRegressor(**default_params)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._features_used = []
    
    def _add_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position indicator features."""
        df = df.copy()
        pos = df['position'].fillna('').str.upper()
        
        df['is_forward'] = pos.str.contains('FW|CF|ST|LW|RW').astype(int)
        df['is_midfielder'] = pos.str.contains('CM|DM|AM|LM|RM|MF').astype(int)
        df['is_defender'] = pos.str.contains('CB|LB|RB|WB|DF').astype(int)
        df['is_goalkeeper'] = pos.str.contains('GK').astype(int)
        
        return df
    
    def _add_season_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add season one-hot encoded features."""
        df = df.copy()
        season = df['season'].fillna('') if 'season' in df.columns else pd.Series('', index=df.index)
        
        # One-hot encode seasons (BPS rules change slightly each season)
        df['season_2021_22'] = (season == '2021-22').astype(int)
        df['season_2022_23'] = (season == '2022-23').astype(int)
        df['season_2023_24'] = (season == '2023-24').astype(int)
        df['season_2024_25'] = (season == '2024-25').astype(int)
        df['season_2025_26'] = (season == '2025-26').astype(int)
        
        return df
    
    def _compute_baseline_bps(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute baseline BPS by subtracting major event BPS from total BPS.
        
        baseline_bps = bps - (goals * goal_bps) - (assists * 9) - (cs * cs_bps) + penalties
        """
        df = df.copy()
        
        # Get position for position-dependent BPS
        fpl_pos = df.get('fpl_position', 'MID')
        
        # Calculate BPS from major events
        goal_bps = df.apply(lambda r: BPS_RULES['goal'].get(r.get('fpl_position', 'MID'), 18), axis=1)
        cs_bps = df.apply(lambda r: BPS_RULES['clean_sheet'].get(r.get('fpl_position', 'MID'), 0), axis=1)
        
        # Handle goals and assists
        if 'goals' in df.columns:
            goals = df['goals'].fillna(0)
        else:
            goals = pd.Series(0, index=df.index)
        
        if 'assists' in df.columns:
            assists = df['assists'].fillna(0)
        else:
            assists = pd.Series(0, index=df.index)
        
        # Clean sheet: 1 if opponent_goals == 0 and player played 60+ mins
        if 'opponent_goals' in df.columns:
            opponent_goals = df['opponent_goals'].fillna(0)
        else:
            opponent_goals = pd.Series(1, index=df.index)  # Default to 1 (no CS)
        
        if 'minutes' in df.columns:
            minutes = df['minutes'].fillna(0)
        else:
            minutes = pd.Series(60, index=df.index)
        
        clean_sheet = ((opponent_goals == 0) & (minutes >= 60)).astype(int)
        
        # Goals conceded penalty
        concede_penalty = df.apply(
            lambda r: BPS_RULES['goal_conceded'].get(r.get('fpl_position', 'MID'), 0) * opponent_goals[r.name]
            if minutes[r.name] >= 60 else 0, 
            axis=1
        )
        
        # Total BPS from major events
        major_event_bps = (goals * goal_bps) + (assists * 9) + (clean_sheet * cs_bps) + concede_penalty
        
        # Baseline = total - major events
        if 'bps' in df.columns:
            total_bps = df['bps'].fillna(0)
        else:
            total_bps = pd.Series(0, index=df.index)
        
        baseline = total_bps - major_event_bps
        
        # Floor at 0 (can't have negative baseline in practice)
        return np.maximum(baseline, 0)
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train the baseline BPS model."""
        df = df.copy()
        df = self._add_position_features(df)
        df = self._add_season_features(df)
        
        # Only train on players who played 60+ mins (bonus eligible)
        played_mask = (df['minutes'] >= 60) if 'minutes' in df.columns else pd.Series(True, index=df.index)
        df = df[played_mask].copy()
        
        # Compute target: baseline BPS
        if 'bps' in df.columns:
            # We have actual BPS data - compute baseline by subtracting major events
            df['baseline_bps'] = self._compute_baseline_bps(df)
            self._has_bps_data = True
            if verbose:
                print("  Using actual BPS data for training")
        else:
            # No BPS data - estimate baseline from stats directly
            # Use a heuristic based on typical BPS contributions
            df['baseline_bps'] = self._estimate_baseline_bps(df)
            self._has_bps_data = False
            if verbose:
                print("  Estimating baseline BPS from stats (no actual BPS data)")
        
        # Get available features
        available_features = [f for f in self.FEATURES if f in df.columns]
        
        for feat in available_features:
            df[feat] = df[feat].fillna(0)
        
        X = df[available_features].fillna(0).astype(float)
        y = df['baseline_bps'].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Sample weights by minutes
        sample_weights = df['minutes'].values.copy()
        sample_weights = sample_weights / sample_weights.mean()
        
        if verbose:
            print(f"Training BaselineBPSModel on {len(X)} samples...")
            print(f"  Features used: {len(available_features)}")
            print(f"  Avg baseline BPS: {y.mean():.1f}")
        
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        self.is_fitted = True
        self._features_used = available_features
        
        if verbose:
            y_pred = self.model.predict(X_scaled)
            print(f"  MAE: {mean_absolute_error(y, y_pred):.2f}")
            print(f"  R²: {r2_score(y, y_pred):.3f}")
        
        return self
    
    def _estimate_baseline_bps(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate baseline BPS from stats when actual BPS data is not available.
        
        This uses typical BPS contribution values:
        - Playing 60+ mins: ~6 points
        - Passes completed: ~1 per 10 passes
        - Tackles/interceptions: ~2-3 each
        - Recoveries: ~1 each
        - Key passes: ~2 each
        - etc.
        """
        baseline = np.zeros(len(df))
        
        # Base for playing 60+ mins
        baseline += 6  # Base appearance score
        
        # Passing contribution (estimate from rolling stats)
        if 'passes_per90_roll5' in df.columns:
            passes = df['passes_per90_roll5'].fillna(0) * (df['minutes'].fillna(0) / 90)
            baseline += passes * 0.1  # ~1 BPS per 10 passes
        
        # Key passes
        if 'key_passes_per90_roll5' in df.columns:
            key_passes = df['key_passes_per90_roll5'].fillna(0) * (df['minutes'].fillna(0) / 90)
            baseline += key_passes * 2
        
        # Tackles
        if 'tackles_per90_roll5' in df.columns:
            tackles = df['tackles_per90_roll5'].fillna(0) * (df['minutes'].fillna(0) / 90)
            baseline += tackles * 2
        
        # Interceptions
        if 'interceptions_per90_roll5' in df.columns:
            ints = df['interceptions_per90_roll5'].fillna(0) * (df['minutes'].fillna(0) / 90)
            baseline += ints * 3
        
        # Recoveries
        if 'recoveries_per90_roll5' in df.columns:
            recoveries = df['recoveries_per90_roll5'].fillna(0) * (df['minutes'].fillna(0) / 90)
            baseline += recoveries * 1
        
        # SCA (shot creating actions)
        if 'sca_per90_roll5' in df.columns:
            sca = df['sca_per90_roll5'].fillna(0) * (df['minutes'].fillna(0) / 90)
            baseline += sca * 1.5
        
        return pd.Series(baseline, index=df.index)
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict baseline BPS."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        df = df.copy()
        df = self._add_position_features(df)
        df = self._add_season_features(df)
        
        for feat in self._features_used:
            if feat not in df.columns:
                df[feat] = 0
        
        X = df[self._features_used].fillna(0).astype(float)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        
        # Floor at 0
        return np.maximum(preds, 0)
    
    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        return pd.DataFrame({
            'feature': self._features_used,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


class BonusModelMC:
    """
    Monte Carlo simulation-based bonus model.
    
    Instead of predicting bonus directly, this model:
    1. Uses a BaselineBPSModel to predict baseline BPS (from "boring" stats)
    2. Uses existing model predictions (goals, assists, CS probability)
    3. Runs Monte Carlo simulation to determine bonus from BPS rankings
    """
    
    TARGET = 'bonus'
    
    # Class-level FEATURES for compatibility with feature selection
    FEATURES = BaselineBPSModel.FEATURES.copy()
    
    def __init__(self, n_simulations: int = 1000, **xgb_params):
        self.n_simulations = n_simulations
        self.baseline_model = BaselineBPSModel(**xgb_params)
        self.is_fitted = False
        
        # Use the class-level FEATURES (which can be modified by feature selection)
        self.baseline_model.FEATURES = self.__class__.FEATURES.copy()
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train the baseline BPS model."""
        if verbose:
            print(f"Training BonusModelMC (Monte Carlo, {self.n_simulations} sims)...")
        
        self.baseline_model.fit(df, verbose=verbose)
        self.is_fitted = True
        self._features_used = self.baseline_model._features_used
        
        return self
    
    def predict(self, df: pd.DataFrame, 
                pred_exp_goals: np.ndarray = None,
                pred_exp_assists: np.ndarray = None,
                pred_cs_prob: np.ndarray = None,
                pred_minutes: np.ndarray = None,
                match_groups: pd.Series = None) -> np.ndarray:
        """
        Predict expected bonus using Monte Carlo simulation.
        
        Args:
            df: DataFrame with player data and features
            pred_exp_goals: Expected goals for each player (from GoalsModel)
            pred_exp_assists: Expected assists for each player (from AssistsModel)
            pred_cs_prob: Clean sheet probability for each player's team
            pred_minutes: Predicted minutes for each player
            match_groups: Series indicating which match each player is in (e.g., 'team_vs_opponent')
        
        Returns:
            Array of expected bonus points (0-3 range, usually 0-0.5 avg)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        df = df.copy()
        n_players = len(df)
        
        # Get predictions from columns if not provided
        if pred_exp_goals is None:
            if 'pred_exp_goals' in df.columns:
                pred_exp_goals = df['pred_exp_goals'].values
            elif 'pred_goals_per90' in df.columns:
                pred_exp_goals = df['pred_goals_per90'].values
            else:
                pred_exp_goals = np.zeros(n_players)
        
        if pred_exp_assists is None:
            if 'pred_exp_assists' in df.columns:
                pred_exp_assists = df['pred_exp_assists'].values
            elif 'pred_assists_per90' in df.columns:
                pred_exp_assists = df['pred_assists_per90'].values
            else:
                pred_exp_assists = np.zeros(n_players)
        
        if pred_cs_prob is None:
            if 'pred_cs_prob' in df.columns:
                pred_cs_prob = df['pred_cs_prob'].values
            else:
                pred_cs_prob = np.full(n_players, 0.2)
        
        if pred_minutes is None:
            if 'pred_minutes' in df.columns:
                pred_minutes = df['pred_minutes'].values
            else:
                pred_minutes = np.full(n_players, 60)
        
        # Predict baseline BPS
        baseline_bps = self.baseline_model.predict(df)
        
        # Scale baseline BPS by predicted minutes (if playing less, lower baseline)
        # Normalize to 90 mins baseline
        minutes_factor = np.clip(pred_minutes / 90.0, 0.1, 1.0)
        baseline_bps = baseline_bps * minutes_factor
        
        # Get FPL position for BPS rules
        fpl_positions = df.get('fpl_position', pd.Series(['MID'] * n_players)).values
        
        # Create match groups if not provided
        if match_groups is None:
            if 'team' in df.columns and 'opponent' in df.columns:
                # Group players by match - NORMALIZE team names to ensure both teams match
                match_groups = df.apply(
                    lambda r: '_vs_'.join(sorted([
                        normalize_team_name(r.get('team', '')), 
                        normalize_team_name(r.get('opponent', ''))
                    ])),
                    axis=1
                ).values
            else:
                # Single match assumption (all players in same match)
                match_groups = np.array(['match'] * n_players)
        else:
            match_groups = match_groups.values if hasattr(match_groups, 'values') else match_groups
        
        # Run simulation
        expected_bonus = self._simulate_bonus(
            baseline_bps=baseline_bps,
            exp_goals=pred_exp_goals,
            exp_assists=pred_exp_assists,
            cs_prob=pred_cs_prob,
            fpl_positions=fpl_positions,
            match_groups=match_groups,
            pred_minutes=pred_minutes
        )
        
        return expected_bonus
    
    def _simulate_bonus(self, baseline_bps: np.ndarray, exp_goals: np.ndarray,
                        exp_assists: np.ndarray, cs_prob: np.ndarray,
                        fpl_positions: np.ndarray, match_groups: np.ndarray,
                        pred_minutes: np.ndarray) -> np.ndarray:
        """
        Run Monte Carlo simulation to compute expected bonus.
        
        VECTORIZED VERSION: Processes all simulations in batch using numpy broadcasting.
        This is ~20-50x faster than the naive loop implementation.
        
        For each simulation:
        1. Sample goals from Poisson(exp_goals)
        2. Sample assists from Poisson(exp_assists)
        3. Sample clean sheets from Bernoulli(cs_prob)
        4. Calculate total BPS
        5. Rank within each match and award 3, 2, 1 points
        """
        n_players = len(baseline_bps)
        n_sims = self.n_simulations
        
        # Pre-compute position-based BPS values (1D arrays)
        goal_bps = np.array([BPS_RULES['goal'].get(pos, 18) for pos in fpl_positions])
        cs_bps = np.array([BPS_RULES['clean_sheet'].get(pos, 0) for pos in fpl_positions])
        mins_mask = (pred_minutes >= 60).astype(np.float32)
        
        # =========================================================================
        # BATCH SAMPLING: Generate all random samples at once
        # Shape: (n_sims, n_players)
        # =========================================================================
        exp_goals_safe = np.maximum(exp_goals, 0)
        exp_assists_safe = np.maximum(exp_assists, 0)
        
        all_goals = np.random.poisson(exp_goals_safe, size=(n_sims, n_players))
        all_assists = np.random.poisson(exp_assists_safe, size=(n_sims, n_players))
        all_cs_rolls = np.random.random((n_sims, n_players))
        
        # Clean sheets: roll < prob AND player plays 60+ mins
        all_clean_sheets = ((all_cs_rolls < cs_prob) * mins_mask).astype(np.int32)
        
        # =========================================================================
        # BATCH BPS CALCULATION: Compute BPS for all simulations at once
        # Shape: (n_sims, n_players)
        # =========================================================================
        all_bps = (
            baseline_bps +  # broadcasts from (n_players,) to (n_sims, n_players)
            all_goals * goal_bps +
            all_assists * BPS_RULES['assist'] +
            all_clean_sheets * cs_bps
        )
        
        # =========================================================================
        # PRE-COMPUTE MATCH MAPPING: Build index arrays once
        # =========================================================================
        unique_matches = np.unique(match_groups)
        match_to_indices = {m: np.where(match_groups == m)[0] for m in unique_matches}
        
        # Initialize bonus accumulator
        total_bonus = np.zeros(n_players, dtype=np.float64)
        
        # =========================================================================
        # VECTORIZED PER-MATCH PROCESSING
        # Instead of (n_sims × n_matches) iterations, we do n_matches iterations
        # where each iteration processes all n_sims simulations at once
        # =========================================================================
        for match_id, match_indices in match_to_indices.items():
            n_match = len(match_indices)
            if n_match == 0:
                continue
            
            # Get BPS for all simulations for this match: shape (n_sims, n_match)
            match_bps = all_bps[:, match_indices]
            
            # Sort each simulation's match BPS (descending): shape (n_sims, n_match)
            sorted_idx = np.argsort(-match_bps, axis=1)
            
            # Get sorted BPS values using advanced indexing
            sim_range = np.arange(n_sims)[:, None]
            sorted_bps = match_bps[sim_range, sorted_idx]
            
            # Initialize match bonus: shape (n_sims, n_match)
            match_bonus = np.zeros((n_sims, n_match), dtype=np.float32)
            
            # First place: all players tied at max BPS get 3 points
            first_bps = sorted_bps[:, 0:1]  # shape (n_sims, 1)
            first_mask = (match_bps == first_bps)
            match_bonus += first_mask * 3
            
            if n_match >= 2:
                # Second place: only if not tied with first
                second_bps = sorted_bps[:, 1:2]  # shape (n_sims, 1)
                has_second = (second_bps < first_bps).astype(np.float32)
                second_mask = (match_bps == second_bps) & (match_bps < first_bps)
                match_bonus += second_mask * 2 * has_second
                
                if n_match >= 3:
                    # Third place: only if not tied with first or second
                    third_bps = sorted_bps[:, 2:3]  # shape (n_sims, 1)
                    has_third = (third_bps < second_bps).astype(np.float32)
                    third_mask = (match_bps == third_bps) & (match_bps < second_bps)
                    match_bonus += third_mask * 1 * has_third
            
            # Sum bonus across all simulations for each player in this match
            total_bonus[match_indices] += match_bonus.sum(axis=0)
        
        # Average across simulations
        expected_bonus = total_bonus / n_sims
        
        return expected_bonus
    
    def predict_simple(self, df: pd.DataFrame) -> np.ndarray:
        """
        Simple prediction without simulation (fallback).
        Uses baseline BPS model and heuristics.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        baseline_bps = self.baseline_model.predict(df)
        
        # Simple heuristic: higher baseline BPS = more likely to get bonus
        # Normalize and scale to 0-3 range
        bps_normalized = np.clip(baseline_bps / 30.0, 0, 1)  # Assume ~30 is a good baseline
        expected_bonus = bps_normalized * 0.5  # Average player gets ~0.5 bonus
        
        return np.clip(expected_bonus, 0, 3)
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance from baseline model."""
        return self.baseline_model.feature_importance()


# Backwards compatibility alias
BonusModel = BonusModelMC

