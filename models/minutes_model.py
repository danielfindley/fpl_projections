"""
Minutes Prediction Model

Predicts expected minutes for players who PLAYED (1+ minute).
Key insight: Among players who play, minutes are heavily skewed toward 90.
- ~50% play 90 minutes (full game)
- ~65% play 60+ minutes (starters)
- The remaining ~35% are subs playing 1-30 minutes

Now includes FPL API availability features:
- Status flags (available, injured, unavailable, doubtful, suspended)
- Chance of playing next round (0-100%)
- Hard filter for 0% chance or injured/unavailable status
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path


class MinutesModel:
    """
    XGBoost model that predicts minutes assuming player WILL play.
    Uses heavy weighting on 90-minute outcomes since they dominate the distribution.
    
    Includes FPL API availability features for prediction mode:
    - fpl_chance_of_playing: Numerical 0-100
    - fpl_status_* flags: One-hot encoded availability status
    
    Hard filter applied: players with 0% chance or injured/unavailable status
    get pred_minutes = 0.
    """
    
    FEATURES = [
        # Minutes history - key predictors
        'last_game_minutes',
        'roll3_minutes_avg',
        'roll5_minutes_avg',
        'roll5_minutes_sum',
        'roll10_minutes_avg',
        
        # Starting likelihood
        'roll5_starts',
        'roll5_full_90s',
        'last_was_start',
        'last_was_full_90',
        'starter_score',
        'full_90_rate',
        
        # LIFETIME PLAYER PROFILE (career-long stats for minutes prediction)
        'lifetime_minutes',                   # Total career minutes (experience indicator)
        'lifetime_minutes_per_appearance',    # Avg minutes when playing (consistency)
        'lifetime_goal_involvements_per90',   # Career G+A per 90 (player quality indicator)
        
        # Productivity
        'goals_roll5',
        'assists_roll5',
        'goal_involvements_roll5',
        
        # Position indicators
        'is_goalkeeper',
        'is_defender',
        'is_midfielder',
        'is_forward',
        
        # Match context
        'is_home',
        
        # Team context
        'team_goals_roll5',
        
        # FPL API availability features (used in prediction only)
        # Numerical: chance of playing (0-100, scaled to 0-1)
        'fpl_chance_of_playing_scaled',
        
        # One-hot status flags
        'fpl_status_available',
        'fpl_status_injured',
        'fpl_status_unavailable',
        'fpl_status_doubtful',
        'fpl_status_suspended',
    ]
    
    # Features used during training (historical data doesn't have FPL status)
    TRAINING_FEATURES = [f for f in FEATURES if not f.startswith('fpl_')]
    
    # Map notebook feature names to model feature names
    FEATURE_ALIASES = {
        'player_roll5_minutes': 'roll5_minutes_avg',
        'player_roll5_goals': 'goals_roll5',
        'player_roll5_assists': 'assists_roll5',
    }
    
    TARGET = 'minutes'
    
    def __init__(self, use_fpl_status: bool = True, **xgb_params):
        """
        Initialize MinutesModel.
        
        Args:
            use_fpl_status: If True, use FPL availability features during prediction
                           and apply hard filters. Default True.
            **xgb_params: XGBoost parameters
        """
        default_params = {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.1,
            'random_state': 42,
            'min_child_weight': 5,
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBRegressor(**default_params)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.train_mean = 67.0
        self.use_fpl_status = use_fpl_status
    
    def _add_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position indicator features."""
        if 'position' in df.columns:
            df['is_goalkeeper'] = df['position'].apply(lambda x: 1 if 'GK' in str(x).upper() else 0)
            df['is_defender'] = df['position'].apply(lambda x: 1 if any(p in str(x).upper() for p in ['CB', 'LB', 'RB', 'WB', 'DF']) else 0)
            df['is_midfielder'] = df['position'].apply(lambda x: 1 if any(p in str(x).upper() for p in ['CM', 'DM', 'AM', 'LM', 'RM', 'MF']) else 0)
            df['is_forward'] = df['position'].apply(lambda x: 1 if any(p in str(x).upper() for p in ['FW', 'CF', 'ST', 'LW', 'RW']) else 0)
        else:
            for col in ['is_goalkeeper', 'is_defender', 'is_midfielder', 'is_forward']:
                if col not in df.columns:
                    df[col] = 0
        return df
    
    def _map_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map notebook feature names to model feature names."""
        df = df.copy()
        for alias, target in self.FEATURE_ALIASES.items():
            if alias in df.columns and target not in df.columns:
                df[target] = df[alias]
        return df
    
    def prepare_features(self, df: pd.DataFrame, compute_rolling: bool = True) -> pd.DataFrame:
        """
        Prepare features from raw data.
        
        Args:
            df: Input dataframe
            compute_rolling: If True, compute rolling features. Set to False if 
                           features are already computed (e.g., from notebook).
        """
        df = df.copy()
        
        # Map any aliases first
        df = self._map_aliases(df)
        
        # Ensure is_home is numeric
        if 'is_home' in df.columns:
            df['is_home'] = pd.to_numeric(df['is_home'].replace({True: 1, False: 0, 'True': 1, 'False': 0}), errors='coerce').fillna(0).astype(int)
        else:
            df['is_home'] = 0
        
        # Minutes - handle case where column doesn't exist (upcoming gameweeks)
        if 'minutes' in df.columns:
            df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(0)
        else:
            df['minutes'] = 0
        
        # Add position features
        df = self._add_position_features(df)
        
        # Team goals
        if 'team_goals_roll5' not in df.columns:
            df['team_goals_roll5'] = 0
        
        # Only compute rolling features if requested AND we have full history
        # Check if we have multiple gameweeks per player (indicating full history)
        has_full_history = False
        if 'player_id' in df.columns and 'gameweek' in df.columns:
            games_per_player = df.groupby('player_id')['gameweek'].count()
            has_full_history = games_per_player.mean() > 1
        
        if compute_rolling and has_full_history:
            df = df.sort_values(['player_id', 'season', 'gameweek'])
            
            # Last game stats
            df['last_game_minutes'] = df.groupby('player_id')['minutes'].shift(1)
            if 'goals' in df.columns:
                df['last_game_goals'] = df.groupby('player_id')['goals'].shift(1)
            if 'assists' in df.columns:
                df['last_game_assists'] = df.groupby('player_id')['assists'].shift(1)
            
            # Starting/90 min indicators
            df['was_start'] = (df['minutes'] >= 60).astype(int)
            df['was_full_90'] = (df['minutes'] >= 89).astype(int)
            df['last_was_start'] = df.groupby('player_id')['was_start'].shift(1).fillna(0)
            df['last_was_full_90'] = df.groupby('player_id')['was_full_90'].shift(1).fillna(0)
            
            # Rolling features
            df['roll3_minutes_avg'] = df.groupby('player_id')['minutes'].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean()
            )
            df['roll5_minutes_avg'] = df.groupby('player_id')['minutes'].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )
            df['roll5_minutes_sum'] = df.groupby('player_id')['minutes'].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).sum()
            )
            df['roll10_minutes_avg'] = df.groupby('player_id')['minutes'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=1).mean()
            )
            
            if 'goals' in df.columns:
                df['goals_roll5'] = df.groupby('player_id')['goals'].transform(
                    lambda x: x.shift(1).rolling(5, min_periods=1).sum()
                ).fillna(0)
            if 'assists' in df.columns:
                df['assists_roll5'] = df.groupby('player_id')['assists'].transform(
                    lambda x: x.shift(1).rolling(5, min_periods=1).sum()
                ).fillna(0)
            
            # Starts and full 90s
            df['roll5_starts'] = df.groupby('player_id')['was_start'].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).sum()
            )
            df['roll5_full_90s'] = df.groupby('player_id')['was_full_90'].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).sum()
            )
            df['full_90_rate'] = df.groupby('player_id')['was_full_90'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=1).mean()
            ).fillna(0.5)
            
            # Clean up
            df = df.drop(columns=['was_start', 'was_full_90'], errors='ignore')
        
        # Compute derived features from what we have
        if 'goals_roll5' in df.columns and 'assists_roll5' in df.columns:
            df['goal_involvements_roll5'] = df['goals_roll5'].fillna(0) + df['assists_roll5'].fillna(0)
        elif 'goal_involvements_roll5' not in df.columns:
            df['goal_involvements_roll5'] = 0
        
        # Starter score (compute from available features)
        roll5_avg = df.get('roll5_minutes_avg', pd.Series([self.train_mean] * len(df)))
        roll5_full = df.get('roll5_full_90s', pd.Series([2.5] * len(df)))
        last_full = df.get('last_was_full_90', pd.Series([0.5] * len(df)))
        
        df['starter_score'] = (
            (roll5_avg.fillna(self.train_mean) / 90) * 0.4 +
            (roll5_full.fillna(2.5) / 5) * 0.4 +
            (last_full.fillna(0.5)) * 0.2
        )
        
        # Fill missing features with sensible defaults
        defaults = {
            'last_game_minutes': self.train_mean,
            'roll3_minutes_avg': self.train_mean,
            'roll5_minutes_avg': self.train_mean,
            'roll5_minutes_sum': self.train_mean * 5,
            'roll10_minutes_avg': self.train_mean,
            'roll5_starts': 3,
            'roll5_full_90s': 2.5,
            'last_was_start': 0.6,
            'last_was_full_90': 0.5,
            'full_90_rate': 0.5,
            'starter_score': 0.5,
            'goals_roll5': 0,
            'assists_roll5': 0,
            'goal_involvements_roll5': 0,
        }
        
        for feat, default in defaults.items():
            if feat not in df.columns:
                df[feat] = default
            else:
                df[feat] = df[feat].fillna(default)
        
        # Ensure all required features exist
        for feat in self.FEATURES:
            if feat not in df.columns:
                df[feat] = 0
        
        # Add FPL status features (defaults for training data)
        df = self._add_fpl_status_features(df)
        
        return df
    
    def _add_fpl_status_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add FPL availability status features.
        
        For training data (historical), these default to 'available' status.
        For prediction data, these should be populated from FPL API.
        """
        df = df.copy()
        
        # Scale chance of playing to 0-1 range
        if 'fpl_chance_of_playing' in df.columns:
            df['fpl_chance_of_playing_scaled'] = df['fpl_chance_of_playing'].fillna(100) / 100.0
        elif 'fpl_chance_of_playing_scaled' not in df.columns:
            df['fpl_chance_of_playing_scaled'] = 1.0  # Default: 100% available
        
        # One-hot status flags (defaults to available for training data)
        status_cols = {
            'fpl_status_available': 1,  # Default: available
            'fpl_status_injured': 0,
            'fpl_status_unavailable': 0,
            'fpl_status_doubtful': 0,
            'fpl_status_suspended': 0,
        }
        
        for col, default in status_cols.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)
        
        return df
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """
        Train the model on players who played at least 1 minute.
        
        Note: Training uses TRAINING_FEATURES (without FPL status) since
        historical data doesn't have FPL API status information.
        """
        # Check if features are already computed (from notebook)
        has_precomputed = all(feat in df.columns for feat in ['roll5_minutes_avg', 'roll5_full_90s', 'full_90_rate'])
        
        # Only compute rolling features if not already present
        df = self.prepare_features(df, compute_rolling=not has_precomputed)
        
        df = df.copy()
        
        # Filter: players who played at least 1 minute
        played_mask = df['minutes'] >= 1
        df_played = df[played_mask].copy()
        
        # Also need some history
        has_history = df_played['last_game_minutes'].notna()
        df_played = df_played[has_history].copy()
        
        # Use TRAINING_FEATURES for fit (no FPL status in historical data)
        # However, if this model was created with modified FEATURES via feature selection,
        # use those features filtered to exclude FPL ones
        training_features = [f for f in self.FEATURES if not f.startswith('fpl_')]
        
        X = df_played[training_features].fillna(0).astype(float)
        y = df_played[self.TARGET]
        
        # Learn the distribution
        self.train_mean = y.mean()
        self.train_median = y.median()
        self.pct_full_90 = (y >= 89).mean()
        self.pct_60_plus = (y >= 60).mean()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # HEAVY sample weighting toward 90 minute outcomes
        sample_weights = np.ones(len(y))
        sample_weights[y >= 89] = 3.0
        sample_weights[(y >= 60) & (y < 89)] = 2.0
        sample_weights[y < 30] = 0.5
        
        if verbose:
            print(f"Training MinutesModel on {len(X)} samples (players who played 1+ min)...")
            print(f"  Features used: {len(training_features)} (excluding FPL status for training)")
            print(f"  Actual distribution:")
            print(f"    Mean: {y.mean():.1f}, Median: {y.median():.1f}")
            print(f"    90 min: {(y >= 89).sum()} ({(y >= 89).mean():.1%})")
            print(f"    60+ min: {(y >= 60).sum()} ({(y >= 60).mean():.1%})")
            print(f"    <30 min: {(y < 30).sum()} ({(y < 30).mean():.1%})")
        
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        self.is_fitted = True
        # Store which features were actually used for training
        self._training_features_used = training_features
        self._features_used = self.FEATURES.copy()
        
        if verbose:
            y_pred = self.model.predict(X_scaled)
            y_pred = np.clip(y_pred, 1, 90)
            print(f"\n  Model predictions:")
            print(f"    Mean: {y_pred.mean():.1f}, Median: {np.median(y_pred):.1f}")
            print(f"    MAE: {mean_absolute_error(y, y_pred):.1f}")
            print(f"    R²: {r2_score(y, y_pred):.3f}")
            print(f"    Predicted 90 min: {(y_pred >= 89).sum()} ({(y_pred >= 89).mean():.1%})")
            print(f"    Predicted 60+ min: {(y_pred >= 60).sum()} ({(y_pred >= 60).mean():.1%})")
        
        return self
    
    def predict(self, df: pd.DataFrame, apply_fpl_filter: bool = True, 
                verbose: bool = False) -> np.ndarray:
        """
        Predict expected minutes.
        
        Args:
            df: DataFrame with player features
            apply_fpl_filter: If True and use_fpl_status is enabled, apply hard
                             filters for injured/unavailable players. Default True.
            verbose: If True, print details about filtered players.
        
        Returns:
            Array of predicted minutes. Players with 0% chance or
            injured/unavailable status get 0 minutes.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # For prediction, DON'T recompute rolling features - use what's in the dataframe
        # The notebook already computed these on the full history
        df = self.prepare_features(df, compute_rolling=False)
        df = df.copy()
        
        # Use training features for model prediction (same features used during fit)
        training_features = getattr(self, '_training_features_used', 
                                    [f for f in self.FEATURES if not f.startswith('fpl_')])
        
        X = df[training_features].fillna(0).astype(float)
        X_scaled = self.scaler.transform(X)
        
        # Get raw predictions
        preds = self.model.predict(X_scaled)
        preds = np.clip(preds, 1, 90)
        
        # Post-process: boost predictions based on historical patterns
        roll5_avg = df['roll5_minutes_avg'].values
        
        for i in range(len(preds)):
            if roll5_avg[i] >= 80:
                preds[i] = max(preds[i], 88)
            elif roll5_avg[i] >= 70:
                preds[i] = max(preds[i], 80)
            elif roll5_avg[i] >= 55:
                preds[i] = max(preds[i], 70)
        
        preds = np.clip(preds, 1, 90)
        
        # Apply FPL availability filter if enabled
        if apply_fpl_filter and self.use_fpl_status:
            preds = self._apply_fpl_availability_filter(df, preds, verbose)
        
        return preds
    
    def _apply_fpl_availability_filter(self, df: pd.DataFrame, preds: np.ndarray,
                                        verbose: bool = False) -> np.ndarray:
        """
        Apply hard filter based on FPL availability status.
        
        Rules:
        - fpl_chance_of_playing == 0: Set to 0 minutes
        - fpl_status in ('i', 'u', 's'): Set to 0 minutes (injured, unavailable, suspended)
        - fpl_chance_of_playing < 25: Set to 0 minutes
        - fpl_chance_of_playing 25-74: Scale down by chance/100
        """
        preds = preds.copy()
        
        # Get FPL status columns
        chance = df.get('fpl_chance_of_playing', pd.Series([100] * len(df))).fillna(100).values
        status = df.get('fpl_status', pd.Series(['a'] * len(df))).fillna('a').values
        
        # Track which players were filtered
        filtered_zero = []
        filtered_scaled = []
        
        for i in range(len(preds)):
            player_chance = chance[i]
            player_status = status[i]
            original_pred = preds[i]
            
            # Hard filter: 0% chance or injured/unavailable/suspended
            if player_chance == 0 or player_status in ('i', 'u', 's'):
                preds[i] = 0
                if verbose and original_pred > 0:
                    name = df.iloc[i].get('player_name', f'Player_{i}')
                    filtered_zero.append((name, player_status, player_chance, original_pred))
            
            # Low chance filter: <25% -> 0
            elif player_chance < 25:
                preds[i] = 0
                if verbose and original_pred > 0:
                    name = df.iloc[i].get('player_name', f'Player_{i}')
                    filtered_zero.append((name, player_status, player_chance, original_pred))
            
            # Medium chance filter: 25-74% -> scale down
            elif player_chance < 75:
                scaled = original_pred * (player_chance / 100)
                preds[i] = scaled
                if verbose:
                    name = df.iloc[i].get('player_name', f'Player_{i}')
                    filtered_scaled.append((name, player_status, player_chance, original_pred, scaled))
        
        if verbose:
            if filtered_zero:
                print(f"  [MinutesModel] Zeroed {len(filtered_zero)} players due to FPL status:")
                for name, status, chance, orig in filtered_zero[:10]:
                    print(f"    {name:25s} (status={status}, chance={chance}%, was {orig:.0f})")
                if len(filtered_zero) > 10:
                    print(f"    ... and {len(filtered_zero) - 10} more")
            
            if filtered_scaled:
                print(f"  [MinutesModel] Scaled {len(filtered_scaled)} players due to FPL doubt:")
                for name, status, chance, orig, scaled in filtered_scaled[:5]:
                    print(f"    {name:25s} (chance={chance}%, {orig:.0f} -> {scaled:.0f})")
        
        return preds
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        # Use _training_features_used (features the model was actually trained on)
        # This excludes FPL status features which aren't used during training
        features = getattr(self, '_training_features_used', 
                          [f for f in self.FEATURES if not f.startswith('fpl_')])
        
        return pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
