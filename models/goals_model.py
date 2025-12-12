"""
Goals Prediction Model

Predicts goals per 90 minutes rate for a player against a given opponent.
Final expected goals = (goals_per90) * (pred_minutes / 90)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path


class GoalsModel:
    """XGBoost model to predict player goals per 90."""
    
    FEATURES = [
        # Player per-90 rates (key predictors)
        'goals_per90_roll1',
        'goals_per90_roll3',
        'xg_per90_roll1',
        'xg_per90_roll3',
        'shots_per90_roll1',
        'shots_per90_roll3',
        'sca_per90_roll1',
        'sca_per90_roll3',
        'gca_per90_roll1',
        'gca_per90_roll3',
        'key_passes_per90_roll1',
        'key_passes_per90_roll3',
        'goals_per90_roll5',
        'goals_per90_roll10',
        'xg_per90_roll5',
        'shots_per90_roll5',
        'sca_per90_roll5',
        'gca_per90_roll5',
        
        # Recent form
        'goals_last1',
        'goals_roll3',
        'xg_roll3',
        
        # Team context (attacking strength)
        'team_xg_roll5',
        'team_goals_roll5',
        'team_shots_roll5',
        
        # Opponent context (defensive weakness)
        'opp_conceded_roll5',
        'opp_xg_against_roll5',
        
        # Match context
        'is_home',
        
        # Position indicators
        'is_forward',
        'is_midfielder',
        'is_defender',
        'is_goalkeeper',
    ]
    
    TARGET = 'goals_per90'
    
    def __init__(self, **xgb_params):
        default_params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.08,
            'random_state': 42,
            'min_child_weight': 3,
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBRegressor(**default_params)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _add_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position indicator features."""
        df = df.copy()
        pos = df['position'].fillna('').str.upper()
        
        df['is_forward'] = pos.str.contains('FW|CF|ST|LW|RW').astype(int)
        df['is_midfielder'] = pos.str.contains('CM|DM|AM|LM|RM|MF').astype(int)
        df['is_defender'] = pos.str.contains('CB|LB|RB|WB|DF').astype(int)
        df['is_goalkeeper'] = pos.str.contains('GK').astype(int)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare per-90 features."""
        df = df.copy()
        df = df.sort_values(['player_id', 'season', 'gameweek'])
        
        # Ensure minutes exists
        df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(0) if 'minutes' in df.columns else 0
        mins_90 = np.maximum(df['minutes'] / 90, 0.01)  # Avoid division by zero
        
        # Per-90 rates for current game
        df['goals_per90'] = df['goals'] / mins_90
        df['xg_per90'] = df.get('xg', 0) / mins_90
        df['shots_per90'] = df.get('shots', 0) / mins_90
        df['sca_per90'] = df.get('sca', 0) / mins_90
        df['gca_per90'] = df.get('gca', 0) / mins_90
        
        # Rolling per-90 rates (using shift to avoid leakage)
        for stat in ['goals', 'xg', 'shots', 'sca', 'gca']:
            per90_col = f'{stat}_per90'
            if per90_col in df.columns:
                df[f'{stat}_per90_roll5'] = df.groupby('player_id')[per90_col].transform(
                    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
                )
                df[f'{stat}_per90_roll10'] = df.groupby('player_id')[per90_col].transform(
                    lambda x: x.shift(1).rolling(10, min_periods=1).mean()
                )
        
        # Recent form (raw counts)
        df['goals_last1'] = df.groupby('player_id')['goals'].shift(1).fillna(0)
        df['goals_roll3'] = df.groupby('player_id')['goals'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).sum()
        ).fillna(0)
        df['xg_roll3'] = df.groupby('player_id')['xg'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).sum()
        ).fillna(0) if 'xg' in df.columns else 0
        
        # Position features
        df = self._add_position_features(df)
        
        # Ensure is_home is numeric
        if 'is_home' in df.columns:
            df['is_home'] = pd.to_numeric(df['is_home'].replace({True: 1, False: 0, 'True': 1, 'False': 0}), errors='coerce').fillna(0)
        else:
            df['is_home'] = 0
        
        # Team/opponent features (fill if missing)
        for col in ['team_xg_roll5', 'team_goals_roll5', 'team_shots_roll5', 'opp_conceded_roll5', 'opp_xg_against_roll5']:
            if col not in df.columns:
                df[col] = 0
        
        # Fill all NaN
        for feat in self.FEATURES:
            if feat in df.columns:
                df[feat] = df[feat].fillna(0)
            else:
                df[feat] = 0
        
        return df
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train the model on players who played at least 1 minute."""
        if 'goals_per90_roll5' not in df.columns:
            df = self.prepare_features(df)
        
        df = df.copy()
        
        # Filter to players who actually played (1+ minutes)
        played_mask = df['minutes'] >= 1
        df_played = df[played_mask].copy()
        
        # Ensure features exist
        for feat in self.FEATURES:
            if feat not in df_played.columns:
                df_played[feat] = 0
        
        X = df_played[self.FEATURES].fillna(0).astype(float)
        y = df_played['goals_per90'].fillna(0)
        
        # Cap extreme values (rare multi-goal games in few minutes)
        y = np.clip(y, 0, 3)  # Max 3 goals per 90
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Sample weights: upweight goalscorers
        sample_weights = np.ones(len(y))
        sample_weights[y > 0] = 2.0
        
        if verbose:
            print(f"Training GoalsModel (per 90) on {len(X)} samples...")
            print(f"  Mean goals/90: {y.mean():.3f}")
            print(f"  Players with goals: {(y > 0).sum()} ({(y > 0).mean():.1%})")
        
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        self.is_fitted = True
        
        if verbose:
            y_pred = self.model.predict(X_scaled)
            print(f"  MAE: {mean_absolute_error(y, y_pred):.4f}")
            print(f"  Predicted mean goals/90: {y_pred.mean():.3f}")
        
        return self
    
    def predict_per90(self, df: pd.DataFrame) -> np.ndarray:
        """Predict goals per 90 rate."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if 'goals_per90_roll5' not in df.columns:
            df = self.prepare_features(df)
        
        for feat in self.FEATURES:
            if feat not in df.columns:
                df = df.copy()
                df[feat] = 0
        
        X = df[self.FEATURES].fillna(0).astype(float)
        X_scaled = self.scaler.transform(X)
        
        return np.clip(self.model.predict(X_scaled), 0, 3)
    
    def predict_expected(self, df: pd.DataFrame, pred_minutes: np.ndarray = None) -> np.ndarray:
        """Predict expected goals given predicted minutes."""
        per90 = self.predict_per90(df)
        
        if pred_minutes is None:
            pred_minutes = df['pred_minutes'].fillna(0).values
        
        return per90 * (pred_minutes / 90)
    
    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        return pd.DataFrame({
            'feature': self.FEATURES,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
