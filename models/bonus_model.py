"""
Bonus Points Prediction Model

Predicts FPL bonus points (0-3) using:
- Predicted probabilities: Pr(Goal), Pr(Assist), Pr(Defcon)
- Rolling xG, xA, defcon stats
- Opponent info
- Match context
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path


class BonusModel:
    """XGBoost model to predict FPL bonus points."""
    
    FEATURES = [
        # Predicted probabilities from other models

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
        'pred_goal_prob',
        'pred_assist_prob', 
        'pred_defcon_prob',
        'pred_minutes',
        
        # Rolling performance - scoring
        'player_roll5_xg',
        'player_roll5_goals',
        'player_roll5_xag',
        'player_roll5_assists',
        'player_roll5_shots',
        
        # Recent form
        'goals_roll3',
        'assists_roll3',
        'goal_involvements_roll5',  # Goals + assists
        
        # Rolling bonus history
        'bonus_roll3',
        'bonus_roll5',
        'bonus_roll10',
        'bonus_rate_roll10',
        
        # Defcon history
        'defcon_roll5',
        'defcon_per90_roll5',
        
        # Position indicators
        'is_forward',
        'is_midfielder', 
        'is_defender',
        'is_goalkeeper',
        
        # Match context
        'is_home',
        
        # Opponent quality
        'opp_conceded_roll5',
        'opp_xg_against_roll5',
        'opp_conceded_roll1',
        'opp_xg_against_roll1',
        'opp_shots_roll1',
        'opp_conceded_roll3',
        'opp_xg_against_roll3',
        'opp_shots_roll3',
        'opp_conceded_roll10',
        'opp_xg_against_roll10',
        'opp_shots_roll10',
        'opp_conceded_roll20',
        'opp_xg_against_roll20',
        'opp_shots_roll20',
        
        # Team context
        'team_xg_roll1',
        'team_goals_roll1',
        'team_shots_roll1',
        'team_xg_roll3',
        'team_goals_roll3',
        'team_shots_roll3',
        'team_xg_roll5',
        'team_goals_roll5',
        'team_shots_roll5',
        'team_xg_roll10',
        'team_goals_roll10',
        'team_shots_roll10',
        'team_xg_roll20',
        'team_goals_roll20',
        'team_shots_roll20',
    ]
    
    TARGET = 'bonus'
    
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
    
    def _add_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position indicator features."""
        df = df.copy()
        pos = df['position'].fillna('').str.upper()
        
        df['is_forward'] = pos.str.contains('FW|CF|ST|LW|RW').astype(int)
        df['is_midfielder'] = pos.str.contains('CM|DM|AM|LM|RM|MF').astype(int)
        df['is_defender'] = pos.str.contains('CB|LB|RB|WB|DF').astype(int)
        df['is_goalkeeper'] = pos.str.contains('GK').astype(int)
        
        return df
    
    def _add_extra_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add extra features for bonus prediction."""
        df = df.copy()
        
        # Goals and assists last 3
        if 'goals_roll3' not in df.columns:
            if 'goals' in df.columns:
                df['goals_roll3'] = df.groupby('player_id')['goals'].transform(
                    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
                ).fillna(0)
            else:
                df['goals_roll3'] = 0
        
        if 'assists_roll3' not in df.columns:
            if 'assists' in df.columns:
                df['assists_roll3'] = df.groupby('player_id')['assists'].transform(
                    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
                ).fillna(0)
            else:
                df['assists_roll3'] = 0
        
        # Goal involvements
        if 'goal_involvements_roll5' not in df.columns:
            df['goal_involvements_roll5'] = (
                df.get('player_roll5_goals', 0) + df.get('player_roll5_assists', 0)
            )
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, bonus_history: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare features for bonus prediction."""
        df = df.copy()
        df = df.sort_values(['player_id', 'season', 'gameweek'])
        
        # Add features
        df = self._add_position_features(df)
        df = self._add_extra_features(df)
        
        # Rolling bonus features (if bonus history available)
        if bonus_history is not None and len(bonus_history) > 0:
            df = df.merge(
                bonus_history[['player_id', 'season', 'gameweek', 'bonus']],
                on=['player_id', 'season', 'gameweek'],
                how='left'
            )
            df['bonus'] = df['bonus'].fillna(0)
            
            # Rolling bonus stats
            df['bonus_roll3'] = df.groupby('player_id')['bonus'].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean()
            )
            df['bonus_roll5'] = df.groupby('player_id')['bonus'].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )
            df['bonus_roll10'] = df.groupby('player_id')['bonus'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=1).mean()
            )
            df['bonus_rate_roll10'] = df['bonus_roll10']
        else:
            df['bonus'] = df.get('bonus', 0)
            df['bonus_roll3'] = 0
            df['bonus_roll5'] = 0
            df['bonus_roll10'] = 0
            df['bonus_rate_roll10'] = 0
        
        return df
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train the model."""
        df = df.copy()
        df = self._add_position_features(df)
        df = self._add_extra_features(df)
        
        # IMPORTANT: Only train on players who ACTUALLY PLAYED (60+ mins for bonus eligibility)
        played_mask = (df['minutes'] >= 60) if 'minutes' in df.columns else pd.Series(True, index=df.index)
        df = df[played_mask]
        
        # Filter to rows with bonus data
        df = df.dropna(subset=['bonus'])
        df = df[df['bonus'] >= 0]
        
        # Get available features
        available_features = [f for f in self.FEATURES if f in df.columns]
        
        # Ensure all features exist
        for feat in available_features:
            if feat not in df.columns:
                df[feat] = 0
        
        X = df[available_features].fillna(0).astype(float)
        y = df[self.TARGET]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if verbose:
            print(f"Training BonusModel on {len(X)} samples (players who played 60+ mins)...")
            print(f"  Features used: {len(available_features)}")
            print(f"  Avg bonus: {y.mean():.2f}")
            print(f"  Players with bonus > 0: {(y > 0).sum()} ({(y > 0).mean():.1%})")
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self._features_used = available_features
        
        if verbose:
            y_pred = self.model.predict(X_scaled)
            print(f"  MAE: {mean_absolute_error(y, y_pred):.3f}")
            print(f"  R²: {r2_score(y, y_pred):.3f}")
            print(f"  Predicted avg: {y_pred.mean():.2f}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict bonus points."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        df = df.copy()
        df = self._add_position_features(df)
        df = self._add_extra_features(df)
        
        for feat in self._features_used:
            if feat not in df.columns:
                df[feat] = 0
        
        X = df[self._features_used].fillna(0).astype(float)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        
        # Clip to valid bonus range [0, 3]
        return np.clip(preds, 0, 3)
    
    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        return pd.DataFrame({
            'feature': self._features_used,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
