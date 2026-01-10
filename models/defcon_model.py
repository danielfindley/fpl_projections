"""
Defcon Prediction Model

Predicts defensive contribution per 90 minutes (CBIT/CBIRT).
- Defenders: CBIT = Clr + Blocks + Int + Tkl (threshold 10+)
- Mids/Forwards: CBIRT = CBIT + Recov (threshold 12+)

Final expected defcon = (defcon_per90) * (pred_minutes / 90)

Uses Poisson distribution to calculate probability of exceeding threshold,
since defcon is a count variable (sum of discrete defensive actions).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import poisson
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path


class DefconModel:
    """XGBoost model to predict player defcon per 90."""
    
    FEATURES = [
        # Rolling per-90 rates (key predictors)
        'defcon_per90_roll5',
        'defcon_per90_roll10',
        'defcon_per90_last1',
        
        # Component stats per-90
        'tackles_per90_roll5',
        'interceptions_per90_roll5',
        'clearances_per90_roll5',
        'blocks_per90_roll5',
        'recoveries_per90_roll5',
        
        # LIFETIME PLAYER PROFILE (career-long defensive stats)
        'lifetime_defcon_per90',        # Career defcon per 90 mins
        'lifetime_tackles_per90',       # Career tackles per 90 mins
        'lifetime_interceptions_per90', # Career interceptions per 90 mins
        'lifetime_clearances_per90',    # Career clearances per 90 mins
        'lifetime_blocks_per90',        # Career blocks per 90 mins
        'lifetime_recoveries_per90',    # Career recoveries per 90 mins
        'lifetime_minutes',             # Total career minutes (experience indicator)
        
        # Recent form
        'defcon_last1',
        'defcon_roll3',
        
        # Threshold hit rate (historical)
        'hit_threshold_roll5',
        'hit_threshold_roll10',
        
        # Position and match context
        'is_defender',
        'is_midfielder',
        'is_home',
        
        # Opponent attacking strength (more attacks = more defensive actions)
        'opp_xg_roll5',
        'opp_shots_roll5',
    ]
    
    TARGET = 'defcon_per90'
    
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
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare per-90 features for defcon prediction."""
        df = df.copy()
        df = df.sort_values(['player_id', 'season', 'gameweek'])
        
        # Ensure player_id exists
        if 'player_id' not in df.columns:
            df['player_id'] = df['player_name'].astype(str) + '_' + df['team'].astype(str)
        
        # Get component stats
        stat_mapping = {
            'tackles': ['tackles', 'Tkl', 'tkl'],
            'interceptions': ['interceptions', 'Int', 'int'],
            'clearances': ['clearances', 'Clr', 'clr'],
            'blocks': ['blocks', 'Blocks'],
            'recoveries': ['recoveries', 'Recov', 'recov'],
        }
        
        for stat, alt_names in stat_mapping.items():
            if stat not in df.columns or df[stat].sum() == 0:
                for alt in alt_names:
                    if alt in df.columns and alt != stat:
                        val = pd.to_numeric(df[alt], errors='coerce').fillna(0)
                        if val.sum() > 0:
                            df[stat] = val
                            break
            if stat not in df.columns:
                df[stat] = 0
            else:
                df[stat] = pd.to_numeric(df[stat], errors='coerce').fillna(0)
        
        # Calculate CBIT and CBIRT
        df['CBIT'] = df['clearances'] + df['blocks'] + df['interceptions'] + df['tackles']
        df['CBIRT'] = df['CBIT'] + df['recoveries']
        
        # Position detection
        pos_col = None
        for col in ['position', 'Pos', 'pos']:
            if col in df.columns:
                pos_col = col
                break
        
        if pos_col:
            df['is_defender'] = df[pos_col].apply(
                lambda x: 1 if any(p in str(x).upper() for p in ['CB', 'LB', 'RB', 'WB', 'DF']) else 0
            )
            df['is_midfielder'] = df[pos_col].apply(
                lambda x: 1 if any(p in str(x).upper() for p in ['CM', 'DM', 'AM', 'LM', 'RM', 'MF']) else 0
            )
        else:
            df['is_defender'] = 0
            df['is_midfielder'] = 0
        
        # Defcon = CBIT for defenders, CBIRT for mids/forwards
        df['defcon'] = np.where(df['is_defender'] == 1, df['CBIT'], df['CBIRT'])
        
        # Threshold for each player
        df['threshold'] = np.where(df['is_defender'] == 1, 10, 12)
        df['hit_threshold'] = (df['defcon'] >= df['threshold']).astype(int)
        
        # Minutes handling
        df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(0) if 'minutes' in df.columns else 0
        mins_90 = np.maximum(df['minutes'] / 90, 0.01)
        
        # Per-90 rates
        df['defcon_per90'] = df['defcon'] / mins_90
        for stat in ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries']:
            df[f'{stat}_per90'] = df[stat] / mins_90
        
        # Rolling per-90 rates
        df['defcon_per90_last1'] = df.groupby('player_id')['defcon_per90'].shift(1)
        df['defcon_per90_roll5'] = df.groupby('player_id')['defcon_per90'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        df['defcon_per90_roll10'] = df.groupby('player_id')['defcon_per90'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )
        
        # Component stats rolling per-90
        for stat in ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries']:
            df[f'{stat}_per90_roll5'] = df.groupby('player_id')[f'{stat}_per90'].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )
        
        # Raw defcon trailing
        df['defcon_last1'] = df.groupby('player_id')['defcon'].shift(1)
        df['defcon_roll3'] = df.groupby('player_id')['defcon'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        
        # Threshold hit rate
        df['hit_threshold_roll5'] = df.groupby('player_id')['hit_threshold'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        df['hit_threshold_roll10'] = df.groupby('player_id')['hit_threshold'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )
        
        # is_home
        if 'is_home' in df.columns:
            df['is_home'] = pd.to_numeric(df['is_home'].replace({True: 1, False: 0, 'True': 1, 'False': 0}), errors='coerce').fillna(0)
        else:
            df['is_home'] = 0
        
        # Opponent features (if not available, default)
        if 'opp_xg_roll5' not in df.columns:
            df['opp_xg_roll5'] = df.get('opp_xg_against_roll5', 1.5)
        if 'opp_shots_roll5' not in df.columns:
            df['opp_shots_roll5'] = 12  # League average
        
        # Fill NaN
        for feat in self.FEATURES:
            if feat in df.columns:
                df[feat] = df[feat].fillna(0)
            else:
                df[feat] = 0
        
        return df
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train the model on players who played at least 1 minute."""
        if 'defcon_per90_roll5' not in df.columns:
            df = self.prepare_features(df)
        
        df = df.copy()
        
        # Filter to players who actually played
        played_mask = df['minutes'] >= 1
        df_played = df[played_mask].copy()
        
        # Ensure features exist
        for feat in self.FEATURES:
            if feat not in df_played.columns:
                df_played[feat] = 0
        
        X = df_played[self.FEATURES].fillna(0).astype(float)
        y = df_played['defcon_per90'].fillna(0)
        
        # Cap extreme values
        y = np.clip(y, 0, 30)  # Max reasonable defcon per 90
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Sample weights: upweight threshold hitters
        # Sample weights: weight by minutes played
        sample_weights = df_played['minutes'].values.copy()
        # Normalize so mean weight = 1.0
        sample_weights = sample_weights / sample_weights.mean()
        
        if verbose:
            print(f"Training DefconModel (per 90) on {len(X)} samples...")
            print(f"  Mean defcon/90: {y.mean():.2f}")
            print(f"  Players hitting threshold: {df_played['hit_threshold'].sum()} ({df_played['hit_threshold'].mean():.1%})")
        
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        self.is_fitted = True
        # Store which features were actually used (matches self.FEATURES after feature selection)
        self._features_used = self.FEATURES.copy()
        
        if verbose:
            y_pred = self.model.predict(X_scaled)
            print(f"  MAE: {mean_absolute_error(y, y_pred):.2f}")
            print(f"  Predicted mean defcon/90: {y_pred.mean():.2f}")
        
        return self
    
    def predict_per90(self, df: pd.DataFrame) -> np.ndarray:
        """Predict defcon per 90 rate."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if 'defcon_per90_roll5' not in df.columns:
            df = self.prepare_features(df)
        
        for feat in self.FEATURES:
            if feat not in df.columns:
                df = df.copy()
                df[feat] = 0
        
        X = df[self.FEATURES].fillna(0).astype(float)
        X_scaled = self.scaler.transform(X)
        
        return np.clip(self.model.predict(X_scaled), 0, 30)
    
    def predict_expected(self, df: pd.DataFrame, pred_minutes: np.ndarray = None) -> np.ndarray:
        """Predict expected defcon given predicted minutes."""
        per90 = self.predict_per90(df)
        
        if pred_minutes is None:
            pred_minutes = df['pred_minutes'].fillna(0).values
        
        return per90 * (pred_minutes / 90)
    
    def predict_proba_above_threshold(self, df: pd.DataFrame, pred_minutes: np.ndarray = None) -> np.ndarray:
        """
        Predict probability of exceeding defcon threshold using Poisson distribution.
        
        Since defcon is a count variable (tackles + interceptions + clearances + blocks + recoveries),
        we model it as Poisson(λ) where λ = expected_defcon.
        
        P(defcon >= threshold) = 1 - P(defcon < threshold) = 1 - CDF(threshold - 1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if 'defcon_per90_roll5' not in df.columns:
            df = self.prepare_features(df)
        
        # Predicted expected defcon (lambda for Poisson)
        expected_defcon = self.predict_expected(df, pred_minutes)
        
        # Ensure lambda is positive (Poisson requires λ > 0)
        expected_defcon = np.maximum(expected_defcon, 0.01)
        
        # Threshold based on position (DEF: 10, MID/FWD: 12)
        thresholds = np.where(df['is_defender'] == 1, 10, 12)
        
        # P(X >= threshold) = 1 - P(X <= threshold - 1) = 1 - CDF(threshold - 1)
        # Using scipy.stats.poisson: poisson.cdf(k, mu) = P(X <= k)
        poisson_prob = 1 - poisson.cdf(thresholds - 1, expected_defcon)
        
        return np.clip(poisson_prob, 0, 1)
    
    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        # Use _features_used if available (matches what model was trained with)
        features = getattr(self, '_features_used', self.FEATURES)
        
        return pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
