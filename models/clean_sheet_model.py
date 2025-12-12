"""
Clean Sheet Prediction Model

Predicts probability of a team keeping a clean sheet in a given fixture.
Team-level model that can be joined to player predictions.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss


class CleanSheetModel:
    """XGBoost model to predict team clean sheet probability."""
    
    FEATURES = [
        # Team's defensive history
        'team_goals_conceded_last1',
        'team_goals_conceded_roll3',
        'team_goals_conceded_roll5',
        'team_goals_conceded_roll10',
        'team_goals_conceded_roll30',  # Long-term defensive quality
        'team_xga_roll5',
        'team_clean_sheets_roll5',
        'team_clean_sheets_roll10',
        'team_clean_sheets_roll30',  # Long-term CS rate
        
        # Home/away separated defensive stats
        'team_xga_roll5_home',
        'team_xga_roll5_away',
        
        # Opponent's attacking history
        'opp_goals_scored_last1',
        'opp_goals_scored_roll3',
        'opp_goals_scored_roll5',
        'opp_goals_scored_roll10',
        'opp_goals_scored_roll30',  # Long-term attacking quality
        'opp_xg_roll5',
        
        # Opponent's attacking history (home/away separated)
        'opp_xg_roll5_home',  # Opponent's xG when playing at home
        'opp_xg_roll5_away',  # Opponent's xG when playing away (use this for home team CS prediction)
        
        # Relative strength features
        'xga_xg_ratio',  # team_xga_roll5 / opp_xg_roll5 (lower = better for CS)
        'defensive_advantage',  # opp_xg_roll5 - team_xga_roll5 (negative = advantage)
        
        # Match context
        'is_home',
        
        # Team identity (encoded)
        'team_encoded',
        'opponent_encoded',
    ]
    
    TARGET = 'clean_sheet'
    
    def __init__(self, **xgb_params):
        """Initialize with optional XGBoost parameters."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,  # Slightly deeper for team encoding
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss',
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBClassifier(**default_params)
        self.is_fitted = False
        self.team_encoder = {}  # Map team names to IDs
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare team-level clean sheet features from player-match data.
        Returns one row per team per match.
        """
        # First, aggregate to team-match level
        # Check if xG data is available
        has_xg = 'xg' in df.columns
        
        agg_dict = {
            'goals': 'sum',  # Goals scored by this team (sum of player goals)
        }
        if has_xg:
            agg_dict['xg'] = 'sum'  # Aggregate xG if available
        
        team_match = df.groupby(['team', 'opponent', 'season', 'gameweek', 'is_home']).agg(agg_dict).reset_index()
        
        # We need goals conceded - that's the opponent's goals scored
        # Create opponent goals lookup
        opponent_goals = team_match[['team', 'season', 'gameweek', 'goals']].copy()
        opponent_goals.columns = ['opponent', 'season', 'gameweek', 'goals_conceded']
        
        team_match = team_match.merge(
            opponent_goals,
            on=['opponent', 'season', 'gameweek'],
            how='left'
        )
        
        # If xG available, also get opponent's xG (which is our xGA)
        if has_xg:
            opponent_xg = team_match[['team', 'season', 'gameweek', 'xg']].copy()
            opponent_xg.columns = ['opponent', 'season', 'gameweek', 'xga']
            team_match = team_match.merge(
                opponent_xg,
                on=['opponent', 'season', 'gameweek'],
                how='left'
            )
        else:
            team_match['xga'] = np.nan
        
        # Clean sheet = conceded 0 goals
        team_match['clean_sheet'] = (team_match['goals_conceded'] == 0).astype(int)
        
        # Sort for rolling calculations
        team_match = team_match.sort_values(['team', 'season', 'gameweek'])
        
        # ============================================================
        # TEAM ENCODING
        # ============================================================
        all_teams = sorted(team_match['team'].unique())
        self.team_encoder = {team: i for i, team in enumerate(all_teams)}
        team_match['team_encoded'] = team_match['team'].map(self.team_encoder)
        team_match['opponent_encoded'] = team_match['opponent'].map(self.team_encoder)
        
        # ============================================================
        # TEAM'S DEFENSIVE HISTORY (goals conceded)
        # For training: use shift(1) so we don't leak current match info
        # But also compute "current" features (no shift) for upcoming GW predictions
        # ============================================================
        team_match['team_goals_conceded_last1'] = (
            team_match.groupby('team')['goals_conceded'].shift(1)
        )
        team_match['team_goals_conceded_roll3'] = (
            team_match.groupby('team')['goals_conceded']
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        team_match['team_goals_conceded_roll5'] = (
            team_match.groupby('team')['goals_conceded']
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
        team_match['team_goals_conceded_roll10'] = (
            team_match.groupby('team')['goals_conceded']
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        )
        team_match['team_goals_conceded_roll30'] = (
            team_match.groupby('team')['goals_conceded']
            .transform(lambda x: x.shift(1).rolling(30, min_periods=5).mean())
        )
        
        # Also compute "current" features (including latest GW) for upcoming predictions
        team_match['team_goals_conceded_last1_current'] = team_match['goals_conceded']
        team_match['team_goals_conceded_roll3_current'] = (
            team_match.groupby('team')['goals_conceded']
            .transform(lambda x: x.rolling(3, min_periods=1).mean())
        )
        team_match['team_goals_conceded_roll5_current'] = (
            team_match.groupby('team')['goals_conceded']
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        
        # Team xGA - use actual xGA if available, otherwise fall back to goals conceded
        if 'xga' in team_match.columns and team_match['xga'].notna().any():
            team_match['team_xga_roll5'] = (
                team_match.groupby('team')['xga']
                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            )
            # Fill any NaN with goals conceded as fallback
            team_match['team_xga_roll5'] = team_match['team_xga_roll5'].fillna(team_match['team_goals_conceded_roll5'])
            
            # Home/away separated xGA
            # Calculate separately for home and away matches
            home_matches = team_match[team_match['is_home'] == True].copy()
            away_matches = team_match[team_match['is_home'] == False].copy()
            
            if len(home_matches) > 0:
                home_matches['team_xga_roll5_home'] = (
                    home_matches.groupby('team')['xga']
                    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
                )
                home_lookup = home_matches[['team', 'season', 'gameweek', 'team_xga_roll5_home']].copy()
                team_match = team_match.merge(home_lookup, on=['team', 'season', 'gameweek'], how='left', suffixes=('', '_home'))
                if 'team_xga_roll5_home_home' in team_match.columns:
                    team_match['team_xga_roll5_home'] = team_match['team_xga_roll5_home_home']
                    team_match = team_match.drop(columns=['team_xga_roll5_home_home'])
            
            if len(away_matches) > 0:
                away_matches['team_xga_roll5_away'] = (
                    away_matches.groupby('team')['xga']
                    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
                )
                away_lookup = away_matches[['team', 'season', 'gameweek', 'team_xga_roll5_away']].copy()
                team_match = team_match.merge(away_lookup, on=['team', 'season', 'gameweek'], how='left', suffixes=('', '_away'))
                if 'team_xga_roll5_away_away' in team_match.columns:
                    team_match['team_xga_roll5_away'] = team_match['team_xga_roll5_away_away']
                    team_match = team_match.drop(columns=['team_xga_roll5_away_away'])
            
            # Fill NaN with overall xGA
            if 'team_xga_roll5_home' not in team_match.columns:
                team_match['team_xga_roll5_home'] = np.nan
            if 'team_xga_roll5_away' not in team_match.columns:
                team_match['team_xga_roll5_away'] = np.nan
            team_match['team_xga_roll5_home'] = team_match['team_xga_roll5_home'].fillna(team_match['team_xga_roll5'])
            team_match['team_xga_roll5_away'] = team_match['team_xga_roll5_away'].fillna(team_match['team_xga_roll5'])
        else:
            # Fallback to goals conceded if no xG data
            team_match['team_xga_roll5'] = team_match['team_goals_conceded_roll5']
            team_match['team_xga_roll5_home'] = team_match['team_xga_roll5']
            team_match['team_xga_roll5_away'] = team_match['team_xga_roll5']
        
        # ============================================================
        # CLEAN SHEET HISTORY
        # ============================================================
        team_match['team_clean_sheets_roll5'] = (
            team_match.groupby('team')['clean_sheet']
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
        )
        team_match['team_clean_sheets_roll10'] = (
            team_match.groupby('team')['clean_sheet']
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
        )
        team_match['team_clean_sheets_roll30'] = (
            team_match.groupby('team')['clean_sheet']
            .transform(lambda x: x.shift(1).rolling(30, min_periods=5).sum())
        )
        
        # ============================================================
        # OPPONENT'S ATTACKING HISTORY
        # ============================================================
        opp_history = team_match.groupby('team').apply(
            lambda g: g.assign(
                goals_last1=g['goals'].shift(1),
                goals_roll3=g['goals'].shift(1).rolling(3, min_periods=1).mean(),
                goals_roll5=g['goals'].shift(1).rolling(5, min_periods=1).mean(),
                goals_roll10=g['goals'].shift(1).rolling(10, min_periods=1).mean(),
                goals_roll30=g['goals'].shift(1).rolling(30, min_periods=5).mean(),
                # Current features (including latest GW) for upcoming predictions
                goals_last1_current=g['goals'],
                goals_roll3_current=g['goals'].rolling(3, min_periods=1).mean(),
                goals_roll5_current=g['goals'].rolling(5, min_periods=1).mean(),
            )
        ).reset_index(drop=True)
        
        # If xG available, also calculate opponent's xG rolling averages
        if 'xg' in opp_history.columns and opp_history['xg'].notna().any():
            opp_history = opp_history.groupby('team').apply(
                lambda g: g.assign(
                    xg_roll5=g['xg'].shift(1).rolling(5, min_periods=1).mean(),
                )
            ).reset_index(drop=True)
        
        opp_lookup = opp_history[['team', 'season', 'gameweek', 
                                   'goals_last1', 'goals_roll3', 'goals_roll5', 
                                   'goals_roll10', 'goals_roll30',
                                   'goals_last1_current', 'goals_roll3_current', 'goals_roll5_current']].copy()
        opp_lookup.columns = ['opponent', 'season', 'gameweek', 
                              'opp_goals_scored_last1', 'opp_goals_scored_roll3', 
                              'opp_goals_scored_roll5', 'opp_goals_scored_roll10',
                              'opp_goals_scored_roll30',
                              'opp_goals_scored_last1_current', 'opp_goals_scored_roll3_current',
                              'opp_goals_scored_roll5_current']
        
        # Add opponent xG if available
        if 'xg_roll5' in opp_history.columns:
            opp_xg_lookup = opp_history[['team', 'season', 'gameweek', 'xg_roll5']].copy()
            opp_xg_lookup.columns = ['opponent', 'season', 'gameweek', 'opp_xg_roll5']
            opp_lookup = opp_lookup.merge(opp_xg_lookup, on=['opponent', 'season', 'gameweek'], how='left')
        
        team_match = team_match.merge(opp_lookup, on=['opponent', 'season', 'gameweek'], how='left')
        
        # Opponent xG - use actual xG if available, otherwise fall back to goals
        if 'opp_xg_roll5' not in team_match.columns or team_match['opp_xg_roll5'].isna().all():
            # Fallback to goals scored if no xG data
            team_match['opp_xg_roll5'] = team_match['opp_goals_scored_roll5']
        else:
            # Fill any NaN with goals as fallback
            team_match['opp_xg_roll5'] = team_match['opp_xg_roll5'].fillna(team_match['opp_goals_scored_roll5'])
        
        # ============================================================
        # OPPONENT'S HOME/AWAY SEPARATED xG
        # ============================================================
        # Calculate opponent's xG when playing at home vs away
        if has_xg and 'xg' in team_match.columns:
            # For each team, calculate their xG when playing at home vs away
            # We need to look at the opponent's historical xG based on where they played
            opp_home_matches = team_match[team_match['is_home'] == True].copy()
            opp_away_matches = team_match[team_match['is_home'] == False].copy()
            
            if len(opp_home_matches) > 0:
                opp_home_matches['opp_xg_roll5_home'] = (
                    opp_home_matches.groupby('team')['xg']
                    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
                )
                opp_home_lookup = opp_home_matches[['team', 'season', 'gameweek', 'opp_xg_roll5_home']].copy()
                opp_home_lookup.columns = ['opponent', 'season', 'gameweek', 'opp_xg_roll5_home']
                team_match = team_match.merge(opp_home_lookup, on=['opponent', 'season', 'gameweek'], how='left')
            
            if len(opp_away_matches) > 0:
                opp_away_matches['opp_xg_roll5_away'] = (
                    opp_away_matches.groupby('team')['xg']
                    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
                )
                opp_away_lookup = opp_away_matches[['team', 'season', 'gameweek', 'opp_xg_roll5_away']].copy()
                opp_away_lookup.columns = ['opponent', 'season', 'gameweek', 'opp_xg_roll5_away']
                team_match = team_match.merge(opp_away_lookup, on=['opponent', 'season', 'gameweek'], how='left')
            
            # Fill NaN with overall opp_xg_roll5
            if 'opp_xg_roll5_home' not in team_match.columns:
                team_match['opp_xg_roll5_home'] = np.nan
            if 'opp_xg_roll5_away' not in team_match.columns:
                team_match['opp_xg_roll5_away'] = np.nan
            team_match['opp_xg_roll5_home'] = team_match['opp_xg_roll5_home'].fillna(team_match['opp_xg_roll5'])
            team_match['opp_xg_roll5_away'] = team_match['opp_xg_roll5_away'].fillna(team_match['opp_xg_roll5'])
        else:
            # Fallback to overall opp_xg_roll5
            team_match['opp_xg_roll5_home'] = team_match['opp_xg_roll5']
            team_match['opp_xg_roll5_away'] = team_match['opp_xg_roll5']
        
        # ============================================================
        # RELATIVE STRENGTH FEATURES
        # ============================================================
        # Use context-appropriate opponent xG for relative strength
        # When predicting home team CS, use opponent's away xG (more relevant)
        # When predicting away team CS, use opponent's home xG
        team_match['opp_xg_context'] = team_match.apply(
            lambda row: row.get('opp_xg_roll5_away', row['opp_xg_roll5']) if row['is_home'] 
                       else row.get('opp_xg_roll5_home', row['opp_xg_roll5']),
            axis=1
        )
        # Fallback to overall if context-specific is missing
        team_match['opp_xg_context'] = team_match['opp_xg_context'].fillna(team_match['opp_xg_roll5'])
        
        # xGA/xG ratio (lower = better for CS, elite defenses have low ratio)
        # Use context-appropriate opponent xG
        team_match['xga_xg_ratio'] = team_match['team_xga_roll5'] / (team_match['opp_xg_context'] + 0.1)  # +0.1 to avoid division by zero
        
        # Defensive advantage (negative = advantage)
        # Use context-appropriate opponent xG
        team_match['defensive_advantage'] = team_match['opp_xg_context'] - team_match['team_xga_roll5']
        
        return team_match
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train the model on team-match data."""
        # Check if features exist
        if 'team_goals_conceded_last1' not in df.columns:
            df = self.prepare_features(df)
        
        # Store encoder from prepared data
        if 'team_encoded' in df.columns:
            # Rebuild encoder from data
            teams = df['team'].unique()
            self.team_encoder = {team: i for i, team in enumerate(sorted(teams))}
        
        # Filter to rows with history
        df = df.dropna(subset=['team_goals_conceded_last1'])
        
        X = df[self.FEATURES].fillna(0)
        y = df[self.TARGET]
        
        if verbose:
            print(f"Training CleanSheetModel on {len(X)} team-matches...")
            print(f"  Clean sheet rate: {y.mean():.1%}")
            print(f"  Teams encoded: {len(self.team_encoder)}")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        if verbose:
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            print(f"  ROC AUC: {roc_auc_score(y, y_pred_proba):.3f}")
        
        return self
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict clean sheet probability."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Apply team encoding if not already done
        df = df.copy()
        if 'team_encoded' not in df.columns or df['team_encoded'].isna().any():
            df['team_encoded'] = df['team'].map(self.team_encoder).fillna(-1)
            df['opponent_encoded'] = df['opponent'].map(self.team_encoder).fillna(-1)
        
        X = df[self.FEATURES].fillna(0)
        return self.model.predict_proba(X)[:, 1]
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        return pd.DataFrame({
            'feature': self.FEATURES,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

