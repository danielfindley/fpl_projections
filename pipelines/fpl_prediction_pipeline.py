"""
FPL Points Prediction Pipeline

Complete pipeline for predicting FPL points for a target gameweek.
Loads raw data, computes features, trains models, and generates predictions.

Supports two modes:
- Prediction mode: Generate predictions for upcoming gameweeks
- Tune mode: Evaluate models using train/test split or cross-validation
"""

import pandas as pd
import numpy as np
import json
import requests
import time
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFE, SelectFromModel

warnings.filterwarnings('ignore')

# Optional Optuna import
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import models
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.minutes_model import MinutesModel
from models.goals_model import GoalsModel
from models.assists_model import AssistsModel
from models.defcon_model import DefconModel
from models.clean_sheet_model import CleanSheetModel
from models.bonus_model_mc import BonusModelMC as BonusModel


# ============================================================================
# TUNE MODE CONFIGURATION AND METRICS
# ============================================================================

@dataclass
class TuneConfig:
    """Configuration for model tuning and evaluation.
    
    Attributes:
        models: List of model names to tune. Options: 'minutes', 'goals', 'assists', 'defcon', 'clean_sheet', 'bonus'
        test_size: Fraction of data to use for testing (default 0.2)
        cv_folds: Number of cross-validation folds (if None, uses train/test split)
        random_state: Random seed for reproducibility
        hyperparams: Dict of model_name -> dict of hyperparameters to override
        feature_selection: Feature selection method ('none', 'importance', 'rfe')
        feature_selection_k: Number of top features to keep (for 'importance' and 'rfe')
        optuna_trials: Number of Optuna trials for hyperparameter optimization (0 to disable)
        optuna_timeout: Timeout in seconds for Optuna optimization (None for no timeout)
    """
    models: List[str] = field(default_factory=lambda: ['goals', 'assists', 'minutes', 'defcon'])
    test_size: float = 0.2
    cv_folds: Optional[int] = None  # If set, uses cross-validation instead of train/test split
    random_state: int = 42
    hyperparams: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Feature selection options
    feature_selection: str = 'none'  # 'none', 'importance', 'rfe'
    feature_selection_k: int = 20  # Number of features to keep
    
    # Optuna hyperparameter optimization
    optuna_trials: int = 0  # Number of trials (0 = disabled)
    optuna_timeout: Optional[int] = None  # Timeout in seconds
    
    # Default hyperparameters for each model type
    @staticmethod
    def default_hyperparams() -> Dict[str, Dict[str, Any]]:
        return {
            'goals': {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.08,
                'min_child_weight': 3,
            },
            'assists': {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.08,
                'min_child_weight': 3,
            },
            'minutes': {
                'n_estimators': 200,
                'max_depth': 4,
                'learning_rate': 0.1,
                'min_child_weight': 5,
            },
            'defcon': {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.08,
                'min_child_weight': 3,
            },
            'bonus': {
                'n_estimators': 150,
                'max_depth': 5,
                'learning_rate': 0.1,
                'min_child_weight': 3,
            },
        }
    
    # Optuna search spaces for each model
    @staticmethod
    def optuna_search_space() -> Dict[str, Dict[str, Any]]:
        """Define the search space for Optuna optimization."""
        return {
            'goals': {
                'n_estimators': ('int', 100, 500),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('float_log', 0.01, 0.3),
                'min_child_weight': ('int', 1, 10),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 1.0),
                'reg_alpha': ('float_log', 1e-8, 10.0),
                'reg_lambda': ('float_log', 1e-8, 10.0),
            },
            'assists': {
                'n_estimators': ('int', 100, 500),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('float_log', 0.01, 0.3),
                'min_child_weight': ('int', 1, 10),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 1.0),
            },
            'minutes': {
                'n_estimators': ('int', 100, 500),
                'max_depth': ('int', 3, 8),
                'learning_rate': ('float_log', 0.01, 0.3),
                'min_child_weight': ('int', 1, 15),
                'subsample': ('float', 0.6, 1.0),
            },
            'defcon': {
                'n_estimators': ('int', 100, 500),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('float_log', 0.01, 0.3),
                'min_child_weight': ('int', 1, 10),
            },
            'bonus': {
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 3, 8),
                'learning_rate': ('float_log', 0.01, 0.3),
                'min_child_weight': ('int', 1, 10),
            },
        }
    
    def get_hyperparams(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameters for a model, merging defaults with overrides."""
        defaults = self.default_hyperparams()
        params = defaults.get(model_name, {}).copy()
        params.update(self.hyperparams.get(model_name, {}))
        params['random_state'] = self.random_state
        return params


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    model_name: str
    mae: float
    rmse: float
    r2: float
    samples: int
    cv_scores: Optional[List[float]] = None  # If using cross-validation
    feature_importance: Optional[pd.DataFrame] = None
    selected_features: Optional[List[str]] = None  # Features after selection
    best_params: Optional[Dict[str, Any]] = None  # Best params from Optuna
    optuna_study: Optional[Any] = None  # Optuna study object
    
    def __str__(self) -> str:
        result = f"\n{'='*60}\n"
        result += f"  {self.model_name.upper()} MODEL METRICS\n"
        result += f"{'='*60}\n"
        result += f"  Samples: {self.samples:,}\n"
        result += f"  MAE:     {self.mae:.4f}\n"
        result += f"  RMSE:    {self.rmse:.4f}\n"
        result += f"  R²:      {self.r2:.4f}\n"
        if self.cv_scores:
            result += f"  CV Scores (MAE): {[f'{s:.4f}' for s in self.cv_scores]}\n"
            result += f"  CV Mean ± Std:   {np.mean(self.cv_scores):.4f} ± {np.std(self.cv_scores):.4f}\n"
        if self.selected_features:
            result += f"  Selected Features: {len(self.selected_features)}\n"
        if self.best_params:
            result += f"  Best Params (Optuna):\n"
            for k, v in self.best_params.items():
                result += f"    {k}: {v}\n"
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'mae': self.mae,
            'rmse': self.rmse,
            'r2': self.r2,
            'samples': self.samples,
            'cv_scores': self.cv_scores,
            'selected_features': self.selected_features,
            'best_params': self.best_params,
        }


# FPL Scoring Rules
FPL_POINTS = {
    'goal': {'GK': 6, 'DEF': 6, 'MID': 5, 'FWD': 4},
    'assist': {'GK': 3, 'DEF': 3, 'MID': 3, 'FWD': 3},
    'clean_sheet': {'GK': 4, 'DEF': 4, 'MID': 1, 'FWD': 0},
    'defcon': 2,
    'appearance_60plus': 2,
    'appearance_under60': 1,
}


def get_fpl_position(pos_str: str) -> str:
    """Convert FBref position to FPL position (fallback only)."""
    if pd.isna(pos_str):
        return 'MID'
    pos = str(pos_str).upper()
    if 'GK' in pos:
        return 'GK'
    elif any(p in pos for p in ['CB', 'LB', 'RB', 'WB', 'DF']):
        return 'DEF'
    elif any(p in pos for p in ['FW', 'CF', 'ST', 'LW', 'RW']):
        return 'FWD'
    return 'MID'


def fetch_fpl_positions(data_dir: Path, player_names: set, verbose: bool = True) -> dict:
    """Fetch FPL positions for players using name matching."""
    if verbose:
        print('[FPL Positions] Fetching positions from FPL API...')
    
    # Get FPL bootstrap data
    bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    
    # Map element_type to position string
    POS_MAP = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    # Build FPL player lookup by name variants
    fpl_players = {}
    for p in bootstrap['elements']:
        pos = POS_MAP.get(p['element_type'], 'MID')
        fpl_id = p['id']
        # Store multiple name variants
        names = [
            p['web_name'],  # e.g., "Salah"
            f"{p['first_name']} {p['second_name']}",  # e.g., "Mohamed Salah"
            p['second_name'],  # e.g., "Salah"
        ]
        for name in names:
            fpl_players[name.lower()] = {'fpl_id': fpl_id, 'fpl_position': pos, 'web_name': p['web_name']}
    
    # Load name matches if available
    name_matches_path = data_dir / 'name_matching' / 'name_matches.json'
    name_matches = {}
    if name_matches_path.exists():
        with open(name_matches_path, encoding='utf-8') as f:
            name_matches = json.load(f)
    
    # Map player names to FPL positions
    player_positions = {}
    matched = 0
    
    for player_name in player_names:
        # Try name_matches first
        if player_name in name_matches:
            match_data = name_matches[player_name]
            if isinstance(match_data, dict) and 'fpl_id' in match_data:
                fpl_id = match_data['fpl_id']
                # Find position by fpl_id
                for p in bootstrap['elements']:
                    if p['id'] == fpl_id:
                        player_positions[player_name] = POS_MAP.get(p['element_type'], 'MID')
                        matched += 1
                        break
                continue
        
        # Try direct name matching
        name_lower = player_name.lower()
        if name_lower in fpl_players:
            player_positions[player_name] = fpl_players[name_lower]['fpl_position']
            matched += 1
            continue
        
        # Try partial matching (last name)
        parts = player_name.split()
        if len(parts) > 1:
            last_name = parts[-1].lower()
            if last_name in fpl_players:
                player_positions[player_name] = fpl_players[last_name]['fpl_position']
                matched += 1
                continue
    
    if verbose:
        print(f'  Matched {matched}/{len(player_names)} players to FPL positions')
    
    # Manual position overrides (for players that are misclassified or missing)
    MANUAL_POSITION_OVERRIDES = {
        'Raúl Jiménez': 'FWD',
        'Matheus Cunha': 'MID',
        'Mateus Gonçalo Espanha Fernandes': 'MID',
    }
    
    # Apply manual overrides
    for player_name, position in MANUAL_POSITION_OVERRIDES.items():
        if player_name in player_names:
            old_position = player_positions.get(player_name, 'Unknown')
            player_positions[player_name] = position
            if verbose and old_position != position:
                print(f'  Manual override: {player_name} ({old_position} → {position})')
    
    return player_positions


def load_raw_data(data_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Load all player-match data from raw CSV files."""
    if verbose:
        print('[1] Loading data from raw CSV files...')
    
    all_records = []
    loaded_count = 0
    skipped_count = 0
    
    for season_dir in sorted(data_dir.iterdir()):
        if not season_dir.is_dir() or season_dir.name in ['features', 'name_matching', 'raw_html', 'intermediate', 'predictions', 'raw']:
            continue
        season = season_dir.name
        
        for gw_dir in sorted(season_dir.iterdir()):
            if not gw_dir.is_dir() or not gw_dir.name.startswith('gw'):
                continue
            try:
                gameweek = int(gw_dir.name.replace('gw', ''))
            except:
                continue
            
            for match_dir in gw_dir.iterdir():
                if not match_dir.is_dir():
                    continue
                
                match_name = match_dir.name
                teams_in_match = match_name.split('_vs_')
                if len(teams_in_match) != 2:
                    continue
                home_team_approx = teams_in_match[0]
                away_team_approx = teams_in_match[1]
                
                for team_dir in match_dir.iterdir():
                    if not team_dir.is_dir():
                        continue
                    
                    team = team_dir.name
                    is_home = home_team_approx in team or team in home_team_approx
                    opponent = away_team_approx if is_home else home_team_approx
                    
                    summary_path = team_dir / 'summary.csv'
                    defense_path = team_dir / 'defense.csv'
                    misc_path = team_dir / 'misc.csv'
                    passing_path = team_dir / 'passing.csv'
                    
                    if not summary_path.exists():
                        skipped_count += 1
                        continue
                    
                    try:
                        summary = pd.read_csv(summary_path)
                        
                        if 'Player' not in summary.columns or 'Min' not in summary.columns:
                            skipped_count += 1
                            continue
                        
                        players = summary[['Player']].copy()
                        players['minutes'] = pd.to_numeric(summary['Min'], errors='coerce').fillna(0)
                        players['position'] = summary['Pos'] if 'Pos' in summary.columns else 'MID'
                        
                        col_map = {'Gls': 'goals', 'Ast': 'assists', 'xG': 'xg', 'npxG': 'npxg',
                                   'xAG': 'xag', 'Sh': 'shots', 'SoT': 'shots_on_target',
                                   'SCA': 'sca', 'GCA': 'gca'}
                        for orig_col, new_col in col_map.items():
                            if orig_col in summary.columns:
                                players[new_col] = pd.to_numeric(summary[orig_col], errors='coerce').fillna(0)
                            else:
                                players[new_col] = 0
                        
                        if defense_path.exists():
                            defense = pd.read_csv(defense_path)
                            if 'Player' in defense.columns and 'Tkl' in defense.columns:
                                defense_cols = defense[['Player']].copy()
                                defense_map = {'Tkl': 'tackles', 'Int': 'interceptions',
                                              'Clr': 'clearances', 'Blocks': 'blocks'}
                                for orig_col, new_col in defense_map.items():
                                    if orig_col in defense.columns:
                                        defense_cols[new_col] = pd.to_numeric(defense[orig_col], errors='coerce').fillna(0)
                                players = players.merge(defense_cols, on='Player', how='left')
                        
                        if misc_path.exists():
                            misc = pd.read_csv(misc_path)
                            if 'Player' in misc.columns and 'Recov' in misc.columns:
                                misc_cols = misc[['Player']].copy()
                                misc_cols['recoveries'] = pd.to_numeric(misc['Recov'], errors='coerce').fillna(0)
                                players = players.merge(misc_cols, on='Player', how='left')
                        
                        if passing_path.exists():
                            passing = pd.read_csv(passing_path)
                            if 'Player' in passing.columns:
                                passing_cols = passing[['Player']].copy()
                                if 'xA' in passing.columns:
                                    passing_cols['xa'] = pd.to_numeric(passing['xA'], errors='coerce').fillna(0)
                                if 'KP' in passing.columns:
                                    passing_cols['key_passes'] = pd.to_numeric(passing['KP'], errors='coerce').fillna(0)
                                if 'PPA' in passing.columns:
                                    passing_cols['ppa'] = pd.to_numeric(passing['PPA'], errors='coerce').fillna(0)
                                players = players.merge(passing_cols, on='Player', how='left')
                        
                        for col in ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries', 'xa', 'key_passes', 'ppa']:
                            if col in players.columns:
                                players[col] = players[col].fillna(0)
                            else:
                                players[col] = 0
                        
                        players['CBIT'] = players['clearances'] + players['blocks'] + players['interceptions'] + players['tackles']
                        players['CBIRT'] = players['CBIT'] + players['recoveries']
                        
                        players['season'] = season
                        players['gameweek'] = gameweek
                        players['team'] = team
                        players['opponent'] = opponent
                        players['is_home'] = is_home
                        players['player_name'] = players['Player']
                        
                        players = players[players['player_name'].notna()]
                        players = players[~players['player_name'].astype(str).str.contains(r'\d+\s*Players?', case=False, na=False, regex=True)]
                        players = players[players['player_name'] != '']
                        
                        if len(players) > 0:
                            all_records.append(players)
                            loaded_count += 1
                            
                    except Exception:
                        skipped_count += 1
                        continue
    
    df = pd.concat(all_records, ignore_index=True)
    df['player_id'] = df['player_name'].astype(str) + '_' + df['team'].astype(str)
    
    if verbose:
        print(f'  Loaded {len(df)} player-match records from {loaded_count} team-matches')
        print(f'  Seasons: {sorted(df["season"].unique())}')
    
    return df


def compute_rolling_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Compute all rolling features for the models."""
    if verbose:
        print('[2] Computing rolling features...')
    
    df = df.sort_values(['player_id', 'season', 'gameweek']).reset_index(drop=True)
    
    # Player rolling features
    for col in ['goals', 'assists', 'xg', 'xag', 'shots', 'sca', 'gca', 'minutes']:
        if col in df.columns:
            df[f'player_roll5_{col}'] = df.groupby('player_id')[col].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )
    
    # Minutes model features
    df['last_game_minutes'] = df.groupby('player_id')['minutes'].shift(1)
    df['roll3_minutes_avg'] = df.groupby('player_id')['minutes'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['roll5_minutes_avg'] = df.groupby('player_id')['minutes'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['roll5_minutes_sum'] = df.groupby('player_id')['minutes'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    df['roll10_minutes_avg'] = df.groupby('player_id')['minutes'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    
    df['was_start'] = (df['minutes'] >= 60).astype(int)
    df['was_full_90'] = (df['minutes'] >= 89).astype(int)
    df['last_was_start'] = df.groupby('player_id')['was_start'].shift(1).fillna(0)
    df['last_was_full_90'] = df.groupby('player_id')['was_full_90'].shift(1).fillna(0)
    df['roll5_starts'] = df.groupby('player_id')['was_start'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()).fillna(2.5)
    df['roll5_full_90s'] = df.groupby('player_id')['was_full_90'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()).fillna(2.5)
    df['full_90_rate'] = df.groupby('player_id')['was_full_90'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(0.5)
    
    df['goals_roll5'] = df.groupby('player_id')['goals'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()).fillna(0)
    df['assists_roll5'] = df.groupby('player_id')['assists'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()).fillna(0)
    df['goal_involvements_roll5'] = df['goals_roll5'] + df['assists_roll5']
    
    df['starter_score'] = (
        (df['roll5_minutes_avg'].fillna(67) / 90) * 0.4 +
        (df['roll5_full_90s'].fillna(2.5) / 5) * 0.4 +
        (df['last_was_full_90'].fillna(0.5)) * 0.2
    )
    
    df = df.drop(columns=['was_start', 'was_full_90'], errors='ignore')
    
    # Per-90 rates
    mins_90 = np.maximum(df['minutes'] / 90, 0.01)
    for col in ['goals', 'assists', 'xg', 'xag', 'shots', 'sca', 'gca', 'key_passes']:
        if col in df.columns:
            df[f'{col}_per90'] = df[col] / mins_90
            df[f'{col}_per90_roll1'] = df.groupby('player_id')[f'{col}_per90'].transform(lambda x: x.shift(1).rolling(1, min_periods=1).mean())
            df[f'{col}_per90_roll3'] = df.groupby('player_id')[f'{col}_per90'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            df[f'{col}_per90_roll5'] = df.groupby('player_id')[f'{col}_per90'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            df[f'{col}_per90_roll10'] = df.groupby('player_id')[f'{col}_per90'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        else:
            df[f'{col}_per90'] = 0
            df[f'{col}_per90_roll1'] = 0
            df[f'{col}_per90_roll3'] = 0
            df[f'{col}_per90_roll5'] = 0
            df[f'{col}_per90_roll10'] = 0
    
    # Recent form
    df['goals_last1'] = df.groupby('player_id')['goals'].shift(1).fillna(0)
    df['goals_roll3'] = df.groupby('player_id')['goals'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum()).fillna(0)
    df['xg_roll3'] = df.groupby('player_id')['xg'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum()).fillna(0)
    df['assists_last1'] = df.groupby('player_id')['assists'].shift(1).fillna(0)
    df['assists_roll3'] = df.groupby('player_id')['assists'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum()).fillna(0)
    df['xag_roll3'] = df.groupby('player_id')['xag'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum()).fillna(0)
    df['goals_roll10'] = df.groupby('player_id')['goals'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum()).fillna(0)
    df['assists_roll10'] = df.groupby('player_id')['assists'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum()).fillna(0)

    # Defcon features
    for col in ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries', 'CBIT', 'CBIRT']:
        if col in df.columns:
            df[f'{col}_roll5'] = df.groupby('player_id')[col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            df[f'{col}_last1'] = df.groupby('player_id')[col].shift(1)
    
    df['is_defender'] = df['position'].apply(lambda x: 1 if any(p in str(x).upper() for p in ['CB', 'LB', 'RB', 'WB', 'DF', 'GK']) else 0)
    df['is_midfielder'] = df['position'].apply(lambda x: 1 if any(p in str(x).upper() for p in ['CM', 'DM', 'AM', 'LM', 'RM', 'MF']) else 0)
    df['is_forward'] = df['position'].apply(lambda x: 1 if any(p in str(x).upper() for p in ['FW', 'CF', 'ST', 'LW', 'RW']) else 0)
    df['is_goalkeeper'] = df['position'].apply(lambda x: 1 if 'GK' in str(x).upper() else 0)
    
    df['defcon'] = df.apply(lambda r: r['CBIT'] if r['is_defender'] == 1 else r['CBIRT'], axis=1)
    df['defcon_roll5'] = df.groupby('player_id')['defcon'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['defcon_last1'] = df.groupby('player_id')['defcon'].shift(1)
    df['defcon_roll3'] = df.groupby('player_id')['defcon'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    
    df['defcon_per90'] = df['defcon'] / np.maximum(df['minutes'] / 90, 0.01)
    df['defcon_per90_roll5'] = df.groupby('player_id')['defcon_per90'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['defcon_per90_roll10'] = df.groupby('player_id')['defcon_per90'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    df['defcon_per90_last1'] = df.groupby('player_id')['defcon_per90'].shift(1)
    
    df['threshold'] = np.where(df['is_defender'] == 1, 10, 12)
    df['hit_threshold'] = (df['defcon'] >= df['threshold']).astype(int)
    df['hit_threshold_roll5'] = df.groupby('player_id')['hit_threshold'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['hit_threshold_roll10'] = df.groupby('player_id')['hit_threshold'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    
    for col in ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries']:
        if col in df.columns:
            df[f'{col}_per90'] = df[col] / np.maximum(df['minutes'] / 90, 0.01)
            df[f'{col}_per90_roll5'] = df.groupby('player_id')[f'{col}_per90'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    
    # Team rolling features
    team_stats = df.groupby(['team', 'season', 'gameweek']).agg({'goals': 'sum', 'xg': 'sum', 'shots': 'sum'}).reset_index()
    team_stats = team_stats.rename(columns={'goals': 'team_goals', 'xg': 'team_xg', 'shots': 'team_shots'})
    team_stats = team_stats.sort_values(['team', 'season', 'gameweek'])
    
    for col in ['team_goals', 'team_xg', 'team_shots']:
        team_stats[f'{col}_roll1'] = team_stats.groupby('team')[col].transform(lambda x: x.shift(1).rolling(1, min_periods=1).mean())
        team_stats[f'{col}_roll3'] = team_stats.groupby('team')[col].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        team_stats[f'{col}_roll5'] = team_stats.groupby('team')[col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        team_stats[f'{col}_roll10'] = team_stats.groupby('team')[col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        team_stats[f'{col}_roll20'] = team_stats.groupby('team')[col].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    
    df = df.merge(team_stats[['team', 'season', 'gameweek', 'team_goals_roll5', 'team_xg_roll5', 'team_shots_roll5','team_goals_roll10', 'team_xg_roll10', 'team_shots_roll10','team_goals_roll20', 'team_xg_roll20', 'team_shots_roll20']],
                  on=['team', 'season', 'gameweek'], how='left')
    
    # Opponent rolling features
    opp_stats = team_stats[['team', 'season', 'gameweek', 'team_goals', 'team_xg', 'team_shots']].copy()
    opp_stats = opp_stats.rename(columns={'team': 'opponent', 'team_goals': 'opp_goals', 'team_xg': 'opp_xg', 'team_shots': 'opp_shots'})
    opp_stats = opp_stats.sort_values(['opponent', 'season', 'gameweek'])
    

    opp_stats['opp_conceded_roll1'] = opp_stats.groupby('opponent')['opp_goals'].transform(lambda x: x.shift(1).rolling(1, min_periods=1).mean())
    opp_stats['opp_xg_against_roll1'] = opp_stats.groupby('opponent')['opp_xg'].transform(lambda x: x.shift(1).rolling(1, min_periods=1).mean())
    opp_stats['opp_shots_roll1'] = opp_stats.groupby('opponent')['opp_shots'].transform(lambda x: x.shift(1).rolling(1, min_periods=1).mean())
    opp_stats['opp_conceded_roll3'] = opp_stats.groupby('opponent')['opp_goals'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    opp_stats['opp_xg_against_roll3'] = opp_stats.groupby('opponent')['opp_xg'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    opp_stats['opp_shots_roll3'] = opp_stats.groupby('opponent')['opp_shots'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    opp_stats['opp_conceded_roll5'] = opp_stats.groupby('opponent')['opp_goals'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    opp_stats['opp_xg_against_roll5'] = opp_stats.groupby('opponent')['opp_xg'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    opp_stats['opp_xg_roll5'] = opp_stats.groupby('opponent')['opp_xg'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    opp_stats['opp_shots_roll5'] = opp_stats.groupby('opponent')['opp_shots'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    opp_stats['opp_conceded_roll10'] = opp_stats.groupby('opponent')['opp_goals'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    opp_stats['opp_xg_against_roll10'] = opp_stats.groupby('opponent')['opp_xg'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    opp_stats['opp_shots_roll10'] = opp_stats.groupby('opponent')['opp_shots'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    opp_stats['opp_xg_roll10'] = opp_stats.groupby('opponent')['opp_xg'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    opp_stats['opp_conceded_roll20'] = opp_stats.groupby('opponent')['opp_goals'].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    opp_stats['opp_xg_against_roll20'] = opp_stats.groupby('opponent')['opp_xg'].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    opp_stats['opp_shots_roll20'] = opp_stats.groupby('opponent')['opp_shots'].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    opp_stats['opp_xg_roll20'] = opp_stats.groupby('opponent')['opp_xg'].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())


    
    df = df.merge(opp_stats[['opponent', 'season', 'gameweek', 'opp_conceded_roll5', 'opp_xg_against_roll5', 'opp_xg_roll5', 'opp_shots_roll5','opp_conceded_roll10', 'opp_xg_against_roll10', 'opp_xg_roll10', 'opp_shots_roll10','opp_conceded_roll20', 'opp_xg_against_roll20', 'opp_xg_roll20', 'opp_shots_roll20']],
                  on=['opponent', 'season', 'gameweek'], how='left')
    
    df['is_home'] = df['is_home'].astype(int)
    
    # Fill NaN
    rolling_cols = [c for c in df.columns if 'roll' in c or 'last1' in c or 'per90' in c]
    for col in rolling_cols:
        df[col] = df[col].fillna(0)
    
    if verbose:
        print(f'  Computed {len(rolling_cols)} rolling features')
    
    return df


def fetch_fpl_bonus_data(data_dir: Path, player_names: set, verbose: bool = True) -> pd.DataFrame:
    """Fetch bonus data from FPL API."""
    if verbose:
        print('[Bonus] Fetching FPL bonus data...')
    
    name_matches_path = data_dir / 'name_matching' / 'name_matches.json'
    with open(name_matches_path, encoding='utf-8') as f:
        name_matches = json.load(f)
    
    fbref_to_fpl_id = {}
    for fbref_name, match_data in name_matches.items():
        if fbref_name in player_names and isinstance(match_data, dict) and 'fpl_id' in match_data:
            fbref_to_fpl_id[fbref_name] = match_data['fpl_id']
    
    if verbose:
        print(f'  Matched {len(fbref_to_fpl_id)} players to FPL IDs')
    
    all_history = []
    fetched = 0
    
    for fbref_name, fpl_id in fbref_to_fpl_id.items():
        try:
            url = f"https://fantasy.premierleague.com/api/element-summary/{fpl_id}/"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'history' in data:
                for gw in data['history']:
                    gw['fbref_name'] = fbref_name
                    gw['fpl_id'] = fpl_id
                    all_history.append(gw)
            
            fetched += 1
            if fetched % 100 == 0 and verbose:
                print(f'    Fetched {fetched}/{len(fbref_to_fpl_id)}...')
            
            time.sleep(0.05)
        except:
            continue
    
    if verbose:
        print(f'  Fetched {len(all_history)} FPL gameweek records')
    
    if len(all_history) > 0:
        fpl_history = pd.DataFrame(all_history)
        fpl_history = fpl_history.rename(columns={'round': 'gameweek'})
        # Include bps (raw BPS score) for Monte Carlo bonus model
        cols = ['fbref_name', 'gameweek', 'bonus']
        if 'bps' in fpl_history.columns:
            cols.append('bps')
        return fpl_history[cols].copy()
    
    return pd.DataFrame()


def calculate_expected_points(row: pd.Series) -> pd.Series:
    """Calculate expected FPL points for a player."""
    pos = row['fpl_position']
    
    exp_goals = row['pred_exp_goals'] * FPL_POINTS['goal'].get(pos, 5)
    exp_assists = row['pred_exp_assists'] * FPL_POINTS['assist'].get(pos, 3)
    exp_cs = row['pred_cs_prob'] * FPL_POINTS['clean_sheet'].get(pos, 0)
    exp_defcon = row.get('pred_defcon_prob', 0) * FPL_POINTS['defcon']
    exp_bonus = row.get('pred_bonus', 0)
    
    pred_mins = row['pred_minutes']
    if pred_mins >= 60:
        exp_appearance = FPL_POINTS['appearance_60plus']
    elif pred_mins >= 1:
        exp_appearance = FPL_POINTS['appearance_under60']
    else:
        exp_appearance = 0
    
    total = exp_goals + exp_assists + exp_cs + exp_defcon + exp_bonus + exp_appearance
    
    return pd.Series({
        'exp_goals_pts': exp_goals,
        'exp_assists_pts': exp_assists,
        'exp_cs_pts': exp_cs,
        'exp_defcon_pts': exp_defcon,
        'exp_bonus_pts': exp_bonus,
        'exp_appearance_pts': exp_appearance,
        'exp_total_pts': total
    })


class FPLPredictionPipeline:
    """Complete FPL prediction pipeline.
    
    Supports two modes:
    - run(): Generate predictions for a target gameweek
    - tune(): Evaluate models using train/test split or cross-validation
    """
    
    def __init__(self, data_dir: str = '../data'):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.df = None
        self.train_df = None
        self.test_df = None
        self.tune_results: Dict[str, ModelMetrics] = {}
    
    def tune(self, config: TuneConfig = None, verbose: bool = True) -> Dict[str, ModelMetrics]:
        """
        Run model tuning and evaluation using train/test split or cross-validation.
        
        Args:
            config: TuneConfig object with tuning settings. If None, uses defaults.
            verbose: Print progress and results
            
        Returns:
            Dictionary of model_name -> ModelMetrics
        """
        if config is None:
            config = TuneConfig()
        
        if verbose:
            print('=' * 70)
            print('FPL MODEL TUNING MODE')
            print('=' * 70)
            print(f"  Models to tune: {config.models}")
            print(f"  Test size: {config.test_size}" if not config.cv_folds else f"  CV folds: {config.cv_folds}")
            print(f"  Random state: {config.random_state}")
            print('=' * 70)
        
        # Step 1: Load data
        self.df = load_raw_data(self.data_dir, verbose)
        
        # Step 2: Compute features
        self.df = compute_rolling_features(self.df, verbose)
        
        # Filter to players who played
        played_df = self.df[self.df['minutes'] >= 1].copy()
        
        if verbose:
            print(f'\n[3] Splitting data for evaluation...')
            print(f'    Total samples (played 1+ min): {len(played_df):,}')
        
        # Run tuning for each model
        results = {}
        
        for model_name in config.models:
            if verbose:
                print(f'\n{"="*70}')
                print(f'TUNING: {model_name.upper()} MODEL')
                print(f'{"="*70}')
            
            if model_name == 'goals':
                metrics = self._tune_goals_model(played_df, config, verbose)
            elif model_name == 'assists':
                metrics = self._tune_assists_model(played_df, config, verbose)
            elif model_name == 'minutes':
                metrics = self._tune_minutes_model(played_df, config, verbose)
            elif model_name == 'defcon':
                metrics = self._tune_defcon_model(played_df, config, verbose)
            elif model_name == 'clean_sheet':
                metrics = self._tune_clean_sheet_model(config, verbose)
            elif model_name == 'bonus':
                metrics = self._tune_bonus_model(played_df, config, verbose)
            else:
                print(f'  WARNING: Unknown model "{model_name}", skipping...')
                continue
            
            results[model_name] = metrics
            if verbose:
                print(metrics)
        
        self.tune_results = results
        
        if verbose:
            print('\n' + '=' * 70)
            print('TUNING COMPLETE - SUMMARY')
            print('=' * 70)
            for name, metrics in results.items():
                print(f"  {name.upper():12} | MAE: {metrics.mae:.4f} | RMSE: {metrics.rmse:.4f} | R²: {metrics.r2:.4f}")
            print('=' * 70)
        
        return results
    
    def _tune_goals_model(self, df: pd.DataFrame, config: TuneConfig, verbose: bool) -> ModelMetrics:
        """Tune and evaluate the goals model."""
        params = config.get_hyperparams('goals')
        
        # Optuna optimization if enabled
        study = None
        if config.optuna_trials > 0:
            params, study = self._optuna_optimize('goals', GoalsModel, df, config, verbose)
        
        if config.cv_folds:
            return self._cv_evaluate_model('goals', GoalsModel, df, config, params, study, verbose)
        else:
            return self._split_evaluate_model('goals', GoalsModel, df, config, params, study, verbose)
    
    def _tune_assists_model(self, df: pd.DataFrame, config: TuneConfig, verbose: bool) -> ModelMetrics:
        """Tune and evaluate the assists model."""
        params = config.get_hyperparams('assists')
        
        # Optuna optimization if enabled
        study = None
        if config.optuna_trials > 0:
            params, study = self._optuna_optimize('assists', AssistsModel, df, config, verbose)
        
        if config.cv_folds:
            return self._cv_evaluate_model('assists', AssistsModel, df, config, params, study, verbose)
        else:
            return self._split_evaluate_model('assists', AssistsModel, df, config, params, study, verbose)
    
    def _tune_minutes_model(self, df: pd.DataFrame, config: TuneConfig, verbose: bool) -> ModelMetrics:
        """Tune and evaluate the minutes model."""
        params = config.get_hyperparams('minutes')
        
        # Optuna optimization if enabled
        study = None
        if config.optuna_trials > 0:
            params, study = self._optuna_optimize('minutes', MinutesModel, df, config, verbose)
        
        if config.cv_folds:
            return self._cv_evaluate_model('minutes', MinutesModel, df, config, params, study, verbose)
        else:
            return self._split_evaluate_model('minutes', MinutesModel, df, config, params, study, verbose)
    
    def _tune_defcon_model(self, df: pd.DataFrame, config: TuneConfig, verbose: bool) -> ModelMetrics:
        """Tune and evaluate the defcon model."""
        params = config.get_hyperparams('defcon')
        
        # Optuna optimization if enabled
        study = None
        if config.optuna_trials > 0:
            params, study = self._optuna_optimize('defcon', DefconModel, df, config, verbose)
        
        if config.cv_folds:
            return self._cv_evaluate_model('defcon', DefconModel, df, config, params, study, verbose)
        else:
            return self._split_evaluate_model('defcon', DefconModel, df, config, params, study, verbose)
    
    def _tune_clean_sheet_model(self, config: TuneConfig, verbose: bool) -> ModelMetrics:
        """Tune and evaluate the clean sheet model."""
        # CS model uses team-level features, different from player models
        cs_model = CleanSheetModel()
        team_features = cs_model.prepare_features(self.df)
        
        # Train/test split on team-match data
        train_teams, test_teams = train_test_split(
            team_features, test_size=config.test_size, random_state=config.random_state
        )
        
        cs_model.fit(train_teams, verbose=False)
        
        # Evaluate
        y_test = test_teams['clean_sheet'].values
        y_pred = cs_model.predict_proba(test_teams)
        
        # For classification, use different metrics
        from sklearn.metrics import roc_auc_score, brier_score_loss
        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        
        if verbose:
            print(f"  Train samples: {len(train_teams):,}")
            print(f"  Test samples: {len(test_teams):,}")
            print(f"  AUC-ROC: {auc:.4f}")
            print(f"  Brier Score: {brier:.4f}")
        
        self.models['clean_sheet'] = cs_model
        
        return ModelMetrics(
            model_name='clean_sheet',
            mae=brier,  # Use Brier score as MAE analog for probability
            rmse=np.sqrt(brier),
            r2=auc,  # Use AUC as R² analog
            samples=len(test_teams),
            feature_importance=cs_model.feature_importance() if hasattr(cs_model, 'feature_importance') else None
        )
    
    def _split_evaluate_model(self, name: str, model_class, df: pd.DataFrame, 
                           config: TuneConfig, params: Dict, study, verbose: bool) -> ModelMetrics:
        """Evaluate a model using temporal train/test split."""
        # Temporal split by gameweek (train on earlier gameweeks, test on later)
        # Sort by season and gameweek to ensure chronological order
        df_sorted = df.sort_values(['season', 'gameweek']).reset_index(drop=True)
        
        # Calculate split point based on test_size
        split_idx = int(len(df_sorted) * (1 - config.test_size))
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        if verbose:
            print(f"  Train samples: {len(train_df):,} (up to GW{train_df['gameweek'].max() if len(train_df) > 0 else 'N/A'})")
            print(f"  Test samples: {len(test_df):,} (from GW{test_df['gameweek'].min() if len(test_df) > 0 else 'N/A'})")
        
        # Feature selection if enabled
        selected_features = None
        if config.feature_selection != 'none':
            # Get model features
            temp_model = model_class(**params)
            original_features = temp_model.FEATURES.copy()
            
            if config.feature_selection == 'importance':
                selected_features = self._select_features_importance(
                    model_class, train_df, original_features,
                    temp_model.TARGET, config.feature_selection_k, params, verbose
                )
            elif config.feature_selection == 'rfe':
                target_col = temp_model.TARGET
                selected_features = self._select_features_rfe(
                    train_df, original_features, target_col, 
                    config.feature_selection_k, verbose
                )
            
            # Update model class to use selected features
            # Note: This creates a custom model with reduced features
            if selected_features:
                model_class.FEATURES = selected_features
        
        # Create and train model on train set for evaluation
        eval_model = model_class(**params)
        eval_model.fit(train_df, verbose=False)
        
        # Get predictions and actuals based on model type
        if name == 'minutes':
            y_test = test_df['minutes'].values
            y_pred = eval_model.predict(test_df)
            target_name = 'minutes'
        elif name == 'goals':
            y_test = test_df['goals_per90'].values if 'goals_per90' in test_df.columns else (test_df['goals'] / np.maximum(test_df['minutes'] / 90, 0.01)).values
            y_pred = eval_model.predict_per90(test_df)
            target_name = 'goals_per90'
        elif name == 'assists':
            y_test = test_df['assists_per90'].values if 'assists_per90' in test_df.columns else (test_df['assists'] / np.maximum(test_df['minutes'] / 90, 0.01)).values
            y_pred = eval_model.predict_per90(test_df)
            target_name = 'assists_per90'
        elif name == 'defcon':
            y_test = test_df['defcon_per90'].values if 'defcon_per90' in test_df.columns else (test_df['defcon'] / np.maximum(test_df['minutes'] / 90, 0.01)).values
            y_pred = eval_model.predict_per90(test_df)
            target_name = 'defcon_per90'
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        if verbose:
            print(f"  Target: {target_name}")
            print(f"  Actual mean: {np.mean(y_test):.4f}")
            print(f"  Predicted mean: {np.mean(y_pred):.4f}")
        
        # Train final model on ALL data for production use
        if verbose:
            print(f"  Training final model on all {len(df):,} samples...")
        final_model = model_class(**params)
        final_model.fit(df, verbose=False)
        
        # Store final model (trained on all data)
        self.models[name] = final_model
        
        return ModelMetrics(
            model_name=name,
            mae=mae,
            rmse=rmse,
            r2=r2,
            samples=len(test_df),
            feature_importance=final_model.feature_importance() if hasattr(final_model, 'feature_importance') else None,
            selected_features=selected_features,
            best_params=params if config.optuna_trials > 0 else None,
            optuna_study=study
        )
    
    def _cv_evaluate_model(self, name: str, model_class, df: pd.DataFrame,
                           config: TuneConfig, params: Dict, study, verbose: bool) -> ModelMetrics:
        """Evaluate a model using k-fold cross-validation."""
        kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        
        cv_maes = []
        cv_rmses = []
        cv_r2s = []
        
        # Feature selection if enabled (do once before CV)
        selected_features = None
        if config.feature_selection != 'none':
            temp_model = model_class(**params)
            original_features = temp_model.FEATURES.copy()
            
            if config.feature_selection == 'importance':
                selected_features = self._select_features_importance(
                    model_class, df, original_features,
                    temp_model.TARGET, config.feature_selection_k, params, verbose
                )
            elif config.feature_selection == 'rfe':
                target_col = temp_model.TARGET
                selected_features = self._select_features_rfe(
                    df, original_features, target_col, 
                    config.feature_selection_k, verbose
                )
            
            if selected_features:
                model_class.FEATURES = selected_features
        
        if verbose:
            print(f"  Running {config.cv_folds}-fold cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            model = model_class(**params)
            model.fit(train_df, verbose=False)
            
            # Get predictions based on model type
            if name == 'minutes':
                y_test = test_df['minutes'].values
                y_pred = model.predict(test_df)
            elif name == 'goals':
                y_test = test_df['goals_per90'].values if 'goals_per90' in test_df.columns else (test_df['goals'] / np.maximum(test_df['minutes'] / 90, 0.01)).values
                y_pred = model.predict_per90(test_df)
            elif name == 'assists':
                y_test = test_df['assists_per90'].values if 'assists_per90' in test_df.columns else (test_df['assists'] / np.maximum(test_df['minutes'] / 90, 0.01)).values
                y_pred = model.predict_per90(test_df)
            elif name == 'defcon':
                y_test = test_df['defcon_per90'].values if 'defcon_per90' in test_df.columns else (test_df['defcon'] / np.maximum(test_df['minutes'] / 90, 0.01)).values
                y_pred = model.predict_per90(test_df)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            cv_maes.append(mae)
            cv_rmses.append(rmse)
            cv_r2s.append(r2)
            
            if verbose:
                print(f"    Fold {fold+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        
        # Train final model on all data for feature importance
        final_model = model_class(**params)
        final_model.fit(df, verbose=False)
        self.models[name] = final_model
        
        return ModelMetrics(
            model_name=name,
            mae=np.mean(cv_maes),
            rmse=np.mean(cv_rmses),
            r2=np.mean(cv_r2s),
            samples=len(df),
            cv_scores=cv_maes,
            feature_importance=final_model.feature_importance() if hasattr(final_model, 'feature_importance') else None,
            selected_features=selected_features,
            best_params=params if config.optuna_trials > 0 else None,
            optuna_study=study
        )
    
    def get_model(self, name: str):
        """Get a trained model by name."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self.models.keys())}")
        return self.models[name]
    
    def get_tune_results(self) -> Dict[str, ModelMetrics]:
        """Get results from the last tune() call."""
        return self.tune_results
    
    # =========================================================================
    # FEATURE SELECTION METHODS
    # =========================================================================
    
    def _select_features_importance(self, model_class, df: pd.DataFrame, 
                                     features: List[str], target_col: str,
                                     k: int, params: Dict, verbose: bool) -> List[str]:
        """Select top k features by importance using a trained model."""
        if verbose:
            print(f"  Feature selection: importance-based (top {k} features)")
        
        # Train a model to get feature importances
        temp_model = model_class(**params)
        temp_model.fit(df, verbose=False)
        
        # Get feature importance
        fi = temp_model.feature_importance()
        top_features = fi.head(k)['feature'].tolist()
        
        if verbose:
            print(f"    Selected {len(top_features)} features")
            print(f"    Top 5: {top_features[:5]}")
        
        return top_features
    
    def _select_features_rfe(self, df: pd.DataFrame, features: List[str], 
                              target_col: str, k: int, verbose: bool) -> List[str]:
        """Select features using Recursive Feature Elimination."""
        import xgboost as xgb
        available_features = [f for f in features if f in df.columns]
        features = available_features
        if verbose:
            print(f"  Feature selection: RFE (top {k} features)")
        
        X = df[features].fillna(0).astype(float)
        y = df[target_col].fillna(0)
        
        # Use XGBoost as base estimator
        base_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
        
        # RFE
        rfe = RFE(base_model, n_features_to_select=k, step=5)
        rfe.fit(X, y)
        
        selected_features = [f for f, selected in zip(features, rfe.support_) if selected]
        
        if verbose:
            print(f"    Selected {len(selected_features)} features")
            print(f"    Top 5: {selected_features[:5]}")
        
        return selected_features
    
    # =========================================================================
    # OPTUNA HYPERPARAMETER OPTIMIZATION
    # =========================================================================
    
    def _optuna_optimize(self, name: str, model_class, df: pd.DataFrame,
                         config: TuneConfig, verbose: bool) -> Tuple[Dict[str, Any], Any]:
        """Run Optuna hyperparameter optimization."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Run: pip install optuna")
        
        if verbose:
            print(f"  Running Optuna optimization ({config.optuna_trials} trials)...")
        
        search_space = config.optuna_search_space().get(name, {})
        
        # Split data for optimization
        train_df, val_df = train_test_split(
            df, test_size=config.test_size, random_state=config.random_state
        )
        
        def objective(trial):
            # Sample hyperparameters
            params = {'random_state': config.random_state}
            
            for param_name, param_config in search_space.items():
                param_type = param_config[0]
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
                elif param_type == 'float_log':
                    params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=True)
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config[1])
            
            # Train model
            model = model_class(**params)
            model.fit(train_df, verbose=False)
            
            # Evaluate
            if name == 'minutes':
                y_val = val_df['minutes'].values
                y_pred = model.predict(val_df)
            elif name == 'bonus':
                y_val = val_df['bonus'].values
                y_pred = model.predict(val_df)
            else:
                target_col = f'{name}_per90'
                if target_col not in val_df.columns:
                    y_val = (val_df[name] / np.maximum(val_df['minutes'] / 90, 0.01)).values
                else:
                    y_val = val_df[target_col].values
                y_pred = model.predict_per90(val_df)
            
            return mean_absolute_error(y_val, y_pred)
        
        # Create and run study
        sampler = TPESampler(seed=config.random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        # Suppress Optuna logs unless verbose
        optuna.logging.set_verbosity(optuna.logging.WARNING if not verbose else optuna.logging.INFO)
        
        study.optimize(
            objective, 
            n_trials=config.optuna_trials, 
            timeout=config.optuna_timeout,
            show_progress_bar=verbose
        )
        
        best_params = study.best_params
        best_params['random_state'] = config.random_state
        
        if verbose:
            print(f"    Best MAE: {study.best_value:.4f}")
            print(f"    Best params: {best_params}")
        
        return best_params, study
    
    # =========================================================================
    # BONUS MODEL TUNING
    # =========================================================================
    
    def _tune_bonus_model(self, df: pd.DataFrame, config: TuneConfig, verbose: bool) -> ModelMetrics:
        """Tune and evaluate the bonus model."""
        params = config.get_hyperparams('bonus')
        
        # Bonus model needs special handling - requires bonus column
        # Filter to players with 60+ minutes (bonus eligibility)
        bonus_df = df[df['minutes'] >= 60].copy()
        
        if 'bonus' not in bonus_df.columns:
            if verbose:
                print("  WARNING: No bonus data available. Fetching from FPL API...")
            # Try to fetch bonus data
            try:
                unique_players = set(bonus_df['player_name'].unique())
                fpl_bonus = fetch_fpl_bonus_data(self.data_dir, unique_players, verbose=False)
                if len(fpl_bonus) > 0:
                    bonus_df = bonus_df.merge(
                        fpl_bonus, 
                        left_on=['player_name', 'gameweek'], 
                        right_on=['fbref_name', 'gameweek'], 
                        how='left'
                    )
                    bonus_df['bonus'] = bonus_df['bonus'].fillna(0)
                else:
                    if verbose:
                        print("  ERROR: Could not fetch bonus data.")
                    return ModelMetrics(
                        model_name='bonus', mae=0, rmse=0, r2=0, samples=0
                    )
            except Exception as e:
                if verbose:
                    print(f"  ERROR fetching bonus data: {e}")
                return ModelMetrics(
                    model_name='bonus', mae=0, rmse=0, r2=0, samples=0
                )
        
        # Filter to rows with valid bonus data
        bonus_df = bonus_df[bonus_df['bonus'].notna() & (bonus_df['bonus'] >= 0)]
        
        if len(bonus_df) < 100:
            if verbose:
                print(f"  WARNING: Only {len(bonus_df)} samples with bonus data. Skipping.")
            return ModelMetrics(
                model_name='bonus', mae=0, rmse=0, r2=0, samples=len(bonus_df)
            )
        
        # Optuna optimization if enabled
        if config.optuna_trials > 0:
            params, study = self._optuna_optimize('bonus', BonusModel, bonus_df, config, verbose)
        else:
            study = None
        
        if config.cv_folds:
            return self._cv_evaluate_bonus_model(bonus_df, config, params, study, verbose)
        else:
            return self._split_evaluate_bonus_model(bonus_df, config, params, study, verbose)
    
    def _split_evaluate_bonus_model(self, df: pd.DataFrame, config: TuneConfig, 
                                 params: Dict, study, verbose: bool) -> ModelMetrics:
        """Evaluate bonus model using train/test split."""
        # Temporal split by gameweek
        df_sorted = df.sort_values(['season', 'gameweek']).reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - config.test_size))
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        if verbose:
            print(f"  Train samples: {len(train_df):,} (up to GW{train_df['gameweek'].max() if len(train_df) > 0 else 'N/A'})")
            print(f"  Test samples: {len(test_df):,} (from GW{test_df['gameweek'].min() if len(test_df) > 0 else 'N/A'})")
        
        # Feature selection if enabled
        selected_features = None
        if config.feature_selection != 'none':
            temp_model = BonusModel(n_simulations=100, **params)  # Few sims for feature selection
            original_features = temp_model.FEATURES.copy()
            
            if config.feature_selection == 'importance':
                selected_features = self._select_features_importance(
                    BonusModel, train_df, original_features,
                    temp_model.TARGET, config.feature_selection_k, params, verbose
                )
            elif config.feature_selection == 'rfe':
                target_col = temp_model.TARGET
                selected_features = self._select_features_rfe(
                    train_df, original_features, target_col, 
                    config.feature_selection_k, verbose
                )
            
            if selected_features:
                BonusModel.FEATURES = selected_features
        
        # Create and train model on train set for evaluation
        eval_model = BonusModel(n_simulations=500, **params)  # Fewer sims for faster tuning
        eval_model.fit(train_df, verbose=False)
        
        y_test = test_df['bonus'].values
        # For tuning, use simple prediction (no other model predictions available)
        # Use historical averages as proxies for predictions
        test_df_copy = test_df.copy()
        test_df_copy['pred_exp_goals'] = test_df_copy.get('xg_per90_roll5', 0).fillna(0) * 0.5
        test_df_copy['pred_exp_assists'] = test_df_copy.get('xag_per90_roll5', 0).fillna(0) * 0.5
        test_df_copy['pred_cs_prob'] = 0.25  # Default CS prob
        test_df_copy['pred_minutes'] = test_df_copy.get('roll5_minutes_avg', 60).fillna(60)
        y_pred = eval_model.predict(test_df_copy)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        if verbose:
            print(f"  Target: bonus (0-3)")
            print(f"  Actual mean: {np.mean(y_test):.4f}")
            print(f"  Predicted mean: {np.mean(y_pred):.4f}")
        
        # Train final model on ALL data
        if verbose:
            print(f"  Training final model on all {len(df):,} samples...")
        final_model = BonusModel(n_simulations=1000, **params)  # Full sims for final model
        final_model.fit(df, verbose=False)
        
        self.models['bonus'] = final_model
        
        return ModelMetrics(
            model_name='bonus',
            mae=mae,
            rmse=rmse,
            r2=r2,
            samples=len(test_df),
            feature_importance=final_model.feature_importance(),
            selected_features=selected_features,
            best_params=params if config.optuna_trials > 0 else None,
            optuna_study=study
        )
    def _cv_evaluate_bonus_model(self, df: pd.DataFrame, config: TuneConfig,
                                params: Dict, study, verbose: bool) -> ModelMetrics:
        """Evaluate bonus model using cross-validation."""
        kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        
        cv_maes = []
        cv_rmses = []
        cv_r2s = []
        
        # Feature selection if enabled (do once before CV)
        selected_features = None
        if config.feature_selection != 'none':
            temp_model = BonusModel(n_simulations=100, **params)  # Few sims for feature selection
            original_features = temp_model.FEATURES.copy()
            
            if config.feature_selection == 'importance':
                selected_features = self._select_features_importance(
                    BonusModel, df, original_features,  # FIX: Use BonusModel, not model_class
                    temp_model.TARGET, config.feature_selection_k, params, verbose
                )
            elif config.feature_selection == 'rfe':
                target_col = temp_model.TARGET
                selected_features = self._select_features_rfe(
                    df, original_features, target_col, 
                    config.feature_selection_k, verbose
                )
            
            if selected_features:
                BonusModel.FEATURES = selected_features  # FIX: Use BonusModel, not model_class
        
        if verbose:
            print(f"  Running {config.cv_folds}-fold cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            
            model = BonusModel(n_simulations=500, **params)  # Fewer sims for faster tuning
            model.fit(train_df, verbose=False)
            
            y_test = test_df['bonus'].values
            # For tuning, use historical averages as proxies for predictions
            test_df['pred_exp_goals'] = test_df.get('xg_per90_roll5', 0).fillna(0) * 0.5
            test_df['pred_exp_assists'] = test_df.get('xag_per90_roll5', 0).fillna(0) * 0.5
            test_df['pred_cs_prob'] = 0.25  # Default CS prob
            test_df['pred_minutes'] = test_df.get('roll5_minutes_avg', 60).fillna(60)
            y_pred = model.predict(test_df)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            cv_maes.append(mae)
            cv_rmses.append(rmse)
            cv_r2s.append(r2)
            
            if verbose:
                print(f"    Fold {fold+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        
        # Train final model on all data
        final_model = BonusModel(n_simulations=1000, **params)  # Full sims for final model
        final_model.fit(df, verbose=False)
        self.models['bonus'] = final_model
        
        return ModelMetrics(
            model_name='bonus',
            mae=np.mean(cv_maes),
            rmse=np.mean(cv_rmses),
            r2=np.mean(cv_r2s),
            samples=len(df),
            cv_scores=cv_maes,
            feature_importance=final_model.feature_importance(),
            selected_features=selected_features,  # FIX: Add this
            best_params=params if config.optuna_trials > 0 else None,
            optuna_study=study
        )
        
    def run(self, target_gw: int, target_season: str = '2025-26', verbose: bool = True) -> pd.DataFrame:
        """Run the complete prediction pipeline."""
        
        if verbose:
            print('=' * 80)
            print(f'FPL PREDICTION PIPELINE - GW{target_gw} {target_season}')
            print('=' * 80)
        
        # Step 1: Load data
        self.df = load_raw_data(self.data_dir, verbose)
        
        # Step 2: Compute features
        self.df = compute_rolling_features(self.df, verbose)
        
        # Step 3: Split train/test
        if verbose:
            print('[3] Splitting train/test data...')
        
        # Check if we have data for target gameweek (past GW) or need to predict upcoming
        existing_data = self.df[(self.df['season'] == target_season) & (self.df['gameweek'] == target_gw)]
        
        if len(existing_data) > 0:
            # Past gameweek with actual data - use for validation
            self.train_df = self.df[~((self.df['season'] == target_season) & (self.df['gameweek'] == target_gw))].copy()
            self.test_df = existing_data.copy()
            if verbose:
                print(f'  Mode: Validation (GW{target_gw} data exists)')
        else:
            # Upcoming gameweek - build test set from upcoming fixtures
            if verbose:
                print(f'  Mode: Prediction (GW{target_gw} is upcoming)')
            self.train_df = self.df.copy()
            self.test_df = self._build_upcoming_test_set(target_gw, target_season, verbose)
        
        if verbose:
            print(f'  Train: {len(self.train_df)} records, Test: {len(self.test_df)} records')
        
        # Step 4: Train and predict with all models
        self._train_minutes_model(verbose)
        self._train_goals_model(verbose)
        self._train_assists_model(verbose)
        self._train_defcon_model(verbose)
        self._train_clean_sheet_model(target_season, target_gw, verbose)
        self._train_bonus_model(verbose)
        
        # Step 5: Get FPL positions and calculate expected points
        if verbose:
            print('[9] Fetching FPL positions and calculating expected points...')
        
        # Fetch official FPL positions
        player_names = set(self.test_df['player_name'].unique())
        fpl_positions = fetch_fpl_positions(self.data_dir, player_names, verbose)
        
        # Apply FPL positions (with fallback to FBref-derived position)
        def get_position(row):
            if row['player_name'] in fpl_positions:
                return fpl_positions[row['player_name']]
            return get_fpl_position(row['position'])  # Fallback
        
        self.test_df['fpl_position'] = self.test_df.apply(get_position, axis=1)
        
        if verbose:
            pos_counts = self.test_df['fpl_position'].value_counts()
            print(f'  Position distribution: {pos_counts.to_dict()}')
        
        points_df = self.test_df.apply(calculate_expected_points, axis=1)
        self.test_df = pd.concat([self.test_df, points_df], axis=1)
        
        # Save predictions
        output_path = self.data_dir / 'predictions' / f'gw{target_gw}_{target_season}_predictions.csv'
        output_path.parent.mkdir(exist_ok=True)
        self.test_df.to_csv(output_path, index=False)
        
        if verbose:
            print(f'\n✓ Predictions saved to: {output_path}')
            print('=' * 80)
            print('PIPELINE COMPLETE!')
            print('=' * 80)
            for model_name,model in self.models.items():
                #print(f'{model_name}: {model.score(self.test_df[model.FEATURES], self.test_df[model_name])}')
                print(model_name,"FEATURE_IMPORTANCE:")
                print(model.feature_importance())
                
        
        return self.test_df
    
    def _train_minutes_model(self, verbose: bool):
        if verbose:
            print('[4] Training minutes model...')
        
        # Check if model already exists (from tuning)
        if 'minutes' in self.models and self.models['minutes'].is_fitted:
            if verbose:
                print('  Using existing tuned minutes model')
        else:
            self.models['minutes'] = MinutesModel()
            self.models['minutes'].fit(self.train_df, verbose=verbose)
        
        self.test_df['pred_minutes'] = self.models['minutes'].predict(self.test_df)
        self.train_df['pred_minutes'] = self.models['minutes'].predict(self.train_df)
    
    def _train_goals_model(self, verbose: bool):
        if verbose:
            print('[5] Training goals model...')
        
        # Check if model already exists (from tuning)
        if 'goals' in self.models and self.models['goals'].is_fitted:
            if verbose:
                print('  Using existing tuned goals model')
        else:
            self.models['goals'] = GoalsModel()
            self.models['goals'].fit(self.train_df, verbose=verbose)
        
        self.test_df['pred_goals_per90'] = self.models['goals'].predict_per90(self.test_df)
        self.test_df['pred_exp_goals'] = self.models['goals'].predict_expected(self.test_df, self.test_df['pred_minutes'].values)
    
    def _train_assists_model(self, verbose: bool):
        if verbose:
            print('[6] Training assists model...')
        
        # Check if model already exists (from tuning)
        if 'assists' in self.models and self.models['assists'].is_fitted:
            if verbose:
                print('  Using existing tuned assists model')
        else:
            self.models['assists'] = AssistsModel()
            self.models['assists'].fit(self.train_df, verbose=verbose)
        
        self.test_df['pred_assists_per90'] = self.models['assists'].predict_per90(self.test_df)
        self.test_df['pred_exp_assists'] = self.models['assists'].predict_expected(self.test_df, self.test_df['pred_minutes'].values)
    
    def _train_defcon_model(self, verbose: bool):
        if verbose:
            print('[7] Training defcon model...')
        
        # Check if model already exists (from tuning)
        if 'defcon' in self.models and self.models['defcon'].is_fitted:
            if verbose:
                print('  Using existing tuned defcon model')
        else:
            self.models['defcon'] = DefconModel()
            self.models['defcon'].fit(self.train_df, verbose=verbose)
        
        self.test_df['pred_defcon_per90'] = self.models['defcon'].predict_per90(self.test_df)
        self.test_df['pred_exp_defcon'] = self.models['defcon'].predict_expected(self.test_df, self.test_df['pred_minutes'].values)
        self.test_df['pred_defcon_prob'] = self.models['defcon'].predict_proba_above_threshold(self.test_df, self.test_df['pred_minutes'].values)
    
    def _train_clean_sheet_model(self, target_season: str, target_gw: int, verbose: bool):
        if verbose:
            print('[8] Training clean sheet model...')
        self.models['clean_sheet'] = CleanSheetModel()
        team_features = self.models['clean_sheet'].prepare_features(self.df)
        team_train = team_features[~((team_features['season'] == target_season) & (team_features['gameweek'] == target_gw))]
        team_test = team_features[(team_features['season'] == target_season) & (team_features['gameweek'] == target_gw)]
        
        self.models['clean_sheet'].fit(team_train, verbose=verbose)
        
        # For upcoming gameweeks, team_test will be empty - need to build it from fixtures
        if len(team_test) == 0 and len(self.test_df) > 0:
            if verbose:
                print('  Building CS predictions for upcoming fixtures...')
            
            # Get unique team/opponent pairs from test_df
            fixtures = self.test_df[['team', 'opponent', 'is_home']].drop_duplicates()
            
            # Get latest team stats from CURRENT SEASON (not previous season!)
            # First try current season data, fall back to previous if team not found
            current_season_data = team_train[team_train['season'] == target_season]
            if verbose:
                print(f'    Using {target_season} data (GW1-{current_season_data["gameweek"].max() if len(current_season_data) > 0 else 0})')
            
            # Get latest stats per team from current season
            # Sort by gameweek to ensure we get the latest row
            current_season_sorted = current_season_data.sort_values(['team', 'gameweek'])
            latest_team_stats = current_season_sorted.groupby('team').last().reset_index()
            
            # Also compute defensive stats directly (goals_conceded in recent matches)
            team_defensive = current_season_sorted.groupby('team').apply(
                lambda g: pd.Series({
                    'team_conceded_last1': g['goals_conceded'].iloc[-1] if len(g) > 0 else 1,
                    'team_conceded_roll3': g['goals_conceded'].tail(3).mean() if len(g) >= 1 else 1,
                    'team_conceded_roll5': g['goals_conceded'].tail(5).mean() if len(g) >= 1 else 1,
                    'latest_gw': g['gameweek'].iloc[-1] if len(g) > 0 else 0,
                })
            ).reset_index()
            
            if verbose:
                # Debug: Show what we're using for Arsenal and Wolves
                print(f'    Teams in current season data: {len(team_defensive)}')
                arsenal_def = team_defensive[team_defensive['team'].str.contains('arsenal', case=False, na=False)]
                wolves_def = team_defensive[team_defensive['team'].str.contains('wolverhampton', case=False, na=False)]
                if len(arsenal_def) > 0:
                    print(f'    DEBUG Arsenal DEF: conceded_last1={arsenal_def.iloc[0]["team_conceded_last1"]}, latest_gw={arsenal_def.iloc[0]["latest_gw"]}')
                else:
                    print(f'    DEBUG Arsenal: NOT FOUND in team_defensive!')
                if len(wolves_def) > 0:
                    print(f'    DEBUG Wolves DEF: conceded_last1={wolves_def.iloc[0]["team_conceded_last1"]}, latest_gw={wolves_def.iloc[0]["latest_gw"]}')
            
            # Also compute each team's attacking form (goals they scored)
            # IMPORTANT: Sort by gameweek to ensure we get the latest data
            team_attacking = current_season_sorted.groupby('team').apply(
                lambda g: pd.Series({
                    'team_goals_last1': g['goals'].iloc[-1] if len(g) > 0 else 0,
                    'team_goals_roll3': g['goals'].tail(3).mean() if len(g) >= 1 else 0,
                    'team_goals_roll5': g['goals'].tail(5).mean() if len(g) >= 1 else 0,
                    'latest_gw': g['gameweek'].iloc[-1] if len(g) > 0 else 0,
                })
            ).reset_index()
            
            if verbose:
                # Debug: Show what we're using for Wolves (Arsenal's opponent)
                wolves_atk = team_attacking[team_attacking['team'].str.contains('wolverhampton', case=False, na=False)]
                arsenal_atk = team_attacking[team_attacking['team'].str.contains('arsenal', case=False, na=False)]
                if len(wolves_atk) > 0:
                    print(f'    DEBUG Wolves ATK: goals_last1={wolves_atk.iloc[0]["team_goals_last1"]}, latest_gw={wolves_atk.iloc[0]["latest_gw"]}')
                else:
                    print(f'    DEBUG Wolves: NOT FOUND in team_attacking!')
                if len(arsenal_atk) > 0:
                    print(f'    DEBUG Arsenal ATK: goals_last1={arsenal_atk.iloc[0]["team_goals_last1"]}, latest_gw={arsenal_atk.iloc[0]["latest_gw"]}')
            
            # For teams not in current season, fall back to previous season
            prev_season_data = team_train[team_train['season'] != target_season]
            prev_latest = prev_season_data.sort_values(['team', 'season', 'gameweek']).groupby('team').last().reset_index()
            
            # Also compute previous season attacking form
            if len(prev_season_data) > 0:
                prev_sorted = prev_season_data.sort_values(['team', 'season', 'gameweek'])
                prev_attacking = prev_sorted.groupby('team').apply(
                    lambda g: pd.Series({
                        'team_goals_last1': g['goals'].iloc[-1] if len(g) > 0 else 0,
                        'team_goals_roll3': g['goals'].tail(3).mean() if len(g) >= 1 else 0,
                        'team_goals_roll5': g['goals'].tail(5).mean() if len(g) >= 1 else 0,
                        'latest_gw': g['gameweek'].iloc[-1] if len(g) > 0 else 0,
                    })
                ).reset_index()
            else:
                prev_attacking = pd.DataFrame()
            
            # Combine: current season takes priority
            teams_in_current = set(latest_team_stats['team'].unique())
            prev_only = prev_latest[~prev_latest['team'].isin(teams_in_current)]
            latest_team_stats = pd.concat([latest_team_stats, prev_only], ignore_index=True)
            
            # Combine attacking stats
            teams_in_current_atk = set(team_attacking['team'].unique()) if len(team_attacking) > 0 else set()
            if len(prev_attacking) > 0:
                prev_atk_only = prev_attacking[~prev_attacking['team'].isin(teams_in_current_atk)]
                team_attacking = pd.concat([team_attacking, prev_atk_only], ignore_index=True)
            
            team_test_rows = []
            for _, fix in fixtures.iterrows():
                team = fix['team']
                opponent = fix['opponent']
                is_home = fix['is_home']
                
                # Get team's defensive stats
                team_stats = latest_team_stats[latest_team_stats['team'] == team]
                opp_stats = latest_team_stats[latest_team_stats['team'] == opponent]
                
                if len(team_stats) == 0:
                    # Try fuzzy match
                    for t in latest_team_stats['team'].unique():
                        if team.lower().replace('_', ' ') in t.lower() or t.lower() in team.lower().replace('_', ' '):
                            team_stats = latest_team_stats[latest_team_stats['team'] == t]
                            break
                
                if len(opp_stats) == 0:
                    for t in latest_team_stats['team'].unique():
                        if opponent.lower().replace('_', ' ') in t.lower() or t.lower() in opponent.lower().replace('_', ' '):
                            opp_stats = latest_team_stats[latest_team_stats['team'] == t]
                            break
                
                row = {
                    'team': team,
                    'opponent': opponent,
                    'season': target_season,
                    'gameweek': target_gw,
                    'is_home': is_home,
                }
                
                # Copy team defensive features - use team_defensive which has correct recent data
                # Team name aliases for matching FPL names to FBref names
                TEAM_ALIASES = {
                    'wolves': 'wolverhampton', 'spurs': 'tottenham', 'man city': 'manchester_city',
                    'man utd': 'manchester_united', 'nottm forest': 'nottingham_forest',
                    "nott'm forest": 'nottingham_forest', 'west ham': 'west_ham',
                }
                
                def match_team_def(search_name, team_list):
                    """Match a team name to the list of teams."""
                    search_lower = search_name.lower().replace('_', ' ')
                    # Direct match
                    for t in team_list:
                        if search_lower == t.lower().replace('_', ' '):
                            return t
                    # Alias match
                    for alias, target in TEAM_ALIASES.items():
                        if alias in search_lower or search_lower in alias:
                            for t in team_list:
                                if target in t.lower():
                                    return t
                    # Partial match
                    for t in team_list:
                        t_lower = t.lower()
                        if search_lower in t_lower or t_lower in search_lower:
                            return t
                        # Check individual words
                        for word in search_lower.split():
                            if len(word) > 3 and word in t_lower:
                                return t
                    return None
                
                team_def = team_defensive[team_defensive['team'] == team] if len(team_defensive) > 0 else pd.DataFrame()
                if len(team_def) == 0 and len(team_defensive) > 0:
                    matched_team = match_team_def(team, team_defensive['team'].unique())
                    if matched_team:
                        team_def = team_defensive[team_defensive['team'] == matched_team]
                
                if len(team_def) > 0:
                    td = team_def.iloc[0]
                    row['team_goals_conceded_last1'] = td.get('team_conceded_last1', 1.0)
                    row['team_goals_conceded_roll3'] = td.get('team_conceded_roll3', 1.0)
                    row['team_goals_conceded_roll5'] = td.get('team_conceded_roll5', 1.0)
                else:
                    row['team_goals_conceded_last1'] = 1.0
                    row['team_goals_conceded_roll3'] = 1.0
                    row['team_goals_conceded_roll5'] = 1.0
                
                # Other features from team_stats
                if len(team_stats) > 0:
                    ts = team_stats.iloc[0]
                    row['team_goals_conceded_roll10'] = ts.get('team_goals_conceded_roll10', 1.0)
                    row['team_goals_conceded_roll30'] = ts.get('team_goals_conceded_roll30', 1.0)
                    row['team_xga_roll5'] = ts.get('team_xga_roll5', 1.0)
                    row['team_clean_sheets_roll5'] = ts.get('team_clean_sheets_roll5', 0)
                    row['team_clean_sheets_roll10'] = ts.get('team_clean_sheets_roll10', 0)
                    row['team_clean_sheets_roll30'] = ts.get('team_clean_sheets_roll30', 0)
                    row['team_xga_roll5_home'] = ts.get('team_xga_roll5_home', 1.0)
                    row['team_xga_roll5_away'] = ts.get('team_xga_roll5_away', 1.0)
                    row['team_encoded'] = ts.get('team_encoded', -1)
                else:
                    row['team_goals_conceded_roll10'] = 1.0
                    row['team_goals_conceded_roll30'] = 1.0
                    row['team_xga_roll5'] = 1.0
                    row['team_clean_sheets_roll5'] = 0
                    row['team_clean_sheets_roll10'] = 0
                    row['team_clean_sheets_roll30'] = 0
                    row['team_xga_roll5_home'] = 1.0
                    row['team_xga_roll5_away'] = 1.0
                    row['team_encoded'] = -1
                
                # Copy opponent attacking features - use team_attacking which has their actual goals scored
                # Team name aliases for matching FPL names to FBref names
                TEAM_ALIASES = {
                    'wolves': 'wolverhampton', 'spurs': 'tottenham', 'man city': 'manchester_city',
                    'man utd': 'manchester_united', 'nottm forest': 'nottingham_forest',
                    "nott'm forest": 'nottingham_forest', 'west ham': 'west_ham',
                }
                
                def match_team(search_name, team_list):
                    """Match a team name to the list of teams."""
                    search_lower = search_name.lower().replace('_', ' ')
                    # Direct match
                    for t in team_list:
                        if search_lower == t.lower().replace('_', ' '):
                            return t
                    # Alias match
                    for alias, target in TEAM_ALIASES.items():
                        if alias in search_lower or search_lower in alias:
                            for t in team_list:
                                if target in t.lower():
                                    return t
                    # Partial match
                    for t in team_list:
                        t_lower = t.lower()
                        if search_lower in t_lower or t_lower in search_lower:
                            return t
                        # Check individual words
                        for word in search_lower.split():
                            if len(word) > 3 and word in t_lower:
                                return t
                    return None
                
                # Look up opponent in team_attacking DataFrame
                opp_atk = team_attacking[team_attacking['team'] == opponent] if len(team_attacking) > 0 else pd.DataFrame()
                if len(opp_atk) == 0 and len(team_attacking) > 0:
                    matched_team = match_team(opponent, team_attacking['team'].unique())
                    if matched_team:
                        opp_atk = team_attacking[team_attacking['team'] == matched_team]
                
                if len(opp_atk) > 0:
                    oa = opp_atk.iloc[0]
                    row['opp_goals_scored_last1'] = oa.get('team_goals_last1', 1.5)
                    row['opp_goals_scored_roll3'] = oa.get('team_goals_roll3', 1.5)
                    row['opp_goals_scored_roll5'] = oa.get('team_goals_roll5', 1.5)
                    row['opp_goals_scored_roll10'] = oa.get('team_goals_roll5', 1.5)  # Fallback to roll5
                    row['opp_goals_scored_roll30'] = oa.get('team_goals_roll5', 1.5)  # Fallback to roll5
                else:
                    row['opp_goals_scored_last1'] = 1.5
                    row['opp_goals_scored_roll3'] = 1.5
                    row['opp_goals_scored_roll5'] = 1.5
                    row['opp_goals_scored_roll10'] = 1.5
                    row['opp_goals_scored_roll30'] = 1.5
                
                # Other opponent features from opp_stats
                # First try to match opp_stats with same alias logic
                if len(opp_stats) == 0 and len(latest_team_stats) > 0:
                    matched_opp = match_team(opponent, latest_team_stats['team'].unique())
                    if matched_opp:
                        opp_stats = latest_team_stats[latest_team_stats['team'] == matched_opp]
                
                if len(opp_stats) > 0:
                    os = opp_stats.iloc[0]
                    row['opp_xg_roll5'] = os.get('team_xga_roll5', 1.5)  # Use opponent's xga as their attacking xG
                    row['opp_xg_roll5_home'] = os.get('team_xga_roll5_home', 1.5)
                    row['opp_xg_roll5_away'] = os.get('team_xga_roll5_away', 1.5)
                    row['opponent_encoded'] = os.get('team_encoded', -1)
                else:
                    row['opp_xg_roll5'] = 1.5
                    row['opp_xg_roll5_home'] = 1.5
                    row['opp_xg_roll5_away'] = 1.5
                    row['opponent_encoded'] = -1
                
                # Compute derived features
                row['xga_xg_ratio'] = row.get('team_xga_roll5', 1.0) / (row.get('opp_xg_roll5', 1.5) + 0.1)
                row['defensive_advantage'] = row.get('opp_xg_roll5', 1.5) - row.get('team_xga_roll5', 1.0)
                
                team_test_rows.append(row)
            
            team_test = pd.DataFrame(team_test_rows)
            if verbose:
                print(f'    Built {len(team_test)} team fixtures for CS prediction')
        
        if len(team_test) > 0:
            team_test['pred_cs_prob'] = self.models['clean_sheet'].predict_proba(team_test)
            
            # Include all CS features in the lookup, not just pred_cs_prob
            # Exclude 'is_home' since it already exists in test_df
            cs_features = self.models['clean_sheet'].FEATURES + ['pred_cs_prob']
            exclude_cols = ['is_home']  # Already in test_df
            cs_cols = ['team', 'season', 'gameweek'] + [c for c in cs_features if c in team_test.columns and c not in exclude_cols]
            cs_lookup = team_test[cs_cols].drop_duplicates(subset=['team', 'season', 'gameweek'])
            
            if verbose:
                # Debug: show Arsenal's features
                arsenal_cs = cs_lookup[cs_lookup['team'].str.contains('arsenal', case=False, na=False)]
                if len(arsenal_cs) > 0:
                    print(f'    Arsenal CS features:')
                    print(f'      team_goals_conceded_last1: {arsenal_cs.iloc[0].get("team_goals_conceded_last1", "N/A")}')
                    print(f'      opp_goals_scored_last1: {arsenal_cs.iloc[0].get("opp_goals_scored_last1", "N/A")}')
                    print(f'      pred_cs_prob: {arsenal_cs.iloc[0]["pred_cs_prob"]:.1%}')
            
            self.test_df = self.test_df.merge(cs_lookup, on=['team', 'season', 'gameweek'], how='left')
        
        self.test_df['pred_cs_prob'] = self.test_df['pred_cs_prob'].fillna(0.2)
    
    def _train_bonus_model(self, verbose: bool):
        if verbose:
            print('[8.5] Training bonus model (Monte Carlo simulation)...')
        
        # Check if model already exists (from tuning)
        if 'bonus' in self.models and self.models['bonus'].is_fitted:
            if verbose:
                print('  Using existing tuned bonus model')
            # Predict using the existing tuned model
            self.test_df['pred_bonus'] = self.models['bonus'].predict(
                self.test_df,
                pred_exp_goals=self.test_df['pred_exp_goals'].values,
                pred_exp_assists=self.test_df['pred_exp_assists'].values,
                pred_cs_prob=self.test_df['pred_cs_prob'].values,
                pred_minutes=self.test_df['pred_minutes'].values,
            )
            if verbose:
                print(f'  Monte Carlo simulation complete')
                print(f'  Avg predicted bonus: {self.test_df["pred_bonus"].mean():.3f}')
            return
        
        # Train new model if not already tuned
        unique_players = set(self.train_df['player_name'].unique()) | set(self.test_df['player_name'].unique())
        fpl_bonus = fetch_fpl_bonus_data(self.data_dir, unique_players, verbose)
        
        if len(fpl_bonus) > 0:
            current_season_train = self.train_df[self.train_df['season'] == '2025-26'].copy()
            current_season_train = current_season_train.merge(
                fpl_bonus, 
                left_on=['player_name', 'gameweek'], 
                right_on=['fbref_name', 'gameweek'], 
                how='left'
            )
            current_season_train['bonus'] = current_season_train['bonus'].fillna(0)
            if 'bps' in current_season_train.columns:
                current_season_train['bps'] = current_season_train['bps'].fillna(0)
            
            train_bonus_df = current_season_train[current_season_train['minutes'] >= 60].copy()
            
            # Train the Monte Carlo bonus model (fits baseline BPS model)
            self.models['bonus'] = BonusModel(n_simulations=1000)
            self.models['bonus'].fit(train_bonus_df, verbose=verbose)
            
            # Predict using Monte Carlo simulation with our model predictions
            # Use the predictions already computed by goals, assists, CS, minutes models
            self.test_df['pred_bonus'] = self.models['bonus'].predict(
                self.test_df,
                pred_exp_goals=self.test_df['pred_exp_goals'].values,
                pred_exp_assists=self.test_df['pred_exp_assists'].values,
                pred_cs_prob=self.test_df['pred_cs_prob'].values,
                pred_minutes=self.test_df['pred_minutes'].values,
            )
            
            if verbose:
                print(f'  Monte Carlo simulation complete')
                print(f'  Avg predicted bonus: {self.test_df["pred_bonus"].mean():.3f}')
        else:
            self.test_df['pred_bonus'] = 0
    
    def _build_upcoming_test_set(self, target_gw: int, target_season: str, verbose: bool) -> pd.DataFrame:
        """Build test set for upcoming gameweek from fixtures and player history.
        
        IMPORTANT: Only uses data from PRIOR gameweeks (not target_gw).
        """
        if verbose:
            print(f'  Building prediction set for GW{target_gw}...')
        
        # IMPORTANT: Use only prior gameweek data for features
        prior_data = self.df[~((self.df['season'] == target_season) & (self.df['gameweek'] >= target_gw))].copy()
        if verbose:
            max_gw = prior_data[prior_data['season'] == target_season]['gameweek'].max() if len(prior_data[prior_data['season'] == target_season]) > 0 else 0
            print(f'    Using historical data up to GW{max_gw}')
        
        # Get fixtures from FPL API
        import requests
        bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
        fpl_teams = {t['id']: t['name'] for t in bootstrap['teams']}
        fpl_teams_short = {t['id']: t['short_name'] for t in bootstrap['teams']}
        fixtures = requests.get("https://fantasy.premierleague.com/api/fixtures/").json()
        gw_fixtures = [f for f in fixtures if f.get('event') == target_gw]
        
        # Build fixtures list with all name variants
        fixtures_list = []
        teams_playing_variants = {}  # Maps normalized names to FPL name
        
        for f in gw_fixtures:
            home = fpl_teams.get(f['team_h'], 'Unknown')
            away = fpl_teams.get(f['team_a'], 'Unknown')
            home_short = fpl_teams_short.get(f['team_h'], '')
            away_short = fpl_teams_short.get(f['team_a'], '')
            fixtures_list.append({'home': home, 'away': away})
            
            # Store multiple name variants for matching
            for name in [home, home_short]:
                norm = name.lower().replace(' ', '_').replace("'", "").replace('-', '_')
                teams_playing_variants[norm] = home
            for name in [away, away_short]:
                norm = name.lower().replace(' ', '_').replace("'", "").replace('-', '_')
                teams_playing_variants[norm] = away
        
        if verbose:
            print(f'    Found {len(gw_fixtures)} fixtures, {len(set(teams_playing_variants.values()))} teams')
        
        # Build team name mapping from our data to FPL names
        def normalize(name):
            return name.lower().replace(' ', '_').replace("'", "").replace('-', '_').replace('.', '')
        
        # Common name mappings
        TEAM_ALIASES = {
            'manchester_city': 'Man City', 'manchester_united': 'Man Utd', 'man_city': 'Man City', 'man_utd': 'Man Utd',
            'tottenham_hotspur': 'Spurs', 'tottenham': 'Spurs', 'spurs': 'Spurs',
            'newcastle_united': 'Newcastle', 'newcastle': 'Newcastle',
            'west_ham_united': 'West Ham', 'west_ham': 'West Ham',
            'nottingham_forest': "Nott'm Forest", 'nottm_forest': "Nott'm Forest", 'nott_forest': "Nott'm Forest",
            'wolverhampton_wanderers': 'Wolves', 'wolves': 'Wolves',
            'brighton_and_hove_albion': 'Brighton', 'brighton': 'Brighton',
            'crystal_palace': 'Crystal Palace', 'leicester_city': 'Leicester',
            'aston_villa': 'Aston Villa', 'ipswich_town': 'Ipswich', 'ipswich': 'Ipswich',
            'arsenal': 'Arsenal', 'chelsea': 'Chelsea', 'liverpool': 'Liverpool',
            'everton': 'Everton', 'fulham': 'Fulham', 'brentford': 'Brentford',
            'bournemouth': 'Bournemouth', 'southampton': 'Southampton',
        }
        
        def match_team_to_fpl(team_name):
            """Match our team name to FPL team name."""
            norm = normalize(team_name)
            
            # Direct match in variants
            if norm in teams_playing_variants:
                return teams_playing_variants[norm]
            
            # Check aliases
            if norm in TEAM_ALIASES:
                alias = TEAM_ALIASES[norm]
                alias_norm = normalize(alias)
                if alias_norm in teams_playing_variants:
                    return teams_playing_variants[alias_norm]
                # Direct name match
                for fpl_name in set(teams_playing_variants.values()):
                    if alias.lower() == fpl_name.lower():
                        return fpl_name
            
            # Partial match
            for variant, fpl_name in teams_playing_variants.items():
                if norm in variant or variant in norm:
                    return fpl_name
                # Also check if any word matches
                for word in norm.split('_'):
                    if len(word) > 3 and word in variant:
                        return fpl_name
            
            return None
        
        # Get latest player data from prior gameweeks only
        current_season_prior = prior_data[prior_data['season'] == target_season]
        latest_player_data = current_season_prior.sort_values(['player_id', 'gameweek']).groupby('player_id').last().reset_index()
        
        if verbose:
            print(f'    Found {len(latest_player_data)} players in prior GW data')
        
        # Build upcoming rows
        upcoming_rows = []
        teams_matched = set()
        teams_unmatched = set()
        
        for _, player_row in latest_player_data.iterrows():
            player_team = str(player_row.get('team', ''))
            fpl_team = match_team_to_fpl(player_team)
            
            if fpl_team is None:
                teams_unmatched.add(player_team)
                continue
            
            teams_matched.add(fpl_team)
            
            # Find opponent from fixtures
            opponent = None
            is_home = 0
            for fix in fixtures_list:
                if fpl_team == fix['home']:
                    opponent = fix['away']
                    is_home = 1
                    break
                elif fpl_team == fix['away']:
                    opponent = fix['home']
                    is_home = 0
                    break
            
            if opponent:
                upcoming_rows.append({
                    'player_name': player_row['player_name'],
                    'team': player_team,  # Keep original for feature matching
                    'team_display': fpl_team,
                    'opponent': opponent,
                    'position': player_row.get('position', 'MID'),
                    'is_home': is_home,
                    'gameweek': target_gw,
                    'season': target_season,
                    'player_id': player_row['player_id']
                })
        
        if verbose:
            print(f'    Teams matched: {len(teams_matched)}/{len(set(teams_playing_variants.values()))}')
            if teams_unmatched:
                print(f'    Teams not matched: {list(teams_unmatched)[:5]}...')
        
        upcoming_df = pd.DataFrame(upcoming_rows)
        
        if len(upcoming_df) == 0:
            print('    WARNING: No players matched to fixtures!')
            return pd.DataFrame()
        
        # Get features from PRIOR data only
        latest_features = prior_data.sort_values(['player_id', 'season', 'gameweek']).groupby('player_id').last().reset_index()
        
        # Merge historical features
        feature_cols = [c for c in latest_features.columns if 'roll' in c or 'last' in c or 'per90' in c or c in [
            'is_defender', 'is_midfielder', 'is_forward', 'is_goalkeeper', 'starter_score', 'full_90_rate'
        ]]
        feature_cols = ['player_id'] + feature_cols
        feature_cols = [c for c in feature_cols if c in latest_features.columns]
        
        test_df = upcoming_df.merge(latest_features[feature_cols], on='player_id', how='left')
        
        # Fill missing with sensible defaults
        for col in test_df.columns:
            if 'roll' in col or 'per90' in col:
                test_df[col] = test_df[col].fillna(0)
        
        test_df['starter_score'] = test_df['starter_score'].fillna(0.5)
        test_df['full_90_rate'] = test_df['full_90_rate'].fillna(0.5)
        test_df['is_home'] = test_df['is_home'].fillna(0).astype(int)
        
        for col in ['is_defender', 'is_midfielder', 'is_forward', 'is_goalkeeper']:
            if col not in test_df.columns:
                test_df[col] = 0
        
        if verbose:
            print(f'    Final test set: {len(test_df)} players with features')
        
        return test_df
    
    def get_top_players(self, n: int = 30) -> pd.DataFrame:
        """Get top N players by expected points."""
        cols = ['player_name', 'team', 'position', 'opponent', 'pred_minutes',
                'pred_exp_goals', 'pred_exp_assists', 'pred_cs_prob', 'pred_bonus', 'exp_total_pts']
        return self.test_df.nlargest(n, 'exp_total_pts')[cols]


def run_pipeline(target_gw: int, target_season: str = '2025-26', data_dir: str = '../data', verbose: bool = True) -> pd.DataFrame:
    """Convenience function to run the pipeline."""
    pipeline = FPLPredictionPipeline(data_dir)
    return pipeline.run(target_gw, target_season, verbose)


if __name__ == '__main__':
    # Run for GW15
    predictions = run_pipeline(target_gw=15, target_season='2025-26')
    print("\nTop 10 Players:")
    print(predictions.nlargest(10, 'exp_total_pts')[['player_name', 'team', 'exp_total_pts']])

