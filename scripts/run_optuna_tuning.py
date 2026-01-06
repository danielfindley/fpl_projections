"""
Standalone Optuna Tuning Script

Run this from command line instead of Jupyter to avoid memory issues:
    python scripts/run_optuna_tuning.py --trials 250 --model goals

This script runs Optuna optimization in isolated batches with proper memory cleanup.
Results are saved to data/tuning_results/ and can be loaded back into the notebook.
"""

import argparse
import json
import gc
import os
import sys
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import after path setup
import optuna
from optuna.samplers import RandomSampler

from pipelines.fpl_prediction_pipeline import (
    FPLPredictionPipeline, TuneConfig, load_raw_data, compute_rolling_features
)
from models.minutes_model import MinutesModel
from models.goals_model import GoalsModel
from models.assists_model import AssistsModel
from models.defcon_model import DefconModel
from models.bonus_model_mc import BonusModelMC as BonusModel


MODEL_CLASSES = {
    'goals': GoalsModel,
    'assists': AssistsModel,
    'minutes': MinutesModel,
    'defcon': DefconModel,
    'bonus': BonusModel,
}


def run_single_batch(args):
    """Run a single batch of trials in isolation. Called via multiprocessing."""
    (model_name, model_class, train_data, val_data, y_val, search_space,
     tune_n_features, min_features, max_features, original_features,
     batch_trials, batch_seed, tuning_max_est) = args
    
    # Suppress optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    best_value = float('inf')
    best_params = None
    
    def objective(trial):
        nonlocal best_value, best_params
        
        params = {'random_state': 42}
        
        for param_name, param_config in search_space.items():
            param_type = param_config[0]
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
            elif param_type == 'float_log':
                params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=True)
        
        # Cap estimators
        if 'n_estimators' in params:
            params['n_estimators'] = min(params['n_estimators'], tuning_max_est)
        
        if tune_n_features:
            n_features = trial.suggest_int('n_features', min_features, max_features)
            selected_features = original_features[:n_features]
            
            class TempModel(model_class):
                FEATURES = selected_features
            
            model = TempModel(**params)
        else:
            model = model_class(**params)
        
        model.fit(train_data, verbose=False)
        
        if model_name == 'minutes':
            y_pred = model.predict(val_data)
        elif model_name == 'bonus':
            y_pred = model.predict(val_data)
        else:
            y_pred = model.predict_per90(val_data)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        # Track best
        if rmse < best_value:
            best_value = rmse
            best_params = trial.params.copy()
        
        # Cleanup
        del model
        gc.collect()
        
        return rmse
    
    # Create study with RandomSampler (light memory)
    sampler = RandomSampler(seed=batch_seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=batch_trials, gc_after_trial=True)
    
    return {
        'best_value': best_value,
        'best_params': best_params,
    }


def run_optimization(model_name: str, total_trials: int, batch_size: int = 25,
                     subsample: float = 0.3, max_est: int = 100):
    """Run optimization for a single model with proper memory isolation."""
    
    print(f"\n{'='*60}")
    print(f"TUNING {model_name.upper()} MODEL")
    print(f"{'='*60}")
    print(f"Total trials: {total_trials}")
    print(f"Batch size: {batch_size}")
    print(f"Data subsample: {subsample*100:.0f}%")
    print(f"Max estimators: {max_est}")
    
    # Load and prepare data
    print("\nLoading data...")
    data_dir = PROJECT_ROOT / 'data'
    pipeline = FPLPredictionPipeline(data_dir=str(data_dir))
    
    # Load raw data and compute features
    df = load_raw_data(str(data_dir))
    df = compute_rolling_features(df)
    
    # Get model info
    model_class = MODEL_CLASSES[model_name]
    temp_model = model_class()
    original_features = temp_model.FEATURES.copy()
    target_col = temp_model.TARGET
    del temp_model
    
    # Split and subsample data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    if subsample < 1.0:
        n_train = int(len(train_df) * subsample)
        n_val = int(len(val_df) * subsample)
        train_df = train_df.sample(n=n_train, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(n=n_val, random_state=42).reset_index(drop=True)
        print(f"Subsampled: {len(train_df)} train, {len(val_df)} val samples")
    
    # Pre-compute validation target
    if model_name == 'minutes':
        y_val = val_df['minutes'].values.copy()
    elif model_name == 'bonus':
        y_val = val_df['bonus'].values.copy()
    else:
        y_val = (val_df[model_name] / np.maximum(val_df['minutes'] / 90, 0.01)).values.copy()
    
    # Get search space
    search_space = TuneConfig.optuna_search_space().get(model_name, {})
    
    # Feature tuning settings
    tune_n_features = True
    min_features = 5
    max_features = min(25, len(original_features))
    
    # Track global best
    global_best_value = float('inf')
    global_best_params = None
    
    # Run batches using multiprocessing for memory isolation
    n_batches = (total_trials + batch_size - 1) // batch_size
    trials_completed = 0
    
    print(f"\nRunning {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        trials_this_batch = min(batch_size, total_trials - trials_completed)
        batch_seed = 42 + batch_idx
        
        print(f"\n  Batch {batch_idx + 1}/{n_batches}: trials {trials_completed + 1}-{trials_completed + trials_this_batch}")
        
        # Prepare args for worker
        args = (
            model_name, model_class, train_df, val_df, y_val, search_space,
            tune_n_features, min_features, max_features, original_features,
            trials_this_batch, batch_seed, max_est
        )
        
        # Run in subprocess for memory isolation
        # Using spawn context for Windows compatibility
        ctx = mp.get_context('spawn')
        with ctx.Pool(1) as pool:
            result = pool.apply(run_single_batch, (args,))
        
        # Check if better
        if result['best_value'] < global_best_value:
            global_best_value = result['best_value']
            global_best_params = result['best_params']
            print(f"    New best RMSE: {global_best_value:.4f}")
        
        trials_completed += trials_this_batch
        
        # Force cleanup
        gc.collect()
    
    # Extract n_features from params
    best_n_features = None
    if global_best_params and 'n_features' in global_best_params:
        best_n_features = global_best_params.pop('n_features')
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS FOR {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Best RMSE: {global_best_value:.4f}")
    print(f"Best params: {global_best_params}")
    if best_n_features:
        print(f"Best n_features: {best_n_features}")
    
    # Save results
    results_dir = data_dir / 'tuning_results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'{model_name}_tuning_{timestamp}.json'
    
    results = {
        'model': model_name,
        'best_rmse': global_best_value,
        'best_params': global_best_params,
        'best_n_features': best_n_features,
        'total_trials': total_trials,
        'timestamp': timestamp,
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run Optuna hyperparameter tuning')
    parser.add_argument('--model', type=str, required=True,
                        choices=['goals', 'assists', 'minutes', 'defcon', 'bonus', 'all'],
                        help='Model to tune')
    parser.add_argument('--trials', type=int, default=250,
                        help='Total number of trials (default: 250)')
    parser.add_argument('--batch-size', type=int, default=25,
                        help='Trials per batch (default: 25)')
    parser.add_argument('--subsample', type=float, default=0.3,
                        help='Fraction of data to use (default: 0.3)')
    parser.add_argument('--max-estimators', type=int, default=100,
                        help='Max n_estimators during tuning (default: 100)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("OPTUNA HYPERPARAMETER TUNING (Subprocess Mode)")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.model == 'all':
        models = ['goals', 'assists', 'minutes', 'defcon', 'bonus']
    else:
        models = [args.model]
    
    all_results = {}
    for model_name in models:
        results = run_optimization(
            model_name=model_name,
            total_trials=args.trials,
            batch_size=args.batch_size,
            subsample=args.subsample,
            max_est=args.max_estimators
        )
        all_results[model_name] = results
    
    print("\n" + "="*60)
    print("ALL TUNING COMPLETE")
    print("="*60)
    for name, res in all_results.items():
        print(f"  {name}: RMSE={res['best_rmse']:.4f}")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()




