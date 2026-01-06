"""
Tune a single model using Optuna and save results to disk.

Usage:
    python scripts/tune_single_model.py <model_name> <n_trials> <output_file>
    
Example:
    python scripts/tune_single_model.py goals 100 data/tuning_results/goals_tuned.json

This script is designed to be called via subprocess from a notebook.
When the subprocess exits, ALL memory is released - like a kernel restart.
"""
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.fpl_prediction_pipeline import FPLPredictionPipeline, TuneConfig


def main():
    if len(sys.argv) < 4:
        print("Usage: python tune_single_model.py <model_name> <n_trials> <output_file>")
        print("Models: goals, assists, minutes, defcon, clean_sheet, bonus")
        sys.exit(1)
    
    model_name = sys.argv[1]
    n_trials = int(sys.argv[2])
    output_file = Path(sys.argv[3])
    
    print(f"\n{'='*60}")
    print(f"TUNING {model_name.upper()} MODEL - {n_trials} Optuna trials")
    print(f"{'='*60}\n")
    
    # Initialize pipeline
    pipeline = FPLPredictionPipeline(data_dir=str(PROJECT_ROOT / 'data'))
    
    # Configure tuning
    config = TuneConfig(
        models=[model_name],
        test_size=0.2,
        random_state=42,
        feature_selection='importance',
        feature_selection_k_min=5,
        feature_selection_k_max=25,
        tune_feature_count=True,
        optuna_trials=n_trials,
        optuna_timeout=3000,
    )
    
    # Run tuning
    results = pipeline.tune(config=config, verbose=True)
    
    # Extract metrics
    metrics = results[model_name]
    
    # Prepare output
    output = {
        'model': model_name,
        'mae': float(metrics.mae),
        'rmse': float(metrics.rmse),
        'r2': float(metrics.r2),
        'samples': int(metrics.samples),
        'best_params': metrics.best_params,
        'selected_features': metrics.selected_features,
    }
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"RESULTS SAVED: {output_file}")
    print(f"{'='*60}")
    print(f"  MAE:  {metrics.mae:.4f}")
    print(f"  RMSE: {metrics.rmse:.4f}")
    print(f"  R²:   {metrics.r2:.4f}")
    if metrics.best_params:
        print(f"  Best params: {metrics.best_params}")


if __name__ == '__main__':
    main()

