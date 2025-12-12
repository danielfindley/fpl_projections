# FPL Ultimate Points Projector

A comprehensive dataset engineering pipeline for Fantasy Premier League player points projection.

## Architecture

```
├── data/
│   ├── raw/              # Immutable source data
│   ├── intermediate/     # Cleaned, standardized data
│   └── features/         # Final feature datasets
├── src/
│   ├── collectors/       # Data collection modules
│   ├── processors/       # Data cleaning & transformation
│   ├── features/         # Feature engineering pipelines
│   └── utils/            # Shared utilities
├── notebooks/            # Exploration & analysis
└── config/               # Configuration files
```

## Data Sources

| Source | Data Provided | Update Frequency |
|--------|---------------|------------------|
| FPL API | Player metadata, points, ownership, prices | Live |
| Understat | xG, xA, shot data at player/match level | Post-match |
| FBRef | Advanced stats, detailed match data | Post-match |

## Feature Categories

### Player Rolling Stats (5-week window)
- Minutes per game
- xG, xA per 90
- Goals, assists per game
- Shots, key passes per game
- Bonus points per game

### Team Form (rolling, point-in-time)
- xG for/against per game
- Goals scored/conceded per game
- Clean sheet rate

### Opponent Context
- Opponent defensive form (xGA, goals conceded)
- Opponent offensive form (xG, goals scored)
- Fixture Difficulty Rating (FDR)

### Derived Features
- Clean sheet probability (from xGA trends + betting odds)
- Expected involvement (xGI = xG + xA)
- Form momentum (acceleration of rolling stats)

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Collect raw data
python -m src.collectors.run_all

# Build feature dataset
python -m src.features.build_dataset

# Export for modeling
python -m src.features.export --format parquet
```

## Point-in-Time Correctness

All rolling features are computed using only data available BEFORE the match in question.
This prevents data leakage and ensures the dataset is valid for backtesting.


