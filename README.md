# FPL Ultimate Points Projector

A comprehensive machine learning pipeline for Fantasy Premier League player points projection, featuring 6 specialized XGBoost models with Monte Carlo simulation for bonus points.

## 📊 Latest Predictions

**[View the latest gameweek predictions →](notebooks/latest_predictions.ipynb)**

The predictions notebook automatically loads the most recent forecast from `data/predictions/` and displays:
- 🏆 Top 15 overall picks
- 🧤 Top goalkeepers
- 🛡️ Top defenders  
- 🎯 Top midfielders
- ⚡ Top forwards
- 💎 Differential picks

---

## Architecture

```
├── data/
│   ├── raw/              # Immutable source data (FBRef)
│   ├── raw_html/         # Scraped match report HTML files
│   ├── intermediate/     # Cleaned, standardized data
│   ├── features/         # Final feature datasets (parquet)
│   ├── name_matching/    # FPL ↔ FBRef player name mappings
│   ├── tuning_results/   # Optuna hyperparameter tuning outputs
│   └── predictions/      # Model predictions output
├── models/               # ML model classes
├── pipelines/            # End-to-end prediction pipeline
├── scripts/              # Data collection & utility scripts
├── notebooks/            # Exploration, tuning & analysis
└── requirements.txt
```

## Data Sources

| Source | Data Provided | Update Frequency |
|--------|---------------|------------------|
| FPL API | Player metadata, availability status, bonus points, prices | Live |
| FBRef | Advanced stats, detailed match data (xG, xA, shots, tackles, etc.) | Post-match |

## Feature Categories

### Player Rolling Stats (1/3/5/10 game windows)
- Minutes per game, starts, full 90s
- xG, xA per 90
- Goals, assists per game
- Shots, key passes, SCA, GCA per game
- Tackles, interceptions, clearances per game

### Lifetime Player Profile
- Career goals/assists per 90
- Career xG/xA per 90
- Career defensive actions per 90
- Total career minutes (experience indicator)

### Team Form (rolling, point-in-time)
- xG for/against per game
- Goals scored/conceded per game
- Clean sheet rate (home/away separated)

### Opponent Context
- Opponent defensive form (xGA, goals conceded)
- Opponent offensive form (xG, goals scored)
- Relative strength ratios

### FPL API Features
- Availability status (injured, doubtful, suspended)
- Chance of playing next round (0-100%)

---

## Models

### 1. Minutes Model (`MinutesModel`)

**Purpose:** Predicts expected minutes for players assuming they will play.

**Architecture:** XGBoost Regressor with sample weighting

**Key Insight:** Among players who play, minutes are heavily bimodal:
- ~50% play exactly 90 minutes (full game)
- ~65% play 60+ minutes (starters)
- ~35% are substitutes playing 1-30 minutes

**Key Features:**
- Recent minutes history (roll3, roll5, roll10)
- Starter score, full 90 rate
- Lifetime minutes per appearance
- FPL availability status (hard filter for injured/unavailable)

**Output:** Expected minutes (0-90)

---

### 2. Goals Model (`GoalsModel`)

**Purpose:** Predicts goals per 90 minutes rate against a specific opponent.

**Architecture:** XGBoost Regressor

**Calculation:** `expected_goals = goals_per90 × (pred_minutes / 90)`

**Key Features:**
- Player xG per 90 (roll1, roll3, roll5, roll10)
- Shots, SCA, GCA per 90
- Lifetime goals and xG per 90
- Team attacking strength (team_xg, team_goals)
- Opponent defensive weakness (opp_goals_conceded, opp_xga)
- Home/away indicator

**Output:** Goals per 90 rate → scaled by predicted minutes

---

### 3. Assists Model (`AssistsModel`)

**Purpose:** Predicts assists per 90 minutes rate against a specific opponent.

**Architecture:** XGBoost Regressor

**Calculation:** `expected_assists = assists_per90 × (pred_minutes / 90)`

**Key Features:**
- Player xAG per 90 (rolling windows)
- Key passes, SCA, GCA per 90
- Lifetime assists and xAG per 90
- Team attacking context
- Opponent defensive weakness

**Output:** Assists per 90 rate → scaled by predicted minutes

---

### 4. Clean Sheet Model (`CleanSheetModel`)

**Purpose:** Predicts probability of a team keeping a clean sheet.

**Architecture:** XGBoost Classifier (binary)

**Key Insight:** Team-level prediction that applies to all defenders/goalkeepers.

**Key Features:**
- Team defensive history (goals conceded, xGA - roll5, roll10, roll30)
- Clean sheet rate (home/away separated)
- Opponent attacking strength (goals scored, xG)
- Relative strength ratios (xGA/xG ratio, defensive advantage)
- Team/opponent identity encodings

**Output:** Clean sheet probability (0.0-1.0)

**FPL Points:**
- GK/DEF: 4 pts for clean sheet (60+ mins)
- MID: 1 pt for clean sheet (60+ mins)

---

### 5. Defcon Model (`DefconModel`)

**Purpose:** Predicts defensive contribution for the "+2 FPL Points" threshold.

**Architecture:** XGBoost Regressor + Poisson Probability

**What is Defcon?**
- **Defenders:** CBIT = Clearances + Blocks + Interceptions + Tackles (threshold: 10+)
- **Midfielders:** CBIRT = CBIT + Recoveries (threshold: 12+)
- Hitting threshold awards **+2 FPL points**

**How It Works:**
1. XGBoost predicts expected defcon (λ) based on player history and opponent
2. Since defcon is a count variable (sum of discrete defensive actions), we model it as **Poisson(λ)**
3. Probability of threshold = **P(X ≥ threshold) = 1 - Poisson.CDF(threshold - 1, λ)**

**Key Features:**
- Rolling defcon per 90 (roll5, roll10)
- Component stats (tackles, interceptions, clearances, blocks, recoveries)
- Lifetime defensive stats per 90
- Historical threshold hit rate
- Opponent attacking strength (more attacks = more defensive actions)

**Output:** Poisson probability of hitting threshold (0.0-1.0)

---

### 6. Bonus Model (`BonusModelMC`)

**Purpose:** Predicts expected bonus points (0-3) using Monte Carlo simulation.

**Architecture:** Two-stage model
1. **BaselineBPSModel** (XGBoost) - Predicts BPS from "boring" stats
2. **Monte Carlo Simulation** - Samples goals/assists/CS and ranks players

**Key Insight:** Bonus points are a **ranking competition** within each match. The top 3 BPS scorers get 3, 2, 1 points. This can't be predicted directly—it depends on what OTHER players in the match do.

**How It Works:**
1. Predict baseline BPS (from passes, tackles, recoveries, etc.)
2. For each simulation (n=1000):
   - Sample goals from Poisson(expected_goals)
   - Sample assists from Poisson(expected_assists)
   - Sample clean sheets from Bernoulli(cs_probability)
   - Calculate total BPS = baseline + goal_bps + assist_bps + cs_bps
   - Rank players within each match
   - Award 3/2/1 points to top 3
3. Average bonus across all simulations

**Key Features (Baseline BPS):**
- Passing stats (touches, passes, progressive passes)
- Defensive stats (tackles, interceptions, clearances)
- Shot creating actions (SCA, GCA)
- Yellow/red card history (negative BPS)
- Season indicators (BPS rules change yearly)
- Recent scoring form (goals/assists per 90)

**Output:** Expected bonus points (0.0-3.0)

---

## Expected Points Calculation

Final FPL points projection combines all model outputs:

```
expected_points = (
    appearance_points(pred_minutes) +
    pred_goals × goal_points[position] +
    pred_assists × 3 +
    pred_cs_prob × cs_points[position] +
    pred_defcon_prob × 2 +
    pred_bonus
)
```

| Position | Goal Pts | CS Pts | Defcon Pts | Appearance (60+) |
|----------|----------|--------|------------|------------------|
| GK       | 6        | 4      | 2          | 2                |
| DEF      | 6        | 4      | 2          | 2                |
| MID      | 5        | 1      | 2          | 2                |
| FWD      | 4        | 0      | 2          | 2                |

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Scrape FBRef data (historical)
python scripts/scrape_all_seasons.py

# Scrape current gameweek
python scripts/scrape_gameweek.py --gameweek 21

# Run predictions
# See notebooks/tune_models.ipynb for full pipeline
```

## Hyperparameter Tuning

Models are tuned using Optuna with configurable:
- Number of trials
- Cross-validation folds
- Feature selection (recursive elimination)
- Search space (learning rate, max depth, etc.)

Tuned parameters are saved to `data/tuning_results/{model}_tuned.json`.

---

## Point-in-Time Correctness

All rolling features are computed using only data available BEFORE the match in question. This prevents data leakage and ensures the dataset is valid for backtesting.

## Data Coverage

- **Seasons:** 2017-18 through 2025-26 (9 seasons)
- **Matches:** ~3,400+ Premier League matches
- **Player-match records:** ~170,000+
