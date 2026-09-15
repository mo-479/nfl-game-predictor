# NFL Game Outcome Prediction Model

Predicts the winner of NFL games using team form, scoring trends, rest days,
and matchup context, built on real historical results (2003–present) from
the open-source [nflverse](https://github.com/nflverse/nfldata) project.

## How it works

1. **`data_loader.py`** - downloads and caches `games.csv` from nflverse
   (one row per NFL game since 1999, with final scores, rest days,
   divisional-game flag, Vegas lines, and weather).

2. **`features.py`** - reshapes the data into a per-team game log and builds
   **leakage-safe** rolling features: trailing 5-game win %, point
   differential, points scored/allowed, and season-to-date win %. Every
   feature for a given game only uses that team's *prior* games (via
   `shift(1)` before rolling), so the model never sees information from the
   game it's predicting. Final features are home-vs-away differentials
   (e.g., `point_diff_diff`, `season_win_pct_diff`) plus rest-day advantage
   and divisional-game flag.

3. **`train.py`** - splits the data **chronologically** (trains on earlier
   seasons, tests on the 3 most recent complete seasons — never a random
   split on time-series sports data), then trains and compares:
   - A naive home-field-only baseline
   - Logistic Regression (interpretable baseline)
   - XGBoost, tuned via `GridSearchCV` with `TimeSeriesSplit` cross-validation

   It reports accuracy, log loss, and ROC-AUC on the held-out seasons, and
   saves a feature-importance chart.

## Results (test seasons: 2024–2026, 585 games)

| Model                     | Accuracy | Log Loss | ROC-AUC |
|---------------------------|----------|----------|---------|
| Home-field-only baseline  | 0.542    | 0.691    | 0.500   |
| Logistic Regression       | 0.636    | 0.633    | 0.696   |
| XGBoost (tuned)           | 0.663    | 0.628    | 0.705   |

The tuned XGBoost model beats the naive home-field baseline by **12
percentage points of accuracy** and improves ROC-AUC from 0.50 to 0.70.
The most predictive features are each team's recent point differential and
season-to-date win rate relative to their opponent — recency and margin of
victory matter more than raw win/loss record.

## Usage

```bash
pip install -r requirements.txt
python train.py
```

This downloads the data (cached locally after the first run), builds
features, trains both models, prints the comparison table above, and saves
`feature_importance.png`.

## Try it live: `app.py`

A small Streamlit app (`app.py`) wraps the trained model so anyone can pick
two teams and get a win-probability prediction, based on each team's actual
current rolling form (computed live from real nflverse data — no
hardcoded stats).

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy for free (no server to manage): push this repo to GitHub, then go to
[share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and
point it at this repo with `app.py` as the entry file. It installs
`requirements.txt` automatically and gives you a public URL.

## Notes / possible extensions

- Add QB-level features (starter changes, injuries) - `games.csv` includes
  `home_qb_name` / `away_qb_name`, which aren't used here yet.
- Incorporate `spread_line` (the closing Vegas spread) as a feature or as a
  separate benchmark — beating the market spread is a much higher bar than
  beating a 50/50 baseline.
- Swap the single chronological split for walk-forward validation
  (retrain each season on all prior seasons) to see how performance
  trends over time.
