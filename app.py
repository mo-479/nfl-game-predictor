"""
app.py

Interactive web app for the NFL game outcome predictor.

Lets anyone pick a home team and an away team, and get a win-probability
prediction from the trained XGBoost model, based on each team's actual
current rolling form (computed live from real nflverse data).

Run locally:
    streamlit run app.py

Deploy for free:
    https://share.streamlit.io (Streamlit Community Cloud) -- point it at
    this repo, set app.py as the entry point, and it installs
    requirements.txt automatically.
"""

import pandas as pd
import streamlit as st
import xgboost as xgb

from data_loader import load_games
from features import (
    FEATURE_COLUMNS,
    ROLLING_WINDOW,
    _add_rolling_form,
    _team_game_log,
    build_feature_table,
)

st.set_page_config(page_title="NFL Game Predictor", page_icon="🏈", layout="centered")

BEST_PARAMS = dict(max_depth=3, learning_rate=0.01, n_estimators=300, subsample=0.8)


@st.cache_data(show_spinner="Downloading historical NFL data...")
def get_data():
    games = load_games(min_season=2010)
    feat_df = build_feature_table(games)
    return games, feat_df


@st.cache_resource(show_spinner="Training model...")
def get_model(feat_df: pd.DataFrame):
    X, y = feat_df[FEATURE_COLUMNS], feat_df["home_win"]
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        **BEST_PARAMS,
    )
    model.fit(X, y)
    return model


@st.cache_data(show_spinner="Computing current team form...")
def get_team_snapshot(games: pd.DataFrame) -> pd.DataFrame:
    """Each team's rolling form as of their most recently played game --
    i.e. what we'd know walking into their NEXT game. Drops relocated /
    defunct abbreviations (e.g. STL, SD, OAK) whose most recent game is
    from several seasons ago, keeping only currently active franchises."""
    log = _team_game_log(games)
    log = _add_rolling_form(log, window=ROLLING_WINDOW)
    snapshot = log.sort_values("gameday").groupby("team").tail(1).set_index("team")
    current_season = games["season"].max()
    return snapshot[snapshot["season"] >= current_season - 1]


def build_matchup_row(home, away, rest_advantage: int, div_game: bool) -> pd.DataFrame:
    return pd.DataFrame([{
        "home_roll_win_pct": home["roll_win_pct"],
        "away_roll_win_pct": away["roll_win_pct"],
        "win_pct_diff": home["roll_win_pct"] - away["roll_win_pct"],
        "home_roll_point_diff": home["roll_point_diff"],
        "away_roll_point_diff": away["roll_point_diff"],
        "point_diff_diff": home["roll_point_diff"] - away["roll_point_diff"],
        "home_season_win_pct": home["season_win_pct"],
        "away_season_win_pct": away["season_win_pct"],
        "season_win_pct_diff": home["season_win_pct"] - away["season_win_pct"],
        "home_roll_points_for": home["roll_points_for"],
        "home_roll_points_against": home["roll_points_against"],
        "away_roll_points_for": away["roll_points_for"],
        "away_roll_points_against": away["roll_points_against"],
        "rest_advantage": rest_advantage,
        "div_game": int(div_game),
    }])[FEATURE_COLUMNS]


# ---- App ----

st.title("🏈 NFL Game Outcome Predictor")
st.caption(
    "XGBoost model trained on real NFL results since 2010 (source: nflverse). "
    "Pick two teams to see a win-probability prediction based on each team's "
    f"trailing {ROLLING_WINDOW}-game form."
)

games, feat_df = get_data()
model = get_model(feat_df)
snapshot = get_team_snapshot(games)
teams = sorted(snapshot.index.unique())

col1, col2 = st.columns(2)
home_team = col1.selectbox("Home team", teams, index=0)
away_team = col2.selectbox("Away team", teams, index=1)

with st.expander("Matchup context (optional)"):
    rest_advantage = st.slider(
        "Home rest-day advantage (home rest days minus away)", -7, 7, 0
    )
    div_game = st.checkbox("Divisional matchup")

if home_team == away_team:
    st.warning("Pick two different teams.")
else:
    home, away = snapshot.loc[home_team], snapshot.loc[away_team]
    row = build_matchup_row(home, away, rest_advantage, div_game)
    prob_home = float(model.predict_proba(row)[0, 1])

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric(f"{home_team} (home)", f"{prob_home:.0%}")
    c2.metric(f"{away_team} (away)", f"{1 - prob_home:.0%}")
    st.progress(prob_home)

    with st.expander("Team form behind this prediction"):
        form = pd.DataFrame({
            home_team: [f"{home['roll_win_pct']:.0%}", f"{home['roll_point_diff']:+.1f}",
                        f"{home['season_win_pct']:.0%}"],
            away_team: [f"{away['roll_win_pct']:.0%}", f"{away['roll_point_diff']:+.1f}",
                        f"{away['season_win_pct']:.0%}"],
        }, index=[f"Last {ROLLING_WINDOW}-game win %", f"Last {ROLLING_WINDOW}-game point diff",
                  "Season-to-date win %"])
        st.table(form)

st.divider()
st.caption(
    "Model: XGBoost, ~66% accuracy / 0.70 ROC-AUC on 2024–2026 held-out games "
    "(vs. a 54% home-field-only baseline). For portfolio/demo purposes only — not betting advice."
)
