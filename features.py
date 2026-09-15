"""
features.py

Turns raw game-level rows into a model-ready feature table.

Key design point: every rolling/team-form feature is computed using only
games that happened BEFORE the game being predicted (via a shift(1) on a
per-team, time-sorted series). This avoids leaking the outcome of the
game we're trying to predict into its own features.
"""

import numpy as np
import pandas as pd

ROLLING_WINDOW = 5  # trailing games used for "recent form" features


def _team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape from one-row-per-game (home vs away) into one-row-per-team-game,
    so we can compute each team's rolling performance over time regardless
    of whether they were home or away in a given week.
    """
    home = games.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "points_for", "away_score": "points_against",
        "home_rest": "rest_days",
    })[["game_id", "season", "week", "gameday", "team", "opponent",
        "points_for", "points_against", "rest_days"]].copy()
    home["is_home"] = 1

    away = games.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "points_for", "home_score": "points_against",
        "away_rest": "rest_days",
    })[["game_id", "season", "week", "gameday", "team", "opponent",
        "points_for", "points_against", "rest_days"]].copy()
    away["is_home"] = 0

    log = pd.concat([home, away], ignore_index=True)
    log["win"] = (log["points_for"] > log["points_against"]).astype(int)
    log["point_diff"] = log["points_for"] - log["points_against"]
    log = log.sort_values(["team", "gameday"]).reset_index(drop=True)
    return log


def _add_rolling_form(log: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    For each team, compute trailing rolling averages using only PRIOR
    games (shift(1) before rolling so the current game is excluded).
    Also adds season-to-date form, which resets each season.
    """
    grouped = log.groupby("team", group_keys=False)

    log["roll_win_pct"] = grouped["win"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    log["roll_point_diff"] = grouped["point_diff"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    log["roll_points_for"] = grouped["points_for"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    log["roll_points_against"] = grouped["points_against"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    season_grouped = log.groupby(["team", "season"], group_keys=False)
    log["season_win_pct"] = season_grouped["win"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # Early-season games have no prior-season rolling history: fill with a
    # neutral prior (0.5 win rate / 0 point differential) rather than NaN.
    fill_values = {
        "roll_win_pct": 0.5, "season_win_pct": 0.5,
        "roll_point_diff": 0.0, "roll_points_for": log["points_for"].mean(),
        "roll_points_against": log["points_against"].mean(),
    }
    log = log.fillna(value=fill_values)
    return log


def build_feature_table(games: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: raw games -> per-team rolling form -> back to one row
    per game with home_* / away_* feature columns plus the label.
    """
    log = _team_game_log(games)
    log = _add_rolling_form(log)

    feature_cols = ["roll_win_pct", "roll_point_diff", "roll_points_for",
                     "roll_points_against", "season_win_pct", "rest_days"]

    home_feats = log[log["is_home"] == 1][["game_id", "team"] + feature_cols]
    home_feats = home_feats.rename(
        columns={c: f"home_{c}" for c in feature_cols} | {"team": "home_team"}
    )

    away_feats = log[log["is_home"] == 0][["game_id", "team"] + feature_cols]
    away_feats = away_feats.rename(
        columns={c: f"away_{c}" for c in feature_cols} | {"team": "away_team"}
    )

    df = games.merge(home_feats, on=["game_id", "home_team"])
    df = df.merge(away_feats, on=["game_id", "away_team"])

    # Differential features tend to be the most predictive: how much
    # better has the home team been playing than the away team lately?
    df["win_pct_diff"] = df["home_roll_win_pct"] - df["away_roll_win_pct"]
    df["point_diff_diff"] = df["home_roll_point_diff"] - df["away_roll_point_diff"]
    df["season_win_pct_diff"] = df["home_season_win_pct"] - df["away_season_win_pct"]
    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
    df["div_game"] = df["div_game"].fillna(0).astype(int)

    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    return df


FEATURE_COLUMNS = [
    "home_roll_win_pct", "away_roll_win_pct", "win_pct_diff",
    "home_roll_point_diff", "away_roll_point_diff", "point_diff_diff",
    "home_season_win_pct", "away_season_win_pct", "season_win_pct_diff",
    "home_roll_points_for", "home_roll_points_against",
    "away_roll_points_for", "away_roll_points_against",
    "rest_advantage", "div_game",
]
