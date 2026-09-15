"""
data_loader.py

Downloads and caches historical NFL game data (1999-present) from the
public nflverse project, an open-source community data repository that
mirrors official NFL schedule and results data.

Source: https://github.com/nflverse/nfldata (games.csv)

Each row is one game with home/away teams, final scores, rest days,
divisional-game flag, Vegas lines, weather, and more.
"""

from pathlib import Path
import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
CACHE_PATH = Path(__file__).parent / "data" / "games_raw.csv"


def load_games(min_season: int = 2003, use_cache: bool = True) -> pd.DataFrame:
    """
    Load historical NFL game results.

    Parameters
    ----------
    min_season : int
        Earliest season to include (earlier seasons have sparser columns
        like rest days / lines, so we default to 2003+).
    use_cache : bool
        If True and a local cache exists, read from disk instead of
        re-downloading.

    Returns
    -------
    pd.DataFrame
        One row per game, sorted by season/week/game date.
    """
    if use_cache and CACHE_PATH.exists():
        df = pd.read_csv(CACHE_PATH)
    else:
        df = pd.read_csv(GAMES_URL)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)

    df = df[df["season"] >= min_season].copy()

    # Only keep games that have been played (drop future/unplayed schedule rows)
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    df["gameday"] = pd.to_datetime(df["gameday"])
    df = df.sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    games = load_games()
    print(f"Loaded {len(games):,} games from {games['season'].min()}"
          f" to {games['season'].max()}")
    print(games[["season", "week", "home_team", "away_team",
                 "home_score", "away_score"]].tail())
