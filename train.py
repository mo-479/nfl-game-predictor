"""
train.py

End-to-end training pipeline for the NFL game outcome prediction model.

1. Load historical games (data_loader.py)
2. Engineer leakage-safe rolling-form features (features.py)
3. Chronological train/test split (train on earlier seasons, test on the
   most recent seasons -- never split randomly on time-series sports data)
4. Train a Logistic Regression baseline and a tuned XGBoost classifier
5. Report accuracy, log loss, and ROC-AUC on the held-out test seasons
6. Plot feature importance from the XGBoost model

Run: python train.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

from data_loader import load_games
from features import build_feature_table, FEATURE_COLUMNS

TEST_SEASONS_HELD_OUT = 3  # most recent N complete seasons used as the test set
RANDOM_STATE = 42


def chronological_split(df: pd.DataFrame, n_test_seasons: int = TEST_SEASONS_HELD_OUT):
    seasons = sorted(df["season"].unique())
    test_seasons = set(seasons[-n_test_seasons:])
    train_df = df[~df["season"].isin(test_seasons)].copy()
    test_df = df[df["season"].isin(test_seasons)].copy()
    return train_df, test_df, sorted(test_seasons)


def train_baseline(X_train, y_train):
    """Logistic Regression on standardized features -- simple, interpretable
    baseline that any more complex model should have to beat."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_xgboost(X_train, y_train):
    """XGBoost classifier tuned with a small grid search, using
    TimeSeriesSplit so validation folds never precede training folds."""
    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [150, 300],
        "subsample": [0.8, 1.0],
    }

    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    cv = TimeSeriesSplit(n_splits=4)
    search = GridSearchCV(
        base_model, param_grid, scoring="neg_log_loss",
        cv=cv, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"Best XGBoost params: {search.best_params_}")
    return search.best_estimator_


def evaluate(name, y_true, y_pred_proba):
    y_pred = (y_pred_proba >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    print(f"{name:22s}  accuracy={acc:.3f}   log_loss={ll:.3f}   roc_auc={auc:.3f}")
    return {"model": name, "accuracy": acc, "log_loss": ll, "roc_auc": auc}


def plot_feature_importance(model, feature_names, out_path="feature_importance.png"):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#1F3864",
    )
    plt.xlabel("XGBoost Feature Importance")
    plt.title("NFL Game Outcome Prediction — Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved feature importance plot to {out_path}")


def main():
    print("Loading games...")
    games = load_games(min_season=2003)

    print("Building features...")
    df = build_feature_table(games)

    train_df, test_df, test_seasons = chronological_split(df)
    print(f"Train: {len(train_df):,} games ({train_df['season'].min()}"
          f"-{train_df['season'].max()})")
    print(f"Test:  {len(test_df):,} games (seasons {test_seasons})")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["home_win"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["home_win"]

    results = []

    # --- Naive baseline: always predict the home team wins ---
    home_field_pred = np.full(len(y_test), y_train.mean())
    results.append(evaluate("Home-field-only baseline", y_test, home_field_pred))

    # --- Logistic Regression baseline ---
    lr_model, scaler = train_baseline(X_train, y_train)
    lr_proba = lr_model.predict_proba(scaler.transform(X_test))[:, 1]
    results.append(evaluate("Logistic Regression", y_test, lr_proba))

    # --- XGBoost (tuned) ---
    print("\nTuning XGBoost (grid search over max_depth / lr / n_estimators)...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost (tuned)", y_test, xgb_proba))

    print("\nSummary:")
    print(pd.DataFrame(results).to_string(index=False))

    plot_feature_importance(xgb_model, FEATURE_COLUMNS)

    return results


if __name__ == "__main__":
    main()
