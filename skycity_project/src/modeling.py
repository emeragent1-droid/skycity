"""
Model development for predicting Total Monthly Net Profit and Net Profit per Order.

Trains and compares:
 - Linear Regression (interpretability baseline)
 - Random Forest Regressor
 - Gradient Boosting Regressor
 - XGBoost Regressor

Evaluates with RMSE, R^2, MAE. Saves the best model + feature list + metrics to /models.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from feature_engineering import FEATURE_COLUMNS

DATA_PATH = "data/skycity_features.csv"
MODELS_DIR = "models"


def get_feature_matrix(df: pd.DataFrame):
    cat_cols = [c for c in df.columns if c.startswith(("CuisineType_", "Segment_", "Subregion_"))]
    feature_cols = FEATURE_COLUMNS + cat_cols
    X = df[feature_cols].astype(float)
    return X, feature_cols


def evaluate(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_all(target: str = "TotalNetProfit"):
    df = pd.read_csv(DATA_PATH)
    X, feature_cols = get_feature_matrix(df)
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        metrics["CV_R2_mean"] = float(cv_scores.mean())
        metrics["CV_R2_std"] = float(cv_scores.std())
        results[name] = metrics
        fitted[name] = model
        print(f"{name:18s} RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  "
              f"R2={metrics['R2']:.4f}  CV_R2={metrics['CV_R2_mean']:.4f}")

    best_name = max(results, key=lambda k: results[k]["R2"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # Feature importance (tree models) or coefficients (linear)
    importance = None
    if best_name == "LinearRegression":
        coefs = best_model.named_steps["model"].coef_
        importance = dict(zip(feature_cols, coefs.tolist()))
    elif hasattr(best_model, "feature_importances_"):
        importance = dict(zip(feature_cols, best_model.feature_importances_.tolist()))

    joblib.dump(best_model, f"{MODELS_DIR}/best_model_{target}.pkl")
    joblib.dump(feature_cols, f"{MODELS_DIR}/feature_columns_{target}.pkl")

    with open(f"{MODELS_DIR}/metrics_{target}.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    if importance:
        imp_series = pd.Series(importance).sort_values(key=abs, ascending=False)
        imp_series.to_csv(f"{MODELS_DIR}/feature_importance_{target}.csv", header=["importance"])

    return results, best_name, best_model, feature_cols


if __name__ == "__main__":
    for target in ["TotalNetProfit", "NetProfitPerOrder"]:
        print(f"\n{'='*60}\nTraining models for target: {target}\n{'='*60}")
        train_all(target)
