"""
CPI Forecasting Model.
Ridge Regression + XGBoost ensemble.
Outputs full probability distribution (mu, sigma) for CPI MoM.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import date, datetime
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Features that have zero lookahead contamination
FEATURE_COLUMNS = [
    "cpi_lag1_mom",
    "cpi_lag2_mom",
    "cpi_lag12_mom",
    "cpi_lag1_yoy",
    "gasoline_mom",
    "gasoline_avg",
    "ppi_mom",
    "import_price_mom",
    "michigan_inflation_expect",
    "shelter_cpi_lag1",
    "shelter_cpi_lag2",
    "energy_cpi_lag1",
    "food_cpi_lag1",
    "month_sin",
    "month_cos",
]

TARGET = "target_cpi_mom"


class CPIForecaster:
    """
    Ensemble CPI forecasting model.
    Ridge (60%) + XGBoost (40%) weighted average.
    Outputs (mu, sigma) of forecast distribution.
    """
    
    VERSION = "1.0.0"
    RIDGE_WEIGHT = 0.60
    XGB_WEIGHT = 0.40
    SIGMA_FLOOR = 0.05  # Minimum uncertainty — never be overconfident
    
    def __init__(self):
        self.ridge = None
        self.xgb = None
        self.scaler = StandardScaler()
        self.ridge_weight = self.RIDGE_WEIGHT
        self.xgb_weight = self.XGB_WEIGHT
        self.is_trained = False
        self.training_errors = []  # Store residuals for sigma estimation
        self.feature_columns = FEATURE_COLUMNS
    
    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and impute feature matrix."""
        X = df[self.feature_columns].copy()
        # Impute missing with column median; fall back to 0.0 if whole column is NaN
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(0.0 if pd.isna(median_val) else median_val)
        return X.values
    
    def train(self, df: pd.DataFrame, verbose: bool = True):
        """
        Train on historical data.
        df must contain feature columns + TARGET column.
        """
        df = df.dropna(subset=[TARGET])
        df = df[df[TARGET].between(-1.5, 1.5)]  # Remove outlier releases
        
        X_raw = self._prepare_X(df)
        y = df[TARGET].values
        
        if verbose:
            logger.info(f"Training on {len(df)} samples, {len(self.feature_columns)} features")
        
        # Scale features (Ridge needs scaling, XGBoost doesn't but we do it anyway for consistency)
        X = self.scaler.fit_transform(X_raw)
        
        # --- Ridge Regression with CV alpha selection ---
        tscv = TimeSeriesSplit(n_splits=8, gap=1)
        
        best_alpha = 1.0
        best_score = np.inf
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            ridge_test = Ridge(alpha=alpha)
            scores = -cross_val_score(ridge_test, X, y, cv=tscv, scoring="neg_mean_squared_error")
            if scores.mean() < best_score:
                best_score = scores.mean()
                best_alpha = alpha
        
        self.ridge = Ridge(alpha=best_alpha)
        self.ridge.fit(X, y)
        
        if verbose:
            logger.info(f"Ridge alpha selected: {best_alpha}, CV RMSE: {np.sqrt(best_score):.4f}")
        
        # --- XGBoost ---
        self.xgb = XGBRegressor(
            n_estimators=100,
            max_depth=3,  # Shallow to prevent overfitting on small dataset
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        self.xgb.fit(X, y)
        
        # --- Compute training residuals for sigma estimation ---
        ridge_preds = self.ridge.predict(X)
        xgb_preds = self.xgb.predict(X)
        ensemble_preds = self.ridge_weight * ridge_preds + self.xgb_weight * xgb_preds
        
        self.training_errors = y - ensemble_preds
        self.is_trained = True
        
        # --- Walk-forward validation to check overfitting ---
        self._walk_forward_validation(df, verbose)
        
        if verbose:
            logger.info(f"✅ Model trained successfully (v{self.VERSION})")
    
    def _walk_forward_validation(self, df: pd.DataFrame, verbose: bool):
        """
        Walk-forward validation using the FULL ensemble (Ridge + XGBoost).

        Each fold:
        - Fits its own scaler on training data only (no test leakage)
        - Selects Ridge alpha via inner TimeSeriesSplit CV
        - Trains both Ridge and XGBoost
        - Predicts with the same ensemble weights used in production

        The residuals from this replace training_errors as the sigma basis,
        so uncertainty estimates reflect actual out-of-sample ensemble error
        rather than in-sample Ridge error.
        """
        n = len(df)
        min_train = min(120, int(n * 0.6))  # At least 10 years of seed data

        # Save in-sample ensemble RMSE before overwriting training_errors
        insample_rmse = float(np.sqrt(np.mean(self.training_errors ** 2)))

        wf_errors = []

        for i in range(min_train, n):
            train_df = df.iloc[:i]
            test_df  = df.iloc[i:i + 1]

            X_train_raw = self._prepare_X(train_df)
            X_test_raw  = self._prepare_X(test_df)
            y_train = train_df[TARGET].values
            y_test  = test_df[TARGET].values

            # Scaler fitted on training fold only — no test-set leakage
            fold_scaler = StandardScaler()
            X_train = fold_scaler.fit_transform(X_train_raw)
            X_test  = fold_scaler.transform(X_test_raw)

            # Ridge alpha selection via inner CV (skip if too few samples)
            best_alpha = 1.0
            if len(train_df) >= 40:
                n_inner = min(5, max(2, len(train_df) // 20))
                inner_cv = TimeSeriesSplit(n_splits=n_inner, gap=1)
                best_score = np.inf
                for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                    scores = -cross_val_score(
                        Ridge(alpha=alpha), X_train, y_train,
                        cv=inner_cv, scoring="neg_mean_squared_error"
                    )
                    if scores.mean() < best_score:
                        best_score = scores.mean()
                        best_alpha = alpha

            fold_ridge = Ridge(alpha=best_alpha)
            fold_ridge.fit(X_train, y_train)

            fold_xgb = XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0
            )
            fold_xgb.fit(X_train, y_train)

            # Ensemble prediction with production weights
            pred = (self.ridge_weight * fold_ridge.predict(X_test)[0] +
                    self.xgb_weight  * fold_xgb.predict(X_test)[0])

            wf_errors.append(float(y_test[0] - pred))

        wf_rmse = float(np.sqrt(np.mean(np.array(wf_errors) ** 2)))

        if verbose:
            logger.info(
                f"Walk-forward RMSE (ensemble): {wf_rmse:.4f} | "
                f"In-sample RMSE (ensemble):    {insample_rmse:.4f} | "
                f"n_folds: {len(wf_errors)}"
            )
            if wf_rmse > 2.0 * insample_rmse:
                logger.warning("⚠️  OVERFITTING DETECTED: walk-forward RMSE >> in-sample RMSE")

        # Replace training_errors with honest walk-forward residuals.
        # sigma will now reflect actual out-of-sample ensemble uncertainty.
        self.training_errors = np.array(wf_errors)
    
    def predict(self, features: dict, spf_forecast: float = None, 
                cleveland_nowcast: float = None) -> dict:
        """
        Generate probabilistic forecast for a single CPI release.
        
        Returns dict with:
        - mu_model: point estimate (MoM %)
        - sigma_model: uncertainty (std dev)
        - component_forecasts: dict of sub-model predictions
        - distribution: scipy norm object for probability calculations
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Build feature row
        feature_row = {col: features.get(col, np.nan) for col in self.feature_columns}
        df_row = pd.DataFrame([feature_row])
        
        # Impute missing
        for col in self.feature_columns:
            if pd.isna(df_row[col].iloc[0]):
                df_row[col] = 0.0  # Zero-impute for prediction
        
        X = self.scaler.transform(df_row.values)
        
        mu_ridge = float(self.ridge.predict(X)[0])
        mu_xgb = float(self.xgb.predict(X)[0])
        mu_ensemble = self.ridge_weight * mu_ridge + self.xgb_weight * mu_xgb
        
        # Blend with external sources if available
        sources = {"ridge": mu_ridge, "xgboost": mu_xgb, "ensemble": mu_ensemble}
        weights = {"ensemble": 0.50}
        
        if cleveland_nowcast is not None:
            sources["cleveland"] = cleveland_nowcast
            weights["cleveland"] = 0.30
            weights["ensemble"] = 0.40
        
        if spf_forecast is not None:
            sources["spf"] = spf_forecast
            weights["spf"] = 0.20
            weights["ensemble"] = 0.30 if cleveland_nowcast else 0.50
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Compute blended mu
        mu_final = 0.0
        for source, w in weights.items():
            if source == "ensemble":
                mu_final += w * mu_ensemble
            elif source == "cleveland":
                mu_final += w * cleveland_nowcast
            elif source == "spf":
                mu_final += w * spf_forecast
        
        # Sigma estimation from walk-forward residuals
        sigma_historical = float(np.std(self.training_errors))
        sigma_final = max(sigma_historical, self.SIGMA_FLOOR)
        
        return {
            "mu_model": round(mu_final, 4),
            "sigma_model": round(sigma_final, 4),
            "mu_ridge": round(mu_ridge, 4),
            "mu_xgboost": round(mu_xgb, 4),
            "mu_ensemble": round(mu_ensemble, 4),
            "mu_cleveland": cleveland_nowcast,
            "mu_spf": spf_forecast,
            "weights_used": weights,
            "distribution": norm(mu_final, sigma_final)
        }
    
    def bucket_probability(self, low: float, high: float, 
                           mu: float, sigma: float) -> float:
        """P(CPI MoM falls in [low, high)) given N(mu, sigma)."""
        if high == np.inf:
            return 1 - norm.cdf(low, mu, sigma)
        if low == -np.inf:
            return norm.cdf(high, mu, sigma)
        return norm.cdf(high, mu, sigma) - norm.cdf(low, mu, sigma)
    
    def compute_edge(self, p_model: float, p_market: float, 
                     fee_rate: float = 0.07) -> float:
        """
        Net edge after fees.
        fee_rate: Kalshi fee as fraction of stake (~7% round-trip).
        """
        return p_model - p_market - fee_rate
    
    def kelly_fraction(self, net_edge: float, p_market: float, 
                       side: str = "buy") -> float:
        """
        Fractional Kelly position size.
        side: 'buy' (long) or 'sell' (short).
        """
        if side == "buy":
            if p_market >= 1.0:
                return 0.0
            raw_kelly = net_edge / (1 - p_market)
        else:
            if p_market <= 0.0:
                return 0.0
            raw_kelly = -net_edge / p_market
        
        fractional = raw_kelly * float(os.getenv("KELLY_FRACTION", 0.25))
        return max(0.0, min(fractional, 0.25))  # Cap at 25% of risk capital
    
    def save(self, path: str = None):
        if path is None:
            path = os.path.join(MODEL_DIR, "cpi_forecaster.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "ridge": self.ridge,
                "xgb": self.xgb,
                "scaler": self.scaler,
                "training_errors": self.training_errors,
                "ridge_weight": self.ridge_weight,
                "xgb_weight": self.xgb_weight,
                "version": self.VERSION,
                "feature_columns": self.feature_columns
            }, f)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str = None):
        if path is None:
            path = os.path.join(MODEL_DIR, "cpi_forecaster.pkl")
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.ridge = state["ridge"]
        self.xgb = state["xgb"]
        self.scaler = state["scaler"]
        self.training_errors = state["training_errors"]
        self.ridge_weight = state["ridge_weight"]
        self.xgb_weight = state["xgb_weight"]
        self.VERSION = state["version"]
        self.feature_columns = state["feature_columns"]
        self.is_trained = True
        logger.info(f"Model loaded from {path} (v{self.VERSION})")


# ─────────────────────────────────────────────
# BRIER SCORE / CALIBRATION UTILITIES
# ─────────────────────────────────────────────

def compute_brier_score(predictions: list, outcomes: list) -> float:
    """
    Brier score for categorical (bucket) predictions.
    predictions: list of dicts {bucket: probability}
    outcomes: list of dicts {bucket: 0 or 1}
    """
    scores = []
    for pred, outcome in zip(predictions, outcomes):
        score = sum((pred.get(b, 0) - outcome.get(b, 0))**2 for b in set(list(pred.keys()) + list(outcome.keys())))
        scores.append(score)
    return float(np.mean(scores))

def compute_brier_skill_score(model_brier: float, naive_brier: float) -> float:
    """BSS = 1 - model_brier/naive_brier. Positive = better than naive."""
    if naive_brier == 0:
        return 0.0
    return 1 - model_brier / naive_brier

def compute_calibration_error(predictions: np.ndarray, outcomes: np.ndarray, 
                               n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i+1])
        if mask.sum() == 0:
            continue
        avg_confidence = predictions[mask].mean()
        avg_accuracy = outcomes[mask].mean()
        ece += mask.mean() * abs(avg_confidence - avg_accuracy)
    return float(ece)


if __name__ == "__main__":
    # Quick test
    from data.ingestion import FeatureBuilder
    
    builder = FeatureBuilder()
    logger.info("Building training dataset...")
    df = builder.build_training_dataset(start_year=1995)
    
    model = CPIForecaster()
    model.train(df)
    model.save()
    
    logger.info("Model trained and saved.")
