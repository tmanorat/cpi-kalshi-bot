"""
CPI Forecasting Model — v2.0.0
Ridge Regression + XGBoost ensemble.
Outputs full probability distribution (mu, sigma) for CPI MoM.

v2.0.0 improvements over v1:
- XGBoost random-search hyperparameter tuning (TimeSeriesSplit-scored, 20 iter)
- Ensemble weights optimised via held-out walk-forward grid search (not hardcoded 60/40)
- Heteroskedastic sigma: exponentially-weighted recent residuals blended with global
- Walk-forward calibration: empirical 1σ/2σ coverage + sigma correction factor
- Feature importance logging at training time (Ridge coefs + XGBoost gain)
- Model rollback protection: compare_to_disk() blocks regressions > 10%
- 4 new macro features: fed_funds_rate, core_cpi_lag1_mom, treasury_10y, unemp_rate
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
from datetime import date, datetime
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS — zero lookahead contamination
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    # CPI lags
    "cpi_lag1_mom",
    "cpi_lag2_mom",
    "cpi_lag12_mom",
    "cpi_lag1_yoy",
    # Energy / gasoline
    "gasoline_mom",
    "gasoline_avg",
    # Producer / import prices
    "ppi_mom",
    "import_price_mom",
    # Inflation expectations
    "michigan_inflation_expect",
    # Shelter (largest CPI component ~35%)
    "shelter_cpi_lag1",
    "shelter_cpi_lag2",
    # Other CPI components
    "energy_cpi_lag1",
    "food_cpi_lag1",
    # Seasonality
    "month_sin",
    "month_cos",
    # Macro factors added in v2.0
    "fed_funds_rate",
    "core_cpi_lag1_mom",
    "treasury_10y",
    "unemp_rate",
]

TARGET = "target_cpi_mom"

# XGBoost random-search parameter space
_XGB_PARAM_GRID = {
    "n_estimators":    [50, 100, 150, 200],
    "max_depth":       [2, 3, 4],
    "learning_rate":   [0.02, 0.05, 0.10],
    "subsample":       [0.70, 0.80, 1.00],
    "colsample_bytree":[0.70, 0.80, 1.00],
    "min_child_weight":[1, 3, 5],
}
_XGB_N_ITER = 20  # Random draws — keeps training under ~60 s on typical hardware


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CPIForecaster:
    """
    Ensemble CPI forecasting model.
    Ridge + XGBoost with learned weights and calibrated heteroskedastic sigma.
    All parameters are data-driven; nothing hardcoded beyond sensible defaults.
    """

    VERSION      = "2.0.0"
    SIGMA_FLOOR  = 0.05   # Minimum uncertainty — never be overconfident
    # Initial weight defaults (overwritten by _optimize_ensemble_weights)
    RIDGE_WEIGHT = 0.60
    XGB_WEIGHT   = 0.40

    def __init__(self):
        self.ridge            = None
        self.xgb              = None
        self.scaler           = StandardScaler()
        self.ridge_weight     = self.RIDGE_WEIGHT
        self.xgb_weight       = self.XGB_WEIGHT
        self.is_trained       = False
        self.training_errors  = np.array([])   # Walk-forward residuals
        self.feature_columns  = FEATURE_COLUMNS
        # v2.0 state
        self.xgb_params:        dict  = {}
        self.sigma_global:      float = 0.15
        self.sigma_recent:      float = 0.15
        self.sigma_correction:  float = 1.0
        self.calibration_stats: dict  = {}

    # ─────────────────────────────────────────────────────────────────────
    # DATA PREPARATION
    # ─────────────────────────────────────────────────────────────────────

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix and impute missing values with column median."""
        X = df[self.feature_columns].copy()
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(0.0 if pd.isna(median_val) else median_val)
        return X.values

    # ─────────────────────────────────────────────────────────────────────
    # HYPERPARAMETER SELECTION
    # ─────────────────────────────────────────────────────────────────────

    def _select_ridge_alpha(self, X: np.ndarray, y: np.ndarray,
                             n_splits: int = 8) -> float:
        """Select Ridge regularisation strength via TimeSeriesSplit CV."""
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)
        best_alpha, best_score = 1.0, np.inf
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            scores = -cross_val_score(
                Ridge(alpha=alpha), X, y,
                cv=tscv, scoring="neg_mean_squared_error"
            )
            if scores.mean() < best_score:
                best_score, best_alpha = scores.mean(), alpha
        return best_alpha

    def _tune_xgboost(self, X: np.ndarray, y: np.ndarray,
                      verbose: bool = True) -> dict:
        """
        Random search over XGBoost hyperparameter space using TimeSeriesSplit CV.
        Returns the best parameter dict found across _XGB_N_ITER random draws.
        """
        tscv = TimeSeriesSplit(n_splits=5, gap=1)
        rng  = random.Random(42)

        best_params, best_score = {}, np.inf

        for _ in range(_XGB_N_ITER):
            params = {k: rng.choice(v) for k, v in _XGB_PARAM_GRID.items()}
            scores = -cross_val_score(
                XGBRegressor(**params, random_state=42, verbosity=0),
                X, y, cv=tscv, scoring="neg_mean_squared_error"
            )
            if scores.mean() < best_score:
                best_score  = scores.mean()
                best_params = dict(params)

        if verbose:
            logger.info(
                f"XGBoost tuning complete: best params={best_params}, "
                f"CV RMSE={np.sqrt(best_score):.4f}"
            )
        return best_params

    # ─────────────────────────────────────────────────────────────────────
    # ENSEMBLE WEIGHT OPTIMISATION
    # ─────────────────────────────────────────────────────────────────────

    def _optimize_ensemble_weights(
        self,
        wf_ridge:   np.ndarray,
        wf_xgb:     np.ndarray,
        wf_actuals: np.ndarray,
        verbose:    bool = True,
    ) -> tuple:
        """
        Grid search for the best ridge_weight on held-out walk-forward predictions.
        This replaces the hardcoded 60/40 split with a data-driven allocation.

        Returns (ridge_weight, xgb_weight) that minimise walk-forward MSE.
        """
        best_w, best_mse = 0.60, np.inf
        for w in np.arange(0.20, 0.86, 0.05):
            ensemble = w * wf_ridge + (1.0 - w) * wf_xgb
            mse = float(np.mean((wf_actuals - ensemble) ** 2))
            if mse < best_mse:
                best_mse, best_w = mse, float(w)

        ridge_w = round(best_w, 2)
        xgb_w   = round(1.0 - ridge_w, 2)  # derive from rounded ridge_w so weights sum to 1.0

        if verbose:
            logger.info(
                f"Ensemble weights optimised: Ridge={ridge_w:.2f}, "
                f"XGBoost={xgb_w:.2f} | WF RMSE={np.sqrt(best_mse):.4f}"
            )
        return ridge_w, xgb_w

    # ─────────────────────────────────────────────────────────────────────
    # SIGMA / CALIBRATION
    # ─────────────────────────────────────────────────────────────────────

    def _compute_ewm_sigma(self, errors: np.ndarray,
                            half_life_months: int = 24) -> float:
        """
        Exponentially-weighted standard deviation of residuals.
        Gives more weight to recent observations so sigma adapts to changing
        economic volatility regimes without discarding historical information.
        """
        n = len(errors)
        if n < 6:
            return float(np.std(errors)) if n > 1 else self.SIGMA_FLOOR
        lam     = 0.5 ** (1.0 / half_life_months)
        weights = np.array([lam ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        ewm_mean = np.sum(weights * errors)
        ewm_var  = np.sum(weights * (errors - ewm_mean) ** 2)
        return float(np.sqrt(ewm_var))

    def _compute_calibration_metrics(
        self,
        wf_errors: np.ndarray,
        verbose:   bool = True,
    ) -> tuple:
        """
        Empirical sigma calibration from walk-forward residuals.

        A well-calibrated N(μ, σ) should have:
          - 68.27% of |errors| ≤ 1σ
          - 95.45% of |errors| ≤ 2σ

        We compute a correction factor:
          sigma_correction = empirical_68th_percentile(|errors|) / raw_sigma

        Applying this in predict() rescales sigma so the model achieves the
        68th-percentile coverage a Gaussian should.  Clamped to [0.5, 2.0]
        to guard against small-sample extremes.

        Returns (sigma_correction, calibration_stats_dict).
        """
        if len(wf_errors) < 10:
            return 1.0, {}

        sigma_raw = float(np.std(wf_errors))
        if sigma_raw < 1e-8:
            return 1.0, {}

        cov_1s = float(np.mean(np.abs(wf_errors) <= sigma_raw))
        cov_2s = float(np.mean(np.abs(wf_errors) <= 2.0 * sigma_raw))

        empirical_68th   = float(np.percentile(np.abs(wf_errors), 68.27))
        sigma_correction = float(np.clip(empirical_68th / sigma_raw, 0.5, 2.0))

        stats_dict = {
            "sigma_raw":            round(sigma_raw, 4),
            "sigma_correction":     round(sigma_correction, 4),
            "coverage_1sigma":      round(cov_1s, 4),
            "coverage_2sigma":      round(cov_2s, 4),
            "target_coverage_1sig": 0.6827,
            "target_coverage_2sig": 0.9545,
            "n_wf_folds":           len(wf_errors),
        }

        if verbose:
            logger.info(
                f"Calibration: 1σ coverage={cov_1s:.1%} (target 68.3%), "
                f"2σ coverage={cov_2s:.1%} (target 95.5%), "
                f"σ-correction={sigma_correction:.3f}"
            )

        return sigma_correction, stats_dict

    # ─────────────────────────────────────────────────────────────────────
    # FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────────────────

    def _log_feature_importance(self, verbose: bool = True):
        """Log top features by Ridge |coefficient| and XGBoost gain."""
        if not verbose or self.ridge is None or self.xgb is None:
            return

        ridge_coefs = pd.Series(
            np.abs(self.ridge.coef_), index=self.feature_columns
        ).sort_values(ascending=False)

        xgb_gains = pd.Series(
            self.xgb.feature_importances_, index=self.feature_columns
        ).sort_values(ascending=False)

        top_n = 8
        logger.info("Top features — Ridge |coef|:")
        for feat, val in ridge_coefs.head(top_n).items():
            logger.info(f"  {feat:<32} {val:.4f}")

        logger.info("Top features — XGBoost gain:")
        for feat, val in xgb_gains.head(top_n).items():
            logger.info(f"  {feat:<32} {val:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # TRAINING PIPELINE
    # ─────────────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, verbose: bool = True):
        """
        Full training pipeline (v2.0):

        1. Clean + scale data
        2. Random-search tune XGBoost (20 iterations, TimeSeriesSplit)
        3. CV-select Ridge alpha
        4. Fit production Ridge + XGBoost on full training data
        5. Walk-forward validation → collect per-fold sub-model predictions
        6. Grid-search optimise ensemble weights on walk-forward predictions
        7. Recompute WF errors with optimised weights
        8. Calibrate sigma (empirical coverage + correction factor)
        9. Compute heteroskedastic sigma components (EWM + global)
        10. Log feature importance
        """
        df = df.dropna(subset=[TARGET])
        df = df[df[TARGET].between(-1.5, 1.5)]  # Remove outlier releases

        X_raw = self._prepare_X(df)
        y     = df[TARGET].values

        if verbose:
            logger.info(
                f"Training on {len(df)} samples, {len(self.feature_columns)} features"
            )

        X = self.scaler.fit_transform(X_raw)

        # Step 2: Tune XGBoost
        self.xgb_params = self._tune_xgboost(X, y, verbose)

        # Step 3: Ridge alpha
        best_alpha = self._select_ridge_alpha(X, y)
        if verbose:
            logger.info(f"Ridge alpha selected: {best_alpha}")

        # Step 4: Production models on full data
        self.ridge = Ridge(alpha=best_alpha)
        self.ridge.fit(X, y)

        self.xgb = XGBRegressor(**self.xgb_params, random_state=42, verbosity=0)
        self.xgb.fit(X, y)

        # Temporary in-sample errors (needed by _walk_forward_validation log)
        ridge_preds = self.ridge.predict(X)
        xgb_preds   = self.xgb.predict(X)
        self.training_errors = (
            y - (self.ridge_weight * ridge_preds + self.xgb_weight * xgb_preds)
        )
        self.is_trained = True

        # Step 5: Walk-forward validation
        wf_errors, wf_ridge, wf_xgb, wf_actuals = self._walk_forward_validation(
            df, verbose, self.xgb_params
        )

        # Step 6: Optimise ensemble weights
        ridge_w, xgb_w = self._optimize_ensemble_weights(
            wf_ridge, wf_xgb, wf_actuals, verbose
        )
        self.ridge_weight = ridge_w
        self.xgb_weight   = xgb_w

        # Step 7: Recompute errors with optimised weights
        wf_errors_opt       = wf_actuals - (self.ridge_weight * wf_ridge + self.xgb_weight * wf_xgb)
        self.training_errors = wf_errors_opt

        # Step 8: Calibration
        self.sigma_correction, self.calibration_stats = (
            self._compute_calibration_metrics(wf_errors_opt, verbose)
        )

        # Step 9: Heteroskedastic sigma
        self.sigma_global = max(float(np.std(wf_errors_opt)),   self.SIGMA_FLOOR)
        self.sigma_recent = max(self._compute_ewm_sigma(wf_errors_opt), self.SIGMA_FLOOR)

        # Step 10: Feature importance
        self._log_feature_importance(verbose)

        if verbose:
            logger.info(
                f"✅ Model trained (v{self.VERSION}) | "
                f"Ridge={self.ridge_weight:.2f}, XGB={self.xgb_weight:.2f} | "
                f"σ_global={self.sigma_global:.4f}, σ_recent={self.sigma_recent:.4f}, "
                f"σ_corr={self.sigma_correction:.3f}"
            )

    def _walk_forward_validation(
        self,
        df:         pd.DataFrame,
        verbose:    bool,
        xgb_params: dict = None,
    ):
        """
        Rolling walk-forward validation using the full Ridge + XGBoost ensemble.

        Each fold uses:
        - Its own StandardScaler fitted on training data only (no test leakage)
        - Ridge alpha selected via inner TimeSeriesSplit CV
        - XGBoost with provided xgb_params (tuned on full data above)

        Returns:
            (wf_errors, wf_ridge_preds, wf_xgb_preds, wf_actuals)
        All as numpy arrays so train() can do weight optimisation and calibration.
        """
        if xgb_params is None:
            xgb_params = {
                "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8,
            }

        n         = len(df)
        min_train = min(120, int(n * 0.6))

        insample_rmse = float(np.sqrt(np.mean(self.training_errors ** 2)))

        wf_errors:      list = []
        wf_ridge_preds: list = []
        wf_xgb_preds:   list = []
        wf_actuals:     list = []

        for i in range(min_train, n):
            train_df = df.iloc[:i]
            test_df  = df.iloc[i:i + 1]

            X_train_raw = self._prepare_X(train_df)
            X_test_raw  = self._prepare_X(test_df)
            y_train     = train_df[TARGET].values
            y_test      = test_df[TARGET].values

            fold_scaler = StandardScaler()
            X_train = fold_scaler.fit_transform(X_train_raw)
            X_test  = fold_scaler.transform(X_test_raw)

            # Inner Ridge alpha selection
            best_alpha = 1.0
            if len(train_df) >= 40:
                n_inner  = min(5, max(2, len(train_df) // 20))
                inner_cv = TimeSeriesSplit(n_splits=n_inner, gap=1)
                best_sc  = np.inf
                for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                    sc = -cross_val_score(
                        Ridge(alpha=alpha), X_train, y_train,
                        cv=inner_cv, scoring="neg_mean_squared_error"
                    )
                    if sc.mean() < best_sc:
                        best_sc, best_alpha = sc.mean(), alpha

            fold_ridge = Ridge(alpha=best_alpha)
            fold_ridge.fit(X_train, y_train)

            fold_xgb = XGBRegressor(**xgb_params, random_state=42, verbosity=0)
            fold_xgb.fit(X_train, y_train)

            r_pred = float(fold_ridge.predict(X_test)[0])
            x_pred = float(fold_xgb.predict(X_test)[0])
            pred   = self.ridge_weight * r_pred + self.xgb_weight * x_pred

            wf_ridge_preds.append(r_pred)
            wf_xgb_preds.append(x_pred)
            wf_actuals.append(float(y_test[0]))
            wf_errors.append(float(y_test[0]) - pred)

        wf_errors_arr = np.array(wf_errors)
        wf_rmse       = float(np.sqrt(np.mean(wf_errors_arr ** 2)))

        if verbose:
            logger.info(
                f"Walk-forward RMSE (ensemble): {wf_rmse:.4f} | "
                f"In-sample RMSE: {insample_rmse:.4f} | "
                f"n_folds: {len(wf_errors)}"
            )
            if wf_rmse > 2.0 * insample_rmse:
                logger.warning(
                    "⚠️  OVERFITTING DETECTED: walk-forward RMSE >> in-sample RMSE"
                )

        return (
            wf_errors_arr,
            np.array(wf_ridge_preds),
            np.array(wf_xgb_preds),
            np.array(wf_actuals),
        )

    # ─────────────────────────────────────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, features: dict, spf_forecast: float = None,
                cleveland_nowcast: float = None) -> dict:
        """
        Generate probabilistic forecast for a single CPI release.

        Sigma is heteroskedastic (EWM-recent blended with global) and
        calibration-corrected so empirical 68th-percentile coverage matches
        the Gaussian 68.27% target.

        Returns dict with:
        - mu_model:    blended point estimate (MoM %)
        - sigma_model: calibrated, heteroskedastic uncertainty (std dev)
        - distribution: scipy norm object for probability calculations
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        feature_row = {col: features.get(col, np.nan) for col in self.feature_columns}
        df_row = pd.DataFrame([feature_row])
        for col in self.feature_columns:
            if pd.isna(df_row[col].iloc[0]):
                df_row[col] = 0.0

        X = self.scaler.transform(df_row.values)

        mu_ridge    = float(self.ridge.predict(X)[0])
        mu_xgb      = float(self.xgb.predict(X)[0])
        mu_ensemble = self.ridge_weight * mu_ridge + self.xgb_weight * mu_xgb

        # Blend with external sources if available
        weights = {"ensemble": 0.50}
        if cleveland_nowcast is not None:
            weights["cleveland"] = 0.30
            weights["ensemble"]  = 0.40
        if spf_forecast is not None:
            weights["spf"]      = 0.20
            weights["ensemble"] = 0.30 if cleveland_nowcast else 0.50

        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        mu_final = 0.0
        for source, w in weights.items():
            if source == "ensemble":
                mu_final += w * mu_ensemble
            elif source == "cleveland":
                mu_final += w * cleveland_nowcast
            elif source == "spf":
                mu_final += w * spf_forecast

        # Heteroskedastic sigma: blend recent EWM with global, then calibrate
        sigma_base       = 0.60 * self.sigma_recent + 0.40 * self.sigma_global
        sigma_calibrated = sigma_base * self.sigma_correction
        sigma_final      = max(sigma_calibrated, self.SIGMA_FLOOR)

        return {
            "mu_model":         round(mu_final, 4),
            "sigma_model":      round(sigma_final, 4),
            "mu_ridge":         round(mu_ridge, 4),
            "mu_xgboost":       round(mu_xgb, 4),
            "mu_ensemble":      round(mu_ensemble, 4),
            "mu_cleveland":     cleveland_nowcast,
            "mu_spf":           spf_forecast,
            "weights_used":     weights,
            "sigma_global":     round(self.sigma_global, 4),
            "sigma_recent":     round(self.sigma_recent, 4),
            "sigma_correction": round(self.sigma_correction, 4),
            "distribution":     norm(mu_final, sigma_final),
        }

    # ─────────────────────────────────────────────────────────────────────
    # TRADING UTILITIES
    # ─────────────────────────────────────────────────────────────────────

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
        """Net edge after fees. fee_rate: Kalshi ~7% round-trip."""
        return p_model - p_market - fee_rate

    def kelly_fraction(self, net_edge: float, p_market: float,
                       side: str = "buy") -> float:
        """Fractional Kelly position size. side: 'buy' or 'sell'."""
        if side == "buy":
            if p_market >= 1.0:
                return 0.0
            raw_kelly = net_edge / (1 - p_market)
        else:
            if p_market <= 0.0:
                return 0.0
            raw_kelly = -net_edge / p_market
        fractional = raw_kelly * float(os.getenv("KELLY_FRACTION", 0.25))
        return max(0.0, min(fractional, 0.25))

    # ─────────────────────────────────────────────────────────────────────
    # MODEL VERSIONING / ROLLBACK PROTECTION
    # ─────────────────────────────────────────────────────────────────────

    def compare_to_disk(self, path: str = None, tolerance: float = 0.10) -> bool:
        """
        Guard against deploying a regressed model.

        Returns True (safe to deploy) if:
          - No saved model exists on disk, OR
          - New walk-forward RMSE ≤ old RMSE × (1 + tolerance)

        Returns False (deploy blocked) if new model is meaningfully worse.
        The caller (engine.train_model) decides whether to abort or force-save.
        """
        if path is None:
            path = os.path.join(MODEL_DIR, "cpi_forecaster.pkl")
        if not os.path.exists(path):
            logger.info("No saved model on disk — safe to deploy new model.")
            return True
        try:
            with open(path, "rb") as f:
                old_state = pickle.load(f)
            old_errors = np.array(old_state.get("training_errors", []))
            if len(old_errors) < 5:
                return True
            old_rmse = float(np.sqrt(np.mean(old_errors ** 2)))
            new_rmse = float(np.sqrt(np.mean(self.training_errors ** 2)))
            if new_rmse > old_rmse * (1.0 + tolerance):
                logger.warning(
                    f"⚠️  Rollback protection: new RMSE={new_rmse:.4f} > "
                    f"old RMSE={old_rmse:.4f} × {1 + tolerance:.2f}. "
                    "Deploy blocked."
                )
                return False
            logger.info(
                f"Rollback check passed: new RMSE={new_rmse:.4f} vs "
                f"old RMSE={old_rmse:.4f} (Δ={new_rmse - old_rmse:+.4f})"
            )
            return True
        except Exception as e:
            logger.warning(f"compare_to_disk error ({e}) — defaulting to deploy=True")
            return True

    # ─────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str = None):
        if path is None:
            path = os.path.join(MODEL_DIR, "cpi_forecaster.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                # Core model state
                "ridge":             self.ridge,
                "xgb":               self.xgb,
                "scaler":            self.scaler,
                "training_errors":   self.training_errors,
                "ridge_weight":      self.ridge_weight,
                "xgb_weight":        self.xgb_weight,
                "version":           self.VERSION,
                "feature_columns":   self.feature_columns,
                # v2.0 additions
                "xgb_params":        self.xgb_params,
                "sigma_global":      self.sigma_global,
                "sigma_recent":      self.sigma_recent,
                "sigma_correction":  self.sigma_correction,
                "calibration_stats": self.calibration_stats,
            }, f)
        logger.info(f"Model saved to {path} (v{self.VERSION})")

    def load(self, path: str = None):
        if path is None:
            path = os.path.join(MODEL_DIR, "cpi_forecaster.pkl")
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.ridge           = state["ridge"]
        self.xgb             = state["xgb"]
        self.scaler          = state["scaler"]
        self.training_errors = state["training_errors"]
        self.ridge_weight    = state["ridge_weight"]
        self.xgb_weight      = state["xgb_weight"]
        self.VERSION         = state["version"]
        self.feature_columns = state["feature_columns"]
        # v2.0 fields — graceful fallback for models saved by v1
        _fallback_sigma = float(np.std(self.training_errors)) if len(self.training_errors) else 0.15
        self.xgb_params        = state.get("xgb_params", {})
        self.sigma_global      = state.get("sigma_global", _fallback_sigma)
        self.sigma_recent      = state.get("sigma_recent", _fallback_sigma)
        self.sigma_correction  = state.get("sigma_correction", 1.0)
        self.calibration_stats = state.get("calibration_stats", {})
        self.is_trained = True
        logger.info(f"Model loaded from {path} (v{self.VERSION})")


# ─────────────────────────────────────────────────────────────────────────────
# BRIER SCORE / CALIBRATION UTILITIES  (module-level helpers)
# ─────────────────────────────────────────────────────────────────────────────

def compute_brier_score(predictions: list, outcomes: list) -> float:
    """
    Brier score for categorical (bucket) predictions.
    predictions: list of dicts {bucket: probability}
    outcomes:    list of dicts {bucket: 0 or 1}
    """
    scores = []
    for pred, outcome in zip(predictions, outcomes):
        keys  = set(list(pred.keys()) + list(outcome.keys()))
        score = sum((pred.get(b, 0) - outcome.get(b, 0)) ** 2 for b in keys)
        scores.append(score)
    return float(np.mean(scores))


def compute_brier_skill_score(model_brier: float, naive_brier: float) -> float:
    """BSS = 1 - model_brier / naive_brier. Positive = better than naive."""
    if naive_brier == 0:
        return 0.0
    return 1 - model_brier / naive_brier


def compute_calibration_error(predictions: np.ndarray, outcomes: np.ndarray,
                               n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_confidence = predictions[mask].mean()
        avg_accuracy   = outcomes[mask].mean()
        ece += mask.mean() * abs(avg_confidence - avg_accuracy)
    return float(ece)


if __name__ == "__main__":
    from data.ingestion import FeatureBuilder

    builder = FeatureBuilder()
    logger.info("Building training dataset...")
    df = builder.build_training_dataset(start_year=1995)

    model = CPIForecaster()
    model.train(df)
    model.save()

    logger.info("Model trained and saved.")
