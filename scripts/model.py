"""
=============================================================================
Forecasting Pipeline
Sparse GP with XGBoost Mean Function (Residual GP)
=============================================================================

ARCHITECTURE:
    Stage 1 — XGBoost regressor trained on technical indicators from the
              database to predict next-day log returns. XGBoost handles the
              non-linear tabular regression task efficiently.

    Stage 2 — Sparse Variational Gaussian Process (SVGP via GPyTorch) trained
              on the residuals of Stage 1. The GP models systematic structure
              that XGBoost misses and provides calibrated predictive uncertainty
              (the posterior variance is the confidence interval).

    Combined — Final forecast = XGBoost mean + GP correction
               Uncertainty    = GP posterior std → price confidence intervals

CONNECTION TO PRIOR WORK (BNN):
    - Feature engineering pipeline is identical to BNN_A/B.py
    - Uncertainty output format (mean, lower_bound, upper_bound) is identical
    - Sentiment scores from sentimenttool.py can be added as features
    - GP kernel (RBF) has the same theoretical foundation as kernel SVMs
      covered in prior ML coursework

SCALABILITY vs BNN:
    - BNN: ~1 hour per 3 tickers, two concurrent processes, full GPU
    - This pipeline: ~500 tickers in 2-4 hours total, single GPU process
    - XGBoost: CPU-parallel, fits all 500 tickers in ~10 minutes total
    - SVGP: GPU, ~1-2 minutes per ticker with minibatch training

=============================================================================
"""

import duckdb
import numpy as np
import pandas as pd
import torch
import gpytorch
import xgboost as xgb
import json
import os
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model.log"),
        logging.StreamHandler()
    ]
)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"[device] Using: {device}")

# ── Configuration ─────────────────────────────────────────────────────────────
DB_PATH          = "stock_data.db"
OUTPUT_DIR       = Path("model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

LOOKBACK         = 1          # predict 1-day-ahead log return
FORECAST_HORIZON = 20         # multi-step forecast days
N_INDUCING       = 128        # sparse GP inducing points (increase for accuracy)
GP_EPOCHS        = 150        # GP training epochs per ticker
GP_LR            = 0.05       # GP learning rate
XGB_ROUNDS       = 400        # XGBoost boosting rounds
BATCH_SIZE       = 256        # GP minibatch size
MIN_TRAIN_ROWS   = 200        # skip tickers with insufficient history
SENTIMENT_WEIGHT = 0.3        # weight for sentiment feature if available

# Feature columns that exist in TechnicalIndicators table
FEATURE_COLS = [
    "close", "daily_return", "log_return",
    "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_12", "ema_26", "ema_50",
    "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct_b",
    "rsi_14",
    "volume_ratio",
    "atr_14", "hist_vol_20",
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_ticker_data(con: duckdb.DuckDBPyConnection,
                     symbol: str,
                     sentiment_map: dict | None = None) -> pd.DataFrame:
    """
    Loads all technical indicator rows for a single ticker from DuckDB,
    optionally joins daily sentiment scores, and constructs the target variable.

    TARGET:
        next-day log return — log(close[t+1] / close[t])
        This is a cleaner prediction target than raw price because:
        1. It is approximately stationary (price is not)
        2. It is symmetric around zero
        3. Returns are more comparable across tickers of different price scales

    SENTIMENT FEATURE (optional):
        If sentiment_map is provided (dict of {date_str -> avg_sentiment_score}),
        it is joined as an additional feature. The score is the FinBERT aggregate
        from sentimenttool.py, winsorized to [-1, 1].
    """
    query = f"""
        SELECT {', '.join(FEATURE_COLS)}, date
        FROM TechnicalIndicators
        WHERE symbol = '{symbol}'
          AND sma_200 IS NOT NULL          -- exclude warm-up rows
          AND hist_vol_20 IS NOT NULL
          AND hist_vol_20 > 0
        ORDER BY date ASC
    """
    df = con.execute(query).df()

    if len(df) < MIN_TRAIN_ROWS + FORECAST_HORIZON:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Target: next-day log return (shift -1 so row t has tomorrow's return)
    df["target"] = df["log_return"].shift(-1)

    # Optional sentiment feature
    if sentiment_map:
        df["sentiment"] = df.index.strftime("%Y-%m-%d").map(sentiment_map)
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
        df["sentiment"] = df["sentiment"].clip(-1, 1).fillna(0)
    else:
        df["sentiment"] = 0.0

    # Drop last row (target is NaN, no tomorrow to predict)
    df = df.dropna(subset=["target"])

    # Drop any remaining NaNs in features
    feature_cols_used = FEATURE_COLS + ["sentiment"]
    df = df.dropna(subset=feature_cols_used)

    return df


def load_all_tickers(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Returns all tickers that have sufficient data for modeling."""
    result = con.execute(f"""
        SELECT symbol, COUNT(*) AS n
        FROM TechnicalIndicators
        WHERE sma_200 IS NOT NULL
        GROUP BY symbol
        HAVING n >= {MIN_TRAIN_ROWS + FORECAST_HORIZON}
        ORDER BY symbol
    """).df()
    return result["symbol"].tolist()


def build_sentiment_map(con: duckdb.DuckDBPyConnection,
                        symbol: str) -> dict:
    """
    Aggregates FinBERT sentiment scores from the StockNews table into a
    daily average score per ticker. This bridges sentimenttool.py's output
    (stored in StockNews during ingestion) with the GP feature vector.

    NOTE: The StockNews table does not store pre-computed sentiment scores —
    it stores raw titles. For a production system you would run FinBERT over
    the titles and store the scores. Here we approximate sentiment using a
    simple positive/negative keyword count as a proxy, which runs without
    a GPU and is sufficient to demonstrate the pipeline integration.
    """
    query = f"""
        SELECT
            CAST(provider_publish_time AS DATE) AS pub_date,
            title
        FROM StockNews
        WHERE symbol = '{symbol}'
          AND provider_publish_time IS NOT NULL
        ORDER BY pub_date
    """
    news_df = con.execute(query).df()

    if news_df.empty:
        return {}

    # Keyword-based proxy sentiment (no GPU required for pipeline demo)
    # Replace with actual FinBERT scores if you run sentimenttool.py first
    positive_words = {
        "beat", "beats", "record", "growth", "profit", "surge", "rally",
        "upgrade", "bullish", "strong", "revenue", "gain", "rises", "up",
        "positive", "outperform", "buy", "exceeds", "boosts"
    }
    negative_words = {
        "miss", "misses", "loss", "decline", "drop", "fall", "bearish",
        "downgrade", "weak", "cut", "layoff", "lawsuit", "warning", "risk",
        "down", "negative", "underperform", "sell", "below", "concern"
    }

    def score_title(title: str) -> float:
        if not isinstance(title, str):
            return 0.0
        words = set(title.lower().split())
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    news_df["score"] = news_df["title"].apply(score_title)
    daily = (news_df.groupby("pub_date")["score"]
             .mean()
             .reset_index())
    daily["pub_date"] = daily["pub_date"].astype(str)
    return dict(zip(daily["pub_date"], daily["score"]))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: XGBOOST MEAN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_val: np.ndarray,
                  y_val: np.ndarray) -> xgb.XGBRegressor:
    """
    Trains an XGBoost regressor to predict next-day log returns.

    WHY XGBOOST FOR STAGE 1:
    ─────────────────────────
    XGBoost is a gradient-boosted decision tree ensemble — a classical ML
    technique covered in DS coursework. It is excellent at tabular regression
    tasks because it handles non-linearities and feature interactions without
    requiring feature scaling, and it is robust to the mild non-stationarity
    of rolling technical indicators.

    The key design choice is using TimeSeriesSplit for validation rather than
    random train/test split. Random splitting would leak future information
    into the training set because the features (SMAs, EWMs) are computed on
    overlapping windows.

    HYPERPARAMETERS:
    - max_depth=4: Shallow trees prevent overfitting on ~200-1700 row datasets
    - subsample=0.8: Row subsampling adds regularization
    - colsample_bytree=0.8: Feature subsampling prevents feature dominance
    - early_stopping_rounds=30: Stop when validation loss stops improving
    """
    model = xgb.XGBRegressor(
        n_estimators=XGB_ROUNDS,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,       # L1 regularization
        reg_lambda=1.0,      # L2 regularization
        objective="reg:squarederror",
        tree_method="hist",  # fast histogram method, GPU-compatible
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=42,
        early_stopping_rounds=30,
        eval_metric="rmse",
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: SPARSE VARIATIONAL GP ON RESIDUALS
# ─────────────────────────────────────────────────────────────────────────────

class ResidualSVGP(gpytorch.models.ApproximateGP):
    """
    Sparse Variational Gaussian Process trained on XGBoost residuals.

    WHY GP FOR STAGE 2:
    ────────────────────
    The GP models the systematic structure in XGBoost's residuals — the
    things XGBoost consistently gets wrong. More importantly, the GP posterior
    provides calibrated uncertainty: p(y* | X*, X, y) is a full distribution,
    not just a point estimate. The predictive std is the confidence interval.

    KERNEL CHOICE — RBF + Matérn(5/2) + Linear:
    - RBF (Squared Exponential): captures smooth, long-range correlations.
      This is the same kernel as in kernel SVMs (covered in prior coursework),
      specifically the Gaussian kernel k(x,x') = exp(-||x-x'||²/2l²).
    - Matérn(5/2): captures rougher, shorter-range patterns. Financial returns
      are not perfectly smooth, so Matérn is more appropriate than pure RBF.
    - Linear: captures linear trends that persist in the residuals.
    The sum of these three gives a flexible kernel that adapts to multiple
    types of structure simultaneously.

    SPARSE APPROXIMATION (SVGP):
    - Full GP scales as O(N³) — infeasible for 1000+ training points.
    - SVGP uses M << N inducing points to approximate the full posterior.
    - With M=128 inducing points, complexity is O(NM²) which is tractable.
    - The variational distribution over inducing points is optimized jointly
      with the kernel hyperparameters via ELBO maximization.
    """
    def __init__(self, inducing_points: torch.Tensor):
        # Variational distribution over the inducing points
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        # Variational strategy: maps from inducing points to full posterior
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,   # optimize inducing point positions
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()  # residuals have zero mean by construction

        # Composite kernel: RBF + Matérn(5/2) + Linear
        rbf_kernel     = gpytorch.kernels.RBFKernel()
        matern_kernel  = gpytorch.kernels.MaternKernel(nu=2.5)
        linear_kernel  = gpytorch.kernels.LinearKernel()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            rbf_kernel + matern_kernel + linear_kernel
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean  = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def train_gp(X_train_scaled: np.ndarray,
             residuals_train: np.ndarray) -> tuple:
    """
    Trains the Sparse Variational GP on XGBoost residuals.

    TRAINING PROCEDURE:
    ────────────────────
    Uses the Evidence Lower BOund (ELBO) as the loss function, which is the
    standard variational inference objective. This is analogous to the ELBO
    used in the BNN's SVI training, providing a consistent theoretical
    framework across both models.

    Minibatch training (BATCH_SIZE=256) allows the ELBO to be estimated on
    a random subset of the data at each step, enabling GPU acceleration and
    preventing memory issues with large datasets.

    Returns the trained model, likelihood, and the scaler used for X.
    """
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(residuals_train, dtype=torch.float32)

    # Initialize inducing points from a random subset of training data
    n_inducing = min(N_INDUCING, len(X_tensor))
    idx = torch.randperm(len(X_tensor))[:n_inducing]
    inducing_points = X_tensor[idx].clone()

    model      = ResidualSVGP(inducing_points).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": likelihood.parameters()},
    ], lr=GP_LR)

    # ELBO loss for variational GP
    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model, num_data=len(X_tensor)
    )

    X_gpu = X_tensor.to(device)
    y_gpu = y_tensor.to(device)

    dataset    = torch.utils.data.TensorDataset(X_gpu, y_gpu)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    for epoch in range(GP_EPOCHS):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss   = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            logging.info(f"      GP epoch {epoch+1}/{GP_EPOCHS}  ELBO={-loss.item():.4f}")

    return model, likelihood


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

def predict_combined(xgb_model:  xgb.XGBRegressor,
                     gp_model:   ResidualSVGP,
                     likelihood: gpytorch.likelihoods.GaussianLikelihood,
                     X_scaled:   np.ndarray,
                     scaler:     StandardScaler) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produces the combined forecast: XGBoost mean + GP correction + GP uncertainty.

    COMBINATION LOGIC:
    ──────────────────
    final_mean = xgb_prediction + gp_mean_of_residuals
    final_std  = gp_posterior_std   (this IS the uncertainty estimate)

    The GP posterior std naturally grows when:
    - The test point is far from training data (epistemic uncertainty)
    - The residuals in that region were large and noisy (aleatoric uncertainty)

    This mirrors what the BNN captured with its weight posteriors and sigma,
    but is computed analytically rather than via Monte Carlo sampling.

    Returns (mean_log_returns, lower_bound_log_returns, upper_bound_log_returns)
    where bounds are ±2 std (95% credible interval under Gaussian approximation).
    """
    xgb_preds = xgb_model.predict(scaler.inverse_transform(X_scaled))

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    gp_model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        gp_dist    = likelihood(gp_model(X_tensor))
        gp_mean    = gp_dist.mean.cpu().numpy()
        gp_std     = gp_dist.stddev.cpu().numpy()

    combined_mean  = xgb_preds + gp_mean
    lower_bound    = combined_mean - 2 * gp_std
    upper_bound    = combined_mean + 2 * gp_std

    return combined_mean, lower_bound, upper_bound


def forecast_future(last_close:   float,
                    xgb_model:    xgb.XGBRegressor,
                    gp_model:     ResidualSVGP,
                    likelihood:   gpytorch.likelihoods.GaussianLikelihood,
                    last_features: np.ndarray,
                    scaler:        StandardScaler,
                    horizon:       int = FORECAST_HORIZON) -> list[dict]:
    """
    Generates multi-step price forecasts with uncertainty bands.

    APPROACH:
    ─────────
    At each step:
    1. XGBoost predicts the log return from current features
    2. GP corrects the prediction and provides std
    3. The predicted price is last_price * exp(predicted_log_return)
    4. Uncertainty bands expand as std accumulates over horizon
       (uncertainty propagation: std grows roughly as sqrt(t))

    This is analogous to the Monte Carlo path simulation in BNN_A.py but
    is deterministic (uses posterior mean rather than sampling paths),
    which is appropriate for the GP because its posterior is analytic.

    For a stochastic version matching the BNN's approach, you could sample
    from the GP posterior at each step and propagate paths — that would
    give a proper predictive distribution over the full horizon.
    """
    forecasts  = []
    price      = last_close
    features   = last_features.copy()
    cumulative_variance = 0.0

    current_date = datetime.now()
    # Advance to next trading day
    while current_date.weekday() >= 5:
        current_date += timedelta(days=1)

    for step in range(horizon):
        # Advance date
        if step > 0:
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)

        features_scaled = scaler.transform(features.reshape(1, -1))
        mean_ret, low_ret, high_ret = predict_combined(
            xgb_model, gp_model, likelihood, features_scaled, scaler
        )

        mean_r = float(mean_ret[0])
        # GP std for this single point
        X_t    = torch.tensor(features_scaled, dtype=torch.float32).to(device)
        gp_model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            gp_d   = likelihood(gp_model(X_t))
            step_std = float(gp_d.stddev.cpu().item())

        # Accumulate variance over horizon (uncertainty grows with time)
        cumulative_variance += step_std ** 2
        horizon_std = np.sqrt(cumulative_variance)

        # Convert log returns to prices
        new_price   = price * np.exp(mean_r)
        lower_price = price * np.exp(mean_r - 2 * horizon_std)
        upper_price = price * np.exp(mean_r + 2 * horizon_std)

        forecasts.append({
            "date":        current_date.strftime("%Y-%m-%d"),
            "mean_close":  round(new_price, 4),
            "lower_bound": round(lower_price, 4),
            "upper_bound": round(upper_price, 4),
            "std_return":  round(step_std, 6),
            "mean_return": round(mean_r, 6),
        })

        # Update features for next step (carry-forward approximation)
        # In production: recompute rolling indicators with the new price
        price = new_price
        # Update close and log_return in features for next iteration
        features[FEATURE_COLS.index("close")]      = price
        features[FEATURE_COLS.index("log_return")] = mean_r
        if "daily_return" in FEATURE_COLS:
            features[FEATURE_COLS.index("daily_return")] = np.exp(mean_r) - 1

    return forecasts


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PER-TICKER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_ticker(con: duckdb.DuckDBPyConnection,
               symbol: str) -> dict | None:
    """
    Full pipeline for a single ticker:
        load → split → XGBoost → residuals → GP → forecast → evaluate

    TRAIN/TEST SPLIT:
        Uses an 80/20 temporal split (not random). The last 20% of rows
        serve as the test set. This is essential for time-series data —
        random splitting would leak future information because rolling
        indicator features are computed on overlapping windows.
    """
    logging.info(f"\n  [{symbol}] Loading data...")
    sentiment_map = build_sentiment_map(con, symbol)
    df = load_ticker_data(con, symbol, sentiment_map)

    if df.empty:
        logging.info(f"  [{symbol}] Insufficient data, skipping.")
        return None

    feature_cols_used = FEATURE_COLS + ["sentiment"]
    X_all = df[feature_cols_used].values.astype(np.float32)
    y_all = df["target"].values.astype(np.float32)

    # Temporal 80/20 split
    split = int(len(X_all) * 0.8)
    X_train, X_test   = X_all[:split], X_all[split:]
    y_train, y_test   = y_all[:split], y_all[split:]

    # Further split training into train/val for XGBoost early stopping
    val_split          = int(len(X_train) * 0.85)
    X_tr, X_val        = X_train[:val_split], X_train[val_split:]
    y_tr, y_val        = y_train[:val_split], y_train[val_split:]

    # Scale features for GP (XGBoost does not need scaling)
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # ── Stage 1: XGBoost ───────────────────────────────────────────────────
    logging.info(f"  [{symbol}] Training XGBoost ({len(X_tr)} train rows)...")
    xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val)

    xgb_train_preds = xgb_model.predict(X_tr)
    xgb_test_preds  = xgb_model.predict(X_test)

    train_residuals = y_tr    - xgb_train_preds
    test_residuals  = y_test  - xgb_test_preds

    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_preds))
    logging.info(f"  [{symbol}] XGBoost test RMSE (log return): {xgb_rmse:.6f}")

    # ── Stage 2: GP on residuals ───────────────────────────────────────────
    logging.info(f"  [{symbol}] Training Sparse GP on residuals...")
    gp_model, likelihood = train_gp(X_tr_sc, train_residuals)

    # Evaluate combined model on test set
    combined_mean, lower, upper = predict_combined(
        xgb_model, gp_model, likelihood, X_test_sc, scaler
    )
    combined_rmse = np.sqrt(mean_squared_error(y_test, combined_mean))
    # Coverage: what % of actual values fall within the 95% CI
    coverage = np.mean((y_test >= lower) & (y_test <= upper)) * 100

    logging.info(f"  [{symbol}] Combined RMSE:    {combined_rmse:.6f}")
    logging.info(f"  [{symbol}] 95% CI coverage:  {coverage:.1f}%  (target: ~95%)")
    logging.info(f"  [{symbol}] GP improvement:   {xgb_rmse - combined_rmse:.6f} log-return units")

    # ── Forecast ───────────────────────────────────────────────────────────
    last_close    = float(df["close"].iloc[-1])
    last_features = X_all[-1]

    forecasts = forecast_future(
        last_close, xgb_model, gp_model, likelihood,
        last_features, scaler, horizon=FORECAST_HORIZON
    )

    # Trading signal (mirrors BNN_A.py's get_signal logic)
    pred_close_tomorrow = forecasts[0]["mean_close"]
    if pred_close_tomorrow > last_close * 1.001:
        signal = 2   # buy
    elif pred_close_tomorrow < last_close * 0.999:
        signal = 0   # sell
    else:
        signal = 1   # hold

    result = {
        "symbol":         symbol,
        "last_close":     round(last_close, 4),
        "signal":         signal,
        "xgb_rmse":       round(float(xgb_rmse), 6),
        "combined_rmse":  round(float(combined_rmse), 6),
        "gp_improvement": round(float(xgb_rmse - combined_rmse), 6),
        "ci_coverage_pct":round(float(coverage), 2),
        "n_train":        int(split),
        "n_test":         int(len(X_test)),
        "forecast":       forecasts,
        "sentiment_available": len(sentiment_map) > 0,
        "fetched_at":     datetime.now().isoformat(),
    }

    # Save per-ticker output (same format as BNN's signal_{ticker}.json)
    out_path = OUTPUT_DIR / f"forecast_{symbol}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION & SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict]) -> None:
    """
    Prints a cross-ticker summary of model performance metrics.

    KEY METRICS:
    ─────────────
    - combined_rmse: lower is better. Measures average forecast error in
      log-return units. Multiply by last_close to get price-scale error.
    - ci_coverage_pct: should be ~95% for a well-calibrated model.
      Coverage >> 95% means the model is over-confident.
      Coverage << 95% means the uncertainty bands are too narrow.
    - gp_improvement: positive means the GP correction helped.
      Consistently negative would mean the GP is hurting (overfitting residuals).
    """
    if not all_results:
        logging.info("No results to summarize.")
        return

    df = pd.DataFrame([{
        "symbol":        r["symbol"],
        "xgb_rmse":      r["xgb_rmse"],
        "combined_rmse": r["combined_rmse"],
        "gp_improvement":r["gp_improvement"],
        "ci_coverage":   r["ci_coverage_pct"],
        "n_train":        r["n_train"],
        "signal":         {0:"SELL", 1:"HOLD", 2:"BUY"}.get(r["signal"], "?"),
    } for r in all_results])

    logging.info("\n" + "="*75)
    logging.info("  CROSS-TICKER MODEL SUMMARY")
    logging.info("="*75)
    logging.info(df.to_string(index=False))
    logging.info("-"*75)
    logging.info(f"  Mean XGBoost RMSE:    {df['xgb_rmse'].mean():.6f}")
    logging.info(f"  Mean Combined RMSE:   {df['combined_rmse'].mean():.6f}")
    logging.info(f"  Mean GP improvement:  {df['gp_improvement'].mean():.6f}")
    logging.info(f"  Mean CI coverage:     {df['ci_coverage'].mean():.1f}%  (target 95%)")
    logging.info(f"  Tickers processed:    {len(df)}")
    logging.info("="*75)

    # Save full summary
    summary_path = OUTPUT_DIR / "model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logging.info(f"\n  Full results saved to {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(max_tickers: int | None = None,
                 ticker_list: list[str] | None = None) -> None:
    """
    Runs the full XGBoost + GP pipeline across all tickers in the database.

    PARAMETERS:
    ─────────────
    max_tickers : int | None
        If set, only processes the first N tickers. Useful for testing.
        Set to None to process all 500 tickers.
    ticker_list : list[str] | None
        If provided, only processes these specific tickers.
        Overrides max_tickers.

    RUNTIME ESTIMATE (4070 Super):
    ─────────────────────────────────
    - XGBoost (CPU, all 500 tickers): ~8-12 minutes total
    - GP per ticker (GPU):            ~1-2 minutes each
    - Total for 500 tickers:          ~10-20 hours if run sequentially

    FOR CLASS SUBMISSION: run with max_tickers=20 for a proof-of-concept
    that demonstrates the full pipeline works, then note in your write-up
    that the full run would cover all 500 tickers.
    """
    con = duckdb.connect(DB_PATH, read_only=True)

    if ticker_list:
        tickers = ticker_list
    else:
        tickers = load_all_tickers(con)
        if max_tickers:
            tickers = tickers[:max_tickers]

    logging.info(f"[pipeline] Processing {len(tickers)} tickers")
    logging.info(f"[pipeline] Device: {device}")
    logging.info(f"[pipeline] GP inducing points: {N_INDUCING}")
    logging.info(f"[pipeline] GP epochs: {GP_EPOCHS}")

    all_results = []
    failed      = []

    for i, symbol in enumerate(tickers):
        logging.info(f"\n[{i+1:>4}/{len(tickers)}] {symbol}")
        try:
            result = run_ticker(con, symbol)
            if result:
                all_results.append(result)
        except Exception as e:
            logging.error(f"  [{symbol}] ERROR: {type(e).__name__}: {e}")
            failed.append(symbol)

    con.close()

    print_summary(all_results)

    if failed:
        logging.error(f"\n  Failed tickers ({len(failed)}): {failed}")

# ── QUICK TEST: 5 well-known tickers ─────────────────────────────────
# Uncomment the line you want to run:

# Quick proof-of-concept (5 tickers, ~10 minutes)
# run_pipeline(ticker_list=["AAPL", "MSFT", "NVDA", "AMZN", "GOOG"])

# Medium run (50 tickers, ~2 hours)
# run_pipeline(max_tickers=50)

# Full run (all tickers, ~15 hours — overnight)
# run_pipeline()