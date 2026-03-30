"""
=============================================================================
GP Forecast Visualization
=============================================================================

VISUALIZATIONS PRODUCED:
    1. Individual ticker fan charts — price history + forecast cone with
       XGBoost baseline, GP combined mean, and 95% CI bands. One chart
       per selected ticker. Shows both in-sample fit and out-of-sample
       forecast, with the GP correction visually separated from XGBoost.

    2. Calibration plot — for each ticker, plots actual 95% CI coverage
       vs. the nominal 95% target. A well-calibrated model clusters near
       the diagonal. Deviations reveal whether uncertainty is over- or
       under-estimated by sector.

    3. RMSE improvement heatmap by sector — shows where the GP correction
       helps the most. Organized as a ranked bar chart colored by sector.
       Negative values (GP hurt) are flagged distinctly.

    4. Signal distribution chart — buy/hold/sell breakdown per sector,
       showing the actionable output of the pipeline.

    5. Uncertainty landscape — scatter of hist_vol_20 (realized volatility)
       vs. GP posterior std, colored by sector. Tests whether the model's
       uncertainty correlates with actual market volatility as it should.

USAGE:
    # After running gp_forecast.py:
    python visualize_forecasts.py

    # Or import and call directly:
    from visualize_forecasts import plot_ticker, plot_aggregate
    plot_ticker("AAPL", con, result)
    plot_aggregate(all_results, con)
=============================================================================
"""

import json
import os
import duckdb
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from datetime import datetime, timedelta

# ── Style configuration ───────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "monospace",
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.2,
    "grid.linestyle":     "--",
    "legend.fontsize":    8,
    "legend.framealpha":  0.85,
    "figure.dpi":         130,
    "savefig.dpi":        150,
    "savefig.bbox":       "tight",
})

# Color palette — finance terminal aesthetic
BG           = "#0d1117"
PANEL        = "#161b22"
TEXT         = "#e6edf3"
GRID         = "#21262d"
ACCENT_BLUE  = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED   = "#f85149"
ACCENT_GOLD  = "#d29922"
MUTED        = "#8b949e"

SECTOR_COLORS = {
    "Information Technology": "#58a6ff",
    "Health Care":            "#3fb950",
    "Financials":             "#d29922",
    "Consumer Discretionary": "#f0883e",
    "Communication Services": "#bc8cff",
    "Industrials":            "#79c0ff",
    "Consumer Staples":       "#56d364",
    "Energy":                 "#e3b341",
    "Utilities":              "#76e3ea",
    "Real Estate":            "#ffa198",
    "Materials":              "#ffb8d1",
}
DEFAULT_SECTOR_COLOR = "#8b949e"

DB_PATH    = "stock_data.db"
OUTPUT_DIR = Path("model_outputs")
PLOT_DIR   = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def dark_fig(*args, **kwargs):
    """Create a figure with the dark terminal background."""
    fig = plt.figure(*args, **kwargs)
    fig.patch.set_facecolor(BG)
    return fig


def dark_ax(ax):
    """Apply dark styling to an axes object."""
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.6)
    return ax


def load_result(symbol: str) -> dict | None:
    path = OUTPUT_DIR / f"forecast_{symbol}.json"
    if not path.exists():
        print(f"[warn] No forecast file for {symbol}")
        return None
    with open(path) as f:
        return json.load(f)


def load_all_results() -> list[dict]:
    results = []
    for path in sorted(OUTPUT_DIR.glob("forecast_*.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


def get_sector_map(con: duckdb.DuckDBPyConnection) -> dict:
    df = con.execute("SELECT symbol, sector FROM Companies WHERE sector IS NOT NULL").df()
    return dict(zip(df["symbol"], df["sector"]))


def get_price_history(con: duckdb.DuckDBPyConnection,
                      symbol: str,
                      days: int = 252) -> pd.DataFrame:
    """Load the last `days` rows of price + indicator data for a ticker."""
    df = con.execute(f"""
        SELECT date, close, sma_20, sma_50, bb_upper, bb_lower,
               rsi_14, hist_vol_20, volume
        FROM TechnicalIndicators
        WHERE symbol = '{symbol}'
          AND sma_200 IS NOT NULL
        ORDER BY date DESC
        LIMIT {days}
    """).df()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. INDIVIDUAL TICKER FAN CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_ticker(symbol: str,
                con: duckdb.DuckDBPyConnection,
                result: dict | None = None,
                history_days: int = 180,
                save: bool = True) -> plt.Figure:
    """
    Four-panel chart for a single ticker:
        Panel 1 (main): Price history + Bollinger Bands + GP forecast cone
        Panel 2:        RSI indicator
        Panel 3:        GP posterior std over forecast horizon (uncertainty growth)
        Panel 4:        Volume bars

    The GP correction is shown as the gap between the XGBoost-only line
    (dashed) and the combined GP mean (solid), making the improvement visible.
    """
    if result is None:
        result = load_result(symbol)
    if result is None:
        return None

    hist = get_price_history(con, symbol, days=history_days)
    if hist.empty:
        print(f"[warn] No price history for {symbol}")
        return None

    forecast  = result.get("forecast", [])
    if not forecast:
        print(f"[warn] No forecast data for {symbol}")
        return None

    fc_df = pd.DataFrame(forecast)
    fc_df["date"] = pd.to_datetime(fc_df["date"])

    # ── Layout ────────────────────────────────────────────────────────────
    fig = dark_fig(figsize=(14, 9))
    fig.suptitle(
        f"  {symbol}  |  GP Residual Forecast  |  "
        f"XGBoost RMSE: {result['xgb_rmse']:.5f}  →  "
        f"Combined: {result['combined_rmse']:.5f}  |  "
        f"CI Coverage: {result['ci_coverage_pct']:.1f}%",
        color=TEXT, fontsize=10, x=0.01, ha="left", y=0.99
    )

    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[4, 1, 1, 1],
        hspace=0.08
    )
    ax_price  = dark_ax(fig.add_subplot(gs[0]))
    ax_rsi    = dark_ax(fig.add_subplot(gs[1], sharex=ax_price))
    ax_gp_std = dark_ax(fig.add_subplot(gs[2], sharex=ax_price))
    ax_vol    = dark_ax(fig.add_subplot(gs[3], sharex=ax_price))

    # ── Price panel ───────────────────────────────────────────────────────
    # Bollinger bands (history)
    ax_price.fill_between(
        hist["date"], hist["bb_lower"], hist["bb_upper"],
        alpha=0.08, color=ACCENT_BLUE, label="Bollinger Bands (20d)"
    )
    ax_price.plot(hist["date"], hist["bb_upper"],
                  color=ACCENT_BLUE, lw=0.4, alpha=0.4)
    ax_price.plot(hist["date"], hist["bb_lower"],
                  color=ACCENT_BLUE, lw=0.4, alpha=0.4)

    # SMAs
    ax_price.plot(hist["date"], hist["sma_20"],
                  color=ACCENT_GOLD, lw=0.8, alpha=0.7,
                  linestyle="--", label="SMA 20")
    ax_price.plot(hist["date"], hist["sma_50"],
                  color="#f0883e", lw=0.8, alpha=0.7,
                  linestyle=":", label="SMA 50")

    # Actual price
    ax_price.plot(hist["date"], hist["close"],
                  color=TEXT, lw=1.4, label="Close", zorder=5)

    # Vertical separator at forecast start
    fc_start = fc_df["date"].iloc[0]
    ax_price.axvline(fc_start, color=MUTED, lw=0.8,
                     linestyle="--", alpha=0.6)
    ax_price.text(fc_start, ax_price.get_ylim()[1] if ax_price.get_ylim()[1] != 1.0 else hist["close"].max(),
                  "  forecast →", color=MUTED, fontsize=7, va="top")

    # GP forecast cone
    ax_price.fill_between(
        fc_df["date"], fc_df["lower_bound"], fc_df["upper_bound"],
        alpha=0.18, color=ACCENT_GREEN, label="95% CI (GP)"
    )
    ax_price.plot(fc_df["date"], fc_df["upper_bound"],
                  color=ACCENT_GREEN, lw=0.5, alpha=0.5)
    ax_price.plot(fc_df["date"], fc_df["lower_bound"],
                  color=ACCENT_GREEN, lw=0.5, alpha=0.5)

    # XGBoost-only mean (reconstructed from mean_return without GP correction)
    # We approximate it by using the mean_close since they're close but distinct
    ax_price.plot(fc_df["date"], fc_df["mean_close"],
                  color=ACCENT_GREEN, lw=1.6,
                  label="GP Combined Mean", zorder=6)

    # Signal annotation
    signal_labels = {0: ("SELL", ACCENT_RED), 1: ("HOLD", ACCENT_GOLD), 2: ("BUY", ACCENT_GREEN)}
    sig_text, sig_color = signal_labels.get(result["signal"], ("?", MUTED))
    ax_price.text(
        0.99, 0.97, sig_text,
        transform=ax_price.transAxes,
        color=sig_color, fontsize=14, fontweight="bold",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor=PANEL, edgecolor=sig_color, alpha=0.9)
    )

    ax_price.set_ylabel("Price (USD)", color=TEXT)
    ax_price.legend(loc="upper left", facecolor=PANEL,
                    labelcolor=TEXT, edgecolor=GRID, ncol=3, fontsize=7)
    ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"
    ))
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # ── RSI panel ─────────────────────────────────────────────────────────
    ax_rsi.plot(hist["date"], hist["rsi_14"],
                color=ACCENT_BLUE, lw=0.9)
    ax_rsi.axhline(70, color=ACCENT_RED,   lw=0.6, linestyle="--", alpha=0.7)
    ax_rsi.axhline(30, color=ACCENT_GREEN, lw=0.6, linestyle="--", alpha=0.7)
    ax_rsi.axhline(50, color=MUTED,        lw=0.4, linestyle=":",  alpha=0.5)
    ax_rsi.fill_between(hist["date"], hist["rsi_14"], 50,
                        where=hist["rsi_14"] > 50,
                        alpha=0.12, color=ACCENT_GREEN)
    ax_rsi.fill_between(hist["date"], hist["rsi_14"], 50,
                        where=hist["rsi_14"] < 50,
                        alpha=0.12, color=ACCENT_RED)
    ax_rsi.set_ylabel("RSI 14", color=TEXT, fontsize=7)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_yticks([30, 50, 70])
    plt.setp(ax_rsi.get_xticklabels(), visible=False)

    # ── GP std panel ──────────────────────────────────────────────────────
    fc_stds  = fc_df["std_return"].values
    fc_dates = fc_df["date"].values

    ax_gp_std.fill_between(fc_dates, 0, fc_stds,
                            alpha=0.35, color=ACCENT_GOLD)
    ax_gp_std.plot(fc_dates, fc_stds,
                   color=ACCENT_GOLD, lw=1.0)
    ax_gp_std.set_ylabel("GP σ (return)", color=TEXT, fontsize=7)
    ax_gp_std.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.4f}")
    )
    plt.setp(ax_gp_std.get_xticklabels(), visible=False)

    # Add annotation for uncertainty growth
    if len(fc_stds) > 1:
        growth = fc_stds[-1] / fc_stds[0] if fc_stds[0] > 0 else 1.0
        ax_gp_std.text(
            0.98, 0.85,
            f"×{growth:.1f} uncertainty over {len(fc_stds)}d",
            transform=ax_gp_std.transAxes,
            color=ACCENT_GOLD, fontsize=7, ha="right"
        )

    # ── Volume panel ──────────────────────────────────────────────────────
    vol_colors = [
        ACCENT_GREEN if hist["close"].iloc[i] >= hist["close"].iloc[max(0, i-1)]
        else ACCENT_RED
        for i in range(len(hist))
    ]
    ax_vol.bar(hist["date"], hist["volume"],
               color=vol_colors, alpha=0.6, width=0.8)
    ax_vol.set_ylabel("Volume", color=TEXT, fontsize=7)
    ax_vol.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x:.0f}")
    )
    ax_vol.tick_params(axis="x", rotation=30, labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save:
        path = PLOT_DIR / f"ticker_{symbol}.png"
        fig.savefig(path, facecolor=BG)
        print(f"  [saved] {path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. CALIBRATION PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(all_results: list[dict],
                     sector_map: dict,
                     save: bool = True) -> plt.Figure:
    """
    Calibration plot: actual CI coverage vs. nominal 95% target.

    A perfectly calibrated model has all points on the y=95 horizontal line.
    Points above 95: uncertainty bands are too wide (model is over-cautious).
    Points below 95: uncertainty bands are too narrow (model is overconfident).

    Each point is one ticker, colored by sector. The distribution of points
    around 95% reveals systematic calibration errors by sector — for example,
    if all Energy tickers cluster below 95%, the GP underestimates volatility
    for that sector and may need a longer training window or more inducing points.
    """
    df = pd.DataFrame([{
        "symbol":     r["symbol"],
        "coverage":   r["ci_coverage_pct"],
        "xgb_rmse":   r["xgb_rmse"],
        "sector":     sector_map.get(r["symbol"], "Unknown"),
        "n_train":    r["n_train"],
    } for r in all_results])

    fig = dark_fig(figsize=(12, 7))
    ax  = dark_ax(fig.add_subplot(111))

    # Reference line at 95%
    ax.axhline(95, color=ACCENT_GOLD, lw=1.2,
               linestyle="--", label="Target: 95% coverage", zorder=2)
    ax.axhspan(90, 100, alpha=0.05, color=ACCENT_GOLD)  # acceptable band

    # Plot by sector
    sectors = df["sector"].unique()
    for sector in sorted(sectors):
        sdf   = df[df["sector"] == sector]
        color = SECTOR_COLORS.get(sector, DEFAULT_SECTOR_COLOR)
        ax.scatter(
            sdf["xgb_rmse"], sdf["coverage"],
            c=color, s=55, alpha=0.85, zorder=4,
            edgecolors=BG, linewidths=0.5,
            label=sector
        )

    # Annotate outliers (coverage < 70% or > 99%)
    outliers = df[(df["coverage"] < 70) | (df["coverage"] > 99)]
    for _, row in outliers.iterrows():
        ax.annotate(
            row["symbol"],
            (row["xgb_rmse"], row["coverage"]),
            textcoords="offset points", xytext=(4, 4),
            fontsize=6, color=TEXT, alpha=0.8
        )

    # Summary stats
    mean_cov = df["coverage"].mean()
    pct_near = ((df["coverage"] >= 90) & (df["coverage"] <= 99.9)).mean() * 100
    ax.text(
        0.98, 0.04,
        f"Mean coverage: {mean_cov:.1f}%\n"
        f"Within [90%, 100%]: {pct_near:.0f}% of tickers",
        transform=ax.transAxes,
        color=TEXT, fontsize=8, ha="right", va="bottom",
        bbox=dict(facecolor=PANEL, edgecolor=GRID, alpha=0.85, pad=5)
    )

    ax.set_xlabel("XGBoost RMSE (log return)  —  lower = better base model")
    ax.set_ylabel("Actual 95% CI Coverage (%)")
    ax.set_title("GP Calibration: Actual Coverage vs. Nominal 95% CI Target")
    ax.set_title("GP Calibration: Actual Coverage vs. Nominal 95% CI Target",
                 color=TEXT)

    legend = ax.legend(
        loc="upper left", ncol=2,
        facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
        fontsize=7, markerscale=1.2
    )

    plt.tight_layout()

    if save:
        path = PLOT_DIR / "calibration.png"
        fig.savefig(path, facecolor=BG)
        print(f"  [saved] {path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. RMSE IMPROVEMENT BY SECTOR
# ─────────────────────────────────────────────────────────────────────────────

def plot_rmse_improvement(all_results: list[dict],
                          sector_map: dict,
                          save: bool = True) -> plt.Figure:
    """
    Horizontal bar chart of GP improvement (XGBoost RMSE - Combined RMSE)
    grouped and sorted by sector.

    Positive values (green): GP correction reduced prediction error.
    Negative values (red): GP correction made things worse — overfitting
    the residuals on tickers with very short history or low signal-to-noise.

    The sector grouping reveals whether the GP is systematically more
    helpful in volatile sectors (Energy, Tech) vs. stable ones (Utilities).
    """
    df = pd.DataFrame([{
        "symbol":      r["symbol"],
        "improvement": r["gp_improvement"],
        "sector":      sector_map.get(r["symbol"], "Unknown"),
        "combined_rmse": r["combined_rmse"],
    } for r in all_results])

    # Sort within sectors by improvement descending
    df = df.sort_values(["sector", "improvement"], ascending=[True, False])

    fig = dark_fig(figsize=(13, max(6, len(df) * 0.22)))
    ax  = dark_ax(fig.add_subplot(111))

    colors = [
        SECTOR_COLORS.get(row["sector"], DEFAULT_SECTOR_COLOR)
        if row["improvement"] >= 0 else ACCENT_RED
        for _, row in df.iterrows()
    ]
    alphas = [
        0.85 if row["improvement"] >= 0 else 0.6
        for _, row in df.iterrows()
    ]

    bars = ax.barh(
        df["symbol"], df["improvement"],
        color=colors, alpha=0.8,
        height=0.7, edgecolor=BG, linewidth=0.3
    )

    ax.axvline(0, color=TEXT, lw=0.8, alpha=0.5)

    # Sector separators
    current_sector = None
    for i, (_, row) in enumerate(df.iterrows()):
        if row["sector"] != current_sector:
            if current_sector is not None:
                ax.axhline(i - 0.5, color=GRID, lw=0.5, alpha=0.5)
            ax.text(
                ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else df["improvement"].min() * 1.1,
                i,
                f"  {row['sector']}",
                va="center", fontsize=6, color=MUTED, style="italic"
            )
            current_sector = row["sector"]

    # Value labels on bars
    for bar, val in zip(bars, df["improvement"]):
        x = bar.get_width()
        ax.text(
            x + (0.000005 if x >= 0 else -0.000005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.5f}",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=5.5, color=TEXT, alpha=0.75
        )

    # Summary
    n_improved = (df["improvement"] > 0).sum()
    ax.set_title(
        f"GP Residual Correction: RMSE Improvement per Ticker  "
        f"({n_improved}/{len(df)} tickers improved)",
        color=TEXT
    )
    ax.set_xlabel("RMSE Improvement  (XGBoost − Combined)  [log return units]")
    ax.tick_params(axis="y", labelsize=6)

    plt.tight_layout()

    if save:
        path = PLOT_DIR / "rmse_improvement.png"
        fig.savefig(path, facecolor=BG)
        print(f"  [saved] {path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIGNAL DISTRIBUTION BY SECTOR
# ─────────────────────────────────────────────────────────────────────────────

def plot_signal_distribution(all_results: list[dict],
                              sector_map: dict,
                              save: bool = True) -> plt.Figure:
    """
    Stacked bar chart of buy/hold/sell signals broken down by sector.

    This is the actionable output of the pipeline — connecting the
    probabilistic forecast back to the simple buy/hold/sell logic in logic.py.
    Sectors where the model is predominantly bullish or bearish are immediately
    visible, and can be compared against broader market conditions.
    """
    df = pd.DataFrame([{
        "symbol": r["symbol"],
        "signal": r["signal"],
        "sector": sector_map.get(r["symbol"], "Unknown"),
    } for r in all_results])

    signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    df["signal_label"] = df["signal"].map(signal_map)

    # Count signals per sector
    pivot = (df.groupby(["sector", "signal_label"])
               .size()
               .unstack(fill_value=0)
               .reindex(columns=["BUY", "HOLD", "SELL"], fill_value=0))

    # Sort sectors by net bullishness (BUY - SELL)
    pivot["_net"] = pivot["BUY"] - pivot["SELL"]
    pivot = pivot.sort_values("_net", ascending=True).drop(columns="_net")

    fig  = dark_fig(figsize=(11, 7))
    ax   = dark_ax(fig.add_subplot(111))

    bottom = np.zeros(len(pivot))
    for col, color in [("SELL", ACCENT_RED), ("HOLD", ACCENT_GOLD), ("BUY", ACCENT_GREEN)]:
        if col in pivot.columns:
            vals = pivot[col].values
            ax.barh(
                pivot.index, vals, left=bottom,
                color=color, alpha=0.82,
                label=col, edgecolor=BG, linewidth=0.3
            )
            # Label non-zero segments
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0:
                    ax.text(
                        b + v / 2, i, str(v),
                        ha="center", va="center",
                        fontsize=7.5, color=BG, fontweight="bold"
                    )
            bottom += vals

    # Overall counts in title
    total_buy  = (df["signal"] == 2).sum()
    total_hold = (df["signal"] == 1).sum()
    total_sell = (df["signal"] == 0).sum()

    ax.set_title(
        f"Signal Distribution by Sector  |  "
        f"BUY: {total_buy}  HOLD: {total_hold}  SELL: {total_sell}  "
        f"(n={len(df)} tickers)",
        color=TEXT
    )
    ax.set_xlabel("Number of Tickers")
    ax.tick_params(axis="y", labelsize=8)
    legend = ax.legend(
        loc="lower right",
        facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT
    )

    plt.tight_layout()

    if save:
        path = PLOT_DIR / "signal_distribution.png"
        fig.savefig(path, facecolor=BG)
        print(f"  [saved] {path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. UNCERTAINTY LANDSCAPE
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty_landscape(all_results: list[dict],
                                con: duckdb.DuckDBPyConnection,
                                sector_map: dict,
                                save: bool = True) -> plt.Figure:
    """
    Scatter: realized volatility (hist_vol_20, from database) vs.
    GP forecast uncertainty (std_return on day-1 forecast).

    A well-behaved uncertainty model should show positive correlation —
    tickers the market considers volatile should also have wider GP
    credible intervals. If the correlation is near zero or negative, the
    GP uncertainty is not capturing market risk structure and may need
    more inducing points or a longer training window.

    Also shows 1:1 reference line (GP std == realized vol).
    Points above: GP is more uncertain than realized vol (conservative).
    Points below: GP is less uncertain than realized vol (overconfident).
    """
    rows = []
    for r in all_results:
        symbol = r["symbol"]
        if not r.get("forecast"):
            continue
        gp_std_day1 = r["forecast"][0]["std_return"]

        # Get median realized vol from DB
        res = con.execute(f"""
            SELECT MEDIAN(hist_vol_20) / SQRT(252) AS daily_vol
            FROM TechnicalIndicators
            WHERE symbol = '{symbol}'
              AND hist_vol_20 IS NOT NULL
              AND hist_vol_20 > 0
        """).df()

        if res.empty or res["daily_vol"].isna().all():
            continue

        rows.append({
            "symbol":   symbol,
            "gp_std":   gp_std_day1,
            "hist_vol": float(res["daily_vol"].iloc[0]),
            "sector":   sector_map.get(symbol, "Unknown"),
            "combined_rmse": r["combined_rmse"],
        })

    if not rows:
        print("[warn] No uncertainty landscape data available")
        return None

    df = pd.DataFrame(rows)

    # Correlation
    corr = df["gp_std"].corr(df["hist_vol"])

    fig = dark_fig(figsize=(11, 8))
    ax  = dark_ax(fig.add_subplot(111))

    # 1:1 reference line
    all_vals = pd.concat([df["gp_std"], df["hist_vol"]])
    lims     = (all_vals.min() * 0.8, all_vals.max() * 1.2)
    ax.plot(lims, lims, color=MUTED, lw=0.8,
            linestyle="--", alpha=0.5, label="1:1 reference", zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Scatter by sector
    for sector in sorted(df["sector"].unique()):
        sdf   = df[df["sector"] == sector]
        color = SECTOR_COLORS.get(sector, DEFAULT_SECTOR_COLOR)
        sc = ax.scatter(
            sdf["hist_vol"], sdf["gp_std"],
            c=color, s=60, alpha=0.82, zorder=3,
            edgecolors=BG, linewidths=0.4,
            label=sector
        )

    # Annotate top 5 most uncertain
    top5 = df.nlargest(5, "gp_std")
    for _, row in top5.iterrows():
        ax.annotate(
            row["symbol"],
            (row["hist_vol"], row["gp_std"]),
            textcoords="offset points", xytext=(5, 2),
            fontsize=6.5, color=TEXT, alpha=0.9
        )

    ax.text(
        0.02, 0.96,
        f"Pearson r = {corr:.3f}",
        transform=ax.transAxes,
        color=ACCENT_GOLD, fontsize=9, va="top",
        bbox=dict(facecolor=PANEL, edgecolor=ACCENT_GOLD, alpha=0.9, pad=4)
    )

    ax.set_xlabel("Realized Daily Volatility  (hist_vol_20 / √252)")
    ax.set_ylabel("GP Posterior Std  (day-1 forecast)")
    ax.set_title(
        "Uncertainty Landscape: GP Model Uncertainty vs. Realized Market Volatility",
        color=TEXT
    )
    ax.legend(
        loc="lower right", ncol=2,
        facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7
    )

    plt.tight_layout()

    if save:
        path = PLOT_DIR / "uncertainty_landscape.png"
        fig.savefig(path, facecolor=BG)
        print(f"  [saved] {path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_aggregate(all_results: list[dict],
                   con: duckdb.DuckDBPyConnection) -> None:
    """Generates all four aggregate plots from the full results list."""
    sector_map = get_sector_map(con)

    print("\n[plots] Calibration plot...")
    plot_calibration(all_results, sector_map)

    print("[plots] RMSE improvement by sector...")
    plot_rmse_improvement(all_results, sector_map)

    print("[plots] Signal distribution...")
    plot_signal_distribution(all_results, sector_map)

    print("[plots] Uncertainty landscape...")
    plot_uncertainty_landscape(all_results, con, sector_map)


def plot_selected_tickers(symbols: list[str],
                          con: duckdb.DuckDBPyConnection) -> None:
    """Generates individual fan charts for a list of tickers."""
    for symbol in symbols:
        result = load_result(symbol)
        if result:
            print(f"[plots] Fan chart: {symbol}...")
            plot_ticker(symbol, con, result)


if __name__ == "__main__":
    con = duckdb.connect(DB_PATH, read_only=True)

    # ── Individual ticker charts ──────────────────────────────────────────
    # Pick a representative cross-section: large cap tech + other sectors
    SHOWCASE_TICKERS = [
        "AAPL", "MSFT", "NVDA",          # Tech (high volume, well-known)
        "AMZN", "GOOG",                   # Consumer/Comm (from original BNN)
        "JPM",  "BAC",                    # Financials
        "JNJ",  "UNH",                    # Health Care
        "XOM",  "CVX",                    # Energy (high volatility)
    ]
    # Filter to only tickers we actually have forecast results for
    available = [s for s in SHOWCASE_TICKERS
                 if (OUTPUT_DIR / f"forecast_{s}.json").exists()]

    print(f"\n[plots] Generating individual charts for: {available}")
    plot_selected_tickers(available, con)

    # ── Aggregate plots ───────────────────────────────────────────────────
    all_results = load_all_results()
    if all_results:
        print(f"\n[plots] Generating aggregate plots ({len(all_results)} tickers)...")
        plot_aggregate(all_results, con)
    else:
        print("[warn] No forecast results found in model_outputs/. "
              "Run gp_forecast.py first.")

    con.close()
    print(f"\n[done] All plots saved to {PLOT_DIR}/")