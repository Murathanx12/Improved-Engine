"""
Module 9: Visualization
========================

All chart creation functions for the Market Prediction Engine.
Each function returns a matplotlib Figure.

Usage:
    from finpredict.utils.charts import chart_projection, chart_crash_probability
"""

import numpy as np
from io import BytesIO
import warnings

# matplotlib is optional for headless environments; provide stubs if absent
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

    # create lightweight stubs that mimic minimal necessary interface
    class _DummyFig:
        def savefig(self, *args, **kwargs):
            pass

        def tight_layout(self, *args, **kwargs):
            pass

    class _DummyAx:
        def __getattr__(self, name):
            def _dummy(*args, **kwargs):
                return None

            return _dummy

    class _DummyPlt:
        class style:
            @staticmethod
            def use(arg):
                pass

        def subplots(self, *args, **kwargs):
            return _DummyFig(), _DummyAx()

        def close(self, fig=None):
            pass

    class _DummyDates:
        class DateFormatter:
            def __init__(self, fmt):
                pass

    plt = _DummyPlt()
    mdates = _DummyDates()

from finpredict.config import config

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
plt.style.use("dark_background")


# Lazy import for reportlab (only needed when generating PDFs)
def _get_inch():
    from reportlab.lib.units import inch

    return inch


CHART_COLORS = {
    "primary": "#FFD700",  # Gold
    "secondary": "#FF6B35",  # Orange
    "accent": "#00D4FF",  # Cyan
    "positive": "#00FF88",  # Green
    "negative": "#FF3366",  # Red
    "neutral": "#888888",
    "bg": "#0D1117",
    "text": "#E6EDF3",
}


def fig_to_image(fig, width=None, height=None):
    """Convert matplotlib figure to ReportLab Image."""
    from reportlab.lib.units import inch
    from reportlab.platypus import Image as RLImage

    if width is None:
        width = 7 * inch
    if height is None:
        height = 3.5 * inch
    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor=CHART_COLORS["bg"],
        edgecolor="none",
    )
    buf.seek(0)
    img = RLImage(buf, width=width, height=height)
    plt.close(fig)
    return img


def chart_backtest(data, bt_results):
    """Backtest validation chart: actual vs predicted S&P 500 since 2000."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=CHART_COLORS["bg"])
    ax.set_facecolor(CHART_COLORS["bg"])

    # Actual S&P 500
    bt_data = data[data.index >= config["data"]["backtest_start"]]
    ax.plot(
        bt_data.index,
        bt_data["SP500"],
        color=CHART_COLORS["accent"],
        linewidth=1.5,
        label="Actual S&P 500",
        zorder=3,
    )

    if len(bt_results) > 0:
        # Prediction bands
        ax.fill_between(
            bt_results["date"],
            bt_results["pred_p05"],
            bt_results["pred_p95"],
            alpha=0.25,
            color=CHART_COLORS["secondary"],
            label="90% Prediction Band",
        )
        ax.plot(
            bt_results["date"],
            bt_results["pred_mean"],
            "--",
            color=CHART_COLORS["primary"],
            linewidth=1,
            label="Model Prediction",
            alpha=0.8,
        )

    # 7% baseline
    baseline_start = float(bt_data["SP500"].iloc[0])
    days = (bt_data.index - bt_data.index[0]).days
    baseline = baseline_start * (1.07 ** (days / 365.25))
    ax.plot(
        bt_data.index,
        baseline,
        ":",
        color=CHART_COLORS["neutral"],
        linewidth=1,
        label="7% Annual Baseline",
        alpha=0.6,
    )

    ax.set_yscale("log")
    ax.set_ylabel("S&P 500 (log scale)", color=CHART_COLORS["text"])
    ax.set_title(
        "Backtest Validation: Model Accuracy (2000–Present)",
        color=CHART_COLORS["primary"],
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(
        loc="upper left",
        fontsize=8,
        facecolor=CHART_COLORS["bg"],
        edgecolor=CHART_COLORS["neutral"],
        labelcolor=CHART_COLORS["text"],
    )
    ax.tick_params(colors=CHART_COLORS["text"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def chart_projection(mc_results, current_price):
    """5-year Monte Carlo projection with confidence bands."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=CHART_COLORS["bg"])
    ax.set_facecolor(CHART_COLORS["bg"])

    days = np.arange(len(mc_results["mean_path"]))
    years = days / config["simulation"]["trading_days_per_year"]

    ax.fill_between(
        years,
        mc_results["p05"],
        mc_results["p95"],
        alpha=0.15,
        color=CHART_COLORS["secondary"],
        label="90% Confidence",
    )
    ax.fill_between(
        years,
        mc_results["p25"],
        mc_results["p75"],
        alpha=0.25,
        color=CHART_COLORS["primary"],
        label="50% Confidence",
    )
    ax.plot(
        years,
        mc_results["mean_path"],
        color=CHART_COLORS["primary"],
        linewidth=2,
        label=f"Mean: ${mc_results['final_mean']:,.0f}",
    )
    ax.plot(
        years,
        mc_results["median_path"],
        "--",
        color=CHART_COLORS["accent"],
        linewidth=1.5,
        label=f"Median: ${mc_results['final_median']:,.0f}",
    )

    # 7% baseline
    baseline = current_price * (1.07**years)
    ax.plot(
        years,
        baseline,
        ":",
        color=CHART_COLORS["neutral"],
        linewidth=1,
        label="7% Baseline",
        alpha=0.6,
    )

    ax.axhline(current_price, color=CHART_COLORS["text"], linestyle="-", alpha=0.3)
    ax.set_xlabel("Years", color=CHART_COLORS["text"])
    ax.set_ylabel("S&P 500 Level", color=CHART_COLORS["text"])
    ax.set_title(
        f'5-Year Market Projection ({config["simulation"]["num_simulations"]:,} Monte Carlo Simulations)',
        color=CHART_COLORS["primary"],
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(
        loc="upper left",
        fontsize=8,
        facecolor=CHART_COLORS["bg"],
        edgecolor=CHART_COLORS["neutral"],
        labelcolor=CHART_COLORS["text"],
    )
    ax.tick_params(colors=CHART_COLORS["text"])
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def chart_crash_probability(mc_results):
    """Crash probability by time horizon — the PRIMARY deliverable."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=CHART_COLORS["bg"])
    ax.set_facecolor(CHART_COLORS["bg"])

    labels = list(mc_results["crash_probs"].keys())
    values = list(mc_results["crash_probs"].values())

    bar_colors = []
    for v in values:
        if v < 15:
            bar_colors.append(CHART_COLORS["positive"])
        elif v < 30:
            bar_colors.append(CHART_COLORS["primary"])
        else:
            bar_colors.append(CHART_COLORS["negative"])

    bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.5, alpha=0.9)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            color=CHART_COLORS["text"],
            fontweight="bold",
            fontsize=10,
        )

    ax.set_ylabel("Crash Probability (%)", color=CHART_COLORS["text"])
    ax.set_xlabel("Time Horizon", color=CHART_COLORS["text"])
    ax.set_title(
        "Crash Probability by Time Horizon (≥20% Drawdown)",
        color=CHART_COLORS["primary"],
        fontsize=14,
        fontweight="bold",
    )
    ax.tick_params(colors=CHART_COLORS["text"])
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    return fig


def chart_risk_score(data):
    """Historical composite risk score with VIX overlay."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), facecolor=CHART_COLORS["bg"], gridspec_kw={"height_ratios": [1, 0.6]}
    )

    bt_data = data[data.index >= config["data"]["backtest_start"]]

    # Risk score
    ax1.set_facecolor(CHART_COLORS["bg"])
    risk = bt_data["Risk_Score"]
    ax1.fill_between(
        risk.index,
        0,
        risk,
        where=risk > 0,
        color=CHART_COLORS["negative"],
        alpha=0.4,
        label="High Risk",
    )
    ax1.fill_between(
        risk.index,
        0,
        risk,
        where=risk <= 0,
        color=CHART_COLORS["positive"],
        alpha=0.4,
        label="Low Risk",
    )
    ax1.axhline(0, color=CHART_COLORS["neutral"], linestyle="--", alpha=0.5)
    current_risk = float(risk.iloc[-1])
    ax1.plot(
        risk.index[-1], current_risk, "o", color=CHART_COLORS["primary"], markersize=8, zorder=5
    )
    ax1.set_ylabel("Risk Score (σ)", color=CHART_COLORS["text"])
    ax1.set_title(
        "Composite Risk Indicator", color=CHART_COLORS["primary"], fontsize=14, fontweight="bold"
    )
    ax1.legend(
        fontsize=8,
        facecolor=CHART_COLORS["bg"],
        edgecolor=CHART_COLORS["neutral"],
        labelcolor=CHART_COLORS["text"],
    )
    ax1.tick_params(colors=CHART_COLORS["text"])
    ax1.grid(True, alpha=0.15)

    # VIX
    if "VIX" in bt_data.columns:
        ax2.set_facecolor(CHART_COLORS["bg"])
        vix = bt_data["VIX"]
        ax2.fill_between(
            vix.index,
            0,
            vix,
            where=vix > 30,
            color=CHART_COLORS["negative"],
            alpha=0.3,
            label="Fear (VIX>30)",
        )
        ax2.plot(vix.index, vix, color=CHART_COLORS["secondary"], linewidth=0.8, label="VIX")
        ax2.axhline(20, color=CHART_COLORS["neutral"], linestyle="--", alpha=0.4)
        ax2.axhline(30, color=CHART_COLORS["negative"], linestyle="--", alpha=0.4)
        ax2.set_ylabel("VIX Level", color=CHART_COLORS["text"])
        ax2.set_xlabel("VIX (Fear Index)", color=CHART_COLORS["text"])
        ax2.tick_params(colors=CHART_COLORS["text"])
        ax2.legend(
            fontsize=8,
            facecolor=CHART_COLORS["bg"],
            edgecolor=CHART_COLORS["neutral"],
            labelcolor=CHART_COLORS["text"],
        )
        ax2.grid(True, alpha=0.15)

    fig.tight_layout()
    return fig


def chart_scenarios(mc_results, current_price):
    """Scenario analysis — each scenario as a separate projected path."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=CHART_COLORS["bg"])
    ax.set_facecolor(CHART_COLORS["bg"])

    scenario_colors = {
        "Base Case": "#4CAF50",
        "AI Productivity Boom": "#2196F3",
        "Soft Landing": "#00BCD4",
        "Market Correction": "#FF9800",
        "Stagflation": "#795548",
        "Recession": "#9C27B0",
        "AI Bubble Collapse": "#E91E63",
        "Geopolitical Crisis": "#F44336",
    }

    forecast_days = (
        config["simulation"]["forecast_years"] * config["simulation"]["trading_days_per_year"]
    )
    years = np.linspace(0, config["simulation"]["forecast_years"], forecast_days + 1)

    for name, info in mc_results["scenarios"].items():
        mu = info["return"]
        info["volatility"]
        # Expected path using continuous compounding (matches jump-diffusion process)
        # E[S(t)] = S(0) * exp(mu * t) — the drift already accounts for diffusion
        mean_path = current_price * np.exp(mu * years)
        c = scenario_colors.get(name, "#FFFFFF")
        prob = info["probability"]
        info["mean_final"]
        ret = info["total_return"]
        ax.plot(
            years, mean_path, color=c, linewidth=2, label=f"{name} ({prob*100:.0f}%) → {ret:+.1f}%"
        )

    # 7% baseline
    baseline = current_price * (1.07**years)
    ax.plot(
        years,
        baseline,
        ":",
        color=CHART_COLORS["neutral"],
        linewidth=1,
        label="7% Baseline",
        alpha=0.5,
    )

    ax.plot(0, current_price, "*", color=CHART_COLORS["primary"], markersize=12, zorder=5)

    ax.set_xlabel("Years", color=CHART_COLORS["text"])
    ax.set_ylabel("S&P 500 Level", color=CHART_COLORS["text"])
    ax.set_title(
        "Scenario Analysis: Alternative Futures",
        color=CHART_COLORS["primary"],
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(
        fontsize=8,
        facecolor=CHART_COLORS["bg"],
        edgecolor=CHART_COLORS["neutral"],
        labelcolor=CHART_COLORS["text"],
    )
    ax.tick_params(colors=CHART_COLORS["text"])
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def chart_sectors(sector_results, sp500_return):
    """Sector performance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=CHART_COLORS["bg"])
    ax.set_facecolor(CHART_COLORS["bg"])

    names = list(sector_results.keys())
    returns = [sector_results[n].get("expected_total", sector_results[n].get("expected_return", 0)) for n in names]

    # Sort by return
    sorted_pairs = sorted(zip(names, returns), key=lambda x: x[1])
    names, returns = zip(*sorted_pairs)

    bar_colors = [
        CHART_COLORS["positive"] if r > sp500_return else CHART_COLORS["secondary"] for r in returns
    ]

    bars = ax.barh(names, returns, color=bar_colors, edgecolor="white", linewidth=0.3, alpha=0.85)

    for bar, val in zip(bars, returns):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}%",
            va="center",
            color=CHART_COLORS["text"],
            fontsize=9,
        )

    ax.axvline(
        sp500_return,
        color=CHART_COLORS["primary"],
        linestyle="--",
        linewidth=1.5,
        label=f"S&P 500: {sp500_return:+.1f}%",
    )

    ax.set_xlabel("Expected 5-Year Return (%)", color=CHART_COLORS["text"])
    ax.set_title(
        "Sector Performance Projections",
        color=CHART_COLORS["primary"],
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(
        fontsize=9,
        facecolor=CHART_COLORS["bg"],
        edgecolor=CHART_COLORS["neutral"],
        labelcolor=CHART_COLORS["text"],
    )
    ax.tick_params(colors=CHART_COLORS["text"])
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    return fig


def chart_stocks(stock_results, sp500_return):
    """Individual stock expected return bar chart with cap tier annotations."""
    if not stock_results:
        return None

    fig, ax = plt.subplots(
        figsize=(10, max(4, len(stock_results) * 0.4)), facecolor=CHART_COLORS["bg"]
    )
    ax.set_facecolor(CHART_COLORS["bg"])

    # Sort by expected return
    sorted_stocks = sorted(stock_results.items(), key=lambda x: x[1]["expected_return"])
    tickers = [s[0] for s in sorted_stocks]
    returns = [s[1]["expected_return"] for s in sorted_stocks]

    bar_colors = [
        CHART_COLORS["positive"] if r > sp500_return else CHART_COLORS["secondary"] for r in returns
    ]

    bars = ax.barh(tickers, returns, color=bar_colors, edgecolor="white", linewidth=0.3, alpha=0.85)

    for bar, val, (tick, info) in zip(bars, returns, sorted_stocks):
        label = f'{val:+.1f}% ({info["cap_tier"]})'
        ax.text(
            max(bar.get_width() + 1, 2),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            color=CHART_COLORS["text"],
            fontsize=8,
        )

    ax.axvline(
        sp500_return,
        color=CHART_COLORS["primary"],
        linestyle="--",
        linewidth=1.5,
        label=f"S&P 500: {sp500_return:+.1f}%",
    )

    ax.set_xlabel("Expected 5-Year Return (%)", color=CHART_COLORS["text"])
    ax.set_title(
        "Individual Stock Projections (CAGR-Capped)",
        color=CHART_COLORS["primary"],
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(
        fontsize=9,
        facecolor=CHART_COLORS["bg"],
        edgecolor=CHART_COLORS["neutral"],
        labelcolor=CHART_COLORS["text"],
    )
    ax.tick_params(colors=CHART_COLORS["text"])
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    return fig


def chart_combined_projection_crash(mc_results, current_price):
    """
    COMBINED Projection + Crash Probability Chart.
    User's primary feature request: "I want the current market projection combined
    with the crash probability analysis... more realistic graph with ups and downs."

    Top panel: Monte Carlo fan chart with crash vs non-crash paths distinguished
    Bottom panel: Rolling crash probability by horizon with danger zones
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), facecolor=CHART_COLORS["bg"], gridspec_kw={"height_ratios": [2.5, 1]}
    )
    for ax in [ax1, ax2]:
        ax.set_facecolor(CHART_COLORS["bg"])

    all_paths = mc_results["all_paths"]
    n_days = all_paths.shape[0]
    n_sims = all_paths.shape[1]
    x_years = np.arange(n_days) / 252

    # --- TOP PANEL: Projection with crash/non-crash path coloring ---

    # Classify each path: did it experience a ≥20% peak-to-trough crash?
    running_peak = np.maximum.accumulate(all_paths, axis=0)
    drawdowns = (all_paths - running_peak) / running_peak
    max_dd_per_sim = drawdowns.min(axis=0)
    crash_mask = max_dd_per_sim <= -0.20  # Paths that experienced a crash

    # Plot a sample of paths (too many = visual noise)
    sample_n = min(200, n_sims)
    sample_idx = np.random.choice(n_sims, sample_n, replace=False)

    for i in sample_idx:
        color = "#FF336633" if crash_mask[i] else "#00FF8811"
        ax1.plot(x_years, all_paths[:, i], color=color, linewidth=0.3, alpha=0.15)

    # Confidence bands
    ax1.fill_between(
        x_years,
        mc_results["p05"],
        mc_results["p95"],
        alpha=0.15,
        color=CHART_COLORS["secondary"],
        label="90% CI",
    )
    ax1.fill_between(
        x_years,
        mc_results["p25"],
        mc_results["p75"],
        alpha=0.20,
        color=CHART_COLORS["accent"],
        label="50% CI",
    )

    # Mean and median
    ax1.plot(
        x_years,
        mc_results["mean_path"],
        color=CHART_COLORS["primary"],
        linewidth=2,
        label=f"Mean: ${mc_results['final_mean']:,.0f}",
    )
    ax1.plot(
        x_years,
        mc_results["median_path"],
        "--",
        color=CHART_COLORS["accent"],
        linewidth=1.5,
        label=f"Median: ${mc_results['final_median']:,.0f}",
    )

    # Current price line
    ax1.axhline(
        current_price, color=CHART_COLORS["neutral"], linestyle=":", linewidth=0.8, alpha=0.6
    )

    # Crash zone: shade below 80% of current price
    crash_level = current_price * 0.80
    ax1.axhline(
        crash_level,
        color=CHART_COLORS["negative"],
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"-20% crash level (${crash_level:,.0f})",
    )

    crash_pct = float(crash_mask.mean() * 100)
    ax1.set_title(
        f"5-Year Projection — {crash_pct:.0f}% of paths experience ≥20% crash",
        color=CHART_COLORS["primary"],
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylabel("S&P 500", color=CHART_COLORS["text"])
    ax1.legend(
        loc="upper left",
        fontsize=8,
        facecolor=CHART_COLORS["bg"],
        edgecolor=CHART_COLORS["neutral"],
        labelcolor=CHART_COLORS["text"],
    )
    ax1.tick_params(colors=CHART_COLORS["text"])
    ax1.grid(alpha=0.1)

    # --- BOTTOM PANEL: Crash probability by horizon ---
    crash_probs = mc_results["crash_probs"]
    horizons = list(crash_probs.keys())
    probs = list(crash_probs.values())

    bar_colors = []
    for p in probs:
        if p < 15:
            bar_colors.append(CHART_COLORS["positive"])
        elif p < 35:
            bar_colors.append(CHART_COLORS["primary"])
        elif p < 60:
            bar_colors.append(CHART_COLORS["secondary"])
        else:
            bar_colors.append(CHART_COLORS["negative"])

    bars = ax2.bar(horizons, probs, color=bar_colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, probs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            color=CHART_COLORS["text"],
            fontsize=9,
            fontweight="bold",
        )

    # Danger zone backgrounds
    ax2.axhspan(0, 15, alpha=0.05, color="green")
    ax2.axhspan(35, 60, alpha=0.05, color="orange")
    ax2.axhspan(60, 100, alpha=0.05, color="red")

    ax2.set_ylabel("Crash Probability (%)", color=CHART_COLORS["text"])
    ax2.set_xlabel("Horizon", color=CHART_COLORS["text"])
    ax2.set_title(
        "Cumulative Crash Probability (≥20% peak-to-trough drawdown)",
        color=CHART_COLORS["primary"],
        fontsize=12,
        fontweight="bold",
    )
    ax2.tick_params(colors=CHART_COLORS["text"])
    ax2.set_ylim(0, max(probs) * 1.15 if probs else 100)
    ax2.grid(axis="y", alpha=0.1)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# DIAGNOSTIC WALK-FORWARD PLOTS
# ═══════════════════════════════════════════════════════════════════════


def chart_ml_crash_timeline(bt_results, **style):
    """ML crash probability over time with actual crash periods shaded red.

    This is the most important validation chart: does the model's crash
    probability actually rise before crashes and fall during calm periods?
    """
    if bt_results is None or len(bt_results) == 0:
        return None

    report_cfg = config.get("reporting", {})
    fig_w = style.get("figure_width", report_cfg.get("figure_width", 7))
    fig_h = style.get("figure_height", report_cfg.get("figure_height", 3.5))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=CHART_COLORS["bg"])
    ax.set_facecolor(CHART_COLORS["bg"])

    dates = bt_results["date"]
    crash_prob = bt_results["ml_crash_12m"] * 100
    actual_crash = bt_results["actual_crash_12m"].astype(bool)

    # Shade actual crash periods red
    for i in range(len(dates)):
        if actual_crash.iloc[i]:
            ax.axvspan(
                dates.iloc[i],
                dates.iloc[min(i + 1, len(dates) - 1)],
                alpha=0.15,
                color="red",
                linewidth=0,
            )

    # Plot ML crash probability
    ax.plot(dates, crash_prob, color=CHART_COLORS["negative"], linewidth=1.5, label="ML Crash Prob")
    ax.axhline(y=20, color=CHART_COLORS["text"], linestyle="--", alpha=0.3, linewidth=0.8)

    ax.set_ylabel("Crash Probability (%)", color=CHART_COLORS["text"])
    ax.set_title(
        "ML Crash Probability vs Actual Crashes (shaded red)",
        color=CHART_COLORS["primary"],
        fontsize=11,
        fontweight="bold",
    )
    ax.tick_params(colors=CHART_COLORS["text"])
    ax.legend(
        facecolor=CHART_COLORS["bg"],
        edgecolor=CHART_COLORS["neutral"],
        labelcolor=CHART_COLORS["text"],
    )
    ax.grid(alpha=0.1)

    fig.tight_layout()
    return fig


def chart_return_scatter(bt_results, **style):
    """Predicted vs actual 12-month return scatter plot with 45-degree line.

    Points near the diagonal = accurate predictions. Systematic bias shows
    as the cloud drifting above or below the line.
    """
    if bt_results is None or len(bt_results) == 0:
        return None

    report_cfg = config.get("reporting", {})
    fig_w = style.get("figure_width", report_cfg.get("figure_width", 7))
    fig_h = style.get("figure_height", report_cfg.get("figure_height", 3.5))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=CHART_COLORS["bg"])
    ax.set_facecolor(CHART_COLORS["bg"])

    actual = bt_results["actual_return_12m"] * 100
    predicted = bt_results["ml_return_12m"] * 100

    ax.scatter(actual, predicted, alpha=0.5, s=20, color=CHART_COLORS["accent"], edgecolors="none")

    # 45-degree line (perfect prediction)
    lims = [min(actual.min(), predicted.min()) - 5, max(actual.max(), predicted.max()) + 5]
    ax.plot(lims, lims, "--", color=CHART_COLORS["text"], alpha=0.4, linewidth=1)

    ax.set_xlabel("Actual 12m Return (%)", color=CHART_COLORS["text"])
    ax.set_ylabel("Predicted 12m Return (%)", color=CHART_COLORS["text"])
    ax.set_title(
        "ML Return Predictions vs Actual",
        color=CHART_COLORS["primary"],
        fontsize=11,
        fontweight="bold",
    )
    ax.tick_params(colors=CHART_COLORS["text"])
    ax.grid(alpha=0.1)

    fig.tight_layout()
    return fig


def chart_calibration_diagram(bt_results, **style):
    """Reliability diagram: predicted crash probability bins vs observed frequency.

    A well-calibrated model has points on the diagonal. Points above = under-confident
    (actual rate higher than predicted). Points below = over-confident.
    """
    if bt_results is None or len(bt_results) == 0:
        return None

    report_cfg = config.get("reporting", {})
    fig_w = style.get("figure_width", report_cfg.get("figure_width", 7))
    fig_h = style.get("figure_height", report_cfg.get("figure_height", 3.5))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=CHART_COLORS["bg"])
    ax.set_facecolor(CHART_COLORS["bg"])

    predicted = bt_results["ml_crash_12m"]
    actual = bt_results["actual_crash_12m"].astype(float)

    # Bin predictions into groups
    bins = [0, 0.10, 0.20, 0.30, 0.50, 1.0]
    bin_centers = []
    observed_rates = []

    for i in range(len(bins) - 1):
        mask = (predicted >= bins[i]) & (predicted < bins[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i + 1]) / 2 * 100)
            observed_rates.append(actual[mask].mean() * 100)

    if bin_centers:
        ax.bar(
            bin_centers,
            observed_rates,
            width=8,
            alpha=0.7,
            color=CHART_COLORS["secondary"],
            edgecolor=CHART_COLORS["text"],
            linewidth=0.5,
        )
        ax.plot(
            [0, 100],
            [0, 100],
            "--",
            color=CHART_COLORS["text"],
            alpha=0.4,
            linewidth=1,
            label="Perfect calibration",
        )

    ax.set_xlabel("Predicted Crash Probability (%)", color=CHART_COLORS["text"])
    ax.set_ylabel("Observed Crash Frequency (%)", color=CHART_COLORS["text"])
    ax.set_title(
        "Crash Model Calibration (Reliability Diagram)",
        color=CHART_COLORS["primary"],
        fontsize=11,
        fontweight="bold",
    )
    ax.tick_params(colors=CHART_COLORS["text"])
    ax.legend(
        facecolor=CHART_COLORS["bg"],
        edgecolor=CHART_COLORS["neutral"],
        labelcolor=CHART_COLORS["text"],
    )
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.1)

    fig.tight_layout()
    return fig


def chart_shap_waterfall(shap_contributions, title="Crash Probability Drivers (SHAP)"):
    """Horizontal bar chart showing top SHAP contributors to crash probability.

    Args:
        shap_contributions: list of (feature_name, shap_value) tuples sorted by |SHAP|.
                            Positive = pushes crash prob UP; negative = pushes it DOWN.
        title: chart title string

    Returns matplotlib Figure, or None if no data.
    """
    if not _MPL_AVAILABLE or not shap_contributions:
        return None

    top = shap_contributions[:10]
    names = [t[0] for t in top]
    values = [t[1] for t in top]

    # Reverse so largest absolute value is at the top
    names = names[::-1]
    values = values[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(CHART_COLORS["bg"])
    ax.set_facecolor(CHART_COLORS["bg"])

    colors_bars = [CHART_COLORS["negative"] if v > 0 else CHART_COLORS["positive"] for v in values]
    bars = ax.barh(names, values, color=colors_bars, edgecolor="none", height=0.6)

    ax.axvline(0, color=CHART_COLORS["neutral"], linewidth=0.8, alpha=0.6)

    for bar, val in zip(bars, values):
        sign = "+" if val > 0 else ""
        ax.text(
            val + (0.001 if val >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{val:.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            color=CHART_COLORS["text"],
            fontsize=7.5,
        )

    ax.set_xlabel("SHAP Value (impact on log-odds of crash)", color=CHART_COLORS["text"])
    ax.set_title(title, color=CHART_COLORS["primary"], fontsize=11, fontweight="bold")
    ax.tick_params(colors=CHART_COLORS["text"], labelsize=8)
    ax.grid(axis="x", alpha=0.12)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=CHART_COLORS["negative"], label="Increases crash probability"),
        Patch(facecolor=CHART_COLORS["positive"], label="Decreases crash probability"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        facecolor=CHART_COLORS["bg"],
        edgecolor=CHART_COLORS["neutral"],
        labelcolor=CHART_COLORS["text"],
        fontsize=8,
    )

    fig.tight_layout()
    return fig


def chart_stress_test(stress_results, current_price):
    """Horizontal bar chart showing S&P 500 level after historical crisis replays.

    Args:
        stress_results: dict from run_stress_tests()
        current_price: current S&P 500 level

    Returns matplotlib Figure.
    """
    if not _MPL_AVAILABLE or not stress_results:
        return None

    names = list(stress_results.keys())
    trough_prices = [stress_results[n]["trough_price"] for n in names]
    drop_pcts = [stress_results[n]["drop_pct"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(CHART_COLORS["bg"])
    ax.set_facecolor(CHART_COLORS["bg"])

    y_pos = range(len(names))
    bars = ax.barh(
        list(y_pos),
        drop_pcts,
        color=CHART_COLORS["negative"],
        edgecolor="none",
        height=0.6,
        alpha=0.85,
    )

    for bar, price, pct in zip(bars, trough_prices, drop_pcts):
        ax.text(
            pct - 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"${price:,.0f}  ({pct:.1f}%)",
            va="center",
            ha="right",
            color=CHART_COLORS["text"],
            fontsize=8.5,
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, color=CHART_COLORS["text"], fontsize=9)
    ax.set_xlabel("Peak-to-Trough Drop (%)", color=CHART_COLORS["text"])
    ax.set_title(
        f"Historical Stress Test — S&P 500 from ${current_price:,.0f}",
        color=CHART_COLORS["primary"],
        fontsize=11,
        fontweight="bold",
    )
    ax.tick_params(colors=CHART_COLORS["text"])
    ax.invert_xaxis()
    ax.grid(axis="x", alpha=0.12)

    fig.tight_layout()
    return fig


def chart_credit_stress(fred_data):
    """3-panel chart of credit stress indicators: TED spread, HY OAS, IG OAS.

    Args:
        fred_data: dict from fetch_fred_data(), keyed by series name.

    Returns matplotlib Figure, or None if data unavailable.
    """
    if not _MPL_AVAILABLE or not fred_data:
        return None

    panels = [
        ("ted_spread", "TED Spread (%)", "Interbank Stress"),
        ("hy_oas", "HY OAS (bps)", "High-Yield Spread"),
        ("ig_oas", "IG OAS (bps)", "Invest.-Grade Spread"),
    ]

    available = [
        (key, label, title)
        for key, label, title in panels
        if key in fred_data and len(fred_data[key]) > 10
    ]

    if not available:
        return None

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.5 * n), sharex=False)
    fig.patch.set_facecolor(CHART_COLORS["bg"])

    if n == 1:
        axes = [axes]

    for ax, (key, label, title) in zip(axes, available):
        series = fred_data[key].dropna()
        if series.empty:
            ax.set_visible(False)
            continue

        ax.set_facecolor(CHART_COLORS["bg"])
        ax.plot(series.index, series.values, color=CHART_COLORS["accent"], linewidth=1.2, alpha=0.9)
        ax.fill_between(series.index, series.values, alpha=0.12, color=CHART_COLORS["accent"])

        # Mark current value
        current_val = float(series.iloc[-1])
        hist_avg = float(series.mean())
        ax.axhline(
            hist_avg,
            color=CHART_COLORS["neutral"],
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
            label=f"Avg: {hist_avg:.2f}",
        )
        ax.axhline(
            current_val,
            color=CHART_COLORS["primary"],
            linewidth=1.0,
            linestyle="-",
            alpha=0.8,
            label=f"Now: {current_val:.2f}",
        )

        ax.set_ylabel(label, color=CHART_COLORS["text"], fontsize=8)
        ax.set_title(title, color=CHART_COLORS["primary"], fontsize=9, fontweight="bold")
        ax.tick_params(colors=CHART_COLORS["text"], labelsize=7)
        ax.legend(
            facecolor=CHART_COLORS["bg"],
            edgecolor=CHART_COLORS["neutral"],
            labelcolor=CHART_COLORS["text"],
            fontsize=7,
            loc="upper left",
        )
        ax.grid(alpha=0.1)

    fig.tight_layout(h_pad=0.6)
    return fig
