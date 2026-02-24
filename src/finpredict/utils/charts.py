"""
Module 9: Visualization
========================

All chart creation functions for the Market Prediction Engine.
Each function returns a matplotlib Figure.

Usage:
    from finpredict.utils.charts import chart_projection, chart_crash_probability
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import warnings

from finpredict.config import config

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
plt.style.use("dark_background")

# Lazy import for reportlab (only needed when generating PDFs)
def _get_inch():
    from reportlab.lib.units import inch
    return inch


CHART_COLORS = {
    'primary': '#FFD700',     # Gold
    'secondary': '#FF6B35',   # Orange
    'accent': '#00D4FF',      # Cyan
    'positive': '#00FF88',    # Green
    'negative': '#FF3366',    # Red
    'neutral': '#888888',
    'bg': '#0D1117',
    'text': '#E6EDF3',
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
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=CHART_COLORS['bg'], edgecolor='none')
    buf.seek(0)
    img = RLImage(buf, width=width, height=height)
    plt.close(fig)
    return img


def chart_backtest(data, bt_results):
    """Backtest validation chart: actual vs predicted S&P 500 since 2000."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=CHART_COLORS['bg'])
    ax.set_facecolor(CHART_COLORS['bg'])

    # Actual S&P 500
    bt_data = data[data.index >= config["data"]["backtest_start"]]
    ax.plot(bt_data.index, bt_data['SP500'], color=CHART_COLORS['accent'],
            linewidth=1.5, label='Actual S&P 500', zorder=3)

    if len(bt_results) > 0:
        # Prediction bands
        ax.fill_between(bt_results['date'], bt_results['pred_p05'],
                        bt_results['pred_p95'], alpha=0.25,
                        color=CHART_COLORS['secondary'], label='90% Prediction Band')
        ax.plot(bt_results['date'], bt_results['pred_mean'],
                '--', color=CHART_COLORS['primary'], linewidth=1,
                label='Model Prediction', alpha=0.8)

    # 7% baseline
    baseline_start = float(bt_data['SP500'].iloc[0])
    days = (bt_data.index - bt_data.index[0]).days
    baseline = baseline_start * (1.07 ** (days / 365.25))
    ax.plot(bt_data.index, baseline, ':', color=CHART_COLORS['neutral'],
            linewidth=1, label='7% Annual Baseline', alpha=0.6)

    ax.set_yscale('log')
    ax.set_ylabel('S&P 500 (log scale)', color=CHART_COLORS['text'])
    ax.set_title('Backtest Validation: Model Accuracy (2000–Present)',
                 color=CHART_COLORS['primary'], fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, facecolor=CHART_COLORS['bg'],
              edgecolor=CHART_COLORS['neutral'], labelcolor=CHART_COLORS['text'])
    ax.tick_params(colors=CHART_COLORS['text'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def chart_projection(mc_results, current_price):
    """5-year Monte Carlo projection with confidence bands."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=CHART_COLORS['bg'])
    ax.set_facecolor(CHART_COLORS['bg'])

    days = np.arange(len(mc_results['mean_path']))
    years = days / config["simulation"]["trading_days_per_year"]

    ax.fill_between(years, mc_results['p05'], mc_results['p95'],
                    alpha=0.15, color=CHART_COLORS['secondary'], label='90% Confidence')
    ax.fill_between(years, mc_results['p25'], mc_results['p75'],
                    alpha=0.25, color=CHART_COLORS['primary'], label='50% Confidence')
    ax.plot(years, mc_results['mean_path'], color=CHART_COLORS['primary'],
            linewidth=2, label=f"Mean: ${mc_results['final_mean']:,.0f}")
    ax.plot(years, mc_results['median_path'], '--', color=CHART_COLORS['accent'],
            linewidth=1.5, label=f"Median: ${mc_results['final_median']:,.0f}")

    # 7% baseline
    baseline = current_price * (1.07 ** years)
    ax.plot(years, baseline, ':', color=CHART_COLORS['neutral'],
            linewidth=1, label='7% Baseline', alpha=0.6)

    ax.axhline(current_price, color=CHART_COLORS['text'], linestyle='-', alpha=0.3)
    ax.set_xlabel('Years', color=CHART_COLORS['text'])
    ax.set_ylabel('S&P 500 Level', color=CHART_COLORS['text'])
    ax.set_title(f'5-Year Market Projection ({config["simulation"]["num_simulations"]:,} Monte Carlo Simulations)',
                 color=CHART_COLORS['primary'], fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, facecolor=CHART_COLORS['bg'],
              edgecolor=CHART_COLORS['neutral'], labelcolor=CHART_COLORS['text'])
    ax.tick_params(colors=CHART_COLORS['text'])
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def chart_crash_probability(mc_results):
    """Crash probability by time horizon — the PRIMARY deliverable."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=CHART_COLORS['bg'])
    ax.set_facecolor(CHART_COLORS['bg'])

    labels = list(mc_results['crash_probs'].keys())
    values = list(mc_results['crash_probs'].values())

    bar_colors = []
    for v in values:
        if v < 15:
            bar_colors.append(CHART_COLORS['positive'])
        elif v < 30:
            bar_colors.append(CHART_COLORS['primary'])
        else:
            bar_colors.append(CHART_COLORS['negative'])

    bars = ax.bar(labels, values, color=bar_colors, edgecolor='white', linewidth=0.5, alpha=0.9)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                color=CHART_COLORS['text'], fontweight='bold', fontsize=10)

    ax.set_ylabel('Crash Probability (%)', color=CHART_COLORS['text'])
    ax.set_xlabel('Time Horizon', color=CHART_COLORS['text'])
    ax.set_title('Crash Probability by Time Horizon (≥20% Drawdown)',
                 color=CHART_COLORS['primary'], fontsize=14, fontweight='bold')
    ax.tick_params(colors=CHART_COLORS['text'])
    ax.grid(axis='y', alpha=0.15)
    fig.tight_layout()
    return fig


def chart_risk_score(data):
    """Historical composite risk score with VIX overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), facecolor=CHART_COLORS['bg'],
                                    gridspec_kw={'height_ratios': [1, 0.6]})

    bt_data = data[data.index >= config["data"]["backtest_start"]]

    # Risk score
    ax1.set_facecolor(CHART_COLORS['bg'])
    risk = bt_data['Risk_Score']
    ax1.fill_between(risk.index, 0, risk, where=risk > 0,
                     color=CHART_COLORS['negative'], alpha=0.4, label='High Risk')
    ax1.fill_between(risk.index, 0, risk, where=risk <= 0,
                     color=CHART_COLORS['positive'], alpha=0.4, label='Low Risk')
    ax1.axhline(0, color=CHART_COLORS['neutral'], linestyle='--', alpha=0.5)
    current_risk = float(risk.iloc[-1])
    ax1.plot(risk.index[-1], current_risk, 'o', color=CHART_COLORS['primary'],
             markersize=8, zorder=5)
    ax1.set_ylabel('Risk Score (σ)', color=CHART_COLORS['text'])
    ax1.set_title('Composite Risk Indicator', color=CHART_COLORS['primary'],
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, facecolor=CHART_COLORS['bg'],
               edgecolor=CHART_COLORS['neutral'], labelcolor=CHART_COLORS['text'])
    ax1.tick_params(colors=CHART_COLORS['text'])
    ax1.grid(True, alpha=0.15)

    # VIX
    if 'VIX' in bt_data.columns:
        ax2.set_facecolor(CHART_COLORS['bg'])
        vix = bt_data['VIX']
        ax2.fill_between(vix.index, 0, vix, where=vix > 30,
                         color=CHART_COLORS['negative'], alpha=0.3, label='Fear (VIX>30)')
        ax2.plot(vix.index, vix, color=CHART_COLORS['secondary'], linewidth=0.8, label='VIX')
        ax2.axhline(20, color=CHART_COLORS['neutral'], linestyle='--', alpha=0.4)
        ax2.axhline(30, color=CHART_COLORS['negative'], linestyle='--', alpha=0.4)
        ax2.set_ylabel('VIX Level', color=CHART_COLORS['text'])
        ax2.set_xlabel('VIX (Fear Index)', color=CHART_COLORS['text'])
        ax2.tick_params(colors=CHART_COLORS['text'])
        ax2.legend(fontsize=8, facecolor=CHART_COLORS['bg'],
                   edgecolor=CHART_COLORS['neutral'], labelcolor=CHART_COLORS['text'])
        ax2.grid(True, alpha=0.15)

    fig.tight_layout()
    return fig


def chart_scenarios(mc_results, current_price):
    """Scenario analysis — each scenario as a separate projected path."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=CHART_COLORS['bg'])
    ax.set_facecolor(CHART_COLORS['bg'])

    scenario_colors = {
        'Base Case': '#4CAF50',
        'AI Productivity Boom': '#2196F3',
        'Soft Landing': '#00BCD4',
        'Market Correction': '#FF9800',
        'Stagflation': '#795548',
        'Recession': '#9C27B0',
        'AI Bubble Collapse': '#E91E63',
        'Geopolitical Crisis': '#F44336',
    }

    forecast_days = config["simulation"]["forecast_years"] * config["simulation"]["trading_days_per_year"]
    years = np.linspace(0, config["simulation"]["forecast_years"], forecast_days + 1)

    for name, info in mc_results['scenarios'].items():
        mu = info['return']
        vol = info['volatility']
        # Expected path using continuous compounding (matches jump-diffusion process)
        # E[S(t)] = S(0) * exp(mu * t) — the drift already accounts for diffusion
        mean_path = current_price * np.exp(mu * years)
        c = scenario_colors.get(name, '#FFFFFF')
        prob = info['probability']
        final = info['mean_final']
        ret = info['total_return']
        ax.plot(years, mean_path, color=c, linewidth=2,
                label=f"{name} ({prob*100:.0f}%) → {ret:+.1f}%")

    # 7% baseline
    baseline = current_price * (1.07 ** years)
    ax.plot(years, baseline, ':', color=CHART_COLORS['neutral'],
            linewidth=1, label='7% Baseline', alpha=0.5)

    ax.plot(0, current_price, '*', color=CHART_COLORS['primary'],
            markersize=12, zorder=5)

    ax.set_xlabel('Years', color=CHART_COLORS['text'])
    ax.set_ylabel('S&P 500 Level', color=CHART_COLORS['text'])
    ax.set_title('Scenario Analysis: Alternative Futures',
                 color=CHART_COLORS['primary'], fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, facecolor=CHART_COLORS['bg'],
              edgecolor=CHART_COLORS['neutral'], labelcolor=CHART_COLORS['text'])
    ax.tick_params(colors=CHART_COLORS['text'])
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def chart_sectors(sector_results, sp500_return):
    """Sector performance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=CHART_COLORS['bg'])
    ax.set_facecolor(CHART_COLORS['bg'])

    names = list(sector_results.keys())
    returns = [sector_results[n]['expected_return'] for n in names]

    # Sort by return
    sorted_pairs = sorted(zip(names, returns), key=lambda x: x[1])
    names, returns = zip(*sorted_pairs)

    bar_colors = [CHART_COLORS['positive'] if r > sp500_return else CHART_COLORS['secondary']
                  for r in returns]

    bars = ax.barh(names, returns, color=bar_colors, edgecolor='white',
                   linewidth=0.3, alpha=0.85)

    for bar, val in zip(bars, returns):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', color=CHART_COLORS['text'], fontsize=9)

    ax.axvline(sp500_return, color=CHART_COLORS['primary'], linestyle='--',
               linewidth=1.5, label=f'S&P 500: {sp500_return:+.1f}%')

    ax.set_xlabel('Expected 5-Year Return (%)', color=CHART_COLORS['text'])
    ax.set_title('Sector Performance Projections',
                 color=CHART_COLORS['primary'], fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, facecolor=CHART_COLORS['bg'],
              edgecolor=CHART_COLORS['neutral'], labelcolor=CHART_COLORS['text'])
    ax.tick_params(colors=CHART_COLORS['text'])
    ax.grid(axis='x', alpha=0.15)
    fig.tight_layout()
    return fig


def chart_stocks(stock_results, sp500_return):
    """Individual stock expected return bar chart with cap tier annotations."""
    if not stock_results:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(stock_results) * 0.4)),
                           facecolor=CHART_COLORS['bg'])
    ax.set_facecolor(CHART_COLORS['bg'])

    # Sort by expected return
    sorted_stocks = sorted(stock_results.items(),
                           key=lambda x: x[1]['expected_return'])
    tickers = [s[0] for s in sorted_stocks]
    returns = [s[1]['expected_return'] for s in sorted_stocks]

    bar_colors = [CHART_COLORS['positive'] if r > sp500_return else CHART_COLORS['secondary']
                  for r in returns]

    bars = ax.barh(tickers, returns, color=bar_colors, edgecolor='white',
                   linewidth=0.3, alpha=0.85)

    for bar, val, (tick, info) in zip(bars, returns, sorted_stocks):
        label = f'{val:+.1f}% ({info["cap_tier"]})'
        ax.text(max(bar.get_width() + 1, 2), bar.get_y() + bar.get_height()/2,
                label, va='center', color=CHART_COLORS['text'], fontsize=8)

    ax.axvline(sp500_return, color=CHART_COLORS['primary'], linestyle='--',
               linewidth=1.5, label=f'S&P 500: {sp500_return:+.1f}%')

    ax.set_xlabel('Expected 5-Year Return (%)', color=CHART_COLORS['text'])
    ax.set_title('Individual Stock Projections (CAGR-Capped)',
                 color=CHART_COLORS['primary'], fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, facecolor=CHART_COLORS['bg'],
              edgecolor=CHART_COLORS['neutral'], labelcolor=CHART_COLORS['text'])
    ax.tick_params(colors=CHART_COLORS['text'])
    ax.grid(axis='x', alpha=0.15)
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor=CHART_COLORS['bg'],
                                    gridspec_kw={'height_ratios': [2.5, 1]})
    for ax in [ax1, ax2]:
        ax.set_facecolor(CHART_COLORS['bg'])

    all_paths = mc_results['all_paths']
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
        color = '#FF336633' if crash_mask[i] else '#00FF8811'
        ax1.plot(x_years, all_paths[:, i], color=color, linewidth=0.3, alpha=0.15)

    # Confidence bands
    ax1.fill_between(x_years, mc_results['p05'], mc_results['p95'],
                     alpha=0.15, color=CHART_COLORS['secondary'], label='90% CI')
    ax1.fill_between(x_years, mc_results['p25'], mc_results['p75'],
                     alpha=0.20, color=CHART_COLORS['accent'], label='50% CI')

    # Mean and median
    ax1.plot(x_years, mc_results['mean_path'], color=CHART_COLORS['primary'],
             linewidth=2, label=f"Mean: ${mc_results['final_mean']:,.0f}")
    ax1.plot(x_years, mc_results['median_path'], '--', color=CHART_COLORS['accent'],
             linewidth=1.5, label=f"Median: ${mc_results['final_median']:,.0f}")

    # Current price line
    ax1.axhline(current_price, color=CHART_COLORS['neutral'], linestyle=':',
                linewidth=0.8, alpha=0.6)

    # Crash zone: shade below 80% of current price
    crash_level = current_price * 0.80
    ax1.axhline(crash_level, color=CHART_COLORS['negative'], linestyle='--',
                linewidth=1, alpha=0.5, label=f'-20% crash level (${crash_level:,.0f})')

    crash_pct = float(crash_mask.mean() * 100)
    ax1.set_title(f'5-Year Projection — {crash_pct:.0f}% of paths experience ≥20% crash',
                  color=CHART_COLORS['primary'], fontsize=14, fontweight='bold')
    ax1.set_ylabel('S&P 500', color=CHART_COLORS['text'])
    ax1.legend(loc='upper left', fontsize=8, facecolor=CHART_COLORS['bg'],
               edgecolor=CHART_COLORS['neutral'], labelcolor=CHART_COLORS['text'])
    ax1.tick_params(colors=CHART_COLORS['text'])
    ax1.grid(alpha=0.1)

    # --- BOTTOM PANEL: Crash probability by horizon ---
    crash_probs = mc_results['crash_probs']
    horizons = list(crash_probs.keys())
    probs = list(crash_probs.values())

    bar_colors = []
    for p in probs:
        if p < 15:
            bar_colors.append(CHART_COLORS['positive'])
        elif p < 35:
            bar_colors.append(CHART_COLORS['primary'])
        elif p < 60:
            bar_colors.append(CHART_COLORS['secondary'])
        else:
            bar_colors.append(CHART_COLORS['negative'])

    bars = ax2.bar(horizons, probs, color=bar_colors, edgecolor='white',
                   linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, probs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.0f}%', ha='center', va='bottom',
                 color=CHART_COLORS['text'], fontsize=9, fontweight='bold')

    # Danger zone backgrounds
    ax2.axhspan(0, 15, alpha=0.05, color='green')
    ax2.axhspan(35, 60, alpha=0.05, color='orange')
    ax2.axhspan(60, 100, alpha=0.05, color='red')

    ax2.set_ylabel('Crash Probability (%)', color=CHART_COLORS['text'])
    ax2.set_xlabel('Horizon', color=CHART_COLORS['text'])
    ax2.set_title('Cumulative Crash Probability (≥20% peak-to-trough drawdown)',
                  color=CHART_COLORS['primary'], fontsize=12, fontweight='bold')
    ax2.tick_params(colors=CHART_COLORS['text'])
    ax2.set_ylim(0, max(probs) * 1.15 if probs else 100)
    ax2.grid(axis='y', alpha=0.1)

    fig.tight_layout()
    return fig



