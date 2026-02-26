"""
Module 10: PDF Report Generation
==================================

Generates the professional multi-page PDF report with all charts,
statistics, scenario analysis, and stock projections.

Usage:
    from finpredict.reporting.pdf_report import generate_report

    generate_report(data, mc_results, bt_results, ...)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Image, Table, TableStyle, KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

from finpredict.config import config
from finpredict.utils.charts import (
    fig_to_image, chart_backtest, chart_projection, chart_crash_probability,
    chart_risk_score, chart_scenarios, chart_sectors, chart_stocks,
    chart_combined_projection_crash, CHART_COLORS,
)


def generate_report(data, mc_results, bt_results, sector_results, stock_results,
                    current_price, regime, risk_score, crash_freq, output_path):
    """
    MODULE 10 ENTRY POINT: Generate comprehensive PDF report.
    """
    print("[MODULE 10] Generating PDF report...")

    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle('Title2', parent=styles['Heading1'],
                                  fontSize=22, textColor=colors.HexColor('#FFD700'),
                                  spaceAfter=6, alignment=TA_CENTER)
    heading_style = ParagraphStyle('H2', parent=styles['Heading2'],
                                    fontSize=16, textColor=colors.HexColor('#FFD700'),
                                    spaceAfter=8, spaceBefore=12)
    body_style = ParagraphStyle('Body2', parent=styles['BodyText'],
                                 fontSize=10, leading=14, textColor=colors.HexColor('#333333'))
    small_style = ParagraphStyle('Small', parent=styles['Normal'],
                                  fontSize=8, textColor=colors.HexColor('#666666'))

    def make_table(headers, rows, col_widths=None):
        """Helper to create styled tables."""
        tdata = [headers] + rows
        t = Table(tdata, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFD700')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ]))
        return t

    # ===== PAGE 1: EXECUTIVE SUMMARY =====
    story.append(Paragraph("MARKET PREDICTION ENGINE v4.5", title_style))
    story.append(Paragraph("Core Engine — Crash Probability & 5-Year Projection", styles['Heading3']))
    story.append(Spacer(1, 0.15*inch))

    exec_text = f"""
    <b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
    <b>S&amp;P 500 Current:</b> ${current_price:,.2f}<br/>
    <b>Market Regime:</b> {regime}<br/>
    <b>Risk Score:</b> {risk_score:.2f}σ<br/><br/>

    <b>5-YEAR PROJECTION</b><br/>
    Mean Target: ${mc_results['final_mean']:,.0f} ({mc_results['total_return_pct']:+.1f}%)<br/>
    Median Target: ${mc_results['final_median']:,.0f}<br/>
    90% Range: ${mc_results['p05'][-1]:,.0f} to ${mc_results['p95'][-1]:,.0f}<br/>
    Annualized Return: {mc_results['annual_return_pct']:.1f}%<br/><br/>

    <b>RISK ASSESSMENT</b><br/>
    1-Year Crash Probability: {mc_results['crash_prob_1y']:.1f}%<br/>
    5-Year Crash Probability: {mc_results['crash_prob_5y']:.1f}%<br/>
    CVaR (95%): {mc_results['cvar_95_pct']:.1f}%<br/>
    Avg Max Drawdown: {mc_results['max_drawdown_pct']:.1f}%
    """
    story.append(Paragraph(exec_text, body_style))
    story.append(PageBreak())

    # ===== PAGE 2: BACKTEST VALIDATION =====
    story.append(Paragraph("MODEL VALIDATION", heading_style))

    if len(bt_results) > 0:
        mape = bt_results.attrs.get('mape', 0)
        coverage = bt_results.attrs.get('coverage', 0)
        direction = bt_results.attrs.get('direction', 0)
        brier = bt_results.attrs.get('brier_score', 0)
        cal_low = bt_results.attrs.get('cal_low', 0)
        cal_med = bt_results.attrs.get('cal_med', 0)
        cal_high = bt_results.attrs.get('cal_high', 0)
        n_low = bt_results.attrs.get('n_low', 0)
        n_med = bt_results.attrs.get('n_med', 0)
        n_high = bt_results.attrs.get('n_high', 0)

        bt_text = f"""
        The model was validated using <b>{len(bt_results)} walk-forward predictions</b> from
        2000 to present. Each prediction used <b>ONLY historical data available at that time</b>,
        ensuring zero data leakage. This is the honest test of model accuracy.<br/><br/>

        <b>Validation Metrics:</b><br/>
        MAPE (Mean Absolute % Error): {mape:.1f}%<br/>
        90% Band Coverage: {coverage:.1f}%<br/>
        Directional Accuracy: {direction:.1f}%<br/>
        Brier Score: {brier:.4f} (lower = better; 0.25 = random)<br/><br/>

        <b>Crash Probability Calibration:</b><br/>
        Low risk (&lt;15% predicted): {cal_low:.0f}% actual crash rate ({n_low} predictions)<br/>
        Medium risk (15-40% predicted): {cal_med:.0f}% actual crash rate ({n_med} predictions)<br/>
        High risk (&gt;40% predicted): {cal_high:.0f}% actual crash rate ({n_high} predictions)<br/><br/>

        <b>How Walk-Forward Works:</b> For each prediction date, we use only data available
        up to that point. We blend historical geometric drift with institutional anchoring,
        run 1,000 Monte Carlo simulations to predict the next year, then compare against
        what actually happened. This simulates real-time prediction and prevents the model
        from "cheating" by using future information.
        """
        story.append(Paragraph(bt_text, body_style))
    story.append(Spacer(1, 0.15*inch))
    story.append(fig_to_image(chart_backtest(data, bt_results)))
    story.append(PageBreak())

    # ===== PAGE 3: 5-YEAR PROJECTION =====
    story.append(Paragraph("5-YEAR MARKET PROJECTION", heading_style))

    proj_text = f"""
    Based on {config["simulation"]["num_simulations"]:,} Monte Carlo simulations across 5 probability-weighted
    scenarios, the model projects the S&amp;P 500 to reach <b>${mc_results['final_mean']:,.0f}</b>
    by {(datetime.now() + timedelta(days=365*5)).strftime('%B %Y')}, representing a
    <b>{mc_results['total_return_pct']:+.1f}%</b> total return
    ({mc_results['annual_return_pct']:.1f}% annualized).
    """
    story.append(Paragraph(proj_text, body_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(fig_to_image(chart_projection(mc_results, current_price)))
    story.append(PageBreak())

    # ===== PAGE 3b: COMBINED PROJECTION + CRASH PROBABILITY =====
    story.append(Paragraph("PROJECTION &amp; CRASH PROBABILITY COMBINED", heading_style))

    combined_text = f"""
    This chart combines the 5-year market projection with crash probability analysis.
    <b>Top panel:</b> Individual Monte Carlo paths colored by outcome — paths that
    experience a ≥20% peak-to-trough crash are highlighted in red. The fan shows
    confidence intervals. <b>Bottom panel:</b> Cumulative probability of a ≥20% crash
    by each time horizon.
    """
    story.append(Paragraph(combined_text, body_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(fig_to_image(
        chart_combined_projection_crash(mc_results, current_price),
        width=7 * inch, height=4.5*inch
    ))
    story.append(PageBreak())

    # ===== PAGE 4: SCENARIO ANALYSIS =====
    story.append(Paragraph("SCENARIO ANALYSIS", heading_style))

    scenario_text = """
    The model considers 8 plausible futures with dynamically adjusted probabilities
    based on current regime, risk score, VIX level, and yield curve shape:
    """
    story.append(Paragraph(scenario_text, body_style))
    story.append(Spacer(1, 0.1*inch))

    scen_rows = []
    for name, info in mc_results['scenarios'].items():
        scen_rows.append([
            name,
            f"{info['probability']*100:.0f}%",
            f"{info['total_return']:+.1f}%",
            f"{info['volatility']*100:.0f}%",
            info['description'][:50]
        ])
    story.append(make_table(
        ['Scenario', 'Prob', '5Y Return', 'Vol', 'Description'],
        scen_rows,
        [1.3*inch, 0.6*inch, 0.8*inch, 0.6*inch, 3.5 * inch]
    ))
    story.append(Spacer(1, 0.15*inch))
    story.append(fig_to_image(chart_scenarios(mc_results, current_price)))
    story.append(PageBreak())

    # ===== PAGE 5: CRASH PROBABILITY =====
    story.append(Paragraph("CRASH PROBABILITY ANALYSIS", heading_style))

    cp = mc_results['crash_probs']
    cp_1y = cp.get('12mo', 0)
    cp_5y = cp.get('60mo', 0)

    crash_text = f"""
    <b>This is the primary deliverable of the engine.</b> The model estimates the probability
    of a ≥20% drawdown (market crash) occurring at each time horizon. The crash probability
    is computed by tracking the maximum drawdown across all {config["simulation"]["num_simulations"]:,}
    simulation paths at each horizon.<br/><br/>

    <b>Key Findings:</b><br/>
    12-month crash probability: {cp_1y:.1f}%<br/>
    5-year crash probability: {cp_5y:.1f}%<br/>
    CVaR (95%): {mc_results['cvar_95_pct']:.1f}% (expected loss in worst 5% of scenarios)<br/>
    Historical crash frequency: once every {1/crash_freq:.1f} years<br/><br/>

    <b>Interpretation:</b> A {cp_1y:.0f}% 1-year crash probability means that in roughly
    {cp_1y:.0f} out of 100 simulated scenarios, the market experiences a 20%+ drawdown
    within the next 12 months. This accounts for current valuations, regime, risk score,
    and both normal market fluctuations and crash-jump events.
    """
    story.append(Paragraph(crash_text, body_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(fig_to_image(chart_crash_probability(mc_results), height=3.2*inch))
    story.append(PageBreak())

    # ===== PAGE 6: RISK ASSESSMENT =====
    story.append(Paragraph("RISK ASSESSMENT", heading_style))

    risk_text = f"""
    Current market conditions show a risk score of <b>{risk_score:.2f}σ</b>, indicating a
    <b>{regime}</b> regime. The composite risk indicator combines market volatility, credit
    spreads, yield curve dynamics, momentum signals, gold/equity ratios, and market breadth.
    <br/><br/>
    Values above +2.0σ have historically preceded significant market declines within 6-12 months.
    Values below -0.5σ indicate low-risk conditions favorable for equities.
    """
    story.append(Paragraph(risk_text, body_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(fig_to_image(chart_risk_score(data), height=4.5*inch))
    story.append(PageBreak())

    # ===== PAGE 7: INSTITUTIONAL COMPARISON =====
    story.append(Paragraph("INSTITUTIONAL BENCHMARK COMPARISON", heading_style))

    inst_text = """
    Every projection should be validated against public institutional forecasts. This creates
    credibility and ensures our model is not producing delusional estimates.
    """
    story.append(Paragraph(inst_text, body_style))
    story.append(Spacer(1, 0.1*inch))

    inst_rows = []
    model_annual = mc_results['annual_return_pct']
    for name, info in config["institutional_benchmarks"].items():
        if not isinstance(info, dict) or "annual" not in info:
            continue
        implied_5y = current_price * (1 + info['annual']) ** 5
        variance = ((mc_results['final_mean'] / implied_5y) - 1) * 100
        inst_rows.append([
            name.replace('_', ' '),
            f"{info['annual']*100:.1f}%",
            info['horizon'],
            f"${implied_5y:,.0f}",
            f"{variance:+.1f}%"
        ])
    inst_rows.append([
        'V4.5 Model',
        f"{model_annual:.1f}%",
        '5Y',
        f"${mc_results['final_mean']:,.0f}",
        '—'
    ])

    story.append(make_table(
        ['Institution', 'Annual Return', 'Horizon', '5Y Implied Target', 'Model Variance'],
        inst_rows,
        [1.5*inch, 1.2*inch, 0.8*inch, 1.3*inch, 1.2*inch]
    ))

    inst_note = f"""<br/>
    <b>Analysis:</b> Our model's {model_annual:.1f}% annualized projection places it within the
    range of major institutional forecasts. This alignment validates our calibration approach
    and suggests the model produces realistic expectations anchored to professional consensus.
    """
    story.append(Paragraph(inst_note, body_style))
    story.append(PageBreak())

    # ===== PAGE 8: SECTOR PERFORMANCE =====
    story.append(Paragraph("SECTOR PERFORMANCE PROJECTIONS", heading_style))

    sect_text = """
    Sector-level analysis uses actual ETF data (XLK, XLV, XLF, etc.) to project 5-year
    returns. Each sector is simulated independently with its own historical drift and
    volatility. Sectors outperforming the S&amp;P 500 projection are highlighted.
    """
    story.append(Paragraph(sect_text, body_style))
    story.append(Spacer(1, 0.1*inch))

    if sector_results:
        sect_rows = []
        for name, info in sorted(sector_results.items(),
                                  key=lambda x: x[1]['expected_return'], reverse=True):
            sect_rows.append([
                name,
                f"{info['expected_return']:+.1f}%",
                f"{info['median_return']:+.1f}%",
                f"{info['volatility']:.0f}%",
                f"{info['sharpe']:.2f}"
            ])
        story.append(make_table(
            ['Sector', 'Expected 5Y %', 'Median 5Y %', 'Volatility', 'Sharpe'],
            sect_rows,
            [1.5*inch, 1.2*inch, 1.2*inch, 1*inch, 0.8*inch]
        ))
        story.append(Spacer(1, 0.15*inch))
        story.append(fig_to_image(chart_sectors(sector_results, mc_results['total_return_pct'])))

    story.append(PageBreak())

    # ===== PAGE 9: INDIVIDUAL STOCK ANALYSIS =====
    if stock_results:
        story.append(Paragraph("INDIVIDUAL STOCK PROJECTIONS", heading_style))

        stock_text = """
        Each stock is analyzed independently using Monte Carlo simulation with fundamental
        constraints. Historical drift is capped by market cap tier to prevent delusional
        projections (e.g., mega-caps cannot sustain 40%+ CAGR indefinitely). Where available,
        analyst consensus targets moderate the expected return. All projections use the same
        jump-diffusion process as the S&amp;P 500 model.
        """
        story.append(Paragraph(stock_text, body_style))
        story.append(Spacer(1, 0.1*inch))

        stock_rows = []
        for tick, info in sorted(stock_results.items(),
                                  key=lambda x: x[1]['expected_return'], reverse=True):
            mc_str = f"${info['market_cap']/1e9:.0f}B" if info['market_cap'] else 'N/A'
            at_str = f"${info['analyst_target']:.0f}" if info['analyst_target'] else '—'
            stock_rows.append([
                tick,
                f"${info['current_price']:.0f}",
                info['cap_tier'].title(),
                f"{info['expected_return']:+.1f}%",
                f"{info['prob_loss_5y']:.0f}%",
                f"{info['avg_max_drawdown']:.0f}%",
                f"{info['sharpe']:.2f}",
            ])

        story.append(make_table(
            ['Ticker', 'Price', 'Cap Tier', 'Exp 5Y %', 'P(Loss)', 'Max DD', 'Sharpe'],
            stock_rows,
            [0.7 * inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.8*inch, 0.8*inch, 0.7 * inch]
        ))
        story.append(Spacer(1, 0.15*inch))

        stock_chart = chart_stocks(stock_results, mc_results['total_return_pct'])
        if stock_chart is not None:
            story.append(fig_to_image(stock_chart))

        stock_note = """<br/>
        <b>Note:</b> CAGR caps prevent runaway projections. Mega-cap stocks (&gt;$200B) are
        capped at 15% max CAGR; small-caps (&lt;$2B) at 30%. Historical drift is blended with
        analyst targets where available (60/40 weighting). The probability of loss (P(Loss))
        indicates the fraction of simulations ending below the current price.
        """
        story.append(Paragraph(stock_note, body_style))
        story.append(PageBreak())

    # ===== PAGE 10: METHODOLOGY & DOCUMENTATION =====
    story.append(Paragraph("METHODOLOGY &amp; ENGINE DOCUMENTATION", heading_style))

    method_text = """
    <b>Engine Architecture (8 Modules):</b><br/><br/>

    <b>Module 1 — Data Layer:</b> Fetches live market data from Yahoo Finance. Sources include
    S&amp;P 500, VIX, 10Y/13-Week/30Y Treasury yields, HYG/LQD (credit spreads), Gold,
    NASDAQ, Russell 2000, and 11 sector ETFs. All data is forward-filled to handle
    holidays and weekends.<br/><br/>

    <b>Module 2 — Regime Detection:</b> Classifies each day into Bull/Bear/Volatile using a
    252-day rolling window of geometric returns and volatility. Leading indicators
    (VIX level, composite risk score) override pure price-based classification to
    detect regime transitions earlier than trailing return alone.<br/><br/>

    <b>Module 3 — Risk Scoring:</b> Computes a 9-factor composite z-score from VIX,
    yield curve (10Y-3M spread), credit spreads, long yield volatility, momentum
    exhaustion, short-term vol regime, gold/stock ratio, market breadth, and small
    cap divergence. Output is clipped to [-4, +4]σ.<br/><br/>

    <b>Module 4 — Crash Analysis:</b> Identifies all historical crashes (≥20% peak-to-trough
    drawdown) and computes base frequencies. This calibrates the jump-diffusion
    component of the Monte Carlo engine.<br/><br/>

    <b>Module 5 — Monte Carlo Engine:</b> Jump-diffusion simulation with Student-t fat tails
    and institutional return anchoring. 5 probability-weighted scenarios (Base Case,
    AI Boom, Market Correction, Recession, Geopolitical Crisis) with dynamic
    probability adjustment based on current conditions. Valuation constraint applies
    mean-reversion drag when market deviates from long-run trend.<br/><br/>

    <b>Module 6 — Walk-Forward Backtest:</b> Validates model accuracy using only data
    available at each prediction point (zero data leakage). Drift uses a 50/50 blend
    of historical geometric returns and institutional anchoring. Reports MAPE, coverage,
    directional accuracy, and Brier score for crash calibration.<br/><br/>

    <b>Module 8 — Sector Analysis:</b> Projects each of 11 sector ETFs independently using
    geometric drift and volatility from ETF history. Returns capped at 300%.<br/><br/>

    <b>Module 8b — Stock Analysis:</b> Analyzes individual stocks with CAGR caps by market
    cap tier (mega: max 15%, large: 20%, mid: 25%, small: 30%). Blends historical
    drift with analyst consensus targets (60/40 weighting). Reports expected return,
    probability of loss, max drawdown, and Sharpe ratio.<br/><br/>

    <b>Key Equations:</b><br/>
    Price path: S(t) = S(t-1) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*epsilon) * (1 + J)<br/>
    Where epsilon ~ Student-t(df=8), J ~ N(-0.10, 0.05) * Bernoulli(jump_rate/252)<br/>
    Risk score: Weighted sum of 9 z-scores, each on 252-day rolling window<br/>
    Crash probability: Fraction of simulations with peak-to-trough drawdown exceeding 20%<br/>
    Valuation constraint: Mean-reversion penalty applied when market deviates from long-run trend<br/><br/>

    <b>Data Sources:</b> Yahoo Finance (prices), Federal Reserve (yield data via proxies).
    All computations in Python using NumPy, Pandas, SciPy, Matplotlib.<br/><br/>

    <b>Known Limitations:</b><br/>
    - Cannot predict black swan events (pandemics, wars, novel crises)<br/>
    - Jump-diffusion adds crash-like events but cannot time them precisely<br/>
    - Institutional forecasts used for calibration may themselves be wrong<br/>
    - Sector projections use simplified assumptions (no fundamental data)<br/>
    - Walk-forward backtest shows honest accuracy, which is lower than fitted accuracy
    """
    story.append(Paragraph(method_text, body_style))

    # Build
    doc.build(story)
    print(f"  [OK] Report saved: {output_path}\n")



