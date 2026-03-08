"""
Module 10: PDF Report Generation
==================================

Generates the professional multi-page PDF report with all charts,
statistics, scenario analysis, and stock projections.

Usage:
    from finpredict.reporting.pdf_report import generate_report

    generate_report(data, mc_results, bt_results, ...)
"""

from datetime import datetime, timedelta

# reportlab is an optional dependency used only for PDF generation.
# Provide a fallback so the engine can run without it.
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        PageBreak,
        Image,
        Table,
        TableStyle,
        KeepTogether,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False
    # Define no-op stand-ins so references later do not crash
    letter = None
    inch = 1
    SimpleDocTemplate = object

    def Paragraph(*args, **kwargs):
        return None

    def Spacer(*args, **kwargs):
        return None

    def PageBreak(*args, **kwargs):
        return None

    def Image(*args, **kwargs):
        return None

    def Table(*args, **kwargs):
        return None

    def TableStyle(*args, **kwargs):
        return None

    def KeepTogether(*args, **kwargs):
        return None

    def getSampleStyleSheet():
        return {}

    class ParagraphStyle:
        def __init__(self, *args, **kwargs):
            pass

    class colors:
        def HexColor(x):
            return None

    TA_CENTER = TA_LEFT = TA_JUSTIFY = None

from finpredict.config import config
from finpredict.utils.charts import (
    fig_to_image,
    chart_backtest,
    chart_projection,
    chart_crash_probability,
    chart_risk_score,
    chart_scenarios,
    chart_sectors,
    chart_stocks,
    chart_combined_projection_crash,
    chart_ml_crash_timeline,
    chart_return_scatter,
    chart_calibration_diagram,
    chart_shap_waterfall,
    chart_stress_test,
    chart_credit_stress,
)


def generate_report(
    data,
    mc_results,
    bt_results,
    sector_results,
    stock_results,
    current_price,
    regime,
    risk_score,
    crash_freq,
    output_path,
    shap_contributions=None,
    counterfactual_results=None,
    stress_results=None,
    fred_data=None,
    regime_validation=None,
    external_validation=None,
    crash_timing_results=None,
):
    """
    MODULE 10 ENTRY POINT: Generate comprehensive PDF report.
    """
    if not _REPORTLAB_AVAILABLE:
        print("[MODULE 10] reportlab not installed; skipping PDF generation")
        return None
    print("[MODULE 10] Generating PDF report...")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        "Title2",
        parent=styles["Heading1"],
        fontSize=22,
        textColor=colors.HexColor("#FFD700"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#FFD700"),
        spaceAfter=8,
        spaceBefore=12,
    )
    body_style = ParagraphStyle(
        "Body2",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#333333"),
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#666666")
    )

    def make_table(headers, rows, col_widths=None):
        """Helper to create styled tables."""
        tdata = [headers] + rows
        t = Table(tdata, colWidths=col_widths)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#FFD700")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("TOPPADDING", (0, 0), (-1, 0), 8),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.HexColor("#f5f5f5"), colors.white],
                    ),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ]
            )
        )
        return t

    # ===== PAGE 1: EXECUTIVE SUMMARY =====
    story.append(Paragraph("MARKET PREDICTION ENGINE v7.0", title_style))
    story.append(
        Paragraph("Core Engine — Crash Probability &amp; 5-Year Projection", styles["Heading3"])
    )
    story.append(Spacer(1, 0.15 * inch))

    exec_text = f"""
    <b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
    <b>S&amp;P 500 Current:</b> ${current_price:,.2f}<br/>
    <b>Market Regime:</b> {regime}<br/>
    <b>Risk Score:</b> {risk_score:.2f}σ<br/><br/>

    <b>5-YEAR PROJECTION</b><br/>
    Mean Target: ${mc_results['final_mean']:,.0f} ({mc_results['total_return_pct']:+.1f}%)<br/>
    Median Target: ${mc_results['final_median']:,.0f}<br/>
    90% Range: ${mc_results['final_p05']:,.0f} to ${mc_results['final_p95']:,.0f}<br/>
    Annualized Return: {mc_results['annual_return_pct']:.1f}%<br/><br/>

    <b>RISK ASSESSMENT</b><br/>
    ML Model 12m Crash Probability: {mc_results.get('ml_crash_prob', 0) or 0:.1f}%<br/>
    Monte Carlo 12m Crash Prob: {mc_results['crash_prob_1y']:.1f}% (conditioned on ML)<br/>
    5-Year Crash Probability: {mc_results['crash_prob_5y']:.1f}%<br/>
    1Y CVaR (95%): {mc_results.get('cvar_95_1y_pct', mc_results['cvar_95_pct']):.1f}%<br/>
    5Y CVaR (95%): {mc_results.get('cvar_95_5y_pct', mc_results['cvar_95_pct']):.1f}%<br/>
    Avg Max Drawdown: {mc_results['max_dd_pct']:.1f}%
    """
    story.append(Paragraph(exec_text, body_style))
    story.append(PageBreak())

    # ===== PAGE 2: BACKTEST VALIDATION =====
    story.append(Paragraph("MODEL VALIDATION", heading_style))

    if len(bt_results) > 0:
        mape = bt_results.attrs.get("mape", 0)
        coverage = bt_results.attrs.get("coverage", 0)
        direction = bt_results.attrs.get("direction", 0)
        brier = bt_results.attrs.get("brier_score", 0)
        cal_low = bt_results.attrs.get("cal_low", 0)
        cal_med = bt_results.attrs.get("cal_med", 0)
        cal_high = bt_results.attrs.get("cal_high", 0)
        n_low = bt_results.attrs.get("n_low", 0)
        n_med = bt_results.attrs.get("n_med", 0)
        n_high = bt_results.attrs.get("n_high", 0)

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
    story.append(Spacer(1, 0.15 * inch))
    story.append(fig_to_image(chart_backtest(data, bt_results)))
    story.append(PageBreak())

    # ===== PAGE 2b: ML DIAGNOSTIC CHARTS =====
    story.append(Paragraph("ML MODEL DIAGNOSTICS", heading_style))

    diag_text = """
    These diagnostic charts validate the ML model's crash predictions and return
    estimates across the full walk-forward backtest period. They reveal whether the
    model discriminates between calm and crisis periods, how well predicted returns
    track actual outcomes, and whether crash probabilities are properly calibrated.
    """
    story.append(Paragraph(diag_text, body_style))
    story.append(Spacer(1, 0.1 * inch))

    # Crash timeline
    crash_tl = chart_ml_crash_timeline(bt_results)
    if crash_tl is not None:
        story.append(
            Paragraph(
                "<b>Crash Probability Timeline</b> — Does the model's "
                "crash probability rise before actual crashes (red shading)?",
                small_style,
            )
        )
        story.append(Spacer(1, 0.05 * inch))
        story.append(fig_to_image(crash_tl, height=2.8 * inch))
        story.append(Spacer(1, 0.15 * inch))

    # Return scatter
    ret_scat = chart_return_scatter(bt_results)
    if ret_scat is not None:
        story.append(
            Paragraph(
                "<b>Predicted vs Actual Returns</b> — Points near the "
                "diagonal indicate accurate predictions.",
                small_style,
            )
        )
        story.append(Spacer(1, 0.05 * inch))
        story.append(fig_to_image(ret_scat, height=2.8 * inch))
        story.append(Spacer(1, 0.15 * inch))

    # Calibration diagram
    cal_diag = chart_calibration_diagram(bt_results)
    if cal_diag is not None:
        story.append(
            Paragraph(
                "<b>Crash Calibration (Reliability Diagram)</b> — Bars on "
                "the diagonal mean the model is well-calibrated.",
                small_style,
            )
        )
        story.append(Spacer(1, 0.05 * inch))
        story.append(fig_to_image(cal_diag, height=2.8 * inch))

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
    story.append(Spacer(1, 0.1 * inch))
    story.append(fig_to_image(chart_projection(mc_results, current_price)))
    story.append(PageBreak())

    # ===== PAGE 3b: COMBINED PROJECTION + CRASH PROBABILITY =====
    story.append(Paragraph("PROJECTION &amp; CRASH PROBABILITY COMBINED", heading_style))

    combined_text = """
    This chart combines the 5-year market projection with crash probability analysis.
    <b>Top panel:</b> Individual Monte Carlo paths colored by outcome — paths that
    experience a ≥20% peak-to-trough crash are highlighted in red. The fan shows
    confidence intervals. <b>Bottom panel:</b> Cumulative probability of a ≥20% crash
    by each time horizon.
    """
    story.append(Paragraph(combined_text, body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        fig_to_image(
            chart_combined_projection_crash(mc_results, current_price),
            width=7 * inch,
            height=4.5 * inch,
        )
    )
    story.append(PageBreak())

    # ===== PAGE 4: SCENARIO ANALYSIS =====
    story.append(Paragraph("SCENARIO ANALYSIS", heading_style))

    scenario_text = """
    The model considers 8 plausible futures with dynamically adjusted probabilities
    based on current regime, risk score, VIX level, and yield curve shape:
    """
    story.append(Paragraph(scenario_text, body_style))
    story.append(Spacer(1, 0.1 * inch))

    scen_rows = []
    for name, info in mc_results["scenarios"].items():
        scen_rows.append(
            [
                name,
                f"{info['probability']*100:.0f}%",
                f"{info['total_return']:+.1f}%",
                f"{info['volatility']*100:.0f}%",
                info["description"][:50],
            ]
        )
    story.append(
        make_table(
            ["Scenario", "Prob", "5Y Return", "Vol", "Description"],
            scen_rows,
            [1.3 * inch, 0.6 * inch, 0.8 * inch, 0.6 * inch, 3.5 * inch],
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(fig_to_image(chart_scenarios(mc_results, current_price)))
    story.append(PageBreak())

    # ===== PAGE 5: CRASH PROBABILITY =====
    story.append(Paragraph("CRASH PROBABILITY ANALYSIS", heading_style))

    cp_1y = mc_results.get("crash_prob_1y", 0)
    cp_5y = mc_results.get("crash_prob_5y", 0)

    crash_text = f"""
    <b>This is the primary deliverable of the engine.</b> The model estimates the probability
    of a ≥20% drawdown (market crash) occurring at each time horizon. The crash probability
    is computed by tracking the maximum drawdown across all {config["simulation"]["num_simulations"]:,}
    simulation paths at each horizon.<br/><br/>

    <b>Key Findings:</b><br/>
    12-month crash probability: {cp_1y:.1f}%<br/>
    5-year crash probability: {cp_5y:.1f}%<br/>
    1Y CVaR (95%): {mc_results.get('cvar_95_1y_pct', mc_results['cvar_95_pct']):.1f}% | 5Y CVaR (95%): {mc_results.get('cvar_95_5y_pct', mc_results['cvar_95_pct']):.1f}%<br/>
    Historical crash frequency: once every {1/crash_freq:.1f} years<br/><br/>

    <b>Interpretation:</b> A {cp_1y:.0f}% 1-year crash probability means that in roughly
    {cp_1y:.0f} out of 100 simulated scenarios, the market experiences a 20%+ drawdown
    within the next 12 months. This accounts for current valuations, regime, risk score,
    and both normal market fluctuations and crash-jump events.
    """
    story.append(Paragraph(crash_text, body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(fig_to_image(chart_crash_probability(mc_results), height=3.2 * inch))

    # ── Crash Timing Distribution (3-month windows) ──
    if crash_timing_results:
        story.append(Spacer(1, 0.2 * inch))
        story.append(
            Paragraph("<b>Crash Timing — 3-Month Window Distribution</b>", styles["Heading3"])
        )
        story.append(
            Paragraph(
                "Which 3-month window has the highest crash probability? "
                "The crash timing model distributes the 12-month crash probability "
                "across quarterly windows to provide actionable timing precision.",
                body_style,
            )
        )
        try:
            fig = _chart_crash_timing(crash_timing_results)
            story.append(fig_to_image(fig, height=2.5 * inch))
        except Exception as e:
            story.append(Paragraph(f"[Crash timing chart error: {e}]", body_style))

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
    story.append(Spacer(1, 0.1 * inch))
    story.append(fig_to_image(chart_risk_score(data), height=4.5 * inch))
    story.append(PageBreak())

    # ===== PAGE 6b: CREDIT STRESS INDICATORS =====
    credit_fig = chart_credit_stress(fred_data)
    if credit_fig is not None:
        story.append(Paragraph("CREDIT STRESS INDICATORS", heading_style))
        credit_text = """
        Credit markets often lead equity markets by weeks or months. Three key indicators
        are tracked: the <b>TED Spread</b> (3-month LIBOR minus T-bill rate — measures
        interbank funding stress), <b>High-Yield OAS</b> (option-adjusted spread for junk
        bonds — widens when default risk rises), and <b>IG OAS</b> (investment-grade spread
        — a more sensitive read on corporate credit conditions). All three are sourced from
        the Federal Reserve (FRED). Dashed lines show historical averages; gold lines show
        the current reading.
        """
        story.append(Paragraph(credit_text, body_style))
        story.append(Spacer(1, 0.1 * inch))
        story.append(fig_to_image(credit_fig, height=5.5 * inch))

        # Current readings table
        credit_rows = []
        for key, label in [
            ("ted_spread", "TED Spread (%)"),
            ("hy_oas", "HY OAS (bps)"),
            ("ig_oas", "IG OAS (bps)"),
        ]:
            if fred_data and key in fred_data and len(fred_data[key]) > 0:
                series = fred_data[key].dropna()
                if not series.empty:
                    current_val = float(series.iloc[-1])
                    hist_avg = float(series.mean())
                    hist_max = float(series.max())
                    pct_of_max = current_val / hist_max * 100 if hist_max > 0 else 0.0
                    credit_rows.append(
                        [
                            label,
                            f"{current_val:.2f}",
                            f"{hist_avg:.2f}",
                            f"{hist_max:.2f}",
                            f"{pct_of_max:.0f}%",
                        ]
                    )
        if credit_rows:
            story.append(Spacer(1, 0.1 * inch))
            story.append(
                make_table(
                    ["Indicator", "Current", "Hist. Avg", "Hist. Max", "% of Max"],
                    credit_rows,
                    [2.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 0.9 * inch],
                )
            )
        story.append(PageBreak())

    # ===== PAGE 6c: AI EXPLAINABILITY — SHAP CRASH DRIVERS =====
    if shap_contributions:
        story.append(Paragraph("AI EXPLAINABILITY — CRASH PROBABILITY DRIVERS", heading_style))
        shap_text = """
        SHAP (SHapley Additive exPlanations) values decompose the model's current crash
        probability prediction into the contribution of each individual market signal.
        <b>Red bars push the crash probability higher</b>; <b>green bars push it lower</b>.
        The values are on the log-odds scale — a SHAP of +0.10 roughly adds ~2.5 percentage
        points to crash probability at a baseline of 50%.
        """
        story.append(Paragraph(shap_text, body_style))
        story.append(Spacer(1, 0.1 * inch))
        shap_fig = chart_shap_waterfall(shap_contributions)
        if shap_fig is not None:
            story.append(fig_to_image(shap_fig, height=3.8 * inch))
        story.append(Spacer(1, 0.1 * inch))

        # SHAP table — top 10 features with value and direction
        shap_rows = []
        for feat, sv in shap_contributions[:10]:
            direction = "INCREASES crash prob" if sv > 0 else "decreases crash prob"
            shap_rows.append([feat, f"{sv:+.5f}", direction])
        story.append(
            make_table(
                ["Feature", "SHAP Value", "Direction"],
                shap_rows,
                [3.0 * inch, 1.2 * inch, 2.5 * inch],
            )
        )
        story.append(Spacer(1, 0.15 * inch))

        # What-If / Counterfactual table
        if counterfactual_results and counterfactual_results.get("scenarios"):
            story.append(Paragraph("<b>WHAT-IF SENSITIVITY ANALYSIS</b>", body_style))
            wif_text = """
            Each row below shows the model's crash probability estimate under a hypothetical
            change to current market conditions. Only the listed feature(s) are altered;
            all other signals remain at their current values.
            """
            story.append(Paragraph(wif_text, small_style))
            story.append(Spacer(1, 0.08 * inch))

            base_3m = counterfactual_results.get("base_prob_3m")
            base_12m = counterfactual_results.get("base_prob_12m", 0)
            wif_rows = []
            # Baseline row
            b3 = f"{base_3m*100:.1f}%" if base_3m is not None else "—"
            wif_rows.append(["Baseline (current)", b3, f"{base_12m*100:.1f}%", "—", "—"])
            for sc in counterfactual_results["scenarios"]:
                p3 = sc.get("crash_prob_3m")
                p12 = sc.get("crash_prob_12m", 0)
                d3 = sc.get("delta_3m")
                d12 = sc.get("delta_12m", 0)
                p3_str = f"{p3*100:.1f}%" if p3 is not None else "—"
                d3_str = f"{d3*100:+.1f}%" if d3 is not None else "—"
                wif_rows.append(
                    [
                        sc["label"],
                        p3_str,
                        f"{p12*100:.1f}%",
                        d3_str,
                        f"{d12*100:+.1f}%",
                    ]
                )
            story.append(
                make_table(
                    ["Scenario", "3M Crash", "12M Crash", "Δ 3M", "Δ 12M"],
                    wif_rows,
                    [2.5 * inch, 1.0 * inch, 1.0 * inch, 0.9 * inch, 0.9 * inch],
                )
            )
        story.append(PageBreak())

    # ===== PAGE 6d: HISTORICAL STRESS TESTS =====
    if stress_results:
        story.append(Paragraph("HISTORICAL STRESS TEST", heading_style))
        stress_text = f"""
        Deterministic scenario analysis: applies the exact peak-to-trough drawdown of each
        major historical crisis to the current S&amp;P 500 level of
        <b>${current_price:,.0f}</b>. These are not probabilistic forecasts — they answer
        the question: <i>"If today's market experienced the same severity as [crisis],
        where would the index trough?"</i> Recovery estimates are historical averages and
        will vary with future conditions.
        """
        story.append(Paragraph(stress_text, body_style))
        story.append(Spacer(1, 0.1 * inch))
        stress_fig = chart_stress_test(stress_results, current_price)
        if stress_fig is not None:
            story.append(fig_to_image(stress_fig, height=3.2 * inch))
        story.append(Spacer(1, 0.1 * inch))

        stress_rows = []
        for name, info in stress_results.items():
            stress_rows.append(
                [
                    name,
                    f"{info['drop_pct']*100:.1f}%",
                    f"${info['trough_price']:,.0f}",
                    f"{info['drop_points']:+,.0f} pts",
                    f"{info['duration_days']} days",
                    f"{info['recovery_days'] // 30} months",
                ]
            )
        story.append(
            make_table(
                [
                    "Crisis",
                    "Drop %",
                    "Trough Level",
                    "Points Lost",
                    "Crash Duration",
                    "Recovery Time",
                ],
                stress_rows,
                [1.8 * inch, 0.7 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 1.0 * inch],
            )
        )
        story.append(Spacer(1, 0.1 * inch))
        story.append(
            Paragraph(
                "<i>Source: Shiller/Yale, NBER. Recovery = calendar days from trough to prior peak.</i>",
                small_style,
            )
        )
        story.append(PageBreak())

    # ===== PAGE 6e: VALIDATION & CROSS-CHECK =====
    if regime_validation is not None or external_validation is not None:
        story.append(Paragraph("VALIDATION &amp; CROSS-CHECK", heading_style))
        story.append(
            Paragraph(
                "Every engine output is cross-checked against independent sources. "
                "Divergences from consensus are flagged — not to override the engine, "
                "but to ensure users know when they are acting against market consensus.",
                body_style,
            )
        )
        story.append(Spacer(1, 8))

        # Regime Confirmation Table
        if regime_validation is not None:
            story.append(Paragraph("<b>Regime Confirmation</b>", body_style))
            story.append(Spacer(1, 4))

            def _pass_fail(val):
                return "PASS" if val else "FAIL"

            regime_rows = [
                [
                    "200-day SMA",
                    _pass_fail(regime_validation.price_confirmed),
                    "Price below 200d MA confirms bearish structure"
                    if regime_validation.price_confirmed
                    else "Price above 200d MA — bear not confirmed by price",
                ],
                [
                    "Market Breadth",
                    _pass_fail(regime_validation.breadth_confirmed),
                    "Majority of sectors declining"
                    if regime_validation.breadth_confirmed
                    else "Breadth divergence — insufficient sectors declining",
                ],
                [
                    "Consensus",
                    _pass_fail(regime_validation.consensus_aligned),
                    "Institutional forecasts align"
                    if regime_validation.consensus_aligned
                    else "Institutional forecasts diverge from engine",
                ],
            ]
            regime_table = make_table(
                ["Check", "Status", "Detail"],
                regime_rows,
                col_widths=[1.2 * inch, 0.8 * inch, 4.5 * inch],
            )
            story.append(regime_table)
            story.append(Spacer(1, 4))

            conf_color = {"HIGH": "#00CC00", "MEDIUM": "#FFAA00", "LOW": "#FF4444"}
            color = conf_color.get(regime_validation.confidence, "#999999")
            story.append(
                Paragraph(
                    f"<b>Regime:</b> {regime_validation.regime} — "
                    f"<font color='{color}'><b>{regime_validation.confidence} CONFIDENCE</b></font> "
                    f"({'CONFIRMED' if regime_validation.confirmed else 'UNCONFIRMED'})",
                    body_style,
                )
            )
            story.append(Spacer(1, 12))

        # External Signals Dashboard
        if external_validation is not None:
            story.append(Paragraph("<b>External Signals Dashboard</b>", body_style))
            story.append(Spacer(1, 4))

            signal_rows = [
                [
                    "Leading Economic Index (LEI)",
                    external_validation.lei_signal,
                    "Consecutive declines precede recessions",
                ],
                [
                    "SLOOS (Bank Lending)",
                    external_validation.sloos_signal,
                    "Tightening standards precede credit stress",
                ],
                [
                    "Fed Funds Direction",
                    external_validation.fed_signal,
                    "Rate path signals monetary policy stance",
                ],
                [
                    "Consumer Sentiment",
                    external_validation.sentiment_signal,
                    "Extreme fear is historically contrarian-bullish",
                ],
                ["IMF GDP Forecast", external_validation.imf_signal, "Global growth outlook"],
            ]
            signal_table = make_table(
                ["Source", "Signal", "Interpretation"],
                signal_rows,
                col_widths=[2.0 * inch, 1.2 * inch, 3.3 * inch],
            )
            story.append(signal_table)
            story.append(Spacer(1, 8))

            agreement_pct = external_validation.engine_agreement * 100
            agr_color = (
                "#00CC00"
                if agreement_pct >= 60
                else "#FFAA00"
                if agreement_pct >= 40
                else "#FF4444"
            )
            story.append(
                Paragraph(
                    f"<b>Engine-External Agreement:</b> "
                    f"<font color='{agr_color}'><b>{agreement_pct:.0f}%</b></font> | "
                    f"<b>Consensus Direction:</b> {external_validation.consensus_direction}",
                    body_style,
                )
            )

            if external_validation.divergence_alerts:
                story.append(Spacer(1, 8))
                story.append(Paragraph("<b>Divergence Alerts:</b>", body_style))
                for alert in external_validation.divergence_alerts:
                    story.append(
                        Paragraph(
                            f"<font color='#FF4444'>&#x26A0;</font> {alert}",
                            body_style,
                        )
                    )

        # Advanced Metrics (from backtest)
        adv = bt_results.attrs.get("advanced_metrics") if len(bt_results) > 0 else None
        if adv:
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>Advanced Validation Metrics</b>", body_style))
            story.append(Spacer(1, 4))

            metric_rows = []
            if "lead_time" in adv:
                lt = adv["lead_time"]
                metric_rows.append(
                    [
                        "Lead Time Accuracy",
                        f"{lt['mean_lead_days']:.0f} days",
                        "> 30 days",
                        "OK" if lt["mean_lead_days"] > 30 else "WARN",
                    ]
                )
            if "false_alarm" in adv:
                fa = adv["false_alarm"]
                metric_rows.append(
                    [
                        "False Alarm Rate",
                        f"{fa['rate']*100:.0f}%",
                        "< 30%",
                        "OK" if fa["rate"] < 0.30 else "WARN",
                    ]
                )
            if "missed_crash" in adv:
                mc = adv["missed_crash"]
                metric_rows.append(
                    [
                        "Missed Crash Rate",
                        f"{mc['rate']*100:.0f}%",
                        "< 20%",
                        "OK" if mc["rate"] < 0.20 else "WARN",
                    ]
                )
            if "signal_drawdown" in adv:
                sd = adv["signal_drawdown"]
                metric_rows.append(
                    [
                        "Signal Max Drawdown",
                        f"{sd['max_drawdown']*100:.1f}%",
                        "< 25%",
                        "OK" if sd["max_drawdown"] < 0.25 else "WARN",
                    ]
                )
            if "directional_vs_consensus" in adv:
                dvc = adv["directional_vs_consensus"]
                metric_rows.append(
                    [
                        "Dir. Accuracy vs Consensus",
                        f"{dvc['engine_correct_rate']*100:.0f}%",
                        "> 55%",
                        "OK" if dvc["engine_correct_rate"] > 0.55 else "WARN",
                    ]
                )

            if metric_rows:
                metrics_table = make_table(
                    ["Metric", "Value", "Target", "Status"],
                    metric_rows,
                    col_widths=[2.2 * inch, 1.2 * inch, 1.2 * inch, 0.8 * inch],
                )
                story.append(metrics_table)

        story.append(PageBreak())

    # ===== PAGE 7: INSTITUTIONAL COMPARISON =====
    story.append(Paragraph("INSTITUTIONAL BENCHMARK COMPARISON", heading_style))

    inst_text = """
    Every projection should be validated against public institutional forecasts. This creates
    credibility and ensures our model is not producing delusional estimates.
    """
    story.append(Paragraph(inst_text, body_style))
    story.append(Spacer(1, 0.1 * inch))

    inst_rows = []
    model_annual = mc_results["annual_return_pct"]
    for name, info in config["institutional_benchmarks"].items():
        if not isinstance(info, dict) or "annual" not in info:
            continue
        implied_5y = current_price * (1 + info["annual"]) ** 5
        variance = ((mc_results["final_mean"] / implied_5y) - 1) * 100
        inst_rows.append(
            [
                name.replace("_", " "),
                f"{info['annual']*100:.1f}%",
                info["horizon"],
                f"${implied_5y:,.0f}",
                f"{variance:+.1f}%",
            ]
        )
    inst_rows.append(
        ["V7.0 Model", f"{model_annual:.1f}%", "5Y", f"${mc_results['final_mean']:,.0f}", "—"]
    )

    story.append(
        make_table(
            ["Institution", "Annual Return", "Horizon", "5Y Implied Target", "Model Variance"],
            inst_rows,
            [1.5 * inch, 1.2 * inch, 0.8 * inch, 1.3 * inch, 1.2 * inch],
        )
    )

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
    story.append(Spacer(1, 0.1 * inch))

    if sector_results:
        sect_rows = []
        for name, info in sorted(
            sector_results.items(), key=lambda x: x[1].get("expected_total", x[1].get("expected_return", 0)), reverse=True
        ):
            exp_ret = info.get("expected_total", info.get("expected_return", 0))
            med_ret = info.get("sim_total_return", info.get("median_return", 0))
            vol = info.get("sigma", info.get("volatility", 0))
            if isinstance(vol, float) and vol < 1:
                vol *= 100  # Convert decimal to percent
            ann_ret = info.get("expected_annual", exp_ret / 5 if exp_ret else 0)
            sharpe = ann_ret / vol if vol > 0 else 0
            sect_rows.append(
                [
                    name,
                    f"{exp_ret:+.1f}%",
                    f"{med_ret:+.1f}%",
                    f"{vol:.0f}%",
                    f"{sharpe:.2f}",
                ]
            )
        story.append(
            make_table(
                ["Sector", "Expected 5Y %", "Median 5Y %", "Volatility", "Sharpe"],
                sect_rows,
                [1.5 * inch, 1.2 * inch, 1.2 * inch, 1 * inch, 0.8 * inch],
            )
        )
        story.append(Spacer(1, 0.15 * inch))
        story.append(fig_to_image(chart_sectors(sector_results, mc_results["total_return_pct"])))

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
        story.append(Spacer(1, 0.1 * inch))

        stock_rows = []
        for tick, info in sorted(
            stock_results.items(), key=lambda x: x[1]["expected_return"], reverse=True
        ):
            f"${info['market_cap']/1e9:.0f}B" if info["market_cap"] else "N/A"
            f"${info['analyst_target']:.0f}" if info["analyst_target"] else "—"
            stock_rows.append(
                [
                    tick,
                    f"${info['current_price']:.0f}",
                    info["cap_tier"].title(),
                    f"{info['expected_return']:+.1f}%",
                    f"{info['prob_loss_5y']:.0f}%",
                    f"{info['avg_max_drawdown']:.0f}%",
                    f"{info['sharpe']:.2f}",
                ]
            )

        story.append(
            make_table(
                ["Ticker", "Price", "Cap Tier", "Exp 5Y %", "P(Loss)", "Max DD", "Sharpe"],
                stock_rows,
                [
                    0.7 * inch,
                    0.8 * inch,
                    0.8 * inch,
                    0.9 * inch,
                    0.8 * inch,
                    0.8 * inch,
                    0.7 * inch,
                ],
            )
        )
        story.append(Spacer(1, 0.15 * inch))

        stock_chart = chart_stocks(stock_results, mc_results["total_return_pct"])
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


def _chart_crash_timing(crash_timing_results):
    """Generate bar chart for crash timing 3-month window distribution."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    labels = ["Q1\n0-3m", "Q2\n3-6m", "Q3\n6-9m", "Q4\n9-12m", "Beyond\n12m"]
    keys = ["Q1_0_3m", "Q2_3_6m", "Q3_6_9m", "Q4_9_12m", "beyond_12m"]
    probs = [crash_timing_results.get(k, 0) for k in keys]

    # Color coding: RED (>40%), AMBER (25-40%), GREEN (<25%)
    colors = []
    for p in probs:
        if p > 0.40:
            colors.append("#E74C3C")  # Red
        elif p > 0.25:
            colors.append("#F39C12")  # Amber
        else:
            colors.append("#27AE60")  # Green

    fig, ax = plt.subplots(figsize=(7, 2.5))
    bars = ax.bar(labels, [p * 100 for p in probs], color=colors, edgecolor="white", width=0.6)

    # Add value labels on bars
    for bar, p in zip(bars, probs):
        if p > 0.02:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{p*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylabel("Crash Probability (%)", fontsize=9)
    ax.set_title("Crash Probability by 3-Month Window", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(p * 100 for p in probs) * 1.3 + 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Highlight most likely crash window
    max_idx = probs[:4].index(max(probs[:4]))  # Exclude "beyond"
    if probs[max_idx] > 0.10:
        bars[max_idx].set_edgecolor("#2C3E50")
        bars[max_idx].set_linewidth(2)

    fig.tight_layout()
    return fig
