"""
Evaluation Module
==================

Publishable metrics for crash probability model evaluation:
Brier Score, Brier Skill Score, reliability diagrams, AUC over time,
and baseline model comparisons.
"""

from finpredict.evaluation.metrics import (
    brier_score,
    brier_skill_score,
    reliability_diagram,
    roc_auc_over_time,
    prediction_spread_check,
)
from finpredict.evaluation.comparison import (
    baseline_crash_prob_vix,
    baseline_crash_prob_yield_curve,
    baseline_crash_prob_climatology,
    compare_models,
)
