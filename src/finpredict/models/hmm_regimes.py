"""
HMM Regime Detection
======================

Uses Hidden Markov Models to detect market regimes as latent states.
The model learns what "bull", "bear", and "crisis" look like from the
statistical properties of returns, VIX, and volatility rather than
from manually-set thresholds.

Key advantage over rule-based: outputs regime PROBABILITIES rather than
binary signals. [0.15, 0.60, 0.25] for [bull, bear, crisis] becomes
the scenario weight input for Monte Carlo simulations.

Usage:
    from finpredict.models.hmm_regimes import fit_hmm_regimes, get_regime_probs

    hmm_result = fit_hmm_regimes(data)
    regime_probs = get_regime_probs(hmm_result)
"""

import numpy as np
import pandas as pd
from typing import NamedTuple

from finpredict.config import config


class HMMResult(NamedTuple):
    """Results from fitting an HMM regime model."""
    model: object                    # Fitted GaussianHMM
    regime_labels: pd.Series         # Regime name per day
    regime_probs: np.ndarray         # Current [bull, bear, crisis] probabilities
    current_regime: str              # Current regime name
    transition_matrix: np.ndarray    # State transition probabilities
    state_means: np.ndarray          # Mean of each state
    state_vols: np.ndarray           # Volatility of each state
    success: bool
    feature_mean: np.ndarray = None  # Standardization mean for scoring new data
    feature_std: np.ndarray = None   # Standardization std for scoring new data


def fit_hmm_regimes(
    data: pd.DataFrame,
    n_states: int = 3,
    n_fits: int = 10,
) -> HMMResult:
    """
    Fit a 3-state Gaussian HMM on market features.

    Features used:
        1. Log returns (5-day smoothed)
        2. VIX level (if available)
        3. 20-day realized volatility

    Runs multiple fits with different seeds to avoid local optima.

    Args:
        data: DataFrame with SP500, optionally VIX
        n_states: Number of hidden states (default 3)
        n_fits: Number of random restarts

    Returns:
        HMMResult with fitted model and regime assignments
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  [WARN] hmmlearn not available, falling back to rule-based regimes")
        return _fallback_result(data)

    print("[HMM] Fitting Hidden Markov Model for regime detection...")

    # Build feature matrix
    returns = data["SP500"].pct_change().dropna()
    log_ret = np.log(1 + returns)

    # Feature 1: Smoothed log returns (5-day rolling mean, annualized)
    ret_smooth = log_ret.rolling(5).mean() * 252

    # Feature 2: Realized volatility (20-day)
    real_vol = returns.rolling(20).std() * np.sqrt(252)

    # Feature 3: VIX if available
    features_list = [ret_smooth, real_vol]
    feature_names = ["returns", "vol"]

    if "VIX" in data.columns:
        vix = data["VIX"].reindex(returns.index)
        features_list.append(vix)
        feature_names.append("vix")

    # Combine and clean
    features = pd.concat(features_list, axis=1).dropna()
    if len(features) < 500:
        print("  [WARN] Not enough data for HMM, falling back to rule-based")
        return _fallback_result(data)

    X = features.values

    # Standardize features for numerical stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # Prevent division by zero
    X_norm = (X - X_mean) / X_std

    # Fit with multiple random seeds, pick best log-likelihood
    best_score = -np.inf
    best_model = None

    for seed in range(n_fits):
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=200,
                random_state=seed,
                tol=1e-4,
            )
            model.fit(X_norm)
            score = model.score(X_norm)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        print("  [WARN] All HMM fits failed, falling back to rule-based")
        return _fallback_result(data)

    # Decode states
    states = best_model.predict(X_norm)
    probs = best_model.predict_proba(X_norm)

    # Label states by their mean return (in original scale)
    state_mean_returns = []
    state_mean_vols = []
    for s in range(n_states):
        mask = states == s
        if mask.sum() > 0:
            state_mean_returns.append(float(X[mask, 0].mean()))  # Annualized return
            state_mean_vols.append(float(X[mask, 1].mean()))     # Realized vol
        else:
            state_mean_returns.append(0)
            state_mean_vols.append(0.2)

    # Sort states: highest return = Bull, lowest = Crisis, middle = Bear
    sorted_indices = np.argsort(state_mean_returns)
    label_map = {
        sorted_indices[0]: "Crisis",
        sorted_indices[1]: "Bear",
        sorted_indices[2]: "Bull",
    }
    if n_states > 3:
        for i in range(3, n_states):
            label_map[sorted_indices[i]] = f"State_{i}"

    # Map to labels
    state_names = [label_map[s] for s in states]
    regime_series = pd.Series(state_names, index=features.index)

    # Extend to full data index (fill gaps with nearest)
    full_regimes = pd.Series("Unknown", index=data.index)
    full_regimes.loc[regime_series.index] = regime_series
    # Forward-fill and back-fill the Unknown gaps
    mask = full_regimes == "Unknown"
    full_regimes[mask] = np.nan
    full_regimes = full_regimes.ffill().bfill().fillna("Bull")

    # Current regime probabilities (reorder to match bull/bear/crisis)
    current_probs_raw = probs[-1]
    current_probs = np.zeros(n_states)
    for raw_idx, name in label_map.items():
        ordered_idx = ["Bull", "Bear", "Crisis"].index(name) if name in ["Bull", "Bear", "Crisis"] else raw_idx
        if ordered_idx < n_states:
            current_probs[ordered_idx] = current_probs_raw[raw_idx]

    current = label_map[states[-1]]

    print(f"  [HMM] Regime distribution:")
    for s_idx in sorted_indices:
        name = label_map[s_idx]
        count = (np.array(states) == s_idx).sum()
        pct = count / len(states) * 100
        print(f"    {name}: {pct:.0f}% of days (mean ret={state_mean_returns[s_idx]*100:.1f}%, "f"vol={state_mean_vols[s_idx]*100:.1f}%)")
    print(f"  [HMM] Current regime: {current} "
          f"(P={current_probs_raw[states[-1]]:.0%})")
    print(f"  [HMM] Log-likelihood: {best_score:.0f}\n")

    return HMMResult(
        model=best_model,
        regime_labels=full_regimes,
        regime_probs=current_probs,
        current_regime=current,
        transition_matrix=best_model.transmat_,
        state_means=np.array(state_mean_returns),
        state_vols=np.array(state_mean_vols),
        success=True,
        feature_mean=X_mean,
        feature_std=X_std,
    )


def get_regime_probs(hmm_result: HMMResult) -> dict:
    """
    Convert HMM regime probabilities to scenario weight adjustments.

    Returns:
        dict with keys 'bull_prob', 'bear_prob', 'crisis_prob'
    """
    if not hmm_result.success:
        return {"bull_prob": 0.50, "bear_prob": 0.30, "crisis_prob": 0.20}

    probs = hmm_result.regime_probs
    # Normalize in case they don't sum to 1
    total = probs.sum()
    if total > 0:
        probs = probs / total

    return {
        "bull_prob": float(probs[0]),
        "bear_prob": float(probs[1]),
        "crisis_prob": float(probs[2]) if len(probs) > 2 else 0.0,
    }


def _fallback_result(data: pd.DataFrame) -> HMMResult:
    """Fallback when HMM is not available."""
    regimes = pd.Series("Bull", index=data.index)
    return HMMResult(
        model=None,
        regime_labels=regimes,
        regime_probs=np.array([0.5, 0.3, 0.2]),
        current_regime="Bull",
        transition_matrix=np.eye(3),
        state_means=np.array([0.10, -0.05, -0.30]),
        state_vols=np.array([0.15, 0.20, 0.35]),
        success=False,
    )
