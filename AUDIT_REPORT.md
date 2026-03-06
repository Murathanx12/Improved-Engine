# Deep Technical Audit: finpredict v7

**Date:** 2026-03-06
**Auditor:** Senior Quant Researcher & ML Engineer
**Scope:** Full codebase — every file read line-by-line
**Codebase:** 15+ modules, 6,000+ lines of Python, 90+ unit tests

---

## SECTION 1: BUGS FOUND

### B1 — Meta-Stacker Receives Fabricated Values for Missing Models [CRITICAL]

**Files:** `src/finpredict/ml/meta_stacker.py:242,253,262` | `src/finpredict/main.py:289-291` | `src/finpredict/simulation/backtest.py:289-291`

**Problem:** When a model (XGB, LSTM, TCN) is unavailable at inference time, `predict_proba()` substitutes a hardcoded `0.12` as the prediction value:

```python
# meta_stacker.py:242
pred = model_predictions.get(model_name)
if pred is None:
    pred = 0.12  # Default — PROBLEM
```

During training, those models' predictions were `NaN` (rows filtered by line 151: `valid = ~(np.isnan(X).any(axis=1))`). At inference, the stacker receives a fabricated `0.12` it never saw during training — a distribution shift that corrupts the learned logistic regression weights. The stacker was trained on real OOS predictions with real inter-model correlations; injecting a constant breaks those correlations.

Additionally, in `main.py:289-291` and `backtest.py:289-291`:
```python
"xgb": xgb_crash_12m if xgb_crash_12m is not None else lgb_crash_12m,
"lstm": lstm_crash_12m if lstm_crash_12m is not None else lgb_crash_12m,
"tcn": tcn_crash_12m if tcn_crash_12m is not None else lgb_crash_12m,
```
Missing models are replaced with LGB's prediction, making the meta-stacker see LGB's signal 2-4 times. This collapses the ensemble diversity.

**Fix:** Track which models were available during training (`self._available_models` is already set at line 75). At inference, build meta-features ONLY from those same models. If a model that was available during training is now missing, fall back to simple averaging instead of injecting fabricated values.

**Test:** Verify that `predict_proba(model_predictions={"lgb": 0.3})` when trained with `["lgb", "xgb", "lstm", "tcn"]` falls back to simple average, not stacker with faked inputs.

---

### B2 — `predict_individual` Returns 0.12 for None Models [MEDIUM]

**File:** `src/finpredict/ml/sequence_model.py:604`

**Problem:**
```python
else:
    result[name] = np.full(n_sequences, 0.12)
```
When LSTM or TCN is `None`, `predict_individual()` returns an array of constant 0.12 values. This constant gets fed to the meta-stacker as if it were a real prediction.

**Fix:** Return `None` instead of a constant array. Let callers (`backtest.py`, `main.py`) handle the `None` case explicitly.

---

### B3 — Ongoing Crash at Dataset End Silently Dropped [HIGH]

**File:** `src/finpredict/risk/crashes.py:65-82`

**Problem:** The crash detection loop records a crash only when the exit condition (recovery to within 5% of peak) is met at line 74. If the dataset ends during an ongoing crash (e.g., data ends during March 2020 sell-off), the crash is never appended:

```python
for i in range(len(data)):
    dd = drawdown.iloc[i]
    if dd <= -threshold and not in_crash:
        in_crash = True
        crash_start = data.index[i]
        crash_min = dd
    elif in_crash:
        crash_min = min(crash_min, dd)
        if dd > -exit_recovery:   # ← Only records on recovery
            crashes.append({...})
            in_crash = False
# ← Loop ends. If in_crash is still True, crash is lost.
```

This means `crash_freq` underestimates frequency, and the crash is invisible to lead time / missed crash evaluation metrics.

**Fix:** After the loop, if `in_crash` is still `True`, append the ongoing crash:
```python
if in_crash:
    crashes.append({
        "start": crash_start,
        "end": data.index[-1],
        "max_dd": crash_min,
        "duration_days": (data.index[-1] - crash_start).days,
        "severity": "Ongoing",
    })
```

**Test:** Create a price series that drops 25% at the end without recovering. Verify `identify_crashes()` returns it.

---

### B4 — HMM Standardization Parameters Not Saved [HIGH]

**File:** `src/finpredict/models/hmm_regimes.py:99-102`

**Problem:**
```python
X_mean = X.mean(axis=0)     # Local variable
X_std = X.std(axis=0)       # Local variable
X_std[X_std == 0] = 1
X_norm = (X - X_mean) / X_std
```

`X_mean` and `X_std` are local variables. The `HMMResult` namedtuple (line 28-38) doesn't include these parameters. When scoring new data in a walk-forward context, there's no way to apply the same standardization. The model was trained on standardized data, but any new prediction call would need to re-standardize with different statistics.

In practice the code only calls `fit_hmm_regimes()` once on the full dataset, so this doesn't break the current pipeline, but it makes incremental scoring impossible and constitutes subtle forward leakage (the HMM is fitted on the full dataset including future data when used in the backtest).

**Fix:** Add `X_mean` and `X_std` fields to `HMMResult`.

---

### B5 — Regime Detection Mixes Log Returns and Simple Returns [MEDIUM]

**File:** `src/finpredict/risk/regimes.py:59-60`

**Problem:**
```python
ann_ret = w.mean() * 252          # w = log_returns (line 55)
ann_vol = returns.loc[date_window].std() * np.sqrt(252)  # returns = simple returns (line 42)
```

The annualized return uses log returns but the annualized volatility uses simple returns. For typical market levels the difference is small (~0.5%), but during crises (high vol), log vs simple returns diverge significantly. This mixes scales when comparing `ann_vol` against `high_vol_threshold` (0.30).

**Fix:** Use the same return type for both. Recommend log returns for both since they're additive over time:
```python
ann_vol = w.std() * np.sqrt(252)  # Use log returns for vol too
```

---

### B6 — Sharpe Annualization Hardcoded `sqrt(4)` [MEDIUM]

**File:** `src/finpredict/evaluation/metrics.py:459`

**Problem:**
```python
sharpe = (arr.mean() / arr.std() * np.sqrt(4)) if arr.std() > 0 else 0.0
```

`np.sqrt(4)` assumes quarterly rebalancing (4 periods/year). But the backtest step is configurable via `bt_cfg["step_months"]` (default 3). If `step_months` changes to 1 (monthly), the annualization factor should be `sqrt(12)`, not `sqrt(4)`.

**Fix:** Compute from actual period count:
```python
n_periods = 12 / step_months  # step_months from config
sharpe = (arr.mean() / arr.std() * np.sqrt(n_periods))
```

---

### B7 — FRED Forward-Fill Ignores Publication Lag [MEDIUM]

**File:** `src/finpredict/ml/features.py:339`

**Problem:**
```python
s = s.reindex(df.index).ffill()
```

FRED monthly data (unemployment, CPI, industrial production) is forward-filled to daily frequency immediately. But FRED data has a publication lag of 1-4 weeks — the January unemployment value isn't published until early February. Forward-filling the January value into January trading days creates ~1 month of look-ahead bias for every FRED feature.

**Fix:** Shift FRED data forward by the typical publication lag before forward-filling:
```python
s = s.shift(21).reindex(df.index).ffill()  # 21 trading days ≈ 1 month lag
```

---

### B8 — CAPE Proxy Uses Price Average Instead of Earnings [MEDIUM]

**File:** `src/finpredict/ml/features.py:460-461`

**Problem:**
```python
sp_10yr_avg = sp.rolling(2520, min_periods=1260).mean()
cape_proxy = sp / sp_10yr_avg.replace(0, np.nan)
earnings_yield = 1.0 / cape_proxy.replace(0, np.nan)
```

This computes `price / price_10yr_avg`, not `price / earnings_10yr_avg`. The real CAPE ratio is P/E10. The "earnings yield" derived from this is systematically wrong — if the market doubled over 10 years, `cape_proxy ≈ 2.0`, `earnings_yield ≈ 0.50` (50%), which is absurd. True earnings yield is typically 3-6%.

**Fix:** Rename to `price_to_avg_ratio` to avoid confusion. The ERP computation using this is directionally useful (higher ratio = more expensive = lower expected return) even if the magnitude is wrong, so it can stay as a feature under a correct name. The `erp` feature magnitude will be wrong but the direction and z-score will still be informative.

---

### B9 — Lookup Table Blend Dampens ML Signal During Regime Shifts [LOW]

**File:** `src/finpredict/ml/crash_model.py:590-594`

**Problem:**
```python
if abs(calibrated[i] - lookup_prob) > 0.15:
    calibrated[i] = 0.5 * calibrated[i] + 0.5 * lookup_prob
```

When the ML prediction diverges from the empirical lookup table by >15pp, the code blends 50/50 with lookup table values (12%, 25%, 35%, or 50%). During genuine regime shifts (where ML should diverge from historical base rates), this dampens the signal exactly when it matters most. The model already has Platt scaling calibration — this second layer of blending is redundant and counterproductive.

**Fix:** Make the divergence threshold and blend ratio configurable in `engine_config.yaml`. Consider removing the lookup table entirely.

---

### B10 — Missing Models Replaced with LGB Prediction in Backtest [LOW]

**File:** `src/finpredict/simulation/backtest.py:289-291`

**Problem:** Same as B1 but specifically in the backtest walk-forward loop:
```python
"xgb": xgb_crash_12m if xgb_crash_12m is not None else lgb_crash_12m,
"lstm": lstm_crash_12m if lstm_crash_12m is not None else lgb_crash_12m,
"tcn": tcn_crash_12m if tcn_crash_12m is not None else lgb_crash_12m,
```

**Fix:** Pass `None` values and let the meta-stacker handle partial inputs, or fall back to simple averaging of available models.

---

### B11 — XGB 6m/3m OOS Predictions Fall Back to LGB [LOW]

**File:** `src/finpredict/simulation/backtest.py:270-280`

**Problem:**
```python
oos_predictions["xgb"]["6m"].append(
    float(xgb_model.predict_proba(current_features, "6m")[0])
    if xgb_model.is_trained and "6m" in xgb_model.models else lgb_crash_6m
)
```

When XGB doesn't have a 6m or 3m model, the code uses LGB's prediction as the XGB OOS entry. This means the meta-stacker's XGB column is actually LGB for those horizons, creating artificial correlation between the two models.

**Fix:** Store `None` and let the meta-stacker filter rows with incomplete model coverage.

---

### B12 — PyTorch DataLoader Not Seeded [LOW]

**File:** `src/finpredict/ml/sequence_model.py:401`

**Problem:**
```python
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
```

`shuffle=True` without a `generator` argument means the batch ordering is non-deterministic. Two runs with `random_state=42` will produce different batch orderings, leading to different LSTM/TCN weights.

**Fix:**
```python
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    generator=torch.Generator().manual_seed(self.random_state),
)
```

---

## SECTION 2: METHODOLOGY IMPROVEMENTS

### M1 — Walk-Forward Purge Gap Coordination [Low Effort]

**Area:** Walk-forward methodology
**Files:** `engine_config.yaml`, `crash_model.py:309`, `sequence_model.py:356`, `xgboost_model.py`

**Current State:** `crash_model._train_single()` uses `{"3m": 70, "6m": 140, "12m": 265}`. `sequence_model.train()` always uses `gap_days = 265`. These are independently defined.

**Proposed:** Centralize purge gap definitions in `engine_config.yaml`. Each horizon should use `horizon_days + max_feature_window` (e.g., 252 + 21 = 273 for 12m with 21-day rolling features).

**Why for crash prediction:** Inconsistent purge gaps mean the meta-stacker trains on predictions from models with different leakage profiles, corrupting the ensemble.

---

### M2 — Embargo Period After Training Window [Low Effort]

**Area:** Walk-forward methodology
**Files:** `crash_model.py:311-322`, `sequence_model.py:359-386`

**Current State:** No explicit embargo after the training window end. The validation set starts immediately after the purge gap.

**Proposed:** Add a 5-day embargo period after each validation window to prevent autocorrelation leakage through overlapping feature windows. Reference: Lopez de Prado, *Advances in Financial Machine Learning*, Ch. 7.

**Why:** Without embargo, the first validation samples can have features partially derived from training targets due to rolling windows.

---

### M3 — Platt Scaling Double-Dip [Medium Effort]

**Area:** Calibration
**Files:** `crash_model.py:383-413`

**Current State:** Platt scaling calibrator is fitted on the validation set (line 395), then the Brier score is evaluated on the *same* validation set (line 413). This means calibration quality metrics are optimistically biased — the calibrator has seen these exact data points.

**Proposed:** Either (a) hold out a separate calibration fold from the validation set (3-way split: train/calibrate/evaluate), or (b) use the walk-forward backtest OOS predictions as the calibration input.

**Why:** Evaluating calibration quality on the data used to fit the calibrator tells you nothing about generalization. The reported Brier score improvement from calibration may be illusory.

---

### M4 — Exponential Temporal Weighting [Low Effort]

**Area:** Sample weighting
**Files:** `crash_model.py:299`, `engine_config.yaml`

**Current State:** `np.linspace(0.5, 1.5, n_samples)` — linear ramp with 3x ratio oldest-to-newest.

**Proposed:** Exponential decay: `np.exp(-decay * (n_samples - np.arange(n_samples)))` with `decay` configured in YAML. This better reflects the non-stationarity of financial regimes.

**Why:** Market microstructure has changed dramatically (HFT, passive investing, central bank QE). Exponential decay naturally adapts to structural breaks by rapidly downweighting pre-break data.

---

### M5 — Feature Selection / Redundancy Reduction [Medium Effort]

**Area:** Feature quality
**Files:** `features.py` (add function), `crash_model.py:128` (apply before training)

**Current State:** 80+ features with many correlated pairs (e.g., `vol_1m`/`vol_1w` ~0.85 correlation, `mom_1m`/`mom_2w` ~0.80). LightGBM handles correlated features but each adds noise.

**Proposed:** Add a feature selection step: compute pairwise correlation matrix, drop features with >0.90 correlation (keeping the one with higher importance from previous training). Alternatively, use mutual information with the target to rank features.

**Why:** With ~10% crash base rate and 80+ features, the effective sample size per feature is small. Feature reduction improves the logistic model (5 features) and the meta-stacker (few parameters), and reduces overfitting in LightGBM.

---

### M6 — Dynamic Crash Threshold Scaled by Volatility [Medium Effort]

**Area:** Label quality
**Files:** `features.py:634-637`, `crashes.py:67`

**Current State:** Fixed -20% drawdown threshold for crash labeling.

**Proposed:** Scale threshold by current volatility regime: `threshold = -0.20 * (VIX_long_run_avg / VIX_current)`. In low-vol regimes (VIX=12), a -15% drop is a severe event; in high-vol regimes (VIX=35), -20% might be a normal correction.

**Why:** Fixed thresholds produce different numbers of positive labels depending on the volatility regime. The model learns to detect high-vol crashes but misses slow-onset ones in calm markets.

---

### M7 — Combinatorial Purged Cross-Validation (CPCV) [High Effort]

**Area:** Walk-forward methodology
**Files:** `backtest.py` (new function), `crash_model.py`

**Current State:** Single expanding window split per prediction date.

**Proposed:** Implement CPCV (Lopez de Prado Ch. 12) — generates multiple purged train/test splits from the same time series, producing more reliable OOS performance estimates from limited data.

**Why:** With ~8-12 crashes in 30 years, a single expanding window may happen to include/exclude specific crash events, making metrics highly sensitive to the split point. CPCV averages over many splits.

---

### M8 — Online HMM / Regime Model Update in Backtest [Medium Effort]

**Area:** Regime model
**Files:** `main.py:167`, `backtest.py`

**Current State:** HMM is fit once on the full dataset. In the walk-forward backtest, regime probabilities at time *t* include data from after *t*.

**Proposed:** Refit HMM at each backtest step using only data up to that point, or use an online Bayesian filter. This is computationally expensive but eliminates forward leakage through regime labels.

**Why:** The HMM's regime probabilities are used as meta-stacker features and Monte Carlo weights. If they're computed from future data, the backtest metrics are optimistically biased.

---

### M9 — Bootstrap Over Model Parameters for Uncertainty [Medium Effort]

**Area:** Uncertainty quantification
**Files:** `monte_carlo.py`, `garch.py` (add bootstrap method)

**Current State:** Monte Carlo treats GARCH parameters (alpha, beta, gamma, xi, rho) as known constants.

**Proposed:** Bootstrap the GARCH fit 20-50 times on resampled returns, extract distributions of parameters, and sample from those distributions in each MC scenario.

**Why:** Model parameter uncertainty is substantial. The difference between GARCH persistence=0.96 and 0.99 is the difference between 10-day and 100-day vol half-life, dramatically affecting path distributions.

---

### M10 — Automatic Block Bootstrap Size Selection [Medium Effort]

**Area:** Monte Carlo simulation
**Files:** `monte_carlo.py:52`, `engine_config.yaml`

**Current State:** Fixed `block_bootstrap_size: 21` in config.

**Proposed:** Implement Politis-Romano (2004) automatic block length selection, which optimizes block size based on the autocorrelation structure of the actual return data.

**Why:** 21-day blocks may break volatility clusters that last 40-60 days, losing the serial correlation structure that block bootstrap is designed to preserve.

---

### M11 — Crash Timing Classifier Evaluation [Low Effort]

**Area:** Evaluation
**Files:** `backtest.py`, `evaluation/metrics.py`

**Current State:** `CrashTimingClassifier` predictions are shown in the PDF report but never evaluated against actual crash timing in the backtest.

**Proposed:** Add a timing accuracy metric: for each historical crash, check if the timing classifier's highest-probability window contained the actual crash onset. Report as "timing hit rate."

**Why:** Without evaluation, the timing predictions could be random noise presented as actionable intelligence.

---

### M12 — Regime-Conditional BSS and AUC [Low Effort]

**Area:** Evaluation
**Files:** `evaluation/metrics.py`, `backtest.py`

**Current State:** BSS is computed globally across all backtest predictions.

**Proposed:** Stratify BSS, AUC, and calibration error by HMM regime. Report "BSS during Bull," "BSS during Bear," "BSS during Crisis."

**Why:** A model that works only in Bull markets (70% of days) will look good overall but fail when it matters most. Regime-conditional metrics expose this.

---

## SECTION 3: NOVEL IMPROVEMENTS

### N1 — Conformal Prediction for Coverage-Guaranteed Crash Intervals

**Description:** Replace ad-hoc p10/p90 quantile predictions with split conformal prediction intervals. Given a target coverage (e.g., 90%), conformal prediction provides mathematically guaranteed finite-sample coverage on exchangeable data, regardless of the underlying distribution.

**Why for crash prediction:** Current quantile predictions have no coverage guarantee — the "80% interval" may actually contain 60% of outcomes. For a financial product, this distinction matters: "our model says 20-40% crash probability with 90% guaranteed coverage" is fundamentally different from "our model guesses 20-40%."

**Complexity:** Medium
**Files:** `crash_model.py` (add conformal wrapper), `return_model.py`, `evaluation/metrics.py` (coverage test)

---

### N2 — Regime-Conditional Calibration

**Description:** Instead of a single Platt scaling calibrator, train separate calibrators per HMM regime (Bull/Bear/Crisis). At inference time, compute the regime-weighted blend of calibrated probabilities.

**Why for crash prediction:** The mapping from raw LightGBM score to true crash probability is regime-dependent. A raw score of 0.3 in a Bull market (low base rate) means something fundamentally different than 0.3 in a Bear market (high base rate). A single calibrator cannot capture this — it learns an average mapping that's wrong in all regimes.

**Complexity:** Low
**Files:** `crash_model.py:383-396` (train per-regime calibrators)

---

### N3 — Temporal Attention Mechanism for LSTM

**Description:** Replace the LSTM's final-hidden-state output (`h_n[-1]` at line 166 of `sequence_model.py`) with a self-attention layer that computes weighted importance over all 60 timesteps. This allows the model to "focus" on specific past days rather than relying on the recency bias of LSTM's forget gate.

**Why for crash prediction:** Pre-crash stress build-up is non-monotonic. VIX might spike 45 days ago, normalize for 3 weeks, then spike again. The LSTM's recency bias dilutes the first spike's signal. Attention would preserve it by learning that "VIX spikes in the past 60 days" matter regardless of when they occurred.

**Complexity:** Medium
**Files:** `sequence_model.py:111-167` (add attention layer to `LSTMCrashModel`)

---

### N4 — MIDAS Mixed-Frequency Features

**Description:** Instead of forward-filling monthly FRED data to daily frequency (which introduces stale data and potential look-ahead bias), use a MIDAS (Mixed Data Sampling) approach. MIDAS uses polynomial distributed lag weights to combine weekly/monthly macro data with daily market data at their native frequencies.

**Why for crash prediction:** Forward-fill creates up to a month of stale data per FRED feature. MIDAS eliminates this while extracting richer information from the frequency mismatch — it can learn that "the most recent monthly unemployment reading matters more than the one 3 months ago" without imposing that assumption.

**Complexity:** High
**Files:** `features.py:332-351`, new `ml/midas.py`

---

### N5 — Adversarial Distribution Shift Detection

**Description:** Train a domain classifier (simple logistic regression) to distinguish "training set features" from "current features." If the classifier achieves >60% accuracy, the current market conditions are distributionally different from the training data, and model confidence should be reduced.

**Why for crash prediction:** The current anomaly detector (`anomaly_detector.py`) uses Isolation Forest, which flags individual feature outliers. But it can miss *joint* distribution shifts where each feature is individually normal but their combination is unprecedented (e.g., high VIX + positive momentum + inverted yield curve + low credit spreads simultaneously). A domain classifier captures these multivariate shifts.

**Complexity:** Medium
**Files:** `anomaly_detector.py` (add `DomainShiftDetector` class)

---

### N6 — Ensemble Disagreement as Uncertainty Signal

**Description:** Compute the standard deviation of crash probabilities across the 4 models (LGB, XGB, LSTM, TCN). When disagreement is high (>15pp spread), this indicates model uncertainty that should widen the confidence interval and trigger a flag in the PDF report.

**Why for crash prediction:** If LGB says 10% and LSTM says 60%, the ensemble mean of 35% is misleading. The 50pp spread is itself an important signal — it typically occurs when the market is transitioning between regimes (temporal models detect it before tree models) or when one model class is overfitting to noise. Practitioners need to see the disagreement, not just the average.

**Complexity:** Low
**Files:** `meta_stacker.py` (add `model_disagreement()` method), `main.py` (display), `reporting/pdf_report.py`

---

### N7 — Options-Implied Crash Probability from Full VIX Term Structure

**Description:** Go beyond the current VIX/VIX3M ratio by computing crash probability from the full VIX term structure (VIX9D, VIX, VIX3M, VIX6M). The shape of the curve — not just the spot level — distinguishes "fear of immediate crash" (steep inversion at the short end) from "elevated uncertainty about the future" (flat elevated curve).

**Why for crash prediction:** The current `vix_term_structure_ratio` is a single ratio. The full term structure contains richer information: deep backwardation at the 9-day point has preceded nearly every flash crash since 2010. VIX6M data is available free from CBOE via yfinance.

**Complexity:** Medium
**Files:** `data/fetchers.py` (fetch VIX9D, VIX6M), `features.py` (compute term structure slope features)

---

### N8 — Cross-Asset Correlation Regime Change Detection

**Description:** Track rolling correlations between SPX-TLT, SPX-Gold, SPX-USD in a 3D correlation space. Use a multivariate changepoint detector to identify when the correlation structure breaks down (all correlations approaching 1.0 = "everything sells off together" = systemic risk). More systematic than the current `risk_off_binary_21d` feature.

**Why for crash prediction:** In normal markets, stocks and bonds are negatively correlated (diversification works). Before major crashes, cross-asset correlations spike toward 1.0 (diversification fails). This structural break — the *breakdown of the correlation regime* — is one of the strongest crash precursors and is not well captured by individual cross-asset features or simple pairwise correlations.

**Complexity:** Medium
**Files:** `features.py` (add rolling correlation matrix eigenvalue features), `anomaly_detector.py` (multivariate changepoint)

---

## SUMMARY TABLE

### Bugs by Severity

| ID | Severity | File | One-Line Description |
|---|---|---|---|
| B1 | CRITICAL | `meta_stacker.py` | Fabricated 0.12 values for missing models corrupt ensemble |
| B3 | HIGH | `crashes.py` | Ongoing crash at dataset end silently dropped |
| B4 | HIGH | `hmm_regimes.py` | HMM standardization params not saved for incremental scoring |
| B7 | MEDIUM | `features.py` | FRED forward-fill ignores ~1 month publication lag |
| B8 | MEDIUM | `features.py` | CAPE proxy uses price average, not earnings |
| B5 | MEDIUM | `regimes.py` | Mixes log returns (drift) with simple returns (vol) |
| B6 | MEDIUM | `metrics.py` | Sharpe annualization hardcoded sqrt(4) |
| B2 | MEDIUM | `sequence_model.py` | predict_individual returns constant for None models |
| B9 | LOW | `crash_model.py` | Lookup table blend dampens ML signal during regime shifts |
| B10 | LOW | `backtest.py` | Missing models replaced with LGB prediction |
| B11 | LOW | `backtest.py` | XGB 6m/3m OOS falls back to LGB |
| B12 | LOW | `sequence_model.py` | DataLoader shuffle not seeded |

### Methodology Improvements by Effort

| ID | Effort | Area | Description |
|---|---|---|---|
| M1 | Low | Walk-forward | Centralize purge gap definitions |
| M2 | Low | Walk-forward | Add embargo period after training window |
| M4 | Low | Sample weighting | Exponential temporal decay |
| M11 | Low | Evaluation | Crash timing accuracy metric |
| M12 | Low | Evaluation | Regime-conditional BSS/AUC |
| M3 | Medium | Calibration | Fix Platt scaling double-dip |
| M5 | Medium | Features | Feature redundancy reduction |
| M6 | Medium | Labels | Dynamic crash threshold |
| M8 | Medium | Regime model | Online HMM in backtest |
| M9 | Medium | Uncertainty | Bootstrap over GARCH parameters |
| M10 | Medium | Monte Carlo | Auto block bootstrap sizing |
| M7 | High | Walk-forward | Combinatorial purged CV |

### Novel Improvements by Complexity

| ID | Complexity | Idea |
|---|---|---|
| N2 | Low | Regime-conditional calibration |
| N6 | Low | Ensemble disagreement reporting |
| N1 | Medium | Conformal prediction intervals |
| N3 | Medium | Temporal attention for LSTM |
| N5 | Medium | Adversarial distribution shift detection |
| N7 | Medium | Full VIX term structure features |
| N8 | Medium | Cross-asset correlation regime detection |
| N4 | High | MIDAS mixed-frequency features |
