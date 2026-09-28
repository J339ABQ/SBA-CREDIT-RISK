# %% [markdown]
# # 02 - Modelling: leakage-controlled PD model, out-of-time validation, calibration, explainability
#
# **Pipeline of this notebook**
# 1. Out-of-time split and why (economic-cycle drift)
# 2. Leak-free preprocessing `Pipeline` (fit on train only)
# 3. Hyper-parameter tuning with **forward-chaining CV inside the training period**
# 4. Baseline logistic regression vs gradient boosting (LightGBM)
# 5. Discrimination on the untouched test vintages: ROC-AUC, KS, Gini
# 6. Calibration: assessment, correction, and behaviour under drift
# 7. Explainability: permutation importance + SHAP, in plain English
# 8. Persist the model artefact used by the API
#
# > Data note: results are computed on whatever `data/raw/` holds - the synthetic FOIA-schema stand-in unless real SBA files were dropped in
# > `data/raw/real/` (see README). The banner in the next cell says which.

# %%
import sys, inspect
sys.path.insert(0, "..")
import warnings; warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from IPython.display import Markdown, display

from src import config, evaluation as ev, explain, features, modeling, train
from src.evaluation import PALETTE, save_fig, set_style

set_style()
pd.options.display.float_format = "{:,.4f}".format
display(Markdown(f"**Data source loaded: `{train.data_source().upper()}`**"))

# %% [markdown]
# ## 1. Out-of-time split (and why not a random split)
#
# Credit risk models are deployed on **future** loans, and the economy in the future is not the economy of the past: default rates move with
# the cycle (see the vintage chart in notebook 01). A random split scatters every vintage across train and test, so the test set shares
# the training set's macro conditions and the model looks better than it will be in production. We instead train on early vintages and
# evaluate on later ones:
#
# | Window | Approval years | Use |
# |---|---|---|
# | Train | 2010-2016 | fit models, tune hyper-parameters (forward-chaining CV *inside* this window) |
# | Validation | 2017-2018 | fit the probability calibrator, choose the approval cut-off |
# | Test | 2019-2021 | **touched once** for final reporting; includes the COVID-shock vintage |

# %%
splits = train.prepare()
s = splits
summary = pd.DataFrame({name: {"approval years": f"{getattr(config, name.upper() + '_YEARS')[0]}-{getattr(config, name.upper() + '_YEARS')[1]}",
                               "loans": len(getattr(s, name)), "defaults": int(getattr(s, name)['default'].sum()),
                               "default rate": getattr(s, name)['default'].mean()} for name in ("train", "valid", "test")}).T
display(summary)
print("Forward-chaining CV folds inside the training window (train on all earlier years, validate on the next):")
for tr_i, va_i in modeling.forward_chaining_splits(s.train["approval_year"]):
    print(f"  train years <= {s.train['approval_year'].iloc[tr_i].max()}  ({len(tr_i):>6,} loans)  ->  validate on {s.train['approval_year'].iloc[va_i].max()}  ({len(va_i):>5,} loans)")

# %% [markdown]
# ## 2. Leak-free preprocessing
# Feature engineering is a pure function of one row (`features.make_features`, origination-time fields only - see the leakage registry in notebook 01).
# All *fitted* preprocessing (imputation medians, scaler statistics, one-hot categories) lives **inside** the sklearn `Pipeline`, so cross-validation and
# out-of-time scoring re-fit it on training folds only. Logistic regression gets scaling; trees do not need it.

# %%
print(inspect.getsource(features.build_preprocessor))
print("Model inputs:", len(features.FEATURES), "engineered features ->", features.NUMERIC_FEATURES, features.CATEGORICAL_FEATURES)

# %% [markdown]
# ## 3. Tuning + training (train window only)
# `train.train()` runs the same stages the CLI runs (`python -m src.train`): forward-chaining CV grid for the logistic baseline (regularisation `C`),
# 20-iteration randomised search for LightGBM, then calibration and cut-off selection on the validation window. The search space is shown below.

# %%
print(inspect.getsource(modeling.tune_gbm))

# %%
res = train.train(n_iter=20, splits=splits)
m = res["metrics"]
cv = pd.DataFrame(res["gbm_search"].cv_results_).sort_values("rank_test_score").head(5)
cols = ["mean_test_score", "std_test_score"] + [c for c in cv.columns if c.startswith("param_")]
display(cv[cols].rename(columns=lambda c: c.replace("param_clf__", "")).round(4).reset_index(drop=True))
print(f"Forward-chaining CV AUC  ->  logistic: {m['cv_auc']['logistic']:.4f} (C={res['base_search'].best_params_['clf__C']})   |   LightGBM: {m['cv_auc']['lightgbm']:.4f}")

# %% [markdown]
# ## 4-5. Discrimination on validation and out-of-time test
# * **ROC-AUC** - probability a random defaulter is scored riskier than a random non-defaulter.
# * **Gini** = 2*AUC - 1 (the industry-standard rescaling).
# * **KS** - maximum gap between the cumulative score distributions of defaulters and non-defaulters.

# %%
rows = []
for name in ("valid", "test"):
    for k, label in (("logistic", "Logistic regression (baseline)"), ("lightgbm_raw", "LightGBM"), ("lightgbm_calibrated", "LightGBM + calibration")):
        r = m[name][k]
        rows.append({"window": name, "model": label, "AUC": r["auc"], "Gini": r["gini"], "KS": r["ks"], "Brier": r["brier"], "mean PD": r["mean_pd"], "observed rate": r["default_rate"]})
tbl = pd.DataFrame(rows)
display(tbl.style.hide(axis="index").format({c: "{:.3f}" for c in ["AUC", "Gini", "KS", "Brier", "mean PD", "observed rate"]}))
lo, hi = m["test"]["auc_ci95_lightgbm"]
print(f"Test AUC (LightGBM) = {m['test']['lightgbm_raw']['auc']:.3f}, 95% bootstrap CI [{lo:.3f}, {hi:.3f}]. The lift over the baseline is small: "
      f"{m['test']['lightgbm_raw']['auc'] - m['test']['logistic']['auc']:+.3f} AUC.")

y_test = s.y("test").to_numpy()
f1 = ev.plot_roc({"Logistic": (y_test, res["raw"]["test"]["lr"]), "LightGBM": (y_test, res["raw"]["test"]["gbm"])}); save_fig(f1, "model_roc"); plt.show()
f2 = ev.plot_ks(y_test, res["raw"]["test"]["gbm"]); save_fig(f2, "model_ks"); plt.show()

# %% [markdown]
# **Reading it (honestly).** The linear baseline is already strong, and gradient boosting is **not materially better** here: LightGBM has a tiny edge on
# validation and test AUC (+0.002 / +0.004, well inside the bootstrap CI), while the logistic model actually wins the forward-chaining CV. The
# tabular-data literature says the same thing for well-engineered, low-signal credit data. We still ship LightGBM because (a) it was ahead on the
# validation window - the window we are allowed to use for model choice, not the test window, (b) it handles missing values and non-monotone effects
# without hand-built bins, and (c) LightGBM gives exact per-loan reason codes for the API. The logistic model remains a fully legitimate
# choice where a scorecard-style, regulator-friendly model is preferred - the trade-off costs almost nothing in ranking power.
#
# Validation and test AUC are close, so *ranking* (who is riskier) transfers well across the cycle. Whether the *probability level* transfers is a
# separate question, answered next.
#
# ### Rank-ordering by decile

# %%
f3 = ev.plot_decile_lift(y_test, res["cal_test"]); save_fig(f3, "model_decile_lift"); plt.show()
t = ev.calibration_table(y_test, res["cal_test"])
print(f"Top-decile default rate is {t['obs_rate'].iloc[-1] / t['obs_rate'].iloc[0]:.1f}x the bottom decile; top 10% of scores capture "
      f"{(t['n'].iloc[-1] * t['obs_rate'].iloc[-1]) / y_test.sum():.0%} of all defaults.")

# %% [markdown]
# ## 6. Calibration: do predicted PDs match observed default rates?
# A PD that ranks well but is off in level produces wrong expected losses and wrong cut-offs. Three views:
# 1. Reliability curve (deciles) on the **validation** window, before vs after calibration.
# 2. The same on the **test** window - the honest check.
# 3. Predicted vs observed **by vintage** - the drift diagnostic.
#
# The calibrator (Platt vs isotonic) is chosen by cross-fitting across the two validation vintages (fit on 2017, score 2018 and vice versa), then fit on both.

# %%
print("Calibrator selection (mean out-of-vintage Brier within the validation window):", {k: round(v, 5) for k, v in m["calibration_cv_brier"].items()}, "->", m["calibration_method"])
yv = s.y("valid").to_numpy()
fv = ev.plot_calibration({"LightGBM raw": (yv, res["raw"]["valid"]["gbm"]), f"+ {m['calibration_method']}": (yv, res["cal_valid"])}, "Calibration - validation (2017-18)")
save_fig(fv, "model_calibration_valid"); plt.show()
ft = ev.plot_calibration({"LightGBM raw": (y_test, res["raw"]["test"]["gbm"]), f"+ {m['calibration_method']}": (y_test, res["cal_test"])}, "Calibration - out-of-time test (2019-21)")
save_fig(ft, "model_calibration_test"); plt.show()

# %%
bv = pd.DataFrame(m["test"]["by_vintage"]).set_index("approval_year")
bv["gap (pp)"] = (bv["predicted"] - bv["observed"]) * 100
display(bv.style.format({"observed": "{:.2%}", "predicted": "{:.2%}", "gap (pp)": "{:+.2f}", "n": "{:,.0f}"}))
slope, icpt = m["test"]["calibration_slope_intercept"]
print(f"Test ECE: raw {m['test']['ece_raw']:.4f} | calibrated {m['test']['ece_calibrated']:.4f}.  Calibration slope {slope:.2f}, intercept {icpt:+.2f} (ideal 1.00 / 0.00).")
print(f"Score PSI train->test: {m['test']['psi_score_train_vs_test']:.3f}  (<0.10 = population stable: the *mix* of borrowers barely moved).")
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(bv.index, bv["observed"] * 100, "o-", color=PALETTE["bad"], lw=2, label="Observed default rate")
ax.plot(bv.index, bv["predicted"] * 100, "s--", color=PALETTE["cal"], lw=2, label="Mean calibrated PD")
ax.set(title="Test vintages: predicted vs observed", ylabel="%", xlabel="Approval year"); ax.set_xticks(bv.index); ax.legend()
save_fig(fig, "model_calibration_by_vintage"); plt.show()

# %% [markdown]
# ### Written calibration assessment
# * **Within the calibration window** the reliability curve tracks the diagonal after Platt scaling (mean PD equals the observed rate by construction).
# * **Out of time, the levels are off in the way economic-cycle drift predicts.** The model was trained on 2010-16 and calibrated on the benign 2017-18
#   vintages, so it expects roughly 6-7% defaults; the 2019-21 vintages actually ran higher, with the worst gap in **2020**. The *score PSI* is small
#   (borrower mix stable) and AUC holds, so this is a shift in the **base rate** (the macro environment), not a broken model.
# * Note the raw LightGBM probabilities were already reasonably calibrated; calibrating on a benign window *improves in-window fit but does not protect
#   against a later regime change* - and can look slightly worse on a stressed test window. We report that rather than hide it.
# * **What we do about it:** (1) the cut-off analysis in notebook 03 is based on calibrated PDs but is stress-tested with a PD multiplier; (2) production
#   guidance is to monitor predicted-vs-observed by vintage and re-calibrate the intercept as soon as new vintages mature, or add a macro overlay.
#
# ## 7. Explainability
# ### Permutation importance (AUC drop when a feature is shuffled, on the test window)

# %%
imp = ev.permutation_importance_auc(res["gbm"], s.X("test"), y_test, n_repeats=5)
display(imp.head(12).style.hide(axis="index").format({"auc_drop": "{:.4f}", "std": "{:.4f}"}))
top = imp.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.barh(top["feature"], top["auc_drop"], xerr=top["std"], color=PALETTE["gbm"])
ax.set(title="Permutation importance (test window)", xlabel="AUC drop when feature is shuffled")
save_fig(fig, "model_feature_importance"); plt.show()

# %% [markdown]
# ### SHAP: direction and size of each driver
# SHAP values decompose each loan's predicted log-odds into per-feature contributions. Encoded columns (one-hot levels) are summed back to
# their business variable for the bar chart; the beeswarm shows direction (red = high feature value).

# %%
Xs = s.X("test").sample(4000, random_state=0)
prep = res["gbm"].named_steps["prep"]
Xt_s = prep.transform(Xs)
explainer = shap.TreeExplainer(res["gbm"].named_steps["clf"])
sv = explainer.shap_values(Xt_s)
sv = sv[1] if isinstance(sv, list) else sv
grouped = explain.group_contributions(sv, Xt_s.columns)
mean_abs = grouped.abs().mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4.4))
mean_abs.head(12).iloc[::-1].plot.barh(ax=ax, color=PALETTE["gbm"])
ax.set(title="Mean |SHAP| by business variable (log-odds)", xlabel="mean |SHAP value|")
save_fig(fig, "model_shap_importance"); plt.show()
plt.figure(figsize=(8, 5.5))
shap.summary_plot(sv, Xt_s, max_display=14, show=False, plot_size=None)
plt.title("SHAP summary (encoded features)", fontweight="bold")
save_fig(plt.gcf(), "model_shap_summary"); plt.show()

# %% [markdown]
# ### Plain-English interpretation of the top drivers
# Read together with the EDA (and remember these are SHAP/permutation results on the data loaded - synthetic unless real files are present):
# 1. **Business age** (largest driver) - start-ups and businesses under two years old carry the highest risk; established firms (>2 years) are the safest.
#    The lender is lending on a track record that either exists or does not.
# 2. **Loan term** - short-term loans (working capital) are riskier than long-term loans (typically real-estate/equipment backed); the effect is non-monotonic.
# 3. **Loan size** - the smallest loans (and to a lesser degree the very largest) default more; mid-sized loans are safest.
# 4. **Industry (NAICS sector)** - accommodation/food, arts/recreation and construction push PD up; health care, real estate and professional services pull it down.
# 5. **Processing method and collateral** - Express (streamlined, less documentation) and un-collateralised loans are riskier than PLP/collateralised ones.
# 6. **Rate spread over prime** - the lender's own price carries real signal (they charge more for risk they can see) but is far from sufficient on its own.
# 7. **State and rate type** - small effects.
#
# **Caveat that matters:** on the synthetic stand-in these drivers reflect how the generator was built (stylised facts from the SBA literature), so this
# ranking demonstrates the *explanation workflow*; on real FOIA data the ordering is an empirical finding and should be re-read from these same plots.
#
# ## 8. Save the model artefact used by the API
# The `PDModel` bundles the fitted pipeline, the calibrator, the recommended cut-off (selected on validation in notebook 03) and metadata.

# %%
train.save(res)
print("saved:", config.MODEL_PATH.relative_to(config.ROOT), "and", config.METRICS_PATH.relative_to(config.ROOT))
print("Recommended cut-off (calibrated PD <=):", f"{res['cutoff']:.2%}", "- derived on the validation window; rationale in notebook 03.")
