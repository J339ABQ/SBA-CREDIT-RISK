"""Business decision layer: expected loss, approval-cut-off economics, pricing, rules baseline.

EL = PD x LGD x EAD
  PD  : model (calibrated 36-month probability of charge-off)
  EAD : exposure at default as a fraction of GrossApproval. DATA-DERIVED: mean(GrossChargeOffAmount /
        GrossApproval) over charged-off training loans (the charge-off amount is the balance outstanding
        when the loan was written off). Falls back to 0.75 if unavailable.
  LGD : NOT in the FOIA data (recoveries are not published). ASSUMED 45% - the Basel II foundation-IRB
        supervisory LGD for senior unsecured corporate exposures - and stress-tested from 30% to 70%.
        Real SBA-loan recoveries depend on collateral, guarantee and liquidation; treat 45% as a placeholder
        to be replaced with the lender's own workout data.
Everything is expressed per dollar of GrossApproval so it is scale-free, then aggregated in dollars.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Economics:
    lgd: float = 0.45               # ASSUMPTION (see module docstring)
    ead_factor: float = 0.75        # overwritten by estimate_ead_factor() when charge-off amounts are available
    net_margin: float = 0.015       # ASSUMPTION: annual risk-free-adjusted margin after funding, opex and cost of capital, on avg balance
    avg_balance_factor: float = 0.85  # ASSUMPTION: avg outstanding / GrossApproval over the horizon (amortisation)
    horizon_years: float = 3.0      # matches the 36-month default window

    @property
    def revenue_rate(self) -> float:
        """Expected net revenue per $ of approval over the horizon (before credit losses)."""
        return self.net_margin * self.avg_balance_factor * self.horizon_years

    def el_rate(self, pd_) -> np.ndarray:
        """Expected loss per $ approved = PD x LGD x EAD-factor."""
        return np.asarray(pd_) * self.lgd * self.ead_factor

    @property
    def breakeven_pd(self) -> float:
        return self.revenue_rate / (self.lgd * self.ead_factor)

    def to_dict(self) -> dict:
        return {**asdict(self), "revenue_rate": self.revenue_rate, "breakeven_pd": self.breakeven_pd}


def estimate_ead_factor(train: pd.DataFrame) -> float:
    d = train[(train["default"] == 1) & train["GrossChargeOffAmount"].notna() & (train["GrossApproval"] > 0)]
    if len(d) < 50:
        return 0.75
    return float(np.clip((d["GrossChargeOffAmount"] / d["GrossApproval"]).mean(), 0.2, 1.0))


def realised_loss(df: pd.DataFrame, econ: Economics) -> np.ndarray:
    """Ex-post loss $: charge-off amount (actual EAD) x assumed LGD for defaulted loans, else 0."""
    ead = df["GrossChargeOffAmount"].fillna(df["GrossApproval"] * econ.ead_factor).to_numpy()
    return np.where(df["default"].to_numpy() == 1, ead * econ.lgd, 0.0)


def cutoff_curve(df: pd.DataFrame, pd_cal: np.ndarray, econ: Economics, cutoffs=None) -> pd.DataFrame:
    """For each PD cut-off (approve iff PD <= cut-off) compute volume, risk, loss and profit trade-offs."""
    if cutoffs is None:
        cutoffs = np.unique(np.round(np.quantile(pd_cal, np.linspace(0.02, 1.0, 99)), 4))
    gross = df["GrossApproval"].to_numpy()
    y = df["default"].to_numpy()
    loss = realised_loss(df, econ)
    exp_loss = econ.el_rate(pd_cal) * gross
    revenue = econ.revenue_rate * gross
    rows = []
    for c in cutoffs:
        a = pd_cal <= c
        if a.sum() == 0:
            continue
        rows.append({
            "cutoff": c, "approval_rate": a.mean(), "approved_n": int(a.sum()),
            "approved_dollars": gross[a].sum(), "default_rate": y[a].mean(),
            "expected_loss": exp_loss[a].sum(), "realised_loss": loss[a].sum(),
            "el_rate_of_approved": exp_loss[a].sum() / gross[a].sum(),
            "expected_profit": (revenue[a] - exp_loss[a]).sum(),
            "realised_profit": (revenue[a] - loss[a]).sum(),
        })
    return pd.DataFrame(rows)


def recommend_cutoff(curve: pd.DataFrame, tolerance: float = 0.01) -> pd.Series:
    """Flat-maximum rule: the *tightest* cut-off whose expected profit is within `tolerance` of the maximum.

    Expected profit is very flat near its peak, and it depends on margin/LGD assumptions we cannot verify.
    Giving up <=1% of modelled profit buys a materially lower default rate and loss, so we take the
    conservative end of the plateau instead of the knife-edge optimum.
    """
    best = curve["expected_profit"].max()
    ok = curve[curve["expected_profit"] >= (1 - tolerance) * best]
    return ok.loc[ok["cutoff"].idxmin()]


def rules_baseline_decline(features: pd.DataFrame) -> np.ndarray:
    """A plausible manual policy: decline start-ups, and new restaurants/hotels (NAICS 72)."""
    startup = features["business_age"].fillna("").str.startswith("Startup")
    new_food = features["business_age"].fillna("").str.startswith("New_Business") & (features["naics_sector"] == "72")
    return (startup | new_food).to_numpy()


def compare_strategies(df: pd.DataFrame, pd_cal: np.ndarray, decline_rules: np.ndarray, econ: Economics,
                       model_cutoff: float) -> pd.DataFrame:
    """Approve-all vs rules policy vs model at (a) the recommended cut-off and (b) the rules policy's own approval rate."""
    gross = df["GrossApproval"].to_numpy()
    y = df["default"].to_numpy()
    loss = realised_loss(df, econ)
    exp_loss = econ.el_rate(pd_cal) * gross
    rev = econ.revenue_rate * gross

    def summarise(name, a):
        return {"strategy": name, "approval_rate": a.mean(), "approved_dollars": gross[a].sum(),
                "default_rate": y[a].mean(), "expected_loss": exp_loss[a].sum(), "realised_loss": loss[a].sum(),
                "loss_rate_realised": loss[a].sum() / gross[a].sum(), "realised_profit": (rev[a] - loss[a]).sum()}

    approve_all = np.ones(len(df), bool)
    rules = ~decline_rules
    k = int(round(rules.mean() * len(df)))
    matched = np.zeros(len(df), bool)
    matched[np.argsort(pd_cal, kind="stable")[:k]] = True
    at_cut = pd_cal <= model_cutoff
    out = pd.DataFrame([
        summarise("Approve everyone", approve_all),
        summarise("Rules policy (decline start-ups & new NAICS-72)", rules),
        summarise("Model, same approval rate as rules", matched),
        summarise(f"Model @ recommended cut-off ({model_cutoff:.1%})", at_cut),
    ])
    base = out.loc[0, "loss_rate_realised"]
    out["loss_rate_vs_approve_all"] = out["loss_rate_realised"] / base - 1
    return out


def pricing_table(df: pd.DataFrame, pd_cal: np.ndarray, econ: Economics, n_bands: int = 5,
                  actual_spread: np.ndarray | None = None) -> pd.DataFrame:
    """Risk-based pricing: annualised expected-loss charge each PD band must earn on top of the base margin."""
    d = pd.DataFrame({"pd": pd_cal, "gross": df["GrossApproval"].to_numpy()})
    if actual_spread is not None:
        d["spread"] = actual_spread
    d["band"] = pd.qcut(d["pd"], n_bands, labels=[f"Q{i}" for i in range(1, n_bands + 1)])
    g = d.groupby("band", observed=True)
    t = g.agg(loans=("pd", "size"), mean_pd=("pd", "mean")).reset_index()
    # annualised EL as % of average balance:  PD*LGD*EAD / (avg_balance_factor * horizon)
    t["annual_el_pct"] = 100 * t["mean_pd"] * econ.lgd * econ.ead_factor / (econ.avg_balance_factor * econ.horizon_years)
    t["required_spread_pct"] = t["annual_el_pct"] + 100 * econ.net_margin
    if actual_spread is not None:
        t["actual_spread_over_prime_pct"] = g["spread"].mean().to_numpy()
    return t
