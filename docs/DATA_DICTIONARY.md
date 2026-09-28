# Data dictionary

## Source
SBA **7(a)** and **504** loan-level FOIA data - <https://data.sba.gov/dataset/7-a-504-foia> (landing page: <https://www.sba.gov/about-sba/open-government/foia>).
Public, updated quarterly, one row per approved loan.

> **Provenance note.** The sandbox this project was built in could not reach `sba.gov` / `data.sba.gov` (HTTP 403 from the network policy), so the field list below is
> written from the published FOIA schema **as I know it, not verified against a fresh download**. `src/data_prep.py` matches headers case-insensitively, maps known spelling
> variants (e.g. `TerminMonths` / `TermInMonths`) and fills any absent optional column with NaN, so a slightly different extract still loads. The committed results come from a
> **synthetic stand-in** (`src/synthetic.py`) that uses these same column names and value vocabularies.

## Raw columns used

| Raw column | Meaning | Used as | Known at origination? |
|---|---|---|---|
| `Program` | 7(a) vs 504 | feature `program` | yes |
| `ApprovalDate` | date SBA approved the loan | vintage for the split; year used only to de-trend the rate | yes (split only) |
| `FirstDisbursementDate` | date of first disbursement | **anchors the 36-month outcome window** | **no** - never a feature |
| `GrossApproval` | approved loan amount, USD | `log_gross_approval`, denominators, EL dollars | yes |
| `SBAGuaranteedApproval` | SBA-guaranteed portion, USD | `guarantee_pct` | yes |
| `TermInMonths` (`TerminMonths` in 7(a)) | loan term | `term_months` | yes |
| `InitialInterestRate` | rate at origination, % | `interest_rate`, `rate_spread` (rate minus prime that year) | yes |
| `FixedOrVariableInterestInd` | F / V | `rate_type` | yes |
| `ProcessingMethod` | PLP / Regular / Express / ... | `processing_method` | yes |
| `NaicsCode` | industry code | first 2 digits -> `naics_sector` | yes |
| `BorrState` | borrower state | `borrower_state` | yes |
| `BusinessType` | corporation / individual / partnership | `business_type` | yes |
| `BusinessAge` | start-up / new (<=2y) / existing / change of ownership / unanswered | `business_age` | yes |
| `CollateralInd` | collateral pledged | `collateral_ind` | yes |
| `FranchiseCode` | franchise identifier | `is_franchise` (present/absent) | yes |
| `NonProfit` | non-profit flag (some extracts) | `non_profit` | yes |
| `JobsSupported` | jobs created/retained (projected) | `jobs_supported`, `jobs_per_100k` | yes |
| `ThirdPartyDollars` | 504: third-party lender amount | `third_party_ratio` | yes |
| `LoanStatus` | PIF / CHGOFF / EXEMPT / CANCLD / COMMIT | **label construction + filtering only** | **no** |
| `ChargeOffDate` | date charged off | **label construction only** | **no** |
| `GrossChargeOffAmount` | amount charged off | EAD estimate + realised loss in the *business* evaluation only; never a model input | **no** |
| `PaidInFullDate`, `SoldSecMrktInd`, `AsOfDate` | post-origination / extract fields | excluded (see leakage) | **no** |

Free-text and identifier columns (`BorrName`, addresses, bank names/IDs) are not used.

## Target variable

```
keep    : LoanStatus in {PIF, CHGOFF, EXEMPT}
          AND ApprovalDate, FirstDisbursementDate present
          AND FirstDisbursementDate + 36 months <= AS_OF_DATE        (full window observed; otherwise right-censored)
default : LoanStatus == "CHGOFF" AND ChargeOffDate <= FirstDisbursementDate + 36 months   -> 1
          everything else kept                                                             -> 0
```

* `CANCLD` (approved, never funded) and `COMMIT` (not yet funded) are dropped: they were never exposed to default.
* `EXEMPT` denotes loans still in force whose status SBA does not disclose - treated as not defaulted *within the window*, which is valid only because the window has fully elapsed.
* Charge-offs **after** month 36 are labelled 0 for this target (the count is reported in notebook 01). This is a deliberate horizon choice: the model predicts *36-month* PD.
* Implemented in `src/data_prep.py::build_target`; edge cases (window boundary, censoring, dropped statuses) are unit-tested in `tests/test_data_prep.py`.

## Features excluded due to leakage risk, and why

| Excluded | Why |
|---|---|
| `LoanStatus` | The outcome itself. |
| `ChargeOffDate` | Exists only for defaulted loans; the label is derived from it. |
| `GrossChargeOffAmount` | Realised loss; non-null only after default. |
| `PaidInFullDate` | Exists only after repayment (survivor marker). |
| `SoldSecMrktInd` | Secondary-market sale happens after origination; investors buy performing loans. |
| `AsOfDate` | Extract date; correlates with loan age, not with borrower quality. |
| `FirstDisbursementDate` | Post-approval event; used only to anchor the window. |
| Calendar year of approval | Used for the out-of-time split and to de-trend the rate; a model must not memorise vintages. |
| Names / addresses / bank identifiers | Memorisation risk, PII-adjacent, no generalisable signal. |
| Loan age, balances, delinquency, servicing flags | Not part of the origination record. |

Enforcement: `src/features.py` (`LEAKAGE_EXCLUDED`, `assert_no_leakage`) and `tests/test_features.py` (including a test that flips every post-origination field and asserts the feature matrix does not change).
