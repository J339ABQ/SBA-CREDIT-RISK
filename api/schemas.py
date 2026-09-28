"""Request / response models for the PD scoring API (validation lives here)."""
from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BusinessAge = Literal["Existing or more than 2 years old", "New Business or 2 years or less",
                      "Startup, Loan Funds will Open Business", "Change of Ownership", "Unanswered"]


class LoanApplication(BaseModel):
    """Origination-time attributes of one loan application (field names follow the SBA FOIA columns)."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "program": "7A", "gross_approval": 150000, "sba_guaranteed_approval": 112500, "term_months": 120,
        "initial_interest_rate": 9.5, "fixed_or_variable": "V", "processing_method": "PLP", "naics_code": "722511",
        "borrower_state": "TX", "business_type": "CORPORATION", "business_age": "New Business or 2 years or less",
        "collateral_ind": "Y", "is_franchise": False, "non_profit": False, "jobs_supported": 6,
        "approval_date": "2019-06-15"}})

    program: Literal["7A", "504"] = "7A"
    gross_approval: float = Field(gt=0, le=20_000_000, description="Approved loan amount, USD")
    sba_guaranteed_approval: float = Field(ge=0, description="SBA-guaranteed portion, USD")
    term_months: int = Field(ge=1, le=480)
    initial_interest_rate: float | None = Field(default=None, ge=0, le=40, description="Percent; omit if unknown")
    fixed_or_variable: Literal["F", "V"] | None = None
    processing_method: str | None = Field(default=None, description="PLP / Regular / Express / Other / 504")
    naics_code: str | None = Field(default=None, pattern=r"^\d{2,6}$")
    borrower_state: str | None = Field(default=None, pattern=r"^[A-Za-z]{2}$")
    business_type: Literal["CORPORATION", "INDIVIDUAL", "PARTNERSHIP"] | None = None
    business_age: BusinessAge = "Unanswered"
    collateral_ind: Literal["Y", "N"] | None = None
    is_franchise: bool = False
    non_profit: bool = False
    jobs_supported: float = Field(default=0, ge=0, le=10_000)
    third_party_dollars: float | None = Field(default=None, ge=0, description="504 only: third-party lender amount")
    approval_date: dt.date = Field(default_factory=dt.date.today)

    @field_validator("borrower_state")
    @classmethod
    def _upper(cls, v):
        return v.upper() if v else v

    @model_validator(mode="after")
    def _guarantee_le_gross(self):
        if self.sba_guaranteed_approval > self.gross_approval:
            raise ValueError("sba_guaranteed_approval cannot exceed gross_approval")
        return self

    def to_raw_row(self) -> dict:
        """Map to the canonical raw (FOIA-style) columns that src.features.make_features expects."""
        return {
            "Program": self.program, "ApprovalDate": self.approval_date.isoformat(), "GrossApproval": self.gross_approval,
            "SBAGuaranteedApproval": self.sba_guaranteed_approval, "TermInMonths": self.term_months,
            "InitialInterestRate": self.initial_interest_rate, "FixedOrVariableInterestInd": self.fixed_or_variable,
            "ProcessingMethod": self.processing_method, "NaicsCode": self.naics_code, "BorrState": self.borrower_state,
            "BusinessType": self.business_type, "BusinessAge": self.business_age, "CollateralInd": self.collateral_ind,
            "FranchiseCode": "Y" if self.is_franchise else None, "NonProfit": "Y" if self.non_profit else None,
            "JobsSupported": self.jobs_supported, "ThirdPartyDollars": self.third_party_dollars,
        }


class Driver(BaseModel):
    feature: str
    value: str
    effect: Literal["raises risk", "lowers risk"]
    log_odds_contribution: float


class ScoreResponse(BaseModel):
    pd_raw: float = Field(description="Uncalibrated model probability of charge-off within 36 months")
    pd_calibrated: float = Field(description="Calibrated 36-month PD (use this one)")
    decision: Literal["APPROVE", "DECLINE"]
    cutoff: float = Field(description="Approve iff pd_calibrated <= cutoff")
    risk_band: int = Field(ge=1, le=5, description="1 = lowest-risk quintile of the calibration window, 5 = highest")
    expected_loss_rate: float = Field(description="PD x LGD x EAD-factor, per $ approved (LGD is an assumption)")
    expected_loss_usd: float
    top_drivers: list[Driver]


class BatchRequest(BaseModel):
    loans: list[LoanApplication] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
    approved: int
    declined: int
