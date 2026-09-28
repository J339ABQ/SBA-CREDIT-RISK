import copy

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import LoanApplication
from src import config, predict

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="run `python -m src.train` first")

SAFE = {"program": "7A", "gross_approval": 600000, "sba_guaranteed_approval": 450000, "term_months": 300,
        "initial_interest_rate": 7.0, "fixed_or_variable": "V", "processing_method": "PLP", "naics_code": "621111",
        "borrower_state": "CA", "business_type": "CORPORATION", "business_age": "Existing or more than 2 years old",
        "collateral_ind": "Y", "is_franchise": False, "non_profit": False, "jobs_supported": 12, "approval_date": "2019-03-01"}
RISKY = {**SAFE, "gross_approval": 20000, "sba_guaranteed_approval": 17000, "term_months": 60, "naics_code": "722511",
         "business_age": "Startup, Loan Funds will Open Business", "collateral_ind": "N", "processing_method": "Express",
         "initial_interest_rate": 12.5, "jobs_supported": 2}


@pytest.fixture(scope="module")
def model():
    return predict.load_model()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def row(d):
    return pd.DataFrame([LoanApplication(**d).to_raw_row()])


# ---------------------------------------------------------------- inference
def test_scores_are_probabilities_and_calibrated_is_used_for_decision(model):
    r = predict.score(model, row(SAFE))[0]
    assert 0 < r["pd_raw"] < 1 and 0 < r["pd_calibrated"] < 1
    assert r["decision"] == ("APPROVE" if r["pd_calibrated"] <= model.cutoff else "DECLINE")
    assert r["expected_loss_rate"] == pytest.approx(r["pd_calibrated"] * model.lgd * model.ead_factor)
    assert r["expected_loss_usd"] == pytest.approx(r["expected_loss_rate"] * SAFE["gross_approval"])


def test_risky_profile_scores_higher_and_gets_declined_safe_approved(model):
    safe, risky = predict.score(model, pd.concat([row(SAFE), row(RISKY)], ignore_index=True))
    assert risky["pd_calibrated"] > 2 * safe["pd_calibrated"]
    assert safe["decision"] == "APPROVE" and risky["decision"] == "DECLINE"
    assert risky["risk_band"] > safe["risk_band"] and risky["risk_band"] == 5


def test_scoring_is_deterministic_and_batch_equals_single(model):
    a = predict.score(model, row(SAFE))[0]
    b = predict.score(model, pd.concat([row(RISKY), row(SAFE)], ignore_index=True))[1]
    assert a["pd_calibrated"] == pytest.approx(b["pd_calibrated"])


def test_missing_and_unseen_inputs_still_score(model):
    d = {k: v for k, v in RISKY.items() if k not in ("initial_interest_rate", "naics_code", "borrower_state", "collateral_ind")}
    r = predict.score(model, row({**d, "borrower_state": None}))[0]
    assert 0 < r["pd_calibrated"] < 1
    assert 0 < predict.score(model, row({**SAFE, "naics_code": "99", "borrower_state": "ZZ"}))[0]["pd_calibrated"] < 1


def test_drivers_explain_the_risky_loan(model):
    r = predict.score(model, row(RISKY))[0]
    assert len(r["top_drivers"]) == 3
    assert any(d["effect"] == "raises risk" for d in r["top_drivers"])
    assert {"business_age", "naics_sector", "term_months", "log_gross_approval", "collateral_ind"} & {d["feature"] for d in r["top_drivers"]}


def test_model_meta_states_training_source_and_cutoff(model):
    assert model.meta["trained_on"] in {"synthetic", "real"} and 0 < model.cutoff < 0.5
    assert model.meta["train_years"][1] < model.meta["calibration_years"][0]  # trained strictly before calibration window


def test_missing_model_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        predict.load_model(tmp_path / "nope.joblib")


# ---------------------------------------------------------------------- API
def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model-info").json()
    assert {"cutoff", "lgd_assumption", "trained_on"} <= set(info)


def test_predict_ok_schema(client):
    r = client.post("/predict", json=RISKY)
    assert r.status_code == 200
    b = r.json()
    assert set(b) == {"pd_raw", "pd_calibrated", "decision", "cutoff", "risk_band", "expected_loss_rate", "expected_loss_usd", "top_drivers"}
    assert b["decision"] == "DECLINE" and 1 <= b["risk_band"] <= 5


def test_predict_minimal_payload_uses_defaults(client):
    r = client.post("/predict", json={"gross_approval": 50000, "sba_guaranteed_approval": 25000, "term_months": 84})
    assert r.status_code == 200 and 0 < r.json()["pd_calibrated"] < 1


@pytest.mark.parametrize("patch", [
    {"sba_guaranteed_approval": 10_000_000},                 # guarantee > loan
    {"gross_approval": -5}, {"gross_approval": 0},
    {"term_months": 0}, {"term_months": 10_000},
    {"borrower_state": "Texas"}, {"naics_code": "abc"},
    {"business_age": "brand new"}, {"program": "8A"}, {"initial_interest_rate": 250},
])
def test_predict_rejects_invalid_input_with_422(client, patch):
    assert client.post("/predict", json={**SAFE, **patch}).status_code == 422


def test_predict_missing_required_field_is_422(client):
    body = copy.deepcopy(SAFE); del body["gross_approval"]
    assert client.post("/predict", json=body).status_code == 422


def test_state_is_case_insensitive(client):
    a = client.post("/predict", json={**SAFE, "borrower_state": "ca"}).json()
    b = client.post("/predict", json=SAFE).json()
    assert a["pd_calibrated"] == pytest.approx(b["pd_calibrated"])


def test_batch_counts_and_order(client):
    r = client.post("/predict/batch", json={"loans": [SAFE, RISKY, SAFE]}).json()
    assert [x["decision"] for x in r["results"]] == ["APPROVE", "DECLINE", "APPROVE"]
    assert (r["approved"], r["declined"]) == (2, 1)


def test_batch_limits(client):
    assert client.post("/predict/batch", json={"loans": []}).status_code == 422
    assert client.post("/predict/batch", json={"loans": [SAFE] * 1001}).status_code == 422
