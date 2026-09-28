"""FastAPI service: POST a loan application, get a calibrated PD and an approve/decline decision.

    uvicorn api.main:app --reload         # local
    docker compose up --build             # container
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src import predict

from .schemas import BatchRequest, BatchResponse, LoanApplication, ScoreResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = predict.load_model()
    yield


app = FastAPI(
    title="SBA small-business PD API", version="1.0.0", lifespan=lifespan,
    description="Calibrated 36-month probability of default for SBA 7(a)/504 loan applications, with an "
                "approve/decline decision at the recommended cut-off. Model trained on the data source named at /model-info "
                "- if that says 'synthetic' the outputs are a demonstration, not real credit advice.")


def _model(request: Request):
    m = getattr(request.app.state, "model", None)
    if m is None:  # e.g. TestClient used without lifespan
        m = request.app.state.model = predict.load_model()
    return m


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": _model(request) is not None}


@app.get("/model-info")
def model_info(request: Request):
    m = _model(request)
    return {"cutoff": m.cutoff, "lgd_assumption": m.lgd, "ead_factor": m.ead_factor, **m.meta}


@app.post("/predict", response_model=ScoreResponse)
def predict_one(loan: LoanApplication, request: Request):
    try:
        return predict.score(_model(request), pd.DataFrame([loan.to_raw_row()]))[0]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"scoring failed: {type(e).__name__}") from e


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest, request: Request):
    res = predict.score(_model(request), pd.DataFrame([l.to_raw_row() for l in req.loans]))
    approved = sum(r["decision"] == "APPROVE" for r in res)
    return {"results": res, "approved": approved, "declined": len(res) - approved}
