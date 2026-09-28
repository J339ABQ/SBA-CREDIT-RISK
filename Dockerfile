FROM python:3.11-slim

# libgomp is required by LightGBM's compiled library
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# only what the API needs: source package, API, trained model artefact
COPY src/ src/
COPY api/ api/
COPY models/pd_model.joblib models/pd_model.joblib

ENV MODEL_PATH=/app/models/pd_model.joblib PYTHONUNBUFFERED=1
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
