.PHONY: install data train notebooks test api docker all
install:      ## install full environment
	pip install -r requirements.txt
data:         ## real SBA data if reachable, else synthetic stand-in
	python -m src.data_download || python -m src.synthetic
train:        ## tune, calibrate, choose cut-off, save models/pd_model.joblib + reports/metrics.json
	python -m src.train
notebooks:    ## execute all notebooks top-to-bottom
	python scripts/build_notebooks.py
test:
	python -m pytest -q
api:
	uvicorn api.main:app --reload
docker:
	docker compose up --build
all: data train notebooks test
