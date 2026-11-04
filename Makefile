.PHONY: setup up down reset-db test papers products products-us curated stats reparse retag prune study-years index reindex search

setup:            ## create venv + install
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]"

up:               ## start Postgres, Qdrant, MinIO
	docker compose up -d

down:
	docker compose down

reset-db:         ## drop all data volumes (destructive)
	docker compose down -v

test:
	.venv/bin/pytest -q

papers:           ## ingest open-access vet papers (LIMIT=200)
	.venv/bin/petfood-ingest papers --limit $${LIMIT:-200}

products:         ## ingest cat/dog food from Open Pet Food Facts (LIMIT=500)
	.venv/bin/petfood-ingest products --limit $${LIMIT:-500}

products-us:      ## US-sold products only (LIMIT=300)
	.venv/bin/petfood-ingest products --country united-states --limit $${LIMIT:-300}

curated:          ## load data/products_curated.csv
	.venv/bin/petfood-ingest curated --path data/products_curated.csv

stats:
	.venv/bin/petfood-ingest stats

reparse:          ## re-parse stored paper XML from S3 (tables, figure captions)
	.venv/bin/petfood-ingest reparse

retag:            ## recompute paper topic tags with current rules
	.venv/bin/petfood-ingest retag

prune:            ## list papers outside the scope rule; ARGS=--yes deletes them
	.venv/bin/petfood-ingest prune $(ARGS)

study-years:      ## extract study period per paper via DeepSeek (ARGS="--limit 10" for a trial)
	.venv/bin/petfood-ingest study-years $(ARGS)

index:            ## chunk + embed everything into Qdrant (first run downloads models ~400MB)
	.venv/bin/petfood-index build

reindex:          ## drop the Qdrant collection and rebuild from Postgres
	.venv/bin/petfood-index build --recreate

search:           ## make search Q="kidney disease diet" ARGS="--species cat --type paper"
	.venv/bin/petfood-index search "$(Q)" $(ARGS)
