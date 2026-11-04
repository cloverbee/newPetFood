# PetFood RAG

An LLM-powered cat and dog food recommender. Its answers are grounded in open-access veterinary
research and real product data. It uses hybrid retrieval (dense + BM25) in Qdrant with reranking,
plus a router that sends simple requests to a local Qwen3 model and complex ones to DeepSeek.

See [docs/architecture.md](docs/architecture.md) for the design and
[docs/decisions.md](docs/decisions.md) for the trade-offs.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo, docker-compose (Postgres, Qdrant, SeaweedFS S3), schema | ✅ |
| 1 | Ingestion: Europe PMC papers → S3 + Postgres; Open Pet Food Facts + curated products | ✅ |
| 2 | Chunking, embeddings, Qdrant hybrid search (dense + BM25), RRF, cross-encoder rerank | ✅ |
| 3 | LLM clients (Ollama / DeepSeek) + router | ⏳ |
| 4 | Recommendation API + simple UI | ⏳ |
| 5 | OpenTelemetry, Prometheus, Grafana | ⏳ |
| 6 | Retrieval and recommendation evaluation | ⏳ |

## Quickstart (macOS)

Requires Docker Desktop and Python ≥ 3.11.

```bash
cp .env.example .env          # then set CONTACT_EMAIL (Europe PMC / OPFF ask for it)
make setup                    # creates .venv and installs dependencies
make up                       # starts Postgres :5432, Qdrant :6333, S3 (SeaweedFS) :8333
make test                     # offline unit tests

make papers LIMIT=200         # ~5–10 min: open-access cat nutrition papers → S3 + Postgres (scope: D13)
make products LIMIT=300       # ~1 min per 100 products (OPFF rate limit), cats + dogs
make stats                    # what got loaded
```

Faster product loading for larger amounts: download the
[Open Pet Food Facts dump](https://static.openpetfoodfacts.org/data/openpetfoodfacts-products.jsonl.gz),
then run `.venv/bin/petfood-ingest products --from-file openpetfoodfacts-products.jsonl.gz --limit 5000 --require-ingredients`.

To add your own products, fill in `data/products_curated.csv` (see the column notes in
`src/petfood/ingest/curated.py`), then run `make curated`.

If port 5432 is already in use, set `POSTGRES_PORT=5433` in `.env` and update `DATABASE_URL` to match.

## Search (Phase 2)

```bash
make setup                    # re-run once: installs qdrant-client + fastembed
make reparse                  # re-read stored XML from S3: adds tables + figure captions
make retag                    # refresh paper topic tags with the improved rules
make study-years              # when each study's data was collected (needs DEEPSEEK_API_KEY); see D12
make index                    # chunk + embed into Qdrant; first run downloads ~400 MB of models

make search Q="diet for a cat with chronic kidney disease" ARGS="--species cat --type paper"
make search Q="wet food for senior cat" ARGS="--species cat --type product --exclude chicken fish"
make search Q="struvite" ARGS="--mode sparse"    # compare: dense | sparse | hybrid | hybrid_rerank
make search Q="obesity in cats" ARGS="--type paper --pub-from 2018 --study-from 2015"
```

`--pub-from/--pub-to` filter on publication year. `--study-from/--study-to` filter on when the
study ran; papers that don't say are kept unless their received/publication year rules them out.
Try `make study-years ARGS="--limit 10"` first, then check the results with
`psql ... -c "SELECT source_id, study_year_start, study_year_end, study_year_evidence FROM papers WHERE study_year_status='found'"`.

Each result shows its rerank score plus where it ranked in the dense and BM25 lists, so you
can see how fusion and reranking changed the order.

## Tests

```bash
make test                                                       # unit tests, no services needed

# integration tests TRUNCATE tables — always point them at a separate test database
docker compose exec postgres createdb -U petfood petfood_test
PETFOOD_TEST_DB=postgresql://petfood:petfood@localhost:5432/petfood_test .venv/bin/pytest -m integration
```

## Disclaimer

This is an educational project and not veterinary advice. Pets with medical conditions should be seen by a veterinarian.
