# Design decisions

Short records of each design decision, written so I can explain the trade-offs in an interview.
Each one also says where it lives in the code, how to run it, and what's still open, so work
can pick up again after a break.

## Where we are (2026-11-05)

**Phases:** 0–2 done (infra, ingestion, hybrid search). Phase 4's core is in: `petfood-answer`
retrieves products + papers, has DeepSeek write a cited answer, and checks it in code (D14).
Phase 3 has started: a local Qwen3-4B client via Ollama exists and `petfood-answer --llm local`
uses it (D15). There is still **no router**: the model is chosen by hand with `--llm`.

**Data loaded right now** (services up via `make up`; check with `make stats`):

| What | Count | Notes |
|---|---|---|
| Papers | 38 | All cats, all full text. Left after `prune --yes` applied the D13 scope to the earlier 240 |
| Table sections | 150 | From `make reparse` (D11) |
| Study years | 8 found · 22 not_stated · 8 no_dates · 0 pending | `make study-years` has run on every paper (D12) |
| Products | 165 (75 cat, 90 dog) | Open Pet Food Facts; curated CSV is a stub (D2). **Mostly empty:** of 75 cat foods, 34 have ingredients, 11 protein, 2 moisture (D16) |
| Chunks | 1,508 paper + 165 product = 1,673 | Qdrant `petfood_chunks` has 1,673 points, so it's in sync |
| Tests | 122 pass, 4 skipped | Skipped = Postgres integration tests (need `PETFOOD_TEST_DB`) |

**Pick up here:**
0. **Product data is the weakest link (D16):** most cat products have no ingredients or nutrients,
   so answers can't compare them. Options: OPFF bulk dump with `--require-ingredients`, or fill
   `data/products_curated.csv` for ~20 common US cat foods.
1. **Allergen tagging gaps (D4):** "poultry" and "meat by-products" don't count as chicken. The
   LLM noticed this itself in a real answer; the tags should catch it first.
2. **Test questions exist (D19):** 20 in `eval/questions.toml`, run with `make eval`. Next:
   fix the two problems the first run found (allergy vet note; products pushed on knowledge
   questions), then use the eval to decide `--k-rerank 15`, skipping product rerank, and local vs DeepSeek.
3. **Phase 3:** the rules router (D7), choosing `--llm local` vs `deepseek` per question, with a
   DeepSeek retry when the local answer fails `check_answer()`. The 20 questions decide whether
   qwen3:4b is good enough (see D15's first comparison: it isn't yet).
4. Full corpus load: on hold by choice (2026-11-04); 38 papers is enough to build the pipeline.
   When wanted: `make papers LIMIT=700`, then reparse, retag, study-years, index.

**Repo:** first commit `75119cd` pushed to `github.com/cloverbee/newPetFood` (`main`) on 2026-11-03.
Backups (`.phase*.tar.gz`, `.tables.tar.gz`), `Claude outputs/` and the sample `PMC*.xml/.md`
files in the root are git-ignored on purpose.

---

## D1. Paper source: Europe PMC rather than scraping journals
- Europe PMC's REST API returns metadata, the abstract, the **license**, and JATS full text for open-access articles. One API covers PubMed Central plus European content.
- Only **Creative Commons** licensed papers are stored. We keep and quote full text, so the license has to permit that. Each paper's license is stored so it can be shown on citations.
- Papers are filtered after the search with species keyword rules. The keyword query pulls in some human-only papers, and these get dropped.
- **Code:** `ingest/europepmc.py`. `EuropePMCClient` (search with cursor paging, `full_text_xml`, tenacity retries on 429/5xx); `license_ok()` accepts `cc by*`, `cc0`, `cc-by*`; `record_to_paper()` maps a search record to `Paper`. `CONTACT_EMAIL` goes in the User-Agent, which Europe PMC asks for.
- **Run:** `make papers LIMIT=200`. Flags on `petfood-ingest papers`: `--query`, `--skip-existing`, `--max-new`. Every run is logged to `ingestion_runs`.
- **History:** 1st run fetched 100 and kept 94; 2nd fetched 250 and kept 146 (2026-10-16, before D13). Skips are mostly non-CC licenses and out-of-scope titles.

## D2. Product source: Open Pet Food Facts + curated CSV, not Amazon or Petco scraping
- Amazon's and Petco's terms of service prohibit scraping. Petco has no public API, and Amazon's PA-API needs an Associates account and returns little nutrition data.
- Open Pet Food Facts is open data (ODbL) with ingredients and nutrition values.
- The weak spot is **coverage of US brands**. The curated CSV fills gaps with data from manufacturer pages.
- The `products.source` column keeps the design source-agnostic, so a retailer API can be added later without schema changes.
- **Code:** `ingest/openpetfoodfacts.py` (`OPFFClient` searches categories `en:cat-food` / `en:dog-food`; `iter_dump()` reads the bulk `.jsonl.gz`; `record_to_product()` normalizes and tags). `ingest/curated.py` loads the CSV and documents its columns.
- **Run:** `make products LIMIT=300` (rate-limited, ~1 min per 100), `make products-us`, `make curated`. For thousands, use the dump: `petfood-ingest products --from-file … --require-ingredients`.
- **Status:** 165 products loaded (192 fetched, 27 skipped). `data/products_curated.csv` is only a header row so far.
- **Open:** US brand coverage is unmeasured. Guaranteed-analysis values in OPFF are often missing or per-100g vs as-fed; `_num()` caps values at 100 to reject junk.

## D3. Postgres is the source of truth and Qdrant is a derived index
- Relational data (products, papers, evaluation labels, query logs) needs joins, constraints and audit history.
- Vectors can be regenerated whenever the embedding model or chunking changes. Keeping `chunks` in Postgres makes re-indexing a batch job, not a data migration.
- **Code:** schema in `db/init/001_schema.sql`: tables `papers`, `paper_sections`, `products`, `chunks`, `ingestion_runs`. Docker applies it only to an **empty** volume; later columns (D12) use `ALTER TABLE … ADD COLUMN IF NOT EXISTS`, so re-apply with `petfood-ingest init-db` or `psql "$DATABASE_URL" -f db/init/001_schema.sql`. SQL access is in `db.py` (upserts keyed on `(source, source_id)`).
- **Rule:** after anything that changes papers or products (ingest, reparse, retag, prune, study-years), run `make index`. `make reindex` drops and rebuilds the collection.
- **Infra (`docker-compose.yml`):** Postgres 16 on :5432, Qdrant v1.12.4 on :6333 (client pinned `<1.13` to match), SeaweedFS S3 on :8333. `make reset-db` deletes all volumes.

## D4. Allergen safety comes from deterministic filters, not the LLM
- Rule-based tagging produces `allergen_tags`, and the filtering happens in SQL or as a Qdrant payload filter **before** retrieval.
- Tagging is **conservative**: "chicken fat" and "fish oil" are tagged as chicken and fish. Wrongly calling a food "safe" costs more than being over-cautious.
- This is also easy to test. Constraint-violation rate is one of the Phase 6 metrics.
- **Code:** `ingest/tagging.py`. `PROTEIN_SOURCES` (chicken, turkey, duck, poultry, beef, lamb, pork, fish, shellfish, venison, rabbit, egg, insect) and `OTHER_ALLERGENS` (grain, gluten, dairy, soy) are word lists. `_FALSE_POSITIVES` blanks out "grain-free", "no added wheat" and the like before matching. `tag_ingredients()` also sets `protein_sources` and `grain_free`; `detect_life_stage()` and `detect_food_form()` fill those columns.
- **Filter:** `SearchFilters.exclude_allergens` in `retrieval/store.py` becomes a Qdrant `must_not` on `allergen_tags`. CLI: `--exclude chicken fish`.
- **Products with no ingredient list are dropped whenever allergens are excluded** (added 2026-11-05). Tags come from ingredients, so such a product has no tags and used to look allergen-free. Rule in `SearchFilters.to_qdrant()`, re-checked in `recommend/answer.py` `check_answer()`.
- **Open:** "poultry" is a separate tag from "chicken", so excluding chicken does not exclude "poultry meal". "Meat by-products", "animal digest" and unnamed "liver" get no tag at all. Fix before showing recommendations to real users: chicken should imply poultry, and unnamed animal sources should count as every meat.

## D5. Raw files go in object storage
- The raw JATS XML is kept in S3 (SeaweedFS locally), so we can re-parse later (for example with better section handling) without downloading everything from Europe PMC again.
- SeaweedFS is S3-compatible locally (MinIO stopped publishing Docker images). Moving to AWS S3 is a configuration change: clear `S3_ENDPOINT_URL` and set real keys.
- **Code:** `storage.py` (`ObjectStore` on boto3). Keys look like `papers/europepmc/PMC1234567.xml`; bucket `petfood-papers`. `papers.s3_key` points to the file.
- **Used by:** `make reparse` (D11), `make study-years` (reads `<history>` dates, D12), `make prune` (deletes S3 first, then Postgres, so a failure leaves a list of what's still to clean).

## D6. Hybrid retrieval with RRF, then reranking (Phase 2)
- Dense embeddings miss exact clinical terms ("struvite", "taurine"). BM25 misses paraphrases. Combining both covers each one's weakness.
- RRF needs no score calibration between the two retrievers.
- A cross-encoder reranker on about 30 candidates gives the largest precision gain for its cost, and only the top 5 go to the LLM. Fewer tokens means lower cost and fewer distractions for the model.
- **Code:** `retrieval/pipeline.py`. `HybridRetriever(k_dense=30, k_sparse=30, k_rerank=30, rrf_k=60, low_value_penalty=0.5)`; `retrieve(query, top_k=5, filters, mode, max_per_doc=2)`. Returns each candidate's dense, sparse, RRF and rerank ranks/scores plus per-stage timings.
- **Run:** `make search Q="…" ARGS="--species cat --type paper"`.
- **Open:** none of these numbers is tuned yet. That waits for the Phase 6 eval set.

## D7. Router: rules first, then a small classifier (Phase 3)
- Start with transparent rules: number of conditions, medical keywords, multi-pet questions.
- Next, a logistic regression on the query embeddings we already compute. It's nearly free at inference time, and DeepSeek labels about 500 queries as training data.
- Move to a fine-tuned DistilBERT/ModernBERT only if the metrics show the simpler router falls short.
- Metrics: routing accuracy, cost per request, and the quality gap between the local model and DeepSeek.
- **Status:** not started. Exists so far: `llm/deepseek.py` (`DeepSeekClient.chat_json`, temperature 0, JSON mode, retries). Config is ready: `OLLAMA_BASE_URL`, `LOCAL_MODEL=qwen3:8b`, `DEEPSEEK_MODEL=deepseek-chat`.
- **Next:** an Ollama client with the same `chat_json(system, user)` shape, so the router and D12 can swap models; then the rules router, then a labeled query set.

## D8. Chunking: section-aware windows with a contextual header (Phase 2)
- Chunks never cross section boundaries. Each one is about 300 words, split at sentence boundaries, with about 50 words of overlap.
- Every chunk starts with `"<paper title> — <section heading>"`. A passage that only says "the cats lost weight" becomes findable by what the study was about. This helps BM25 and dense search alike.
- Methods sections are indexed but flagged `low_value`, and the reranker score is down-weighted. They rarely answer an owner's question but can still be retrieved when needed.
- Results are capped at 2 chunks per paper, so the LLM sees evidence from several studies instead of one paper repeated.
- Chunk ids are `uuid5(doc_type:doc_id:ord)`. Re-indexing overwrites points instead of duplicating them.
- **Code:** `retrieval/chunking.py`: `chunk_paper(max_words=300, overlap_words=50, min_words=25)` drops tiny trailing windows but keeps a short section that stands alone; `chunk_product()` renders one chunk per product. `retrieval/indexer.py` saves chunks to Postgres, then upserts to Qdrant.
- **Payload per paper chunk:** `doc_type, doc_id, source_id, title, species, topics, pub_year, study_year_start/end/latest, journal, license, doi, section_type, heading, low_value, body`. Indexed fields are listed in `store.py` `_PAYLOAD_INDEXES`.
- **Penalty detail:** reranker scores can be negative logits, so Methods scores are multiplied by 0.5 when positive and divided by 0.5 when negative. Both push them down.

## D9. Models: bge-small + Qdrant BM25 + bge-reranker-base, all through fastembed
- **fastembed** runs ONNX models on the CPU, with no PyTorch and a small install. It's fast on Apple Silicon.
- **Dense: `BAAI/bge-small-en-v1.5`** (384 dimensions). Good quality for its size, and the corpus embeds in about a minute.
- **Sparse: `Qdrant/bm25` + `Modifier.IDF`.** The client sends term frequencies and Qdrant applies IDF over the live collection, so it's real BM25 with no separate Elasticsearch to run.
- **Rerank: `BAAI/bge-reranker-base`**, a cross-encoder. It reads the query and passage together, so it's far more precise than comparing two separate embeddings. Too slow to run over the whole corpus, so it only rescores the top 30 after fusion.
- All three model names are set in `.env`, so we can swap them and measure the difference in Phase 6.
- **Code:** `retrieval/embeddings.py`. `FastEmbedDense`, `FastEmbedBM25`, `FastEmbedReranker` behind small Protocols. `HashDense`, `HashSparse` and `OverlapReranker` are offline fakes so unit tests need no model download (`make_store(fake=True)` in `retrieval/cli.py`).
- **First run** of `make index` downloads ~400 MB of models.
- **Model cache (2026-11-07):** fastembed downloads to `FASTEMBED_CACHE_DIR` (default `~/.cache/fastembed`, ~1.3 GB), passed as `cache_dir` by `make_store()`/`make_retriever()`. fastembed's default was a macOS temp folder the system can clear, which would force a re-download. Existing files were moved there.

## D10. Fusion is done in our code, not with Qdrant's built-in RRF
- Qdrant can fuse server-side. Doing it in our code keeps each candidate's dense rank, BM25 rank, RRF rank and rerank score. That gives free ablations for evaluation (`--mode dense|sparse|hybrid|hybrid_rerank`) and makes it possible to debug why a result came first.
- The cost is two network round-trips instead of one, which is negligible locally.
- **Code:** `rrf()` and `dedupe_by_doc()` in `retrieval/pipeline.py`; `ChunkStore.search_dense/search_sparse` in `retrieval/store.py`. Vector names: `dense`, `bm25`.

## D11. Tables are their own sections and chunks
- v1 dropped tables to keep prose clean. Reviewing a real paper showed that loses the most useful facts in nutrition papers: diet composition, nutrient percentages and outcomes.
- Each `<table-wrap>` now becomes a `table` section: label + caption, markdown rows, footnotes. Figure captions become `figure` sections. Prose still excludes them, so nothing is duplicated.
- Tables are split by **rows**, and every piece repeats the caption and header row, so `| Crude protein | 45.2 |` never loses its context.
- This needed no re-download: `petfood-ingest reparse` re-parsed the raw XML already kept in S3. That's the payoff of D5.
- **Code:** `ingest/jats.py` (`table_to_text()`, `_add_floats()`; `_SKIP_TAGS` keeps tables, figures, formulas and supplements out of prose). `window_table()` in `retrieval/chunking.py`.
- **Status:** 150 table sections across the 38 papers. The reviewed papers were PMC3395784 and PMC6533943 (copies sit in the repo root, git-ignored).
- **Open:** merged cells (`rowspan`/`colspan`) are flattened, not expanded, so some wide tables read oddly.

## D12. Two years per paper: publication year and study period
- **Publication year** comes straight from Europe PMC's `pubYear` and is always known. Filter with `--pub-from/--pub-to`.
- **Study period** (when the data was collected) has no metadata field and most papers never state it, so `petfood-ingest study-years` extracts it in three steps:
  1. A regex keeps sentences that still mention a year once citations ("Smith et al., 2014", "(AAFCO, 2016)"), citation fragments and measurements ("2000 ng/g") are stripped. Papers with none never reach the LLM.
  2. DeepSeek reads the numbered sentences and answers `{start, end, sentence_ids}`. The prompt tells it to ignore cited studies, literature searches and guideline years, and to answer null for reviews.
  3. Code checks the answer: the years must appear in the cited sentences, start ≤ end, and the study must end no later than the **received** year (from JATS `<history>`), else the publication year. Anything that fails is stored as `rejected` with the reason, never as a filterable year.
- **Unknown study periods still filter (option B).** A study can't end after the manuscript was submitted, so every paper gets `study_year_latest` = stated end, else received year, else publication year. `--study-from Y` keeps papers whose `study_year_latest ≥ Y`. `--study-to Y` keeps papers whose stated start ≤ Y, *and* papers with no stated start, since we can't rule them out. So "studies from before 2010" is a weak filter for papers that don't state their dates.
- Year filters apply to papers only; products pass through unchanged.
- The evidence sentence is stored next to the years, so any value can be checked by eye.
- **Code:** `ingest/study_years.py` (`candidate_sentences`, `SYSTEM_PROMPT`, `validate`, `extract_study_years`). Limits: scan sections in order methods → abstract → other → results; at most 25 candidate sentences of ≤500 chars; years 1950–2099; reject a start more than 30 years before the upper bound. Reviews (by JATS article type or a title word like "review", "meta-analysis", "guidelines") skip the LLM as `not_stated`. Filters: `_or_unknown()` in `retrieval/store.py`. Columns: `received_year`, `study_year_start/end/status/evidence/note` on `papers`.
- **Status values:** `found` · `not_stated` (model said null, or review) · `no_dates` (no candidate sentences, LLM never called) · `rejected` (failed a check; years only in `study_year_note`).
- **Run:** needs `DEEPSEEK_API_KEY` in `.env`. `make study-years ARGS="--limit 10"` for a trial; it skips done papers unless `--force`. One failed response is logged and counted as `error`, and a re-run picks it up. Then `make index`.
- **Result on the 38 papers:** 8 found, 22 not_stated, 8 no_dates, 0 rejected.
- **Check by eye:** `SELECT source_id, study_year_start, study_year_end, study_year_evidence FROM papers WHERE study_year_status='found'`.

## D13. Corpus scope: cats *and* diet in the title, sorted by relevance
- v1 matched species and diet words anywhere in title **or abstract**, sorted by citations. Of 240 ingested papers, 198 were off-topic: plant physiology, fungi, human cohorts, dairy cows, dog genomes and behaviour. They mention a dog or "food" once in the abstract, and highly cited off-topic papers rose to the top.
- **Cats only.** The papers serve cat food recommendations, and in a cat-and-dog query dog papers outnumber cat papers about 2:1. Titles naming both ("dog and cat foods") still count.
- Both conditions must hit the **title**: a cat word (cat, feline, kitten) **and** a diet word (diet, nutrition, food, feeding, kibble, supplement, obesity, body condition, digestibility, …). Generic titles ("…for pet foods", "companion animals") count only if the abstract mentions cats.
- What a trial fetch of 30 papers showed:
  - "gut", "microbiota", "diarrhea" and "body weight" pulled in digestive-disease, drug and heart-scan papers with no diet in them. They're no longer diet words; diet-and-microbiome papers still match on "diet" or "fed".
  - "pet" matched PET imaging and plastics, so bare "pet" was dropped.
  - Europe PMC ignores case, so "CAT" (catalase) matches "cat"; `in_scope()` rejects the all-caps form.
  - Sorting by citations surfaced classic gut-microbiome papers. Sorting by relevance surfaces food studies: kibble vs wet food, digestibility, minerals and contaminants in cat foods.
- About 680 papers match. The term lists in `ingest/europepmc.py` build both the API query and `in_scope()`, a local check. `record_to_paper` applies it too, so a custom `--query` can't bypass the scope. `petfood-ingest prune` applies it to papers already stored, in Postgres and S3.
- Trade-off: a relevant paper whose title has no diet word, such as one on a disease managed by diet, is missed. Recall can be widened later by adding terms; precision problems poison every answer, so precision comes first.
- Correction, erratum and retraction notices are skipped. Title markup (`&lt;i&gt;`) is cleaned.
- **Code:** `CAT_TERMS`, `GENERIC_TERMS`, `DIET_TERMS`, `DEFAULT_QUERY` and `in_scope()` at the top of `ingest/europepmc.py`. The query also requires `OPEN_ACCESS:y AND IN_EPMC:y AND PUB_YEAR:[2005 TO 2026]`.
- **Status:** prune has run, leaving 38 papers. The full ~680 have **not** been ingested yet (see "Pick up here").
- **Open: recall for disease papers.** Candidate fix: a second title rule of cat word + disease word (renal, kidney, CKD, urinary, struvite, oxalate, urolith, allergy, adverse food reaction, diabetes, hepatic lipidosis), but only when the abstract mentions diet. Measure precision on a trial fetch of 30 first, as was done for v2.
- **Open:** the 2005 lower bound is a guess; older feeding trials (e.g. taurine work from the 1980s–90s) are excluded.
- **Note:** the paper tag `species` can still include `dog` (20 of the 38 papers) because titles like "dog and cat foods" are in scope. The `--species cat` filter still works.

## D14. Answer generation: label the context, let code check the model (Phase 4 core)
- Retrieval stops at a list of chunks. `answer()` adds the generation step: one product search (filtered by the pet's species, life stage, food form and allergens) plus one paper search (species, optional year filters), then one DeepSeek call, then code checks.
- **Labels instead of free text.** The prompt lists products as `[P1]…` and paper excerpts as `[E1]…` (with title, section, publication year and study period). The model must recommend by P-label and cite by E-label, so every claim can be checked exactly and mapped back to a product id or PMC id.
- **Code decides what survives, not the model** (`check_answer()`): unknown product labels and invented citations are dropped; recommended products are re-checked against the allergen, ingredient-list and species rules even though the search already filtered them; at most 3 recommendations; leftover P-labels in prose are replaced with product names. Everything removed is listed in `warnings`.
- **The vet note is a rule:** it's added when the profile lists a condition or the question hits a medical topic (obesity, renal, urinary, diabetes, food allergy, GI, cardiac, hepatic). The model can't skip it.
- **No context, no LLM call:** if both searches return nothing, the answer says so instead of letting the model guess.
- **Code:** `src/petfood/recommend/answer.py` (`PetProfile`, `PaperFilters`, `gather_context`, `build_prompt`, `check_answer`, `answer`); terminal wrapper `src/petfood/recommend/cli.py`; tests `tests/test_answer.py` (in-memory Qdrant, fake encoders, fake LLM). The LLM is any `(system, user) -> dict` callable, so Phase 3 can swap in Qwen3 without touching this code.
- **Run:** `petfood-answer "question" --species cat --life-stage senior --food-form wet --exclude chicken --condition overweight`, or `make answer Q="…" ARGS="…"`. `--show-prompt` prints the prompt without calling the LLM; `--json` prints everything (sources, warnings, timings); `--no-rerank` skips the cross-encoder.
- **Cost** (see D18 for the current numbers): the reranker dominates: ~11–13 s per answer on CPU (two searches) vs ~40 ms with `--no-rerank`. DeepSeek takes ~2–2.5 s. Loading the reranker once in a long-running API process won't fix this, because the time is in scoring, not loading: fewer rerank candidates or a smaller reranker are the levers (Phase 6 can measure the quality cost).
- **First real runs (2026-11-05):** the hydration question got 3 wet foods with correct citations. The overweight + chicken-allergy question got no recommendation, because the 3 remaining chicken-free wet foods didn't fit, and the model said so and pointed to a vet. That's the intended behaviour with a thin corpus.
- **Open:** the product search uses the whole question as its query, so "overweight" doesn't steer it to low-calorie foods; filtering or sorting on `kcal_per_100g`/`fat_pct` would. With no life stage given, kitten foods can appear for an adult cat.

## D15. Local model: Qwen3-4B on Ollama, thinking off, schema-enforced JSON (Phase 3)
- **Why Ollama:** one install, a local HTTP server that keeps the model loaded, Apple-GPU (Metal) support, and an API shaped like DeepSeek's. vLLM is the production choice for many users on NVIDIA GPUs but has no Apple-GPU support; switching later is a URL change. MLX is the fallback if Ollama is too slow on the Mac.
- **Why 4B, not 8B:** the Mac has 16 GB and Docker reserves 7.6 GB. qwen3:4b uses ~2.5 GB. Below ~3–4B, models break strict JSON and citation rules.
- **Install (2026-11-05):** Homebrew failed because `/etc/hosts` pins `raw.githubusercontent.com` to an old address (`199.232.28.133`) with the wrong TLS certificate. That line was left alone on purpose. Ollama 0.40.15 was installed from the official release tarball into `~/.local/ollama`, linked as `~/.local/bin/ollama`. Models live in `~/.ollama/models`.
- **Start it** (not auto-started; needed after every reboot): `nohup ollama serve > ~/.ollama/serve.log 2>&1 &`. Check: `curl localhost:11434/api/version`.
- **Measured on the M3 (Metal GPU):** ~31 tokens/s generation; first call loads the model in ~3 s.
  - Thinking on: correct answers but ~700 tokens of reasoning first (~26 s for one sentence).
  - Thinking off + plain JSON mode: valid JSON with invented keys (`task`, `reasoning`).
  - **Thinking off + a JSON schema in `format`: correct shape in 1.7 s.** That's the setting used.
- **Where the model files live (2026-11-07):** `~/.ollama/models` is a symlink to `/Volumes/PSSD/ollama-models` on the external SSD (2.3 GB moved; a 27B would be ~17 GB). The symlink works however Ollama is started, with no env var. Plug PSSD in before `ollama serve`; without it, Ollama reports the model missing and nothing is damaged (the files are only read, then held in memory). The old copy is at `~/.ollama/models.old-internal`, to delete once PSSD is trusted. **Speed:** PSSD was behind a USB 2.0 hub (480 Mb/s): 41 MB/s read, so loading qwen3:4b took 74 s (the 2.5 GB file alone takes 61 s to read; ~1.5 s from the internal drive). Plug it straight into a Mac port. Databases (Docker volumes) stay on the internal drive: they write constantly and an unplug mid-write can corrupt them.
- **Context window:** `LOCAL_NUM_CTX=8192` is sent as `options.num_ctx`. Ollama's default 4096 was nearly full: the answer prompt is ~3,789 tokens, and an overflow silently drops the start of the prompt (the rules). Costs ~0.6 GB more memory (3.2 → 3.8 GB loaded).
- **Code:** `src/petfood/llm/ollama.py` `OllamaClient.chat_json(system, user, schema=None)`: `think` from `LOCAL_THINK` (default false), temperature 0, retries on 5xx, clear errors when the server is down (`ollama serve`) or the model is missing (`ollama pull`). `ANSWER_SCHEMA` in `recommend/answer.py`. Config: `LOCAL_MODEL=qwen3:4b`, `LOCAL_THINK=false`. Tests: `tests/test_ollama.py` (HTTP mocked).
- **Run:** `petfood-answer "question" --llm local` (default `--llm deepseek`).
- **First comparison** ("Is wet food better than dry for hydration?", same retrieval, `--no-rerank`):

  | | qwen3:4b (local) | deepseek-chat (API) |
  |---|---|---|
  | Generation time | ~19 s | ~2.6 s |
  | Citations in summary | none; "research shows…" uncited | inline [E1][E3] |
  | Faithfulness | said the research doesn't compare wet vs dry (E1 does); **gave Fussy Cat 84.5% moisture, copied from another product** (DB has no value) | claims match the excerpts and product cards |
  | Picks | same 3 wet foods | same 3 wet foods |

  The long prompt (~8 product cards + 5 excerpts) is what's slow locally: the laptop has to read it all before writing.
- **Open:** local-model latency: try fewer products/excerpts for local runs, or MLX. (The invented-number problem is handled by D16.)

## D16. Show missing values; check every number against its source
- **Why:** qwen3:4b gave Fussy Cat "84.5% moisture", copied from the Mars card next to it. LLMs write the most likely next word; the only moisture number in the prompt was the likeliest fit, and a 4B model is weak at keeping each fact attached to its product ("attribute binding"). The prompt made it worse: a missing value was silently left out, so nothing said "unknown".
- **Missing values are stated in the prompt:** each product gets `Not listed: protein, fat, fiber, moisture, calories, ingredients` for whatever it lacks, plus a rule: only state a product number that appears in that product's own entry. This line is **not** in the search index: "moisture: not listed" on every card would make a keyword search for "moisture" match every product. Code: `product_entry()`, `_missing_facts()` in `recommend/answer.py`; `fiber_pct` and `moisture_pct` added to the product payload (`chunking.py` `chunk_product`), re-indexed 2026-11-05.
- **Number check in code** (`_drop_unsupported_numbers()`, called from `check_answer()`): every number in a recommendation's reason must appear in that product's entry, an excerpt the reason cites, or the question. Every number in the summary must appear somewhere in the prompt. A sentence that fails is removed with a warning; a recommendation with no sentence left is dropped. Numbers glued to letters ("B12") are ignored; [E#] labels are removed before matching. The prompt text and the check use the same `product_entry()` / `evidence_entry()`, so they compare against exactly what the model saw.
- **Limits:** the summary is checked against the whole prompt, so a summary that pins a real number on the wrong product still passes; only reasons are checked per product. Numbers written as words ("two") aren't checked. Rounding ("about 85%" for 84.5) is rejected, which is strict but safe.
- **Result on the same hydration question:** both models now attach 84.5% to the right product and say "moisture is not listed" for the others. No sentence needed removing. qwen3:4b took 19 s on the first run and 65 s on this one with a nearly identical prompt; the cause wasn't investigated (likely memory pressure or a model reload), so local latency is unreliable on this laptop.
- **What it exposed:** product data is mostly empty. Of 75 cat products: 34 have an ingredient list, 11 protein, 13 calories, **2 moisture**. 7 of the 8 products in the hydration prompt had no nutrition values or ingredients. Answers can't compare products on facts that aren't there, so better product data (D2) matters more than model tuning right now.

## D17. Read the question for filters; an unstated life stage means adult food
- **The failure (2026-11-06):** "What wet food is good for my adult cat?" got *no recommendation*. Three causes, all on the product side (the papers were fine):
  1. "wet" and "adult" in the question weren't filters; only `--food-form`/`--life-stage` flags were. The search returned kitten foods, a dry food, a raw food and a product with no data.
  2. Most labels don't state a life stage. Cat wet foods: 8 kitten, **0 adult**, 20 not stated. The model read "not stated" as "not suitable".
  3. The prompt let the model recommend none whenever details were missing.
- **Life-stage rule (owner's call, backed by AAFCO):** AAFCO defines only two nutrient profiles, *growth* and *adult maintenance*; "senior" is marketing. So a label without a life stage is an adult food.

  | Requested | Matches labels |
  |---|---|
  | kitten | kitten, growth, all (unstated **excluded**: kittens need growth food) |
  | adult | adult, all, **unstated** |
  | senior | senior, adult, all, **unstated** |

  Code: `LIFE_STAGE_MATCHES` and `_life_stage_condition()` in `retrieval/store.py` (Qdrant `IsEmpty` matches the unstated ones). The database keeps `life_stage` NULL (the label really doesn't say); only matching and the prompt treat it as adult. The prompt line reads "Life stage: adult (the label doesn't say, so it's an adult maintenance food)". This also changes `make search --life-stage senior`, which used to exclude adult foods.
- **Filters from the question:** `profile_from_question()` in `recommend/answer.py` sets food form and life stage when the caller didn't. It only sets one when the question names exactly one ("is wet better than dry?" sets nothing). Ages count: "12-year-old" → senior (from 11 years, `SENIOR_FROM_YEARS`), "4 year old" → adult. Age phrases are removed before matching stage words, so "4 year old cat" isn't also read as "old cat". Explicit flags always win. The answer reports the profile it used (`Answer.profile`; the CLI prints a `filters:` line).
- **Prompt rule:** every listed product already fits the pet's needs; missing details aren't a reason to reject one. Recommend the best 1–3, say what isn't listed, prefer products with ingredients or nutrients listed, and recommend none only if no listed product can provide what's asked.
- **Result:** the same question now recommends three wet foods, with reasons drawn from what each label lists ("Moisture is not listed").
- **Open:**
  - Some product values look wrong, e.g. a wet food at 320 kcal/100g (typical of dry food), probably from Open Pet Food Facts on the wrong basis. The number check can't catch it because the value is in the data. A sanity check at ingest (wet food kcal or moisture out of range) would.
  - Search took 48 s on that run against ~13 s before. Mostly explained by D18 (padding inside rerank batches); the remaining run-to-run variance is probably memory pressure on 16 GB.
  - "Chicken Breast in Broth" may be a topper rather than a complete diet; products don't record "complete" vs "complementary" yet.

## D18. Rerank one pair at a time on CPU
- **Problem (2026-11-06):** an answer took 20–50 s, almost all of it in the cross-encoder. fastembed reranks in batches of 64 by default. A batch is one rectangular tensor, so every (question, text) pair is padded to the longest pair in it, and the model computes through the padding (attention cost grows with length²). One long product card made all 20 product pairs as long as itself.
- **Measured** (M3, 16 GB, CPU via ONNX; scores identical to 4 decimals at every batch size):

  | Batch size | 20 product cards | 30 paper chunks |
  |---|---|---|
  | 64 (fastembed default) | 17.7 s | 26.6 s |
  | 8 | – | 24.1 s |
  | 4 | 6.9 s | – |
  | **1** | **4.3 s** | **16.5 s** |

  Same-process A/B of `gather_context()` for one question, alternating: batch 64 → 29.7 s, 29.9 s; batch 1 → 18.8 s, 19.1 s.
- **Decision:** `FastEmbedReranker(model_name, batch_size=1)` in `retrieval/embeddings.py`. On a CPU, batching gains little (the cores are already busy with one pair) and padding costs a lot. **On a GPU, raise it**: there, batching is where the speed comes from.
- **Tried and rejected:** the CoreML execution provider (Apple GPU/Neural Engine) was slower (43–67 s). Only 405 of 641 graph nodes run on CoreML, so the model is split into 75 partitions with copies between them.
- **Not a cloud GPU yet:** one developer running a few dozen questions a day doesn't justify paying for and running a GPU server. Revisit for real users (Phase 4 deployment), a bigger local LLM (27B needs a ~24 GB GPU anyway), or a hosted rerank API (Cohere/Jina/Voyage).
- **Next levers, unmeasured for quality:** rerank 15 paper candidates instead of 30 (`k_rerank`); skip reranking products (8 reach the model, filters do most of the narrowing). The ~20 test questions should decide both.

## D19. A small eval set of owner questions, scored automatically (start of Phase 6)
- **Why:** changes were judged by running one or two questions by hand. That's how the "no recommendation" problem was found (D17), but it can't show whether a change helps overall, and it can't decide open questions like `k_rerank` 15 vs 30 or whether qwen3:4b is good enough.
- **Questions:** `eval/questions.toml`, 20 `[[q]]` entries written for the owner to edit: choosing a food (wet/dry/kitten/senior/picky/treats), allergies (`exclude = [...]`, set like a profile form since the app doesn't read allergens from text), health (overweight, kidney and urinary marked GAP: no papers yet), and knowledge questions (raw, vegan, grain-free, probiotics, fish oil, insects, heavy metals, gut bacteria). `[q.expect]` says what a good answer must do: `life_stage`/`food_form` (`"none"` = no filter), `min_recs`, `vet_note`, `papers` (title fragments).
- **Runner:** `petfood-eval` / `make eval` (`src/petfood/eval/run.py`). Loads models once, runs each question through `answer()`, writes `eval/results/<time>-<llm>.jsonl` (git-ignored: regenerated every run), prints a table and the reason for each failure. Flags: `--llm local`, `--only id …`, `--k-rerank N`, `--no-rerank`.
- **Checks** (`score()`, pure, tested in `tests/test_eval.py`): filters read from the question; enough recommendations; every recommended product fits the profile used (form, life stage with the D17 rule, no excluded allergen, has an ingredient list); vet note; expected paper retrieved and cited; nothing removed by `check_answer`. They catch basic mistakes, not bad answers: read the answers too.
- **Committed, not private:** the questions and runner are versioned with the code; run outputs aren't. A private set built from real users' questions would come later and stay out of the public repo.
- **Baseline (2026-11-07, DeepSeek, k_rerank 30, `20261107-214134-deepseek.jsonl`):** no errors; filters 11/11, recs 17/17, products fit 17/17, vet note **6/7**, paper retrieved **12/13**, cited 12/13, nothing removed 20/20; **13.0 s average** per answer (15.7 s slowest), since models load once.
- **Problems it found (to fix next):**
  1. *Chicken allergy got no vet note.* The `food_allergy` topic pattern (`ingest/tagging.py`) matches "food allergy" but not "allergic to chicken". A real bug.
  2. *Products pushed on knowledge questions.* "Can my cat eat a vegan diet?" said no listed food is vegan, then recommended three meat foods; "Do dry foods contain heavy metals?" recommended three unrelated products. Cause: D17's prompt rule ("recommend none only if nothing can serve") makes the model always pick. The checks missed it: needs a `max_recs = 0` expectation on such questions.
  - Also: picky-eater didn't retrieve the "food preference" paper, but that paper is about a new protein, not picky eating; the expectation is weak.

## D20. Latency breakdown per stage
- **Why:** answers took 13–50 s and the cause kept being guessed (padding, memory pressure, DeepSeek). Each stage now records its own time, so the slow part is measured, not guessed.
- **Code:**
  - `ChunkStore.search_dense/search_sparse(timings=)` split query embedding from the Qdrant call.
  - `HybridRetriever.retrieve()` adds `rerank_n` (texts scored).
  - `gather_context(timings=)` prefixes each search's stages with `products.` / `papers.`.
  - `answer()` adds `prompt`, `prompt_chars`, `generate`, `check`.
  - The clients record `last_stats`: DeepSeek gives tokens in/out/cached and **attempts** (more than 1 = a retry, the likely cause of the 108 s call on 2026-11-06); Ollama gives load / read-prompt / write times. Callers copy it into `Answer.llm_stats`.
  - `recommend/latency.py` groups and formats it: `petfood-answer` prints a bar chart per answer (plus model start-up, not counted in the total), and `petfood-eval` prints average / median / p90 / max per stage.
  - Tests: `tests/test_latency.py`, plus a `last_stats` test in `tests/test_ollama.py`.
- **Measured (20 eval questions, DeepSeek, `k_rerank` 30, batch 1, 2026-11-08, on battery with Low Power Mode on):**

  | Stage | avg | p90 | max | share |
  |---|---|---|---|---|
  | products: embed + Qdrant | 0.05 s | 0.1 s | 0.1 s | 0% |
  | products: rerank (0–30 cards, avg ~20) | 3.0 s | 4.6 s | 6.1 s | 13% |
  | papers: embed + Qdrant | 0.04 s | 0.06 s | 0.1 s | 0% |
  | **papers: rerank (always 30 excerpts)** | **16.5 s** | 18.7 s | 20.2 s | **74%** |
  | DeepSeek (~3,100 tokens in, ~300 out, no retries) | 2.8 s | 3.3 s | 3.5 s | 12% |
  | prompt + checks + other | <0.01 s | | | 0% |
  | **total** | **22.4 s** | 25.1 s | 27.6 s | |

  Model start-up: ~3 s per command (once per eval run).
- **Findings:**
  1. **Reranking paper excerpts is three quarters of the time**, ~0.55 s per excerpt (they're long, ~300 words). Product cards are shorter: ~0.15–0.2 s each. Embedding, Qdrant, prompt building and the checks are all negligible.
  2. **Power mode doubles CPU stages.** The same eval on 2026-11-07 (plugged in) had search at 10.2 s average; on battery with Low Power Mode it was 19.6 s. DeepSeek, which isn't on the Mac, was unchanged (2.7 vs 2.8 s). Compare timings only between runs in the same power state. This also explains much of the earlier "unexplained" run-to-run variance (D17, D18).
  3. Knowledge questions rerank up to 30 products that the answer then mostly shouldn't recommend.
- **Levers, in order of payoff:** fewer paper candidates (`--k-rerank 15` should roughly halve 74% of the time); shorter paper text for the reranker (e.g. first ~150 words); skip product reranking (13%); a smaller reranker model. Each needs an eval run to check the quality cost (D19).
- **`--k-rerank 15` tried (2026-11-08, plugged in, same session as a k=30 run):**
  - **Speed:** 12.1 s → **7.9 s** average (paper rerank 7.8 → 3.9 s, product rerank 1.3 → 0.8 s). Plugged in, k=30 took 12.1 s, against 22.4 s on battery: the power-mode effect again.
  - **Checks:** identical (filters 11/11, recs 17/17, fit 17/17, vet 6/7, found/cited 12/13). Two "removed sentence" warnings at 15 are LLM noise: those two questions got exactly the same excerpts in both runs, and the answers still differed.
  - **But the evidence changed a lot:** only **62 of 98** excerpts given to the model were the same. Retrieval is deterministic and rerank scores don't depend on k, so the other 36 (37%) were excerpts the reranker picked from **fusion ranks 16–30**. The first-stage ranking (dense + BM25 + RRF) often places the best excerpt below 15, and the reranker rescues it. Some swaps look worse: e.g. wet-vs-dry-water lost the hydration study and the kibble-vs-canned microbiome paper and gained the ultra-processed review.
  - **Also learned:** DeepSeek at temperature 0 isn't deterministic; identical prompts gave different wording. A single answer comparison can't show a quality difference, only patterns over many questions.
  - **Decision:** keep 30 for now. The checks are too coarse to see the evidence loss (they only look for one expected paper). Next: shorter text for the reranker (same 30 candidates, less to read per candidate), or k=20; and a stronger quality check (graded relevance of the 5 excerpts).
