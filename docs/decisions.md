# Design decisions

Short records of each design decision, written so I can explain the trade-offs in an interview.

## D1. Paper source: Europe PMC rather than scraping journals
- Europe PMC's REST API returns metadata, the abstract, the **license**, and JATS full text for open-access articles. One API covers PubMed Central plus European content.
- Only **Creative Commons** licensed papers are stored. We keep and quote full text, so the license has to permit that. Each paper's license is stored so it can be shown on citations.
- Papers are filtered after the search with species keyword rules. The keyword query pulls in some human-only papers, and these get dropped.

## D2. Product source: Open Pet Food Facts + curated CSV, not Amazon or Petco scraping
- Amazon's and Petco's terms of service prohibit scraping. Petco has no public API, and Amazon's PA-API needs an Associates account and returns little nutrition data.
- Open Pet Food Facts is open data (ODbL) with ingredients and nutrition values.
- The weak spot is **coverage of US brands**. The curated CSV fills gaps with data from manufacturer pages.
- The `products.source` column keeps the design source-agnostic, so a retailer API can be added later without schema changes.

## D3. Postgres is the source of truth and Qdrant is a derived index
- Relational data (products, papers, evaluation labels, query logs) needs joins, constraints and audit history.
- Vectors can be regenerated whenever the embedding model or chunking changes. Keeping `chunks` in Postgres makes re-indexing a batch job, not a data migration.

## D4. Allergen safety comes from deterministic filters, not the LLM
- Rule-based tagging produces `allergen_tags`, and the filtering happens in SQL or as a Qdrant payload filter **before** retrieval.
- Tagging is **conservative**: "chicken fat" and "fish oil" are tagged as chicken and fish. Wrongly calling a food "safe" costs more than being over-cautious.
- This is also easy to test. Constraint-violation rate is one of the Phase 6 metrics.

## D5. Raw files go in object storage
- The raw JATS XML is kept in S3 (SeaweedFS locally), so we can re-parse later (for example with better section handling) without downloading everything from Europe PMC again.
- SeaweedFS is S3-compatible locally (MinIO stopped publishing Docker images). Moving to AWS S3 is a configuration change: clear `S3_ENDPOINT_URL` and set real keys.

## D6. Hybrid retrieval with RRF, then reranking (Phase 2)
- Dense embeddings miss exact clinical terms ("struvite", "taurine"). BM25 misses paraphrases. Combining both covers each one's weakness.
- RRF needs no score calibration between the two retrievers.
- A cross-encoder reranker on about 30 candidates gives the largest precision gain for its cost, and only the top 5 go to the LLM. Fewer tokens means lower cost and fewer distractions for the model.

## D7. Router: rules first, then a small classifier (Phase 3)
- Start with transparent rules: number of conditions, medical keywords, multi-pet questions.
- Next, a logistic regression on the query embeddings we already compute. It's nearly free at inference time, and DeepSeek labels about 500 queries as training data.
- Move to a fine-tuned DistilBERT/ModernBERT only if the metrics show the simpler router falls short.
- Metrics: routing accuracy, cost per request, and the quality gap between the local model and DeepSeek.

## D8. Chunking: section-aware windows with a contextual header (Phase 2)
- Chunks never cross section boundaries. Each one is about 300 words, split at sentence boundaries, with about 50 words of overlap.
- Every chunk starts with `"<paper title> — <section heading>"`. A passage that only says "the cats lost weight" becomes findable by what the study was about. This helps BM25 and dense search alike.
- Methods sections are indexed but flagged `low_value`, and the reranker score is down-weighted. They rarely answer an owner's question but can still be retrieved when needed.
- Results are capped at 2 chunks per paper, so the LLM sees evidence from several studies instead of one paper repeated.
- Chunk ids are `uuid5(doc_type:doc_id:ord)`. Re-indexing overwrites points instead of duplicating them.

## D9. Models: bge-small + Qdrant BM25 + bge-reranker-base, all through fastembed
- **fastembed** runs ONNX models on the CPU, with no PyTorch and a small install. It's fast on Apple Silicon.
- **Dense: `BAAI/bge-small-en-v1.5`** (384 dimensions). Good quality for its size, and the corpus embeds in about a minute.
- **Sparse: `Qdrant/bm25` + `Modifier.IDF`.** The client sends term frequencies and Qdrant applies IDF over the live collection, so it's real BM25 with no separate Elasticsearch to run.
- **Rerank: `BAAI/bge-reranker-base`**, a cross-encoder. It reads the query and passage together, so it's far more precise than comparing two separate embeddings. Too slow to run over the whole corpus, so it only rescores the top 30 after fusion.
- All three model names are set in `.env`, so we can swap them and measure the difference in Phase 6.

## D10. Fusion is done in our code, not with Qdrant's built-in RRF
- Qdrant can fuse server-side. Doing it in our code keeps each candidate's dense rank, BM25 rank, RRF rank and rerank score. That gives free ablations for evaluation (`--mode dense|sparse|hybrid|hybrid_rerank`) and makes it possible to debug why a result came first.
- The cost is two network round-trips instead of one, which is negligible locally.

## D11. Tables are their own sections and chunks
- v1 dropped tables to keep prose clean. Reviewing a real paper showed that loses the most useful facts in nutrition papers: diet composition, nutrient percentages and outcomes.
- Each `<table-wrap>` now becomes a `table` section: label + caption, markdown rows, footnotes. Figure captions become `figure` sections. Prose still excludes them, so nothing is duplicated.
- Tables are split by **rows**, and every piece repeats the caption and header row, so `| Crude protein | 45.2 |` never loses its context.
- This needed no re-download: `petfood-ingest reparse` re-parsed the raw XML already kept in S3. That's the payoff of D5.

## D12. Two years per paper: publication year and study period
- **Publication year** comes straight from Europe PMC's `pubYear` and is always known. Filter with `--pub-from/--pub-to`.
- **Study period** (when the data was collected) has no metadata field and most papers never state it, so `petfood-ingest study-years` extracts it in three steps:
  1. A regex keeps sentences that still mention a year once citations ("Smith et al., 2014", "(AAFCO, 2016)"), citation fragments and measurements ("2000 ng/g") are stripped. Papers with none never reach the LLM.
  2. DeepSeek reads the numbered sentences and answers `{start, end, sentence_ids}`. The prompt tells it to ignore cited studies, literature searches and guideline years, and to answer null for reviews.
  3. Code checks the answer: the years must appear in the cited sentences, start ≤ end, and the study must end no later than the **received** year (from JATS `<history>`), else the publication year. Anything that fails is stored as `rejected` with the reason, never as a filterable year.
- **Unknown study periods still filter (option B).** A study can't end after the manuscript was submitted, so every paper gets `study_year_latest` = stated end, else received year, else publication year. `--study-from Y` keeps papers whose `study_year_latest ≥ Y`. `--study-to Y` keeps papers whose stated start ≤ Y, *and* papers with no stated start, since we can't rule them out. So "studies from before 2010" is a weak filter for papers that don't state their dates.
- Year filters apply to papers only; products pass through unchanged.
- The evidence sentence is stored next to the years, so any value can be checked by eye.

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
