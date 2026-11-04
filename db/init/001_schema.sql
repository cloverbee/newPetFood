-- Pet food RAG — core schema (Phase 1).
-- Runs automatically on first `docker compose up` (empty volume only).
-- Re-apply manually with: psql "$DATABASE_URL" -f db/init/001_schema.sql  (idempotent)

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ───────────── Veterinary papers ─────────────
CREATE TABLE IF NOT EXISTS papers (
    id                 BIGSERIAL PRIMARY KEY,
    source             TEXT NOT NULL DEFAULT 'europepmc',
    source_id          TEXT NOT NULL,              -- e.g. PMC1234567
    pmid               TEXT,
    doi                TEXT,
    title              TEXT NOT NULL,
    abstract           TEXT,
    authors            JSONB NOT NULL DEFAULT '[]',
    journal            TEXT,
    pub_year           INT,
    license            TEXT,                       -- cc by, cc0, cc by-nc ...
    species            TEXT[] NOT NULL DEFAULT '{}',  -- {cat,dog}
    topics             TEXT[] NOT NULL DEFAULT '{}',  -- {obesity,renal,allergy,...}
    s3_key             TEXT,                       -- raw JATS XML in object storage
    fulltext_available BOOLEAN NOT NULL DEFAULT FALSE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, source_id)
);
-- Study period (when the data was collected), extracted from the text by `petfood-ingest study-years`.
-- Most papers never state it, so these stay NULL; received_year then bounds the study from above.
ALTER TABLE papers ADD COLUMN IF NOT EXISTS received_year       INT;   -- JATS <date date-type="received">
ALTER TABLE papers ADD COLUMN IF NOT EXISTS study_year_start    INT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS study_year_end      INT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS study_year_status   TEXT;  -- found | no_dates | not_stated | rejected
ALTER TABLE papers ADD COLUMN IF NOT EXISTS study_year_evidence TEXT;  -- the sentence(s) the years came from
ALTER TABLE papers ADD COLUMN IF NOT EXISTS study_year_note     TEXT;  -- why a model answer was rejected
CREATE INDEX IF NOT EXISTS papers_species_idx ON papers USING GIN (species);
CREATE INDEX IF NOT EXISTS papers_topics_idx  ON papers USING GIN (topics);

-- Parsed sections of the full text (abstract, intro, methods, results, discussion...)
CREATE TABLE IF NOT EXISTS paper_sections (
    id           BIGSERIAL PRIMARY KEY,
    paper_id     BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    ord          INT NOT NULL,
    section_type TEXT NOT NULL,      -- abstract | introduction | methods | results | discussion | conclusion | other
    heading      TEXT,
    text         TEXT NOT NULL,
    UNIQUE (paper_id, ord)
);

-- ───────────── Products ─────────────
CREATE TABLE IF NOT EXISTS products (
    id                BIGSERIAL PRIMARY KEY,
    source            TEXT NOT NULL,             -- openpetfoodfacts | curated
    source_id         TEXT NOT NULL,             -- barcode or curated slug
    name              TEXT NOT NULL,
    brand             TEXT,
    species           TEXT,                      -- cat | dog | NULL (unknown)
    life_stage        TEXT,                      -- kitten | puppy | adult | senior | all
    food_form         TEXT,                      -- dry | wet | raw | treat | NULL
    ingredients_text  TEXT,
    ingredients       TEXT[] NOT NULL DEFAULT '{}',   -- normalized, in label order
    protein_sources   TEXT[] NOT NULL DEFAULT '{}',   -- {chicken,fish,...}
    allergen_tags     TEXT[] NOT NULL DEFAULT '{}',   -- {chicken,grain,dairy,...}
    grain_free        BOOLEAN,
    -- Guaranteed analysis, as-fed % (NULL when unknown)
    protein_pct       NUMERIC(5,2),
    fat_pct           NUMERIC(5,2),
    fiber_pct         NUMERIC(5,2),
    moisture_pct      NUMERIC(5,2),
    kcal_per_100g     NUMERIC(7,2),
    labels            TEXT[] NOT NULL DEFAULT '{}',   -- e.g. aafco-complete, organic
    url               TEXT,
    image_url         TEXT,
    raw               JSONB,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, source_id)
);
CREATE INDEX IF NOT EXISTS products_species_idx   ON products (species);
CREATE INDEX IF NOT EXISTS products_allergen_idx  ON products USING GIN (allergen_tags);
CREATE INDEX IF NOT EXISTS products_protein_idx   ON products USING GIN (protein_sources);
CREATE INDEX IF NOT EXISTS products_name_trgm_idx ON products USING GIN (name gin_trgm_ops);

-- ───────────── Chunks (filled in Phase 2; Qdrant holds the vectors) ─────────────
CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY,                -- same id as the Qdrant point
    doc_type    TEXT NOT NULL,                   -- paper | product
    doc_id      BIGINT NOT NULL,
    ord         INT NOT NULL,
    text        TEXT NOT NULL,
    token_count INT,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_type, doc_id, ord)
);

-- ───────────── Ingestion bookkeeping ─────────────
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status      TEXT NOT NULL DEFAULT 'running',  -- running | ok | failed
    n_fetched   INT NOT NULL DEFAULT 0,
    n_upserted  INT NOT NULL DEFAULT 0,
    n_skipped   INT NOT NULL DEFAULT 0,
    error       TEXT,
    params      JSONB NOT NULL DEFAULT '{}'
);
