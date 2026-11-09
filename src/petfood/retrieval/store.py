"""Qdrant collection: one point per chunk, with a dense vector and a BM25 sparse vector."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from qdrant_client import QdrantClient, models

from petfood.retrieval.chunking import Chunk
from petfood.retrieval.embeddings import DenseEncoder, Sparse, SparseEncoder

DENSE = "dense"
SPARSE = "bm25"

# Payload fields we filter on -> index them so filtering stays fast as the corpus grows.
_PAYLOAD_INDEXES = {
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "species": models.PayloadSchemaType.KEYWORD,
    "topics": models.PayloadSchemaType.KEYWORD,
    "section_type": models.PayloadSchemaType.KEYWORD,
    "allergen_tags": models.PayloadSchemaType.KEYWORD,
    "life_stage": models.PayloadSchemaType.KEYWORD,
    "food_form": models.PayloadSchemaType.KEYWORD,
    "pub_year": models.PayloadSchemaType.INTEGER,
    "study_year_start": models.PayloadSchemaType.INTEGER,
    "study_year_latest": models.PayloadSchemaType.INTEGER,
    "doc_id": models.PayloadSchemaType.INTEGER,
    "low_value": models.PayloadSchemaType.BOOL,
}


def _or_unknown(key: str, rng: models.Range) -> models.Filter:
    """Range condition that also keeps points where the field is missing or null: an unknown
    value can't rule a paper out."""
    return models.Filter(should=[
        models.FieldCondition(key=key, range=rng),
        models.IsEmptyCondition(is_empty=models.PayloadField(key=key)),
    ])


# Which product labels suit each requested life stage (D17). AAFCO defines only two nutrient
# profiles: growth (kittens/puppies) and adult maintenance; "senior" is a marketing label. So a
# label that states no life stage is an adult food: it suits adults and seniors, never kittens.
LIFE_STAGE_MATCHES = {
    "kitten": ["kitten", "growth", "all"],
    "puppy": ["puppy", "growth", "all"],
    "adult": ["adult", "all"],
    "senior": ["senior", "adult", "all"],
}
UNSTATED_LIFE_STAGE_OK = {"adult", "senior"}


def _life_stage_condition(stage: str) -> models.Condition:
    stated = models.FieldCondition(key="life_stage",
                                   match=models.MatchAny(any=LIFE_STAGE_MATCHES.get(stage, [stage, "all"])))
    if stage not in UNSTATED_LIFE_STAGE_OK:
        return stated
    return models.Filter(should=[stated, models.IsEmptyCondition(is_empty=models.PayloadField(key="life_stage"))])


@dataclass
class SearchFilters:
    doc_type: str | None = None                 # "paper" | "product"
    species: str | None = None                  # "cat" | "dog"
    exclude_allergens: Sequence[str] = ()       # products containing any of these are dropped
    life_stage: str | None = None               # products only
    food_form: str | None = None                # products only
    pub_year_from: int | None = None            # papers only, inclusive
    pub_year_to: int | None = None
    # Papers whose study period could overlap [from, to]. Most papers don't state it: their
    # start is unknown and their end is bounded by the received/published year (D12).
    study_year_from: int | None = None
    study_year_to: int | None = None
    exclude_low_value: bool = False             # drop Methods-section chunks

    def to_qdrant(self) -> models.Filter | None:
        must, must_not = [], []
        if self.doc_type:
            must.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=self.doc_type)))
        if self.species:
            must.append(models.FieldCondition(key="species", match=models.MatchAny(any=[self.species])))
        if self.exclude_allergens:
            must_not.append(models.FieldCondition(
                key="allergen_tags", match=models.MatchAny(any=list(self.exclude_allergens))))
            # Allergen tags come from the ingredient list. A product without one has no tags and
            # would look allergen-free, so it can't be cleared and is dropped too (D4: conservative).
            must_not.append(models.Filter(must=[
                models.FieldCondition(key="doc_type", match=models.MatchValue(value="product")),
                models.FieldCondition(key="has_ingredients", match=models.MatchValue(value=False)),
            ]))
        if self.life_stage:
            must.append(_life_stage_condition(self.life_stage))
        if self.food_form:
            must.append(models.FieldCondition(key="food_form", match=models.MatchValue(value=self.food_form)))
        # year filters constrain papers only; products have no years and pass through
        years = []
        if self.pub_year_from or self.pub_year_to:
            years.append(models.FieldCondition(
                key="pub_year", range=models.Range(gte=self.pub_year_from, lte=self.pub_year_to)))
        if self.study_year_from:
            years.append(_or_unknown("study_year_latest", models.Range(gte=self.study_year_from)))
        if self.study_year_to:
            years.append(_or_unknown("study_year_start", models.Range(lte=self.study_year_to)))
        if years:
            is_paper = models.FieldCondition(key="doc_type", match=models.MatchValue(value="paper"))
            must.append(models.Filter(should=[models.Filter(must_not=[is_paper]),
                                              models.Filter(must=years)]))
        if self.exclude_low_value:
            must_not.append(models.FieldCondition(key="low_value", match=models.MatchValue(value=True)))
        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)


@dataclass
class Hit:
    id: str
    score: float
    payload: dict


class ChunkStore:
    def __init__(self, client: QdrantClient, collection: str, dense: DenseEncoder, sparse: SparseEncoder):
        self.client = client
        self.collection = collection
        self.dense = dense
        self.sparse = sparse

    # ── collection management ──
    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            return
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=self.dense.dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field, schema in _PAYLOAD_INDEXES.items():
            self.client.create_payload_index(self.collection, field, field_schema=schema)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    # ── writes ──
    def upsert(self, chunks: Sequence[Chunk], batch_size: int = 128) -> int:
        n = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            dense = self.dense.embed_documents(texts)
            sparse = self.sparse.embed_documents(texts)
            self.client.upsert(self.collection, points=[
                models.PointStruct(
                    id=c.id,
                    vector={DENSE: d, SPARSE: models.SparseVector(indices=s.indices, values=s.values)},
                    payload=c.payload | {"text": c.text, "ord": c.ord},
                )
                for c, d, s in zip(batch, dense, sparse)
            ])
            n += len(batch)
        return n

    def delete_doc(self, doc_type: str, doc_id: int) -> None:
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type)),
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
        ])))

    # ── reads ──
    # `timings`, if given, receives "<kind>_embed" and "<kind>_qdrant" in ms (latency breakdown)
    def search_dense(self, query: str, limit: int, filters: SearchFilters | None = None,
                     timings: dict | None = None) -> list[Hit]:
        t0 = perf_counter()
        vec = self.dense.embed_query(query)
        return self._timed(vec, DENSE, limit, filters, timings, "dense", t0)

    def search_sparse(self, query: str, limit: int, filters: SearchFilters | None = None,
                      timings: dict | None = None) -> list[Hit]:
        t0 = perf_counter()
        q: Sparse = self.sparse.embed_query(query)
        if not q.indices:
            return []
        vec = models.SparseVector(indices=q.indices, values=q.values)
        return self._timed(vec, SPARSE, limit, filters, timings, "sparse", t0)

    def _timed(self, vec, using, limit, filters, timings, kind, t0) -> list[Hit]:
        t1 = perf_counter()
        hits = self._query(vec, using, limit, filters)
        if timings is not None:
            timings[f"{kind}_embed"] = (t1 - t0) * 1000
            timings[f"{kind}_qdrant"] = (perf_counter() - t1) * 1000
        return hits

    def _query(self, vector, using: str, limit: int, filters: SearchFilters | None) -> list[Hit]:
        res = self.client.query_points(
            self.collection, query=vector, using=using, limit=limit, with_payload=True,
            query_filter=filters.to_qdrant() if filters else None,
        )
        return [Hit(id=str(p.id), score=p.score, payload=p.payload or {}) for p in res.points]
