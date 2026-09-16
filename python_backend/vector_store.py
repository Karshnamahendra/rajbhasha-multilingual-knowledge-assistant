import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger("VectorStore")
logger.setLevel(logging.INFO)


class VectorStore:

    def __init__(self):
        self.qdrant_url = getattr(settings, "QDRANT_URL", "http://localhost:6333")
        logger.info(f"[VectorStore] Connecting to Qdrant at {self.qdrant_url}...")

        try:
            self.client = QdrantClient(url=self.qdrant_url, timeout=10)
            # Verify connectivity
            self.client.get_collections()
            logger.info(f"[VectorStore] Successfully connected to Qdrant at {self.qdrant_url}")
        except Exception as e:
            logger.error(f"[VectorStore] Failed to connect to Qdrant at {self.qdrant_url}: {e}")
            raise RuntimeError(f"Qdrant vector database is unavailable at {self.qdrant_url}: {e}")

        # -----------------------------------------
        # Embedding Model (384-dimensional)
        # -----------------------------------------
        self.model_name = getattr(settings, "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
        logger.info(f"[VectorStore] Loading SentenceTransformer model '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dimension = 384

        # -----------------------------------------
        # Collection Names
        # -----------------------------------------
        self.index_collection_name = getattr(settings, "QDRANT_INDEX_COLLECTION_NAME", "rajbhasha_index_collection")
        self.content_collection_name = getattr(settings, "QDRANT_CONTENT_COLLECTION_NAME", "rajbhasha_content_collection")
        self.table_collection_name = getattr(settings, "QDRANT_TABLE_COLLECTION_NAME", "rajbhasha_table_collection")

        # Ensure collections exist in Qdrant
        self._ensure_collections()

        logger.info(
            f"[VectorStore] Qdrant Collections initialized: "
            f"Index='{self.index_collection_name}', "
            f"Content='{self.content_collection_name}', "
            f"Table='{self.table_collection_name}'"
        )

    def _ensure_collections(self):
        """Create collections with 384-dim COSINE distance if they do not exist."""
        for coll_name in [self.index_collection_name, self.content_collection_name, self.table_collection_name]:
            try:
                if not self.client.collection_exists(coll_name):
                    self.client.create_collection(
                        collection_name=coll_name,
                        vectors_config=models.VectorParams(
                            size=self.embedding_dimension,
                            distance=models.Distance.COSINE
                        )
                    )
                    logger.info(f"[VectorStore] Created Qdrant collection: {coll_name}")
            except Exception as e:
                logger.error(f"[VectorStore] Error ensuring collection {coll_name}: {e}")
                raise

    def embed_fn(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Compatible embedding callable for external components."""
        if isinstance(texts, str):
            texts = [texts]
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs.tolist()

    @staticmethod
    def _to_uuid(raw_id: str) -> str:
        """Deterministic UUIDv5 conversion to guarantee idempotency across re-ingestion."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))

    # =========================================================
    # ADD / UPSERT CHUNKS
    # =========================================================

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        collection_type: str = "content"
    ):
        if not chunks:
            return

        target_coll = self.index_collection_name if collection_type == "index" else (
            self.table_collection_name if collection_type == "table" else self.content_collection_name
        )

        texts = [str(c.get("text", "")) for c in chunks]
        embeddings = self.model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)

        points = []
        for idx, c in enumerate(chunks):
            raw_id = str(c.get("id") or f"chunk_{idx}")
            point_id = self._to_uuid(raw_id)
            raw_meta = c.get("metadata", {})

            doc_id = str(raw_meta.get("document_id", "default_doc"))
            doc_name = str(raw_meta.get("document_name", raw_meta.get("source_filename", doc_id)))
            doc_type = str(raw_meta.get("document_type", "magazine")).lower()
            year_val = raw_meta.get("year")
            try:
                year_int = int(year_val) if year_val is not None else None
            except (ValueError, TypeError):
                year_int = None
            content_type_val = str(raw_meta.get("content_type", collection_type)).lower()
            page_val = raw_meta.get("page", raw_meta.get("page_number"))
            try:
                page_int = int(page_val) if page_val is not None else None
            except (ValueError, TypeError):
                page_int = None

            payload = {
                "id": raw_id,
                "chunk_id": raw_id,
                "document_id": doc_id,
                "document_name": doc_name,
                "document_type": doc_type,
                "year": year_int,
                "content_type": content_type_val,
                "page": page_int,
                "page_number": page_int,
                "source_filename": str(raw_meta.get("source_filename", doc_name)),
                "location": str(raw_meta.get("location", f"page {page_int or 'doc'}")),
                "collection_type": str(collection_type),
                "section": str(raw_meta.get("section", "General")),
                "has_normalized": bool(raw_meta.get("has_normalized", False)),
                "text": str(c.get("text", "")),
                "chunk_type": str(raw_meta.get("chunk_type", collection_type))
            }

            if raw_meta.get("author"):
                payload["author"] = str(raw_meta.get("author"))
            if raw_meta.get("article_title"):
                payload["article_title"] = str(raw_meta.get("article_title"))

            if collection_type == "table":
                payload.update({
                    "chunk_type": "table",
                    "table_id": str(raw_meta.get("table_id", "")),
                    "table_index": int(raw_meta.get("table_index", 0)),
                    "row_count": int(raw_meta.get("row_count", 0)),
                    "column_count": int(raw_meta.get("column_count", 0)),
                    "extraction_method": str(raw_meta.get("extraction_method", "unknown")),
                    "extraction_confidence": float(raw_meta.get("extraction_confidence", 0.0)),
                    "field_label": str(raw_meta.get("field_label", "")),
                    "field_value_state": str(raw_meta.get("field_value_state", "")),
                })

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embeddings[idx].tolist(),
                    payload=payload
                )
            )

        # Batch upsert into Qdrant
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=target_coll,
                points=batch,
                wait=True
            )

        logger.info(
            f"[VectorStore] Upserted {len(chunks)} chunks into Qdrant [{target_coll}]."
        )

    # =========================================================
    # VECTOR SEARCH (ACROSS DUAL COLLECTIONS OR TARGETED)
    # =========================================================

    def search_by_vector(
        self,
        query: str,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        year: Optional[int] = None,
        content_type: str = "all",
        collection_type: str = "all",
        n_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Perform a vector similarity search across Qdrant collections with optional filters."""
        if not query:
            return []

        query_vector = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False).tolist()

        # Build dynamic multi-attribute filter
        conditions = []
        if document_id:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                )
            )
        if document_type:
            conditions.append(
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value=document_type.lower())
                )
            )
        if year is not None:
            conditions.append(
                models.FieldCondition(
                    key="year",
                    match=models.MatchValue(value=int(year))
                )
            )
        if content_type and content_type != "all":
            conditions.append(
                models.FieldCondition(
                    key="content_type",
                    match=models.MatchValue(value=content_type.lower())
                )
            )

        query_filter = models.Filter(must=conditions) if conditions else None

        # Resolve which physical collections to query
        effective_coll = collection_type if collection_type != "all" else content_type
        collections_to_query = []
        if effective_coll == "index":
            collections_to_query = [self.index_collection_name]
        elif effective_coll == "content":
            collections_to_query = [self.content_collection_name]
        elif effective_coll == "table":
            collections_to_query = [self.table_collection_name]
        else:  # "all"
            collections_to_query = [
                self.index_collection_name,
                self.content_collection_name,
                self.table_collection_name
            ]

        candidates = []
        seen_ids = set()

        for coll_name in collections_to_query:
            try:
                response = self.client.query_points(
                    collection_name=coll_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=n_results,
                    with_payload=True,
                    with_vectors=False
                )
                points = response.points

                for pt in points:
                    payload = pt.payload or {}
                    raw_id = payload.get("id") or payload.get("chunk_id") or str(pt.id)
                    if raw_id not in seen_ids:
                        seen_ids.add(raw_id)
                        # Cosine similarity is in pt.score (1.0 = identical).
                        # Distance = max(0.0, 1.0 - score) for compatibility with reranker/retriever
                        score = float(pt.score) if pt.score is not None else 0.0
                        distance = max(0.0, 1.0 - score)
                        candidates.append({
                            "id": raw_id,
                            "text": payload.get("text", ""),
                            "metadata": payload,
                            "distance": distance,
                            "score": score
                        })
            except Exception as e:
                logger.warning(f"[VectorStore] Search query failed on Qdrant collection {coll_name}: {e}")

        # If we got any candidates, return the top ones sorted by distance
        if candidates:
            candidates.sort(key=lambda x: x.get("distance", 1.0))
            return candidates[:n_results]

        # ----------- Fallback: simple keyword search via scroll -----------
        logger.info("[VectorStore] No vector results; falling back to keyword match.")
        fallback_results = []
        for coll_name in collections_to_query:
            try:
                scroll_res, _ = self.client.scroll(
                    collection_name=coll_name,
                    scroll_filter=query_filter,
                    limit=200,
                    with_payload=True
                )
                for pt in scroll_res:
                    payload = pt.payload or {}
                    text = payload.get("text", "")
                    if query.lower() in text.lower():
                        raw_id = payload.get("id") or payload.get("chunk_id") or str(pt.id)
                        fallback_results.append({
                            "id": raw_id,
                            "text": text,
                            "metadata": payload,
                            "distance": 0.0,
                            "score": 1.0
                        })
            except Exception as e:
                logger.warning(f"[VectorStore] Keyword fallback failed on {coll_name}: {e}")

        fallback_results.sort(key=lambda x: len(x["text"]))
        return fallback_results[:n_results]

    # =========================================================
    # FETCH DISTINCT AUTHORS FOR A QUERY
    # =========================================================

    def get_authors_for_query(
        self,
        query: str,
        n_results: int = 20
    ) -> List[str]:
        if not query:
            return []

        results = self.search_by_vector(query, collection_type="content", n_results=n_results)
        authors: List[str] = []
        seen: set = set()
        for entry in results:
            meta = entry.get("metadata", {})
            author = meta.get("author")
            if author and author not in seen:
                seen.add(author)
                authors.append(author)
        return authors

    # =========================================================
    # GET ALL CHUNKS FOR ONE DOCUMENT
    # =========================================================

    def get_all_document_chunks(
        self,
        document_id: Optional[str] = None,
        collection_type: str = "all"
    ) -> List[Dict[str, Any]]:
        query_filter = None
        if document_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )

        collections_to_get = []
        if collection_type == "index":
            collections_to_get = [self.index_collection_name]
        elif collection_type == "content":
            collections_to_get = [self.content_collection_name]
        elif collection_type == "table":
            collections_to_get = [self.table_collection_name]
        else:
            collections_to_get = [
                self.index_collection_name,
                self.content_collection_name,
                self.table_collection_name
            ]

        candidates = []
        for coll_name in collections_to_get:
            try:
                offset = None
                while True:
                    scroll_res, next_offset = self.client.scroll(
                        collection_name=coll_name,
                        scroll_filter=query_filter,
                        limit=250,
                        offset=offset,
                        with_payload=True
                    )
                    for pt in scroll_res:
                        payload = pt.payload or {}
                        raw_id = payload.get("id") or payload.get("chunk_id") or str(pt.id)
                        candidates.append({
                            "id": raw_id,
                            "text": payload.get("text", ""),
                            "metadata": payload
                        })
                    if not next_offset:
                        break
                    offset = next_offset
            except Exception as e:
                logger.warning(f"[VectorStore] get_all_document_chunks failed on {coll_name}: {e}")

        return candidates

    # =========================================================
    # DELETE ONE DOCUMENT ACROSS ALL COLLECTIONS
    # =========================================================

    def delete_document(
        self,
        document_id: str
    ):
        if not document_id:
            return

        delete_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                )
            ]
        )

        for coll_name in [self.index_collection_name, self.content_collection_name, self.table_collection_name]:
            try:
                self.client.delete(
                    collection_name=coll_name,
                    points_selector=models.FilterSelector(filter=delete_filter),
                    wait=True
                )
            except Exception as e:
                logger.warning(f"[VectorStore] Delete from {coll_name} notice: {e}")

        logger.info(f"[VectorStore] Deleted document '{document_id}' from Qdrant collections.")

    # =========================================================
    # INSPECT CHUNK VECTOR (FOR FRONTEND PIPELINE MODAL)
    # =========================================================

    def get_chunk_vector(self, chunk_id: str) -> Optional[List[float]]:
        """Retrieve the raw 384-dimensional embedding vector for a given chunk ID."""
        if not chunk_id:
            return None
        point_id = self._to_uuid(chunk_id)
        for coll_name in [self.content_collection_name, self.index_collection_name, self.table_collection_name]:
            try:
                records = self.client.retrieve(
                    collection_name=coll_name,
                    ids=[point_id],
                    with_vectors=True
                )
                if records and records[0].vector:
                    vec = records[0].vector
                    if isinstance(vec, list):
                        return vec
                    elif hasattr(vec, "tolist"):
                        return vec.tolist()
            except Exception:
                pass
        return None

    # =========================================================
    # COLLECTION STATISTICS
    # =========================================================

    def get_collection_stats(self, collection_name: str) -> dict:
        """Return points_count, vectors_count, and status for a collection."""
        try:
            info = self.client.get_collection(collection_name)
            points_count = info.points_count if info.points_count is not None else 0
            vectors_count = getattr(info, "vectors_count", points_count)
            if vectors_count is None:
                vectors_count = points_count
            status_val = info.status.value if hasattr(info.status, "value") else str(info.status)
            return {
                "points_count": points_count,
                "vectors_count": vectors_count,
                "status": status_val
            }
        except Exception as e:
            logger.warning(f"[VectorStore] get_collection_stats failed for {collection_name}: {e}")
            return {"points_count": 0, "vectors_count": 0, "status": "unknown"}

    def get_stats(self) -> dict:
        """Return statistics across all 3 Rajbhasha collections."""
        return {
            "index": self.get_collection_stats(self.index_collection_name),
            "content": self.get_collection_stats(self.content_collection_name),
            "table": self.get_collection_stats(self.table_collection_name)
        }
