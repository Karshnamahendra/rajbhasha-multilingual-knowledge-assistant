"""Unified Qdrant vector database client managing dense embeddings, sparse payload indices, and RRF fusion."""
import os
import re
import math
import uuid
import zlib
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
            self.client = QdrantClient(url=self.qdrant_url, timeout=3)
            # Verify connectivity
            self.client.get_collections()
            logger.info(f"[VectorStore] Successfully connected to Qdrant at {self.qdrant_url}")
        except Exception as e:
            fallback_dir = os.path.join(getattr(settings, "DATA_DIR", "data"), "qdrant_storage")
            os.makedirs(fallback_dir, exist_ok=True)
            logger.warning(
                f"[VectorStore] Cannot connect to Qdrant server at {self.qdrant_url} ({e}). "
                f"Falling back to embedded local Qdrant at '{fallback_dir}'..."
            )
            try:
                self.client = QdrantClient(path=fallback_dir)
                self.client.get_collections()
                logger.info(f"[VectorStore] Successfully initialized embedded Qdrant at '{fallback_dir}'")
            except Exception as e2:
                logger.error(f"[VectorStore] Failed to initialize embedded Qdrant: {e2}")
                raise RuntimeError(f"Qdrant is unavailable at {self.qdrant_url} and embedded fallback failed: {e2}")

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
        self.hierarchical_collection_name = getattr(settings, "QDRANT_HIERARCHICAL_COLLECTION_NAME", "rajbhasha_hierarchical_collection")

        # Ensure collections exist in Qdrant
        self._ensure_collections()

        logger.info(
            f"[VectorStore] Qdrant Collections initialized: "
            f"Index='{self.index_collection_name}', "
            f"Content='{self.content_collection_name}', "
            f"Table='{self.table_collection_name}', "
            f"Hierarchical='{self.hierarchical_collection_name}'"
        )

    @staticmethod
    def compute_sparse_vector(text: str) -> models.SparseVector:
        """Computes deterministic token hashes (31-bit positive integers) and sublinear weights for Qdrant sparse vectors."""
        tokens = re.findall(r'[\w\u0900-\u097F]+', (text or "").lower())
        if not tokens:
            return models.SparseVector(indices=[], values=[])
        tf: Dict[int, int] = {}
        for tok in tokens:
            idx = zlib.crc32(tok.encode("utf-8")) & 0x7FFFFFFF
            tf[idx] = tf.get(idx, 0) + 1
        indices = []
        values = []
        for idx, count in sorted(tf.items()):
            indices.append(idx)
            values.append(float(round(1.0 + math.log(count), 4)))
        return models.SparseVector(indices=indices, values=values)

    def _ensure_collections(self):
        """Create collections with named dense (384-dim COSINE) and native sparse vectors."""
        for coll_name in [
            self.index_collection_name,
            self.content_collection_name,
            self.table_collection_name,
            self.hierarchical_collection_name
        ]:
            try:
                needs_creation = True
                if self.client.collection_exists(coll_name):
                    try:
                        coll_info = self.client.get_collection(coll_name)
                        has_dense = isinstance(coll_info.config.params.vectors, dict) and "dense" in coll_info.config.params.vectors
                        has_sparse = coll_info.config.params.sparse_vectors and "sparse" in coll_info.config.params.sparse_vectors
                        if has_dense and has_sparse:
                            needs_creation = False
                        else:
                            logger.info(f"[VectorStore] Upgrading collection '{coll_name}' to named dense and sparse vectors...")
                            self.client.delete_collection(coll_name)
                    except Exception:
                        pass

                if needs_creation:
                    self.client.create_collection(
                        collection_name=coll_name,
                        vectors_config={
                            "dense": models.VectorParams(
                                size=self.embedding_dimension,
                                distance=models.Distance.COSINE
                            )
                        },
                        sparse_vectors_config={
                            "sparse": models.SparseVectorParams(
                                index=models.SparseIndexParams(on_disk=False)
                            )
                        }
                    )
                    logger.info(f"[VectorStore] Created Qdrant collection with named dense + sparse vectors: {coll_name}")

                # Ensure payload text index for multilingual search
                for text_field in ["text", "hierarchy_path_text"]:
                    try:
                        self.client.create_payload_index(
                            collection_name=coll_name,
                            field_name=text_field,
                            field_schema=models.TextIndexParams(
                                type="text",
                                tokenizer=models.TokenizerType.MULTILINGUAL,
                                lowercase=True
                            )
                        )
                    except Exception:
                        pass

                # Ensure payload metadata indices for fast filtering
                for field_name, schema_type in [
                    ("document_id", models.PayloadSchemaType.KEYWORD),
                    ("document_type", models.PayloadSchemaType.KEYWORD),
                    ("content_type", models.PayloadSchemaType.KEYWORD),
                    ("chunk_type", models.PayloadSchemaType.KEYWORD),
                    ("year", models.PayloadSchemaType.INTEGER),
                    ("page", models.PayloadSchemaType.INTEGER),
                    ("source_page", models.PayloadSchemaType.INTEGER),
                    ("author", models.PayloadSchemaType.KEYWORD),
                    ("node_type", models.PayloadSchemaType.KEYWORD),
                    ("node_key", models.PayloadSchemaType.KEYWORD),
                    ("file_name", models.PayloadSchemaType.KEYWORD),
                    ("quarter", models.PayloadSchemaType.KEYWORD),
                    ("report_period", models.PayloadSchemaType.KEYWORD),
                ]:
                    try:
                        self.client.create_payload_index(
                            collection_name=coll_name,
                            field_name=field_name,
                            field_schema=schema_type
                        )
                    except Exception:
                        pass
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

            filename_val = str(raw_meta.get("filename", raw_meta.get("source_filename", doc_name)))
            source_val = str(raw_meta.get("source", raw_meta.get("source_filename", doc_name)))

            payload = {
                "id": raw_id,
                "chunk_id": raw_id,
                "document_id": doc_id,
                "filename": filename_val,
                "source": source_val,
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
                    "chunk_type": str(raw_meta.get("chunk_type", "table")),
                    "table_id": str(raw_meta.get("table_id", "")),
                    "table_index": int(raw_meta.get("table_index", 0)),
                    "row_count": int(raw_meta.get("row_count", 0)),
                    "column_count": int(raw_meta.get("column_count", 0)),
                    "headers": raw_meta.get("headers", []),
                    "row_index": int(raw_meta.get("row_index")) if raw_meta.get("row_index") is not None else None,
                    "extraction_method": str(raw_meta.get("extraction_method", "unknown")),
                    "extraction_confidence": float(raw_meta.get("extraction_confidence", 0.0)),
                    "field_label": str(raw_meta.get("field_label", "")),
                    "field_value_state": str(raw_meta.get("field_value_state", "")),
                })

            sparse_vec = self.compute_sparse_vector(str(c.get("text", "")))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": embeddings[idx].tolist(),
                        "sparse": sparse_vec
                    },
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
    # ADD / UPSERT HIERARCHICAL NODES
    # =========================================================

    def add_hierarchical_points(
        self,
        points_data: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """Indexes hierarchical nodes (both leaf factual values and branch structural paths) into Qdrant.

        Generates:
          - 384-dimensional dense vectors from the complete hierarchy_path_text
          - Deterministic lexical sparse vectors from the complete hierarchy_path_text
          - Deterministic UUID5 point IDs ensuring idempotent upsertion without duplicates
        """
        if not points_data:
            return 0

        target_coll = self.hierarchical_collection_name

        # 1. Extract texts to embed
        texts = [str(p.get("hierarchy_path_text") or p.get("text", "")) for p in points_data]
        embeddings = self.model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)

        # 2. Build PointStruct list
        qdrant_points = []
        for idx, p in enumerate(points_data):
            path_text = texts[idx]
            doc_id = str(p.get("document_id", "default_doc"))
            raw_id = str(p.get("id") or f"{doc_id}::hier::{path_text}")
            point_id = self._to_uuid(raw_id)

            file_name = str(p.get("file_name", p.get("filename", doc_id)))
            doc_type = str(p.get("document_type", "magazine")).lower()

            year_val = p.get("year")
            try:
                year_int = int(year_val) if year_val is not None else None
            except (ValueError, TypeError):
                year_int = None

            page_val = p.get("source_page", p.get("page", p.get("page_number")))
            try:
                page_int = int(page_val) if page_val is not None else None
            except (ValueError, TypeError):
                page_int = None

            sparse_vec = self.compute_sparse_vector(path_text)

            payload = {
                "id": raw_id,
                "document_id": doc_id,
                "file_name": file_name,
                "filename": file_name,
                "source_filename": file_name,
                "document_type": doc_type,
                "year": year_int,
                "quarter": p.get("quarter"),
                "report_period": p.get("report_period"),
                "source_page": page_int,
                "page": page_int,
                "page_number": page_int,
                "node_key": str(p.get("node_key", "")),
                "node_value": str(p.get("node_value")) if p.get("node_value") is not None else None,
                "hierarchy_path": p.get("hierarchy_path", []),
                "hierarchy_path_text": path_text,
                "text": path_text,
                "node_type": str(p.get("node_type", "leaf")),
            }

            qdrant_points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": embeddings[idx].tolist(),
                        "sparse": sparse_vec
                    },
                    payload=payload
                )
            )

        # 3. Batch upsert into Qdrant
        for i in range(0, len(qdrant_points), batch_size):
            batch = qdrant_points[i:i + batch_size]
            self.client.upsert(
                collection_name=target_coll,
                points=batch,
                wait=True
            )

        logger.info(
            f"[VectorStore] Upserted {len(points_data)} hierarchical points into Qdrant [{target_coll}]."
        )
        return len(points_data)

    def delete_hierarchical_points(self, document_id: str) -> None:
        """Deletes all hierarchical points for a given document_id from the hierarchical collection."""
        try:
            self.client.delete(
                collection_name=self.hierarchical_collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                ),
                wait=True
            )
            logger.info(f"[VectorStore] Deleted hierarchical points for document_id='{document_id}'")
        except Exception as e:
            logger.warning(f"[VectorStore] Non-critical warning deleting hierarchical points for {document_id}: {e}")

    def get_hierarchical_points(
        self,
        document_id: Optional[str] = None,
        limit: int = 50,
        node_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Debug inspection method: scrolls hierarchical points with payloads and vector details."""
        try:
            must_filters = []
            if document_id:
                must_filters.append(
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                )
            if node_type:
                must_filters.append(
                    models.FieldCondition(
                        key="node_type",
                        match=models.MatchValue(value=node_type)
                    )
                )
            query_filter = models.Filter(must=must_filters) if must_filters else None

            records, _ = self.client.scroll(
                collection_name=self.hierarchical_collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True
            )

            results = []
            for r in records:
                vec_info: Dict[str, Any] = {}
                if isinstance(r.vector, dict):
                    if "dense" in r.vector and r.vector["dense"] is not None:
                        vec_info["dense_dim"] = len(r.vector["dense"])
                    if "sparse" in r.vector and r.vector["sparse"] is not None:
                        sparse_obj = r.vector["sparse"]
                        indices = getattr(sparse_obj, "indices", [])
                        vec_info["sparse_tokens_count"] = len(indices)

                p = r.payload or {}
                results.append({
                    "id": str(r.id),
                    "document_id": p.get("document_id"),
                    "file_name": p.get("file_name"),
                    "document_type": p.get("document_type"),
                    "year": p.get("year"),
                    "quarter": p.get("quarter"),
                    "report_period": p.get("report_period"),
                    "source_page": p.get("source_page"),
                    "node_key": p.get("node_key"),
                    "node_value": p.get("node_value"),
                    "hierarchy_path": p.get("hierarchy_path"),
                    "hierarchy_path_text": p.get("hierarchy_path_text"),
                    "node_type": p.get("node_type"),
                    "vectors": vec_info
                })
            return results
        except Exception as e:
            logger.error(f"[VectorStore] Error retrieving hierarchical points: {e}")
            return []

    def search_children_by_path_prefix(
        self,
        path_prefix: List[str],
        document_id: Optional[str] = None,
        n_results: int = 8,
    ) -> List[Dict[str, Any]]:
        """Fetches hierarchy nodes whose hierarchy_path starts with path_prefix.

        Used for generic branch node expansion: when a branch node is retrieved as
        relevant evidence, this method finds its direct children and deeper descendants
        so the generator can answer questions about the branch's contents.

        Works with both server and embedded Qdrant by fetching all points for the
        document and filtering client-side on path prefix equality.
        No document-specific logic — works for any hierarchy shape.

        Args:
            path_prefix: The hierarchy_path list of the branch node.
            document_id: Optional document scope to limit the scan.
            n_results: Maximum number of child nodes to return.

        Returns:
            List of child node dicts compatible with the retrieval pipeline format.
        """
        if not path_prefix:
            return []

        prefix_len = len(path_prefix)

        # Build a document-scoped filter when available (reduces scan size)
        doc_filter = None
        if document_id:
            doc_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )

        try:
            all_points, _ = self.client.scroll(
                collection_name=self.hierarchical_collection_name,
                scroll_filter=doc_filter,
                limit=500,   # bounded — collection is typically <2000 nodes
                with_payload=True,
                with_vectors=False,
            )
            if not all_points and doc_filter:
                raw_points, _ = self.client.scroll(
                    collection_name=self.hierarchical_collection_name,
                    limit=500,
                    with_payload=True,
                    with_vectors=False,
                )
                if document_id:
                    all_points = [
                        pt for pt in raw_points
                        if (pt.payload or {}).get("document_id") in (document_id, None)
                        or (pt.payload or {}).get("file_name") == document_id
                    ]
                else:
                    all_points = raw_points

            children: List[Dict[str, Any]] = []
            prefix_list = list(path_prefix)

            for pt in all_points:
                payload = pt.payload or {}
                pt_path = payload.get("hierarchy_path", [])

                # Include only nodes that are STRICTLY DEEPER than the prefix
                # (i.e., their path starts with the prefix and has at least one more element)
                if (
                    len(pt_path) > prefix_len
                    and pt_path[:prefix_len] == prefix_list
                ):
                    children.append({
                        "id": payload.get("id") or str(pt.id),
                        "text": payload.get("hierarchy_path_text", ""),
                        "metadata": payload,
                        "document_id": payload.get("document_id"),
                        "file_name": payload.get("file_name"),
                        "document_type": payload.get("document_type"),
                        "year": payload.get("year"),
                        "quarter": payload.get("quarter"),
                        "report_period": payload.get("report_period"),
                        "node_key": payload.get("node_key", ""),
                        "node_value": payload.get("node_value"),
                        "hierarchy_path": pt_path,
                        "hierarchy_path_text": payload.get("hierarchy_path_text", ""),
                        "source_page": payload.get("source_page"),
                        "node_type": payload.get("node_type", "leaf"),
                        # Synthetic score — will be re-ranked by EvidenceReranker
                        "score": 0.50,
                        "vector_sim": 0.50,
                        "rrf_score": 0.0,
                        "rerank_score": 0.50,
                    })

            # Return direct children first (depth = prefix_len + 1), then deeper
            children.sort(key=lambda c: len(c["hierarchy_path"]))
            return children[:n_results]

        except Exception as e:
            logger.warning(f"[VectorStore] Branch child expansion failed for prefix {path_prefix}: {e}")
            return []

    # =========================================================
    # HIERARCHICAL SEARCH (DENSE, SPARSE, HYBRID RRF)
    # =========================================================

    def _build_hierarchical_filter(
        self,
        document_id: Optional[str] = None,
        file_name: Optional[str] = None,
        document_type: Optional[str] = None,
        year: Optional[Union[int, str]] = None,
        quarter: Optional[Union[int, str]] = None,
        report_period: Optional[str] = None,
        source_page: Optional[Union[int, str]] = None,
        node_type: Optional[str] = None,
    ) -> Optional[models.Filter]:
        """Builds dynamic Qdrant metadata filters for the hierarchical collection."""
        conditions = []
        if document_id:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                )
            )
        if file_name:
            conditions.append(
                models.FieldCondition(
                    key="file_name",
                    match=models.MatchValue(value=file_name)
                )
            )
        if document_type and document_type.lower() != "all":
            conditions.append(
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value=document_type.lower())
                )
            )
        if year is not None:
            try:
                year_int = int(year)
                conditions.append(
                    models.FieldCondition(
                        key="year",
                        match=models.MatchValue(value=year_int)
                    )
                )
            except (ValueError, TypeError):
                pass
        if quarter is not None:
            conditions.append(
                models.FieldCondition(
                    key="quarter",
                    match=models.MatchValue(value=quarter)
                )
            )
        if report_period:
            conditions.append(
                models.FieldCondition(
                    key="report_period",
                    match=models.MatchValue(value=report_period)
                )
            )
        if source_page is not None:
            try:
                page_int = int(source_page)
                conditions.append(
                    models.FieldCondition(
                        key="source_page",
                        match=models.MatchValue(value=page_int)
                    )
                )
            except (ValueError, TypeError):
                pass
        if node_type:
            conditions.append(
                models.FieldCondition(
                    key="node_type",
                    match=models.MatchValue(value=node_type)
                )
            )
        return models.Filter(must=conditions) if conditions else None

    def search_hierarchical_dense(
        self,
        query: str,
        document_id: Optional[str] = None,
        file_name: Optional[str] = None,
        document_type: Optional[str] = None,
        year: Optional[Union[int, str]] = None,
        quarter: Optional[Union[int, str]] = None,
        report_period: Optional[str] = None,
        source_page: Optional[Union[int, str]] = None,
        node_type: Optional[str] = None,
        n_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Dense semantic search across hierarchical collection using paraphrase-multilingual-MiniLM-L12-v2."""
        if not query:
            return []

        query_vector = self.model.encode(query).tolist()
        query_filter = self._build_hierarchical_filter(
            document_id=document_id,
            file_name=file_name,
            document_type=document_type,
            year=year,
            quarter=quarter,
            report_period=report_period,
            source_page=source_page,
            node_type=node_type
        )

        candidates: List[Dict[str, Any]] = []
        seen_ids = set()

        try:
            response = self.client.query_points(
                collection_name=self.hierarchical_collection_name,
                query=query_vector,
                using="dense",
                query_filter=query_filter,
                limit=n_results * 2,
                with_payload=True,
                with_vectors=False
            )
            for pt in response.points:
                payload = pt.payload or {}
                raw_id = payload.get("id") or str(pt.id)
                if raw_id not in seen_ids:
                    seen_ids.add(raw_id)
                    score = float(pt.score) if pt.score is not None else 0.0
                    distance = max(0.0, 1.0 - score)
                    candidates.append({
                        "id": raw_id,
                        "text": payload.get("hierarchy_path_text") or payload.get("text", ""),
                        "metadata": payload,
                        "distance": distance,
                        "score": score,
                        "vector_sim": score,
                        "dense_score": score,
                        # Direct payload attributes for reranker and evidence verification
                        "document_id": payload.get("document_id"),
                        "file_name": payload.get("file_name"),
                        "document_type": payload.get("document_type"),
                        "year": payload.get("year"),
                        "quarter": payload.get("quarter"),
                        "report_period": payload.get("report_period"),
                        "node_key": payload.get("node_key", ""),
                        "node_value": payload.get("node_value"),
                        "hierarchy_path": payload.get("hierarchy_path", []),
                        "hierarchy_path_text": payload.get("hierarchy_path_text", ""),
                        "source_page": payload.get("source_page"),
                        "node_type": payload.get("node_type", "leaf")
                    })
        except Exception as e:
            logger.warning(f"[VectorStore] Dense hierarchical search failed: {e}")

        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        for rank, cand in enumerate(candidates):
            cand["dense_rank"] = rank
        return candidates[:n_results]

    def search_hierarchical_sparse(
        self,
        query: str,
        query_terms: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        file_name: Optional[str] = None,
        document_type: Optional[str] = None,
        year: Optional[Union[int, str]] = None,
        quarter: Optional[Union[int, str]] = None,
        report_period: Optional[str] = None,
        source_page: Optional[Union[int, str]] = None,
        node_type: Optional[str] = None,
        n_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Sparse lexical retrieval across hierarchical collection using deterministic token hashes."""
        combined_terms = [query] if query else []
        if query_terms:
            combined_terms.extend([t for t in query_terms if t not in combined_terms])
        search_text = " ".join(combined_terms)
        q_sparse = self.compute_sparse_vector(search_text)
        query_filter = self._build_hierarchical_filter(
            document_id=document_id,
            file_name=file_name,
            document_type=document_type,
            year=year,
            quarter=quarter,
            report_period=report_period,
            source_page=source_page,
            node_type=node_type
        )

        sparse_hits: List[Dict[str, Any]] = []
        seen_ids = set()

        if q_sparse.indices:
            try:
                res = self.client.query_points(
                    collection_name=self.hierarchical_collection_name,
                    query=q_sparse,
                    using="sparse",
                    query_filter=query_filter,
                    limit=n_results * 2,
                    with_payload=True,
                    with_vectors=False
                )
                for pt in res.points:
                    payload = pt.payload or {}
                    raw_id = payload.get("id") or str(pt.id)
                    if raw_id not in seen_ids:
                        seen_ids.add(raw_id)
                        score = float(pt.score) if pt.score is not None else 0.0
                        sparse_hits.append({
                            "id": raw_id,
                            "text": payload.get("hierarchy_path_text") or payload.get("text", ""),
                            "metadata": payload,
                            "sparse_score": score,
                            "bm25_score": score,
                            "score": score,
                            # Direct payload attributes for reranker and evidence verification
                            "document_id": payload.get("document_id"),
                            "file_name": payload.get("file_name"),
                            "document_type": payload.get("document_type"),
                            "year": payload.get("year"),
                            "quarter": payload.get("quarter"),
                            "report_period": payload.get("report_period"),
                            "node_key": payload.get("node_key", ""),
                            "node_value": payload.get("node_value"),
                            "hierarchy_path": payload.get("hierarchy_path", []),
                            "hierarchy_path_text": payload.get("hierarchy_path_text", ""),
                            "source_page": payload.get("source_page"),
                            "node_type": payload.get("node_type", "leaf")
                        })
            except Exception as e:
                logger.warning(f"[VectorStore] Sparse hierarchical search failed: {e}")

        sparse_hits.sort(key=lambda x: x.get("sparse_score", 0.0), reverse=True)
        for rank, item in enumerate(sparse_hits[:n_results]):
            item["sparse_rank"] = rank
        return sparse_hits[:n_results]

    def search_hierarchical_hybrid(
        self,
        query: str,
        query_terms: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        file_name: Optional[str] = None,
        document_type: Optional[str] = None,
        year: Optional[Union[int, str]] = None,
        quarter: Optional[Union[int, str]] = None,
        report_period: Optional[str] = None,
        source_page: Optional[Union[int, str]] = None,
        node_type: Optional[str] = None,
        n_results: int = 20,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """Unified Hierarchical Search: Dense + Sparse + RRF fusion on rajbhasha_hierarchical_collection."""
        dense_results = self.search_hierarchical_dense(
            query=query,
            document_id=document_id,
            file_name=file_name,
            document_type=document_type,
            year=year,
            quarter=quarter,
            report_period=report_period,
            source_page=source_page,
            node_type=node_type,
            n_results=n_results * 2
        )

        sparse_results = self.search_hierarchical_sparse(
            query=query,
            query_terms=query_terms,
            document_id=document_id,
            file_name=file_name,
            document_type=document_type,
            year=year,
            quarter=quarter,
            report_period=report_period,
            source_page=source_page,
            node_type=node_type,
            n_results=n_results * 2
        )

        fused = self.fuse_rrf(dense_results, sparse_results, k=rrf_k)
        return fused[:n_results]

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
                    using="dense",
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

        # Sort candidates by dense similarity (pt.score descending)
        if candidates:
            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            for rank, cand in enumerate(candidates):
                cand["dense_rank"] = rank
                cand["dense_score"] = cand.get("score", 0.0)
                cand["vector_sim"] = cand.get("score", 0.0)
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
                            "score": 1.0,
                            "dense_rank": 0,
                            "dense_score": 1.0,
                            "vector_sim": 1.0
                        })
            except Exception as e:
                logger.warning(f"[VectorStore] Keyword fallback failed on {coll_name}: {e}")

        fallback_results.sort(key=lambda x: len(x["text"]))
        return fallback_results[:n_results]

    # =========================================================
    # SPARSE / BM25-STYLE RETRIEVAL
    # =========================================================

    def _extract_search_terms(self, text: str) -> List[str]:
        """Extract significant words, clauses (e.g., 3(3)), numbers, and phrases."""
        if not text:
            return []
        clauses = re.findall(r'\b\d+\(\d+\)', text)
        numbers = re.findall(r'\b\d+\b', text)
        words = re.findall(r'[\w\u0900-\u097F]+', text.lower())
        terms = [w for w in words if len(w) > 1 or w.isdigit()]
        combined = list(dict.fromkeys(clauses + numbers + terms))
        return combined

    def search_sparse_bm25(
        self,
        query: str,
        query_terms: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        year: Optional[int] = None,
        content_type: str = "all",
        collection_type: str = "all",
        n_results: int = 20,
        k1: float = 1.5,
        b: float = 0.75
    ) -> List[Dict[str, Any]]:
        """
        BM25-style sparse retrieval supporting exact words, names, numbers,
        article titles, clauses (e.g. 3(3)), and terminology via Qdrant payload search.
        """
        if not query and not query_terms:
            return []

        # Extract search terms
        extracted = self._extract_search_terms(query)
        if query_terms:
            for qt in query_terms:
                if qt:
                    extracted.extend(self._extract_search_terms(qt))
        search_terms = list(dict.fromkeys([t.lower() for t in extracted if t]))
        if not search_terms:
            return []

        # Build dynamic metadata filter
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

        # Resolve target collections
        effective_coll = collection_type if collection_type != "all" else content_type
        if effective_coll == "index":
            collections_to_query = [self.index_collection_name]
        elif effective_coll == "content":
            collections_to_query = [self.content_collection_name]
        elif effective_coll == "table":
            collections_to_query = [self.table_collection_name]
        else:
            collections_to_query = [
                self.index_collection_name,
                self.content_collection_name,
                self.table_collection_name
            ]

        # 1. Native Qdrant sparse vector search using named sparse vector index
        q_sparse = self.compute_sparse_vector(query or " ".join(search_terms))
        sparse_hits: List[Dict[str, Any]] = []
        seen_ids = set()

        if q_sparse.indices:
            for coll_name in collections_to_query:
                try:
                    res = self.client.query_points(
                        collection_name=coll_name,
                        query=q_sparse,
                        using="sparse",
                        query_filter=query_filter,
                        limit=n_results * 2,
                        with_payload=True,
                        with_vectors=False
                    )
                    for pt in res.points:
                        payload = pt.payload or {}
                        raw_id = payload.get("id") or payload.get("chunk_id") or str(pt.id)
                        if raw_id not in seen_ids:
                            seen_ids.add(raw_id)
                            score = float(pt.score) if pt.score is not None else 0.0
                            sparse_hits.append({
                                "id": raw_id,
                                "text": payload.get("text", ""),
                                "metadata": payload,
                                "sparse_score": score,
                                "bm25_score": score,
                                "score": score
                            })
                except Exception as e:
                    logger.debug(f"[VectorStore] Native sparse query fallback on {coll_name}: {e}")

        if sparse_hits:
            sparse_hits.sort(key=lambda x: x.get("sparse_score", 0.0), reverse=True)
            for rank, item in enumerate(sparse_hits[:n_results]):
                item["sparse_rank"] = rank
            return sparse_hits[:n_results]

        # 2. Fallback: Retrieve candidate points from Qdrant with filters
        candidate_pool: List[Dict[str, Any]] = []
        seen_ids = set()

        for coll_name in collections_to_query:
            try:
                scroll_res, _ = self.client.scroll(
                    collection_name=coll_name,
                    scroll_filter=query_filter,
                    limit=300,
                    with_payload=True
                )
                for pt in scroll_res:
                    payload = pt.payload or {}
                    raw_id = payload.get("id") or payload.get("chunk_id") or str(pt.id)
                    if raw_id not in seen_ids:
                        seen_ids.add(raw_id)
                        candidate_pool.append({
                            "id": raw_id,
                            "text": payload.get("text", ""),
                            "metadata": payload
                        })
            except Exception as e:
                logger.warning(f"[VectorStore] Sparse candidate fetch failed on {coll_name}: {e}")

        if not candidate_pool:
            return []

        # ==========================================
        # BM25 Mathematical Scoring
        # ==========================================
        N = len(candidate_pool)
        doc_tokens_list: List[List[str]] = []
        df: Dict[str, int] = {}

        for item in candidate_pool:
            text = item.get("text", "")
            tokens = re.findall(r'[\w\u0900-\u097F]+', text.lower())
            doc_tokens_list.append(tokens)
            token_set = set(tokens)
            for term in search_terms:
                if term in token_set:
                    df[term] = df.get(term, 0) + 1

        total_tokens = sum(len(toks) for toks in doc_tokens_list)
        avgdl = total_tokens / max(1, N)

        # Compute IDF for each term: ln(1.0 + (N - df + 0.5) / (df + 0.5))
        idf: Dict[str, float] = {}
        for term in search_terms:
            n_t = df.get(term, 0)
            idf[term] = math.log(1.0 + (N - n_t + 0.5) / (n_t + 0.5))

        # Check numeric page or year terms
        num_terms = [int(t) for t in search_terms if t.isdigit() and len(t) <= 4]
        query_phrase = query.lower().strip() if query else ""

        scored_candidates = []
        for idx, item in enumerate(candidate_pool):
            tokens = doc_tokens_list[idx]
            dl = len(tokens)
            meta = item.get("metadata", {})
            text_lower = item.get("text", "").lower()

            counts: Dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1

            # Exact author or article title token boost
            author_text = str(meta.get("author", "")).lower()
            title_text = str(meta.get("article_title", "")).lower()

            score = 0.0
            matched_terms_count = 0

            for term in search_terms:
                tf = counts.get(term, 0)
                # Boost term frequency if matched directly in author or article title metadata
                if term in author_text or term in title_text:
                    tf += 3.0

                if tf > 0:
                    matched_terms_count += 1
                    denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                    score += idf[term] * ((tf * (k1 + 1.0)) / denom)

            # Exact phrase match boost if multi-word phrase appears in chunk
            if len(search_terms) > 1 and query_phrase and query_phrase in text_lower:
                score += 3.0

            # Exact page match boost
            chunk_page = meta.get("page") or meta.get("page_number")
            if chunk_page is not None:
                try:
                    if int(chunk_page) in num_terms:
                        score += 3.5
                except (ValueError, TypeError):
                    pass

            # Exact year match boost
            chunk_year = meta.get("year")
            if chunk_year is not None:
                try:
                    if int(chunk_year) in num_terms:
                        score += 2.0
                except (ValueError, TypeError):
                    pass

            if score > 0:
                item_copy = dict(item)
                item_copy["bm25_score"] = round(score, 4)
                item_copy["score"] = round(score, 4)
                item_copy["distance"] = 0.0
                scored_candidates.append(item_copy)

        # Sort descending by BM25 score
        scored_candidates.sort(key=lambda x: x["bm25_score"], reverse=True)
        for rank, cand in enumerate(scored_candidates):
            cand["sparse_rank"] = rank
            cand["sparse_score"] = cand["bm25_score"]

        return scored_candidates[:n_results]

    # =========================================================
    # RECIPROCAL RANK FUSION (RRF)
    # =========================================================

    @staticmethod
    def fuse_rrf(
        dense_candidates: List[Dict[str, Any]],
        sparse_candidates: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) strictly combines ranked lists from dense
        and sparse retrievers without comparing disparate raw score scales.
        Formula: RRF(d) = sum( 1.0 / (k + rank + 1) )
        """
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # 1. Ingest Dense Candidates
        for rank, cand in enumerate(dense_candidates):
            cid = str(cand.get("id") or cand.get("chunk_id"))
            cand_copy = dict(cand)
            cand_copy["dense_rank"] = rank
            cand_copy["dense_score"] = cand.get("score", cand.get("vector_sim", 0.0))
            cand_copy["vector_sim"] = cand_copy["dense_score"]
            chunk_map[cid] = cand_copy
            scores[cid] = 1.0 / (k + rank + 1)

        # 2. Ingest Sparse Candidates
        for rank, cand in enumerate(sparse_candidates):
            cid = str(cand.get("id") or cand.get("chunk_id"))
            if cid not in chunk_map:
                cand_copy = dict(cand)
                cand_copy["dense_rank"] = None
                cand_copy["dense_score"] = 0.0
                cand_copy["vector_sim"] = 0.0
                chunk_map[cid] = cand_copy

            chunk_map[cid]["sparse_rank"] = rank
            chunk_map[cid]["sparse_score"] = cand.get("score", cand.get("bm25_score", 0.0))
            chunk_map[cid]["lexical_score"] = chunk_map[cid]["sparse_score"]
            chunk_map[cid]["bm25_score"] = chunk_map[cid]["sparse_score"]
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank + 1))

        # 3. Assign RRF Scores
        for cid, chunk in chunk_map.items():
            chunk["rrf_score"] = round(scores[cid], 6)

        # 4. Sort descending by RRF score
        sorted_chunks = sorted(chunk_map.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_chunks

    # =========================================================
    # HYBRID SEARCH (DENSE + SPARSE/BM25 + RRF)
    # =========================================================

    def search_hybrid(
        self,
        query: str,
        query_terms: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        year: Optional[int] = None,
        content_type: str = "all",
        collection_type: str = "all",
        n_results: int = 20,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Executes both dense semantic search and sparse BM25 search in Qdrant,
        then fuses the results using Reciprocal Rank Fusion (RRF).
        """
        dense_results = self.search_by_vector(
            query=query,
            document_id=document_id,
            document_type=document_type,
            year=year,
            content_type=content_type,
            collection_type=collection_type,
            n_results=n_results * 2
        )

        sparse_results = self.search_sparse_bm25(
            query=query,
            query_terms=query_terms,
            document_id=document_id,
            document_type=document_type,
            year=year,
            content_type=content_type,
            collection_type=collection_type,
            n_results=n_results * 2
        )

        fused = self.fuse_rrf(dense_results, sparse_results, k=rrf_k)
        return fused[:n_results]

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

        for coll_name in [self.index_collection_name, self.content_collection_name, self.table_collection_name, self.hierarchical_collection_name]:
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
        for coll_name in [self.content_collection_name, self.index_collection_name, self.table_collection_name, self.hierarchical_collection_name]:
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
        """Return statistics across all Rajbhasha collections."""
        return {
            "index": self.get_collection_stats(self.index_collection_name),
            "content": self.get_collection_stats(self.content_collection_name),
            "table": self.get_collection_stats(self.table_collection_name),
            "hierarchical": self.get_collection_stats(self.hierarchical_collection_name)
        }
