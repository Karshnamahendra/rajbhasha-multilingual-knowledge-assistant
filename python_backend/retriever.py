"""Production hybrid retriever fusing dense Qdrant vector search and sparse BM25 retrieval via RRF."""
import math
import re
import logging
from typing import List, Dict, Any, Optional, Set

from vector_store import VectorStore
from query_analyzer import SemanticQueryAnalyzer
from reranker import ReciprocalRankFusion
from config import settings

logger = logging.getLogger("HybridRetriever")
logger.setLevel(logging.INFO)


class HybridRetriever:
    """
    Production Hybrid Retriever combining:
    1. Dense semantic retrieval via SentenceTransformer on Qdrant
    2. Sparse / BM25-style retrieval via exact words, names, numbers, clauses, and titles
    3. Reciprocal Rank Fusion (RRF) to merge ranked lists without score scale conflicts
    """

    def __init__(self, vector_store: VectorStore, query_analyzer: SemanticQueryAnalyzer):
        self.vector_store = vector_store
        self.analyzer = query_analyzer
        self.rrf_k = getattr(settings, "RRF_K", 60)

    def _lexical_score(self, query_terms: List[str], text: str) -> float:
        """Compute term frequency weighted lexical score for candidate text."""
        if not text or not query_terms:
            return 0.0
        text_lower = text.lower()
        score = 0.0
        for term in query_terms:
            t_low = term.lower()
            count = text_lower.count(t_low)
            if count > 0:
                score += (1.0 + math.log(count)) * (len(t_low) ** 0.5)
        return round(score, 4)

    def _extract_sparse_terms(
        self,
        query: str,
        analysis: Dict[str, Any],
        expanded_queries: List[str]
    ) -> List[str]:
        """
        Extract exact words, names, numbers, article titles, clauses (e.g., 3(3)),
        and terminology from query, analyzer entities, and transliterations.
        """
        terms: Set[str] = set()

        # 1. Clauses and subsection numbers like 3(3), 8(4), 10.1
        clauses = re.findall(r'\b\d+(?:\(\d+\)|\.\d+)?\b', query)
        for c in clauses:
            terms.add(c)

        # 2. General words and Devanagari tokens
        words = re.findall(r'[\w\u0900-\u097F]+', query.lower())
        for w in words:
            if len(w) > 1 or w.isdigit() or re.match(r'^[\u0900-\u097F]$', w):
                terms.add(w)

        # 3. Target title and resolved entities from analyzer
        target_title = analysis.get("target_title")
        if target_title:
            terms.add(target_title.lower())
            for w in re.findall(r'[\w\u0900-\u097F]+', target_title.lower()):
                if len(w) > 1 or w.isdigit() or re.match(r'^[\u0900-\u097F]$', w):
                    terms.add(w)
            trans = self.analyzer.algorithmic_transliterate(target_title)
            if trans:
                terms.add(trans.lower())

        for ent in analysis.get("entities", []):
            terms.add(ent.lower())
            for w in re.findall(r'[\w\u0900-\u097F]+', ent.lower()):
                if len(w) > 1 or w.isdigit() or re.match(r'^[\u0900-\u097F]$', w):
                    terms.add(w)

        # 4. Expanded queries
        for q_var in expanded_queries:
            for w in re.findall(r'[\w\u0900-\u097F]+', q_var.lower()):
                if len(w) > 1 or w.isdigit() or re.match(r'^[\u0900-\u097F]$', w):
                    terms.add(w)

        return list(terms)

    def _locate_target_pages_from_index(
        self,
        target_title: Optional[str],
        entities: List[str],
        document_id: Optional[str]
    ) -> Set[int]:
        """
        Scans index_collection to dynamically discover page numbers or section coordinates
        for a specific requested article, poem, or story without hardcoding.
        """
        target_pages: Set[int] = set()
        if not target_title and not entities:
            return target_pages

        search_terms = []
        if target_title:
            search_terms.append(target_title)
            trans = self.analyzer.algorithmic_transliterate(target_title)
            if trans:
                search_terms.append(trans)
        for ent in entities:
            search_terms.append(ent)

        # Retrieve index chunks for the document
        idx_chunks = self.vector_store.get_all_document_chunks(document_id, collection_type="index")
        if not idx_chunks:
            # Fallback search by vector in index collection
            for st in search_terms[:2]:
                res = self.vector_store.search_by_vector(
                    st,
                    document_id=document_id,
                    collection_type="index",
                    n_results=5
                )
                idx_chunks.extend(res)

        for chunk in idx_chunks:
            text = chunk.get("text", "")
            lines = text.splitlines()
            for line in lines:
                line_lower = line.lower()
                for term in search_terms:
                    t_low = term.lower()
                    if t_low in line_lower:
                        # Extract trailing page numbers or digits associated with this listing
                        # e.g., "Semiconductor ... 12" or "नई सुबह - पृष्ठ 15"
                        found_nums = re.findall(r'\b(\d{1,3})\b', line)
                        for num_str in found_nums:
                            try:
                                p_val = int(num_str)
                                if 1 <= p_val <= 300:
                                    target_pages.add(p_val)
                                    target_pages.add(p_val + 1)
                            except ValueError:
                                pass

        if target_pages:
            logger.info(
                f"[HybridRetriever] Localized '{target_title or entities}' to target page(s) "
                f"{sorted(list(target_pages))} via Index Collection."
            )
        return target_pages

    # =========================================================================
    # CORE HYBRID RETRIEVAL (DENSE + SPARSE/BM25 + RRF)
    # =========================================================================

    def retrieve_candidates(
        self,
        query: str,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        year: Optional[int] = None,
        query_mode: str = "AUTO",
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Executes unified Hybrid Retrieval:
        1. Dense semantic search across query variations
        2. Sparse BM25 exact keyword search for names, numbers, clauses, and titles
        3. Reciprocal Rank Fusion (RRF) combining dense and sparse results
        4. Preserves all metadata (document_id, document_type, content_type, page, year, source, chunk_id)
        """
        # 1. Analyze query, detect language/script, and generate normalized variants
        analysis = self.analyzer.analyze(query, requested_mode=query_mode, document_id=document_id)
        resolved_mode = analysis.get("query_mode", "CONTENT_BASED")
        expanded_queries = self.analyzer.expand_query(analysis)

        # Filters
        filter_doc_type = document_type or analysis.get("target_document_type")
        filter_year = year if year is not None else analysis.get("target_year")

        logger.info(
            f"[HybridRetriever] Query: '{query}' | Script: {analysis.get('detected_script')} | "
            f"Mode: {resolved_mode} | DocumentID: {document_id} | DocType: {filter_doc_type} | Year: {filter_year}"
        )

        # Extract comprehensive search terms for Sparse BM25 retrieval
        sparse_terms = self._extract_sparse_terms(query, analysis, expanded_queries)

        dense_candidates: List[Dict[str, Any]] = []
        sparse_candidates: List[Dict[str, Any]] = []

        # =========================================================================
        # BRANCH 1: INDEX_BASED (TOC, Article Counts, Author Mappings)
        # =========================================================================
        if resolved_mode == "INDEX_BASED":
            intent = analysis.get("intent", "semantic_qa")
            target_title = analysis.get("target_title")
            entities = analysis.get("entities", [])

            # A. Full scan of index collection for structural / list queries
            full_scan_chunks: List[Dict[str, Any]] = []
            if intent in ("find_author", "count", "list", "structure_query") or analysis.get("is_count_or_list"):
                all_idx = self.vector_store.get_all_document_chunks(document_id, collection_type="index")
                for chunk in all_idx:
                    item_copy = dict(chunk)
                    item_copy["vector_sim"] = 0.70
                    item_copy["distance"] = 0.30
                    item_copy["score"] = 0.70
                    full_scan_chunks.append(item_copy)

            # B. Dense search across expanded queries in index_collection
            dense_map: Dict[str, Dict[str, Any]] = {}
            for q_var in expanded_queries:
                v_res = self.vector_store.search_by_vector(
                    q_var,
                    document_id=document_id,
                    document_type=filter_doc_type,
                    year=filter_year,
                    collection_type="index",
                    n_results=top_k
                )
                for item in v_res:
                    cid = item["id"]
                    score = item.get("score", 0.0)
                    if cid not in dense_map or score > dense_map[cid].get("score", 0.0):
                        dense_map[cid] = item

            # Include full-scan chunks if not already present
            for chunk in full_scan_chunks:
                cid = chunk["id"]
                if cid not in dense_map:
                    dense_map[cid] = chunk

            # Order dense candidates
            dense_candidates = sorted(dense_map.values(), key=lambda x: x.get("score", 0.0), reverse=True)
            for rank, cand in enumerate(dense_candidates):
                cand["dense_rank"] = rank
                cand["dense_score"] = cand.get("score", 0.0)
                cand["vector_sim"] = cand.get("score", 0.0)

            # C. Sparse BM25 search in index_collection
            sparse_candidates = self.vector_store.search_sparse_bm25(
                query=query,
                query_terms=sparse_terms,
                document_id=document_id,
                document_type=filter_doc_type,
                year=filter_year,
                collection_type="index",
                n_results=top_k
            )

            # Fallback: if index results are few, also search early content pages (1-8)
            if len(dense_candidates) + len(sparse_candidates) < 5:
                early_cnt = self.vector_store.search_by_vector(
                    query,
                    document_id=document_id,
                    document_type=filter_doc_type,
                    year=filter_year,
                    collection_type="content",
                    n_results=10
                )
                for item in early_cnt:
                    page = item.get("metadata", {}).get("page_number") or item.get("metadata", {}).get("page", 999)
                    if page <= 8 and item["id"] not in [c["id"] for c in dense_candidates]:
                        dense_candidates.append(item)

        # =========================================================================
        # BRANCH 2: CONTENT_BASED (Deep Content, Articles, Stories, Reports, Tables)
        # =========================================================================
        else:
            target_title = analysis.get("target_title")
            entities = analysis.get("entities", [])

            # Step 2a: Use Index Collection to dynamically discover page numbers
            target_pages = self._locate_target_pages_from_index(target_title, entities, document_id)

            # Determine target collection: include rajbhasha_table_collection for table/report queries
            is_table_query = (
                resolved_mode in ("TABLE_BASED", "FORM_BASED", "REPORT_METADATA") or
                analysis.get("table_intent") or
                filter_doc_type == "report" or
                any(k in query.lower() for k in [
                    "table", "तालिका", "ब्यौरा", "विवरण", "संख्या", "प्रतिशत", "quarter",
                    "report", "तिमाही", "धारा 3", "3(3)", "लक्ष्य", "उपलब्धि", "पत्र", "फाइल",
                    "बैठक", "कार्यशाला", "मंत्रालय", "समिति"
                ])
            )
            coll_type = "table" if resolved_mode == "TABLE_BASED" else ("all" if is_table_query else "content")

            # Step 2b: Dense search across all query variants in target collection(s)
            dense_map: Dict[str, Dict[str, Any]] = {}
            for q_var in expanded_queries:
                v_res = self.vector_store.search_by_vector(
                    q_var,
                    document_id=document_id,
                    document_type=filter_doc_type,
                    year=filter_year,
                    collection_type=coll_type,
                    n_results=top_k
                )
                for item in v_res:
                    cid = item["id"]
                    score = item.get("score", 0.0)
                    # Boost dense similarity if chunk matches localized page from TOC
                    item_page = item.get("metadata", {}).get("page_number") or item.get("metadata", {}).get("page", 0)
                    if target_pages and item_page in target_pages:
                        score = min(1.0, score + 0.20)
                        item["score"] = score

                    if cid not in dense_map or score > dense_map[cid].get("score", 0.0):
                        dense_map[cid] = item

            # If specific target pages were discovered, pull chunks directly from those pages
            if target_pages:
                doc_content = self.vector_store.get_all_document_chunks(document_id, collection_type="content")
                for chunk in doc_content:
                    c_page = chunk.get("metadata", {}).get("page_number") or chunk.get("metadata", {}).get("page", 0)
                    if c_page in target_pages:
                        cid = chunk["id"]
                        if cid not in dense_map:
                            item_copy = dict(chunk)
                            item_copy["score"] = 0.85
                            item_copy["vector_sim"] = 0.85
                            item_copy["distance"] = 0.15
                            dense_map[cid] = item_copy

            dense_candidates = sorted(dense_map.values(), key=lambda x: x.get("score", 0.0), reverse=True)
            for rank, cand in enumerate(dense_candidates):
                cand["dense_rank"] = rank
                cand["dense_score"] = cand.get("score", 0.0)
                cand["vector_sim"] = cand.get("score", 0.0)

            # Step 2c: Sparse BM25 search across target collection(s) (including table collection)
            sparse_candidates = self.vector_store.search_sparse_bm25(
                query=query,
                query_terms=sparse_terms,
                document_id=document_id,
                document_type=filter_doc_type,
                year=filter_year,
                collection_type=coll_type,
                n_results=top_k
            )

        # =========================================================================
        # 3. RECIPROCAL RANK FUSION (RRF) COMBINATION
        # =========================================================================
        fused_candidates = self.vector_store.fuse_rrf(
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            k=self.rrf_k
        )

        # 4. Attach lexical scores and verify metadata integrity
        for cand in fused_candidates:
            # Ensure lexical_score is populated for downstream reranker
            sparse_score = cand.get("sparse_score") or cand.get("bm25_score")
            if sparse_score is not None:
                cand["lexical_score"] = sparse_score
            else:
                cand["lexical_score"] = self._lexical_score(sparse_terms, cand.get("text", ""))

            # Ensure vector_sim is populated
            if "vector_sim" not in cand:
                cand["vector_sim"] = cand.get("dense_score", 0.0)

        logger.info(
            f"[HybridRetriever] Retrieval complete: Dense={len(dense_candidates)}, "
            f"Sparse={len(sparse_candidates)} -> Fused RRF={len(fused_candidates)}"
        )

        return fused_candidates

    # =========================================================================
    # HIERARCHICAL UNIFIED RETRIEVAL (DENSE + SPARSE + RRF)
    # =========================================================================

    def retrieve_hierarchical_candidates(
        self,
        query: str,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        year: Optional[int] = None,
        quarter: Optional[str] = None,
        report_period: Optional[str] = None,
        source_page: Optional[int] = None,
        top_k: int = 20,
        analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Primary Unified Knowledge Retrieval across rajbhasha_hierarchical_collection.
        1. Analyzes query and extracts normalized/expanded variations
        2. Executes dense vector search across query variations
        3. Executes sparse lexical search across extracted terms
        4. Fuses ranked candidates with Reciprocal Rank Fusion (RRF)
        5. Preserves all hierarchical payload fields for downstream EvidenceReranker and Generator
        """
        if analysis is None:
            analysis = self.analyzer.analyze(query, requested_mode="AUTO", document_id=document_id)

        expanded_queries = self.analyzer.expand_query(analysis)
        filter_doc_type = document_type or analysis.get("target_document_type")
        filter_year = year if year is not None else analysis.get("target_year")

        sparse_terms = self._extract_sparse_terms(query, analysis, expanded_queries)

        logger.info(
            f"[HybridRetriever:Hierarchical] Query='{query}' | FilterDocID={document_id} | "
            f"FilterDocType={filter_doc_type} | FilterYear={filter_year} | TopK={top_k}"
        )

        # 1. Multi-query Dense Search across expanded variations
        dense_map: Dict[str, Dict[str, Any]] = {}
        primary_queries = [query]
        for v in (analysis.get("normalized_variants") or []):
            if v and v not in primary_queries:
                primary_queries.append(v)

        for q_var in expanded_queries:
            is_auxiliary = (q_var not in primary_queries and len(str(q_var).split()) <= 3)
            scale = 0.70 if is_auxiliary else 1.0
            dense_hits = self.vector_store.search_hierarchical_dense(
                query=q_var,
                document_id=document_id,
                document_type=filter_doc_type,
                year=filter_year,
                quarter=quarter,
                report_period=report_period,
                source_page=source_page,
                n_results=top_k * 2
            )
            for hit in dense_hits:
                cid = hit["id"]
                score = (hit.get("score", 0.0) or 0.0) * scale
                if cid not in dense_map or score > dense_map[cid].get("score", 0.0):
                    hit_copy = dict(hit)
                    hit_copy["score"] = score
                    dense_map[cid] = hit_copy

        dense_candidates = sorted(dense_map.values(), key=lambda x: x.get("score", 0.0), reverse=True)
        for rank, cand in enumerate(dense_candidates):
            cand["dense_rank"] = rank
            cand["dense_score"] = cand.get("score", 0.0)
            cand["vector_sim"] = cand.get("score", 0.0)

        # 2. Sparse Search
        sparse_candidates = self.vector_store.search_hierarchical_sparse(
            query=query,
            query_terms=sparse_terms,
            document_id=document_id,
            document_type=filter_doc_type,
            year=filter_year,
            quarter=quarter,
            report_period=report_period,
            source_page=source_page,
            n_results=top_k * 2
        )

        # 3. Reciprocal Rank Fusion
        fused_candidates = self.vector_store.fuse_rrf(
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            k=self.rrf_k
        )

        # 4. Attach lexical_score and vector_sim
        for cand in fused_candidates:
            sparse_score = cand.get("sparse_score") or cand.get("bm25_score")
            if sparse_score is not None:
                cand["lexical_score"] = sparse_score
            else:
                cand["lexical_score"] = self._lexical_score(sparse_terms, cand.get("text", ""))

            if "vector_sim" not in cand:
                cand["vector_sim"] = cand.get("dense_score", 0.0)

            # Ensure metadata payload fields are exposed at top-level
            meta = cand.get("metadata", {})
            for field_name in [
                "node_key", "node_value", "hierarchy_path", "hierarchy_path_text",
                "source_page", "node_type", "document_id", "file_name", "document_type",
                "year", "quarter", "report_period"
            ]:
                if field_name not in cand and field_name in meta:
                    cand[field_name] = meta[field_name]

        logger.info(
            f"[HybridRetriever:Hierarchical] Retrieval complete: Dense={len(dense_candidates)}, "
            f"Sparse={len(sparse_candidates)} -> Fused RRF={len(fused_candidates)}"
        )

        return fused_candidates[:top_k]

