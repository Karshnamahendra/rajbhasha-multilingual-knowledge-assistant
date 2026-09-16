import math
import re
import logging
from typing import List, Dict, Any, Optional, Set
from vector_store import VectorStore
from query_analyzer import SemanticQueryAnalyzer
from reranker import ReciprocalRankFusion

logger = logging.getLogger("HybridRetriever")
logger.setLevel(logging.INFO)

class HybridRetriever:
    def __init__(self, vector_store: VectorStore, query_analyzer: SemanticQueryAnalyzer):
        self.vector_store = vector_store
        self.analyzer = query_analyzer

    def _lexical_score(self, query_terms: List[str], text: str) -> float:
        if not text:
            return 0.0
        text_lower = text.lower()
        score = 0.0
        for term in query_terms:
            t_low = term.lower()
            count = text_lower.count(t_low)
            if count > 0:
                score += (1.0 + math.log(count)) * (len(t_low) ** 0.5)
        return score

    def _locate_target_pages_from_index(
        self,
        target_title: str,
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
                res = self.vector_store.search_by_vector(st, document_id=document_id, collection_type="index", n_results=5)
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
                                    # Also include adjacent page for multi-page articles
                                    target_pages.add(p_val + 1)
                            except ValueError:
                                pass

        if target_pages:
            logger.info(f"[HybridRetriever] Localized '{target_title or entities}' to target page(s) {sorted(list(target_pages))} via Index Collection.")
        return target_pages

    def retrieve_candidates(
        self,
        query: str,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        year: Optional[int] = None,
        query_mode: str = "AUTO",
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        # 1. Analyze query, detect language/script, and generate normalized variants
        analysis = self.analyzer.analyze(query, requested_mode=query_mode, document_id=document_id)
        resolved_mode = analysis.get("query_mode", "CONTENT_BASED")
        expanded_queries = self.analyzer.expand_query(analysis)

        # Use query-extracted filters if not explicitly provided
        filter_doc_type = document_type or analysis.get("target_document_type")
        filter_year = year if year is not None else analysis.get("target_year")

        logger.info(f"[HybridRetriever] Query: '{query}' | Script: {analysis.get('detected_script')} | Mode: {resolved_mode} | DocumentID: {document_id} | DocType: {filter_doc_type} | Year: {filter_year}")

        candidate_runs: List[List[Dict[str, Any]]] = []
        candidate_map: Dict[str, Dict[str, Any]] = {}

        # =========================================================================
        # MODE 1: INDEX BASED (Table of Contents, Article Counts, Author Mappings)
        # =========================================================================
        if resolved_mode == "INDEX_BASED":
            intent = analysis.get("intent", "semantic_qa")
            target_title = analysis.get("target_title")
            entities = analysis.get("entities", [])

            # A. Always pull ALL index chunks for find_author / count / list queries
            if intent in ("find_author", "count", "list", "structure_query") or analysis.get("is_count_or_list"):
                all_idx_chunks = self.vector_store.get_all_document_chunks(document_id, collection_type="index")
                for chunk in all_idx_chunks:
                    c_id = chunk["id"]
                    item_copy = dict(chunk)
                    item_copy["vector_sim"] = 0.70  # baseline for full-scan index chunks
                    item_copy["distance"] = 0.30
                    candidate_map[c_id] = item_copy

            # B. Vector search across all query variants in index_collection
            for q_var in expanded_queries:
                vec_res = self.vector_store.search_by_vector(
                    q_var,
                    document_id=document_id,
                    document_type=filter_doc_type,
                    year=filter_year,
                    collection_type="index",
                    n_results=top_k
                )
                if vec_res:
                    candidate_runs.append(vec_res)
                for item in vec_res:
                    c_id = item["id"]
                    sim = max(0.0, 1.0 - (item.get("distance", 0.0) / 2.0))
                    sim = min(1.0, sim + 0.15)
                    if c_id not in candidate_map or sim > candidate_map[c_id].get("vector_sim", 0.0):
                        item_copy = dict(item)
                        item_copy["vector_sim"] = sim
                        candidate_map[c_id] = item_copy

            # C. Fallback: also search content collection pages 1-8
            for q_var in expanded_queries[:2]:
                cnt_res = self.vector_store.search_by_vector(
                    q_var,
                    document_id=document_id,
                    document_type=filter_doc_type,
                    year=filter_year,
                    collection_type="content",
                    n_results=8
                )
                for item in cnt_res:
                    item_page = item.get("metadata", {}).get("page_number", 999)
                    if item_page <= 8:
                        c_id = item["id"]
                        sim = max(0.0, 1.0 - (item.get("distance", 0.0) / 2.0))
                        if c_id not in candidate_map:
                            item_copy = dict(item)
                            item_copy["vector_sim"] = sim
                            candidate_map[c_id] = item_copy

        # =========================================================================
        # MODE 2: CONTENT BASED (Deep Semantic Search, Story/Poem Contents)
        # =========================================================================
        else:  # CONTENT_BASED
            target_title = analysis.get("target_title")
            entities = analysis.get("entities", [])

            # Step 2a: Use Index Collection to localize target section/page numbers
            target_pages = self._locate_target_pages_from_index(target_title, entities, document_id)

            # Step 2b: Search Content Collection across all query variants
            for q_var in expanded_queries:
                vec_res = self.vector_store.search_by_vector(
                    q_var,
                    document_id=document_id,
                    document_type=filter_doc_type,
                    year=filter_year,
                    collection_type="content",
                    n_results=top_k
                )
                if vec_res:
                    candidate_runs.append(vec_res)
                for item in vec_res:
                    c_id = item["id"]
                    sim = max(0.0, 1.0 - (item.get("distance", 0.0) / 2.0))
                    
                    # Boost score if chunk matches localized page from TOC
                    item_page = item.get("metadata", {}).get("page_number", 0)
                    if target_pages and item_page in target_pages:
                        sim = min(1.0, sim + 0.35)

                    if c_id not in candidate_map or sim > candidate_map[c_id].get("vector_sim", 0.0):
                        item_copy = dict(item)
                        item_copy["vector_sim"] = sim
                        candidate_map[c_id] = item_copy

            # Step 2c: If specific target pages were discovered, pull chunks directly from those pages
            if target_pages:
                doc_content_chunks = self.vector_store.get_all_document_chunks(document_id, collection_type="content")
                for chunk in doc_content_chunks:
                    c_page = chunk.get("metadata", {}).get("page_number", 0)
                    if c_page in target_pages:
                        c_id = chunk["id"]
                        if c_id not in candidate_map:
                            item_copy = dict(chunk)
                            item_copy["vector_sim"] = 0.85
                            item_copy["distance"] = 0.15
                            candidate_map[c_id] = item_copy

        # =========================================================================
        # 3. Reciprocal Rank Fusion (RRF) & Lexical Coverage
        # =========================================================================
        if candidate_runs:
            fused_ranked = ReciprocalRankFusion.fuse(candidate_runs, k=60)
            for f_item in fused_ranked:
                c_id = f_item["id"]
                if c_id in candidate_map:
                    candidate_map[c_id]["rrf_score"] = f_item.get("rrf_score", 0.0)

        # Build comprehensive keyword list from all expanded variants
        keywords = set(query.lower().split())
        for q_var in expanded_queries:
            for w in q_var.lower().split():
                if len(w) > 1:
                    keywords.add(w)
        for ent in analysis.get("entities", []):
            keywords.add(ent.lower())
        if analysis.get("target_title"):
            keywords.add(analysis["target_title"].lower())

        candidates = list(candidate_map.values())
        for cand in candidates:
            cand["lexical_score"] = self._lexical_score(list(keywords), cand.get("text", ""))

        return candidates
