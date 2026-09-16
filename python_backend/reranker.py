import re
from typing import List, Dict, Any, Optional

class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) combines candidate lists from multiple query variants.
    """
    @staticmethod
    def fuse(
        candidate_lists: List[List[Dict[str, Any]]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for candidates in candidate_lists:
            for rank, chunk in enumerate(candidates):
                c_id = chunk["id"]
                if c_id not in chunk_map:
                    chunk_map[c_id] = chunk
                    scores[c_id] = 0.0
                scores[c_id] += 1.0 / (k + rank + 1)

        # Attach RRF score to chunks
        for c_id, chunk in chunk_map.items():
            chunk["rrf_score"] = scores[c_id]

        sorted_chunks = sorted(chunk_map.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_chunks

class EvidenceReranker:
    def __init__(self):
        pass

    def rerank(
        self,
        query_analysis: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        entities = query_analysis.get("entities", [])
        intent = query_analysis.get("intent", "semantic_qa")
        is_count_list = query_analysis.get("is_count_or_list", False)
        orig_query = query_analysis.get("original_query", "")

        query_tokens = set(re.findall(r"[\w\u0900-\u097F]+", orig_query.lower()))
        stopwords = {
            "what", "is", "written", "in", "the", "poem", "article", "story", "about",
            "kya", "hai", "me", "mein", "batao", "explain", "describe", "ka", "ke", "ki",
            "se", "par", "ko", "and", "or", "to", "of", "a", "an"
        }
        significant_tokens = {t for t in query_tokens if t not in stopwords and len(t) > 1}

        scored = []
        for cand in candidates:
            text = cand["text"]
            text_lower = text.lower()
            meta = cand.get("metadata", {})
            score = 0.0

            # 1. Similarity and RRF baselines
            vec_sim = cand.get("vector_sim", 0.0)
            score += vec_sim * 0.25

            rrf_score = cand.get("rrf_score", 0.0)
            score += min(0.20, rrf_score * 3.0)

            # 2. Lexical and significant token overlap
            lex_norm = min(1.0, cand.get("lexical_score", 0.0) / 10.0)
            score += lex_norm * 0.15

            token_matches = sum(1 for t in significant_tokens if t in text_lower)
            if significant_tokens:
                score += (token_matches / len(significant_tokens)) * 0.20

            # 3. Entity match across English & Devanagari
            for ent in entities:
                ent_lower = ent.lower()
                if ent_lower in text_lower:
                    score += 0.35

            # 4. Target title matching
            target_title = query_analysis.get("target_title")
            if target_title and target_title.lower() in text_lower:
                score += 0.50

            # 4b. Exact title-in-line scoring: find a line that contains the exact title
            #     This handles TOC dot-leader entries like:
            #     "Semiconductor Ecosystem ..... Saurabh Behera ..... 14"
            if target_title:
                for line in text.splitlines():
                    if target_title.lower() in line.lower():
                        score += 0.80  # very strong boost for the exact TOC line
                        break

            # 5. Strong Boost for Table of Contents / Index chunks on authorship/count queries
            is_toc = (
                meta.get("collection_type") == "index"
                or "TABLE OF CONTENTS" in text.upper()
                or "विषय सूची" in text
                or "अनुक्रमणिका" in text
                or "विषय-सूची" in text
            )
            if (intent == "find_author" or is_count_list) and is_toc:
                score += 1.20  # dominant boost — index TOC must outrank all content chunks

            # 6. Author indicator patterns in the chunk text
            if re.search(r'(लेखक|द्वारा|author\s*:?|by\s*:?|written\s+by)', text, re.IGNORECASE):
                score += 0.30

            # 7. Numbered list / colon / byline patterns
            if re.search(r'(\d+[\.\:\-]|लेखक|द्वारा|author|by\s*:)', text, re.IGNORECASE):
                score += 0.15

            cand["rerank_score"] = round(score, 4)
            scored.append(cand)

        # Sort descending
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_n]