"""Evidence re-ranking engine using Reciprocal Rank Fusion (RRF) and multilingual semantic scoring."""
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
        variants = query_analysis.get("normalized_variants", [])
        combined_query = " ".join([orig_query] + [v for v in variants if v and v != orig_query])

        raw_words = re.findall(r"[\w\u0900-\u097F]+", combined_query.lower())
        # Standard multilingual stopwords (functional words, question words, auxiliaries)
        stopwords = {
            # English
            "what", "is", "are", "under", "written", "in", "the", "poem", "article", "story", "about",
            "kya", "hai", "me", "mein", "batao", "explain", "describe", "ka", "ke", "ki",
            "se", "par", "ko", "and", "or", "to", "of", "a", "an", "who", "which", "where",
            "how", "did", "done", "doing", "been", "being", "antargat", "aur", "kaun", "hain", "kare", "diye",
            # Note: 'kitne', 'kitna', 'kitni', 'many', 'much' are intentionally NOT here:
            # they are quantitative question-intent signals used for value-type alignment.
            # Hindi Devanagari function words & auxiliaries
            "है", "हैं", "हो", "था", "थी", "थे", "का", "के", "की", "को", "में", "से", "पर",
            "और", "या", "ने", "तो", "ही", "भी", "गए", "गई", "गईं", "गया",
            "किया", "किए", "कर", "करके", "करना", "करने", "दिए", "दिया", "हुए", "हुआ", "हुई",
            "किस", "कौन", "किसे", "कहाँ", "कब", "क्यों", "कैसे", "अंतर्गत"
            # Note: कितने, कितनी, कितना intentionally NOT in stopwords:
            # they signal quantitative intent and help disambiguate count vs percentage leaves.
        }
        # Keep significant query tokens including single Devanagari characters and digits
        significant_tokens = {
            t for t in raw_words
            if t not in stopwords and (len(t) > 1 or re.match(r'^[\u0900-\u097F]$', t) or t.isalnum())
        }
        # Generic query n-grams (phrases)
        query_bigrams = {" ".join(raw_words[i:i+2]) for i in range(len(raw_words)-1)}
        query_phrases = {
            bg for bg in query_bigrams
            if not all(w in stopwords for w in bg.split())
        }

        scored = []
        for cand in candidates:
            text = cand["text"]
            text_lower = text.lower()
            meta = cand.get("metadata", {})
            score = 0.0

            # 1. Similarity and RRF baselines
            vec_sim = cand.get("vector_sim", 0.0)
            score += vec_sim * 0.75

            rrf_score = cand.get("rrf_score", 0.0)
            score += min(0.35, rrf_score * 4.0)

            # 2. Lexical and significant token overlap
            lex_norm = min(1.0, cand.get("lexical_score", 0.0) / 10.0)
            score += lex_norm * 0.15

            text_tokens = set(re.findall(r"[\w\u0900-\u097F]+", text_lower))
            token_matches = sum(1 for t in significant_tokens if t in text_tokens)
            if significant_tokens:
                score += (token_matches / len(significant_tokens)) * 0.20

            # 3. Entity match across English & Devanagari (bounded to prevent alias inflation)
            matched_ents = {ent.lower() for ent in entities if ent.lower() in text_lower}
            if matched_ents:
                score += min(0.40, len(matched_ents) * 0.20)

            # 4. Target title matching
            target_title = query_analysis.get("target_title")
            if target_title and target_title.lower() in text_lower:
                score += 0.50

            # 4b. Exact title-in-line scoring for structured lines
            if target_title:
                for line in text.splitlines():
                    if target_title.lower() in line.lower():
                        score += 0.50
                        break

            # =========================================================================
            # 5. GENERIC HIERARCHICAL NODE & PATH SCORING
            # Evaluates: node_key, node_value, hierarchy_path, hierarchy_path_text,
            # query token overlap, and n-gram alignment without document-specific rules.
            # =========================================================================
            h_path = cand.get("hierarchy_path") or meta.get("hierarchy_path") or []
            h_path_text = cand.get("hierarchy_path_text") or meta.get("hierarchy_path_text") or text
            node_key = str(cand.get("node_key") or meta.get("node_key") or "").lower()
            node_val = cand.get("node_value") or meta.get("node_value")
            node_type = str(cand.get("node_type") or meta.get("node_type") or "")

            if h_path or node_key or node_val is not None:
                # B. Hierarchy path coverage: proportion of query content terms appearing along path
                h_path_lower = [str(elem).lower() for elem in h_path]
                h_path_combined = (" > ".join(h_path_lower) + " " + node_key + " " + str(node_val or "")).lower()
                path_tokens = set(re.findall(r"[\w\u0900-\u097F]+", h_path_combined))

                matched_path_tokens = {t for t in significant_tokens if t in path_tokens}
                if significant_tokens:
                    coverage = len(matched_path_tokens) / len(significant_tokens)
                    score += coverage * 0.80

                # A. Value-bearing leaf bonus: factual QA seeks concrete leaf values when path has relevant content.
                # For branch enumeration, structural branches and descriptive definitions are sought;
                # pure numeric table cells (e.g. letter counts in progress reports) should not outscore structural definitions.
                if node_type == "leaf" and node_val is not None and str(node_val).strip() != "":
                    is_pure_num = bool(re.fullmatch(r"[-+]?\d+(?:[,.]\d+)?%?", str(node_val).strip()))
                    if intent == "branch_enumeration" and is_pure_num:
                        score -= 0.15
                    elif not significant_tokens or len(matched_path_tokens) > 0:
                        score += 0.35

                # C. Generic multi-word phrase matching across hierarchy path
                matched_phrases = sum(1 for phr in query_phrases if phr in h_path_combined)
                if matched_phrases > 0:
                    score += min(0.60, matched_phrases * 0.20)

                # D. Node Key Match: specificity of the requested metric/attribute
                node_key_tokens = set(re.findall(r"[\w\u0900-\u097F]+", node_key))
                key_token_matches = sum(1 for t in significant_tokens if t in node_key_tokens)
                if key_token_matches > 0:
                    score += min(0.60, key_token_matches * 0.30)
                key_phrase_matches = sum(1 for phr in query_phrases if phr in node_key)
                if key_phrase_matches > 0:
                    score += min(0.40, key_phrase_matches * 0.35)

                # D2. Immediate PARENT context coverage.
                # Use h_path[-2] (the actual parent of this node) when depth > 1.
                # This measures section/context relevance without rewarding branch nodes
                # for having their own path text match the query (which was causing
                # branch nodes to unfairly outscore their sibling leaf nodes).
                # Single-level nodes (depth=1) skip D2 — their full context is in h_path[-1].
                if len(h_path) >= 2:
                    imm_parent = str(h_path[-2]).lower()
                    imm_parent_tokens = set(re.findall(r"[\w\u0900-\u097F]+", imm_parent))
                    imm_parent_tokens = {t for t in imm_parent_tokens if t not in stopwords}
                    if imm_parent_tokens:
                        imm_cov = len(significant_tokens.intersection(imm_parent_tokens)) / len(imm_parent_tokens)
                        score += imm_cov * 0.40

                # G. Generic quantitative-intent and expected-value-type alignment.
                QUANT_TOKENS = {
                    "कितने", "कितनी", "कितना", "कितनों",  # Hindi count questions
                    "kitne", "kitna", "kitni", "kitnon",  # Roman Hindi
                    "many", "much", "count", "total",     # English
                    "संख्या", "कुल"                       # Hindi count/total nouns
                }
                raw_word_set = set(raw_words)
                query_is_count_intent = bool(
                    significant_tokens.intersection(QUANT_TOKENS) or
                    raw_word_set.intersection(QUANT_TOKENS)
                )
                expected_val_type = (query_analysis.get("expected_value_type") or "").upper()
                if expected_val_type == "DURATION":
                    cand_text_dur = (str(node_val or "") + " " + node_key + " " + text_lower).lower()
                    has_dur = bool(re.search(r'(?<![\w\u0900-\u097F])(?:वर्ष|साल|माह|महीने|दिन|घंटे|years?|months?|days?|hours?)(?![\w\u0900-\u097F])', cand_text_dur, re.IGNORECASE))
                    if has_dur:
                        score += 2.50
                    elif node_type == "leaf":
                        score -= 2.50
                elif expected_val_type == "MONETARY":
                    cand_text_mon = (str(node_val or "") + " " + node_key + " " + text_lower).lower()
                    has_mon = bool(re.search(r'(?:रुपए|रुपये|रु\.|rs\.?|inr|नगद|पुरस्कार|cash|prize)', cand_text_mon, re.IGNORECASE))
                    if has_mon:
                        score += 2.50
                    elif node_type == "leaf" and re.fullmatch(r"[-+]?\d+(?:[,.]\d+)?", str(node_val or "").strip()):
                        score -= 2.50
                elif expected_val_type == "DATE":
                    cand_text_date = (str(node_val or "") + " " + node_key + " " + text_lower).lower()
                    has_date = bool(re.search(r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|(?:दिनांक|date|जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)', cand_text_date, re.IGNORECASE))
                    if has_date:
                        score += 2.50
                    elif node_type == "leaf" and re.fullmatch(r"[-+]?\d+(?:[,.]\d+)?", str(node_val or "").strip()):
                        score -= 2.50
                elif expected_val_type == "PERCENTAGE" and node_val is not None:
                    val_s = str(node_val).strip()
                    if "%" in val_s:
                        score += 0.40
                    elif re.fullmatch(r"[-+]?\d+(?:[,.]\d+)?", val_s):
                        score -= 0.40
                elif expected_val_type in {"DESCRIPTION", "PERSON"} or intent in {"person_query", "name_query"}:
                    if node_type == "leaf" and node_val is not None:
                        val_s = str(node_val).strip()
                        if re.fullmatch(r"[-+]?\d+(?:[,.]\d+)?%?", val_s) or len(val_s.split()) <= 2:
                            score -= 3.00
                        else:
                            score += 2.00
                    else:
                        score += 0.50
                elif (expected_val_type == "COUNT" or query_is_count_intent) and node_val is not None:
                    val_s = str(node_val).strip()
                    if re.fullmatch(r"[-+]?\d+(?:[,.]\d+)?", val_s):
                        score += 0.40
                    elif "%" in val_s:
                        score -= 0.50
                    elif bool(re.fullmatch(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', val_s)):
                        score -= 0.50


                # Target language alignment for hierarchical nodes
                target_langs = query_analysis.get("target_languages", [])
                if target_langs:
                    target_l = set(target_langs)
                    _L_PATS = {
                        "hindi": r'(?<![\w\u0900-\u097F])(?:हिन्दी|हिंदी|hindi)(?![\w\u0900-\u097F])',
                        "english": r'(?<![\w\u0900-\u097F])(?:अंग्रेजी|अंग्रेज़ी|english)(?![\w\u0900-\u097F])',
                        "bilingual": r'(?<![\w\u0900-\u097F])(?:द्विभाषी|द्विभाषीय|bilingual)(?![\w\u0900-\u097F])',
                    }
                    node_langs = set()
                    for lname, lpat in _L_PATS.items():
                        if re.search(lpat, node_key, re.IGNORECASE):
                            node_langs.add(lname)
                    if node_langs:
                        if target_l & node_langs:
                            score += 0.40
                        else:
                            score -= 0.50


                # H. Generic sibling contrast & attribute alignment.
                # Contrasts dual/plural intent ("दोनों", "दो", "द्वि", "द्विभाषी", "both", "bilingual", "dual")
                # with exclusive/singular intent ("केवल", "मात्र", "सिर्फ", "एकल", "only", "solely", "single", "exclusively").
                DUALITY_TOKENS = {
                    "दोनों", "दो", "द्वि", "द्विभाषी",
                    "both", "bilingual", "dual", "pair", "two"
                }
                EXCLUSIVE_TOKENS = {
                    "केवल", "मात्र", "सिर्फ", "एकल",
                    "only", "solely", "single", "exclusively"
                }
                query_has_duality = any(dt in combined_query.lower() for dt in DUALITY_TOKENS)
                query_has_exclusive = any(et in combined_query.lower() for et in EXCLUSIVE_TOKENS)

                cand_key_has_duality = any(dt in node_key for dt in DUALITY_TOKENS)
                cand_key_has_exclusive = any(et in node_key for et in EXCLUSIVE_TOKENS)

                if query_has_duality:
                    if cand_key_has_duality:
                        score += 0.40
                    elif cand_key_has_exclusive:
                        score -= 0.40

                if query_has_exclusive:
                    if cand_key_has_exclusive:
                        score += 0.40
                    elif cand_key_has_duality:
                        score -= 0.40

                # E. Entity & target title matches within hierarchy path (bounded)
                matched_path_ents = {ent.lower() for ent in entities if ent.lower() in h_path_combined}
                if matched_path_ents:
                    score += min(0.40, len(matched_path_ents) * 0.20)
                if target_title and target_title.lower() in h_path_combined:
                    score += 0.40

                # F. Node Value Match (when query asks to verify or match a value)
                if node_val is not None:
                    val_str = str(node_val).lower().strip()
                    val_tokens = set(re.findall(r"[\w\u0900-\u097F]+", val_str))
                    if val_tokens and any(t in val_tokens for t in significant_tokens if not t.isdigit()):
                        score += 0.25

                # I. Generic section & category alignment for branch enumeration and structured queries
                query_sections = re.findall(r"(?:धारा|section|dhara)\s*(\d+(?:\(\d+\))?)", combined_query.lower())
                if query_sections:
                    for q_sec in query_sections:
                        cand_sections = re.findall(r"(?:धारा|section|dhara)\s*(\d+(?:\(\d+\))?)", (h_path_combined + " " + text_lower))
                        if cand_sections:
                            if any(cs == q_sec for cs in cand_sections):
                                score += 0.40
                            else:
                                score -= 0.60

                if intent == "branch_enumeration":
                    cat_terms = {"क्षेत्र", "region", "regions", "भाग", "श्रेणी", "category", "वर्ग"}
                    q_cat_terms = {ct for ct in cat_terms if ct in combined_query.lower()}
                    if q_cat_terms:
                        cand_cat_matches = sum(1 for ct in q_cat_terms if ct in node_key or ct in h_path_combined)
                        if cand_cat_matches > 0:
                            score += 0.45

            cand["rerank_score"] = round(score, 4)
            scored.append(cand)

        # Sort descending
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_n]