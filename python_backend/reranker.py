"""Evidence re-ranking engine using Reciprocal Rank Fusion (RRF) and multilingual semantic scoring."""
import re
from typing import List, Dict, Any, Optional


class StructuralRecordMatcher:
    """Three-state eligibility checks over query and candidate structure."""

    _terminology_memory = None

    @staticmethod
    def _norm(value: Any) -> str:
        return " ".join(re.findall(r"[\w\u0900-\u097F]+", str(value or "").casefold()))

    @classmethod
    def _field_labels(cls, value: Any, document_id: Any = None) -> set:
        """Build exact comparable labels using existing multilingual normalizers."""
        label = cls._norm(value)
        if not label:
            return set()
        labels = {label}
        try:
            from index_knowledge_layer import romanize_generic
            roman = cls._norm(romanize_generic(label))
            if roman:
                labels.add(roman)
        except Exception:
            pass

        try:
            from terminology_memory import DynamicTerminologyMemory
            if cls._terminology_memory is None:
                cls._terminology_memory = DynamicTerminologyMemory()
            key = cls._terminology_memory.resolve_term(label, document_id=document_id)
            if key and key in cls._terminology_memory.terms:
                record = cls._terminology_memory.terms[key]
                labels.add(cls._norm(record.get("canonical")))
                for alias in (record.get("english", set()) | record.get("hindi", set())
                              | record.get("roman", set()) | record.get("ocr_variants", set())):
                    normalized_alias = cls._norm(alias)
                    if normalized_alias:
                        labels.add(normalized_alias)
        except Exception:
            pass

        # FieldNormalizer is the project's existing cross-lingual field
        # canonicalizer. Only accept a canonical identity shared by both inputs;
        # it cannot turn aggregation or row-entity metadata into a field label.
        try:
            from structured_store import FieldNormalizer
            canonical = FieldNormalizer.canonical(label)
            descriptor = FieldNormalizer.CANONICAL_FIELDS.get(canonical) if canonical else None
            label_tokens = FieldNormalizer._normalize_tokens(label)
            descriptor_tokens = FieldNormalizer._normalize_tokens(descriptor) if descriptor else set()
            if canonical and label_tokens.intersection(descriptor_tokens):
                labels.add(cls._norm(canonical))
        except Exception:
            pass
        labels.update(re.sub(r"[^\w\u0900-\u097F]", "", item) for item in tuple(labels))
        return {item for item in labels if item}

    @classmethod
    def _candidate_field(cls, candidate: Dict[str, Any], meta: Dict[str, Any]) -> Any:
        # Explicit field metadata takes precedence over generic hierarchy labels.
        for key in ("column_attribute", "field", "field_name", "field_key", "field_label",
                    "column_name", "attribute", "attribute_name", "attribute_key"):
            value = candidate.get(key)
            if value in (None, ""):
                value = meta.get(key)
            if value not in (None, ""):
                return value

        node_key = candidate.get("node_key") or meta.get("node_key")
        if node_key not in (None, ""):
            # A node key equal to the separately stored row or parent identity
            # describes that entity, not the field containing its value.
            row_entity = candidate.get("row_entity") or meta.get("row_entity")
            parent_key = candidate.get("parent_key") or meta.get("parent_key")
            node_label = cls._norm(node_key)
            if node_label and node_label not in {cls._norm(row_entity), cls._norm(parent_key)}:
                return node_key

        # A value-bearing leaf's final path label can identify its field when no
        # stronger field metadata exists. Never use node_value as an identity.
        path = candidate.get("hierarchy_path") or meta.get("hierarchy_path") or []
        if isinstance(path, str):
            path = [part.strip() for part in re.split(r"\s*(?:>|/|\\)\s*", path) if part.strip()]
        value = candidate.get("node_value", meta.get("node_value"))
        node_type = cls._norm(candidate.get("node_type") or meta.get("node_type")).upper()
        if len(path) >= 2 and value not in (None, "") and node_type == "LEAF":
            last = path[-1]
            row_entity = candidate.get("row_entity") or meta.get("row_entity")
            parent_key = candidate.get("parent_key") or meta.get("parent_key")
            if cls._norm(last) not in {cls._norm(row_entity), cls._norm(parent_key)}:
                return last
        return None

    @classmethod
    def evaluate_candidate(cls, analysis: Dict[str, Any], candidate: Dict[str, Any]) -> str:
        meta = candidate.get("metadata") or {}

        def get(*keys):
            for key in keys:
                value = candidate.get(key)
                if value is None or value == "":
                    value = meta.get(key)
                if value is not None and value != "":
                    return value
            return None

        identity = []
        requested_field = analysis.get("requested_field")
        candidate_field = cls._candidate_field(candidate, meta)
        if requested_field and requested_field != "branch_children":
            query_labels = cls._field_labels(requested_field, get("document_id"))
            candidate_labels = cls._field_labels(candidate_field, get("document_id"))
            if candidate_labels:
                identity.append("VALID" if query_labels.intersection(candidate_labels) else "CONTRADICTED")
            else:
                identity.append("UNKNOWN")

        expected_type = cls._norm(analysis.get("expected_value_type")).upper()
        candidate_type = cls._norm(get("value_type")).upper()
        if expected_type:
            if candidate_type:
                # Query intent (for example COUNT or PERSON) and stored value
                # shape (for example NUMERIC or TEXT) are different type systems.
                # Compare them through generic compatibility groups rather than
                # requiring their labels to be identical.
                compatible_candidate_types = {
                    "COUNT": {"NUMERIC", "NUMBER", "INTEGER", "COUNT"},
                    "PERCENTAGE": {"PERCENTAGE"},
                    "DATE": {"DATE", "TEXT"},
                    "DURATION": {"DURATION", "NUMERIC", "TEXT"},
                    "MONETARY": {"MONETARY", "NUMERIC", "TEXT"},
                    "PERSON": {"PERSON", "NAME", "TEXT"},
                    "DESCRIPTION": {"DESCRIPTION", "TEXT"},
                }
                compatible = compatible_candidate_types.get(expected_type, {expected_type})
                identity.append("VALID" if candidate_type in compatible else "CONTRADICTED")
            else:
                identity.append("UNKNOWN")

        # Keep the requested entity/section separate from the requested field.
        # Compare only against explicit structural identity (or path labels when
        # the payload lacks a dedicated slot); never infer identity from value.
        requested_entity = analysis.get("requested_entity")
        if requested_entity:
            entity_values = [get("row_entity"), get("entity"), get("entity_name")]
            entity_values = [value for value in entity_values if value not in (None, "")]
            if not entity_values:
                path = get("hierarchy_path") or []
                if isinstance(path, str):
                    path = [part.strip() for part in re.split(r"\s*(?:>|/|\\)\s*", path) if part.strip()]
                entity_values = list(path[:-1]) if path else []
                if not entity_values:
                    entity_values = [get("parent_key")]
            entity_values = [value for value in entity_values if value not in (None, "")]
            qlabels = cls._field_labels(requested_entity, get("document_id"))
            if entity_values:
                candidate_labels = set().union(*(
                    cls._field_labels(value, get("document_id")) for value in entity_values
                ))
                identity.append("VALID" if qlabels.intersection(candidate_labels) else "CONTRADICTED")
            else:
                identity.append("UNKNOWN")

        requested_section = analysis.get("parent_section")
        if requested_section:
            section_values = [get("root_section"), get("section"), get("parent_section")]
            section_values = [value for value in section_values if value not in (None, "")]
            if not section_values:
                path = get("hierarchy_path") or []
                if isinstance(path, str):
                    path = [part.strip() for part in re.split(r"\s*(?:>|/|\\)\s*", path) if part.strip()]
                section_values = path
            section_values = [value for value in section_values if value not in (None, "")]
            def section_labels(value):
                labels = cls._field_labels(value, get("document_id"))
                number_parts = re.findall(r"\d+(?:\s*\(\s*\d+\s*\))?", str(value or ""))
                if number_parts:
                    labels.add("section-number:" + " ".join(
                        re.sub(r"\s+", "", part) for part in number_parts
                    ))
                return labels

            qlabels = section_labels(requested_section)
            if section_values:
                candidate_labels = set().union(*(
                    section_labels(value) for value in section_values
                ))
                identity.append("VALID" if qlabels.intersection(candidate_labels) else "CONTRADICTED")
            else:
                identity.append("UNKNOWN")

        # Compare only explicitly analyzed dimensions; candidate metadata is optional.
        def values(value):
            if isinstance(value, (list, tuple, set)):
                return [item for item in value if item not in (None, "")]
            return [value] if value not in (None, "") else []

        for query_keys, candidate_keys in (
            (("action", "action_relation", "direction", "requested_direction", "source_direction", "target_direction"),
             ("action", "action_relation", "direction", "relationship_direction", "source_direction", "target_direction")),
            (("source_language",), ("source_language",)),
            (("target_language", "reply_language"), ("target_language", "reply_language")),
            (("region", "requested_region", "target_region"), ("region", "region_name", "row_entity")),
        ):
            qvalue = next((analysis.get(k) for k in query_keys if values(analysis.get(k))), None)
            qvalues = values(qvalue)
            if not qvalues:
                continue
            cvalue = get(*candidate_keys)
            cvalues = values(cvalue)
            if not cvalues:
                identity.append("UNKNOWN")
                continue
            qlabels = set().union(*(cls._field_labels(value, get("document_id")) for value in qvalues))
            clabels = set().union(*(cls._field_labels(value, get("document_id")) for value in cvalues))
            identity.append("VALID" if qlabels.intersection(clabels) else "CONTRADICTED")

        if "CONTRADICTED" in identity:
            return "CONTRADICTED"

        if analysis.get("intent") not in {"branch_enumeration", "list", "structure_query"}:
            value = get("node_value")
            node_type = cls._norm(get("node_type")).upper()
            has_value = value is not None and str(value).strip() != ""
            if node_type in {"BRANCH", "HEADING", "PARENT"} or not has_value:
                identity.append("UNKNOWN")

        # A factual record needs enough observed structure to qualify as VALID.
        if not identity and not any(get(k) for k in ("root_section", "parent_key", "row_entity", "column_attribute", "node_key", "hierarchy_path", "structural_record_id")):
            return "UNKNOWN"
        return "VALID" if identity and all(state == "VALID" for state in identity) else "UNKNOWN"

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

        structural_states = {
            str(cand.get("id", id(cand))): StructuralRecordMatcher.evaluate_candidate(query_analysis, cand)
            for cand in candidates
        }
        has_factual_leaf = any(
            StructuralRecordMatcher._norm(c.get("node_type") or (c.get("metadata") or {}).get("node_type")).upper() not in {"BRANCH", "HEADING", "PARENT"}
            and (c.get("node_value") is not None or (c.get("metadata") or {}).get("node_value") is not None)
            for c in candidates
        )

        scored = []
        for cand in candidates:
            text = cand["text"]
            text_lower = text.lower()
            meta = cand.get("metadata", {})
            structural_state = structural_states[str(cand.get("id", id(cand)))]
            cand["structural_eligibility"] = structural_state
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

            # Keep structural eligibility lexicographically ahead of every soft signal.
            # If a factual leaf exists in this result set, heading/branch candidates
            # cannot be selected as the final factual evidence.
            cand["rerank_score"] = round(score, 4)
            node_type_norm = StructuralRecordMatcher._norm(cand.get("node_type") or meta.get("node_type")).upper()
            if has_factual_leaf and node_type_norm in {"BRANCH", "HEADING", "PARENT"}:
                structural_state = "CONTRADICTED"
                cand["structural_eligibility"] = structural_state
            scored.append(cand)

        # Explicit contradictions are ineligible. VALID evidence always precedes
        # UNKNOWN structure; soft scores only order candidates within each state.
        scored = [c for c in scored if c.get("structural_eligibility") != "CONTRADICTED"]
        state_order = {"VALID": 0, "UNKNOWN": 1, "CONTRADICTED": 2}
        scored.sort(key=lambda x: (state_order.get(x.get("structural_eligibility"), 1), -x["rerank_score"]))
        return scored[:top_n]
