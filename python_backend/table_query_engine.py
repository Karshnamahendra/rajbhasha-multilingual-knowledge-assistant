"""Embedding-first, non-hardcoded cross-lingual lookup over extracted table data.

Strategy:
  1. Build rich candidate descriptions from the document's own text
     (table section header + row label + column subheader).
  2. Score via multilingual cosine similarity (sentence-transformers model).
  3. Boost with algorithmic phonetic token overlap (romanize_generic) - no dictionaries.
  4. Structural regex boosters for Section 3(3) and region markers (ka/kha/ga)
     which are literal patterns from the document, not synonym lists.
  5. Multi-clause splitting on conjunctions (aur/and/tatha/evam/va).
"""
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from table_store import StructuredTableStore
from index_knowledge_layer import romanize_generic  # algorithmic, no dictionaries
from terminology_memory import DynamicTerminologyMemory

# ---------------------------------------------------------------------------
# Module-level regex constants - purely structural document patterns
# ---------------------------------------------------------------------------
_CONJUNCTION_RE = re.compile(r'\b(?:aur|and|\u0924\u0925\u093e|\u090f\u0935\u0902|va)\b', re.IGNORECASE)

# Section 3(3) - matches literal notation used in Rajbhasha forms
_SECTION_33_RE = re.compile(
    r'(?:\u0927\u093e\u0930\u093e\s*3\s*\(\s*3\s*\))'
    r'|(?:section\s*3\s*\(\s*3\s*\))'
    r'|(?:\b3\s*\(\s*3\s*\))'
    r'|(?:\b3\s*[-/]\s*3\b)',
    re.IGNORECASE,
)

# Region ka / kha / ga - structural markers from the document forms.
# The roman patterns allow optional quote/punctuation between the region label and
# 'kshetra'/'zone' so that queries like "'Ka' kshetra" or "ka' kshetra" are matched.
_REGION_KA_RE = re.compile(
    r"(?:[\u2018\u2019'\u201c\u201d`]?\s*\u0915\s*[\u2018\u2019'\u201c\u201d`]?\s*\u0915\u094d\u0937\u0947\u0924\u094d\u0930)"
    r"|(?:(?:region|zone|kshetr|kshetra)\s*(?:a|ka)\b)"
    r"|(?:\bka\b[\s\u2018\u2019'\u201c\u201d`\"]*(?:region|kshetra|kshetr|zone)\b)"
    r"|(?:[\u2018\u2019'\u201c\u201d]ka[\u2018\u2019'\u201c\u201d])"
    r"|(?:[\u2018\u2019]\u0915[\u2018\u2019])",
    re.IGNORECASE,
)
_REGION_KHA_RE = re.compile(
    r"(?:[\u2018\u2019'\u201c\u201d`]?\s*\u0916\s*[\u2018\u2019'\u201c\u201d`]?\s*\u0915\u094d\u0937\u0947\u0924\u094d\u0930)"
    r"|(?:(?:region|zone|kshetr|kshetra)\s*(?:b|kha)\b)"
    r"|(?:\bkha\b[\s\u2018\u2019'\u201c\u201d`\"]*(?:region|kshetra|kshetr|zone)\b)"
    r"|(?:[\u2018\u2019'\u201c\u201d]kha[\u2018\u2019'\u201c\u201d])"
    r"|(?:[\u2018\u2019]\u0916[\u2018\u2019])",
    re.IGNORECASE,
)
_REGION_GA_RE = re.compile(
    r"(?:[\u2018\u2019'\u201c\u201d`]?\s*\u0917\s*[\u2018\u2019'\u201c\u201d`]?\s*\u0915\u094d\u0937\u0947\u0924\u094d\u0930)"
    r"|(?:(?:region|zone|kshetr|kshetra)\s*(?:c|ga)\b)"
    r"|(?:\bga\b[\s\u2018\u2019'\u201c\u201d`\"]*(?:region|kshetra|kshetr|zone)\b)"
    r"|(?:[\u2018\u2019'\u201c\u201d]ga[\u2018\u2019'\u201c\u201d])"
    r"|(?:[\u2018\u2019]\u0917[\u2018\u2019])",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phonetic_reduce(tok: str) -> str:
    """Algorithmic phonetic reduction for cross-lingual Indo-Aryan / English script matching."""
    t = tok.lower().strip()
    t = re.sub(r'[^a-z0-9]', '', t)
    if not t:
        return ""
    t = t.replace('ph', 'f').replace('w', 'v').replace('z', 'j').replace('sh', 's')
    t = t.replace('oo', 'u').replace('ee', 'i').replace('aa', 'a').replace('ei', 'e')
    t = re.sub(r'([b-df-hj-np-tv-z])a?n([b-df-hj-np-tv-z])', r'\1n\2', t)
    t = re.sub(r'(en|on|an|s|es|a|e|i|o|u)$', '', t)
    t = re.sub(r'(.)\1+', r'\1', t)
    return t


def _normalize_tokens(text: str, memory: Optional[DynamicTerminologyMemory] = None) -> Set[str]:
    """
    Lower-case token set using original script, algorithmic phonetic
    Roman transliteration via romanize_generic, phonetic reduction, and
    dynamic cross-lingual expansion via DynamicTerminologyMemory.
    """
    if not text:
        return set()
    lower = text.lower().strip()
    roman = romanize_generic(lower)
    tokens: Set[str] = set()
    for chunk in (lower, roman):
        for tok in re.findall(r'[\w\u0900-\u097F]+', chunk):
            if len(tok) > 1:
                tokens.add(tok)
                p = _phonetic_reduce(tok)
                if p and len(p) > 1:
                    tokens.add(p)
    # Dynamic cross-lingual expansion via DynamicTerminologyMemory
    mem = memory or DynamicTerminologyMemory()
    c_detected = mem.detect_concepts(text)
    for ckey in c_detected.get("all", set()):
        rec = mem.terms.get(ckey, {})
        for form in rec.get("english", set()) | rec.get("hindi", set()) | rec.get("roman", set()):
            for part in form.lower().split():
                if len(part) > 1:
                    tokens.add(part)
                    p = _phonetic_reduce(part)
                    if p and len(p) > 1:
                        tokens.add(p)
    return tokens


def _token_overlap_score(q_tokens: Set[str], c_tokens: Set[str]) -> float:
    """Query token recall: fraction of query tokens found in candidate."""
    if not q_tokens or not c_tokens:
        return 0.0
    inter = len(q_tokens & c_tokens)
    if not inter:
        return 0.0
    recall = inter / len(q_tokens)
    union = len(q_tokens | c_tokens)
    jaccard = inter / union if union else 0.0
    return 0.8 * recall + 0.2 * jaccard


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _detect_regions(text: str) -> Set[str]:
    regions: Set[str] = set()
    if _REGION_KA_RE.search(text):
        regions.add('region_ka')
    if _REGION_KHA_RE.search(text):
        regions.add('region_kha')
    if _REGION_GA_RE.search(text):
        regions.add('region_ga')
    return regions


def _get_table_section_label(table: Dict[str, Any]) -> str:
    """
    Returns the primary semantic topic of the table using the document's own
    column header text (first non-trivial, non-'Column X' header).
    """
    for h in table.get('headers', []):
        if not h.startswith('Column ') and not re.fullmatch(r'\d+\.?', h.strip()):
            return h
    return ''


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class StructuredTableEngine:
    """
    Embedding-first non-hardcoded cross-lingual table query engine.

    Final score = EMBED_W * cosine_similarity
                + LEX_W   * phonetic_token_overlap
                + structural_boosts (Section 3(3), regions - literal doc patterns)
                + concept & constraint scoring (language, role, status)
    """

    EMBED_W              = 8.0
    LEX_W                = 12.0
    SECTION_33_BOOST     = 12.0
    REGION_MATCH_BOOST   = 8.0
    REGION_MISMATCH_PEN  = -15.0
    EMPTY_VAL_PEN        = -8.0
    MIN_SCORE_THRESHOLD  = 3.0

    def __init__(
        self,
        store: StructuredTableStore,
        embed_fn: Optional[Callable] = None,
        memory: Optional[DynamicTerminologyMemory] = None,
    ):
        self.store = store
        self.embed_fn = embed_fn
        self.memory = memory or DynamicTerminologyMemory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        document_id: Optional[str],
        table_ids: Iterable[str],
        field_labels: Iterable[str] = (),
        query_variants: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:

        tables = self.store.get_tables(document_id)
        if table_ids:
            t_set = set(table_ids)
            tables = [t for t in tables if t.get('table_id') in t_set]
        if not tables:
            return None

        operation = self._operation(query)

        # Arithmetic operations (max/min/sum/average/difference/percentage)
        if operation in {'max', 'min', 'average', 'sum', 'difference', 'percentage'}:
            q_tokens = _normalize_tokens(query, self.memory)
            for table in tables:
                calc = self._calculate_table(query, table, operation, q_tokens)
                if calc:
                    return calc

        # Multi-clause compound query (e.g. "X aur Y kitni thi?")
        raw_clauses = _CONJUNCTION_RE.split(query)
        clauses = [c.strip() for c in raw_clauses if len(c.strip().split()) >= 2]

        selected: List[Dict[str, Any]] = []

        if len(clauses) >= 2:
            per_clause = []
            for clause in clauses:
                cands = self._score_candidates(clause, tables, query_variants=query_variants)
                if cands and cands[0]['score'] >= self.MIN_SCORE_THRESHOLD:
                    per_clause.append(cands[0])
            # Accept multi-clause result only when all clauses matched from same table
            if (len(per_clause) == len(clauses)
                    and len({c['table']['table_id'] for c in per_clause}) == 1):
                selected = per_clause

        if not selected:
            scored = self._score_candidates(query, tables, query_variants=query_variants)
            if not scored or scored[0]['score'] < self.MIN_SCORE_THRESHOLD:
                return None

            best = scored[0]
            selected = [best]

            q_regions = _detect_regions(query)
            if best.get('is_regional_row') and not q_regions and best['score'] >= 6.0:
                # No region specified -> aggregate all regional rows for same column
                best_tid = best['table']['table_id']
                req_col = best['details'].get('requested_column')
                regional = [
                    c for c in scored
                    if c['table']['table_id'] == best_tid
                    and c['details'].get('requested_column') == req_col
                    and c.get('is_regional_row')
                ]
                seen: Set[tuple] = set()
                unique_reg = []
                for c in regional:
                    key = tuple(sorted(c.get('regions', ())))
                    if key and key not in seen:
                        seen.add(key)
                        unique_reg.append(c)
                if len(unique_reg) > 1:
                    selected = unique_reg
            elif q_regions:
                best_tid = best['table']['table_id']
                req_col = best['details'].get('requested_column')
                selected = []
                for region in q_regions:
                    match = next(
                        (c for c in scored
                         if c['table']['table_id'] == best_tid
                         and (not req_col or c['details'].get('requested_column') == req_col)
                         and region in c.get('regions', set())),
                        None,
                    )
                    if match:
                        selected.append(match)

        if not selected:
            return None

        q_tokens = _normalize_tokens(query, self.memory)
        final_answer = self._format_final_answer(selected, q_tokens, query=query)
        for c in selected:
            if 'details' in c and 'label' in c['details']:
                c['details']['label'] = self._strip_clause_bullet(c['details']['label'], query)
        best_table = selected[0]['table']
        return {
            'answer': final_answer,
            'details': {'matches': [c['details'] for c in selected]},
            'evidence': [{
                'chunk_id': f"{best_table['table_id']}:summary",
                'document_id': best_table['document_id'],
                'text': final_answer,
                'page': best_table.get('page_number'),
                'section': best_table.get('section', 'General'),
                'score': 1.0,
                'chunk_type': 'table',
                'table_id': best_table['table_id'],
                'extraction_confidence': best_table.get('extraction_confidence', 0.0),
            }],
        }

    # ------------------------------------------------------------------
    # Candidate extraction
    # ------------------------------------------------------------------

    def _build_candidates(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For every scorable cell, build a rich text description from the
        document's own labels (table section header + row label + col subheader).
        Fully dynamic and universal across matrix and key-value form tables.
        """
        candidates = []
        for table in tables:
            table_header = _get_table_section_label(table)
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            if not headers and not rows:
                continue

            # Detect key-value form table (contains ':' in headers or rows)
            has_colon_in_hdr = any(str(h).strip() == ':' for h in headers)
            has_colon_in_rows = any(any(str(v).strip() == ':' for v in r.values()) for r in rows)

            if has_colon_in_hdr or has_colon_in_rows:
                all_rows = []
                if has_colon_in_hdr:
                    # Headers row absorbed row 0
                    all_rows.append({h: h for h in headers})
                all_rows.extend(rows)

                parent_context = ""
                for r in all_rows:
                    # Find colon column
                    c_idx = -1
                    for idx, h in enumerate(headers):
                        if str(r.get(h, '')).strip() == ':':
                            c_idx = idx
                            break
                    if c_idx >= 1:
                        lbl_col = headers[c_idx - 1]
                        lbl = str(r.get(lbl_col, '')).strip()
                        val_col = headers[c_idx + 1] if c_idx + 1 < len(headers) else None
                        val = str(r.get(val_col, '')).strip() if val_col else ""

                        prefix_parts = []
                        for h in headers[:c_idx - 1]:
                            pv = str(r.get(h, '')).strip()
                            if pv and pv != ':':
                                prefix_parts.append(pv)
                        prefix = " ".join(prefix_parts).strip()

                        full_label_parts = []
                        if prefix:
                            full_label_parts.append(prefix)
                        if lbl:
                            full_label_parts.append(lbl)
                        row_lbl = " ".join(full_label_parts).strip()

                        if any(kw in lbl for kw in ['कार्यालयों', 'अनुभागों', 'समिति', 'अधिकारी', 'कार्मिक']):
                            parent_context = lbl

                        desc_parts = [p for p in [table_header, parent_context, row_lbl] if p]
                        seen_p = []
                        for p in desc_parts:
                            if p not in seen_p:
                                seen_p.append(p)
                        desc = " ".join(seen_p)
                        desc_rom = romanize_generic(desc)

                        if row_lbl and val and val != ':':
                            candidates.append({
                                'table': table,
                                'desc': desc,
                                'desc_roman': desc_rom,
                                'display': f"{row_lbl}: {val}",
                                'details': {
                                    'requested_column': lbl,
                                    'value': val,
                                    'label': row_lbl,
                                },
                                'val': val,
                                'regions': _detect_regions(desc),
                                'is_regional_row': bool(_detect_regions(desc)),
                            })
                continue

            # Matrix / standard table
            header_rows = []
            data_rows = []
            for r in rows:
                numeric_cells = [
                    int(str(v).strip()) for v in r.values()
                    if re.fullmatch(r'\d+', str(v).strip())
                ]
                is_numbering = (
                    len(numeric_cells) >= 2
                    and numeric_cells == list(range(1, len(numeric_cells) + 1))
                )
                if not data_rows and (not numeric_cells or is_numbering):
                    header_rows.append(r)
                else:
                    data_rows.append(r)

            # Build column subheaders
            ctx: Dict[str, List[str]] = {col: [] for col in headers}
            for r in header_rows:
                inherited = ''
                for col in headers:
                    val = str(r.get(col, '')).strip()
                    meaningful = val and val != ':' and not val.isdigit()
                    if meaningful:
                        inherited = val
                        ctx[col].append(val)
                    elif inherited:
                        ctx[col].append(inherited)

            subheaders = {
                col: ' | '.join(dict.fromkeys(parts))
                for col, parts in ctx.items() if parts
            }
            if not subheaders:
                real = [h for h in headers
                        if not h.startswith('Column ') and not re.fullmatch(r'\d+\.?', h)]
                if len(real) >= 1:
                    subheaders = {h: h for h in headers}

            for r in data_rows:
                label_parts = []
                data_cols = []
                for h in headers:
                    v = str(r.get(h, '')).strip()
                    if not v or v == ':':
                        continue
                    is_num = bool(re.fullmatch(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?', v))
                    if not is_num and len(data_cols) == 0:
                        label_parts.append(v)
                    else:
                        data_cols.append((h, v))

                row_lbl = " ".join(label_parts).strip()
                row_regions = _detect_regions(row_lbl)

                for col_name, val in data_cols:
                    subhead = subheaders.get(col_name, col_name)
                    is_dummy_subhead = (
                        subhead.startswith('Column ')
                        or re.fullmatch(r'\d+\.?', subhead)
                        or not subhead
                    )

                    if is_dummy_subhead:
                        display_lbl = row_lbl
                        desc_parts = [table_header, row_lbl]
                    else:
                        if subhead in row_lbl:
                            display_lbl = row_lbl
                            desc_parts = [table_header, row_lbl]
                        else:
                            display_lbl = f"{row_lbl} ({subhead})"
                            desc_parts = [table_header, row_lbl, subhead]

                    seen_p = []
                    for p in desc_parts:
                        if p and p not in seen_p:
                            seen_p.append(p)
                    desc = " ".join(seen_p)
                    desc_rom = romanize_generic(desc)

                    candidates.append({
                        'table': table,
                        'desc': desc,
                        'desc_roman': desc_rom,
                        'display': f"{display_lbl}: {val}",
                        'details': {
                            'requested_column': col_name,
                            'value': val,
                            'label': display_lbl,
                        },
                        'val': val,
                        'regions': row_regions,
                        'is_regional_row': bool(row_regions),
                    })

        return candidates

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        query: str,
        tables: List[Dict[str, Any]],
        query_variants: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:

        candidates = self._build_candidates(tables)
        if not candidates:
            return []

        all_queries = [query]
        if query_variants:
            for v in query_variants:
                if v and v.strip() and v not in all_queries:
                    all_queries.append(v)

        query_token_sets = [_normalize_tokens(q, self.memory) for q in all_queries if q and q.strip()]

        q_has_33 = any(bool(_SECTION_33_RE.search(q)) for q in all_queries)
        q_regions = set()
        for q in all_queries:
            q_regions.update(_detect_regions(q))

        q_det = self.memory.detect_concepts(query)
        if query_variants:
            for v in query_variants:
                if v and v.strip():
                    v_det = self.memory.detect_concepts(v)
                    for cat, keys in v_det.items():
                        q_det.setdefault(cat, set()).update(keys)

        q_concepts = q_det.get("all", set())
        q_langs = q_det.get("language", set())

        # Batch embed for efficiency: embed query variants + both Devanagari & Roman candidate descs
        q_embs: Optional[List[np.ndarray]] = None
        c_embs: Optional[np.ndarray] = None
        c_rom_embs: Optional[np.ndarray] = None

        if self.embed_fn is not None:
            try:
                c_descs = [c['desc'] for c in candidates]
                all_texts = all_queries + c_descs
                all_emb = np.array(self.embed_fn(all_texts), dtype=np.float32)
                n_q = len(all_queries)
                q_embs = [all_emb[j] for j in range(n_q)]
                c_embs = all_emb[n_q:]
            except Exception:
                q_embs = None
                c_embs = None

        scored = []
        for i, cand in enumerate(candidates):
            # 1. Lexical overlap across query representations (algorithmic phonetic + dynamic memory)
            c_tokens = _normalize_tokens(cand['desc'], self.memory)
            lex_score = max((_token_overlap_score(q_toks, c_tokens) for q_toks in query_token_sets), default=0.0)

            # 2. Embedding similarity: natural cross-lingual alignment against candidate text
            emb_score = 0.0
            if q_embs is not None and c_embs is not None:
                emb_score = max(_cosine_sim(qe, c_embs[i]) for qe in q_embs)

            score = self.EMBED_W * emb_score + self.LEX_W * lex_score

            # 3. Structural boost: Section 3(3)
            if q_has_33 and _SECTION_33_RE.search(cand['desc']):
                score += self.SECTION_33_BOOST

            # 4. Region alignment
            c_regions = cand.get('regions', set())
            if q_regions:
                if q_regions & c_regions:
                    score += self.REGION_MATCH_BOOST
                else:
                    score += self.REGION_MISMATCH_PEN

            # 5. Dynamic Concept & Semantic Constraint Matching via DynamicTerminologyMemory
            c_cell_label = str(cand.get('details', {}).get('label') or cand.get('display') or '')
            c_desc_det = self.memory.detect_concepts(cand['desc'])
            c_cell_det = self.memory.detect_concepts(c_cell_label)
            c_concepts = c_desc_det.get("all", set()) | c_cell_det.get("all", set())
            c_cell_langs = c_cell_det.get("language", set())
            c_desc_langs = c_desc_det.get("language", set())

            # 5a. Dynamic Language constraint matching
            if q_langs:
                # Query explicitly asked for a specific language (e.g. English, Hindi, Bilingual)
                if q_langs & c_cell_langs:
                    score += 9.0
                elif q_langs & c_desc_langs:
                    score += 4.0

                # Penalize candidates specifying conflicting languages
                conflicting_cell_langs = c_cell_langs - q_langs
                if conflicting_cell_langs:
                    score -= 12.0
                elif not (q_langs & c_desc_langs):
                    score -= 6.0
            else:
                # Query did NOT ask for a specific language sub-item (general or total count requested)
                if c_cell_langs:
                    score -= 5.0
                if "total" in q_concepts and "total" in c_concepts:
                    score += 5.0

            # 5b. Role, Status, Action, Entity concepts dynamically matched from DynamicTerminologyMemory
            for cat in ["role", "status", "action", "entity"]:
                cat_terms = self.memory.get_terms_by_category(cat)
                for spec_concept in cat_terms:
                    if spec_concept in q_concepts and spec_concept in c_concepts:
                        score += 7.0
                    elif spec_concept in q_concepts and spec_concept not in c_concepts:
                        score -= 4.0
                    elif spec_concept not in q_concepts and spec_concept in c_cell_det.get("all", set()):
                        if cat in ["role", "status"]:
                            score -= 5.0

            # 6. Empty value penalty
            if not cand['val'] or cand['val'] == ':':
                score += self.EMPTY_VAL_PEN

            # Type alignment: count queries seek numeric counts, not boolean or label strings
            is_count_query = bool(re.search(r'\b(how many|how much|kitn[ei]|count|sankhya|number|kul|total)\b', query, re.IGNORECASE))
            if is_count_query:
                val_str = str(cand['val']).strip()
                val_has_digit = bool(re.search(r'\d', val_str))
                val_is_bool = bool(re.search(r'^(हां|नहीं|हाँ|yes|no|na|n/a)(?:[/|\s]|$)', val_str, re.IGNORECASE))
                if val_is_bool or not val_has_digit:
                    score += self.EMPTY_VAL_PEN

            scored.append({
                'table': cand['table'],
                'score': round(score, 3),
                'display': cand['display'],
                'details': cand['details'],
                'regions': c_regions,
                'is_regional_row': cand['is_regional_row'],
            })

        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Answer formatting (dynamic - no hardcoded word lists)
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_clause_bullet(text: str, query: str = '') -> str:
        """
        Strips table clause/bullet index prefixes like '(क) ', '(ख) ', '(ग) ',
        'iii. ', '1. ', '(1) ' from evidence display text unless the user query
        specifically referenced that clause/bullet marker (e.g. 'part (kha)', 'point 1').
        """
        if not text:
            return text

        m = re.match(
            r'^\s*(?:\(\s*([क-हa-zA-Z0-9ivxlcdmIVXLCDM]+)\s*\)|([क-हa-zA-Z0-9ivxlcdmIVXLCDM]+)[\.\)])\s*',
            text
        )
        if not m:
            return text

        marker = m.group(1) or m.group(2)
        q_lower = query.lower() if query else ''

        # Direct bracket check in query: (क), (kha), (a), etc.
        if marker and f'({marker.lower()})' in q_lower:
            return text

        dev_to_rom = {
            'क': ['ka', 'k'],
            'ख': ['kha', 'kh'],
            'ग': ['ga', 'g'],
            'घ': ['gha', 'gh'],
            'ङ': ['nga'],
            'च': ['cha'],
            'छ': ['chha'],
            'ज': ['ja', 'j'],
        }

        asked = False
        if marker in dev_to_rom:
            if marker in (query or ''):
                pat = r'(?<![\w\u0900-\u097F])' + re.escape(marker) + r'(?![\w\u0900-\u097F])'
                if re.search(pat, query):
                    asked = True
            for rom in dev_to_rom[marker]:
                if re.search(r'\b' + re.escape(rom) + r'\b', q_lower) and any(
                    kw in q_lower for kw in ['clause', 'part', 'bhag', 'bhaag', 'point', 'section', 'para', f'({rom})']
                ):
                    asked = True
        else:
            pat = r'\b' + re.escape(marker.lower()) + r'\b'
            if re.search(pat, q_lower) and any(
                kw in q_lower for kw in ['clause', 'part', 'bhag', 'bhaag', 'point', 'section', 'item', f'({marker.lower()})']
            ):
                asked = True

        if asked:
            return text

        return text[m.end():].strip()

    @classmethod
    def _format_final_answer(
        cls,
        selected: List[Dict[str, Any]],
        q_tokens: Set[str],
        query: str = '',
    ) -> str:
        unique_cands = list({c['display']: c for c in selected}.values())
        cleaned_displays = [cls._strip_clause_bullet(c['display'], query) for c in unique_cands]
        evidence_text = '\n'.join(cleaned_displays)
        values = [str(c.get('details', {}).get('value', '')).strip() for c in unique_cands]

        # Case 1: Exactly 2 results where one is hindi-medium and other is broader
        if len(unique_cands) == 2:
            hindi_toks = _normalize_tokens('\u0939\u093f\u0902\u0926\u0940 \u0939\u093f\u0928\u094d\u0926\u0940 hindi')
            t0 = _normalize_tokens(unique_cands[0]['display'])
            t1 = _normalize_tokens(unique_cands[1]['display'])
            is_h0 = bool(t0 & hindi_toks)
            is_h1 = bool(t1 & hindi_toks)
            if is_h0 != is_h1:
                total_val = values[1] if is_h0 else values[0]
                hindi_val = values[0] if is_h0 else values[1]
                summary = (
                    f'**Answer:** **\u0915\u0941\u0932 {total_val} '
                    f'\u092d\u0947\u091c\u0940 \u0917\u0908\u0902 '
                    f'\u0914\u0930 \u0909\u0928\u092e\u0947\u0902 \u0938\u0947 '
                    f'{hindi_val} \u0939\u093f\u0902\u0926\u0940 \u092e\u0947\u0902 \u0925\u0940\u0902\u0964**'
                )
                return f'{evidence_text}\n\n{summary}'

        # Case 2: Multiple numeric rows (e.g. regional rows) -> sum
        numeric_vals = []
        for v in values:
            m = re.search(r'[-+]?\d+', v)
            if m:
                numeric_vals.append(int(m.group()))

        if len(numeric_vals) == len(values) and len(values) > 1:
            total_num = sum(numeric_vals)
            summary = f'**Answer:** **{total_num}.**'
            return f'{evidence_text}\n\n{summary}'

        # Case 3: Single result
        if len(values) == 1:
            summary = f'**Answer:** **{values[0]}.**'
            return f'{evidence_text}\n\n{summary}'

        return evidence_text

    # ------------------------------------------------------------------
    # Arithmetic operations
    # ------------------------------------------------------------------

    def _calculate_table(
        self,
        query: str,
        table: Dict[str, Any],
        operation: str,
        q_tokens: Set[str],
    ) -> Optional[Dict[str, Any]]:
        headers = table.get('headers', [])
        numeric_headers = []
        for h in headers:
            vals = [self._number(row.get(h, '')) for row in table.get('rows', [])]
            if sum(1 for v in vals if v is not None) >= 2:
                numeric_headers.append(h)
        if not numeric_headers:
            return None
        return self._calculate(operation, numeric_headers[0], table.get('rows', []), table)

    @staticmethod
    def _operation(query: str) -> str:
        lower = query.lower()
        if re.search(r'(?<![\u0900-\u097F])(\u0905\u0902\u0924\u0930|\u092b\u0930\u094d\u0915|\u0918\u091f|\u092c\u0922\u093c)(?![\u0900-\u097F])|\b(difference|diff|increase|decrease|minus)\b', lower):
            return 'difference'
        if re.search(r'\b(highest|maximum|max|largest|\u0938\u092c\u0938\u0947 \u0905\u0927\u093f\u0915|\u0905\u0927\u093f\u0915\u0924\u092e)\b', lower):
            return 'max'
        if re.search(r'\b(lowest|minimum|min|smallest|\u0938\u092c\u0938\u0947 \u0915\u092e|\u0928\u094d\u092f\u0942\u0928\u0924\u092e)\b', lower):
            return 'min'
        if re.search(r'\b(average|mean|\u0914\u0938\u0924)\b', lower):
            return 'average'
        if re.search(r'\b(percentage|percent|\u092a\u094d\u0930\u0924\u093f\u0936\u0924)\b', lower):
            return 'percentage'
        return 'lookup'

    def _calculate(
        self,
        operation: str,
        header: str,
        rows: List[Dict[str, str]],
        table: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        parsed = [(row, self._number(row.get(header, ''))) for row in rows]
        parsed = [(row, value) for row, value in parsed if value is not None]
        if not parsed:
            return None
        if operation == 'difference' and len(parsed) >= 2:
            first, last = parsed[0], parsed[-1]
            amount = abs(last[1][0] - first[1][0])
            unit = self._common_unit(first[1][1], last[1][1])
            text = (f'Difference in {header}: {self._format(amount, unit)} '
                    f'({self._row_label(first[0], table)} to {self._row_label(last[0], table)}).')
        elif operation == 'max':
            row, (amount, unit) = max(parsed, key=lambda x: x[1][0])
            text = f'Highest {header}: {self._format(amount, unit)} ({self._row_label(row, table)}).'
        elif operation == 'min':
            row, (amount, unit) = min(parsed, key=lambda x: x[1][0])
            text = f'Lowest {header}: {self._format(amount, unit)} ({self._row_label(row, table)}).'
        elif operation in {'sum', 'average'}:
            units = {unit for _, (_, unit) in parsed if unit}
            if len(units) > 1:
                return None
            amount = sum(value[0] for _, value in parsed)
            if operation == 'average':
                amount /= Decimal(len(parsed))
            text = f'{operation.capitalize()} {header}: {self._format(amount, next(iter(units), ""))}.'
        else:
            return None
        return self._result(table, text, {
            'requested_column': header,
            'operation': operation,
            'matches': [dict(row) for row, _ in parsed],
        })

    @staticmethod
    def _number(value: str) -> Optional[Tuple[Decimal, str]]:
        match = re.search(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', str(value))
        if not match:
            return None
        try:
            return Decimal(match.group().replace(',', '')), str(value)[match.end():].strip()
        except InvalidOperation:
            return None

    @staticmethod
    def _common_unit(first: str, second: str) -> str:
        return first if first.lower() == second.lower() else ''

    @staticmethod
    def _format(amount: Decimal, unit: str) -> str:
        value = (
            format(amount.normalize(), 'f').rstrip('0').rstrip('.')
            if amount % 1 else str(int(amount))
        )
        return f'{value} {unit}'.strip()

    @staticmethod
    def _row_label(row: Dict[str, str], table: Dict[str, Any]) -> str:
        headers = table.get('headers') or ['Row']
        for header in headers:
            value = str(row.get(header, '')).strip()
            if value and value != ':' and not re.fullmatch(r'\d+', value):
                return value
        first = headers[0]
        return f'{first} {row.get(first, "")}'.strip()

    @staticmethod
    def _result(table: Dict[str, Any], answer: str, details: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'answer': answer,
            'details': details,
            'evidence': [{
                'chunk_id': f"{table['table_id']}:summary",
                'document_id': table['document_id'],
                'text': answer,
                'page': table.get('page_number'),
                'section': table.get('section', 'General'),
                'score': 1.0,
                'chunk_type': 'table',
                'table_id': table['table_id'],
                'extraction_confidence': table.get('extraction_confidence', 0.0),
            }],
        }
