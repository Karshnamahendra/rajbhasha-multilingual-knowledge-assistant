"""Persistent, document-scoped facts extracted from report metadata and forms."""
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np

from index_knowledge_layer import romanize_generic
from doc_scope import get_scope, in_scope


class FieldNormalizer:
    """Dynamic embedding-based cross-lingual field recognizer.

    Uses multilingual sentence embeddings and algorithmic transliteration
    for Hindi/English/Roman-Hindi matching.  No hardcoded synonym lists —
    the ``paraphrase-multilingual-MiniLM-L12-v2`` model handles cross-lingual
    bridging dynamically.
    """

    # Canonical field semantic anchors — bilingual concept descriptors.
    # Multilingual sentence embedding bridges queries dynamically.
    CANONICAL_FIELDS = {
        "reporting_period": "quarterly reporting period timahi अवधि तिमाही",
        "office_name": "office name organization karyalay कार्यालय का नाम",
        "address": "postal address full address location pata पूरा पता कार्यालय का पता",
        "phone": "phone telephone mobile number contact फोन दूरभाष",
        "email": "email address electronic mail ईमेल",
        "employees": "employee staff personnel worker count कर्मचारी कार्मिक",
    }

    _embed_fn = None
    _transliterate_fn = None
    _field_embeddings: Optional[Dict[str, np.ndarray]] = None
    _field_tokens: Optional[Dict[str, Set[str]]] = None
    _embedding_cache: Dict[str, np.ndarray] = {}
    _CACHE_MAX = 500
    SIMILARITY_THRESHOLD = 0.35

    @classmethod
    def initialize(cls, embed_fn, transliterate_fn=None):
        """Called once at pipeline startup to pre-compute field embeddings."""
        cls._embed_fn = embed_fn
        cls._transliterate_fn = transliterate_fn
        cls._embedding_cache = {}
        descriptions = list(cls.CANONICAL_FIELDS.values())
        raw = embed_fn(descriptions)
        cls._field_embeddings = {
            field: np.array(raw[i], dtype=np.float32)
            for i, field in enumerate(cls.CANONICAL_FIELDS.keys())
        }
        cls._field_tokens = {
            field: cls._normalize_tokens(desc)
            for field, desc in cls.CANONICAL_FIELDS.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _normalize_tokens(cls, text: str) -> Set[str]:
        if not text:
            return set()
        lower = text.lower().strip()
        roman = romanize_generic(lower)
        tokens: Set[str] = set()
        for chunk in (lower, roman):
            for tok in re.findall(r'[\w\u0900-\u097F]+', chunk):
                if len(tok) > 1:
                    tokens.add(tok)
        return tokens

    @classmethod
    def _get_variants(cls, text: str) -> List[str]:
        variants = [text]
        roman = romanize_generic(text)
        if roman and roman.strip() != text.strip():
            variants.append(roman)
        if cls._transliterate_fn:
            try:
                deva = cls._transliterate_fn(text)
                if deva and deva.strip() != text.strip():
                    variants.append(deva)
            except Exception:
                pass
        return variants

    @classmethod
    def _cosine_sim(cls, a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @classmethod
    def _compute_field_scores(cls, value: str) -> Dict[str, float]:
        if cls._embed_fn is None or cls._field_embeddings is None:
            return {}

        variants = cls._get_variants(value)
        # Check cache or embed variants
        uncached = [v for v in variants if v not in cls._embedding_cache]
        if uncached:
            if len(cls._embedding_cache) + len(uncached) >= cls._CACHE_MAX:
                cls._embedding_cache.clear()
            raw = cls._embed_fn(uncached)
            for v, emb in zip(uncached, raw):
                cls._embedding_cache[v] = np.array(emb, dtype=np.float32)

        v_embs = [cls._embedding_cache[v] for v in variants if v in cls._embedding_cache]
        if not v_embs and variants:
            raw = cls._embed_fn(variants)
            v_embs = [np.array(e, dtype=np.float32) for e in raw]
        q_tokens = cls._normalize_tokens(value)

        scores: Dict[str, float] = {}
        for field, f_emb in cls._field_embeddings.items():
            base_sim = max(cls._cosine_sim(vemb, f_emb) for vemb in v_embs)
            f_tokens = cls._field_tokens.get(field, set()) if cls._field_tokens else set()
            tok_overlap = len(q_tokens & f_tokens) / max(len(q_tokens), 1)
            scores[field] = base_sim + 0.3 * tok_overlap

        return scores

    # ------------------------------------------------------------------
    # Public API (same signatures as the old class)
    # ------------------------------------------------------------------

    @classmethod
    def canonical(cls, text: str) -> Optional[str]:
        """Return the canonical field name for *text*, or ``None``."""
        if not text or not str(text or "").strip():
            return None
        value = " ".join(str(text).lower().split())

        scores = cls._compute_field_scores(value)
        if not scores:
            return None

        best_field = max(scores, key=scores.get)
        best_score = scores[best_field]

        if best_score < cls.SIMILARITY_THRESHOLD:
            return None

        # Preserve priority: prefer "address" over "office_name"
        # when query asks for address/location and doesn't explicitly mention "name".
        if best_field == "office_name":
            addr_score = scores.get("address", 0.0)
            has_name = any(m in value for m in ("नाम", "naam", "name"))
            if not has_name and addr_score >= cls.SIMILARITY_THRESHOLD and addr_score > best_score - 0.12:
                return "address"

        return best_field

    @classmethod
    def matches(cls, text: str, field: str) -> bool:
        """Check whether *text* semantically matches a specific *field*."""
        if not text or not str(text or "").strip():
            return False
        if field not in cls.CANONICAL_FIELDS:
            return False
        value = " ".join(str(text).lower().split())
        scores = cls._compute_field_scores(value)
        return scores.get(field, 0.0) >= cls.SIMILARITY_THRESHOLD


class StructuredRecordStore:
    def __init__(self, storage_dir: Optional[str] = None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.storage_dir = storage_dir or os.path.join(base, "structured_store")
        os.makedirs(self.storage_dir, exist_ok=True)

    def _path(self, document_id: str) -> str:
        safe_id = re.sub(r"[^\w.\-]", "_", document_id)
        return os.path.join(self.storage_dir, f"{safe_id}_records.json")

    def save_records(self, document_id: str, filename: str, records: List[Dict[str, Any]]) -> None:
        with open(self._path(document_id), "w", encoding="utf-8") as target:
            json.dump({"schema_version": 1, "document_id": document_id, "filename": filename,
                       "records": records}, target, ensure_ascii=False, indent=2)

    def get_records(self, document_id: Optional[str]) -> List[Dict[str, Any]]:
        if document_id:
            return self._read(self._path(document_id))
        records: List[Dict[str, Any]] = []
        for name in os.listdir(self.storage_dir):
            if name.endswith("_records.json"):
                records.extend(self._read(os.path.join(self.storage_dir, name)))
        if get_scope():
            records = [r for r in records if in_scope(r.get("document_id"))]
        return records

    def find(self, document_id: Optional[str], field: str) -> Optional[Dict[str, Any]]:
        matches = [record for record in self.get_records(document_id)
                   if record.get("field") == field and record.get("value_state") == "present"]
        # Prefer metadata/form labels over a coincidental table cell.
        matches.sort(key=lambda r: (r.get("source_type") != "metadata", r.get("page") is None,
                                    field == "address" and "continuation" not in r.get("label", "").lower()))
        return matches[0] if matches else None

    def find_all(self, document_id: Optional[str], field: str) -> List[Dict[str, Any]]:
        """Return ALL matching records for a field (e.g. address has multiple lines)."""
        matches = [record for record in self.get_records(document_id)
                   if record.get("field") == field and record.get("value_state") == "present"]
        # Put non-continuation records first so the primary value comes before the street line.
        matches.sort(key=lambda r: ("continuation" in r.get("label", "").lower(),
                                    r.get("source_type") != "metadata"))
        return matches

    def delete_document(self, document_id: str) -> None:
        path = self._path(document_id)
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def _read(path: str) -> List[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as source:
                return json.load(source).get("records", [])
        except (OSError, ValueError, TypeError):
            return []


def build_structured_records(document_id: str, pages: Iterable[Dict[str, Any]],
                             tables: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create verified facts from extracted labels/values; never invent a value."""
    records: List[Dict[str, Any]] = []
    seen = set()

    def add(source_type: str, page: Any, section: str, label: str, value: Any,
            table_id: Optional[str] = None, forced_field: Optional[str] = None):
        field = forced_field or FieldNormalizer.canonical(label)
        if not field:
            return
        # A table heading can mention a quarter while its numeric cells are counts;
        # only a report-metadata extraction may assert a reporting period.
        if field == "reporting_period" and source_type != "metadata":
            return
        text_value = "" if value is None else str(value).strip()
        state = "blank" if text_value == "" else "present"
        key = (field, text_value, page, table_id)
        if key in seen:
            return
        seen.add(key)
        records.append({"source_type": source_type, "document_id": document_id, "page": page,
                        "section": section or "General", "table_id": table_id, "field": field,
                        "label": label.strip(), "value": value, "value_state": state, "confidence": 1.0})
        # A combined "office name and full address" field provides verified
        # evidence for either request; no document value is duplicated by hand.
        if field == "office_name" and FieldNormalizer.matches(label, "address"):
            add(source_type, page, section, label, value, table_id, forced_field="address")

    date_pattern = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
    for page_data in pages:
        page = page_data.get("page_number")
        lines = [line.strip() for line in (page_data.get("text") or "").splitlines() if line.strip()]
        for line_index, line in enumerate(lines):
            date = date_pattern.search(line)
            if date and FieldNormalizer.canonical(line) == "reporting_period":
                add("metadata", page, "Report metadata", line, date.group())
            if ":" in line:
                label, value = line.split(":", 1)
                add("metadata", page, "Report metadata", label, value)
                # PDF/DOCX cover pages often put a labelled office name/address
                # on one line and the physical address on the next. Preserve that
                # source line as a separate address fact, without guessing it.
                next_line = lines[line_index + 1] if line_index + 1 < len(lines) else ""
                if (FieldNormalizer.matches(label, "office_name") and FieldNormalizer.matches(label, "address")
                        and next_line and re.search(r"\d|sector|s?\.?\s*-?\s*\d|सैक्टर|सेक्टर|pin|पिन", next_line, re.IGNORECASE)):
                    add("metadata", page, "Report metadata", "address continuation", next_line)

    for table in tables:
        section = table.get("section") or table.get("heading") or "General"
        page = table.get("page_number")
        table_id = table.get("table_id")
        for item in table.get("structured_fields", []):
            source = "metadata" if FieldNormalizer.canonical(item.get("label", "")) in {"office_name", "address", "phone", "email", "reporting_period"} else "form"
            add(source, page, section, item.get("label", ""), item.get("value"), table_id)

        rows = table.get("rows", [])
        headers = table.get("headers", [])
        # Some Word/PDF forms use the first data rows as multi-row headers. Include
        # their labels when interpreting later numeric/value rows.
        context_by_column = {header: [] for header in headers}
        for row in rows:
            numeric_values = sum(1 for value in row.values() if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(value).strip()))
            if numeric_values == 0:
                for header, value in row.items():
                    if value and not re.fullmatch(r"\d+", str(value).strip()):
                        context_by_column.setdefault(header, []).append(str(value))
                continue
            numeric_sequence = [int(str(value).strip()) for value in row.values()
                                if re.fullmatch(r"\d+", str(value).strip())]
            # Forms often have a row of column numbers (1, 2, 3) immediately
            # before their values. Those numbers describe columns, not facts.
            if len(numeric_sequence) >= 2 and numeric_sequence == list(range(1, len(numeric_sequence) + 1)):
                continue
            for header, value in row.items():
                if value == "" or not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(value).strip()):
                    continue
                label = " | ".join([section, header] + context_by_column.get(header, [])[-2:])
                add("form" if len(headers) <= 4 else "table", page, section, label, value, table_id)
    return records