"""Deterministic numbers from report tables: region totals and report-vs-report comparison.

Why this exists
---------------
Retrieval + LLM is good at finding *one* value, but it cannot reliably
  * add values that live in different rows   ("तीनों क्षेत्रों को कुल कितने पत्र" -> 270+86+99)
  * line up the same metric across two reports ("2024 vs 2025")
Numbers for these answers must come from the document, and arithmetic must be
done in code, never by the LLM. This module reads the tables already saved by
StructuredTableStore at upload time, turns them into flat metric records, and
answers those two kinds of questions directly.

Record shape (one per value row in a report table):
    {document_id, filename, period, section_no, section, region, region_header,
     metric, metric_key, raw_value, value, is_percent}

Nothing here is document-specific: sections are detected from "N." rows,
regions from ‘क’/‘ख’/‘ग’ (Region A/B/C) rows, values from the last filled cell.
"""
from __future__ import annotations

import difflib
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from doc_scope import in_scope, get_scope

# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

_REGION_RE = re.compile(r"[‘'\"`]\s*(क|ख|ग)\s*[’'\"`]\s*क्षेत्र")
_REGION_EN_RE = re.compile(r"region\s*[‘'\"`]?\s*([abc])\b", re.I)
_REGION_EN_TO_HI = {"a": "क", "b": "ख", "c": "ग"}
_SECTION_NO_RE = re.compile(r"^\s*(\d{1,2})\s*[.)]?\s*$")
_NUMBER_RE = re.compile(r"^\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(%)?\s*\**\s*$")
_PERIOD_RE = re.compile(r"(\d{1,2})[._-](\d{1,2})[._-](\d{4})")
_TOKEN_RE = re.compile(r"[a-z0-9]+|[ऀ-ॿ]+")

# Words that carry no meaning for matching a metric label
_STOP = {
    # hindi
    "की", "का", "के", "में", "मे", "है", "हैं", "था", "थे", "और", "या", "कितने", "कितनी", "कितना",
    "क्या", "कौन", "इनमें", "इन", "से", "को", "पर", "तक", "गए", "गये", "गई", "दिए", "किये", "किए",
    "संख्या", "कुल", "तीनों", "सभी", "सब", "दोनों", "क्षेत्र", "क्षेत्रों", "रिपोर्ट", "तुलना", "करें", "करो",
    "बताओ", "बताइए", "वर्ष", "साल", "तिमाही",
    # english
    "the", "of", "in", "to", "from", "a", "an", "and", "or", "is", "are", "was", "were", "how", "many",
    "much", "what", "which", "no", "number", "total", "all", "three", "both", "region", "regions",
    "these", "this", "out", "against", "by", "for", "compare", "comparison", "vs", "report", "year",
    "quarter", "between", "with", "only",
    # roman hindi
    "ki", "ka", "ke", "me", "mein", "hai", "hain", "kitne", "kitni", "kitna", "kya", "kaun", "se", "ko",
    "kul", "teeno", "teenon", "sabhi", "sab", "dono", "kshetra", "kshetron", "tulna", "karo", "batao",
    "aur", "saal",
    # comparison / display words
    "अंतर", "antar", "difference", "फर्क", "farak", "growth", "वृद्धि", "badlav", "बदलाव", "बनाम", "banaam",
    "versus", "दिखाओ", "dikhao", "show", "graph", "chart", "ग्राफ", "चार्ट", "बीच", "beech",
    "ग्राफ़", "दो", "two", "reports", "please", "ye", "yeh", "these", "those", "do", "gaye", "gayi", "gaya",
    "गया", "ग्राफ़", "banao", "बनाओ", "dikhaiye", "दिखाइए", "between", "vich", "रिपोर्टों", "reporton",
}

# Roman Hindi / English words -> a Devanagari stem found in the report labels
_ALIASES = {
    "patra": "पत्र", "patr": "पत्र", "patron": "पत्र", "letter": "पत्र", "letters": "पत्र",
    "bheje": "भेजे", "bheja": "भेजे", "issued": "भेजे", "sent": "भेजे",
    "prapt": "प्राप्त", "received": "प्राप्त", "aaye": "प्राप्त", "aye": "प्राप्त",
    "uttar": "उत्तर", "reply": "उत्तर", "replied": "उत्तर",
    "hindi": "हिंदी", "angrezi": "अंग्रेजी", "english": "अंग्रेजी",
    "dwibhashi": "द्विभाषी", "bilingual": "द्विभाषी",
    "tippani": "टिप्पणियों", "noting": "टिप्पणियों", "notings": "टिप्पणियों",
    "prishth": "पृष्ठों", "pages": "पृष्ठों",
    "karyashala": "कार्यशाला", "workshop": "कार्यशाला", "workshops": "कार्यशाला",
    "prashikshit": "प्रशिक्षित", "trained": "प्रशिक्षित",
    "adhikari": "अधिकारियों", "officers": "अधिकारियों",
    "karmachari": "कर्मचारियों", "employees": "कर्मचारियों",
    "kagjat": "कागजात", "kagaz": "कागजात", "documents": "कागजात",
    "pratishat": "प्रतिशत", "percentage": "प्रतिशत", "percent": "प्रतिशत",
    "baithak": "बैठक", "meeting": "बैठक", "meetings": "बैठक",
    "samiti": "समिति", "committee": "समिति",
    "patro": "पत्र", "patron": "पत्र", "patra": "पत्र", "bheji": "भेजे", "bhejne": "भेजे",
    "tippaniyon": "टिप्पणियों", "tippaniyan": "टिप्पणियों", "tippaniya": "टिप्पणियों", "notes": "टिप्पणियों",
    "likhi": "लिखी", "likhe": "लिखी", "likha": "लिखी", "written": "लिखी",
    "kagzat": "कागजात", "kagazat": "कागजात", "kagjaat": "कागजात", "kaagjat": "कागजात", "papers": "कागजात",
    "jari": "जारी", "dhara": "धारा", "section": "धारा",
    "angreji": "अंग्रेजी", "dvibhashi": "द्विभाषी",
    "adhikariyon": "अधिकारियों", "karmchari": "कर्मचारियों", "karmchariyon": "कर्मचारियों",
    "karmachariyon": "कर्मचारियों", "staff": "कर्मचारियों",
    "uttaron": "उत्तर", "jawab": "उत्तर", "answered": "उत्तर",
    "karyashalaon": "कार्यशाला", "karyashalayen": "कार्यशाला",
}

# Direction words: "को / to" = letters sent to a region, "से / from" = received from it
_DIRECTION_TO = {"को", "ko", "to", "bheje", "भेजे", "issued", "sent"}
_DIRECTION_FROM = {"से", "se", "from", "प्राप्त", "prapt", "received"}

_COMPARE_RE = re.compile(
    r"(?<![\wऀ-ॿ])(compare|comparison|vs\.?|versus|तुलना|tulna|बनाम|banaam|अंतर|antar|difference|फर्क|farak|growth|वृद्धि|badlav|बदलाव)(?![\wऀ-ॿ])",
    re.I,
)
_ALL_REGIONS_RE = re.compile(
    r"(?<![\wऀ-ॿ])(तीनों|तीनो|सभी|सब|दोनों|all|three|both|teeno|teenon|sabhi|sab|dono)(?![\wऀ-ॿ])",
    re.I,
)
_REGION_WORD_RE = re.compile(r"क्षेत्र|kshetr|kshetra|region", re.I)
_TOTAL_WORDS = {"कुल", "total", "kul", "टोटल"}
_PERCENT_WORDS = {"प्रतिशत", "percent", "percentage", "pratishat", "%"}
# "region A", "क क्षेत्र", "‘क’ क्षेत्र", "ka kshetra"
_NAMED_REGION_RE = re.compile(
    r"region\s*[‘'\"`]?\s*([abc])\b|(?<![ऀ-ॿ])[‘'\"`]?\s*(क|ख|ग)\s*[’'\"`]?\s*क्षेत्र|\b(ka|kha|ga)\s+kshetr",
    re.I,
)
_ROMAN_REGION = {"ka": "क", "kha": "ख", "ga": "ग"}


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _content_tokens(text: str) -> List[str]:
    out = []
    for tok in _tokens(text):
        if tok in _STOP or len(tok) < 2 or tok.isdigit():
            continue
        out.append(_ALIASES.get(tok, tok))
    return out


def _stem_match(a: str, b: str) -> bool:
    """Loose Hindi-aware match: पत्र ~ पत्रों, कार्यशाला ~ कार्यशालाओं."""
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 3 and long_.startswith(short)


def _overlap(query_toks: Iterable[str], label_toks: Iterable[str]) -> int:
    label_toks = list(label_toks)
    return sum(1 for q in set(query_toks) if any(_stem_match(q, l) for l in label_toks))


def _short_label(label: str) -> str:
    """Hindi part of a bilingual label: 'भेजे गए पत्रों की कुल संख्या (Total no...)' -> 'भेजे गए पत्रों की कुल संख्या'."""
    head = re.split(r"\s*\(", label, maxsplit=1)[0].strip(" :-/")
    return head or label.strip()


def _metric_key(label: str) -> str:
    """Stable key for matching the same row across two reports."""
    english = re.findall(r"\(([^()]*[A-Za-z][^()]*)\)", label)
    base = english[-1] if english else _short_label(label)
    return re.sub(r"[^a-z0-9ऀ-ॿ]+", " ", base.lower()).strip()


def _parse_value(raw: str) -> Tuple[Optional[float], bool]:
    m = _NUMBER_RE.match(raw or "")
    if not m:
        return None, False
    return float(m.group(1).replace(",", "")), bool(m.group(2))


def _fmt(v: Optional[float], is_percent: bool = False) -> str:
    if v is None:
        return "—"
    s = f"{v:.2f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return f"{s}%" if is_percent else s


def _period_of(document_id: str, filename: str, tables: List[Dict[str, Any]]) -> str:
    for text in (filename or "", document_id or ""):
        m = _PERIOD_RE.search(text)
        if m:
            return f"{int(m.group(1)):02d}.{int(m.group(2)):02d}.{m.group(3)}"
    for t in tables:
        m = _PERIOD_RE.search(t.get("original_table_text") or "")
        if m:
            return f"{int(m.group(1)):02d}.{int(m.group(2)):02d}.{m.group(3)}"
    years = re.findall(r"20\d{2}", filename or document_id or "")
    return years[0] if years else (filename or document_id)


def _period_sort_key(period: str) -> Tuple[int, int, int]:
    m = _PERIOD_RE.search(period or "")
    if m:
        return int(m.group(3)), int(m.group(2)), int(m.group(1))
    y = re.search(r"20\d{2}", period or "")
    return (int(y.group(0)) if y else 0, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Table -> records
# ─────────────────────────────────────────────────────────────────────────────

def _matrix(table: Dict[str, Any]) -> List[List[str]]:
    headers = list(table.get("headers") or [])
    rows = [[str(r.get(h, "") or "") for h in headers] for r in table.get("rows") or []]
    # Generated names like "Column 3" are placeholders for empty header cells
    head = ["" if re.fullmatch(r"Column \d+", h or "") else h for h in headers]
    return [head] + rows


def records_from_table(table: Dict[str, Any], period: str) -> List[Dict[str, Any]]:
    if table.get("extraction_method") not in {"docx_xml", "pdfplumber"}:
        return []  # form/period fallbacks duplicate the same numbers
    matrix = _matrix(table)
    # A contents/list table ("1 | title | : | author | 7") looks like a run of numbered
    # rows that each end in a number. Report tables have headed sections instead.
    numbered = [r for r in matrix if r and _SECTION_NO_RE.match(r[0].strip() or "x")]
    numbered_with_value = [r for r in numbered if _NUMBER_RE.match((r[-1] or "").strip() or "x")]
    if len(numbered) >= 5 and len(numbered_with_value) >= 0.6 * len(numbered):
        return []
    out: List[Dict[str, Any]] = []
    section_no, section, region, region_header = "", "", None, ""
    for cells in matrix:
        cells = [c.strip() for c in cells]
        filled = [c for c in cells if c and c not in {":", "："}]
        if not filled:
            continue
        # Section row: "3." + heading
        if len(cells) > 1 and _SECTION_NO_RE.match(cells[0]):
            section_no = _SECTION_NO_RE.match(cells[0]).group(1)
            section = next((c for c in cells[1:] if c and c != ":"), "")
            region, region_header = None, ""
            # A section row can itself carry a value (e.g. "7. meeting date : 23.09.2024")
            rest = [c for c in cells[1:] if c and c != ":"]
            if len(rest) >= 2 and rest[-1] != rest[0]:
                raw = rest[-1]
                val, pct = _parse_value(raw)
                out.append(_record(table, period, section_no, section, None, "", rest[0], raw, val, pct))
            continue
        label = max(filled, key=len)
        # Region row: "‘क’ क्षेत्र से / From Region ‘A’"
        m = _REGION_RE.search(label)
        m_en = _REGION_EN_RE.search(label)
        if (m or m_en) and len(filled) == 1:
            region = m.group(1) if m else _REGION_EN_TO_HI[m_en.group(1).lower()]
            region_header = label
            continue
        if len(filled) < 2:
            continue
        raw = filled[-1]
        if raw == label:
            continue
        val, pct = _parse_value(raw)
        out.append(_record(table, period, section_no, section, region, region_header, label, raw, val, pct))
    return out


def _record(table, period, section_no, section, region, region_header, label, raw, val, pct):
    return {
        "document_id": table.get("document_id"),
        "filename": table.get("filename"),
        "table_id": table.get("table_id"),
        "page_number": table.get("page_number"),
        "period": period,
        "section_no": section_no,
        "section": section,
        "region": region,
        "region_header": region_header,
        "metric": label,
        "metric_short": _short_label(label),
        "metric_key": _metric_key(label),
        "raw_value": raw,
        "value": val,
        "is_percent": pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class ReportMetrics:
    def __init__(self, table_store, doc_type_of=None):
        self.table_store = table_store
        # Optional: document_id -> "report" | "magazine" (upload category). Magazines are skipped.
        self.doc_type_of = doc_type_of

    # -- loading ----------------------------------------------------------------
    def records(self, document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        tables = self.table_store.get_tables(None)
        by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for t in tables:
            doc = t.get("document_id")
            if document_ids and doc not in document_ids:
                continue
            if not in_scope(doc):
                continue
            if self.doc_type_of and self.doc_type_of(doc) == "magazine":
                continue
            by_doc.setdefault(doc, []).append(t)
        out: List[Dict[str, Any]] = []
        for doc, doc_tables in by_doc.items():
            period = _period_of(doc, doc_tables[0].get("filename", ""), doc_tables)
            for t in doc_tables:
                out.extend(records_from_table(t, period))
        return out

    # -- scoring ----------------------------------------------------------------
    @staticmethod
    def _score(query: str, rec: Dict[str, Any]) -> float:
        q = _content_tokens(query)
        if not q:
            return 0.0
        label = _content_tokens(rec["metric"])
        context = _content_tokens(f'{rec["section"]} {rec["region_header"]}')
        score = 2.0 * _overlap(q, label) + 0.5 * _overlap(q, context)
        # Direction: "को/to" means sent to the region, "से/from" means received from it
        raw_q = set(_tokens(query))
        ctx = set(_tokens(f'{rec["section"]} {rec["region_header"]}'))
        label_raw = set(_tokens(rec["metric"]))
        if raw_q & _TOTAL_WORDS and label_raw & _TOTAL_WORDS:
            score += 1.5
        wants_pct = bool(raw_q & _PERCENT_WORDS) or "%" in (query or "")
        if rec.get("is_percent"):
            score += 1.5 if wants_pct else -1.5
        if raw_q & _DIRECTION_TO and ctx & _DIRECTION_TO:
            score += 1.0
        if raw_q & _DIRECTION_FROM and ctx & _DIRECTION_FROM:
            score += 1.0
        return score

    @staticmethod
    def named_regions(query: str) -> set:
        found = set()
        for m in _NAMED_REGION_RE.finditer(query or ""):
            if m.group(1):
                found.add(_REGION_EN_TO_HI[m.group(1).lower()])
            elif m.group(2):
                found.add(m.group(2))
            elif m.group(3):
                found.add(_ROMAN_REGION[m.group(3).lower()])
        return found

    # -- intents ------------------------------------------------------------------
    @staticmethod
    def is_comparison(query: str, several_selected: bool = False) -> bool:
        """Two periods named ("2024 vs 2025"), or a compare word while the user has
        explicitly selected several documents. A bare "अंतर" with nothing selected
        is usually a normal question ("हिंदी और अंग्रेजी में अंतर"), not a report comparison."""
        periods = set(re.findall(r"20\d{2}", query or ""))
        if len(periods) >= 2:
            return True
        return several_selected and bool(_COMPARE_RE.search(query or ""))

    @staticmethod
    def is_region_total(query: str) -> bool:
        q = query or ""
        return bool(_ALL_REGIONS_RE.search(q) and _REGION_WORD_RE.search(q)) or "क्षेत्रों" in q and "कुल" in q

    # -- region total -------------------------------------------------------------
    def region_total(self, query: str, document_ids: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        recs = [r for r in self.records(document_ids) if r["region"] and r["value"] is not None]
        if not recs:
            return None
        # Pick the metric (section + label) that best matches the question
        groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for r in recs:
            groups.setdefault((r["document_id"], r["section_no"], r["metric_key"]), []).append(r)
        scored = []
        for key, rows in groups.items():
            regions = {r["region"] for r in rows}
            if len(regions) < 2:
                continue
            scored.append((max(self._score(query, r) for r in rows), key, rows))
        if not scored:
            return None
        scored.sort(key=lambda s: s[0], reverse=True)
        best_score, _, rows = scored[0]
        if best_score < 2.0:
            return None  # nothing clearly matches; let normal RAG answer
        # Same metric in several selected documents: answer for each of them
        best_key = scored[0][1][2]
        best = [s for s in scored if s[0] == best_score and s[1][2] == best_key]
        parts, evidence, results = [], [], []
        for _, (doc, sec, _k), grp in best:
            grp = sorted(grp, key=lambda r: "कखग".find(r["region"]))
            is_pct = any(r["is_percent"] for r in grp)
            if is_pct:
                breakdown = ", ".join(f'{r["region"]}: {_fmt(r["value"], True)}' for r in grp)
                text = (f'**{grp[0]["metric_short"]}** — {breakdown}. '
                        f"प्रतिशत को जोड़ा नहीं जाता, इसलिए क्षेत्रवार मान दिए गए हैं।")
                total = None
            else:
                total = sum(r["value"] for r in grp)
                breakdown = " + ".join(f'{r["region"]}: {_fmt(r["value"])}' for r in grp)
                regions = ", ".join(r["region"] for r in grp)
                text = (f'**{grp[0]["metric_short"]}** ({regions} क्षेत्र): **{_fmt(total)}**\n\n'
                        f"गणना: {breakdown} = {_fmt(total)}")
            if len(best) > 1:
                text = f'📄 {grp[0]["filename"]} ({grp[0]["period"]}):\n{text}'
            parts.append(text)
            evidence.extend(grp)
            results.append({"document_id": doc, "section_no": sec, "metric": grp[0]["metric_short"],
                            "total": total, "values": {r["region"]: r["value"] for r in grp}})
        return {"kind": "region_total", "answer": "\n\n".join(parts), "results": results, "evidence": evidence}

    # -- comparison ---------------------------------------------------------------
    def compare(self, query: str, document_ids: Optional[List[str]] = None,
                max_rows: int = 40) -> Optional[Dict[str, Any]]:
        recs = [r for r in self.records(document_ids) if r["value"] is not None]
        periods = sorted({r["period"] for r in recs}, key=_period_sort_key)
        if len(periods) < 2:
            return None
        docs_by_period = {r["period"]: r["document_id"] for r in recs}

        # Join the same metric across reports: (section, region, metric key); fuzzy fallback on the key
        rows: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for r in recs:
            key = (r["section_no"], r["region"] or "", r["metric_key"])
            if key not in rows:
                # Fuzzy join only onto a row that has no value yet for this report.
                # "...Noting's in Hindi" and "...Noting's in English" are 0.9 alike;
                # joining them put the English numbers under the Hindi row.
                best_ratio, match = 0.0, None
                for k, row in rows.items():
                    if k[0] != key[0] or k[1] != key[1] or r["period"] in row["values"]:
                        continue
                    ratio = difflib.SequenceMatcher(None, k[2], key[2]).ratio()
                    if ratio >= 0.85 and ratio > best_ratio:
                        best_ratio, match = ratio, k
                key = match or key
            row = rows.setdefault(key, {"section_no": r["section_no"], "section": r["section"],
                                        "region": r["region"], "metric": r["metric_short"],
                                        "is_percent": r["is_percent"], "values": {}, "evidence": []})
            row["values"][r["period"]] = r["value"]
            row["evidence"].append(r)

        table = [row for row in rows.values() if sum(p in row["values"] for p in periods) >= 2]
        if not table:
            return None

        wanted_regions = self.named_regions(query)
        if wanted_regions:
            narrowed = [row for row in table if row["region"] in wanted_regions]
            table = narrowed or table

        # Narrow to what the question asks about, if it names something specific
        q_tokens = _content_tokens(re.sub(r"20\d{2}|\d{1,2}[._-]\d{1,2}[._-]\d{4}", " ", query or ""))
        if q_tokens:
            scored = [(max(self._score(query, e) for e in row["evidence"]), row) for row in table]
            top = max(s for s, _ in scored)
            if top < 2.0:
                # The question names something that is not a row in these tables
                # (e.g. "हिंदी और अंग्रेजी में अंतर"): leave it to normal RAG.
                return None
            table = [row for s, row in scored if s >= max(2.0, top - 1.0)]

        first, last = periods[0], periods[-1]
        out_rows = []
        for row in table[:max_rows]:
            a, b = row["values"].get(first), row["values"].get(last)
            change = None if a is None or b is None else b - a
            pct = None
            if change is not None and a not in (None, 0) and not row["is_percent"]:
                pct = round(change / a * 100, 2)
            out_rows.append({
                "section_no": row["section_no"], "section": row["section"], "region": row["region"],
                "metric": row["metric"], "is_percent": row["is_percent"],
                "values": {p: row["values"].get(p) for p in periods},
                "change": change, "change_pct": pct,
            })

        # Chart: one series per report, one bar group per metric (percent rows kept separate)
        chart_rows = [r for r in out_rows if not r["is_percent"]][:12] or out_rows[:12]
        chart = {
            "type": "bar",
            "labels": [(f'{r["region"]} क्षेत्र – ' if r["region"] else "") + r["metric"] for r in chart_rows],
            "series": [{"name": p, "values": [r["values"].get(p) for r in chart_rows]} for p in periods],
            "unit": "%" if chart_rows and chart_rows[0]["is_percent"] else "",
        }

        lines = [f"**{first}** और **{last}** की रिपोर्ट की तुलना ({len(out_rows)} मद):", ""]
        for r in out_rows[:15]:
            name = (f'{r["region"]} क्षेत्र – ' if r["region"] else "") + r["metric"]
            a, b = r["values"].get(first), r["values"].get(last)
            delta = ""
            if r["change"] is not None:
                sign = "+" if r["change"] > 0 else ""
                delta = f' ({sign}{_fmt(r["change"], r["is_percent"])}'
                delta += f', {sign}{_fmt(r["change_pct"])}%)' if r["change_pct"] is not None else ")"
            lines.append(f'- {name}: {_fmt(a, r["is_percent"])} → {_fmt(b, r["is_percent"])}{delta}')
        if len(out_rows) > 15:
            lines.append(f"- … और {len(out_rows) - 15} मद (पूरी तालिका नीचे)")

        evidence = [e for row in table[:max_rows] for e in row["evidence"]]
        return {"kind": "comparison", "answer": "\n".join(lines), "periods": periods,
                "documents": {p: docs_by_period.get(p) for p in periods},
                "comparison": out_rows, "chart_data": chart, "evidence": evidence}

    # -- entry point used by the chat route ------------------------------------------
    def try_answer(self, query: str, document_ids: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        try:
            scope = get_scope()
            several = bool((document_ids and len(document_ids) > 1) or (scope and len(scope) > 1))
            if self.is_comparison(query, several):
                res = self.compare(query, document_ids)
                if res:
                    return res
            if self.is_region_total(query):
                return self.region_total(query, document_ids)
        except Exception as exc:  # never break chat because of this layer
            print(f"[ReportMetrics] skipped: {exc}")
        return None


def evidence_to_sources(evidence: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    """Map records to the source format the frontend already renders."""
    sources, seen = [], set()
    for e in evidence:
        key = (e["document_id"], e["section_no"], e["region"], e["metric_key"])
        if key in seen:
            continue
        seen.add(key)
        where = " › ".join(x for x in [f'धारा/मद {e["section_no"]}' if e["section_no"] else "",
                                       f'{e["region"]} क्षेत्र' if e["region"] else "",
                                       e["metric_short"]] if x)
        text = f'{where}: {e["raw_value"]}'
        sources.append({
            "quote": text, "text": text, "highlightText": text,
            "pageNumber": e.get("page_number") or 1, "page_number": e.get("page_number") or 1,
            "chunkId": e.get("table_id", ""), "chunk_id": e.get("table_id", ""),
            "docName": e["document_id"], "document_name": e["document_id"], "document_id": e["document_id"],
            "document_type": "report", "content_type": "table", "collection_type": "table",
            "section": e["section"], "table_id": e.get("table_id"), "relevanceScore": 100,
        })
        if len(sources) >= limit:
            break
    return sources