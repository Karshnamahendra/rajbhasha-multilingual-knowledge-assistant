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
from index_knowledge_layer import romanize_generic
from region_utils import REGION_EN_TO_HI as _REGION_EN_TO_HI

# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

_REGION_RE = re.compile(r"[‘'\"`]\s*(क|ख|ग)\s*[’'\"`]\s*क्षेत्र")
_REGION_EN_RE = re.compile(r"region\s*[‘'\"`]?\s*([abc])\b", re.I)
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
    "हिन्दी": "हिंदी", "अंग्रेज़ी": "अंग्रेजी",
    "patra": "पत्र", "patr": "पत्र", "patron": "पत्र", "letter": "पत्र", "letters": "पत्र",
    "bheje": "भेजे", "bheja": "भेजे", "issued": "भेजे", "sent": "भेजे",
    "prapt": "प्राप्त", "received": "प्राप्त", "aaye": "प्राप्त", "aye": "प्राप्त",
    "uttar": "उत्तर", "reply": "उत्तर", "replied": "उत्तर",
    "hindi": "हिंदी", "angrezi": "अंग्रेजी", "english": "अंग्रेजी",
    "dwibhashi": "द्विभाषी", "bilingual": "द्विभाषी",
    "tippani": "टिप्पणियों", "noting": "टिप्पणियों", "notings": "टिप्पणियों",
    "prishth": "पृष्ठों", "pages": "पृष्ठों",
    "karyashala": "कार्यशाला", "workshop": "कार्यशाला", "workshops": "कार्यशाला",
    "prashikshit": "प्रशिक्षित", "trained": "प्रशिक्षित", "train": "प्रशिक्षित",
    "adhikari": "अधिकारियों", "officers": "अधिकारी",
    "karmachari": "कर्मचारियों", "employee": "कर्मचारी", "employees": "कर्मचारी",
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
    "correspondence": "पत्राचार", "original": "मूल", "note": "टिप्पणियों",
    "nhi": "नहीं", "nahi": "नहीं", "requirement": "अपेक्षित", "required": "अपेक्षित",
    "need": "अपेक्षित", "needed": "अपेक्षित", "expected": "अपेक्षित",
    "karyashalaon": "कार्यशाला", "karyashalayen": "कार्यशाला",
}

# Direction words: "को / to" = letters sent to a region, "से / from" = received from it
_DIRECTION_TO = {"को", "ko", "to", "bheje", "भेजे", "issued", "sent"}
_DIRECTION_FROM = {"से", "se", "from", "प्राप्त", "prapt", "receive", "received", "aaye", "aye"}
_RECEIPT_TERMS = {"प्राप्त", "prapt", "receive", "received", "aaye", "aye"}
_ANSWER_TERMS = {"उत्तर", "uttar", "जवाब", "jawab", "reply", "replies", "replied",
                 "answer", "answers", "answered", "response", "responses"}
_NO_REPLY_RE = re.compile(
    r"(?:reply|replies|answer|answers|उत्तर|जवाब|uttar|jawab).{0,64}(?:not|required|requirement|need|expected|no|nhi|nahi|नहीं|नही|अपेक्षित|आवश्यक)|"
    r"(?:not|required|requirement|need|expected|no|nhi|nahi|नहीं|नही|अपेक्षित|आवश्यक).{0,64}(?:reply|replies|answer|answers|उत्तर|जवाब|uttar|jawab)",
    re.I,
)
_HINDI_TERMS = {"hindi", "हिंदी", "हिन्दी"}
_ENGLISH_TERMS = {"english", "अंग्रेजी", "अंग्रेज़ी"}


def _language_near_action(query: str, action_terms: set) -> Optional[str]:
    """Return a language explicitly attached to an action (e.g. Hindi replies)."""
    tokens = _tokens(query)
    actions = [i for i, token in enumerate(tokens) if token in action_terms]
    candidates = []
    for i, token in enumerate(tokens):
        language = "hindi" if token in _HINDI_TERMS else "english" if token in _ENGLISH_TERMS else None
        if language:
            nearest_actions = [(abs(i - action), 0 if i >= action else 1) for action in actions]
            distance, side_priority = min(nearest_actions, default=(999, 1))
            if distance <= 5:
                candidates.append((distance, side_priority, language))
    if not candidates:
        return None
    # If a reply action has a language on both sides, prefer the language
    # after the action ("replied in English") over the received language before it.
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]

_COMPARE_RE = re.compile(
    r"(?<![\wऀ-ॿ])(compare|comparison|vs\.?|versus|increase|increased|increasing|decrease|decreased|decreasing|change|changed|तुलना|tulna|बनाम|banaam|अंतर|antar|difference|फर्क|farak|growth|वृद्धि|vriddhi|badlav|बदलाव)(?![\wऀ-ॿ])",
    re.I,
)
_MULTI_PERIOD_RE = re.compile(
    r"(?<![\wऀ-ॿ])(both|two|multiple|dono|dono\s+(?:quarter|timahi|report)|"
    r"all\s+(?:selected\s+)?(?:periods?|quarters?|reports?|files?)|"
    r"whole\s+year|full\s+year|entire\s+year|yearly|throughout\s+(?:the\s+)?year|"
    r"(?:\d+|four|चार|chaar)\s+quarters?|saare\s+quarters?|poore\s+saal|pure\s+saal|saal\s+bhar|"
    r"दोनों|दो\s+तिमाहियों|चार\s+तिमाहियों|सभी\s+तिमाहियों|पूरे\s+साल|पूरे\s+वर्ष|"
    r"दो\s+रिपोर्टों|दो\s+रिपोर्ट|दो\s+अवधियों)(?![\wऀ-ॿ])",
    re.I,
)
_ALL_REGIONS_RE = re.compile(
    r"(?<![\wऀ-ॿ])(तीनों|तीनो|सभी|सब|दोनों|all|three|both|teeno|teenon|sabhi|sab|dono)(?![\wऀ-ॿ])",
    re.I,
)
_REGION_WORD_RE = re.compile(r"क्षेत्र|kshetr|kshetra|region", re.I)
_TOTAL_WORDS = {"कुल", "total", "kul", "टोटल"}
_PERCENT_WORDS = {"प्रतिशत", "percent", "percentage", "pratishat", "%"}
_COUNT_INTENT_RE = re.compile(
    r"(?<![\wऀ-ॿ])(how\s+many|number\s+of|count|कितने|कितनी|कितना|संख्या|kul|kitne|kitni|kitna)(?![\wऀ-ॿ])",
    re.I,
)
# "region A", "क क्षेत्र", "‘क’ क्षेत्र", "ka kshetra"
_NAMED_REGION_RE = re.compile(
    r"regions?\s*[‘'\"`]?\s*(?P<en_after>[abc])\b|"
    r"(?<![a-z])(?P<en_before>[abc])\s+(?:regions?|zones?)\b|"
    r"(?<![ऀ-ॿ])[‘'\"`]?\s*(?P<hi>[कखग])\s*[’'\"`]?\s*क्षेत्र|"
    r"(?<![a-z])[‘'\"`]?\s*(?P<roman_before>ka|kha|kh|k|ga|g)\b[‘'\"`]?\s+(?:kshetr(?:a)?|regions?|zones?)\b|"
    r"(?:kshetr(?:a)?|regions?|zones?)\s+[‘'\"`]?\s*(?P<roman_after>ka|kha|kh|k|ga|g)\b",
    re.I,
)
_REGION_ENUM_EN_RE = re.compile(
    r"\bregions?\s+((?:[‘’'\"`]?\s*[abc]\s*[‘’'\"`]?\s*(?:,|and|&)\s*)*"
    r"[‘’'\"`]?\s*[abc]\s*[‘’'\"`]?)", re.I)
_REGION_ENUM_HI_RE = re.compile(
    r"((?:[‘’'\"`]?\s*[कखग]\s*[’'\"`]?\s*(?:,|और|and)\s*)+"
    r"[‘’'\"`]?\s*[कखग]\s*[’'\"`]?)\s*क्षेत्र", re.I)
_REGION_ENUM_ROMAN_RE = re.compile(
    r"(?<![a-z])((?:kha|kh|ka|k|ga|g)\s*(?:,|aur|and|&)\s*"
    r"(?:kha|kh|ka|k|ga|g)(?:\s*(?:,|aur|and|&)\s*(?:kha|kh|ka|k|ga|g))*)(?![a-z])", re.I)
_ROMAN_REGION = {"k": "क", "ka": "क", "kh": "ख", "kha": "ख", "g": "ग", "ga": "ग"}
_SECTION_REF_RE = re.compile(r"(?<![\wऀ-ॿ])(?:section|sec\.?|धारा|dhara)\s*(\d{1,2})(?:\s*\(\s*(\d{1,2})\s*\))?", re.I)


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


def _query_matches_label(query_token: str, label_token: str) -> bool:
    """Match stems and modest Roman-Hindi spelling variation across scripts."""
    if _stem_match(query_token, label_token):
        return True
    query_token, label_token = query_token.casefold(), label_token.casefold()
    if re.search(r"[a-z]", query_token) and re.search(r"[ऀ-ॿ]", label_token):
        label_token = romanize_generic(label_token).casefold()
    elif re.search(r"[ऀ-ॿ]", query_token) and re.search(r"[a-z]", label_token):
        query_token = romanize_generic(query_token).casefold()
    else:
        return False
    query_token = re.sub(r"[^a-z0-9]", "", query_token)
    label_token = re.sub(r"[^a-z0-9]", "", label_token)
    if min(len(query_token), len(label_token)) < 5 or abs(len(query_token) - len(label_token)) > 3:
        return False
    return difflib.SequenceMatcher(None, query_token, label_token).ratio() >= 0.78


def _overlap(query_toks: Iterable[str], label_toks: Iterable[str]) -> int:
    label_toks = list(label_toks)
    return sum(1 for q in set(query_toks) if any(_query_matches_label(q, l) for l in label_toks))


def _best_section_scope(query: str, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    """Prefer a distinctly named table section, without depending on section numbers."""
    query_terms = _content_tokens(query or "")
    section_terms = [set(_content_tokens(row.get("section") or "")) for row in rows]
    unique_sections = {str(row.get("section") or ""): terms
                       for row, terms in zip(rows, section_terms)}
    section_frequency: Dict[str, int] = {}
    for terms in unique_sections.values():
        for term in terms:
            section_frequency[term] = section_frequency.get(term, 0) + 1
    scores = [
        sum(max((1.0 / section_frequency[term] for term in terms if _query_matches_label(qterm, term)), default=0.0)
            for qterm in set(query_terms))
        for terms in section_terms
    ]
    strongest = max(scores, default=0.0)
    if strongest >= 1.0:
        return [row for row, score in zip(rows, scores) if score >= strongest - 0.25], True
    return rows, False


def _scope_explicit_section(rows: List[Dict[str, Any]], reference: re.Match) -> List[Dict[str, Any]]:
    """Resolve a section reference against report numbering or cited law numbering."""
    wanted_no, wanted_sub = reference.group(1), reference.group(2)
    if wanted_sub:
        citation = re.compile(
            rf"(?<!\d)(?:section|धारा)\s*{re.escape(wanted_no)}\s*\(\s*{re.escape(wanted_sub)}\s*\)",
            re.I,
        )
        cited = [row for row in rows if citation.search(
            f'{row.get("section") or ""} {row.get("metric") or row.get("metric_short") or ""}')]
        if cited:
            return cited
    numbered = [row for row in rows if str(row.get("section_no") or "") == wanted_no]
    if wanted_sub and numbered:
        exact_sub = re.compile(rf"(?<!\d){re.escape(wanted_no)}\s*\(\s*{re.escape(wanted_sub)}\s*\)")
        exact = [row for row in numbered if exact_sub.search(str(row.get("section") or ""))]
        return exact or numbered
    return numbered


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
    method = table.get("extraction_method")
    if method not in {
        "docx_xml", "pdfplumber", "key_value_form", "period_field_pattern"
    }:
        return []
    if method in {"key_value_form", "period_field_pattern"}:
        # Fallback extractors already provide explicit label/value fields. Parse
        # those directly so words such as “Region A” inside a field label are
        # treated as context, not mistaken for a standalone region header row.
        headers = list(table.get("headers") or [])
        if len(headers) < 2:
            return []
        out = []
        for row in table.get("rows") or []:
            label = str(row.get(headers[0]) or "").strip()
            raw = str(row.get(headers[-1]) or "").strip()
            value, is_percent = _parse_value(raw)
            if not label or value is None:
                continue
            out.append(_record(table, period, "", table.get("heading") or "", None, "",
                               label, raw, value, is_percent))
        return out
    matrix = _matrix(table)
    # A contents/list table ("1 | title | : | author | 7") looks like a run of numbered
    # rows that each end in a number. Report tables have headed sections instead.
    numbered = [r for r in matrix if r and _SECTION_NO_RE.match(r[0].strip() or "x")]
    numbered_with_value = [r for r in numbered if _NUMBER_RE.match((r[-1] or "").strip() or "x")]
    if len(numbered) >= 5 and len(numbered_with_value) >= 0.6 * len(numbered):
        return []
    out: List[Dict[str, Any]] = []
    column_labels: Dict[int, str] = {}
    column_groups: Dict[int, str] = {}
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
            column_labels.clear()
            column_groups.clear()
            # A section row can itself carry a value (e.g. "7. meeting date : 23.09.2024")
            rest = [c for c in cells[1:] if c and c != ":"]
            if len(rest) >= 2 and rest[-1] != rest[0]:
                raw = rest[-1]
                val, pct = _parse_value(raw)
                out.append(_record(table, period, section_no, section, None, "", rest[0], raw, val, pct))
            continue
        # Some report tables put several measures in columns under a blank
        # generated header. Keep the human-readable column labels so each cell
        # can be indexed as its own metric instead of retaining only the last
        # value in the row.
        if not any(_REGION_RE.search(c) or _REGION_EN_RE.search(c) for c in cells):
            text_cells = [(i, c) for i, c in enumerate(cells) if c and c not in {":", "："}
                          and not _NUMBER_RE.match(c)]
            numeric_cells = [(i, c) for i, c in enumerate(cells) if _NUMBER_RE.match(c)]
            if len(text_cells) >= 2 and len(numeric_cells) <= 1:
                if all(re.fullmatch(r"\(\s*\d+\s*\)", value) for _, value in text_cells):
                    continue  # column numbering row, not labels
                category_cells = [(i, value) for i, value in text_cells if i >= 2]
                parent = column_labels.get(min((i for i, _ in category_cells), default=-1), "")
                for i, value in text_cells:
                    if i < 1:
                        continue
                    if category_cells and i >= 2 and parent:
                        column_labels[i] = f"{parent} – {value}"
                        column_groups[i] = parent
                    else:
                        column_labels[i] = value
                        if i >= 2:
                            column_groups[i] = value
                continue

        region_cell = next(((i, _REGION_RE.search(c) or _REGION_EN_RE.search(c))
                            for i, c in enumerate(cells)
                            if _REGION_RE.search(c) or _REGION_EN_RE.search(c)), None)
        if region_cell:
            region_index, match = region_cell
            region = match.group(1) if match.re is _REGION_RE else _REGION_EN_TO_HI[match.group(1).lower()]
            region_header = cells[region_index]
            indexed_values = []
            for i, raw_value in enumerate(cells):
                parsed, is_percent = _parse_value(raw_value)
                if parsed is None:
                    continue
                metric_label = column_labels.get(i)
                if metric_label:
                    indexed_values.append((metric_label, raw_value, parsed, is_percent))
            if indexed_values:
                for metric_label, raw_value, parsed, is_percent in indexed_values:
                    out.append(_record(table, period, section_no, section, region, region_header,
                                       metric_label, raw_value, parsed, is_percent))
                continue
        # Header-driven tables without a region column can have several
        # numeric values per row (for example counts split by staff category).
        # Preserve each value with its dynamically extracted column label.
        numeric_values = [(i, raw) for i, raw in enumerate(cells) if _NUMBER_RE.match(raw)]
        if len(numeric_values) >= 2:
            indexed_values = []
            for i, raw in numeric_values:
                metric_label = column_labels.get(i)
                value, pct = _parse_value(raw)
                if metric_label and value is not None:
                    indexed_values.append((i, metric_label, raw, value, pct))
            if indexed_values:
                for i, metric_label, raw, value, pct in indexed_values:
                    out.append(_record(table, period, section_no, section, None, "",
                                       metric_label, raw, value, pct,
                                       aggregation_group=column_groups.get(i)))
                continue
        label = max(filled, key=len)
        # Region row: "‘क’ क्षेत्र से / From Region ‘A’"
        m = _REGION_RE.search(label)
        m_en = _REGION_EN_RE.search(label)
        if (m or m_en) and len(filled) == 1:
            region = m.group(1) if m else _REGION_EN_TO_HI[m_en.group(1).lower()]
            region_header = label
            continue
        # Numbering rows (1 : 2 : 3 : 4) are column guides, not report data.
        if len(filled) > 1 and all(_NUMBER_RE.match(c) for c in filled):
            continue
        if len(filled) < 2:
            continue
        raw = filled[-1]
        if raw == label:
            continue
        val, pct = _parse_value(raw)
        out.append(_record(table, period, section_no, section, region, region_header, label, raw, val, pct))
    return out


def _record(table, period, section_no, section, region, region_header, label, raw, val, pct,
            aggregation_group=None):
    # Some report tables encode the region inside each metric label instead of
    # as a separate header row (e.g. "‘क’ क्षेत्र (letters sent in Hindi %)").
    # Normalize that schema so matching, comparison tables, and charts can use
    # region + metric as separate fields without depending on a report template.
    inline_region = _REGION_RE.search(label or "") or _REGION_EN_RE.search(label or "")
    if region is None and inline_region:
        region = (inline_region.group(1) if inline_region.re is _REGION_RE
                  else _REGION_EN_TO_HI[inline_region.group(1).lower()])
        region_header = inline_region.group(0)
        remainder = (label[:inline_region.start()] + " " + label[inline_region.end():]).strip(" :-–—()'‘’\"`")
        # If the region is the entire visible label and the metric follows in
        # parentheses, retain that metric text instead of turning it blank.
        label = remainder or label
    metric_short = _short_label(label)
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
        "metric_short": metric_short,
        "metric_key": _metric_key(label),
        "raw_value": raw,
        "value": val,
        "is_percent": pct,
        "aggregation_group": aggregation_group,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class ReportMetrics:
    def __init__(self, table_store, doc_type_of=None, embed_fn=None):
        self.table_store = table_store
        # Optional: document_id -> "report" | "magazine" (upload category). Magazines are skipped.
        self.doc_type_of = doc_type_of
        self.embed_fn = embed_fn
        self._embedding_cache: Dict[str, List[float]] = {}

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
            # Prefer native table structure when available; text-form parsing is
            # a fallback for PDFs without extractable table geometry. Mixing both
            # representations of the same page can duplicate every metric.
            native_tables = [t for t in doc_tables
                             if t.get("extraction_method") in {"docx_xml", "pdfplumber"}]
            usable_tables = native_tables or [
                t for t in doc_tables
                if t.get("extraction_method") in {"key_value_form", "period_field_pattern"}
            ]
            for t in usable_tables:
                out.extend(records_from_table(t, period))
        return out

    # -- scoring ----------------------------------------------------------------
    def _semantic_similarity(self, left: str, right: str) -> float:
        """Compare a query with extracted table text using the shared embedder."""
        if not self.embed_fn or not left.strip() or not right.strip():
            return 0.0
        try:
            missing = [text for text in dict.fromkeys((left, right)) if text not in self._embedding_cache]
            if missing:
                for text, vector in zip(missing, self.embed_fn(missing)):
                    self._embedding_cache[text] = vector
            a, b = self._embedding_cache[left], self._embedding_cache[right]
            norm_a = sum(float(x) ** 2 for x in a) ** 0.5 or 1.0
            norm_b = sum(float(x) ** 2 for x in b) ** 0.5 or 1.0
            return sum(float(x) * float(y) for x, y in zip(a, b)) / (norm_a * norm_b)
        except Exception:
            return 0.0

    def _score(self, query: str, rec: Dict[str, Any]) -> float:
        q = _content_tokens(query)
        if not q:
            return 0.0
        label = _content_tokens(rec["metric"])
        context = _content_tokens(f'{rec["section"]} {rec["region_header"]}')
        score = 2.0 * _overlap(q, label) + 0.5 * _overlap(q, context)
        # Exact terms from a table's parent heading are strong evidence for
        # short/ambiguous cell labels; keep this independent of any report data.
        score += 2.5 * _overlap(q, _content_tokens(str(rec.get("section") or "")))
        score += 14.0 * self._semantic_similarity(query, str(rec.get("metric") or ""))
        # Many report tables use short cell labels (for example “हिन्दी में”)
        # whose meaning only becomes clear beside the table's section heading.
        score += 6.0 * self._semantic_similarity(query, str(rec.get("section") or ""))
        score += 10.0 * self._semantic_similarity(
            query, " ".join(str(rec.get(key) or "") for key in ("section", "region_header", "metric"))
        )
        # Direction: "को/to" means sent to the region, "से/from" means received from it
        raw_q = set(_tokens(query))
        label_raw = set(_tokens(rec["metric"]))
        if raw_q & _TOTAL_WORDS and label_raw & _TOTAL_WORDS:
            score += 1.5
        wants_pct = bool(raw_q & _PERCENT_WORDS) or "%" in (query or "")
        asks_count = bool(_COUNT_INTENT_RE.search(query or "")) and not wants_pct
        if rec.get("is_percent"):
            score += 1.5 if wants_pct else -20.0 if asks_count else -1.5
        asks_hindi = bool(raw_q & _HINDI_TERMS)
        asks_english = bool(raw_q & _ENGLISH_TERMS)
        context_raw = set(_tokens(f'{rec["section"]} {rec["region_header"]}'))
        label_hindi = bool(label_raw & {"hindi", "हिंदी", "हिन्दी"})
        label_english = bool(label_raw & {"english", "अंग्रेजी", "अंग्रेज़ी"})
        context_hindi = bool(context_raw & _HINDI_TERMS)
        context_english = bool(context_raw & _ENGLISH_TERMS)
        asks_sent = bool(raw_q & _DIRECTION_TO) or bool(raw_q & {"send", "sent", "issue", "issued"})
        source_language = None if asks_sent else _language_near_action(query, _DIRECTION_FROM)
        if source_language:
            # Source language is indicated by the section heading around the
            # incoming/received action. Row labels often describe reply language,
            # so using them here confuses source and response language.
            section_source_language = _language_near_action(str(rec.get("section") or ""), _DIRECTION_FROM)
            source_matches = section_source_language == source_language
            score += 7.0 if source_matches else -7.0
        response_language = _language_near_action(query, _ANSWER_TERMS)
        asks_answer = bool(raw_q & _ANSWER_TERMS)
        label_is_answer = bool(label_raw & _ANSWER_TERMS)
        asks_no_reply = bool(_NO_REPLY_RE.search(query or ""))
        label_no_reply = bool(label_raw & {"नहीं", "नही", "not", "no"}) and bool(
            label_raw & {"अपेक्षित", "आवश्यक", "required", "expected", "need"}
        )
        if asks_no_reply:
            score += 14.0 if label_no_reply else -14.0
        asks_received_count = bool(raw_q & _RECEIPT_TERMS) and not asks_sent and not asks_answer and not asks_no_reply
        if asks_received_count:
            label_is_received_count = bool(label_raw & _RECEIPT_TERMS)
            score += 10.0 if label_is_received_count else (-10.0 if label_is_answer else 0.0)
        if asks_answer:
            score += 8.0 if label_is_answer and not (asks_no_reply and not label_no_reply) else -8.0
        if asks_answer and response_language:
            row_language_matches = (response_language == "hindi" and label_hindi) or (
                response_language == "english" and label_english)
            score += 8.0 if row_language_matches else -8.0
        elif asks_hindi and asks_english:
            # A multi-language question may also request an overall-total row;
            # do not reject that row just because it has no single language tag.
            if label_hindi or label_english:
                score += 4.0
            elif raw_q & _TOTAL_WORDS and label_raw & _TOTAL_WORDS:
                score += 4.0
            else:
                score -= 10.0
        elif asks_hindi or asks_english:
            hindi_match = asks_hindi and (label_hindi or (not asks_english and context_hindi))
            english_match = asks_english and (label_english or (not asks_hindi and context_english))
            row_matches_language = hindi_match or english_match
            score += 4.0 if row_matches_language else -10.0
        q_sent = bool(raw_q & _DIRECTION_TO) or "send" in raw_q
        q_received = bool(raw_q & _DIRECTION_FROM)
        ctx_sent, ctx_received = bool(context_raw & _DIRECTION_TO), bool(context_raw & _DIRECTION_FROM)
        # Prefer the explicit action in the question over a locative "from".
        # E.g. "letters sent in Hindi from Region A" asks about dispatch, while
        # "letters received from Region A" asks about receipt.
        if q_sent:
            score += 10.0 if ctx_sent else 0.0
            score -= 10.0 if ctx_received else 0.0
        elif q_received:
            score += 10.0 if ctx_received else 0.0
            score -= 10.0 if ctx_sent else 0.0
        return score

    @staticmethod
    def named_regions(query: str) -> set:
        found = set()
        for enum in _REGION_ENUM_EN_RE.finditer(query or ""):
            found.update(_REGION_EN_TO_HI[letter.lower()]
                         for letter in re.findall(r"(?<![a-z])[abc](?![a-z])", enum.group(1), re.I))
        for enum in _REGION_ENUM_HI_RE.finditer(query or ""):
            found.update(re.findall(r"[कखग]", enum.group(1)))
        for enum in _REGION_ENUM_ROMAN_RE.finditer(query or ""):
            found.update(_ROMAN_REGION[letter.lower()]
                         for letter in re.findall(r"(?<![a-z])(?:kha|kh|ka|k|ga|g)(?![a-z])", enum.group(1), re.I))
        for m in _NAMED_REGION_RE.finditer(query or ""):
            region_en = m.group("en_after") or m.group("en_before")
            region_hi = m.group("hi")
            region_roman = m.group("roman_before") or m.group("roman_after")
            if region_en:
                found.add(_REGION_EN_TO_HI[region_en.lower()])
            elif region_hi:
                found.add(region_hi)
            elif region_roman:
                found.add(_ROMAN_REGION[region_roman.lower()])
        return found

    # -- intents ------------------------------------------------------------------
    @staticmethod
    def is_comparison(query: str, several_selected: bool = False) -> bool:
        """Two periods named ("2024 vs 2025"), or a compare word while the user has
        explicitly selected several documents. A bare "अंतर" with nothing selected
        is usually a normal question ("हिंदी और अंग्रेजी में अंतर"), not a report comparison."""
        periods = set(re.findall(r"20\d{2}", query or ""))
        exact_dates = set(re.findall(r"\d{1,2}[._/-]\d{1,2}[._/-]20\d{2}", query or ""))
        if len(periods) >= 2 or len(exact_dates) >= 2:
            return True
        return several_selected and bool(_COMPARE_RE.search(query or "") or _MULTI_PERIOD_RE.search(query or ""))

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
                max_rows: Optional[int] = None) -> Optional[Dict[str, Any]]:
        recs = [r for r in self.records(document_ids) if r["value"] is not None]
        # If the query names report periods, keep comparison scoped to those
        # periods. This prevents unrelated available reports from filling gaps.
        exact_dates = re.findall(r"\d{1,2}[._/-]\d{1,2}[._/-]20\d{2}", query or "")
        requested_years = set(re.findall(r"20\d{2}", query or ""))
        if exact_dates:
            normalize_date = lambda value: tuple(int(part) for part in re.split(r"[._/-]", value))
            wanted_dates = {normalize_date(value) for value in exact_dates}
            recs = [r for r in recs if (match := _PERIOD_RE.search(str(r.get("period", ""))))
                    and normalize_date(match.group(0)) in wanted_dates]
        elif requested_years:
            recs = [r for r in recs if any(year in str(r.get("period", "")) for year in requested_years)]
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
                                        "is_percent": r["is_percent"],
                                        "aggregation_group": r.get("aggregation_group"),
                                        "values": {}, "evidence": []})
            row["values"][r["period"]] = r["value"]
            row["evidence"].append(r)

        table = [row for row in rows.values() if sum(p in row["values"] for p in periods) >= 2]
        if not table:
            return None

        # An explicit section reference is a hard structural constraint, not a
        # loose keyword. It prevents a Section 3(3) question matching Section 2.
        section_ref = _SECTION_REF_RE.search(query or "")
        if section_ref:
            scoped = _scope_explicit_section(table, section_ref)
            if not scoped:
                return None
            table = scoped

        wanted_regions = self.named_regions(query)
        if wanted_regions:
            narrowed = [row for row in table if row["region"] in wanted_regions]
            if not narrowed:
                return None
            table = narrowed

        # If the wording contains a distinctive term from a report section,
        # keep the comparison inside the best matching section. This resolves
        # repeated labels such as “हिन्दी में” that appear under incoming letters,
        # replies, and outgoing correspondence alike.
        table, section_scoped = _best_section_scope(query, table)
        section_scope_rows = list(table)

        # Narrow to what the question asks about, if it names something specific
        q_tokens = _content_tokens(re.sub(r"20\d{2}|\d{1,2}[._-]\d{1,2}[._-]\d{4}", " ", query or ""))
        if q_tokens:
            scored = [(max(self._score(query, e) for e in row["evidence"]), row) for row in table]
            top = max(s for s, _ in scored)
            if top < 2.0:
                # The question names something that is not a row in these tables
                # (e.g. "हिंदी और अंग्रेजी में अंतर"): leave it to normal RAG.
                return None
            # Keep only close semantic matches. Region rows for a requested list
            # remain separate, while unrelated measures (e.g. replies vs letters
            # sent) fall out after direction and whole-row context scoring.
            coordinated = bool(re.search(r",|;|\b(?:and|aur)\b|और", query or "", re.I))
            raw_query_tokens = set(_tokens(query or ""))
            requests_multiple_languages = bool(
                raw_query_tokens & {"hindi", "हिंदी", "हिन्दी"}
                and raw_query_tokens & {"english", "अंग्रेजी", "अंग्रेज़ी"}
            )
            margin = 2.5 if coordinated or requests_multiple_languages else 0.6
            table = [row for score, row in scored if score >= max(2.0, top - margin)]

        # When a query explicitly requests more than one language in a table
        # section, treat those language labels as separate requested measures.
        # Embeddings can rank the short Hindi and English cell labels unevenly;
        # the table's own labels are authoritative once the section is identified.
        query_token_set = set(_tokens(query or ""))
        asks_hindi = bool(query_token_set & {"hindi", "हिंदी", "हिन्दी"})
        asks_english = bool(query_token_set & {"english", "अंग्रेजी", "अंग्रेज़ी"})
        asks_total = bool(query_token_set & _TOTAL_WORDS)
        response_language = _language_near_action(query, _ANSWER_TERMS)
        asks_answer = bool(query_token_set & _ANSWER_TERMS)
        if section_scoped and asks_hindi and asks_english and not (asks_answer and response_language):
            requested_measures = []
            for row in section_scope_rows:
                if row.get("is_percent"):
                    continue
                label_tokens = set(_tokens(row.get("metric") or ""))
                language_match = bool(label_tokens & {"hindi", "हिंदी", "हिन्दी", "english", "अंग्रेजी", "अंग्रेज़ी"})
                total_match = asks_total and bool(label_tokens & _TOTAL_WORDS)
                if language_match or total_match:
                    requested_measures.append(row)
            if requested_measures:
                table = requested_measures

        if asks_total:
            requested_groups = {str(row.get("aggregation_group") or "").casefold()
                                for row in table if row.get("aggregation_group")}
            if requested_groups:
                present = {(row.get("section_no"), row.get("region"), row.get("metric")) for row in table}
                table.extend(row for row in section_scope_rows
                             if str(row.get("aggregation_group") or "").casefold() in requested_groups
                             and (row.get("section_no"), row.get("region"), row.get("metric")) not in present)

        # Sum values that the extracted table explicitly groups under one
        # parent measure (for example trained officers + trained employees).
        # This is driven by the document's column headers, not report-specific
        # question mappings.
        if asks_total:
            grouped_submetrics: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
            ungrouped_rows = []
            for row in table:
                parent = str(row.get("aggregation_group") or "").strip()
                if parent:
                    key = (str(row.get("section_no") or ""), str(row.get("region") or ""), parent.casefold())
                    grouped_submetrics.setdefault(key, []).append(row)
                else:
                    ungrouped_rows.append(row)
            for group_rows in grouped_submetrics.values():
                if len(group_rows) < 2:
                    ungrouped_rows.extend(group_rows)
                    continue
                aggregate = {**group_rows[0], "metric": group_rows[0]["aggregation_group"],
                             "region": group_rows[0].get("region"), "values": {}, "evidence": []}
                for row in group_rows:
                    for period, value in row["values"].items():
                        if value is not None:
                            aggregate["values"][period] = aggregate["values"].get(period, 0) + value
                    aggregate["evidence"].extend(row.get("evidence") or [])
                ungrouped_rows.append(aggregate)
            table = ungrouped_rows

        # A total request without an explicit region means aggregate matching
        # region rows by the extracted metric identity and reporting period.
        if asks_total and not wanted_regions and any(row.get("region") for row in table):
            grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for row in table:
                key = (str(row.get("section_no") or ""), str(row.get("metric") or "").casefold())
                target = grouped.setdefault(key, {**row, "region": None, "values": {}, "evidence": []})
                for period in periods:
                    value = row.get("values", {}).get(period)
                    if value is not None:
                        target["values"][period] = target["values"].get(period, 0) + value
                target["evidence"].extend(row.get("evidence") or [])
            table = list(grouped.values())

        # Count/display only periods that actually contain matched values for
        # this question; unrelated reports in an all-documents scope should not
        # inflate the period count or produce an empty chart series.
        periods = sorted({period for row in table for period, value in row["values"].items()
                          if value is not None}, key=_period_sort_key)
        if len(periods) < 2:
            return None
        docs_by_period = {period: docs_by_period.get(period) for period in periods}
        first, last = periods[0], periods[-1]
        out_rows = []
        selected_rows = table if max_rows is None else table[:max_rows]
        for row in selected_rows:
            a, b = row["values"].get(first), row["values"].get(last)
            change = None if a is None or b is None else b - a
            pct = None
            if change is not None and a not in (None, 0) and not row["is_percent"]:
                pct = round(change / a * 100, 2)
            metric_label = row["metric"]
            region_label = _REGION_RE.search(metric_label or "")
            if row["region"] and region_label and region_label.group(1) == row["region"]:
                metric_label = ""
            out_rows.append({
                "section_no": row["section_no"], "section": row["section"], "region": row["region"],
                "metric": metric_label, "is_percent": row["is_percent"],
                "values": {p: row["values"].get(p) for p in periods},
                "change": change, "change_pct": pct,
            })

        # Chart: one series per report, one bar group per metric (percent rows kept separate)
        asks_percent = bool(set(_tokens(query or "")) & _PERCENT_WORDS) or "%" in (query or "")
        preferred_chart_rows = [r for r in out_rows if r["is_percent"]] if asks_percent else [r for r in out_rows if not r["is_percent"]]
        chart_rows = preferred_chart_rows or out_rows
        chart = {
            "type": "bar",
            "labels": [" – ".join(part for part in ([f'{r["region"]} क्षेत्र' if r["region"] else "", r["metric"]]) if part) for r in chart_rows],
            "series": [{"name": p, "values": [r["values"].get(p) for r in chart_rows]} for p in periods],
            "unit": "%" if chart_rows and chart_rows[0]["is_percent"] else "",
        }

        lines = [f"**{len(periods)} अवधियों** की रिपोर्ट तुलना ({len(out_rows)} मद):", ""]
        for r in out_rows:
            name = " – ".join(part for part in ([f'{r["region"]} क्षेत्र' if r["region"] else "", r["metric"]]) if part)
            a, b = r["values"].get(first), r["values"].get(last)
            period_values = "; ".join(
                f'{period}: {_fmt(r["values"].get(period), r["is_percent"])}'
                for period in periods if r["values"].get(period) is not None
            )
            delta = ""
            if r["change"] is not None:
                sign = "+" if r["change"] > 0 else ""
                delta = f' ({sign}{_fmt(r["change"], r["is_percent"])}'
                delta += f', {sign}{_fmt(r["change_pct"])}%)' if r["change_pct"] is not None else ")"
            lines.append(f'- {name}: {period_values}{delta}')
        evidence = [e for row in selected_rows for e in row["evidence"]]
        return {"kind": "comparison", "answer": "\n".join(lines), "periods": periods,
                "documents": {p: docs_by_period.get(p) for p in periods},
                "comparison": out_rows, "chart_data": chart, "evidence": evidence}

    def lookup(self, query: str, document_ids: Optional[List[str]] = None,
               latest_report_only: bool = False) -> Optional[Dict[str, Any]]:
        """Answer a numeric question from extracted report tables."""
        if not _COUNT_INTENT_RE.search(query or ""):
            return None
        rows = [r for r in self.records(document_ids) if r.get("value") is not None]
        if not rows:
            return None
        section_ref = _SECTION_REF_RE.search(query or "")
        if section_ref:
            rows = _scope_explicit_section(rows, section_ref)
        wanted_regions = self.named_regions(query)
        if wanted_regions:
            rows = [r for r in rows if r.get("region") in wanted_regions]
        if not rows:
            return None

        # For questions that state the language of received letters, scope to
        # sections whose heading identifies that incoming language. This keeps
        # "Hindi letters replied in English" from matching the separate table
        # for "English letters replied in Hindi". Derive it from each heading;
        # no section number, metric value, or report-specific mapping is used.
        query_tokens = set(_tokens(query or ""))
        asks_sent = bool(query_tokens & _DIRECTION_TO) or bool(query_tokens & {"send", "sent", "issue", "issued"})
        source_language = None if asks_sent else _language_near_action(query, _DIRECTION_FROM)
        if source_language:
            source_scoped = [
                row for row in rows
                if _language_near_action(str(row.get("section") or ""), _DIRECTION_FROM) == source_language
            ]
            if source_scoped:
                rows = source_scoped
        rows, _ = _best_section_scope(query, rows)

        scored = [(self._score(query, row), row) for row in rows]
        top = max(score for score, _ in scored)
        if top < 2.0:
            return None
        selected = [row for score, row in scored if score >= max(2.0, top - 0.6)]
        if latest_report_only and selected:
            # A plain fact question over multiple selected reports refers to the
            # latest matching report by default. Explicit comparisons are handled
            # separately and retain every requested period.
            latest_period = max((str(row.get("period") or "") for row in selected),
                                key=_period_sort_key)
            rows = [row for row in rows if str(row.get("period") or "") == latest_period]
            selected = [row for row in selected if str(row.get("period") or "") == latest_period]
        asks_total = bool(set(_tokens(query or "")) & _TOTAL_WORDS)
        raw_query_tokens = set(_tokens(query or ""))
        asks_category = bool(raw_query_tokens & {
            "hindi", "हिंदी", "हिन्दी", "english", "अंग्रेजी", "अंग्रेज़ी",
            "bilingual", "द्विभाषी", "original", "मूल",
        })
        if asks_total and not asks_category:
            # A total question should resolve to the row explicitly labelled as
            # the total, not a similarly worded subtotal such as bilingual-only.
            explicit_totals = [
                row for row in selected
                if set(_tokens(row.get("metric_short") or "")) & {"कुल", "total"}
            ]
            if explicit_totals:
                selected = explicit_totals
        if asks_total:
            requested_groups = {str(row.get("aggregation_group") or "").casefold()
                                for row in selected if row.get("aggregation_group")}
            if requested_groups:
                present = {(row.get("section_no"), row.get("region"), row.get("metric")) for row in selected}
                selected.extend(row for row in rows
                                if str(row.get("aggregation_group") or "").casefold() in requested_groups
                                and (row.get("section_no"), row.get("region"), row.get("metric")) not in present)
            groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
            remaining = []
            for row in selected:
                parent = str(row.get("aggregation_group") or "").strip()
                if parent:
                    groups.setdefault((str(row.get("section_no") or ""),
                                       str(row.get("region") or ""), parent.casefold()), []).append(row)
                else:
                    remaining.append(row)
            for grouped in groups.values():
                if len(grouped) > 1:
                    total = sum(row["value"] for row in grouped)
                    merged = {**grouped[0], "metric_short": grouped[0]["aggregation_group"],
                              "raw_value": _fmt(total), "value": total}
                    remaining.append(merged)
                else:
                    remaining.extend(grouped)
            selected = remaining

        if not selected:
            return None
        answer = "\n".join(f'**{row["metric_short"]}:** {_fmt(row["value"], row["is_percent"])}'
                           for row in selected)
        return {"kind": "report_lookup", "answer": answer, "evidence": selected}

    # -- entry point used by the chat route ------------------------------------------
    def try_answer(self, query: str, document_ids: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        try:
            scope = get_scope()
            several = bool((document_ids and len(document_ids) > 1) or (scope and len(scope) > 1))
            compare_all_periods = bool(
                document_ids is None and not scope and _MULTI_PERIOD_RE.search(query or "")
            )
            if self.is_comparison(query, several) or compare_all_periods:
                res = self.compare(query, document_ids)
                if res:
                    return res
            if self.is_region_total(query):
                result = self.region_total(query, document_ids)
                if result:
                    return result
            # With all/multiple reports in scope, a simple fact lookup returns
            # the newest matching report. If the query asks for both periods or
            # comparison, compare() above returns the full period-by-period result.
            multiple_or_all = document_ids is None or len(document_ids) != 1 or bool(scope and len(scope) > 1)
            return self.lookup(query, document_ids, latest_report_only=multiple_or_all)
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
