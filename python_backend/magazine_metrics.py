"""Magazine questions that need counting or comparing, answered from the contents (TOC) records.

Examples it handles (Hindi, Hinglish, English):
    "AI par kitne lekh hain?"            -> count titles on a topic (with the list)
    "2025 में कितनी कविताएं हैं?"          -> count by category, edition picked by year
    "technical articles kaun se hain?"   -> list by category
    "kis lekhak ne sabse zyada likha?"   -> per-author counts
    "अनिल कुमार ने कितने लेख लिखे?"        -> count for a named author
    "2024 vs 2025 magazine"              -> edition comparison + chart data

Counts are computed in code from the TOC records the index store keeps for every
magazine, so the LLM never does arithmetic. Questions that are not about
counting / listing / comparing are left to normal RAG (returns None).
Nothing here is specific to one magazine.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter, OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

from doc_scope import in_scope, get_scope
from index_knowledge_layer import romanize_generic

_TOKEN_RE = re.compile(r"[a-z0-9]+|[ऀ-ॿ]+")
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

_COUNT_RE = re.compile(r"कितन[ेीा]|कितनों|संख्या|\bkitn[eia]\b|\bhow many\b|\bcount\b|\bnumber of\b|\btotal\b|कुल|\bkul\b", re.I)
_LIST_RE = re.compile(r"कौन[\s-]?से|कौन[\s-]?सी|कौनसे|सूची|\bkaun ?s[ei]\b|\bwhich\b|\blist\b|\bnames?\b|दिखाओ|बताओ|\bbatao\b|\bdikhao\b", re.I)
_AUTHOR_WORD_RE = re.compile(r"लेखक|लेखकों|रचनाकार|(?<![ऀ-ॿ])(?:कवि|कवियों)(?![ऀ-ॿ])|\bauthors?\b|\bwriters?\b|\blekhak\w*|\bkavi\b", re.I)
_NE_RE = re.compile(r"(?<![ऀ-ॿ])ने(?![ऀ-ॿ])|\bne\b", re.I)
_MOST_RE = re.compile(r"सबसे|ज़्यादा|ज्यादा|अधिक|\bmost\b|\bmaximum\b|\bsabse\b|\bzyada\b|\bjyada\b|\btop\b", re.I)
_WORK_RE = re.compile(
    r"लेख|रचना|रचनाएं|रचनाएँ|कविता|कविताएं|कविताएँ|कहानी|कहानियां|कहानियाँ|articles?|poems?|poetry|stor(?:y|ies)|"
    r"\blekh\b|\brachna\w*|\bkavit\w*|\bkahani\w*|entries|लेखक|author|writer|lekhak",
    re.I,
)
_COMPARE_RE = re.compile(r"compare|comparison|\bvs\.?\b|versus|तुलना|\btulna\b|बनाम|अंतर|\bantar\b|difference|फर्क", re.I)

# Categories as they appear in contents tables; matched against section names, not hard-coded to one issue
_CATEGORY_QUERY = [
    # (query pattern, predicate on (section, type), label)
    (re.compile(r"गैर[\s-]?तकनीकी|non[\s-]?technical|gair[\s-]?takn?iki", re.I),
     lambda sec, typ: "गैर" in sec or "non" in sec.lower(), "गैर-तकनीकी"),
    (re.compile(r"तकनीकी|technical|takn?iki", re.I),
     lambda sec, typ: ("तकनीकी" in sec or "technical" in sec.lower()) and "गैर" not in sec and "non" not in sec.lower(), "तकनीकी"),
    (re.compile(r"कविता|कविताएं|कविताएँ|poem|poetry|kavit", re.I),
     lambda sec, typ: typ == "poem" or "कविता" in sec, "कविता"),
    (re.compile(r"कहानी|कहानियां|कहानियाँ|stor(?:y|ies)|kahani", re.I),
     lambda sec, typ: "कहानी" in sec or typ == "story", "कहानी"),
    (re.compile(r"विशेषांक|special", re.I),
     lambda sec, typ: "विशेषांक" in sec, "विशेषांक"),
    (re.compile(r"राजभाषा\s*कॉर्नर|corner", re.I),
     lambda sec, typ: "कॉर्नर" in sec or "corner" in sec.lower(), "राजभाषा कॉर्नर"),
]

# Topic words -> variants seen in Hindi titles
_TOPIC_ALIASES = {
    "ai": ["कृत्रिम बुद्धिमत्ता", "एआई", "आर्टिफिशियल", "artificial intelligence", "ai"],
    "बुद्धिमत्ता": ["कृत्रिम बुद्धिमत्ता", "एआई", "आर्टिफिशियल", "ai"],
    "artificial": ["कृत्रिम बुद्धिमत्ता", "एआई", "आर्टिफिशियल", "artificial intelligence", "ai"],
    "intelligence": ["कृत्रिम बुद्धिमत्ता", "एआई", "आर्टिफिशियल", "artificial intelligence", "ai"],
    "कृत्रिम": ["कृत्रिम बुद्धिमत्ता", "एआई", "आर्टिफिशियल", "ai"],
    "एआई": ["कृत्रिम बुद्धिमत्ता", "एआई", "आर्टिफिशियल", "ai"],
    "cyber": ["साइबर", "साईबर", "cyber"],
    "saibar": ["साइबर", "साईबर", "cyber"],
    "साइबर": ["साइबर", "साईबर", "cyber"],
    "साईबर": ["साइबर", "साईबर", "cyber"],
    "digital": ["डिजिटल", "डि़जिटाइजेशन", "digital"],
    "डिजिटल": ["डिजिटल", "डि़जिटाइजेशन", "digital"],
    "software": ["सॉफ्टवेयर", "सॉफ़्टवेयर", "software"],
    "सॉफ्टवेयर": ["सॉफ्टवेयर", "सॉफ़्टवेयर", "software"],
    "security": ["सुरक्षा", "security", "सिक्योरिटी"],
    "suraksha": ["सुरक्षा", "security", "सिक्योरिटी"],
    "सुरक्षा": ["सुरक्षा", "security", "सिक्योरिटी"],
    "yoga": ["योग"], "yog": ["योग"], "योग": ["योग"],
    "mahakumbh": ["महाकुंभ"], "महाकुंभ": ["महाकुंभ"],
    "rajbhasha": ["राजभाषा"], "राजभाषा": ["राजभाषा"],
}

# Words that never name a topic
_STOP = {
    "की", "का", "के", "में", "मे", "है", "हैं", "था", "थे", "और", "या", "पर", "से", "को", "ने", "कितने", "कितनी",
    "कितना", "कौन", "कौनसे", "सी", "कुल", "संख्या", "सूची", "सभी", "सबसे", "ज्यादा", "ज़्यादा", "अधिक", "लिखे",
    "लिखी", "लिखा", "गए", "गई", "गये", "लेख", "लेखों", "रचना", "रचनाएं", "रचनाएँ", "कविता", "कविताएं", "कविताएँ",
    "कहानी", "कहानियां", "कहानियाँ", "लेखक", "लेखकों", "पत्रिका", "मैगज़ीन", "मैगजीन", "अंक", "विषय", "बारे",
    "तकनीकी", "गैर", "किस", "किसने", "किन", "इस", "उस", "यह", "वह", "तुलना", "बताओ", "दिखाओ", "वर्ष", "साल",
    "par", "pe", "ki", "ka", "ke", "me", "mein", "hai", "hain", "aur", "kitne", "kitni", "kitna", "kaun", "kaunse",
    "se", "si", "kul", "sabse", "zyada", "jyada", "likhe", "likha", "lekh", "lekhak", "kavita", "kavitayen",
    "kahani", "magazine", "patrika", "ank", "technical", "takniki", "gair", "kis", "kisne", "batao", "dikhao",
    "the", "of", "in", "on", "about", "is", "are", "how", "many", "much", "what", "which", "who", "list", "all",
    "articles", "article", "poems", "poem", "stories", "story", "authors", "author", "writers", "writer", "wrote",
    "written", "by", "total", "count", "number", "most", "non", "and", "or", "a", "an", "to", "there", "have",
    "has", "vs", "compare", "comparison", "edition", "issue", "top", "name", "names",
}

_MAX_LIST = 25


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _roman_key(name: str) -> str:
    """Rough phonetic key so 'anil kumar' ~ 'anila kumara' and सरस्वत ~ सारस्वत."""
    rom = romanize_generic(name) if re.search(r"[ऀ-ॿ]", name or "") else (name or "").lower()
    out = []
    for tok in re.findall(r"[a-z]+", rom):
        tok = re.sub(r"a+$", "", tok)                 # inherent vowel: kumara -> kumar
        tok = re.sub(r"(.)\1+", r"\1", tok)            # doubled letters
        tok = tok.replace("aa", "a").replace("w", "v").replace("ee", "i").replace("oo", "u")
        out.append(tok)
    return " ".join(out)


def _same_person(a: str, b: str) -> bool:
    if a == b:
        return True
    ka, kb = _roman_key(a), _roman_key(b)
    if len(ka.split()) != len(kb.split()):
        return False
    return difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.84


def _edition_label(document_id: str) -> str:
    years = _YEAR_RE.findall(document_id or "")
    return years[-1] if years else (document_id or "")


class MagazineMetrics:
    def __init__(self, index_store, doc_type_of: Optional[Callable[[str], Optional[str]]] = None):
        self.index_store = index_store
        self.doc_type_of = doc_type_of

    # -- data ---------------------------------------------------------------------
    def entries(self, document_ids: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """document_id -> TOC records, for magazines in the current scope."""
        records = []
        if document_ids:
            for d in document_ids:
                records.extend(self.index_store.get_records(d))
        else:
            records = self.index_store.get_records(None)
        out: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        for r in records:
            doc = r.get("document_id")
            if not doc or not in_scope(doc):
                continue
            if self.doc_type_of and (self.doc_type_of(doc) or "magazine") == "report":
                continue
            if not (r.get("title_original") or "").strip():
                continue
            out.setdefault(doc, []).append(r)
        # Oldest edition first
        return OrderedDict(sorted(out.items(), key=lambda kv: _edition_label(kv[0])))

    # -- query understanding ------------------------------------------------------------
    @staticmethod
    def _category(query: str):
        for pattern, pred, label in _CATEGORY_QUERY:
            if pattern.search(query or ""):
                return pred, label
        return None, None

    @staticmethod
    def _topic_terms(query: str) -> List[List[str]]:
        """Each topic word in the question -> its variants. [] if the question names no topic."""
        topics = []
        for tok in _tokens(query):
            if tok in _STOP or tok.isdigit() or len(tok) < 2:
                continue
            if any(p.search(tok) for p, _, _ in _CATEGORY_QUERY):
                continue
            group = _TOPIC_ALIASES.get(tok, [tok])
            if group not in topics:          # "कृत्रिम बुद्धिमत्ता" = one topic, not two
                topics.append(group)
        # "साइबर सुरक्षा / cyber security" is one subject: cyber alone decides
        cyber = _TOPIC_ALIASES["cyber"]
        if cyber in topics:
            topics = [g for g in topics if g is cyber or g == cyber or g != _TOPIC_ALIASES["security"]]
        return topics

    @staticmethod
    def _title_matches(title: str, topics: List[List[str]]) -> bool:
        t = (title or "").lower()
        t_rom = romanize_generic(title or "")
        for variants in topics:
            hit = False
            for v in variants:
                v = v.lower()
                if re.fullmatch(r"[a-z]{1,3}", v):          # short latin like "ai": whole word only
                    if re.search(rf"(?<![a-z]){v}(?![a-z])", t):
                        hit = True
                elif v in t or (re.fullmatch(r"[a-z]+", v) and len(v) >= 4 and v in t_rom):
                    hit = True
                elif len(v) >= 4 and any(w.startswith(v) for w in _tokens(t)):
                    hit = True
            if not hit:
                return False
        return True

    def _author_counts(self, entries: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
        groups: List[Tuple[str, List[Dict[str, Any]]]] = []
        for e in entries:
            for name in e.get("author_list") or []:
                for i, (canon, works) in enumerate(groups):
                    if _same_person(canon, name):
                        works.append(e)
                        break
                else:
                    groups.append((name, [e]))
        groups.sort(key=lambda g: (-len(g[1]), g[0]))
        return groups

    def _named_author(self, query: str, entries: List[Dict[str, Any]]) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        q_dev = set(_tokens(query))
        q_rom = _roman_key(" ".join(t for t in _tokens(query) if re.fullmatch(r"[a-z]+", t)))
        best = None
        for canon, works in self._author_counts(entries):
            parts = _tokens(canon)
            if parts and all(p in q_dev for p in parts):
                return canon, works
            key = _roman_key(canon)
            if q_rom and key and len(key) >= 5 and key in q_rom:
                best = best or (canon, works)
        return best

    # -- formatting -----------------------------------------------------------------------
    @staticmethod
    def _line(e: Dict[str, Any]) -> str:
        by = f' — {e["author_original"]}' if e.get("author_original") else ""
        page = f' (पृष्ठ {e["page_number"]})' if e.get("page_number") else ""
        return f'- {e["title_original"]}{by}{page}'

    @staticmethod
    def _breakdown(entries: List[Dict[str, Any]]) -> "OrderedDict[str, int]":
        c: "OrderedDict[str, int]" = OrderedDict()
        for e in entries:
            sec = e.get("section_original") or "General"
            c[sec] = c.get(sec, 0) + 1
        return c

    # -- answers ------------------------------------------------------------------------------
    def _count_or_list(self, query: str, docs: "OrderedDict[str, List[Dict[str, Any]]]") -> Optional[Dict[str, Any]]:
        pred, cat_label = self._category(query)
        topics = self._topic_terms(query)
        if topics:
            # Filler words ("wale", "related", "संबंधित") name no topic: keep only words
            # that occur in at least one title; if none do, this is not a topic count.
            all_titles = [e.get("title_original") for es in docs.values() for e in es]
            topics = [g for g in topics if any(self._title_matches(t, [g]) for t in all_titles)]
            if not topics:
                return None
        wants_list = bool(_LIST_RE.search(query)) and not _COUNT_RE.search(query)
        parts, evidence, results = [], [], []
        for doc, entries in docs.items():
            chosen = entries
            if pred:
                chosen = [e for e in chosen if pred(e.get("section_original") or "", e.get("type") or "")]
            if topics:
                chosen = [e for e in chosen if self._title_matches(e.get("title_original"), topics)]
            head = f'📖 **{doc}**' if len(docs) > 1 else ""
            if not pred and not topics:
                bd = self._breakdown(entries)
                text = f"कुल **{len(entries)}** रचनाएँ: " + ", ".join(f"{k} {v}" for k, v in bd.items())
            else:
                topic_txt = ("‘" + ", ".join(v[0] for v in topics) + "’ विषय पर ") if topics else ""
                noun = cat_label or "रचनाएँ"
                text = f"{topic_txt}{noun}: **{len(chosen)}**"
                if chosen:
                    text += "\n" + "\n".join(self._line(e) for e in chosen[:_MAX_LIST])
                    if len(chosen) > _MAX_LIST:
                        text += f"\n- … और {len(chosen) - _MAX_LIST}"
            if wants_list and not pred and not topics:
                text += "\n" + "\n".join(self._line(e) for e in entries[:_MAX_LIST])
            parts.append((head + "\n" if head else "") + text)
            evidence.extend(chosen if (pred or topics) else entries)
            results.append({"document_id": doc, "count": len(chosen) if (pred or topics) else len(entries),
                            "category": cat_label, "topic": [v[0] for v in topics]})
        if topics and all(r["count"] == 0 for r in results):
            return None  # topic word may be something the TOC simply doesn't name; let RAG try
        return {"kind": "magazine_count", "answer": "\n\n".join(parts), "results": results, "evidence": evidence}

    def _authors(self, query: str, docs: "OrderedDict[str, List[Dict[str, Any]]]") -> Optional[Dict[str, Any]]:
        all_entries = [e for es in docs.values() for e in es]
        named = self._named_author(query, all_entries)
        if named:
            canon, works = named
            lines = [f"**{canon}** की रचनाएँ: **{len(works)}**"]
            for doc in docs:
                mine = [w for w in works if w["document_id"] == doc]
                if mine and len(docs) > 1:
                    lines.append(f"\n📖 {doc}: {len(mine)}")
                lines.extend(self._line(w) for w in mine)
            return {"kind": "author_count", "answer": "\n".join(lines), "evidence": works,
                    "results": [{"author": canon, "count": len(works)}]}
        if not _AUTHOR_WORD_RE.search(query):
            return None
        groups = self._author_counts(all_entries)
        if not groups:
            return None
        top = groups[:15]
        lines = [f"कुल **{len(groups)}** लेखक/रचनाकार। सबसे अधिक रचनाएँ:"]
        lines += [f'- {name}: {len(works)}' for name, works in top]
        chart = {"type": "bar", "labels": [n for n, _ in top[:10]],
                 "series": [{"name": "रचनाएँ", "values": [len(w) for _, w in top[:10]]}], "unit": ""}
        return {"kind": "author_count", "answer": "\n".join(lines), "chart_data": chart,
                "evidence": [w for _, ws in top for w in ws][:30],
                "results": [{"author": n, "count": len(w)} for n, w in groups]}

    def compare(self, query: str, docs: "OrderedDict[str, List[Dict[str, Any]]]") -> Optional[Dict[str, Any]]:
        if len(docs) < 2:
            return None
        editions = {doc: _edition_label(doc) for doc in docs}
        sections = []
        for entries in docs.values():
            for sec in self._breakdown(entries):
                if sec not in sections:
                    sections.append(sec)
        rows = []
        for sec in sections + ["कुल"]:
            values = {}
            for doc, entries in docs.items():
                values[editions[doc]] = len(entries) if sec == "कुल" else self._breakdown(entries).get(sec, 0)
            rows.append({"metric": sec, "region": None, "section_no": "", "section": "", "is_percent": False,
                         "values": values})
        periods = [editions[d] for d in docs]
        first, last = periods[0], periods[-1]
        for r in rows:
            a, b = r["values"].get(first), r["values"].get(last)
            r["change"] = None if a is None or b is None else b - a
            r["change_pct"] = round((b - a) / a * 100, 2) if a else None

        authors = {editions[d]: {n for n, _ in self._author_counts(es)} for d, es in docs.items()}
        a_first, a_last = authors[first], authors[last]
        common = sum(1 for x in a_last if any(_same_person(x, y) for y in a_first))
        topic_rows = []
        for key, label in [("ai", "AI / कृत्रिम बुद्धिमत्ता"), ("cyber", "साइबर")]:
            vals = {editions[d]: sum(1 for e in es if self._title_matches(e["title_original"], [_TOPIC_ALIASES[key]]))
                    for d, es in docs.items()}
            topic_rows.append((label, vals))

        lines = [f"**{first}** और **{last}** अंक की तुलना:", ""]
        for r in rows:
            sign = "+" if (r["change"] or 0) > 0 else ""
            delta = f' ({sign}{r["change"]})' if r["change"] is not None else ""
            name = f'**{r["metric"]}**' if r["metric"] == "कुल" else r["metric"]
            lines.append(f'- {name}: {r["values"].get(first, 0)} → {r["values"].get(last, 0)}{delta}')
        for label, vals in topic_rows:
            lines.append(f'- {label} विषय पर: {vals.get(first, 0)} → {vals.get(last, 0)}')
        lines.append(f"- लेखक: {len(a_first)} → {len(a_last)} ({common} लेखक दोनों अंकों में)")

        chart = {"type": "bar", "labels": [r["metric"] for r in rows if r["metric"] != "कुल"],
                 "series": [{"name": p, "values": [r["values"].get(p, 0) for r in rows if r["metric"] != "कुल"]}
                            for p in periods], "unit": ""}
        evidence = [e for es in docs.values() for e in es][:20]
        return {"kind": "comparison", "answer": "\n".join(lines), "periods": periods,
                "comparison": rows, "chart_data": chart, "evidence": evidence}

    # -- entry point -----------------------------------------------------------------------------
    def try_answer(self, query: str, document_ids: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        try:
            docs = self.entries(document_ids)
            if not docs:
                return None
            q = query or ""
            years = set(_YEAR_RE.findall(q))
            scope = get_scope()
            several = bool((document_ids and len(document_ids) > 1) or (scope and len(scope) > 1))

            # Edition comparison: two years named, or a compare word with several magazines selected
            if len(docs) >= 2 and (len(years) >= 2 or (several and _COMPARE_RE.search(q))):
                picked = OrderedDict((d, e) for d, e in docs.items()
                                     if not years or _edition_label(d) in years) if len(years) >= 2 else docs
                res = self.compare(q, picked if len(picked) >= 2 else docs)
                if res:
                    return res

            if not (_WORK_RE.search(q) and (_COUNT_RE.search(q) or _LIST_RE.search(q) or _MOST_RE.search(q))):
                return None
            # One year named: answer for that edition only
            if len(years) == 1:
                only = OrderedDict((d, e) for d, e in docs.items() if _edition_label(d) in years)
                docs = only or docs

            if _AUTHOR_WORD_RE.search(q) or _NE_RE.search(q):
                res = self._authors(q, docs)
                if res:
                    return res
            return self._count_or_list(q, docs)
        except Exception as exc:  # never break chat because of this layer
            print(f"[MagazineMetrics] skipped: {exc}")
            return None


def toc_evidence_to_sources(evidence: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    sources, seen = [], set()
    for e in evidence:
        key = (e.get("document_id"), e.get("title_original"))
        if key in seen:
            continue
        seen.add(key)
        text = f'{e.get("section_original") or ""} › {e.get("title_original")}'
        if e.get("author_original"):
            text += f' — {e["author_original"]}'
        sources.append({
            "quote": text, "text": text, "highlightText": text,
            "pageNumber": e.get("page_number") or e.get("toc_page") or 1,
            "page_number": e.get("page_number") or e.get("toc_page") or 1,
            "chunkId": f'{e.get("document_id")}:toc:{e.get("serial_number")}',
            "chunk_id": f'{e.get("document_id")}:toc:{e.get("serial_number")}',
            "docName": e.get("document_id"), "document_name": e.get("document_id"),
            "document_id": e.get("document_id"), "document_type": "magazine",
            "content_type": "toc", "collection_type": "index", "section": e.get("section_original"),
            "relevanceScore": 100,
        })
        if len(sources) >= limit:
            break
    return sources