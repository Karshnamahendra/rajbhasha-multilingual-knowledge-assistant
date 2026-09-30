"""Table-of-contents extraction for DOCX magazines whose contents page is a Word table.

Many Rajbhasha magazines lay out "इस अंक में / विषय सूची" as a table:

    ┌──────────────── तकनीकी लेख (merged row = category) ────────────────┐
    │ 1 │ साइबर सुरक्षा : सरकारी प्रयास…  │ : │ सुश्री कान्ती सिंह सेंघर │ 9 │
    │   │                                 │   │ सुश्री रेखा सरस्वत       │   │  ← 2nd author = 2nd paragraph
    └───┴─────────────────────────────────┴───┴──────────────────────────┴───┘

The generic text-based TOCExtractor flattens these cells, so titles swallow the
author ("title : : : author"), co-authors get glued together and page numbers
are lost. Reading the table cells directly keeps every field separate.

Output records use exactly the schema of TOCExtractor, so everything that reads
the index store (author/title lookups, hierarchy builder) keeps working.
Nothing here is specific to one magazine.
"""
from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from index_knowledge_layer import fix_devanagari_ocr, romanize_generic

_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_SERIAL_RE = re.compile(r"^\s*(\d{1,3})\s*[.)]?\s*$")
_PAGE_RE = re.compile(r"^\s*(\d{1,4})\s*$")
_HONORIFIC_RE = re.compile(r"^\s*(?:श्रीमती|सुश्री|श्री|डॉ\.?|डा\.|Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Shri|Smt\.?)\s*", re.I)
# Two names typed into one paragraph: split before a second honorific
_INNER_HONORIFIC_RE = re.compile(r"\s+(?=(?:श्रीमती|सुश्री|श्री|डॉ\.|Dr\.)\s)")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def _cell_paragraphs(cell) -> List[str]:
    paras = []
    for p in cell.findall(f"{_NS}p"):
        text = _clean("".join(n.text or "" for n in p.iter(f"{_NS}t")))
        if text:
            paras.append(text)
    return paras


def split_authors(paragraphs: List[str]) -> List[str]:
    """One author per paragraph; also split 'सुश्री A सुश्री B' typed on one line.
    Returned names have honorifics removed so the same person is counted once."""
    names: List[str] = []
    for para in paragraphs:
        for part in re.split(r"\s*[,;/|]\s*|\s+(?:एवं|और|and)\s+", para):
            for piece in _INNER_HONORIFIC_RE.split(part):
                name = _clean(_HONORIFIC_RE.sub("", piece))
                if name and name not in {"-", ":"} and name not in names:
                    names.append(name)
    return names


def _entry_type(section: str) -> str:
    s = (section or "").lower()
    if any(k in s for k in ["कविता", "poem", "poetry"]):
        return "poem"
    if any(k in s for k in ["कहानी", "story", "fiction"]) and "लेख" not in s:
        return "story"
    if any(k in s for k in ["श्रद्धांजलि", "tribute", "memoriam"]):
        return "tribute"
    if any(k in s for k in ["रिपोर्ट", "कॉर्नर", "report", "corner"]):
        return "report"
    return "article"


def _parse_row(cells) -> Optional[Dict[str, Any]]:
    """serial | title | [:] | author(s) | page   (colon column optional)"""
    paras = [_cell_paragraphs(c) for c in cells]
    flat = [" ".join(p) for p in paras]
    if len(flat) < 3 or not _SERIAL_RE.match(flat[0]) or not _PAGE_RE.match(flat[-1]):
        return None
    middle = list(zip(flat[1:-1], paras[1:-1]))
    middle = [(t, p) for t, p in middle if t and t not in {":", "：", "-", "–"}]
    if not middle:
        return None
    title = middle[0][0]
    author_paras = [x for _, p in middle[1:] for x in p]
    return {
        "serial": int(_SERIAL_RE.match(flat[0]).group(1)),
        "title": title,
        "author_paras": author_paras,
        "page": int(_PAGE_RE.match(flat[-1]).group(1)),
    }


def _category_of(cells) -> Optional[str]:
    texts = [" ".join(_cell_paragraphs(c)) for c in cells]
    texts = [t for t in texts if t]
    if not texts or len(set(texts)) != 1:
        return None
    t = texts[0]
    if re.search(r"\d", t) or len(t) > 60:
        return None
    return t


def _toc_page(pages: Optional[List[Dict[str, Any]]], first_title: str) -> int:
    if pages and first_title:
        probe = first_title[:20]
        for p in pages[:10]:
            if probe and probe in (p.get("text") or ""):
                return int(p.get("page_number") or 1)
    return 1


def extract_docx_toc_records(file_path: str, document_id: str, filename: str,
                             pages: Optional[List[Dict[str, Any]]] = None,
                             min_entries: int = 5) -> List[Dict[str, Any]]:
    """Return TOC records from the first DOCX table that looks like a contents table, else []."""
    try:
        with zipfile.ZipFile(file_path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
    except Exception:
        return []

    for table in root.iter(f"{_NS}tbl"):
        entries: List[Dict[str, Any]] = []
        section = "General"
        for row in table.findall(f"{_NS}tr"):
            cells = row.findall(f"{_NS}tc")
            parsed = _parse_row(cells)
            if parsed:
                parsed["section"] = section
                entries.append(parsed)
                continue
            cat = _category_of(cells)
            if cat:
                section = cat
        if len(entries) < min_entries:
            continue

        toc_p = _toc_page(pages, entries[0]["title"])
        counters: Dict[str, int] = {}
        records = []
        for e in entries:
            title = fix_devanagari_ocr(e["title"])
            sec = fix_devanagari_ocr(e["section"])
            counters[sec] = counters.get(sec, 0) + 1
            authors = split_authors([fix_devanagari_ocr(a) for a in e["author_paras"]])
            author_text = ", ".join(authors) if authors else None
            records.append({
                "document_id": document_id,
                "source_filename": filename,
                "toc_page": toc_p,
                "serial_number": e["serial"],
                "section_original": sec,
                "section_normalized": sec,
                "section_item_index": counters[sec],
                "title_original": title,
                "title_normalized": title,
                "title_roman": romanize_generic(title),
                "author_original": author_text,
                "author_normalized": author_text,
                "author_roman": romanize_generic(author_text) if author_text else None,
                "author_list": authors,
                "page_number": e["page"],
                "type": _entry_type(sec),
                "extraction_method": "docx_toc_table",
            })
        return records
    return []