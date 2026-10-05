"""Extracts, indexes, and resolves Table of Contents (TOC) records for author-article-page queries."""
import os
import re
import json
import logging
import difflib
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Union
from terminology_memory import DynamicTerminologyMemory
from doc_scope import in_scope

logger = logging.getLogger("IndexKnowledgeLayer")
logger.setLevel(logging.INFO)

# =====================================================================
# 1. UNIVERSAL SCRIPT & LIGATURE NORMALIZATION (GENERIC FOR ANY PDF)
# =====================================================================

def fix_devanagari_ocr(text: str) -> str:
    """Repairs common OCR ligature and font extraction artifacts in Hindi Devanagari."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    
    replacements = [
        (r"\u093F\u0947", "वे"),
        (r"िकनीकी", "तकनीकी"),
        (r"ववशेर्\s*ंक", "विशेष अंक"),
        (r"कववि", "कविता"),
        (r"श्रद्\s*ंजभल", "श्रद्धांजलि"),
        (r"र\s*जि\s*र्\s*कॉनषर", "राजभाषा कॉर्नर"),
        (r"कह\s*नी", "कहानी"),
        (r"डाका\s*पैटना", "डार्क पैटर्न"),
        (r"पार[रि]स्स्थत[ति]क[ति]\s*तुंत्र", "पारिस्थितिकी तंत्र"),
        (r"तुंत्र", "तंत्र"),
        (r"चुनौत[ति]या[ाँ|ां]?", "चुनौतियां"),
        (r"अुंतरराष्ट्रीय", "अंतरराष्ट्रीय"),
        (r"अन्द्निर्ा", "अन्न वर्ष"),
        (r"पररयोजना\s*प्रबुंधन\s*प्रणाली", "परियोजना प्रबंधन प्रणाली"),
        (r"प्रबुंधन\s*प्रणाली", "प्रबंधन प्रणाली"),
        (r"जेनरेटटि\s*एआई", "जेनरेटिव एआई"),
        (r"ग्राफफक\s*डडज़ाइन", "ग्राफिक डिज़ाइन"),
        (r"स्स्प्रुंग", "स्प्रिंग"),
        (r"िस\s*िसत", "वेबसाइट"),
        (r"िस\s*साइट", "वेबसाइट"),
        (r"डडजाइतनुंग", "डिजाइनिंग"),
        (r"प्राकृततक\s*भार्ा\s*प्रसुंस्करण", "प्राकृतिक भाषा प्रसंस्करण"),
        (r"ररकॉडा", "रिकॉर्ड"),
        (r"विश्ि", "विश्व"),
        (r"आधुतनक", "आधुनिक"),
        (r"सुन्द्दरता", "सुंदरता"),
        (r"मटहलाओुं", "महिलाओं"),
        (r"बहुभार्ी", "बहुभाषी"),
        (r"सामास्जक", "सामाजिक"),
        (r"भारतीय\s*सुंस्कृतत", "भारतीय संस्कृति"),
        (r"जीिन\s*का\s*सुंदेश", "जीवन का संदेश"),
        (r"अहसमयत", "अहमियत"),
        (r"चुंरयान", "चंद्रयान"),
    ]
    for pattern, repl in replacements:

        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        
    text = re.sub(r'([\u0905-\u0939\u0958-\u095F])\s+([\u093E-\u094D\u0962\u0963])', r'\1\2', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF\u00AD\uf000-\uf8ff]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


DEVA_CONSONANTS = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
    'ष': 'sh', 'स': 's', 'ह': 'h', 'क़': 'q', 'ख़': 'kh',
    'ग़': 'gh', 'ज़': 'z', 'ड़': 'd', 'ढ़': 'dh', 'फ़': 'f',
    'त्र': 'tr', 'ज्ञ': 'gy', 'क्ष': 'ksh'
}

DEVA_MATRAS = {
    'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    'ं': 'n', 'ँ': 'n', 'ः': 'h'
}

DEVA_INDEP_VOWELS = {
    'अ': 'a', 'आ': 'a', 'इ': 'i', 'ई': 'i', 'उ': 'u',
    'ऊ': 'u', 'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au'
}

def romanize_generic(text: str) -> str:
    """Universal phonetic Devanagari-to-Roman transliterator for cross-lingual string matching."""
    if not text:
        return ""
    text = fix_devanagari_ocr(text)
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if i + 1 < n and text[i:i+2] in DEVA_CONSONANTS:
            c_rom = DEVA_CONSONANTS[text[i:i+2]]
            i += 2
            if i < n and text[i] == '्':
                out.append(c_rom)
                i += 1
            elif i < n and text[i] in DEVA_MATRAS:
                out.append(c_rom + DEVA_MATRAS[text[i]])
                i += 1
            else:
                out.append(c_rom + 'a')
        elif c in DEVA_CONSONANTS:
            c_rom = DEVA_CONSONANTS[c]
            i += 1
            if i < n and text[i] == '्':
                out.append(c_rom)
                i += 1
            elif i < n and text[i] in DEVA_MATRAS:
                out.append(c_rom + DEVA_MATRAS[text[i]])
                i += 1
            else:
                out.append(c_rom + 'a')
        elif c in DEVA_INDEP_VOWELS:
            out.append(DEVA_INDEP_VOWELS[c])
            i += 1
        elif c in DEVA_MATRAS:
            out.append(DEVA_MATRAS[c])
            i += 1
        else:
            out.append(c)
            i += 1
    res = ''.join(out)
    res = re.sub(r'[^a-zA-Z0-9\s]', ' ', res)
    res = re.sub(r'\s+', ' ', res)
    return res.lower().strip()


# =====================================================================
# 2. UNIVERSAL CONCEPT NORMALIZER (LANGUAGE & INTENT ONLY)
# =====================================================================

class IndexConceptNormalizer:
    """
    Universal semantic concept matcher for queries across English, Hindi, and Roman-Hindi.
    Handles linguistic markers (ordinals, counts, section synonyms, item types).
    """

    ORDINAL_MAP = {
        1: [
            "first", "1st", "pehla", "pahla", "pehli", "pahli", "पहला", "पहली", "प्रथम", 
            "सबसे पहला", "सबसे पहली"
        ],
        2: [
            "second", "2nd", "dusra", "doosra", "dusri", "doosri", "दूसरा", "दूसरी", 
            "द्वितीय"
        ],
        3: [
            "third", "3rd", "teesra", "tisra", "teesri", "तीसरा", "तीसरी", "तृतीय"
        ],
        4: [
            "fourth", "4th", "chautha", "chauthi", "चौथा", "चौथी", "चतुर्थ"
        ],
        5: [
            "fifth", "5th", "panchva", "panchwa", "पांचवां", "पाँचवाँ", "पंचम"
        ],
        6: [
            "sixth", "6th", "chhattha", "chhatha", "छठा", "छठी", "षष्ठ"
        ],
        7: [
            "seventh", "7th", "saatva", "saatwa", "सातवां", "सातवीं", "सप्तम"
        ],
        8: [
            "eighth", "8th", "aathva", "aathwa", "आठवां", "आठवीं", "अष्टम"
        ],
        9: [
            "ninth", "9th", "nauva", "nauwa", "नौवां", "नौवीं", "नवम"
        ],
        10: [
            "tenth", "10th", "dasva", "daswa", "दसवां", "दसवीं", "दशम"
        ],
        "last": [
            "last", "aakhiri", "akhiri", "antim", "अंतिम", "आखिरी", "सबसे अंतिम", 
            "सबसे आखिरी", "final"
        ]
    }

    COMMON_SECTION_PATTERNS = {
        "technical": [
            "तकनीकी", "तकनीक", "प्रौद्योगिकी", "technical", "tech", "technology", 
            "takniki", "takneeki", "praudyogiki", "takneeki lekh", "takniki lekh"
        ],
        "non_technical": [
            "गैर-तकनीकी", "गैर तकनीकी", "non-technical", "non technical", "general", 
            "gair takniki", "gair-takniki"
        ],
        "story": [
            "कहानी", "कहानियां", "कथा", "story", "stories", "fiction", "kahani", "kahaani"
        ],
        "poem": [
            "कविता", "कविताएं", "कवितायें", "पद्य", "कविताओं", "काव्य", "poem", "poems", 
            "poetry", "kavita", "kavitayein", "padya"
        ],
        "special": [
            "विशेष", "खास", "special", "focus", "feature", "vishesh"
        ],
        "tribute": [
            "श्रद्धांजलि", "स्मृति", "tribute", "tributes", "in memoriam", "shraddhanjali"
        ],
        "editorial": [
            "संपादकीय", "संदेश", "संरक्षक", "प्रस्तावना", "editorial", "foreword", 
            "preface", "message", "sampadkiya"
        ],
        "official": [
            "राजभाषा", "सरकारी", "प्रशासनिक", "official", "administrative", "rajbhasha"
        ]
    }


    @classmethod
    def clean_text(cls, text: str) -> str:
        if not text:
            return ""
        text = fix_devanagari_ocr(text)
        text = re.sub(r"[\'\"`~@#$%^*()_+=\[\]{}|\\<>.,;!?/]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def detect_ordinal(cls, query: str) -> Optional[Union[int, str]]:
        # Strip page references so "page 8" or "page 1" is never confused with an ordinal position
        q_no_page = re.sub(r'\b(?:page|prishth|पेज|पृष्ठ)\s*(?:no\.?|number|संख्या)?\s*\d+\b', ' ', query, flags=re.IGNORECASE)
        q_no_page = re.sub(r'\b\d+\s*(?:page|prishth|पेज|पृष्ठ)\s*(?:par|mein|me|पर|में)?\b', ' ', q_no_page, flags=re.IGNORECASE)
        q_norm = cls.clean_text(q_no_page).lower()
        words = q_norm.split()
        for ord_val, variants in cls.ORDINAL_MAP.items():
            for v in variants:
                if v in words or f" {v} " in f" {q_norm} ":
                    return ord_val
        return None

    @classmethod
    def match_section_concept(cls, query: str, available_sections: List[str]) -> Optional[str]:
        q_norm = cls.clean_text(query).lower()
        q_rom = romanize_generic(query)
        for sec in available_sections:
            sec_norm = cls.clean_text(sec).lower()
            sec_rom = romanize_generic(sec)
            if sec_norm in q_norm or sec_rom in q_rom:
                return sec
        # Evaluate every concept and prefer the most specific matching phrase.
        # This prevents "non technical" from being swallowed by "technical".
        concept_matches = []
        for concept, synonyms in cls.COMMON_SECTION_PATTERNS.items():
            matched_synonyms = [s for s in synonyms if s in q_norm or s in q_rom]
            if matched_synonyms:
                concept_matches.append((max(len(s) for s in matched_synonyms), concept, synonyms))
        for _, _, synonyms in sorted(concept_matches, reverse=True):
            for sec in available_sections:
                sec_clean = cls.clean_text(sec).lower()
                sec_rom = romanize_generic(sec)
                if any(s in sec_clean or s in sec_rom for s in synonyms):
                    return sec
        return None


# =====================================================================
# 3. UNIVERSAL TOC & STRUCTURE EXTRACTOR
# =====================================================================

class TOCExtractor:
    """
    Universal TOC parser that handles both:
    - Tokenized DOCX paragraphs where serial, title, colon, author, page are separate lines
    - Composite PDF lines where each entry is a single line with leader dots / colons
    Uses a robust state-machine approach.
    """

    @staticmethod
    def extract_records_from_pages(
        pages_data: List[Dict[str, Any]],
        document_id: str,
        filename: str
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not pages_data:
            return records

        # ── Phase 1: Identify TOC candidate pages ──
        toc_candidates = []
        for p in pages_data[:8]:
            p_text = p.get("text", "")
            words = p_text.split()

            has_leaders = len(re.findall(r"(\.{2,}|…+|\:{1,2})\s*\d{1,3}", p_text)) >= 2
            has_numbered_list = len(re.findall(r"^\s*\d+[\.\s\:\-]", p_text, re.MULTILINE)) >= 2
            has_toc_kw = any(kw in p_text.lower() for kw in [
                "विषय", "सूची", "अनुक्रमणिका", "contents", "table of contents",
                "index", "articles", "लेख", "कवि", "इस अंक में"
            ])

            is_editorial_board = any(kw in p_text for kw in [
                "संरक्षक", "सलाहकार", "सह संपादक", "संपादक मंडल",
                "board of editors", "editorial board", "फोन:", "ई-मेल"
            ])
            is_editorial_or_body = any(kw in p_text[:100] for kw in [
                "संदेश", "संपादकीय", "Editorial", "Foreword", "Message"
            ])

            if is_editorial_board and not has_numbered_list:
                continue

            if (has_leaders or has_numbered_list or has_toc_kw or p.get("segment_type") == "index") \
                    and len(words) < 1200 and not is_editorial_or_body:
                toc_candidates.append(p)
            elif (len(words) > 500 and not has_numbered_list) or (is_editorial_or_body and (p.get("page_number") or 1) > 4):
                break

        if not toc_candidates and pages_data:
            toc_candidates = pages_data[:2]

        # ── Phase 2: Collect all raw lines from TOC pages ──
        skip_lines = {"---", "•", "–", ""}
        all_raw_lines: List[Tuple[str, int]] = []   # (line_text, toc_page_number)
        for page in toc_candidates:
            p_num = page.get("page_number", 1)
            raw_text = page.get("text", "")
            clean_t = fix_devanagari_ocr(raw_text)
            for l in clean_t.splitlines():
                l_str = l.strip()
                if l_str and l_str not in skip_lines:
                    all_raw_lines.append((l_str, p_num))

        if not all_raw_lines:
            return records

        # ── Phase 3: Detect parsing mode ──
        bare_digit_count = sum(1 for (l, _) in all_raw_lines if re.match(r"^\d{1,3}$", l))
        bare_colon_count = sum(1 for (l, _) in all_raw_lines if l in (":", "："))
        composite_entry_count = sum(
            1 for (l, _) in all_raw_lines
            if bool(re.match(r"^\s*\d+[\.\s\:\-]", l)) and len(l.split()) > 3
        )
        is_tokenized = (bare_digit_count >= 4 and bare_colon_count >= 2) or \
                        (bare_digit_count >= 6 and composite_entry_count < 3)

        # ── Phase 4: Section indicators ──
        section_indicators = [
            "तकनीकी लेख", "गैर-तकनीकी लेख/कहानी", "गैर-तकनीकी लेख", "कहानी",
            "कविता", "कविताएँ", "विशेष", "श्रद्धांजलि", "संपादकीय", "रिपोर्ट", "राजभाषा कॉर्नर"
        ]
  
        def _is_section_heading(text: str) -> bool:
            t = text.strip()
            if not t:
                return False
            return any(t == si or t.lower() == si.lower() for si in section_indicators)

        def _build_record(serial, title, author, page, toc_p, section, sec_counters):
            clean_title = fix_devanagari_ocr(title.strip()) if title else ""
            clean_author = fix_devanagari_ocr(author.strip()) if author else None
            clean_sec = fix_devanagari_ocr(section) if section else "General"
            sec_item_idx = sec_counters.get(clean_sec, 0) + 1
            sec_counters[clean_sec] = sec_item_idx
            sec_lower = clean_sec.lower()
            if any(k in sec_lower for k in ["कविता", "poem", "poetry"]):
                entry_type = "poem"
            elif any(k in sec_lower for k in ["कहानी", "story", "fiction"]):
                entry_type = "story"
            elif any(k in sec_lower for k in ["श्रद्धांजलि", "tribute", "memoriam"]):
                entry_type = "tribute"
            elif any(k in sec_lower for k in ["रिपोर्ट", "कॉर्नर", "report", "corner"]):
                entry_type = "report"
            else:
                entry_type = "article"

            author_list = []
            if clean_author:
                for p_a in re.split(r"[,/|]|\s+एवं\s+|\s+व\s+|\s+and\s+", clean_author):
                    ca = p_a.strip()
                    if ca and ca != "-":
                        author_list.append(ca)

            return {
                "document_id": document_id,
                "source_filename": filename,
                "toc_page": toc_p,
                "serial_number": serial,
                "section_original": clean_sec,
                "section_normalized": clean_sec,
                "section_item_index": sec_item_idx,
                "title_original": clean_title,
                "title_normalized": clean_title,
                "title_roman": romanize_generic(clean_title),
                "author_original": clean_author,
                "author_normalized": clean_author,
                "author_roman": romanize_generic(clean_author) if clean_author else None,
                "author_list": author_list,
                "page_number": page,
                "type": entry_type,
            }

        current_section = "General"
        section_counters: Dict[str, int] = {}

        # ── Phase 5A: Tokenized (DOCX) state-machine parser ──
        if is_tokenized:
            cur_serial: Optional[int] = None
            cur_title = ""
            cur_author = ""
            cur_page: Optional[int] = None
            state = "EXPECT_ITEM"   # or "EXPECT_AUTHOR"

            def _flush_entry(toc_p: int):
                nonlocal cur_serial, cur_title, cur_author, cur_page, state
                if cur_title and cur_title.strip():
                    records.append(_build_record(
                        cur_serial, cur_title, cur_author, cur_page,
                        toc_p, current_section, section_counters
                    ))
                cur_serial, cur_title, cur_author, cur_page = None, "", "", None
                state = "EXPECT_ITEM"

            # Seek to actual TOC start marker
            toc_start_markers = {
                "इस अंक में", "अनुक्रमणिका", "विषय सूची", "विषय-सूची",
                "table of contents", "contents"
            }
            stop_markers = {
                "संरक्षक की कलम से", "संपादक की कलम से", "संदेश", "संपादकीय", "प्रस्तावना",
                "foreword", "message", "editorial", "", "***", "---"
            }
            noise_skip_markers = {
                "संपादक मंडल", "संरक्षक", "सलाहकार", "सह संपादक", "संपादकीय", "संपादक",
                "कवर डिजाइन", "विशेष सहयोग", "फोन:", "ई-मेलः", "ई-मेल:", "वेबसाइट:",
                "board of editors", "editorial board", "patron", "advisor", "editor",
                "email:", "phone:", "website:", "copyright", "सर्वाधिकार"
            }

            toc_start_idx = 0
            for _s_i, (_s_line, _) in enumerate(all_raw_lines):
                if _s_line.strip().lower() in {m.lower() for m in toc_start_markers}:
                    toc_start_idx = _s_i + 1
                    while toc_start_idx < len(all_raw_lines) and \
                            all_raw_lines[toc_start_idx][0].strip().lower() in {m.lower() for m in toc_start_markers}:
                        toc_start_idx += 1
                    break

            i = toc_start_idx
            while i < len(all_raw_lines):
                line, toc_p = all_raw_lines[i]

                # Stop conditions: long narrative paragraph or article body marker
                if any(line == sm or line.startswith(sm) for sm in stop_markers) or \
                        (len(records) >= 3 and len(line.split()) > 25 and not re.match(r"^\s*\d+[\.\s\:\-]", line)):
                    _flush_entry(toc_p)
                    break

                # If no serial yet, skip noise
                if cur_serial is None and any(line.startswith(nm) or nm in line for nm in noise_skip_markers):
                    i += 1
                    continue

                # Section heading
                if _is_section_heading(line):
                    _flush_entry(toc_p)
                    current_section = line
                    i += 1
                    continue

                # Bare number: serial or page
                if re.match(r"^\d{1,3}$", line):
                    num = int(line)
                    if cur_serial is None:
                        cur_serial = num
                        state = "EXPECT_ITEM"
                    elif cur_title and cur_page is None:
                        cur_page = num
                        _flush_entry(toc_p)
                    else:
                        _flush_entry(toc_p)
                        cur_serial = num
                        state = "EXPECT_ITEM"
                    i += 1
                    continue

                # Bare colon/separator
                if line in (":", "：", "-"):
                    state = "EXPECT_AUTHOR"
                    i += 1
                    continue

                # Accumulate title or author
                if cur_serial is not None:
                    if state == "EXPECT_AUTHOR":
                        cur_author = (cur_author + " " + line).strip() if cur_author else line.strip()
                    else:
                        cur_title = (cur_title + " " + line).strip()
                    i += 1
                    continue

                i += 1

            _flush_entry(all_raw_lines[-1][1] if all_raw_lines else 1)

        # ── Phase 5B: Composite (PDF) line-based parser ──
        else:
            global_serial = 0

            for line, toc_p in all_raw_lines:
                if len(records) >= 5 and len(line.split()) > 40 \
                        and not re.match(r"^\s*\d+[\.\s\:\-]", line):
                    break

                # Merge standalone page digits into previous line
                if re.match(r"^\d{1,3}$", line) and records:
                    if records[-1].get("page_number") is None:
                        records[-1]["page_number"] = int(line)
                    continue

                starts_with_num = bool(re.match(r"^\s*(\d+)[\.\s\:\-]+(.*)", line))
                has_page_end = bool(re.search(r"\d{1,3}\s*$", line) and re.search(r"[:：\-—]|\.{2,}", line))
                is_entry_line = starts_with_num or has_page_end

                if not is_entry_line:
                    if _is_section_heading(line):
                        current_section = fix_devanagari_ocr(line.strip())
                    elif len(line.split()) <= 6 and not re.search(r"\d{3,}", line):
                        current_section = fix_devanagari_ocr(line.strip())
                    continue

                # Parse serial number
                serial_num = None
                rest = line
                m_ser = re.match(r"^\s*(\d+)[\.\s\:\-]+(.*)", rest)
                if m_ser:
                    serial_num = int(m_ser.group(1))
                    rest = m_ser.group(2).strip()
                else:
                    global_serial += 1
                    serial_num = global_serial

                # Extract trailing page number
                page_target = None
                rest = re.sub(r"[\uf000-\uf8ff\s\-\.]+$", "", rest)
                m_pg = re.search(r"(\d{1,3})\s*$", rest)
                if m_pg:
                    page_target = int(m_pg.group(1))
                    rest = rest[:m_pg.start()].strip()
                # DOCX table rows arrive as "title : : : author : page" (the ":" column is a
                # cell of its own), so drop the separator left behind by the page cell.
                rest = re.sub(r"[\s:：]+$", "", rest)

                # Separate title and author
                title = ""
                author = ""
                if re.search(r"\.{2,}|…+", rest):
                    parts = re.split(r"\.{2,}|…+", rest)
                    title = parts[0].strip()
                    author = parts[1].strip() if len(parts) > 1 else ""
                elif ":" in rest or "：" in rest:
                    segs = [x.strip() for x in re.split(r"[:：]", rest) if x.strip()]
                    if len(segs) >= 2:
                        title = ": ".join(segs[:-1])
                        author = segs[-1]
                    else:
                        title = segs[0] if segs else rest.strip()
                elif " by " in rest.lower():
                    idx = rest.lower().rfind(" by ")
                    title = rest[:idx].strip()
                    author = rest[idx + 4:].strip()
                elif "-" in rest or "—" in rest:
                    parts = re.split(r"[-—]", rest)
                    title = parts[0].strip()
                    author = parts[1].strip() if len(parts) > 1 else ""
                else:
                    title = rest.strip()

                records.append(_build_record(
                    serial_num, title, author, page_target,
                    toc_p, current_section, section_counters
                ))

        logger.info(f"[TOCExtractor] Extracted {len(records)} structured TOC records for '{filename}' ({document_id})")
        return records


class StructuredIndexStore:
    """
    Document-isolated persistence for structured TOC records.
    Persists to doc_metadata/{document_id}_toc.json.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir is None:
            base = os.path.dirname(os.path.abspath(__file__))
            storage_dir = os.path.join(base, "doc_metadata")
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self._memory_store: Dict[str, List[Dict[str, Any]]] = {}
        self._load_all_from_disk()

    def _get_path(self, doc_id: str) -> str:
        safe_id = re.sub(r'[^\w\.\-]', '_', doc_id)
        return os.path.join(self.storage_dir, f"{safe_id}_toc.json")

    def _load_all_from_disk(self):
        try:
            for fname in os.listdir(self.storage_dir):
                if fname.endswith("_toc.json"):
                    fpath = os.path.join(self.storage_dir, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        doc_id = data.get("document_id") or fname.replace("_toc.json", "")
                        self._memory_store[doc_id] = data.get("records", [])
        except Exception as e:
            logger.warning(f"[StructuredIndexStore] Warning loading disk cache: {e}")

    def save_records(self, document_id: str, filename: str, records: List[Dict[str, Any]]):
        for record in records:
            self._enrich_article_record(record, document_id)
        self._memory_store[document_id] = records
        try:
            fpath = self._get_path(document_id)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump({
                    "document_id": document_id,
                    "filename": filename,
                    "records_count": len(records),
                    "records": records
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[StructuredIndexStore] Failed to write TOC cache for {document_id}: {e}")

    @staticmethod
    def _enrich_article_record(record: Dict[str, Any], document_id: str) -> None:
        """Attach deterministic, document-derived article aliases and provenance."""
        title = (record.get("title_original") or "").strip()
        aliases = {title, IndexConceptNormalizer.clean_text(title), romanize_generic(title)}
        record["entity_type"] = "article"
        record["article_id"] = record.get("article_id") or f"{document_id}:article:{record.get('serial_number', len(title))}"
        record["aliases"] = sorted(a.strip() for a in aliases if a and len(a.strip()) > 1)
        record["provenance"] = {"document_id": document_id, "toc_page": record.get("toc_page"), "source": "toc_or_index"}

    def get_records(self, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not document_id or document_id in ["all", "All Uploaded Documents", ""]:
            all_recs = []
            for doc_key, r_list in self._memory_store.items():
                if not in_scope(doc_key):
                    continue
                for record in r_list:
                    self._enrich_article_record(record, doc_key)
                all_recs.extend(r_list)
            return all_recs
        
        if document_id in self._memory_store:
            records = self._memory_store[document_id]
            for record in records:
                self._enrich_article_record(record, document_id)
            return records

        fpath = self._get_path(document_id)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    recs = data.get("records", [])
                    for record in recs:
                        self._enrich_article_record(record, document_id)
                    self._memory_store[document_id] = recs
                    return recs
            except Exception:
                pass

        return []

    def get_available_sections(self, document_id: Optional[str] = None) -> List[str]:
        records = self.get_records(document_id)
        secs = []
        seen = set()
        for r in records:
            s = r.get("section_original", "")
            if s and s not in seen and s != "General":
                seen.add(s)
                secs.append(s)
        return secs

    def delete_document(self, document_id: str):
        if document_id in self._memory_store:
            del self._memory_store[document_id]
        fpath = self._get_path(document_id)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass


# =====================================================================
# 5. DYNAMIC DETERMINISTIC INDEX QUERY ENGINE
# =====================================================================

class StructuredIndexEngine:
    """
    Deterministic Query Engine for Index-Based Questions (Ordinals, Counts,
    Author by Title, Title by Author, Page by Title, Section Listings, etc.).
    Fully generic across all uploaded PDFs.
    """

    def __init__(self, store: Optional[StructuredIndexStore] = None, embed_fn: Optional[Any] = None):
        self.store = store or StructuredIndexStore()
        self.normalizer = IndexConceptNormalizer()
        self.memory = DynamicTerminologyMemory()
        self.embed_fn = embed_fn
        self._title_embed_cache: Dict[str, Any] = {}
        self._title_translation_cache: Dict[str, List[str]] = {}

    _FIELD_MARKERS = {
        "author": (
            "author", "authors", "writer", "writers", "who wrote", "written by", "written", "poet", "poets", "creator", "creators",
            "लेखक", "लेखकों", "लेखिका", "लेखिकाएं", "रचयिता", "रचनाकार", "रचनाकारों", "कवि", "कवियों", "कवयित्री", "शायर", "किसने लिखा", "किसने रचा",
            "kisne likha", "kisne racha", "kisne write", "lekhak kaun", "lekhika kaun", "ke lekhak", "ka lekhak", "ki lekhika",
            "lekhak", "lekhika", "rachayita", "rachnakar", "rachnakaron",
            "kavi", "kavitri", "kiski rachna", "kiski kavita", "kiski kahani", "kiske dwara"
        ),
        "page": (
            "page", "pages", "page number", "start page", "पेज", "पृष्ठ", "kis page",
            "kis prishth", "किस पेज", "किस पृष्ठ", "which page", "where does"
        ),
        "category": (
            "category", "categories", "section", "sections", "वर्ग", "श्रेणी",
            "category kya", "section ka", "किस अनुभाग", "किस वर्ग"
        ),
        "title": (
            "title", "titles", "article name", "which article", "which poem", "शीर्षक",
            "लेख का नाम", "कविता का नाम", "कौन सा लेख", "कौन सी कविता",
            "kaunsa lekh", "kaun sa lekh", "kaun si kavita", "kaun sa article", "kya likha"
        ),
    }


    _ENTITY_NOISE = (
        "who", "is", "are", "wrote", "write", "author", "authors", "writer", "writers", "poet", "poets", "of", "the", "article",
        "story", "poem", "title", "page", "number", "kisne", "likha", "likhi", "kiya", "racha", "rachi",
        "ka", "ke", "ki", "mein", "me", "hai", "hain", "kaun", "kya", "wala", "wali", "on", "about", "titled", "named", "called",
        "lekh", "kavita", "kahani", "lekhak", "lekhika", "rachayita", "rachnakar", "kavi", "kavitri", "shayar", "par", "kis", "by", "written",
        "लेखक", "लेखिका", "रचयिता", "रचनाकार", "कवि", "कवयित्री", "किसने", "लिखा", "लिखी", "रचा", "रची", "लेख", "कविता", "कहानी",
        "का", "के", "की", "में", "है", "हैं", "कौन", "क्या", "पृष्ठ", "पेज", "शीर्षक", "पर", "किस", "बताओ"
    )

    # Words that only form the question around a title, never the title itself.
    # They are removed from the END (and the English lead-in from the START) of
    # the text, so a title word in the middle ("नाम", "रचना") is never touched.
    _QUESTION_TAIL = {
        "किसकी", "किसका", "किसके", "किसने", "किसको", "kiski", "kiska", "kiske", "kisne", "kisko",
        "लिखा", "लिखी", "लिखे", "रचा", "रची", "likha", "likhi", "likhe", "racha", "rachi", "rachit",
        "है", "हैं", "था", "थी", "थे", "hai", "hain", "he", "tha", "thi", "the",
        "कौन", "क्या", "kaun", "kon", "kya", "नाम", "naam", "name", "names", "बताओ", "बताइए", "बताएं", "बताएँ",
        "batao", "bataiye", "bataye", "बता", "bata", "दो", "do", "please", "plz", "pls",
        "रचना", "rachna", "रचनाकार", "rachnakar", "लेखक", "लेखिका", "lekhak", "lekhika", "कवि", "कवयित्री",
        "kavi", "kaviyitri", "kavitri", "रचयिता", "rachayita", "author", "authors", "writer", "writers",
        "poet", "poets", "कविता", "kavita", "poem", "लेख", "lekh", "article", "कहानी", "kahani", "story",
        "का", "के", "की", "ka", "ke", "ki", "वाला", "वाली", "वाले", "wala", "wali", "wale", "vala", "vali",
        "गया", "गई", "गए", "गयी", "gaya", "gayi", "gaye", "ne", "ने", "is", "was", "are", "were", "by",
    }
    _QUESTION_LEAD = {
        "who", "whose", "is", "are", "was", "were", "the", "a", "an", "writer", "writers", "author", "authors",
        "poet", "poets", "creator", "of", "wrote", "write", "written", "tell", "me", "name", "please",
        "article", "poem", "story", "titled", "called", "named", "did",
    }

    @classmethod
    def _strip_question_edges(cls, text: str) -> str:
        words = text.split()
        while words and words[-1].lower().strip(".!") in cls._QUESTION_TAIL:
            words.pop()
        if words and words[0].lower() in {"who", "whose", "tell", "please", "name"}:
            while words and words[0].lower() in cls._QUESTION_LEAD:
                words.pop(0)
        return " ".join(words)

    def _requested_field(self, query: str) -> Optional[str]:
        q = IndexConceptNormalizer.clean_text(query).lower()
        for field, markers in self._FIELD_MARKERS.items():
            if any(marker.lower() in q for marker in markers):
                return field
        return None

    def _extract_article_target(self, query: str) -> str:
        # 1. Quoted titles: match paired double quotes or paired single quotes
        quoted_matches = re.findall(r'["“„”]([^"“„”]+)["“„”]|[\'‘‚’]([^\'‘‚’]+)[\'‘‚’]', query)
        if quoted_matches:
            for m in quoted_matches:
                val = m[0] or m[1]
                if val and len(val.strip()) >= 2:
                    return val.strip()

        # 2. Possessives remove BEFORE replacing punctuation
        text = re.sub(r"['’]s\b|s['’]\b", "", query, flags=re.IGNORECASE)

        # 3. Strip prefixes like "Who wrote", "Who is the author of", "Who are the authors of"
        text = re.sub(r'^(?:who\s+(?:is\s+the\s+author\s+of|are\s+the\s+authors\s+of|wrote|is\s+author\s+of|authored))\s+', ' ', text, flags=re.IGNORECASE)

        # 4. Strip suffixes like "ke lekhak kaun hain", "ke lekhak", "ka lekhak", "kisne likha", "के लेखक कौन हैं", "kis page par hai"
        text = re.sub(r'\b(?:ke\s+lekhak|ka\s+lekhak|ki\s+lekhika|के\s+लेखक|का\s+लेखक|की\s+लेखिका)\s*(?:kaun\s+(?:hai|hain)?|कौन\s*(?:है|हैं)?)?\s*$', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:kisne\s+likha|kisne\s+likhi|kisne\s+racha|किसने\s+लिखा|किसने\s+लिखी|किसने\s+रचा)\s*(?:hai|hain|है|हैं)?\s*$', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:kis\s+page\s+par|kis\s+page|kis\s+prishth\s+par|kis\s+prishth|किस\s+पेज\s+पर|किस\s+पृष्ठ\s+पर)\s*(?:hai|hain|है|हैं)?\s*$', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:exists?\s+or\s+not|exists?|hai\s+ya\s+nahi|hai\s+kya|मौजूद\s+है|उपलब्ध\s+है)\s*$', ' ', text, flags=re.IGNORECASE)

        # 5. Punctuation aur extra symbols remove karein
        text = re.sub(r"[\"'“”‘’?:,।–—\-]", " ", text)

        # 6. Entity noise markers remove karein
        # Strip the question words around the title ("... kiski rachna hai",
        # "who is the writer of ...") before the global noise pass.
        text = self._strip_question_edges(text)
        # Devanagari vowel signs are not \w, so a plain \w boundary let "है"
        # match inside "हैं" and leave a stray "ं" in the title.
        for marker in self._ENTITY_NOISE:
            text = re.sub(r"(?<![\w\u0900-\u097F])" + re.escape(marker) + r"(?![\w\u0900-\u097F])", " ", text, flags=re.IGNORECASE)

        # 7. Extra spaces clean karein
        cleaned_target = re.sub(r"\s+", " ", text).strip()
        # Drop only stray marks at the START. A mark at the end belongs to the word
        # (माँ, कहानियां), so stripping it there turned "माँ" into "मा".
        cleaned_target = re.sub(r"^[\u0900-\u0903\s]+", "", cleaned_target).strip()
        return cleaned_target

    def _extract_author_target(self, query: str) -> str:
        """Extract author name from a TITLE_BY_AUTHOR / ARTICLE_BY_AUTHOR query, preserving proper nouns."""
        text = query.strip()
        text = re.sub(r'["\'\?:,।–—]', ' ', text)
        # "X ne kitne lekh likhe" / "X ने कितने लेख लिखे हैं" -> keep only what is before ne/ने
        m_ne = re.search(r'(?:^|\s)(?:ne|ने)\s', text, flags=re.IGNORECASE)
        if m_ne and m_ne.start() > 0:
            text = text[:m_ne.start()]
        # "how many articles did X write"
        m_did = re.search(r'\bhow\s+many\s+\w+\s+(?:did|has|have)\s+(.+?)\s+(?:write|written)\b', text, flags=re.IGNORECASE)
        if m_did:
            text = m_did.group(1)
        # "articles written by X" / "X द्वारा लिखे गए लेख"
        m_by = re.search(r'\b(?:written|authored)\s+by\s+(.+)$', text, flags=re.IGNORECASE)
        if m_by:
            text = m_by.group(1)
        m_dwara = re.search(r'^(.+?)\s+(?:द्वारा|dwara)\s', text, flags=re.IGNORECASE)
        if m_dwara:
            text = m_dwara.group(1)
        text = re.sub(r'\b(?:how\s+many|total|kul|kitne|kitni)\b|कितने|कितनी|कुल', ' ', text, flags=re.IGNORECASE)
        
        # Strip prefixes like "Which article was written by", "What did", "Who wrote"
        text = re.sub(r'^(?:which\s+(?:article|poem|story|report|work)\s+(?:was|were|is|are)?\s*written\s+by|what\s+did|which\s+(?:article|poem|story|report)\s+did)\s+', ' ', text, flags=re.IGNORECASE)
        
        # Strip suffixes like "ne kaun sa lekh likha hai", "ne kya likha", etc.
        text = re.sub(r'\bne\s+(?:kaun\s*(?:sa|si|se)\s+)?(?:lekh|kavita|kahani|report|article|poem|story)?\s*(?:likha|likhi|likhe)?\s*(?:hai|hain)?\s*$', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bने\s+(?:कौन\s*(?:सा|सी|से)\s+)?(?:लेख|कविता|कहानी|रिपोर्ट)?\s*(?:लिखा|लिखी|लिखे)?\s*(?:है|हैं)?\s*$', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:ka\s+article|ka\s+lekh|ki\s+report|ki\s+kavita|का\s+लेख|की\s+कविता|की\s+रचना)\s+(?:kaunsa|kaun\s*sa|kaunsi|kaun\s*si|कौन\s*सा|कौन\s*सी)?\s*(?:hai|hain|है|हैं)?\s*$', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:write|wrote)\s*\??\s*$', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:likha|likhi|likhe|लिखा|लिखी|लिखे)\s*(?:hai|hain|है|हैं)?\s*$', ' ', text, flags=re.IGNORECASE)
        
        # Strip honorifics and noisy functional words
        noise_words = {
            'ms', 'mrs', 'mr', 'dr', 'shri', 'sushri', 'shrimati',
            'which', 'article', 'poem', 'story', 'report', 'was', 'is', 'written', 'by', 'the', 'did',
            'ne', 'ka', 'ke', 'ki', 'hai', 'hain', 'kya', 'kaun', 'sa', 'si', 'se', 'lekh', 'kavita', 'kahani', 'report',
            'dwara', 'dwaraa', 'gaya', 'gaye', 'gayi',
            'सुश्री', 'श्रीमती', 'श्री', 'डॉ', 'ने', 'का', 'के', 'की', 'है', 'हैं', 'क्या', 'कौन', 'सा', 'सी', 'से', 'लेख', 'कविता', 'कहानी', 'रिपोर्ट', 'द्वारा', 'गया', 'गए', 'गई', 'लिखित'
        }
        words = text.split()
        cleaned_words = [w for w in words if w.lower().rstrip('.') not in noise_words and len(w.strip()) > 0]
        return ' '.join(cleaned_words).strip()


    _AUTHOR_STOPS = {
        "and", "aur", "evam", "va", "or", "dr", "ms", "mrs", "mr", "shri", "sushri", "shrimati", "kumari", "km",
        "डॉ", "सुश्री", "श्रीमती", "श्री", "कुमारी", "कु", "और", "एवं", "व", "तथा",
    }

    def _strict_author_match(self, query_author: str, cand_author: str, cand_roman: Optional[str] = None) -> bool:
        """Every word of the asked name must match a whole word of the author's name.

        Fuzzy substring scoring treats "सुनीता" and "नीता" as the same name, so for
        author lists a word only matches when it is identical, or when both romanised
        forms share the first letter and the consonant skeleton (sunita / suneeta,
        naveen / navin), or are near-identical long words.
        """
        def words(text):
            return [w for w in re.findall(r"[a-zA-Z\u0900-\u097F]+", (text or "").lower())
                    if w not in self._AUTHOR_STOPS and len(w) > 1]

        def roman(w):
            if not re.search(r"[\u0900-\u097F]", w):
                return w
            # ड़ / ढ़ are spoken as "r" (अरोड़ा = Arora)
            w = w.replace("\u095c", "र").replace("\u095d", "र").replace("ड\u093c", "र").replace("ढ\u093c", "र")
            return romanize_generic(w).lower().strip()

        q_words = words(query_author)
        c_words = words(cand_author) + words(cand_roman)
        if not q_words or not c_words:
            return False
        c_forms = [(cw, roman(cw)) for cw in c_words]
        for qw in q_words:
            qr = roman(qw)
            ok = False
            for cw, cr in c_forms:
                if qw == cw or (qr and qr == cr):
                    ok = True
                elif qr and cr and qr[0] == cr[0]:
                    qs, cs = self._consonant_skeleton(qr), self._consonant_skeleton(cr)
                    if len(qs) >= 2 and qs == cs:
                        ok = True
                    elif min(len(qr), len(cr)) >= 5 and difflib.SequenceMatcher(None, qr, cr).ratio() >= 0.85:
                        ok = True
                if ok:
                    break
            if not ok:
                return False
        return True

    def _answer_author_works(self, query: str, author_q: str, records: List[Dict[str, Any]]):
        """Count + list every work of an author from the TOC records.

        "सुनीता ने कितने लेख लिखे?" ->
            सुनीता अरोड़ा ने कुल 2 रचनाएँ लिखी हैं:
            1. <title> (पृष्ठ 12)
            2. <title> (पृष्ठ 40)
        Different people with the same first name are listed separately.
        """
        q_low = query.lower()
        is_hindi = bool(re.search(r"[\u0900-\u097F]|\b(?:ne|kitne|kitni|likhe|likha|lekh|kavita|hai|hain)\b", q_low))
        want_poem = any(k in q_low for k in ["poem", "kavita", "कविता", "कविताएं", "कवितायें", "काव्य"])
        want_story = any(k in q_low for k in ["story", "stories", "kahani", "कहानी", "कहानियां"])

        hits = []
        for rec in records:
            a = rec.get("author_original")
            if not a:
                continue
            if not self._strict_author_match(author_q, a, rec.get("author_roman")):
                continue
            sec = (rec.get("section_original") or "").lower()
            rtype = (rec.get("type") or "").lower()
            if want_poem and not ("poem" in rtype or "कविता" in sec):
                continue
            if want_story and not ("story" in rtype or "कहानी" in sec):
                continue
            hits.append(rec)

        if not hits:
            return "", []

        # group by (author, document) keeping TOC order, drop duplicate titles
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        seen = set()
        for rec in hits:
            key_t = (rec.get("document_id"), rec.get("author_original"), rec.get("title_original"))
            if key_t in seen:
                continue
            seen.add(key_t)
            groups.setdefault((rec.get("author_original"), rec.get("document_id")), []).append(rec)

        multi_doc = len({d for _, d in groups}) > 1
        noun_hi_pl = "कविताएँ" if want_poem else ("कहानियाँ" if want_story else "रचनाएँ")
        noun_hi_sg = "कविता" if want_poem else ("कहानी" if want_story else "रचना")
        noun_en_pl = "poems" if want_poem else ("stories" if want_story else "works")
        noun_en_sg = "poem" if want_poem else ("story" if want_story else "work")
        blocks = []
        for (author, doc_id), recs in groups.items():
            doc_label = re.sub(r"\.(?:docx?|pdf)$", "", str(doc_id or ""), flags=re.IGNORECASE)
            lines = []
            for i, r in enumerate(recs, 1):
                pg = r.get("page_number")
                pg_txt = (f" (पृष्ठ {pg})" if is_hindi else f" (page {pg})") if pg else ""
                lines.append(f"{i}\\. {r.get('title_original')}{pg_txt}  ")
            n = len(recs)
            if is_hindi:
                head = (f"**{author}** ने कुल **{n}** {noun_hi_pl} लिखी हैं" if n > 1
                        else f"**{author}** ने **1** {noun_hi_sg} लिखी है")
                head += f" ({doc_label} में):" if multi_doc and doc_label else ":"
            else:
                head = f"**{author}** wrote **{n}** {noun_en_pl if n > 1 else noun_en_sg}"
                head += f" (in {doc_label}):" if multi_doc and doc_label else ":"
            blocks.append(head + "\n\n" + "\n".join(lines))
        return "\n\n".join(blocks), hits

    def _extract_topic_keywords(self, query: str) -> str:
        """Extract the topic/subject from a topic-search query like 'AI se related article kisne likhe' or 'ai related all articles'."""
        text = re.sub(r'\brelated\s+top\b', 'related to', query.strip(), flags=re.IGNORECASE)
        text = re.sub(r'[\"\'\'\?:,।–—]', ' ', text)
        # Strip known intent/noise phrases
        noise_patterns = [
            r'\b(?:se\s+related|related\s+to|related|relted)\b',
            r'\b(?:all|saari|saare|sabhi|sab|jitni|jitne|jitna|pure|sare|total|kul)\b',
            r'\b(?:article|articles|lekh|lekhon|kavita|kavitaye|kavitayein|kahani|report|answers?)\b',
            r'\b(?:kisne|kisna|kaun|kya|kiski|kiska|kiske|likhe|likha|likhi|likh|racha|rachi|rachna)\b',
            r'\b(?:hai|hain|he|h|tha|the|thi|batao|bataiye|naam\s+do|list\s+do|do|de|den|dein|dijiye)\b',
            r'\b(?:ke|ka|ki|se|ne|me|mein|par|ko|to|about|on|in|the|a|an|of|is|are|was|were|who|what|which|bhi)\b',
            r'\b(?:wrote|written|write|author|authors|writer)\b',
        ]
        for pat in noise_patterns:
            text = re.sub(pat, ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _consonant_skeleton(w: str) -> str:
        w = w.lower().strip()
        w = re.sub(r'c(?=[eiy])', 's', w)
        w = w.replace("j", "g").replace("z", "s").replace("sh", "s").replace("kh", "k").replace("th", "t").replace("dh", "d").replace("ph", "f").replace("bh", "b").replace("v", "w").replace("ai", "e").replace("au", "o").replace("q", "k").replace("c", "k").replace("x", "ks")
        w = w.replace("m", "n").replace("h", "")
        w = re.sub(r"[aeiouy]+", "", w)
        return re.sub(r'(.)\1+', r'\1', w)

    def _score_author_match(self, query_author: str, cand_author: str, cand_roman: Optional[str] = None) -> float:
        """Computes accurate author match score taking into account co-authors and transliterations."""
        if not query_author or not cand_author:
            return 0.0
        stops = {
            "and", "aur", "evam", "va", "or", "to", "the", "of", "dr", "ms", "mrs", "mr", "shri", "sushri", "shrimati",
            "डॉ", "सुश्री", "श्रीमती", "श्री", "और", "एवं", "व", "तथा"
        }
        q_words = [w for w in re.findall(r'[a-zA-Z\u0900-\u097F]+', query_author.lower()) if w not in stops and len(w) > 1]
        cand_text = f"{cand_author} {cand_roman or ''}"
        c_words = [w for w in re.findall(r'[a-zA-Z\u0900-\u097F]+', cand_text.lower()) if w not in stops and len(w) > 1]
        
        if not q_words or not c_words:
            return 0.0

        q_skels = [self._consonant_skeleton(w) for w in q_words]
        c_skels = [self._consonant_skeleton(w) for w in c_words]

        matched_q = 0
        used_c_indices = set()
        for i, qw in enumerate(q_words):
            matched = False
            qs = q_skels[i] if i < len(q_skels) else ""
            for j, cw in enumerate(c_words):
                if j in used_c_indices:
                    continue
                cs = c_skels[j] if j < len(c_skels) else ""
                # 1. Exact original word match
                if qw == cw:
                    matched = True
                    used_c_indices.add(j)
                    break
                # 2. High fuzzy match on original words
                if len(qw) >= 4 and len(cw) >= 4 and difflib.SequenceMatcher(None, qw, cw).ratio() >= 0.80:
                    matched = True
                    used_c_indices.add(j)
                    break
                # 3. Exact consonant skeleton match
                if qs and cs and qs == cs:
                    matched = True
                    used_c_indices.add(j)
                    break
                # 4. High-confidence prefix match (length >= 4 and high ratio)
                if len(qw) >= 4 and len(cw) >= 4 and (qw.startswith(cw[:4]) or cw.startswith(qw[:4])) and difflib.SequenceMatcher(None, qw, cw).ratio() >= 0.75:
                    matched = True
                    used_c_indices.add(j)
                    break
            if matched:
                matched_q += 1

        if not q_words:
            return 0.0
        raw_score = matched_q / len(q_words)
        # If query has multiple author words (e.g. "Rahul Sharma"), matching only a common surname ("Sharma")
        # gives raw_score = 0.50, which is insufficient evidence of the specific author.
        if len(q_words) >= 2 and raw_score < 0.60:
            return 0.0
        return raw_score

    # These are grammatical/linking words, not title evidence.  Keeping them
    # out of the coverage calculation prevents a title from matching merely
    # because both strings contain common Hindi/English glue words.
    _TITLE_STOP_TOKENS = {
        "a", "an", "the", "and", "or", "of", "for", "in", "on", "to", "by",
        "ka", "ke", "ki", "mein", "men", "me", "se", "aur", "hai", "hain",
        "article", "lekh", "poem", "kavita", "story", "kahani"
    }

    @staticmethod
    def _title_phonetic_key(value: str) -> str:
        """Return a compact, generic Roman-Hindi/English sound key."""
        value = value.lower()
        value = re.sub(r"[^a-z0-9]", "", value)
        # A bare "c" is "k" ("covid" ~ "kovida") or "s" before e/i ("social" ~
        # "soshala"); "ch" keeps its own sound.  Hindi anusvara before b/p is
        # romanized "n" ("kunbha") where people type "m" ("kumbh"/"palampur").
        # English "-tion" is written "-शन" ("migration" ~ "maigreshana").
        value = re.sub(r"tion(?=s?$)", "shan", value)
        value = value.replace("ch", "\x00")
        value = re.sub(r"c(?=[eiy])", "s", value).replace("c", "k").replace("\x00", "ch")
        value = re.sub(r"n(?=[bp])", "m", value)
        # Normalize spelling variants before removing vowels.  This is a
        # phonetic rule set, not a document or title alias table.
        value = (value.replace("sch", "s").replace("sh", "s")
                      .replace("ph", "f").replace("bh", "b")
                      .replace("dh", "d").replace("th", "t")
                      .replace("kh", "k").replace("ch", "c")
                      .replace("v", "w").replace("z", "s")
                      .replace("q", "k"))
        value = re.sub(r"[aeiouy]+", "", value)
        return re.sub(r"(.)\1+", r"\1", value)

    @staticmethod
    def _spelled_key(word: str) -> str:
        """Like the phonetic key, but keeps vowels (and drops the final inherent "a")."""
        w = re.sub(r"[^a-z0-9]", "", word.lower())
        w = re.sub(r"tion(?=s?$)", "shan", w)
        w = w.replace("ch", "\x00")
        w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k").replace("\x00", "ch")
        w = re.sub(r"n(?=[bp])", "m", w)
        w = (w.replace("sh", "s").replace("ph", "f").replace("bh", "b").replace("dh", "d")
              .replace("th", "t").replace("kh", "k").replace("v", "w").replace("z", "j")
              .replace("ee", "i").replace("oo", "u").replace("aa", "a"))
        w = re.sub(r"(.)\1+", r"\1", w)
        return w[:-1] if len(w) > 3 and w.endswith("a") else w

    def _spelled_similarity(self, query_tokens: List[str], record: Dict[str, Any]) -> float:
        roman = (record.get("title_roman") or romanize_generic(record.get("title_original", ""))).lower()
        cand = [self._spelled_key(t) for t in re.findall(r"[a-z0-9]+", roman)
                if t not in self._TITLE_STOP_TOKENS and len(t) > 1]
        if not cand:
            return 0.0
        best = [max(difflib.SequenceMatcher(None, self._spelled_key(q), c).ratio() for c in cand)
                for q in query_tokens]
        return sum(best) / len(best)

    def _title_concept_tokens(self, value: str) -> set:
        """Return title tokens plus document terminology aliases.

        A TOC title is often written in Hindi while its question is written in
        English (or vice versa).  The terminology memory already owns those
        language equivalents; use it as retrieval evidence without storing a
        document-specific translated title.
        """
        roman_value = romanize_generic(value).lower()
        tokens = {
            token for token in re.findall(r"[a-z0-9]+", roman_value)
            if token not in self._TITLE_STOP_TOKENS and len(token) > 1
        }

        words = IndexConceptNormalizer.clean_text(value).split()
        for start in range(len(words)):
            # Prefer the longest known phrase, so "Artificial Intelligence"
            # is resolved as one concept instead of two unrelated words.
            for size in range(min(4, len(words) - start), 0, -1):
                phrase = " ".join(words[start:start + size])
                key = self.memory.resolve_term(phrase)
                if not key or key not in self.memory.terms:
                    continue
                term = self.memory.terms[key]
                aliases = {term.get("canonical", "")}
                for field in ("english", "hindi", "roman", "ocr_variants"):
                    aliases.update(term.get(field, set()))
                for alias in aliases:
                    tokens.update(
                        token for token in re.findall(r"[a-z0-9]+", romanize_generic(alias).lower())
                        if token not in self._TITLE_STOP_TOKENS and len(token) > 1
                    )
                break
        return tokens

    def _title_match_details(self, query_title: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Score a title only from title evidence; never from author fields."""
        query_clean = self.normalizer.clean_text(query_title).lower()
        candidate_clean = self.normalizer.clean_text(record.get("title_original", "")).lower()
        query_roman = romanize_generic(query_title).lower()
        candidate_roman = (record.get("title_roman") or romanize_generic(record.get("title_original", ""))).lower()

        if not query_clean or not candidate_clean:
            return {"lexical_score": 0.0, "phonetic_score": 0.0, "token_coverage": 0.0, "final_score": 0.0, "match_method": "empty"}
        if query_clean == candidate_clean or query_roman == candidate_roman:
            return {"lexical_score": 1.0, "phonetic_score": 1.0, "token_coverage": 1.0, "final_score": 1.0, "match_method": "exact_normalized"}

        q_tokens = [t for t in re.findall(r"[a-z0-9]+", query_roman) if t not in self._TITLE_STOP_TOKENS and len(t) > 1]
        c_tokens = [t for t in re.findall(r"[a-z0-9]+", candidate_roman) if t not in self._TITLE_STOP_TOKENS and len(t) > 1]
        q_tokens = list(dict.fromkeys(q_tokens))
        c_tokens = list(dict.fromkeys(c_tokens))
        if not q_tokens or not c_tokens:
            return {"lexical_score": 0.0, "phonetic_score": 0.0, "token_coverage": 0.0, "final_score": 0.0, "match_method": "no_informative_tokens"}

        def token_matches(left: str, right: str) -> bool:
            if left == right:
                return True
            left_key, right_key = self._title_phonetic_key(left), self._title_phonetic_key(right)
            if len(left_key) >= 3 and left_key == right_key:
                return True
            # Short words differ only by the inherent final "a" of the Devanagari
            # romanization ("char" ~ "chara", "log" ~ "loga", "media" ~ "midiya").
            if (len(left_key) == 2 and left_key == right_key
                    and min(len(left), len(right)) >= 3 and left[0] == right[0]):
                return True
            return len(left_key) >= 4 and len(right_key) >= 4 and difflib.SequenceMatcher(None, left_key, right_key).ratio() >= 0.84

        # Keep the ordinary title words for phrase matching, then add aliases
        # only for concept coverage.  This prevents a shared concept such as
        # AI from being treated as an exact full-title match.
        concept_q_tokens = self._title_concept_tokens(query_title)
        concept_c_tokens = self._title_concept_tokens(record.get("title_original", ""))
        matched = sum(1 for token in q_tokens if any(token_matches(token, candidate) for candidate in c_tokens))
        coverage = matched / len(q_tokens)
        concept_overlap = len(concept_q_tokens & concept_c_tokens)
        concept_coverage = concept_overlap / len(concept_q_tokens) if concept_q_tokens else 0.0
        q_phrase_key = self._title_phonetic_key(query_roman)
        c_phrase_key = self._title_phonetic_key(candidate_roman)
        phrase_score = difflib.SequenceMatcher(None, q_phrase_key, c_phrase_key).ratio() if q_phrase_key and c_phrase_key else 0.0
        # Short titles need an almost exact phonetic phrase, while a long title
        # is accepted only when its informative-token coverage is very high.
        if len(q_tokens) == 1:
            final = max(coverage if phrase_score >= 0.88 else 0.0, phrase_score if phrase_score >= 0.92 else 0.0)
        else:
            final = 0.80 * coverage + 0.20 * phrase_score
        return {
            "lexical_score": coverage,
            "phonetic_score": phrase_score,
            "token_coverage": coverage,
            "concept_coverage": concept_coverage,
            "final_score": final,
            "match_method": "token_phonetic"
        }

    def _resolve_title_confidently(self, query_title: str, records: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Rank every document-local title and enforce threshold plus margin."""
        ranked = []
        for record in records:
            details = self._title_match_details(query_title, record)
            ranked.append((details["final_score"], record, details))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            return None, {"best_score": 0.0, "second_best_score": 0.0, "score_margin": 0.0, "candidates": []}

        best_score, best_record, best_details = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = best_score - second_score
        token_count = len([t for t in re.findall(r"[a-z0-9]+", romanize_generic(query_title)) if t not in self._TITLE_STOP_TOKENS and len(t) > 1])
        threshold = 0.92 if token_count <= 1 else (0.78 if token_count <= 3 else 0.80)
        required_margin = 0.0 if best_score >= 0.98 else 0.10
        diagnostics = {
            "best_score": best_score,
            "second_best_score": second_score,
            "score_margin": margin,
            "threshold": threshold,
            "candidates": [
                {"title": rec.get("title_original"), "author": rec.get("author_original"), **details}
                for score, rec, details in ranked[:10]
            ]
        }
        # Keep the API diagnostic compact, but log every evaluated record so a
        # production mismatch can be audited without rerunning with a debugger.
        for _, record, candidate in ranked:
            logger.info("[CANDIDATE] TITLE=%r AUTHOR=%r LEXICAL_SCORE=%.3f PHONETIC_SCORE=%.3f TOKEN_COVERAGE=%.3f FINAL_SCORE=%.3f", record.get("title_original"), record.get("author_original"), candidate["lexical_score"], candidate["phonetic_score"], candidate["token_coverage"], candidate["final_score"])
        # A Roman-Hindi query can use translated tokens (for example a Hindi
        # word in place of an English loanword) while retaining the title's
        # overall sound sequence.  Accept that only when both the phrase-level
        # phonetic signal and informative-token coverage are strong and the
        # nearest alternative is clearly behind it.
        strong_phonetic_title = (
            best_details["phonetic_score"] >= 0.82
            and best_details["token_coverage"] >= 0.50
            and margin >= 0.10
        )
        # Roman Hindi spellings are a lossy transliteration of Devanagari
        # titles (for example, "Aasaan"/"aasaan" for "आसान").  A near-exact
        # whole-title phonetic sequence plus several matching title tokens is
        # stronger evidence than the strict lexical score alone.
        high_confidence_roman_title = (
            best_details["phonetic_score"] >= 0.90
            and best_details["token_coverage"] >= 0.40
            and margin >= 0.10
        )
        # Technical titles often retain multiple English product/domain words
        # inside a Hindi title.  The transliteration may differ ("boot" vs
        # "buta"), so accept only a clearly separated, multi-token phonetic
        # title match rather than requiring an exact spelling.
        strong_cross_script_phonetic_title = (
            best_details["phonetic_score"] >= 0.72
            and best_details["token_coverage"] >= 0.28
            and margin >= 0.15
        )
        # A short or partial title ("Tulsi", "Palampur", "spring boot migration")
        # names one article when every informative word of the question is
        # found in that title and none of them occurs in any other title.
        # (Ranking is by final score, which stays low for a one-word query, so
        # this looks at every title's word coverage, not only the top one.)
        informative = [t for t in re.findall(r"[a-z0-9]+", romanize_generic(query_title).lower())
                       if t not in self._TITLE_STOP_TOKENS and len(t) > 1]
        full = [item for item in ranked if item[2].get("token_coverage", 0.0) >= 1.0]
        touched = [item for item in ranked if item[2].get("token_coverage", 0.0) > 0.0]
        unique_full_token_match = False
        if informative and (len(informative) >= 2 or len(informative[0]) >= 4) and full:
            if len(full) == 1 and len(touched) == 1:
                unique_full_token_match = True
            else:
                # Vowel-less keys can collide ("tulsi" ~ "talasha"); compare the
                # words with their vowels and accept only a clear winner.
                scored = sorted(((self._spelled_similarity(informative, item[1]), item) for item in touched),
                                key=lambda x: x[0], reverse=True)
                top_sim, top_item = scored[0]
                next_sim = scored[1][0] if len(scored) > 1 else 0.0
                if top_item in full and top_sim >= 0.80 and top_sim - next_sim >= 0.20:
                    full = [top_item]
                    unique_full_token_match = True
        if unique_full_token_match and not (best_score >= threshold and margin >= required_margin):
            best_score, best_record, best_details = full[0]
        if (best_score >= threshold and margin >= required_margin) or strong_phonetic_title or high_confidence_roman_title or strong_cross_script_phonetic_title or unique_full_token_match:
            diagnostics["match_method"] = best_details["match_method"]
            logger.info("[INDEX MATCH] SELECTED=%r CONFIDENCE=%.3f MARGIN=%.3f", best_record.get("title_original"), best_score, margin)
            return best_record, diagnostics
        diagnostics["reason"] = "below_title_threshold" if best_score < threshold else "ambiguous_title_match"
        logger.info("[INDEX NO MATCH] REASON=%s BEST_SCORE=%.3f THRESHOLD=%.3f SECOND_BEST=%.3f MARGIN=%.3f", diagnostics["reason"], best_score, threshold, second_score, margin)
        return None, diagnostics

    def _translate_title_for_lookup(self, title: str) -> List[str]:
        """Return local-model Hindi title variants for a final, verified lookup.

        This is intentionally a fallback, not a source of facts: every model
        output is sent back through the deterministic TOC title matcher before
        it can identify an author or any other metadata.
        """
        cache_key = IndexConceptNormalizer.clean_text(title).lower()
        if not cache_key:
            return []
        if cache_key in self._title_translation_cache:
            return self._title_translation_cache[cache_key]

        # Hindi input is already handled by exact/phonetic matching.  Calling
        # the model only for Latin-script titles avoids unnecessary latency.
        if not re.search(r"[A-Za-z]", title):
            self._title_translation_cache[cache_key] = []
            return []

        variants: List[str] = []
        try:
            from urllib import request as urlrequest

            model_name = os.getenv("GENERATOR_MODEL_NAME", "llama3.2")
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

            payload = {
                "model": model_name,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Translate this document title into natural Hindi (Devanagari). "
                        "Preserve product names and technical acronyms. Return only the title, "
                        "with no explanation or quotation marks.\n\nTitle: " + title
                    )
                }],
                "stream": False,
                "options": {"temperature": 0}
            }
            request = urlrequest.Request(
                f"{ollama_base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urlrequest.urlopen(request, timeout=12) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            if response.status == 200:
                translated = response_data.get("message", {}).get("content", "")
                translated = translated.strip().strip("`\"'“”‘’ ")
                # A translation response must be a short title, never prose.
                if 2 <= len(translated) <= 200 and "\n" not in translated:
                    variants.append(translated)
        except Exception as exc:
            logger.info("[TitleTranslation] Local translation unavailable: %s", exc)

        self._title_translation_cache[cache_key] = variants
        return variants

    def _article_match(self, target: str, record: Dict[str, Any], document_id: Optional[str]) -> Dict[str, Any]:
        if document_id and record.get("document_id") != document_id:
            return {"score": 0.0, "match_type": "wrong_document"}
        if not target or not target.strip():
            return {"score": 0.0, "match_type": "no_target"}

        t_clean = IndexConceptNormalizer.clean_text(target).lower()
        t_rom = romanize_generic(target).lower()

        rec_orig = record.get("title_original", "").lower()
        rec_clean = IndexConceptNormalizer.clean_text(record.get("title_original", "")).lower()
        rec_rom = (record.get("title_roman") or romanize_generic(rec_orig)).lower()

        # 1. Exact title matches
        if t_clean == rec_clean or t_rom == rec_rom:
            return {"score": 1.0, "match_type": "exact_title"}

        # 2. Substring containment matches
        if len(t_clean) >= 4 and (t_clean in rec_clean or rec_clean in t_clean):
            return {"score": 0.95, "match_type": "substring_clean"}
        if len(t_rom) >= 4 and (t_rom in rec_rom or rec_rom in t_rom):
            return {"score": 0.95, "match_type": "substring_roman"}

        # 3. Consonant skeleton and token overlap
        stops = {"in", "on", "at", "and", "or", "to", "is", "this", "its", "the", "of", "ka", "ke", "ki", "me", "mein", "men", "par", "hai", "aur", "by", "a", "an", "se", "ko", "ne"}
        rec_rom_tokens = {t for t in rec_rom.split() if t not in stops and len(t) > 1}
        t_rom_tokens = {t for t in t_rom.split() if t not in stops and len(t) > 1}

        best_score = 0.0
        t_skels = {self._consonant_skeleton(t) for t in t_rom_tokens if len(t) > 2}
        t_skels.discard("")
        rec_skels = {self._consonant_skeleton(t) for t in rec_rom_tokens if len(t) > 2}
        rec_skels.discard("")

        if t_skels and rec_skels:
            ov = len(t_skels & rec_skels)
            if ov >= 1:
                ratio = ov / len(t_skels)
                if ratio >= 0.50 or ov >= 2:
                    score = ratio * 0.70 + (ov / len(rec_skels)) * 0.30
                    best_score = max(best_score, score)

        return {
            "score": best_score if best_score >= 0.40 else 0.0,
            "match_type": "cross_lingual_overlap" if best_score >= 0.40 else "insufficient"
        }

    def resolve_article(self, query: str, document_id: Optional[str] = None, target: Optional[str] = None) -> Dict[str, Any]:
        target = target or self._extract_article_target(query)
        ranked = []
        records = self.store.get_records(document_id)
        for record in records:
            result = self._article_match(target, record, document_id=document_id)
            if result["score"]:
                ranked.append({**result, "record": record})
        ranked.sort(key=lambda item: item["score"], reverse=True)

        if not ranked:
            return {"target_entity": target, "resolved": None, "confidence": 0.0, "candidates": []}
        best = ranked[0]
        ambiguous = len(ranked) > 1 and best["score"] - ranked[1]["score"] < 0.05
        return {"target_entity": target, "resolved": None if ambiguous else best["record"], "confidence": best["score"], "match_type": best["match_type"], "candidates": [{"article_id": x["record"].get("article_id"), "title": x["record"].get("title_original"), "confidence": x["score"]} for x in ranked[:5]], "ambiguous": ambiguous}

    def classify_intent(self, query: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        norm_q = self.normalizer.clean_text(query).lower()
        # Common Roman-English typing slip: "related top X" means
        # "related to X" and should remain an index topic-list request.
        norm_q = re.sub(r'\brelated\s+top\b', 'related to', norm_q)
        ordinal = self.normalizer.detect_ordinal(query)
        available_sections = self.store.get_available_sections(document_id)
        matched_section = self.normalizer.match_section_concept(query, available_sections)

        # 1. COUNT Detection
        is_count = (
            any(k in norm_q for k in ["how many", "count", "kitne", "kitni", "kul kitne", "kitna", "कितने", "कुल कितने", "संख्या", "गिनती"])
            or bool(re.search(r'\btotal\s+(?:\w+\s+)*(?:articles?|lekh|kavita|poems?|stories|kahani|items?|records?|documents?|docs?)\b', norm_q))
            or bool(re.search(r'\b(?:total|kul)\s+(?:kitne|kitni|count)\b', norm_q))
            or bool(re.search(r'\b(?:kitne|kitni)\s+(?:\w+\s+)*(?:articles?|lekh|kavita|poems?|stories|kahani|items?|records?|documents?|docs?)\b', norm_q))
            or bool(re.search(r'\b(?:articles?|lekh|kavita|poems?|documents?)\s+(?:count|sankhya|kitne|kitni|total|numbers?)\b', norm_q))
            or bool(re.search(r'\btotal\s+(?:articles?\s+|documents?\s+)?present\b', norm_q))
            or bool(re.search(r'\b(?:is\s+)?document\s+(?:mein|me)\s+(?:total\s+)?kitne\b', norm_q))
            # English patterns: "no of X", "number of X", "total no of X", "total number of X"
            or bool(re.search(r'\b(?:total\s+)?(?:no\.?|number)\s+of\s+(?:\w+\s+)*(?:articles?|lekh|kavita|poems?|stories|kahani|items?|records?|documents?|docs?)\b', norm_q))
            or bool(re.search(r'\b(?:total\s+no\.?|total\s+number)\b', norm_q))
        )

        # 2. LIST Detection
        is_list = any(k in norm_q for k in [
            "list", "name all", "names of", "name of", "titles of", "title of", "names", "titles",
            "sabke naam", "sab batao", "sabhi batao", "naam do", "naam batao", "naam bataiye",
            "kaun kaun se", "kaun-kaun se", "saari", "saare", "sabhi", "sab",
            "jitni", "jitne", "jitna", "kitni kavita", "kitne lekh", "kavitaye", "kavitaon ke naam", "lekhon ke naam",
            "सूची", "सभी लेख", "सभी कविताएं", "बताओ", "सबके नाम", "सारी", "सभी", "जितनी", "जितने",
            "list do", "list karo", "all poems", "all articles", "all stories", "all items"
        ])

        # 3. ARTICLE_BY_AUTHOR (TITLE_BY_AUTHOR) Detection - Check BEFORE AUTHOR_BY_ARTICLE!
        is_title_by_author = (
            any(k in norm_q for k in [
                "which article was written by", "which poem was written by", "which story was written by", "which report was written by",
                "which article did", "which poem did", "which story did", "which report did",
                "what did", "article written by", "poem written by", "story written by", "report written by",
                "ka article kaunsa", "kaun sa lekh", "kaunsa lekh", "kaun si report", "kaunsi report",
                "kaun si kavita", "kaunsi kavita", "kaun sa article", "kaun si kahani", "kaunsi kahani",
                "kya likha", "kaunsa article likha", "kaun sa article likha", "kaunsa lekh likha", "kaun sa lekh likha",
                "कौन सा लेख लिखा", "कौन सी कविता लिखी", "ने कौन सा लेख", "ने कौन सी कविता", "ने क्या लिखा", "द्वारा लिखा गया"
            ])
            or bool(re.search(r'\bwhat\s+did\s+.+?\s+write\b', norm_q))
            or bool(re.search(r'\bwhich\s+(?:article|poem|story|report|work)\s+(?:was\s+written\s+by|did)\s+', norm_q))
            or bool(re.search(r'\bne\s+(?:kaun\s*(?:sa|si)|kya|likha|likhi)\b', norm_q))
            or bool(re.search(r'\bने\s+(?:कौन\s*(?:सा|सी)|क्या|लिखा|लिखी)', norm_q))
        )

        # 3B. AUTHOR_WORKS: "सुनीता ने कितने लेख लिखे", "sunita ne kitne lekh likhe",
        # "how many articles did Sunita write", "articles written by Sunita"
        has_ne_likh = bool(re.search(r'(?:^|\s)(?:ne|ने)\s+.*(?:likh|लिख|rach|रच)', norm_q))
        is_author_works = (
            has_ne_likh
            or bool(re.search(r'\bhow\s+many\s+\w+\s+(?:did|has|have)\s+.+?\s+(?:write|written)\b', norm_q))
            or bool(re.search(r'\b(?:articles?|poems?|stories|works?)\s+(?:written|authored)\s+by\s+\S+', norm_q))
            or bool(re.search(r'(?:द्वारा\s+लिखे|द्वारा\s+लिखी|dwara\s+likh)', norm_q))
        )
        if is_author_works:
            is_title_by_author = True
        wants_author_count = is_author_works and (is_count or is_list or bool(re.search(r'कितनी|कितने|kitni|kitne|how\s+many|कौन[\s-]*कौन|kaun[\s-]*kaun|सभी|सारे|सूची|नाम|\bnaam\b|\blist\b|\ball\b|\bwhich\b', norm_q)))

        # 4. AUTHOR_BY_ARTICLE (AUTHOR_BY_TITLE) Detection
        has_who_word = (
            any(k in norm_q for k in [
                "kisne", "kisna", "kiski", "kiska", "kiske", "who", "whose", "lekhak kaun", "lekhika kaun", "kaun lekhak",
                "किसने", "किसकी", "किसका", "किसके", "कौन लेखक", "लेखक कौन", "लेखिका कौन"
            ])
            or bool(re.search(r'\b(?:kisne|kisna|kiski|kiska|kiske|who)\b', norm_q))
            or bool(re.search(r'(?:किसने|किसकी|किसका|किसके)', norm_q))
        )
        is_author = (
            not is_title_by_author and (
                has_who_word
                or any(k in norm_q for k in [
                    "who wrote", "author", "authors", "writer", "writers", "poet", "poets", "creator", "creators",
                    "kisne likha", "kisne racha", "kisne write", "lekhak kaun", "lekhika kaun", "ke lekhak", "ka lekhak", "ki lekhika",
                    "lekhak", "lekhika", "rachnakar", "rachnakaron", "rachayita", "kavitri", "kiski rachna", "kiski kavita",
                    "लेखक", "लेखकों", "लेखिका", "रचयिता", "रचनाकार", "रचनाकारों", "कवियों", "कवयित्री", "शायर", "किसने लिखा", "किसने रचा", "द्वारा लिखित", "किसका लेख", "written by"
                ])
                or bool(re.search(r'(?<![\w\u0900-\u097F])(?:kavi|कवि|कवियों)(?![\w\u0900-\u097F])', norm_q))
            )
        )

        # 5. PAGE Detection (asking for page of an article, or asking which article is on a page)
        m_page_num = re.search(r'\b(?:page|prishth|पेज|पृष्ठ)\s*(?:no\.?|number|संख्या)?\s*(\d+)\b', norm_q) or \
                     re.search(r'\b(\d+)\s*(?:page|prishth|पेज|पृष्ठ)\b', norm_q)
        is_page = any(k in norm_q for k in [
            "page", "pages", "page number", "kis page par", "kis page", "kis prishth par", "kis prishth", "where does", "start page",
            "पेज", "पृष्ठ", "किस पेज पर", "किस पेज", "किस पृष्ठ पर", "किस पृष्ठ", "which page"
        ])
        is_article_on_page = bool(m_page_num) and (
            any(k in norm_q for k in [
                "which", "kaun", "kaunsa", "kaun sa", "kaun si", "kya", "article on", "lekh on", "poem on", "kavita on",
                "कौन", "कौन सा", "कौन सी", "क्या", "बताओ", "नाम", "kisne", "who", "author"
            ])
            or bool(re.search(r'\b(?:article|lekh|kavita|poem|story|kahani)\s+(?:on|par|mein|me)\b', norm_q))
        )

        # 6. EXISTS Detection
        is_exists = any(k in norm_q for k in [
            "exists or not", "exist or not", "exists", "exist", "hai ya nahi", "hai kya", "मौजूद है", "उपलब्ध है"
        ])

        # 7. TOPIC_SEARCH Detection
        is_topic_search = (
            not is_count and (
                any(k in norm_q for k in [
                    "se related", "related to", "related article", "related lekh",
                    "ke baare mein", "ke baare me", "ke bare mein", "ke bare me",
                    "से संबंधित", "से रिलेटेड", "के बारे में",
                    "topic", "subject", "vishay", "विषय",
                ])
                or bool(re.search(r'\b(?:related|relted)\b', norm_q))
                or bool(re.search(r'\b(?:article|articles|lekh|kavita)\s+(?:on|about)\b', norm_q))
                or bool(re.search(r'\b(?:on|about)\s+.+?\s+(?:article|articles|lekh|kavita)\b', norm_q))
            )
        )

        if is_article_on_page and m_page_num:
            intent = "ARTICLE_BY_PAGE"
        elif ordinal is not None:
            if ordinal == 1:
                intent = "FIRST_ITEM"
            elif ordinal == "last":
                intent = "LAST_ITEM"
            else:
                intent = "NTH_ITEM"
        elif is_author_works:
            intent = "TITLE_BY_AUTHOR"
        elif is_count:
            intent = "COUNT"
        elif is_page:
            intent = "PAGE_BY_TITLE"
        elif is_title_by_author:
            intent = "TITLE_BY_AUTHOR"
        elif is_author:
            intent = "AUTHOR_BY_TITLE"
        elif is_exists:
            intent = "EXISTS"
        elif is_topic_search:
            intent = "TOPIC_SEARCH"
        elif matched_section:
            intent = "ARTICLES_BY_SECTION"
        elif is_list:
            intent = "ARTICLES_BY_SECTION"
        else:
            intent = "STRUCTURED_LOOKUP"

        if intent == "ARTICLE_BY_PAGE" and m_page_num:
            cleaned_entity = m_page_num.group(1)
        elif intent == "TITLE_BY_AUTHOR":
            cleaned_entity = self._extract_author_target(query)
        elif intent == "TOPIC_SEARCH":
            cleaned_entity = self._extract_topic_keywords(query)
        else:
            cleaned_entity = self._extract_article_target(query)

        requested_field = self._requested_field(query)
        if intent == "TITLE_BY_AUTHOR":
            requested_field = "title"
        elif intent == "AUTHOR_BY_TITLE":
            requested_field = "author"
        elif intent == "ARTICLE_BY_PAGE":
            requested_field = "title"
        target_entity = cleaned_entity if requested_field in {"author", "page", "category"} else None

        return {
            "intent": intent,
            "wants_author_count": bool(intent == "TITLE_BY_AUTHOR" and wants_author_count),
            "ordinal": ordinal,
            "section": matched_section,
            "entity_candidate": cleaned_entity if (cleaned_entity and (len(cleaned_entity) >= 1 if intent == "ARTICLE_BY_PAGE" else len(cleaned_entity) >= 2)) else None,
            "requested_field": requested_field,
            "target_entity_type": "article" if target_entity else None,
            "target_entity": target_entity,
            "is_list_query": is_list,
            "is_count_query": is_count
        }

    def _compute_base_score(self, query_text: str, candidate_text: str, roman_candidate: Optional[str] = None) -> float:
        if not query_text or not candidate_text:
            return 0.0
        q_norm = self.normalizer.clean_text(query_text)
        c_norm = self.normalizer.clean_text(candidate_text)
        q_rom = romanize_generic(query_text)
        c_rom = roman_candidate or romanize_generic(candidate_text)
        if q_norm == c_norm or q_rom == c_rom:
            return 1.0
        if len(q_norm) >= 3 and (q_norm in c_norm or c_norm in q_norm):
            shorter, longer = (q_norm, c_norm) if len(q_norm) <= len(c_norm) else (c_norm, q_norm)
            if len(shorter) / max(len(longer), 1) >= 0.40:
                return 0.95
        if len(q_rom) >= 4 and (q_rom in c_rom or c_rom in q_rom):
            shorter, longer = (q_rom, c_rom) if len(q_rom) <= len(c_rom) else (c_norm, q_rom)
            if len(shorter) / max(len(longer), 1) >= 0.40:
                return 0.95

        resolved_key = self.memory.resolve_term(query_text)
        if resolved_key and resolved_key in self.memory.terms:
            rec = self.memory.terms[resolved_key]
            all_aliases = {rec["canonical"].lower()} | {a.lower() for a in rec["english"]} | {a.lower() for a in rec["hindi"]} | {a.lower() for a in rec["roman"]}
            for alias in all_aliases:
                alias_clean = self.normalizer.clean_text(alias)
                if alias_clean in c_norm or c_norm in alias_clean or alias_clean in c_rom:
                    shorter_a, longer_a = (alias_clean, c_norm) if len(alias_clean) <= len(c_norm) else (c_norm, alias_clean)
                    if len(shorter_a) / max(len(longer_a), 1) >= 0.40:
                        return 0.95

        stops = {"in", "the", "of", "ka", "ke", "ki", "me", "mein", "men", "hai", "a", "an", "is", "by", "par", "ko", "ne", "on", "who", "wrote", "se", "aur", "and", "with", "for", "about", "regarding", "from", "into"}
        q_tokens = {t.lower() for t in (q_norm.split() + q_rom.split()) if t.lower() not in stops and len(t) > 1}
        c_tokens = {t.lower() for t in (c_norm.split() + c_rom.split()) if t.lower() not in stops and len(t) > 1}
        if not q_tokens:
            return 0.0

        def _phon_stem(w: str) -> str:
            w = w.lower().strip()
            w = re.sub(r"[aeiou]+", lambda m: m.group(0)[0], w)
            w = w.replace("j", "g").replace("z", "s").replace("sh", "s").replace("kh", "k").replace("th", "t").replace("dh", "d").replace("ph", "f").replace("bh", "b")
            return w.rstrip("a")

        def _token_match(qt: str, ct: str) -> bool:
            if qt == ct:
                return True
            if len(qt) >= 4 and len(ct) >= 4 and difflib.SequenceMatcher(None, qt, ct).ratio() >= 0.75:
                return True
            qs = _phon_stem(qt)
            cs = _phon_stem(ct)
            if qs and cs and qs == cs:
                return True
            if qs and cs and len(qs) >= 4 and len(cs) >= 4 and (qs in cs or cs in qs) and min(len(qs), len(cs)) / max(len(qs), len(cs)) >= 0.75:
                return True
            sk_q = self._consonant_skeleton(qt)
            sk_c = self._consonant_skeleton(ct)
            if sk_q and sk_c and len(sk_q) >= 2 and len(sk_c) >= 2 and len(qt) >= 3 and len(ct) >= 3:
                if sk_q == sk_c:
                    return True
                if len(sk_q) >= 3 and len(sk_c) >= 3 and (sk_q in sk_c or sk_c in sk_q) and min(len(sk_q), len(sk_c)) / max(len(sk_q), len(sk_c)) >= 0.75:
                    return True
                if len(sk_q) >= 4 and len(sk_c) >= 4 and difflib.SequenceMatcher(None, sk_q, sk_c).ratio() >= 0.85:
                    return True
            return False

        score_orig = 0.0
        if q_tokens:
            matched_orig = sum(1 for qt in q_tokens if any(_token_match(qt, ct) for ct in c_tokens))
            score_orig = matched_orig / len(q_tokens)
        if score_orig >= 0.50:
            return score_orig

        ratio_deva = difflib.SequenceMatcher(None, q_norm, c_norm).ratio()
        ratio_rom = difflib.SequenceMatcher(None, q_rom, c_rom).ratio()
        max_ratio = max(ratio_deva, ratio_rom)
        if max_ratio >= 0.65:
            return max_ratio * 0.92

        return 0.0

    def _score_match(self, query_text: str, candidate_text: str, roman_candidate: Optional[str] = None, _depth: int = 0) -> float:
        if not query_text or not candidate_text:
            return 0.0
        base_score = self._compute_base_score(query_text, candidate_text, roman_candidate)
        if _depth > 0 or base_score >= 0.75:
            return base_score

        # Multi-segment matching for titles with subtitles (e.g. "Main Title – Subtitle" or "Main Title : Subtitle")
        q_parts = [p.strip() for p in re.split(r'[:–—\-\(\)]', query_text) if len(p.strip()) >= 3]
        c_parts = [p.strip() for p in re.split(r'[:–—\-\(\)]', candidate_text) if len(p.strip()) >= 3]

        best = base_score
        for qp in [query_text] + q_parts:
            for cp in [candidate_text] + c_parts:
                if qp == query_text and cp == candidate_text:
                    continue
                sc = self._compute_base_score(qp, cp, None)
                if sc > best:
                    best = sc
                    if best >= 0.90:
                        return best
        return best

    def _semantic_title_match_confident(self, query_title: str, records: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Use embeddings only with a strict margin and independent title evidence."""
        if not self.embed_fn:
            return None, {"reason": "embedding_unavailable", "embedding_score": 0.0, "embedding_margin": 0.0}
        try:
            import numpy as np
            query_embedding = np.array(self.embed_fn([query_title])[0])
            query_embedding /= max(float(np.linalg.norm(query_embedding)), 1e-9)
            ranked = []
            for record in records:
                title = record.get("title_original", "")
                if not title:
                    continue
                if title not in self._title_embed_cache:
                    vector = np.array(self.embed_fn([title])[0])
                    self._title_embed_cache[title] = vector / max(float(np.linalg.norm(vector)), 1e-9)
                embedding_score = float(np.dot(query_embedding, self._title_embed_cache[title]))
                lexical = self._title_match_details(query_title, record)
                ranked.append((embedding_score, record, lexical))
            ranked.sort(key=lambda item: item[0], reverse=True)
            if not ranked:
                return None, {"reason": "no_titles", "embedding_score": 0.0, "embedding_margin": 0.0}
            best_score, best_record, lexical = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else 0.0
            margin = best_score - second_score
            # A translation/paraphrase may have no shared Roman/Devanagari
            # tokens at all.  The multilingual encoder is therefore evaluated
            # by its absolute similarity and the gap to the next TOC title;
            # it is never allowed to select merely the nearest candidate.
            lexical_signal = lexical["token_coverage"]
            concept_signal = lexical.get("concept_coverage", 0.0)
            has_independent_title_signal = max(lexical_signal, concept_signal) >= 0.15
            # Terminology memory is populated from the indexed document, so
            # it may supply an additional cross-language signal without a
            # hand-maintained translation table.
            has_cross_lingual_concept_signal = concept_signal >= 0.15
            semantic_threshold = 0.60 if has_cross_lingual_concept_signal else (0.64 if has_independent_title_signal else 0.66)
            required_margin = 0.08 if has_cross_lingual_concept_signal else (0.10 if has_independent_title_signal else 0.12)
            if best_score >= semantic_threshold and margin >= required_margin:
                return best_record, {"embedding_score": best_score, "embedding_margin": margin, "match_method": "semantic_plus_title_evidence", "concept_coverage": concept_signal}
            return None, {"reason": "semantic_threshold_or_margin", "embedding_score": best_score, "embedding_margin": margin, "lexical_score": lexical_signal, "concept_coverage": concept_signal}
        except Exception as exc:
            logger.warning("[SemanticTitleMatch] Error: %s", exc)
            return None, {"reason": "embedding_error", "embedding_score": 0.0, "embedding_margin": 0.0}
        try:
            import numpy as np
            q_emb = np.array(self.embed_fn([query_title])[0])
            q_norm = q_emb / max(float(np.linalg.norm(q_emb)), 1e-9)

            best_rec = None
            best_sim = 0.0

            for rec in records:
                t = rec.get("title_original", "")
                if not t:
                    continue
                if t not in self._title_embed_cache:
                    t_emb = np.array(self.embed_fn([t])[0])
                    self._title_embed_cache[t] = t_emb / max(float(np.linalg.norm(t_emb)), 1e-9)
                t_norm = self._title_embed_cache[t]

                sim = float(np.dot(q_norm, t_norm))
                if sim > best_sim and sim >= threshold:
                    best_sim = sim
                    best_rec = rec
            return best_rec, best_sim
        except Exception as e:
            logger.warning(f"[SemanticTitleMatch] Error: {e}")
            return None, 0.0

    def _resolve_title_for_index_lookup(self, query_title: str, records: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Resolve a title through deterministic, semantic, then translated checks."""
        best_rec, diagnostics = self._resolve_title_confidently(query_title, records)
        if best_rec:
            return best_rec, diagnostics

        sem_rec, sem_diagnostics = self._semantic_title_match_confident(query_title, records)
        diagnostics["semantic"] = sem_diagnostics
        if sem_rec:
            return sem_rec, diagnostics

        translated_variants = self._translate_title_for_lookup(query_title)
        diagnostics["translation_variants"] = translated_variants
        for translated_title in translated_variants:
            translated_rec, translated_diagnostics = self._resolve_title_confidently(translated_title, records)
            if translated_rec:
                diagnostics["match_method"] = "verified_local_translation"
                diagnostics["translation"] = translated_diagnostics
                return translated_rec, diagnostics
        return None, diagnostics

    def execute_index_query(
        self,
        query: str,
        document_id: Optional[str] = None,
        query_mode: str = "AUTO",
        conversation_article_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        records = self.store.get_records(document_id)
        if not records:
            return None

        # Filter by document if document is mentioned in the query
        doc_filter_applied = None
        if not document_id:
            q_lower = query.lower()
            all_docs = list(self.store._memory_store.keys())
            for d in all_docs:
                d_clean = re.sub(r'[\._\-\d]+', ' ', d.lower()).strip()
                d_tokens = [w for w in d_clean.split() if len(w) > 3 and w not in ["final", "edition", "docx", "pdf", "hindi", "report", "proforma", "part"]]
                if d_tokens and any(t in q_lower for t in d_tokens):
                    filtered_by_doc = [r for r in records if r.get("document_id") == d]
                    if filtered_by_doc:
                        records = filtered_by_doc
                        doc_filter_applied = d
                        break

        slot_data = self.classify_intent(query, document_id)
        intent = slot_data["intent"]
        ordinal = slot_data["ordinal"]
        matched_section = slot_data["section"]
        entity_cand = slot_data["entity_candidate"]

        logger.info(
            f"[INDEX] ── NEW QUERY ──\n"
            f"  query      = '{query}'\n"
            f"  intent     = {intent}\n"
            f"  section    = '{matched_section}'\n"
            f"  entity     = '{entity_cand}'\n"
            f"  ordinal    = {ordinal}\n"
            f"  records    = {len(records)} available"
        )

        answer_text = ""
        matched_records: List[Dict[str, Any]] = []

        # =====================================================================
        # INTENT 1: FIRST_ITEM, LAST_ITEM, NTH_ITEM (Ordinal queries)
        # =====================================================================
        if intent in ["FIRST_ITEM", "LAST_ITEM", "NTH_ITEM"] and ordinal is not None:
            filtered = records
            if matched_section:
                filtered = [r for r in records if r.get("section_original") == matched_section or r.get("section_normalized") == matched_section]
            
            if not filtered:
                filtered = records

            target_idx = 0
            if ordinal == "last":
                target_idx = len(filtered) - 1
            elif isinstance(ordinal, int) and 1 <= ordinal <= len(filtered):
                target_idx = ordinal - 1
            else:
                target_idx = 0

            target_rec = filtered[target_idx]
            matched_records.append(target_rec)

            title = target_rec["title_original"]
            author = target_rec["author_original"]
            page = target_rec["page_number"]

            is_hindi = bool(re.search(r"[\u0900-\u097F]|kisne|pehla|kaunsa|hai|kya", query.lower()))
            if is_hindi:
                if author:
                    answer_text = f"{title} (लेखक: {author}, पृष्ठ: {page})"
                else:
                    answer_text = f"{title} (पृष्ठ: {page})"
            else:
                if author:
                    answer_text = f"{title} (Author: {author}, Page: {page})"
                else:
                    answer_text = f"{title} (Page: {page})"

        # =====================================================================
        # INTENT 2: COUNT Queries
        # =====================================================================
        elif intent == "COUNT":
            filtered = records
            section_label = ""
            if matched_section:
                filtered = [r for r in records if r.get("section_original") == matched_section or r.get("section_normalized") == matched_section]
                section_label = matched_section

            count_val = len(filtered)
            matched_records = filtered[:5] if filtered else records[:5]

            is_hindi = bool(re.search(r"[\u0900-\u097F]|kitne|kitni|kul|lekh|hain|hai|btao", query.lower()))
            doc_name_clean = doc_filter_applied.replace(".docx", "").replace(".pdf", "") if doc_filter_applied else ""
            doc_label_hi = f" '{doc_name_clean}'" if doc_name_clean else ""
            doc_label_en = f" in '{doc_name_clean}'" if doc_name_clean else ""

            if section_label:
                if is_hindi:
                    answer_text = f"दस्तावेज़{doc_label_hi} में '{section_label}' अनुभाग में कुल {count_val} लेख / रचनाएं उपलब्ध हैं।"
                else:
                    answer_text = f"There are {count_val} articles/items listed under '{section_label}'{doc_label_en}."
            else:
                if is_hindi:
                    answer_text = f"दस्तावेज़{doc_label_hi} में कुल {count_val} लेख / रचनाएं उपलब्ध हैं।"
                else:
                    answer_text = f"There are {count_val} articles/items listed{doc_label_en}."

        # =====================================================================
        # INTENT 3: AUTHOR_BY_TITLE (e.g. "Dark Pattern kisne likha hai?", "Who wrote Natural Language Processing?")
        # =====================================================================
        elif intent == "AUTHOR_BY_TITLE" and entity_cand:
            best_rec, title_diagnostics = self._resolve_title_for_index_lookup(entity_cand, records)
            slot_data["entity_resolution"] = title_diagnostics

            if best_rec:
                matched_records.append(best_rec)
                author = best_rec.get("author_original")
                title = best_rec["title_original"]
                page = best_rec["page_number"]
                src_doc = best_rec.get("document_id", "")

                is_hindi = bool(re.search(r"[\u0900-\u097F]|kisne|likha|hai|lekhak|ke lekhak", query.lower()))
                # Add source doc note if answer is from a different document
                doc_note_hi = f" [स्रोत: {src_doc}]" if document_id and src_doc and src_doc != document_id else ""
                doc_note_en = f" [Source: {src_doc}]" if document_id and src_doc and src_doc != document_id else ""

                if not author or author.strip() in ("", "-", "None", "अज्ञात लेखक"):
                    if is_hindi:
                        answer_text = f"लेख '{title}' के लेखक की जानकारी उपलब्ध नहीं है (पृष्ठ: {page}){doc_note_hi}।"
                    else:
                        answer_text = f"Author information is not available for '{title}' (Page: {page}){doc_note_en}."
                else:
                    if is_hindi:
                        answer_text = f"{author} (लेख: '{title}', पृष्ठ: {page}){doc_note_hi}"
                    else:
                        answer_text = f"{author} (Article: '{title}', Page: {page}){doc_note_en}"

        # =====================================================================
        # INTENT 4: TITLE_BY_AUTHOR (e.g. "Himani Garg ne kaunsa lekh likha?")
        # =====================================================================
        elif intent == "TITLE_BY_AUTHOR" and entity_cand and slot_data.get("wants_author_count"):
            answer_text, matched_records = self._answer_author_works(query, entity_cand, records)

        elif intent == "TITLE_BY_AUTHOR" and entity_cand:
            # Detect requested work type if specified in query
            req_type = None
            q_lower = query.lower()
            if any(k in q_lower for k in ["poem", "poems", "poetry", "kavita", "kavitayein", "कविता", "कविताएं", "कवितायें", "काव्य"]):
                req_type = "poem"
            elif any(k in q_lower for k in ["story", "stories", "kahani", "kahaani", "कहानी", "कहानियां"]):
                req_type = "story"
            elif any(k in q_lower for k in ["report", "रिपोर्ट"]):
                req_type = "report"
            elif any(k in q_lower for k in ["article", "articles", "lekh", "लेख"]):
                req_type = "article"

            candidates = []
            target_records = records
            for rec in target_records:
                if not rec.get("author_original"):
                    continue
                author_sc = self._score_author_match(entity_cand, rec["author_original"], rec.get("author_roman"))
                general_sc = self._score_match(entity_cand, rec["author_original"], rec.get("author_roman"))
                base_score = max(author_sc, general_sc)
                if base_score >= 0.65:
                    type_match = False
                    if req_type:
                        rec_type = rec.get("type", "").lower()
                        rec_sec = (rec.get("section_original") or "").lower()
                        if req_type == "poem" and ("poem" in rec_type or "कविता" in rec_sec):
                            type_match = True
                        elif req_type == "story" and ("story" in rec_type or "कहानी" in rec_sec):
                            type_match = True
                        elif req_type == "report" and ("report" in rec_type or "रिपोर्ट" in rec_sec or "कॉर्नर" in rec_sec or "corner" in rec_sec):
                            type_match = True
                        elif req_type == "article" and ("article" in rec_type or "लेख" in rec_sec):
                            type_match = True

                    candidates.append((base_score, type_match, rec))

            if candidates:
                candidates.sort(key=lambda x: (round(x[0], 2), 1 if x[1] else 0), reverse=True)
                top_score = round(candidates[0][0], 2)
                top_type_match = candidates[0][1]

                # Collect all works by this author with matching top score and type match
                top_recs = [c[2] for c in candidates if round(c[0], 2) == top_score and c[1] == top_type_match]

                for r in top_recs:
                    matched_records.append(r)

                author = top_recs[0]["author_original"]
                is_hindi = bool(re.search(r"[\u0900-\u097F]|kaunsa|kaun sa|kya|likha|likhi|hai|hain|lekh|kavita|kahani|report", query.lower()))

                if len(top_recs) == 1:
                    title = top_recs[0]["title_original"]
                    page = top_recs[0]["page_number"]
                    if is_hindi:
                        answer_text = f"{title} (लेखक: {author}, पृष्ठ: {page})"
                    else:
                        answer_text = f"{title} (Author: {author}, Page: {page})"
                else:
                    titles_hi = " एवं ".join([f"'{r['title_original']}' (पृष्ठ: {r['page_number']})" for r in top_recs])
                    titles_en = " and ".join([f"'{r['title_original']}' (Page: {r['page_number']})" for r in top_recs])
                    if is_hindi:
                        answer_text = f"{titles_hi} (लेखक: {author})"
                    else:
                        answer_text = f"{titles_en} (Author: {author})"

        # =====================================================================
        # INTENT 4B: ARTICLE_BY_PAGE (e.g. "Page 8 par kaunsa lekh hai?", "Which article is on page 8?")
        # =====================================================================
        elif intent == "ARTICLE_BY_PAGE" and entity_cand:
            try:
                target_page = int(entity_cand)
            except (ValueError, TypeError):
                target_page = -1

            page_matches = [r for r in records if r.get("page_number") == target_page]
            is_hindi = bool(re.search(r"[\u0900-\u097F]|kisne|kaun|hai|kya|batao|lekh", query.lower()))
            has_who = any(k in query.lower() for k in ["who", "author", "kisne", "lekhak", "लेखक", "किसने"])

            if page_matches:
                matched_records = page_matches
                if len(page_matches) == 1:
                    r = page_matches[0]
                    t = r["title_original"]
                    a = r.get("author_original")
                    auth_str = f" (लेखक: {a})" if a else ""
                    auth_str_en = f" (Author: {a})" if a else ""
                    if has_who and a:
                        if is_hindi:
                            answer_text = f"पृष्ठ {target_page} के लेख '{t}' के लेखक {a} हैं।"
                        else:
                            answer_text = f"The author of the article on page {target_page} ('{t}') is {a}."
                    else:
                        if is_hindi:
                            answer_text = f"दस्तावेज़ के पृष्ठ {target_page} पर '{t}'{auth_str} स्थित है।"
                        else:
                            answer_text = f"On page {target_page}: '{t}'{auth_str_en}."
                else:
                    items_hi = [f"'{r['title_original']}'" + (f" (लेखक: {r['author_original']})" if r.get("author_original") else "") for r in page_matches]
                    items_en = [f"'{r['title_original']}'" + (f" (Author: {r['author_original']})" if r.get("author_original") else "") for r in page_matches]
                    if is_hindi:
                        answer_text = f"दस्तावेज़ के पृष्ठ {target_page} पर निम्न रचनाएं स्थित हैं: {', '.join(items_hi)}।"
                    else:
                        answer_text = f"On page {target_page}: {', '.join(items_en)}."
            else:
                if is_hindi:
                    answer_text = f"दस्तावेज़ में पृष्ठ {target_page} पर कोई रचना सूचीबद्ध नहीं है।"
                else:
                    answer_text = f"No article is listed on page {target_page} in the document index."

        # =====================================================================
        # INTENT 5: PAGE_BY_TITLE (e.g. "Dark Pattern kis page par hai?")
        # =====================================================================
        elif intent == "PAGE_BY_TITLE" and entity_cand:
            best_rec, title_diagnostics = self._resolve_title_for_index_lookup(entity_cand, records)
            slot_data["entity_resolution"] = title_diagnostics

            if best_rec:
                matched_records.append(best_rec)
                title = best_rec["title_original"]
                page = best_rec["page_number"]

                is_hindi = bool(re.search(r"[\u0900-\u097F]|kis|page|prishth|par", query.lower()))
                if is_hindi:
                    answer_text = f"'{title}' पृष्ठ {page} पर स्थित है।"
                else:
                    answer_text = f"'{title}' starts on page {page}."

        # =====================================================================
        # INTENT 5B: EXISTS (Check if article exists)
        # =====================================================================
        elif intent == "EXISTS" and entity_cand:
            best_rec, title_diagnostics = self._resolve_title_for_index_lookup(entity_cand, records)
            slot_data["entity_resolution"] = title_diagnostics

            is_hindi = bool(re.search(r"[\u0900-\u097F]|hai|kya|batao|lekh", query.lower()))
            if best_rec:
                matched_records.append(best_rec)
                title = best_rec["title_original"]
                page = best_rec["page_number"]
                author = best_rec.get("author_original")
                auth_str_hi = f", लेखक: {author}" if author else ""
                auth_str_en = f", Author: {author}" if author else ""
                if is_hindi:
                    answer_text = f"हाँ, '{title}' दस्तावेज़ में पृष्ठ {page} पर मौजूद है{auth_str_hi}।"
                else:
                    answer_text = f"Yes, '{title}' is present in the document on page {page}{auth_str_en}."
            else:
                if is_hindi:
                    answer_text = f"दस्तावेज़ में '{entity_cand}' से संबंधित कोई रचना नहीं मिली।"
                else:
                    answer_text = f"No article related to '{entity_cand}' was found in the document."
                matched_records = []

        # =====================================================================
        # INTENT 6: TOPIC_SEARCH (e.g. "AI se related article kisne likhe", "ai related all articles")
        # =====================================================================
        elif intent == "TOPIC_SEARCH" and entity_cand:
            topic_text = entity_cand.strip()
            # Clean noise words from topic_text
            topic_clean_words = [w for w in re.findall(r'\b\w+\b', topic_text.lower()) if w not in {"all", "saari", "saare", "sabhi", "sab", "jitni", "jitne", "jitna", "pure", "sare", "total", "kul", "se", "related", "relted", "article", "articles", "lekh", "kisne", "likhe", "likha", "the", "of", "in", "a", "an", "and", "ke", "ka", "ki", "hai", "hain", "me", "mein", "bhi", "document", "dastavez"}]
            if not topic_clean_words:
                topic_clean_words = re.findall(r'\b\w+\b', topic_text.lower())

            # Build a set of search tokens from the topic keywords using phonetic romanization
            # and DynamicTerminologyMemory - no hardcoded translation map.
            topic_trans_tokens = set()
            # Resolve the complete topic before splitting it into words.  A
            # terminology entry such as "Cyber Security" may have a Hindi
            # alias which is a multi-word title signal.
            if hasattr(self, 'memory') and self.memory:
                resolved = self.memory.resolve_term(topic_text)
                if resolved and resolved in self.memory.terms:
                    term = self.memory.terms[resolved]
                    topic_trans_tokens.add(term.get("canonical", "").lower())
                    for field in ("english", "hindi", "roman", "ocr_variants"):
                        for alias in term.get(field, []):
                            if alias:
                                topic_trans_tokens.add(alias.lower())
            for tok in topic_clean_words:
                topic_trans_tokens.add(tok.lower())
                # Add roman phonetic of any Devanagari token
                if re.search(r'[\u0900-\u097F]', tok):
                    topic_trans_tokens.add(romanize_generic(tok).lower())
                # Ask DynamicTerminologyMemory for known aliases
                if hasattr(self, 'memory') and self.memory:
                    resolved = self.memory.resolve_term(tok)
                    if resolved and hasattr(self.memory, 'terms') and resolved in self.memory.terms:
                        term = self.memory.terms[resolved]
                        for field in ('english', 'hindi', 'roman', 'ocr_variants'):
                            for alias in term.get(field, []):
                                if alias:
                                    topic_trans_tokens.add(alias.lower())
                                    topic_trans_tokens.add(romanize_generic(alias).lower())

            topic_matches = []
            for rec in records:
                title_orig = rec["title_original"]
                title_clean = self.normalizer.clean_text(title_orig).lower()
                title_rom = (rec.get("title_roman") or romanize_generic(title_orig)).lower()
                title_words = set(re.findall(r'\b\w+\b', title_clean))
                title_rom_words = set(re.findall(r'\b\w+\b', title_rom))

                hits = 0
                for tt in topic_trans_tokens:
                    # Short acronym / token (e.g. "ai", "ui", "ux", "ml", "iot")
                    if re.match(r'^[a-z0-9]{1,3}$', tt):
                        if tt in title_rom_words or tt in title_words:
                            hits += 2
                    else:
                        # Devanagari or longer token
                        if tt in title_words or (len(tt) >= 3 and tt in title_clean) or (len(tt) >= 4 and tt in title_rom):
                            hits += 1

                if hits > 0:
                    topic_matches.append((hits, rec))

            topic_matches.sort(key=lambda x: x[0], reverse=True)
            if topic_matches:
                top_recs = [m[1] for m in topic_matches]
                matched_records = top_recs

                is_hindi = bool(re.search(r'[\u0900-\u097F]|kisne|likhe|likha|kaunsa|batao|bataiye|lekh|kavita', query.lower()))
                titles_formatted = []
                for i, r in enumerate(top_recs, 1):
                    auth_str = f" — {r['author_original']}" if r.get("author_original") else ""
                    pg_str = f" (पृष्ठ {r['page_number']})" if r.get("page_number") else f" (Page {r['page_number']})"
                    titles_formatted.append(f"{i}. {r['title_original']}{auth_str}{pg_str}")

                if len(top_recs) == 1:
                    r = top_recs[0]
                    auth_str = f", लेखक: {r['author_original']}" if r.get("author_original") else ""
                    pg_str = f", पृष्ठ: {r['page_number']}" if r.get("page_number") else ""
                    answer_text = f"लेख: '{r['title_original']}'{auth_str}{pg_str}"
                else:
                    topic_display = " ".join(topic_clean_words) if topic_clean_words else topic_text
                    if is_hindi:
                        header = f"दस्तावेज़ में '{topic_display}' से संबंधित {len(top_recs)} लेख मिले:\n"
                    else:
                        header = f"Found {len(top_recs)} articles related to '{topic_display}':\n"
                    answer_text = header + "\n".join(titles_formatted)

        # =====================================================================
        # INTENT 7: ARTICLES_BY_SECTION (List all items in section)
        # =====================================================================
        elif intent == "ARTICLES_BY_SECTION":
            filtered = records
            if matched_section:
                filtered = [r for r in records if r.get("section_original") == matched_section or r.get("section_normalized") == matched_section]
            
            matched_records = filtered
            if filtered:
                titles_formatted = []
                for i, r in enumerate(filtered, 1):
                    auth_str = f" - {r['author_original']}" if r.get("author_original") else ""
                    pg_str = f" (पृष्ठ {r['page_number']})" if r.get("page_number") else ""
                    titles_formatted.append(f"{i}. {r['title_original']}{auth_str}{pg_str}")
                answer_text = "\n".join(titles_formatted)

        # =====================================================================
        # INTENT 8: STRUCTURED_LOOKUP (General entity/title match fallback)
        # =====================================================================
        elif intent == "STRUCTURED_LOOKUP" and entity_cand:
            scored_recs = []
            for rec in records:
                sc = self._score_match(entity_cand, rec["title_original"], rec.get("title_roman"))
                if sc >= 0.35:
                    scored_recs.append((sc, rec))
            scored_recs.sort(key=lambda x: x[0], reverse=True)
            if scored_recs:
                top_sc = scored_recs[0][0]
                top_recs = [r[1] for r in scored_recs if r[0] >= top_sc - 0.15]
                matched_records = top_recs
                if len(top_recs) == 1:
                    r = top_recs[0]
                    auth_str = f", लेखक: {r['author_original']}" if r.get("author_original") else ""
                    pg_str = f", पृष्ठ: {r['page_number']}" if r.get("page_number") else ""
                    answer_text = f"लेख: '{r['title_original']}'{auth_str}{pg_str}"
                else:
                    titles_formatted = []
                    for i, r in enumerate(top_recs, 1):
                        auth_str = f" — {r['author_original']}" if r.get("author_original") else ""
                        pg_str = f" (पृष्ठ {r['page_number']})" if r.get("page_number") else ""
                        titles_formatted.append(f"{i}. {r['title_original']}{auth_str}{pg_str}")
                    answer_text = f"दस्तावेज़ में '{entity_cand}' से संबंधित निम्न रचनाएं मिलीं:\n" + "\n".join(titles_formatted)

        if not answer_text or not matched_records:
            logger.info("[INDEX MATCH] matched_record = None (Will route to normal RAG)")
            return None

        logger.info(f"[INDEX MATCH] matched_record = {matched_records[0].get('title_original')} (Page {matched_records[0].get('page_number')})")

        # Format evidence chunks for frontend
        evidence_chunks = []
        for r in matched_records:
            evidence_chunks.append({
                "chunk_id": f"{r['document_id']}_toc_p{r['toc_page']}_s{r['serial_number']}",
                "text": f"[{r['section_original']}] {r['serial_number']}. {r['title_original']} : {r.get('author_original', '')} (Page {r['page_number']})",
                "page": r["toc_page"],
                "section": r["section_original"],
                "document_id": r.get("document_id"),
                "chunk_type": "index",
                "score": 1.0
            })

        return {
            "answer": answer_text,
            "evidence": evidence_chunks,
            "matched_records": matched_records,
            "slot_data": slot_data
        }