"""
Comprehensive Document Extractor, Cleaner, Metadata Recognizer, and Chunking Engine.
Consolidates PDF/DOCX page extraction, Devanagari normalization, dynamic year/metadata
extraction, and structure-aware chunking into a single unified module.
"""

import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
import logging
from html import unescape
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET
import pypdf

logger = logging.getLogger("Extractor")
logger.setLevel(logging.INFO)

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx"}
DEVA_TO_ASCII_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text Normalization & Ligature Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def fix_devanagari_ligatures(text: str) -> str:
    """Normalizes broken Indian script encodings common in PDF font streams."""
    if not text:
        return ""
    
    # Unicode canonical composition
    text = unicodedata.normalize('NFKC', text)
    
    # Common PDF extraction artifact replacements for Hindi
    replacements = {
        'ि': 'ि',
        'ी': 'ी',
        'े': 'े',
        'ै': 'ै',
        'ो': 'ो',
        'ौ': 'ौ',
        '्': '्',
        'ं': 'ं',
        'ँ': 'ँ',
        '़': '़',
        'ा': 'ा',
        'ु': 'ु',
        'ू': 'ू',
        'ृ': 'ृ',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
        
    # Fix broken whitespace inside words
    text = re.sub(r'([अ-ह])\s+([ा-्])', r'\1\2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def detect_file_type(filename: str) -> str:
    """Returns normalized extension (pdf, docx, doc) or raises ValueError."""
    extension = os.path.splitext(filename or "")[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload a PDF, DOC, or DOCX file.")
    return extension.lstrip(".")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Format-Specific Document Readers
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_pages(file_path: str) -> List[Dict[str, Any]]:
    """Extract pages from a PDF and include author metadata if available."""
    reader = pypdf.PdfReader(file_path)
    doc_author = ""
    try:
        doc_info = reader.metadata
        if doc_info and doc_info.get("/Author"):
            doc_author = str(doc_info.get("/Author"))
    except Exception:
        doc_author = ""
    pages = []
    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned = fix_devanagari_ligatures(raw_text)
        if cleaned:
            pages.append({
                "page_number": idx + 1,
                "location": f"page {idx + 1}",
                "text": cleaned,
                "author": doc_author
            })
    return pages


def extract_docx_pages(file_path: str) -> List[Dict[str, Any]]:
    """Extract paragraphs and tables from DOCX into structured page objects."""
    with zipfile.ZipFile(file_path) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    body = root.find(f"{ns}body")
    if body is None:
        return []

    pages: List[Dict[str, Any]] = []
    current_paragraphs: List[str] = []
    page_num = 1

    for elem in body:
        has_page_break = False
        if elem.tag == f"{ns}tbl":
            for tr in elem.iter(f"{ns}tr"):
                row_pieces = []
                for tc in tr.iter(f"{ns}tc"):
                    cell_txt = fix_devanagari_ligatures("".join(t.text or "" for t in tc.iter(f"{ns}t"))).strip()
                    if cell_txt:
                        row_pieces.append(cell_txt)
                if row_pieces:
                    current_paragraphs.append(" : ".join(row_pieces))
        else:
            pieces = []
            for node in elem.iter():
                if node.tag == f"{ns}br" and node.get(f"{ns}type") == "page":
                    has_page_break = True
                elif node.tag == f"{ns}lastRenderedPageBreak":
                    has_page_break = True
                elif node.tag == f"{ns}t" and node.text:
                    pieces.append(node.text)

            text = fix_devanagari_ligatures("".join(pieces)).strip()
            if text:
                current_paragraphs.append(text)

        if has_page_break and current_paragraphs:
            page_text = "\n".join(current_paragraphs).strip()
            if page_text:
                pages.append({
                    "page_number": page_num,
                    "location": f"page {page_num}",
                    "text": page_text
                })
                page_num += 1
            current_paragraphs = []

    if current_paragraphs:
        page_text = "\n\n".join(current_paragraphs).strip()
        if page_text:
            pages.append({
                "page_number": page_num,
                "location": f"page {page_num}",
                "text": page_text
            })

    # If document has no explicit page breaks but is long, split by logical page blocks (~350-400 words)
    if len(pages) == 1 and len(pages[0].get("text", "").split()) > 450:
        full_text = pages[0]["text"]
        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        virtual_pages = []
        cur_p = []
        cur_words = 0
        v_pnum = 1
        for para in paragraphs:
            p_words = len(para.split())
            if cur_words + p_words > 400 and cur_p:
                virtual_pages.append({
                    "page_number": v_pnum,
                    "location": f"page {v_pnum}",
                    "text": "\n\n".join(cur_p)
                })
                cur_p = [para]
                cur_words = p_words
                v_pnum += 1
            else:
                cur_p.append(para)
                cur_words += p_words
        if cur_p:
            virtual_pages.append({
                "page_number": v_pnum,
                "location": f"page {v_pnum}",
                "text": "\n\n".join(cur_p)
            })
        if virtual_pages:
            pages = virtual_pages

    return pages


def extract_doc_pages(file_path: str) -> List[Dict[str, Any]]:
    """Legacy .doc binary reader using local converters if present."""
    converter = shutil.which("antiword") or shutil.which("catdoc")
    command = [converter, file_path] if converter else None
    if not command:
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if soffice:
            with tempfile.TemporaryDirectory() as out_dir:
                completed = subprocess.run(
                    [soffice, "--headless", "--convert-to", "txt:Text", "--outdir", out_dir, file_path],
                    capture_output=True, text=True, timeout=60, check=False,
                )
                converted = os.path.join(out_dir, os.path.splitext(os.path.basename(file_path))[0] + ".txt")
                if os.path.exists(converted):
                    with open(converted, "r", encoding="utf-8", errors="replace") as source:
                        text = fix_devanagari_ligatures(source.read())
                    return [{"page_number": 1, "location": "document", "text": text}] if text else []
                raise ValueError("Could not extract text from this DOC file.")
        raise ValueError("DOC extraction requires antiword, catdoc, or LibreOffice on the server.")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode != 0:
        raise ValueError("Could not extract text from this DOC file.")
    text = fix_devanagari_ligatures(unescape(completed.stdout))
    return [{"page_number": 1, "location": "document", "text": text}] if text else []


def extract_document_pages(file_path: str) -> List[Dict[str, Any]]:
    """Unified entry point for reading pages from PDF, DOCX, or DOC."""
    file_type = detect_file_type(file_path)
    if file_type == "pdf":
        return extract_pdf_pages(file_path)
    if file_type == "docx":
        return extract_docx_pages(file_path)
    return extract_doc_pages(file_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dynamic Metadata & Year Extractor
# ─────────────────────────────────────────────────────────────────────────────

class DocumentMetadataExtractor:
    """Extracts publication year, report period, and quarter dynamically from front-matter & text."""

    YEAR_PATTERN = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
    YEAR_RANGE_PATTERN = re.compile(r"\b(19[89]\d|20[0-3]\d)\s*[-–—/]\s*(\d{2,4})\b")

    CONTEXTUAL_YEAR_PATTERNS = [
        re.compile(r"(?:वर्ष|year|अंक|edition|विशेषांक|वार्षिक|annual)\s*[:\-–—]?\s*(?:सन्\s*)?(20[0-3]\d|19[89]\d)", re.IGNORECASE),
        re.compile(r"(20[0-3]\d|19[89]\d)\s*(?:का\s*अंक|का\s*वर्ष|की\s*रिपोर्ट)", re.IGNORECASE),
        re.compile(r"(?:दिनांक|date)\s*[:\-–—]?\s*\d{1,2}[./-]\d{1,2}[./-](20[0-3]\d)", re.IGNORECASE)
    ]

    QUARTER_PATTERNS = [
        (re.compile(r"(?:प्रथम|1st|first)\s*तिमाही|Q1|quarter\s*1\b", re.IGNORECASE), "Q1"),
        (re.compile(r"(?:द्वितीय|2nd|second)\s*तिमाही|Q2|quarter\s*2\b", re.IGNORECASE), "Q2"),
        (re.compile(r"(?:तृतीय|3rd|third)\s*तिमाही|Q3|quarter\s*3\b", re.IGNORECASE), "Q3"),
        (re.compile(r"(?:चतुर्थ|4th|fourth)\s*तिमाही|Q4|quarter\s*4\b", re.IGNORECASE), "Q4"),
        (re.compile(r"त्रैमासिक|quarterly", re.IGNORECASE), "Quarterly"),
        (re.compile(r"वार्षिक|annual", re.IGNORECASE), "Annual")
    ]

    @classmethod
    def _normalize_digits(cls, text: str) -> str:
        if not text:
            return ""
        return text.translate(DEVA_TO_ASCII_DIGITS)

    @classmethod
    def extract_year(cls, pages: List[Dict[str, Any]], filename: str) -> Optional[int]:
        front_pages = pages[:3] if pages else []
        for p in front_pages:
            p_text = cls._normalize_digits(p.get("text", ""))
            if not p_text:
                continue

            for pat in cls.CONTEXTUAL_YEAR_PATTERNS:
                m = pat.search(p_text)
                if m:
                    yr = int(m.group(1))
                    if 1980 <= yr <= 2035:
                        logger.info(f"[MetadataExtractor] Found year {yr} via contextual pattern on page {p.get('page_number')}")
                        return yr

            m_range = cls.YEAR_RANGE_PATTERN.search(p_text)
            if m_range:
                yr = int(m_range.group(1))
                if 1980 <= yr <= 2035:
                    logger.info(f"[MetadataExtractor] Found year range starting {yr} on page {p.get('page_number')}")
                    return yr

            sample = p_text[:1200]
            matches = cls.YEAR_PATTERN.findall(sample)
            if matches:
                valid_yrs = [int(y) for y in matches if 1990 <= int(y) <= 2035]
                if valid_yrs:
                    yr = max(set(valid_yrs), key=valid_yrs.count)
                    logger.info(f"[MetadataExtractor] Found year {yr} from front page text sample.")
                    return yr

        clean_fn = cls._normalize_digits(filename or "")
        fn_match = cls.YEAR_PATTERN.search(clean_fn)
        if fn_match:
            yr = int(fn_match.group(1))
            if 1990 <= yr <= 2035:
                logger.info(f"[MetadataExtractor] Found year {yr} in filename: {filename}")
                return yr

        edition_match = re.search(r"edition[_\s]*(\d{2})\b", clean_fn, re.IGNORECASE)
        if edition_match:
            short_yr = int(edition_match.group(1))
            if 20 <= short_yr <= 35:
                yr = 2000 + short_yr
                logger.info(f"[MetadataExtractor] Deduced year {yr} from edition suffix in {filename}")
                return yr

        return None

    @classmethod
    def extract_quarter_and_period(cls, pages: List[Dict[str, Any]], filename: str) -> Tuple[Optional[str], Optional[str]]:
        text_corpus = filename + " "
        if pages:
            text_corpus += " ".join(cls._normalize_digits(p.get("text", "")[:500]) for p in pages[:2])

        quarter = None
        period = None

        for pat, q_val in cls.QUARTER_PATTERNS:
            if pat.search(text_corpus):
                if q_val in {"Q1", "Q2", "Q3", "Q4"}:
                    quarter = q_val
                    period = "Quarterly"
                    break
                elif q_val in {"Quarterly", "Annual"}:
                    period = q_val

        return quarter, period

    @classmethod
    def extract_metadata(
        cls,
        pages: List[Dict[str, Any]],
        filename: str,
        document_type: str = "magazine"
    ) -> Dict[str, Any]:
        year = cls.extract_year(pages, filename)
        quarter, period = cls.extract_quarter_and_period(pages, filename)

        return {
            "document_name": filename,
            "document_type": document_type.lower() if document_type else "magazine",
            "year": year,
            "quarter": quarter,
            "report_period": period,
            "total_pages": len(pages) if pages else 1
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Structure-Aware Chunking
# ─────────────────────────────────────────────────────────────────────────────

def structure_aware_chunking(
    pages: List[Dict[str, Any]],
    document_id: str,
    source_filename: str,
    target_chunk_size: int = 750, 
    chunk_overlap: int = 150,
    document_name: Optional[str] = None,
    document_type: str = "magazine",
    year: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Produces boundary-aware text chunks with uniform container attributes."""
    chunks = []
    chunk_counter = 0
    doc_display_name = document_name or source_filename
    
    for page in pages:
        p_num = page.get("page_number")
        text = page.get("text", "")
        author = page.get("author", "")
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk_text = ""
        for para in paragraphs:
            if len(current_chunk_text) + len(para) > target_chunk_size and current_chunk_text:
                chunk_id = f"{document_id}_chunk_{chunk_counter}"
                chunks.append({
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "text": current_chunk_text.strip(),
                    "document_id": document_id,
                    "document_name": doc_display_name,
                    "document_type": document_type,
                    "year": year,
                    "content_type": "content",
                    "page": p_num,
                    "page_number": p_num,
                    "author": author,
                    "source_filename": source_filename,
                    "metadata": {
                        "document_id": document_id,
                        "document_name": doc_display_name,
                        "document_type": document_type,
                        "year": year,
                        "content_type": "content",
                        "page": p_num,
                        "page_number": p_num,
                        "author": author,
                        "source_filename": source_filename,
                        "chunk_id": chunk_id
                    }
                })
                chunk_counter += 1
                
                # Overlap retention
                words = current_chunk_text.split()
                if len(words) > 30:
                    current_chunk_text = " ".join(words[-20:]) + "\n\n" + para
                else:
                    current_chunk_text = para
            else:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para
                    
        if current_chunk_text.strip():
            chunk_id = f"{document_id}_chunk_{chunk_counter}"
            chunks.append({
                "id": chunk_id,
                "chunk_id": chunk_id,
                "text": current_chunk_text.strip(),
                "document_id": document_id,
                "document_name": doc_display_name,
                "document_type": document_type,
                "year": year,
                "content_type": "content",
                "page": p_num,
                "page_number": p_num,
                "author": author,
                "source_filename": source_filename,
                "metadata": {
                    "document_id": document_id,
                    "document_name": doc_display_name,
                    "document_type": document_type,
                    "year": year,
                    "content_type": "content",
                    "page": p_num,
                    "page_number": p_num,
                    "author": author,
                    "source_filename": source_filename,
                    "chunk_id": chunk_id
                }
            })
            chunk_counter += 1
            
    return chunks
