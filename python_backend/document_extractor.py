"""
Backward-compatibility re-export module. All document extraction has been unified into extractor.py.
"""
from extractor import (
    SUPPORTED_EXTENSIONS,
    detect_file_type,
    extract_pdf_pages,
    extract_docx_pages,
    extract_doc_pages,
    extract_document_pages,
    fix_devanagari_ligatures
)

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "detect_file_type",
    "extract_pdf_pages",
    "extract_docx_pages",
    "extract_doc_pages",
    "extract_document_pages",
    "fix_devanagari_ligatures"
]
