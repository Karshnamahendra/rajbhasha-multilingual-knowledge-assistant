import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("PDFSegmenter")
logger.setLevel(logging.INFO)

class PDFSegmenter:
    """
    Dynamically identifies index/front-matter pages vs. main content pages
    without hardcoded page counts or assumption of identical document layouts.
    """

    TOC_KEYWORDS_EN = [
        r"\bcontents?\b", r"\btable of contents\b", r"\bindex\b",
        r"\beditorial\b", r"\bforeword\b", r"\bpreface\b",
        r"\bboard of editors\b", r"\bpublished by\b", r"\bvolume\b", r"\bissue\b",
        r"\barticles?\b", r"\bcontributors\b"
    ]
    
    TOC_KEYWORDS_HI = [
        r"विषय\s*सूची", r"अनुक्रमणिका", r"सामग्री", r"संपादकीय",
        r"प्राक्कथन", r"भूमिका", r"पत्रिका", r"संपादक\s*मंडल", r"अंक\b",
        r"रचनाएं", r"लेखक\s*सूची", r"विषयानुक्रमणिका"
    ]

    # Pattern for index lines ending in page numbers (e.g., "Semiconductor ... 12" or "कहानी - 15")
    PAGE_LEADER_PATTERN = re.compile(r"(\.{2,}|…+|\-{2,}|—+|\t+)\s*\d{1,3}$|(?:\s{3,}\d{1,3})$", re.MULTILINE)

    def __init__(self, max_frontmatter_scan: int = 15):
        self.max_frontmatter_scan = max_frontmatter_scan

    def _score_page_as_frontmatter(self, page_text: str, page_num: int) -> float:
        if not page_text or not page_text.strip():
            return 0.2  # Blank or visual cover page

        score = 0.0
        text_lower = page_text.lower()
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]

        # 1. Keyword Signal (Bilingual)
        for kw in self.TOC_KEYWORDS_EN + self.TOC_KEYWORDS_HI:
            if re.search(kw, text_lower):
                score += 0.35

        # 2. Leader & TOC Page Mapping Signal
        leader_matches = len(self.PAGE_LEADER_PATTERN.findall(page_text))
        if leader_matches >= 3:
            score += 0.5
        elif leader_matches >= 1:
            score += 0.25

        # 3. Listing vs Prose Ratio
        short_lines = [l for l in lines if len(l.split()) < 12]
        if lines and (len(short_lines) / len(lines)) > 0.7:
            score += 0.2

        # 4. First 2 pages slight structural bias
        if page_num <= 2:
            score += 0.15

        return score

    def segment_pages(self, pages_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Segments extracted pages into index_pages and content_pages.
        pages_data format: [{"page_number": int, "text": str, ...}]
        """
        total_pages = len(pages_data)
        scan_limit = min(self.max_frontmatter_scan, max(3, int(total_pages * 0.35)))
        
        index_pages = []
        content_pages = []
        
        in_frontmatter = True
        consecutive_content_signals = 0

        for page in pages_data:
            p_num = page.get("page_number")
            p_text = page.get("text", "")

            # Word extraction has no reliable physical page number. It still uses
            # the same structural heuristics, without inventing one.
            ordinal = p_num if isinstance(p_num, int) else 1
            if ordinal <= scan_limit and in_frontmatter:
                score = self._score_page_as_frontmatter(p_text, ordinal)
                
                # Check for continuous narrative article flow (prose block > 180 words without TOC markers)
                words = p_text.split()
                is_dense_prose = len(words) > 180 and len(self.PAGE_LEADER_PATTERN.findall(p_text)) == 0

                if score >= 0.45 and not is_dense_prose:
                    consecutive_content_signals = 0
                    page_copy = dict(page)
                    page_copy["segment_type"] = "index"
                    index_pages.append(page_copy)
                    logger.info(f"[PDFSegmenter] Page {p_num} identified as INDEX/FRONT-MATTER (Confidence Score: {score:.2f})")
                    continue
                else:
                    if is_dense_prose or score < 0.25:
                        consecutive_content_signals += 1
                        if consecutive_content_signals >= 2 or ordinal > 3:
                            in_frontmatter = False

            # Content page
            page_copy = dict(page)
            page_copy["segment_type"] = "content"
            content_pages.append(page_copy)

        # Safe Fallback: if no index pages detected
        if not index_pages and pages_data:
            if len(pages_data) == 1:
                logger.info("[PDFSegmenter Fallback] Single-page document detected. Assigning to content_collection.")
                first_page = dict(pages_data[0])
                first_page["segment_type"] = "content"
                content_pages.append(first_page)
            else:
                logger.warning("[PDFSegmenter Fallback] No clear index structure detected. Assigning Page 1 to index_collection.")
                first_page = dict(pages_data[0])
                first_page["segment_type"] = "index"
                index_pages.append(first_page)
                content_pages = [dict(p, segment_type="content") for p in pages_data[1:]]

        logger.info(f"[PDFSegmenter Completed] {len(index_pages)} front-matter/index pages, {len(content_pages)} content pages.")
        return index_pages, content_pages
