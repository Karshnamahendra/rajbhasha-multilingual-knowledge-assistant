"""Universal Hierarchical Document Builder.

Dynamically transforms extracted pages, tables, forms, and TOC records
into a single recursive Key -> Sub-key -> Value hierarchy.

No hardcoded document names, section names, author mappings, or answer mappings.
All knowledge is extracted directly and dynamically from the document structure.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from extractor import fix_devanagari_ligatures
from hierarchical_model import HierarchicalNode, DocumentHierarchy


def clean_text(text: Any) -> str:
    """Normalize whitespace and fix Devanagari ligatures."""
    if text is None:
        return ""
    t = fix_devanagari_ligatures(str(text))
    t = re.sub(r"[\u200B-\u200D\uFEFF\u00AD\uf000-\uf8ff]", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def clean_key(text: Any) -> str:
    """Cleans a structural key by removing leading numeric bullets (1., i., (क)) and trailing colons."""
    t = clean_text(text)
    if not t:
        return ""
    # Strip leading bullet numbering, e.g.:
    # "1.", "1. :", "i.", "ii.", "(क)", "(ख)", "(1)", "1 -", "•", "*"
    t = re.sub(
        r"^(?:[0-9]{1,3}[.)\s:-]+|[a-zA-Z][.)\s:-]+|\([a-zA-Z0-9\u0900-\u097F]+\)[.\s:-]*|[ivxIVX]{1,5}[.)\s:-]+|[•*]\s*)",
        "",
        t,
    ).strip()
    # Strip trailing punctuation/colons
    t = re.sub(r"[:：\s.\-—]+$", "", t).strip()
    return t or clean_text(text)


def clean_val(val: Any) -> str:
    """Cleans an extracted value."""
    if val is None:
        return ""
    v = clean_text(val)
    # Strip leading colons if present
    v = re.sub(r"^[:：\s]+", "", v).strip()
    return v


def is_generic_header(header: str) -> bool:
    """Returns True if the header is an automatic placeholder like 'Column 1', 'Field', or pure digit."""
    h = str(header).strip().lower()
    if not h or re.fullmatch(r"column\s*\d+", h) or h in {"field", "value", "column"}:
        return True
    if re.fullmatch(r"\d+[.)]?", h):
        return True
    return False


def build_toc_hierarchy(toc_records: List[Dict[str, Any]]) -> Optional[HierarchicalNode]:
    """Dynamically converts extracted TOC records into the universal hierarchy.

    Structure:
      अनुक्रमणिका / विषय-सूची
        -> Section Name (if present)
            -> Article Title
                -> लेखक: Author Name
                -> पृष्ठ संख्या: 25
    """
    if not toc_records:
        return None

    toc_root = HierarchicalNode(key="विषय-सूची / अनुक्रमणिका", source_page=toc_records[0].get("toc_page", 1))

    # Group by section preserving document order
    sections: Dict[str, List[HierarchicalNode]] = {}
    standalone_articles: List[HierarchicalNode] = []

    for r in toc_records:
        title = clean_text(r.get("title_original") or r.get("title_normalized"))
        if not title:
            continue
        author = clean_text(r.get("author_original") or r.get("author_normalized"))
        page = r.get("page_number")
        section = clean_text(r.get("section_original") or r.get("section_normalized"))
        toc_p = r.get("toc_page")

        children: List[HierarchicalNode] = []
        if author:
            children.append(HierarchicalNode(key="लेखक", value=author, source_page=toc_p))
        if page:
            children.append(HierarchicalNode(key="पृष्ठ संख्या", value=str(page), source_page=toc_p))

        article_node = HierarchicalNode(key=title, children=children, source_page=toc_p)

        if section and section not in {"General", "सामान्य"}:
            sections.setdefault(section, []).append(article_node)
        else:
            standalone_articles.append(article_node)

    if sections:
        for sec_name, articles in sections.items():
            sec_node = HierarchicalNode(key=sec_name, children=articles)
            toc_root.add_child(sec_node)

    for art in standalone_articles:
        toc_root.add_child(art)

    return toc_root if toc_root.children else None


def build_table_hierarchy(table: Dict[str, Any]) -> Optional[HierarchicalNode]:
    """Dynamically transforms a table into the universal Key -> Sub-key -> Value hierarchy.

    Handles:
      1. Matrix / Multi-column Data Tables (e.g. Category -> Header1: Val1, Header2: Val2)
      2. Key-Value Form Tables (e.g. Question/Field -> Value)
    """
    headers = table.get("headers", [])
    rows = table.get("rows", [])
    page_number = table.get("page_number")

    # 1. Determine the descriptive table title dynamically
    # Avoid 'General' fallback for heading
    raw_heading = clean_text(table.get("heading") or "")
    if raw_heading in {"General", "सामान्य", "Structured table"}:
        raw_heading = ""

    title_candidates = [
        h for h in headers if len(clean_text(h)) >= 3 and not is_generic_header(h)
    ]

    if raw_heading:
        table_title = raw_heading
    elif title_candidates:
        table_title = title_candidates[0]
    else:
        table_title = "तालिका विवरण"

    cleaned_title = clean_key(table_title) or table_title
    table_node = HierarchicalNode(key=cleaned_title, source_page=page_number)

    if not rows:
        return table_node

    header_rows = []
    data_rows = []
    for r in rows:
        non_empty = {col: clean_text(v) for col, v in r.items() if clean_text(v) and clean_text(v) not in {":", "::"}}
        if not non_empty:
            continue
        vals = list(non_empty.values())
        is_bracket_num = len(vals) >= 2 and all(re.fullmatch(r"\(\d+\)", v) for v in vals)
        int_vals = [int(v) for v in vals if re.fullmatch(r"\d+", v)]
        is_seq_num = len(int_vals) == len(vals) and len(int_vals) >= 2 and int_vals == list(range(1, len(int_vals) + 1))
        is_col_index = is_bracket_num or is_seq_num
        num_cells = sum(1 for v in vals if re.fullmatch(r"[-+]?\d+(?:[,.]\d+)?%?", v))

        if data_rows:
            data_rows.append(r)
        elif is_col_index:
            header_rows.append(r)
        elif num_cells >= 2 or (num_cells >= 1 and len(vals) <= 2):
            data_rows.append(r)
        else:
            header_rows.append(r)

    # Multi-level column headers across header rows
    col_subheaders: Dict[str, List[str]] = {col: [] for col in headers}
    for r in header_rows:
        inherited = ""
        for col in headers:
            v = clean_text(r.get(col, ""))
            if v and not re.fullmatch(r"\(\d+\)", v) and v not in {":", "::"}:
                inherited = v
                col_subheaders[col].append(v)
            elif inherited and (is_generic_header(col) or col == table_title):
                # Spanning header into next column
                col_subheaders[col].append(inherited)

    resolved_cols: Dict[str, str] = {}
    for col in headers:
        parts = col_subheaders.get(col, [])
        dedup = []
        for p in parts:
            if not dedup or dedup[-1] != p:
                dedup.append(p)
        if dedup:
            resolved_cols[col] = " - ".join(clean_key(p) for p in dedup if clean_key(p))
        elif not is_generic_header(col) and col != table_title:
            resolved_cols[col] = clean_key(col)

    # Identify if there is a row category column among data rows
    label_col = None
    for col in headers:
        vals = [clean_val(r.get(col, "")) for r in data_rows if clean_val(r.get(col, ""))]
        if vals and len(vals) >= 1 and all(not re.fullmatch(r"[-+]?\d+(?:[,.]\d+)?%?", v) for v in vals):
            label_col = col
            break

    if label_col:
        for r in data_rows:
            row_subject = clean_val(r.get(label_col, ""))
            if not row_subject or row_subject in {":", "::"}:
                continue
            cleaned_row_subject = clean_key(row_subject) or row_subject
            row_node = HierarchicalNode(key=cleaned_row_subject, source_page=page_number)
            for col, col_name in resolved_cols.items():
                if col == label_col:
                    continue
                val = clean_val(r.get(col, ""))
                if val and val not in {":", "::"}:
                    row_node.add_child(HierarchicalNode(key=col_name or col, value=val, source_page=page_number))
            if row_node.children:
                table_node.add_child(row_node)
    else:
        if len(data_rows) == 1:
            r = data_rows[0]
            for col, col_name in resolved_cols.items():
                val = clean_val(r.get(col, ""))
                if val and val not in {":", "::"} and not re.fullmatch(r"\(\d+\)", val):
                    table_node.add_child(HierarchicalNode(key=col_name or col, value=val, source_page=page_number))
        else:
            for r in data_rows:
                row_items = [(resolved_cols.get(c, c), clean_val(r.get(c, ""))) for c in headers if clean_val(r.get(c, ""))]
                row_items = [(k, v) for k, v in row_items if k and v and v not in {":", "::"}]
                if len(row_items) == 2 and not re.fullmatch(r"[-+]?\d+", row_items[0][1]):
                    table_node.add_child(HierarchicalNode(key=clean_key(row_items[0][1]), value=row_items[1][1], source_page=page_number))
                else:
                    for k, v in row_items:
                        table_node.add_child(HierarchicalNode(key=k, value=v, source_page=page_number))

    # Fallback / Key-Value form table mode if no children built
    if not table_node.children:
        for r in rows:
            label = ""
            val = ""
            if title_candidates and title_candidates[0] in r:
                label = clean_val(r.get(title_candidates[0], ""))
                other_vals = [
                    clean_val(r[c]) for c in headers
                    if c != title_candidates[0] and clean_val(r.get(c, "")) not in {"", ":", "::"}
                ]
                val = other_vals[-1] if other_vals else ""
            elif "Field" in r and "Value" in r:
                label = clean_val(r.get("Field", ""))
                val = clean_val(r.get("Value", ""))
            else:
                cells = [clean_val(r.get(h, "")) for h in headers if clean_val(r.get(h, "")) not in {"", ":", "::"}]
                if len(cells) >= 2:
                    label = cells[0]
                    val = cells[-1]
                elif len(cells) == 1:
                    label = cells[0]
                    val = ""

            if label and label != cleaned_title:
                c_lbl = clean_key(label)
                c_val = clean_val(val)
                if c_lbl and c_val:
                    table_node.add_child(HierarchicalNode(key=c_lbl, value=c_val, source_page=page_number))

    # Narrative / sequential text table handling (e.g. Section 11 narrative sub-sections, notes, achievements)
    if not table_node.children:
        clean_rows = []
        for r in rows:
            non_empty = [clean_val(v) for v in r.values() if clean_val(v) and clean_val(v) not in {":", "::"}]
            if non_empty:
                clean_rows.append(non_empty[-1] if len(non_empty) == 1 else " : ".join(non_empty))

        if clean_rows:
            subhead_pat = re.compile(r"^\s*(?:[iIvVxX\d]+|[क-ह]|[A-Za-z])[.)\-]\s*")
            current_key = None
            current_val_pieces = []

            for text in clean_rows:
                is_subhead = bool(subhead_pat.match(text)) or (len(text) <= 60 and not text.endswith(('।', '.', '!', '?')))
                if is_subhead and len(text) <= 100:
                    if current_key and current_val_pieces:
                        val_text = " ".join(current_val_pieces).strip()
                        c_k = clean_key(current_key) or current_key
                        table_node.add_child(HierarchicalNode(key=c_k, value=val_text, source_page=page_number))
                        current_val_pieces = []
                    current_key = text
                else:
                    if current_key:
                        current_val_pieces.append(text)
                    else:
                        c_k = clean_key(text[:50]) or text[:50]
                        table_node.add_child(HierarchicalNode(key=c_k, value=text, source_page=page_number))

            if current_key and current_val_pieces:
                val_text = " ".join(current_val_pieces).strip()
                c_k = clean_key(current_key) or current_key
                table_node.add_child(HierarchicalNode(key=c_k, value=val_text, source_page=page_number))

    return table_node if table_node.children else None



def build_prose_content_hierarchy(pages: List[Dict[str, Any]], existing_keys: Set[str]) -> List[HierarchicalNode]:
    """Extracts standalone Field : Value pairs outside tables into a single consolidated section."""
    content_nodes: List[HierarchicalNode] = []
    seen_keys: Set[str] = set(existing_keys)

    kv_pattern = re.compile(r"^\s*(?:\d+[.)-]\s*)?(.{2,140}?)\s*[:：\u0903]\s*(.*?)\s*$")
    all_kv_children: List[HierarchicalNode] = []


    for page in pages:
        p_num = page.get("page_number")
        text = page.get("text", "")
        if not text:
            continue

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for line in lines:
            m = kv_pattern.match(line)
            if m:
                k_raw, v_raw = m.groups()
                # Skip if the key is purely a numeric index like '1.', '2.', '3'
                if re.fullmatch(r"\d+[.)\s:-]*", k_raw.strip()):
                    continue
                k_clean = clean_key(k_raw)
                v_clean = clean_val(v_raw)
                # Skip if value is a known section title already captured in tables
                if clean_key(v_clean) in seen_keys or v_clean in seen_keys:
                    continue
                if k_clean and len(k_clean) >= 2 and v_clean and k_clean not in seen_keys:
                    if len(v_clean) > 0 and not v_clean.startswith(":"):
                        seen_keys.add(k_clean)
                        all_kv_children.append(HierarchicalNode(key=k_clean, value=v_clean, source_page=p_num))

    if all_kv_children:
        content_nodes.append(HierarchicalNode(
            key="कार्यालय एवं सामान्य विवरण",
            children=all_kv_children,
            source_page=all_kv_children[0].source_page if all_kv_children else 1
        ))

    return content_nodes


def build_document_hierarchy(
    file_path: str,
    pages: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    toc_records: List[Dict[str, Any]],
    doc_meta: Dict[str, Any],
) -> DocumentHierarchy:
    """Orchestrates creation of the universal DocumentHierarchy for ANY document type."""
    doc_id = doc_meta.get("document_id") or doc_meta.get("document_name") or "document"
    filename = doc_meta.get("document_name") or doc_meta.get("filename") or "document"
    doc_type = doc_meta.get("document_type", "magazine")
    year = doc_meta.get("year")
    quarter = doc_meta.get("quarter")
    report_period = doc_meta.get("report_period")
    total_pages = doc_meta.get("total_pages", len(pages))

    root = HierarchicalNode(key=filename)
    registered_keys: Set[str] = set()

    # 1. Build TOC / Index Hierarchy if document is a magazine or has genuine TOC records
    # (Reports often have numbered questions which shouldn't be treated as article TOC)
    is_report = "report" in str(doc_type).lower()
    if toc_records and not is_report:
        toc_node = build_toc_hierarchy(toc_records)
        if toc_node:
            root.add_child(toc_node)
            for ch in toc_node.children:
                registered_keys.add(ch.key)

    # 2. Build Tables & Forms Hierarchy
    # In DOCX reports, native tables (docx_xml) are authoritative; fallback key_value_form tables
    # are used to catch any remaining form fields.
    has_native_tables = any(t.get("extraction_method") in {"docx_xml", "pdfplumber"} for t in tables)
    for table in tables:
        method = table.get("extraction_method")
        if has_native_tables and method in {"key_value_form", "period_field_pattern"}:
            # Fallback form extractor: only keep fields not already captured
            continue
        t_node = build_table_hierarchy(table)
        if t_node and t_node.children:
            root.add_child(t_node)
            registered_keys.add(t_node.key)
            for child in t_node.children:
                registered_keys.add(child.key)

    # 3. Build Prose / Form Line Content Hierarchy
    # Captures header fields (e.g., office name, address, reporting period) outside native tables
    prose_nodes = build_prose_content_hierarchy(pages, existing_keys=registered_keys)
    for p_node in prose_nodes:
        if p_node.children:
            root.add_child(p_node)

    return DocumentHierarchy(
        document_id=doc_id,
        file_name=filename,
        document_type=doc_type,
        year=year,
        quarter=quarter,
        report_period=report_period,
        total_pages=total_pages,
        root=root,
    )
