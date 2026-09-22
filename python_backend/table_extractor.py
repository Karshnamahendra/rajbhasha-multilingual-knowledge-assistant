"""Generic table and form extraction; no document-specific schemas or values."""
import logging
import os
import re
import zipfile
from typing import Any, Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET

from extractor import fix_devanagari_ligatures

logger = logging.getLogger("TableExtractor")


class TableExtractor:
    """Extract native PDF tables where possible and conservative form/key-value tables otherwise."""

    def extract(self, file_path: str, pages: List[Dict[str, Any]], document_id: str,
                filename: str, file_type: str) -> List[Dict[str, Any]]:
        raw_tables: List[Dict[str, Any]] = []
        if file_type == "pdf":
            raw_tables.extend(self._extract_pdf_tables(file_path, pages))
        elif file_type == "docx":
            raw_tables.extend(self._extract_docx_tables(file_path))

        # Forms are deliberately a fallback. This avoids turning a normal one-off
        # sentence with a colon into fabricated structured data.
        raw_tables.extend(self._extract_form_blocks(pages))
        raw_tables.extend(self._extract_period_fields(pages))
        result: List[Dict[str, Any]] = []
        fingerprints = set()
        for index, candidate in enumerate(raw_tables, 1):
            table = self._normalise(candidate, document_id, filename, index)
            if not table:
                continue
            fingerprint = (table["page_number"], tuple(table["headers"]),
                           tuple(tuple(sorted(row.items())) for row in table["rows"]))
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            result.append(table)
            logger.info("[TableExtractor] document=%s page=%s table=%s headers=%s rows=%s confidence=%.2f",
                        document_id, table["page_number"], table["table_id"], table["headers"],
                        len(table["rows"]), table["extraction_confidence"])
        return result

    def _extract_pdf_tables(self, file_path: str, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            import pdfplumber  # optional at runtime; declared in requirements.txt
        except ImportError:
            logger.warning("[TableExtractor] pdfplumber is unavailable; native PDF table extraction skipped.")
            return []
        tables: List[Dict[str, Any]] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_index, pdf_page in enumerate(pdf.pages):
                    page_number = page_index + 1
                    page_text = next((p.get("text", "") for p in pages if p.get("page_number") == page_number), "")
                    for matrix in pdf_page.extract_tables() or []:
                        tables.append({"page_number": page_number, "matrix": matrix,
                                       "method": "pdfplumber", "heading": self._heading(page_text)})
        except Exception as exc:
            logger.warning("[TableExtractor] Native PDF extraction failed: %s", exc)
        return tables

    def _extract_docx_tables(self, file_path: str) -> List[Dict[str, Any]]:
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        try:
            with zipfile.ZipFile(file_path) as archive:
                root = ET.fromstring(archive.read("word/document.xml"))
        except Exception as exc:
            logger.warning("[TableExtractor] DOCX table extraction failed: %s", exc)
            return []
        tables = []
        for table in root.iter(f"{ns}tbl"):
            matrix = []
            for row in table.findall(f"{ns}tr"):
                cells = []
                for cell in row.findall(f"{ns}tc"):
                    cells.append(fix_devanagari_ligatures("".join(node.text or "" for node in cell.iter(f"{ns}t"))))
                if cells:
                    matrix.append(cells)
            tables.append({"page_number": None, "matrix": matrix, "method": "docx_xml", "heading": ""})
        return tables

    def _extract_form_blocks(self, pages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        extracted = []
        # Preserve an empty value: blank, zero, and "not applicable" are three
        # distinct states in a form and must never be collapsed.
        pattern = re.compile(r"^\s*(?:\d+[.)-]\s*)?(.{2,180}?)\s*[:：]\s*(.*?)\s*$")
        for page in pages:
            pairs = []
            numbered_pair = False
            for line in (page.get("text") or "").splitlines():
                match = pattern.match(line)
                if match:
                    key, value = (fix_devanagari_ligatures(item).strip() for item in match.groups())
                    if len(key) >= 2:
                        pairs.append([key, value])
                        numbered_pair = numbered_pair or bool(re.match(r"^\s*\d+[.)-]", line))
            if len(pairs) >= 2 or (pairs and numbered_pair):
                extracted.append({"page_number": page.get("page_number"), "matrix": [["Field", "Value"]] + pairs,
                                  "method": "key_value_form", "heading": self._heading(page.get("text", ""))})
        return extracted

    def _extract_period_fields(self, pages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find a date coupled with a reporting/period label without fixed dates."""
        results = []
        date_pattern = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
        period_signal = re.compile(r"quarter|period|reporting|ended|ending|तिमाही|अवधि|समाप्त", re.IGNORECASE)
        for page in pages:
            for line in (page.get("text") or "").splitlines():
                date = date_pattern.search(line)
                if date and period_signal.search(line):
                    label = (line[:date.start()] + " " + line[date.end():]).strip(" :-–—")
                    results.append({"page_number": page.get("page_number"),
                                    "matrix": [["Field", "Value"], [label or "Reporting period", date.group()]],
                                    "method": "period_field_pattern", "heading": self._heading(page.get("text", ""))})
        return results

    def _normalise(self, candidate: Dict[str, Any], document_id: str, filename: str,
                   table_index: int) -> Optional[Dict[str, Any]]:
        matrix = [[self._clean(cell) for cell in row] for row in candidate.get("matrix", [])]
        matrix = [row for row in matrix if any(row)]
        if len(matrix) < 2:
            return None
        width = max(len(row) for row in matrix)
        if width < 2:
            return None
        matrix = [row + [""] * (width - len(row)) for row in matrix]
        headers = self._unique_headers(matrix[0])
        data_rows = [row for row in matrix[1:] if any(row)]
        if not data_rows:
            return None
        # Retain empty cells for form/table semantics; only entirely empty rows go away.
        rows = [{headers[i]: row[i] for i in range(width)} for row in data_rows]
        rows = [row for row in rows if len(row) >= 1]
        rectangular = sum(1 for row in data_rows if sum(bool(cell) for cell in row) >= max(2, width - 1)) / len(data_rows)
        confidence = round(0.45 + 0.35 * rectangular + (0.15 if candidate.get("method") != "key_value_form" else 0.05), 2)
        page = candidate.get("page_number")
        table_id = f"{document_id}:p{page or 'document'}:t{table_index}"
        return {"table_id": table_id, "document_id": document_id, "filename": filename,
                "page_number": page, "heading": candidate.get("heading") or "",
                "section": candidate.get("heading") or "General", "headers": headers, "rows": rows,
                "column_count": len(headers), "row_count": len(rows), "extraction_method": candidate.get("method"),
                "extraction_confidence": confidence, "original_table_text": self._matrix_text(headers, data_rows),
                "structured_fields": self._structured_fields(headers, rows, candidate.get("method", ""))}

    @staticmethod
    def _structured_fields(headers: List[str], rows: List[Dict[str, str]], method: str) -> List[Dict[str, str]]:
        if len(headers) != 2 or method not in {"key_value_form", "period_field_pattern"}:
            return []
        label_key, value_key = headers
        fields = []
        for row in rows:
            label, value = row.get(label_key, ""), row.get(value_key, "")
            if not label:
                continue
            normalized = value.strip().lower()
            state = "blank" if not value.strip() else ("not_applicable" if normalized in {"n/a", "na", "not applicable", "लागू नहीं"} else "present")
            fields.append({"label": label, "value": value, "value_state": state})
        return fields

    @staticmethod
    def _clean(value: Any) -> str:
        return fix_devanagari_ligatures(str(value or "")).strip()

    @staticmethod
    def _unique_headers(headers: List[str]) -> List[str]:
        seen = {}
        result = []
        for index, value in enumerate(headers, 1):
            base = value or f"Column {index}"
            seen[base] = seen.get(base, 0) + 1
            result.append(base if seen[base] == 1 else f"{base} ({seen[base]})")
        return result

    @staticmethod
    def _matrix_text(headers: List[str], rows: List[List[str]]) -> str:
        return "\n".join([" | ".join(headers)] + [" | ".join(row) for row in rows])

    @staticmethod
    def _heading(page_text: str) -> str:
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        return next((line for line in lines[:12] if 3 <= len(line) <= 140 and not re.match(r"^\d+$", line)), "")


def table_to_text(table: Dict[str, Any]) -> str:
    title = table.get("heading") or "Structured table"
    lines = [f"{title}.", "Columns: " + ", ".join(table.get("headers", [])) + "."]
    for number, row in enumerate(table.get("rows", []), 1):
        values = "; ".join(f"{header} = {value}" for header, value in row.items())
        lines.append(f"Row {number}: {values}.")
    return "\n".join(lines)


def table_chunks(
    tables: List[Dict[str, Any]],
    source_filename: str,
    document_name: Optional[str] = None,
    document_type: str = "report",
    year: Optional[int] = None
) -> List[Dict[str, Any]]:
    chunks = []
    doc_display_name = document_name or source_filename
    for table_index, table in enumerate(tables, 1):
        headers = table.get("headers", [])
        # 1. Holistic Table Summary Chunk (retains complete structural context)
        chunks.append({
            "id": f"{table['table_id']}:summary",
            "text": table_to_text(table),
            "metadata": {
                "document_id": table["document_id"],
                "filename": source_filename,
                "source": source_filename,
                "document_name": doc_display_name,
                "document_type": document_type,
                "year": year,
                "content_type": "table",
                "source_filename": source_filename,
                "page": table.get("page_number"),
                "page_number": table.get("page_number"),
                "location": f"page {table.get('page_number') or 'document'}",
                "section": table.get("section", "General"),
                "headers": headers,
                "chunk_id": f"{table['table_id']}:summary",
                "chunk_type": "table",
                "table_id": table["table_id"],
                "table_index": table_index,
                "row_count": table.get("row_count", 0),
                "column_count": table.get("column_count", 0),
                "extraction_method": table.get("extraction_method", "unknown"),
                "extraction_confidence": table.get("extraction_confidence", 0.0)
            }
        })

        # 2. Multi-row Data Tables: generate row-level chunks preserving column header -> cell value relationships
        rows = table.get("rows", [])
        if len(rows) > 1 and table.get("extraction_method") != "key_value_form":
            for row_idx, row in enumerate(rows, 1):
                row_items = [f"{k} = {v}" for k, v in row.items() if str(v).strip()]
                if row_items:
                    heading_str = table.get("heading") or "Data Table"
                    page_str = table.get("page_number") or "N/A"
                    row_text = f"Table '{heading_str}' (Page {page_str}), Row {row_idx}: " + "; ".join(row_items) + "."
                    chunks.append({
                        "id": f"{table['table_id']}:r{row_idx}",
                        "text": row_text,
                        "metadata": {
                            "document_id": table["document_id"],
                            "filename": source_filename,
                            "source": source_filename,
                            "document_name": doc_display_name,
                            "document_type": document_type,
                            "year": year,
                            "content_type": "table",
                            "source_filename": source_filename,
                            "page": table.get("page_number"),
                            "page_number": table.get("page_number"),
                            "location": f"page {table.get('page_number') or 'document'}",
                            "section": table.get("section", "General"),
                            "headers": headers,
                            "chunk_id": f"{table['table_id']}:r{row_idx}",
                            "chunk_type": "table_row",
                            "table_id": table["table_id"],
                            "table_index": table_index,
                            "row_index": row_idx,
                            "row_count": table.get("row_count", 0),
                            "column_count": table.get("column_count", 0),
                            "extraction_method": table.get("extraction_method", "unknown"),
                            "extraction_confidence": table.get("extraction_confidence", 0.0)
                        }
                    })

        # 3. Form fields receive their own semantic record for exact label/field lookup
        for field_index, field in enumerate(table.get("structured_fields", []), 1):
            chunks.append({
                "id": f"{table['table_id']}:field:{field_index}",
                "text": f"Form field: {field['label']}. Recorded value: {field['value']}. Value state: {field['value_state']}.",
                "metadata": {
                    "document_id": table["document_id"],
                    "filename": source_filename,
                    "source": source_filename,
                    "document_name": doc_display_name,
                    "document_type": document_type,
                    "year": year,
                    "content_type": "table",
                    "source_filename": source_filename,
                    "page": table.get("page_number"),
                    "page_number": table.get("page_number"),
                    "location": f"page {table.get('page_number') or 'document'}",
                    "section": table.get("section", "General"),
                    "headers": headers,
                    "chunk_id": f"{table['table_id']}:field:{field_index}",
                    "chunk_type": "form_field",
                    "table_id": table["table_id"],
                    "table_index": table_index,
                    "field_label": field["label"],
                    "field_value_state": field["value_state"],
                    "row_count": 1,
                    "column_count": 2,
                    "extraction_method": table.get("extraction_method", "unknown"),
                    "extraction_confidence": table.get("extraction_confidence", 0.0)
                }
            })
    return chunks


class StructuredTableStore:
    """Document-isolated persistence for dynamically extracted tables."""

    def __init__(self, storage_dir: Optional[str] = None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.storage_dir = storage_dir or os.path.join(base, "table_store")
        os.makedirs(self.storage_dir, exist_ok=True)

    def _path(self, document_id: str) -> str:
        safe_id = re.sub(r"[^\w.\-]", "_", document_id)
        return os.path.join(self.storage_dir, f"{safe_id}_tables.json")

    def save_tables(self, document_id: str, filename: str, tables: List[Dict[str, Any]]) -> None:
        payload = {
            "schema_version": 1,
            "document_id": document_id,
            "filename": filename,
            "tables_count": len(tables),
            "tables": tables,
        }
        with open(self._path(document_id), "w", encoding="utf-8") as target:
            import json
            json.dump(payload, target, ensure_ascii=False, indent=2)

    def get_tables(self, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not document_id:
            tables: List[Dict[str, Any]] = []
            for name in os.listdir(self.storage_dir):
                if name.endswith("_tables.json"):
                    tables.extend(self._read_path(os.path.join(self.storage_dir, name)))
            return tables
        return self._read_path(self._path(document_id))

    def get_table(self, document_id: str, table_id: str) -> Optional[Dict[str, Any]]:
        return next((table for table in self.get_tables(document_id)
                     if table.get("table_id") == table_id), None)

    def delete_document(self, document_id: str) -> None:
        path = self._path(document_id)
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def _read_path(path: str) -> List[Dict[str, Any]]:
        import json
        try:
            with open(path, "r", encoding="utf-8") as source:
                return json.load(source).get("tables", [])
        except (OSError, ValueError, TypeError):
            return []

