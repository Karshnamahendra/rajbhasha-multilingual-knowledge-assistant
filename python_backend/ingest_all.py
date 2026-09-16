import os
import sys
import json
import datetime
import logging

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IngestAll")

from config import settings
from pipeline import RAGPipeline
from document_extractor import detect_file_type, extract_document_pages

def ingest_directory(upload_dir: str = None) -> dict:
    if upload_dir is None:
        upload_dir = getattr(settings, "UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "uploaded_pdfs"))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    doc_meta_dir = os.path.join(base_dir, "doc_metadata")
    os.makedirs(doc_meta_dir, exist_ok=True)

    if not os.path.exists(upload_dir):
        logger.warning(f"Upload directory does not exist: {upload_dir}")
        return {"processed": 0, "errors": []}

    supported_exts = {".pdf", ".docx", ".doc"}
    files_to_process = [
        f for f in os.listdir(upload_dir)
        if os.path.splitext(f)[1].lower() in supported_exts and not f.startswith("~$")
    ]

    logger.info(f"Found {len(files_to_process)} document(s) in {upload_dir} to ingest.")

    pipeline = RAGPipeline()
    results = {}
    errors = []

    for filename in files_to_process:
        file_path = os.path.join(upload_dir, filename)
        doc_id = filename
        logger.info(f"--- Ingesting: {filename} ---")
        try:
            doc_type = "report" if "report" in filename.lower() else "magazine"
            res = pipeline.index_document(file_path=file_path, document_id=doc_id, filename=filename, document_type=doc_type)
            results[filename] = res

            # Update cached metadata for frontend
            try:
                pages_raw = extract_document_pages(file_path)
                pages_for_frontend = []
                total_chars = 0
                total_words = 0
                full_text_parts = []
                for p in pages_raw:
                    ptext = p.get("text", "")
                    pwords = len(ptext.split())
                    pchars = len(ptext)
                    total_chars += pchars
                    total_words += pwords
                    full_text_parts.append(ptext)
                    pages_for_frontend.append({
                        "pageNumber": p.get("page_number", p.get("pageNumber")),
                        "page_number": p.get("page_number", p.get("pageNumber")),
                        "location": p.get("location"),
                        "text": ptext,
                        "wordCount": pwords,
                        "word_count": pwords,
                        "charCount": pchars,
                        "char_count": pchars,
                    })
                full_text = "\n\n".join(full_text_parts)
                chunks_for_frontend = []
                for i, p in enumerate(pages_for_frontend[:50]):
                    ptext = p["text"]
                    if ptext.strip():
                        chunks_for_frontend.append({
                            "id": f"{doc_id}_chunk_{i}",
                            "chunk_id": f"{doc_id}_chunk_{i}",
                            "docId": doc_id,
                            "document_id": doc_id,
                            "docName": filename,
                            "document_name": filename,
                            "pageNumber": p["pageNumber"],
                            "page_number": p["pageNumber"],
                            "chunkIndex": i,
                            "chunk_index": i,
                            "text": ptext[:500],
                            "charCount": min(500, len(ptext)),
                            "wordCount": len(ptext[:500].split()),
                        })
                meta = {
                    "document_id": doc_id,
                    "filename": filename,
                    "file_type": res.get("file_type", "pdf"),
                    "document_type": res.get("document_type", doc_type),
                    "year": res.get("year"),
                    "quarter": res.get("quarter"),
                    "report_period": res.get("report_period"),
                    "total_pages": res.get("total_pages", len(pages_raw)),
                    "total_words": total_words,
                    "total_chars": total_chars,
                    "full_text": full_text[:5000],
                    "pages": pages_for_frontend,
                    "chunks": chunks_for_frontend,
                    "total_chunks": res.get("total_chunks", 0),
                    "index_chunks": res.get("index_chunks", 0),
                    "content_chunks": res.get("content_chunks", 0),
                    "terminology_records": res.get("terminology_records", 0),
                    "table_records": res.get("table_records", 0),
                    "table_chunks": res.get("table_chunks", 0),
                    "processing_status": res.get("processing_status", []),
                    "upload_date": datetime.datetime.now().isoformat(),
                }
                with open(os.path.join(doc_meta_dir, f"{doc_id}.json"), "w", encoding="utf-8") as mf:
                    json.dump(meta, mf, ensure_ascii=False)
            except Exception as meta_err:
                logger.warning(f"Metadata caching notice for {filename}: {meta_err}")

            logger.info(f"Successfully ingested {filename}: {res.get('total_chunks', 0)} chunks")
        except Exception as e:
            logger.error(f"Failed to ingest {filename}: {e}", exc_info=True)
            errors.append({"filename": filename, "error": str(e)})

    # Summary of Qdrant collection counts
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=getattr(settings, "QDRANT_URL", "http://localhost:6333"))
        colls = [
            getattr(settings, "QDRANT_INDEX_COLLECTION_NAME", "rajbhasha_index_collection"),
            getattr(settings, "QDRANT_CONTENT_COLLECTION_NAME", "rajbhasha_content_collection"),
            getattr(settings, "QDRANT_TABLE_COLLECTION_NAME", "rajbhasha_table_collection")
        ]
        logger.info("=== Qdrant Collection Stats ===")
        for c in colls:
            info = qc.get_collection(c)
            logger.info(f"  {c}: {info.points_count} points")
    except Exception as e:
        logger.warning(f"Could not retrieve Qdrant stats: {e}")

    return {"processed": len(results), "errors": errors, "details": results}

if __name__ == "__main__":
    ingest_directory()
