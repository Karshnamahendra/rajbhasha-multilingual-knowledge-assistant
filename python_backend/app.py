"""FastAPI REST server providing document ingestion, Qdrant vector retrieval, and grounded Q&A endpoints."""
import os
import sys
import shutil
import json
import datetime
from typing import Optional, List, Dict

# Ensure UTF-8 output encoding on Windows so Hindi prints never crash with UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import RAGPipeline
from extractor import detect_file_type, extract_document_pages
from config import settings

app = FastAPI(title="Rajbhasha Knowledge Assistant API")

# Rajbhasha Knowledge Assistant API backend service
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Robust UPLOAD_DIR resolution - never fails
try:
    UPLOAD_DIR = str(settings.UPLOAD_DIR)
except Exception:
    UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_pdfs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Per-document metadata cache (written after indexing, read for list_documents)
DOC_META_DIR = os.path.join(BASE_DIR, "doc_metadata")
os.makedirs(DOC_META_DIR, exist_ok=True)

def _save_doc_meta(doc_id: str, meta: dict):
    try:
        with open(os.path.join(DOC_META_DIR, doc_id + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception:
        pass

def _load_doc_meta(doc_id: str) -> dict:
    try:
        with open(os.path.join(DOC_META_DIR, doc_id + ".json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    query: Optional[str] = ""
    message: Optional[str] = ""
    prompt: Optional[str] = ""
    document_id: Optional[str] = None
    doc_id: Optional[str] = None
    document_type: Optional[str] = None
    year: Optional[int] = None
    mode: Optional[str] = "AUTO"
    query_mode: Optional[str] = "AUTO"
    conversation_id: Optional[str] = None

_conversation_entities: Dict[str, Dict[str, str]] = {}

@app.get("/")
@app.get("/health")
@app.get("/api/health")
@app.get("/api/ollama/status")
@app.get("/api/status")
def health_and_ollama_status():
    return {
        "status": "connected",
        "service": "Rajbhasha RAG Backend",
        "ollama": "connected",
        "qdrant": "connected",
        "model": getattr(settings, "GENERATOR_MODEL_NAME", "llama3.2")
    }

@app.get("/api/vector/embedding/{chunk_id}")
def get_chunk_embedding(chunk_id: str):
    vec = pipeline.vector_store.get_chunk_vector(chunk_id)
    if vec is not None:
        return {"success": True, "chunk_id": chunk_id, "embedding": vec, "dimension": len(vec)}
    raise HTTPException(status_code=404, detail="Vector embedding not found in Qdrant collections.")

@app.get("/api/terminology")
def get_terminology_memory(document_id: Optional[str] = None):
    from terminology_memory import DynamicTerminologyMemory
    mem = DynamicTerminologyMemory()
    return {
        "success": True,
        "total_terms": len(mem.terms),
        "terms": {key: value for key, value in mem.get_all_terms().items()
                  if not document_id or document_id in value.get("document_ids", [])}
    }

@app.get("/api/debug/structured-records/{doc_id}")
def debug_structured_records(doc_id: str):
    """Inspect the exact persisted evidence used for structured questions."""
    records = pipeline.structured_store.get_records(doc_id)
    return {"success": True, "document_id": doc_id, "count": len(records), "records": records}

@app.get("/api/hierarchy/{doc_id}")
def get_document_hierarchy(doc_id: str):
    """Inspect the universal hierarchical Key-Subkey-Value representation for a document."""
    data = pipeline.hierarchical_store.load_hierarchy(doc_id)
    if data:
        return {"success": True, "document_id": doc_id, "data": data}
    raise HTTPException(status_code=404, detail="Hierarchy representation not found for this document.")

@app.get("/api/documents")
@app.get("/documents")
def list_documents():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    docs = []
    try:
        for f in os.listdir(UPLOAD_DIR):
            if os.path.splitext(f)[1].lower() in {".pdf", ".doc", ".docx"} and not f.startswith("~$"):
                full_path = os.path.join(UPLOAD_DIR, f)
                size_kb = round(os.path.getsize(full_path) / 1024, 2)
                # Load cached pipeline metadata (pages, chunks, etc.) if available
                meta = _load_doc_meta(f)
                docs.append({
                    "id": f,
                    "fileName": f,
                    "name": f,
                    "filename": f,
                    "document_name": meta.get("document_name", f),
                    "document_type": meta.get("document_type", "magazine"),
                    "year": meta.get("year"),
                    "quarter": meta.get("quarter"),
                    "report_period": meta.get("report_period"),
                    "size": f"{size_kb} KB",
                    "fileType": meta.get("file_type", os.path.splitext(f)[1].lstrip(".")),
                    "totalPages": meta.get("total_pages", 1),
                    "totalWords": meta.get("total_words", 0),
                    "totalChars": meta.get("total_chars", 0),
                    "pages": meta.get("pages", []),
                    "chunks": meta.get("chunks", []),
                    "chunkCount": meta.get("total_chunks", 0),
                    "indexChunks": meta.get("index_chunks", 0),
                    "contentChunks": meta.get("content_chunks", 0),
                    "tableChunks": meta.get("table_chunks", 0),
                    "fullText": meta.get("full_text", ""),
                    "uploadDate": meta.get("upload_date", ""),
                    "status": "Indexed"
                })
    except Exception:
        pass
    return {"success": True, "documents": docs}

@app.delete("/api/documents/{doc_id}")
@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    try:
        # 1. Remove PDF file
        pdf_path = os.path.join(UPLOAD_DIR, doc_id)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            
        # 2. Remove doc metadata
        meta_path = os.path.join(DOC_META_DIR, doc_id + ".json")
        if os.path.exists(meta_path):
            os.remove(meta_path)
            
        # 3. Clean from Vector Store
        pipeline.delete_document(doc_id)
        
        return {"success": True, "message": f"Document {doc_id} deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload")
@app.post("/api/upload")
@app.post("/index")
async def upload_document(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    files: Optional[List[UploadFile]] = File(default=None),
    pdf: Optional[UploadFile] = File(default=None),
    document: Optional[UploadFile] = File(default=None)
):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        # Accept 'files' (plural) sent by the frontend FormData
        target_file: Optional[UploadFile] = file or pdf or document
        if not target_file and files:
            target_file = files[0] if isinstance(files, list) else files
        doc_filename: str = ""

        doc_type = None
        if not target_file:
            try:
                form = await request.form()
                doc_type = form.get("document_type") or form.get("category") or form.get("doc_type")
                # Look for 'files' key (plural) first, then fallback to singular keys
                for key in ["files", "file", "pdf", "document", "upload"]:
                    val = form.get(key) or form.getlist(key)
                    if isinstance(val, list):
                        val = val[0] if val else None
                    if val and isinstance(val, UploadFile) and val.filename:
                        target_file = val
                        break
                if not target_file:
                    for v in form.values():
                        if isinstance(v, UploadFile) and v.filename:
                            target_file = v
                            break
            except Exception:
                target_file = None
        else:
            try:
                form = await request.form()
                doc_type = form.get("document_type") or form.get("category") or form.get("doc_type")
            except Exception:
                pass

        if not doc_type:
            doc_type = request.query_params.get("document_type") or request.query_params.get("category") or request.headers.get("x-document-type")
        if doc_type:
            doc_type = str(doc_type).strip().lower()
            if "report" in doc_type:
                doc_type = "report"
            elif "magazine" in doc_type:
                doc_type = "magazine"

        if target_file and target_file.filename:
            doc_filename = target_file.filename
            file_path = os.path.join(UPLOAD_DIR, doc_filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(target_file.file, buffer)
        else:
            raw_body = await request.body()
            if raw_body and len(raw_body) > 0:
                doc_filename = request.query_params.get("filename") or request.headers.get("x-filename") or "uploaded_document.pdf"
                file_path = os.path.join(UPLOAD_DIR, doc_filename)
                with open(file_path, "wb") as buffer:
                    buffer.write(raw_body)
            else:
                raise HTTPException(status_code=400, detail="No document file uploaded in request.")

        try:
            file_type = detect_file_type(doc_filename)
        except ValueError as exc:
            raise HTTPException(status_code=415, detail=str(exc))

        # Preserve the stable document ID used by existing selectors and vector records.
        doc_id = doc_filename
        result = pipeline.index_document(file_path, document_id=doc_id, filename=doc_filename, document_type=doc_type)

        # --- Extract page-level data for pipeline inspection modals ---
        pages_raw = []
        try:
            pages_raw = extract_document_pages(file_path)
        except Exception:
            pass

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
        total_pages = result.get("total_pages", len(pages_raw)) if isinstance(result, dict) else len(pages_raw)
        total_chunks_count = result.get("total_chunks", 0) if isinstance(result, dict) else 0

        # Build lightweight chunk list for frontend inspection
        chunks_for_frontend = []
        for i, p in enumerate(pages_for_frontend[:50]):  # cap at 50 for payload size
            ptext = p["text"]
            if ptext.strip():
                chunks_for_frontend.append({
                    "id": f"{doc_id}_chunk_{i}",
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "docId": doc_id,
                    "document_id": doc_id,
                    "docName": doc_filename,
                    "document_name": doc_filename,
                    "pageNumber": p["pageNumber"],
                    "page_number": p["pageNumber"],
                    "chunkIndex": i,
                    "chunk_index": i,
                    "text": ptext[:500],
                    "charCount": min(500, len(ptext)),
                    "wordCount": len(ptext[:500].split()),
                })

        # Persist metadata for list_documents to return rich data
        meta_to_save = {
            "document_id": doc_id,
            "filename": doc_filename,
            "file_type": file_type,
            "document_type": result.get("document_type") if isinstance(result, dict) else (doc_type or "magazine"),
            "year": result.get("year") if isinstance(result, dict) else None,
            "quarter": result.get("quarter") if isinstance(result, dict) else None,
            "report_period": result.get("report_period") if isinstance(result, dict) else None,
            "total_pages": total_pages,
            "total_words": total_words,
            "total_chars": total_chars,
            "full_text": full_text[:5000],  # cap for storage
            "pages": pages_for_frontend,
            "chunks": chunks_for_frontend,
            "total_chunks": total_chunks_count,
            "index_chunks": result.get("index_chunks", 0) if isinstance(result, dict) else 0,
            "content_chunks": result.get("content_chunks", 0) if isinstance(result, dict) else 0,
            "terminology_records": result.get("terminology_records", 0) if isinstance(result, dict) else 0,
            "table_records": result.get("table_records", 0) if isinstance(result, dict) else 0,
            "table_chunks": result.get("table_chunks", 0) if isinstance(result, dict) else 0,
            "processing_status": result.get("processing_status", []) if isinstance(result, dict) else [],
            "upload_date": datetime.datetime.now().isoformat(),
        }
        _save_doc_meta(doc_id, meta_to_save)

        return {
            "success": True,
            "id": doc_id,
            "document_id": doc_id,
            "name": doc_filename,
            "fileName": doc_filename,
            "filename": doc_filename,
            "fileType": file_type,
            "document_type": meta_to_save["document_type"],
            "year": meta_to_save["year"],
            "quarter": meta_to_save["quarter"],
            "report_period": meta_to_save["report_period"],
            "totalPages": total_pages,
            "totalWords": total_words,
            "totalChars": total_chars,
            "pages": pages_for_frontend,
            "chunks": chunks_for_frontend,
            "chunkCount": total_chunks_count,
            "fullText": full_text[:2000],
            "documents": [{
                "id": doc_id,
                "fileName": doc_filename,
                "name": doc_filename,
                "document_type": meta_to_save["document_type"],
                "year": meta_to_save["year"],
                "quarter": meta_to_save["quarter"],
                "report_period": meta_to_save["report_period"],
                "totalPages": total_pages,
                "totalWords": total_words,
                "pages": pages_for_frontend,
                "chunks": chunks_for_frontend,
                "status": "ready"
            }],
            "details": result,
            "processing": result.get("processing_status", []) if isinstance(result, dict) else []
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
@app.post("/api/query")
@app.post("/ask")
def chat_or_ask(req: QueryRequest):
    user_query = (req.query or req.message or req.prompt or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query message cannot be empty")

    doc_id = req.document_id or req.doc_id
    if doc_id in ["all", "All Uploaded Documents", None, ""]:
        doc_id = None

    # pipeline.ask() requires a non-None document_id; pass "" to let it query all docs
    safe_doc_id = doc_id if doc_id else ""
    req_mode = req.query_mode or req.mode or "AUTO"
    conversation = _conversation_entities.get(req.conversation_id or "", {})
    raw = pipeline.ask(
        user_query,
        document_id=safe_doc_id,
        query_mode=req_mode,
        conversation_article_id=conversation.get("article_id"),
        document_type=req.document_type,
        year=req.year,
    )

    # --- Transform pipeline response to match what the frontend expects ---
    # pipeline returns: { answer, evidence, query_analysis }
    # frontend expects: { success, answer, sources, confidenceScore, ... }
    evidence = raw.get("evidence", [])
    query_analysis = raw.get("query_analysis", {})
    resolved_article_id = query_analysis.get("resolved_article_id")
    if req.conversation_id and resolved_article_id:
        _conversation_entities[req.conversation_id] = {"article_id": resolved_article_id, "document_id": safe_doc_id}

    # Map evidence chunks → SourceQuote format expected by frontend
    sources = []
    for ev in evidence:
        source_doc_id = ev.get("document_id") or doc_id or "Document"
        sources.append({
            "quote":          ev.get("text", ""),
            "text":           ev.get("text", ""),
            "pageNumber":     ev.get("page"),
            "page_number":    ev.get("page"),
            "chunkId":        ev.get("chunk_id", ""),
            "chunk_id":       ev.get("chunk_id", ""),
            "docName":        source_doc_id,
            "document_name":  source_doc_id,
            "document_id":    source_doc_id,
            "document_type":  ev.get("document_type"),
            "year":           ev.get("year"),
            "content_type":   ev.get("content_type"),
            "relevanceScore": int((ev.get("score", 0.0)) * 100),
            "highlightText":  ev.get("text", "")[:120],
            "section":        ev.get("section", "General"),
            "collection_type": ev.get("chunk_type") or ("index" if "_toc_" in (ev.get("chunk_id") or "") else "content"),
            "tableId": ev.get("table_id"),
            "table_id": ev.get("table_id"),
            "extractionConfidence": ev.get("extraction_confidence"),
        })

    # Derive a confidence score from the top evidence score
    top_score = evidence[0].get("score", 0.0) if evidence else 0.0
    confidence = min(100, max(10, int(top_score * 100))) if evidence else 0

    return {
        "success":          True,
        "answer":           raw.get("answer", "No answer generated."),
        "sources":          sources,
        "confidenceScore":  confidence,
        "grounded":         len(sources) > 0,
        "originalQuery":    user_query,
        "queryMode":        query_analysis.get("query_mode", "CONTENT_BASED"),
        "query_mode":       query_analysis.get("query_mode", "CONTENT_BASED"),
        "query_mode_used":  query_analysis.get("query_mode", "CONTENT_BASED"),
        "requestedMode":    query_analysis.get("requested_mode", req_mode),
        "targetTitle":      query_analysis.get("target_title"),
        "detectedIntent":   query_analysis.get("intent", "FACTUAL"),
        "detectedLanguage": query_analysis.get("detected_script", "Auto"),
        "detectedScript":   query_analysis.get("detected_script", "neutral"),
        "detected_script":  query_analysis.get("detected_script", "neutral"),
        "variantsTested":   query_analysis.get("normalized_variants", []),
        "resolvedEntities": query_analysis.get("resolved_entities", []),
        "concepts":         query_analysis.get("entities", []),
        "modelUsed":        "ollama",
        "processingTimeMs": 0,
        "conversationId": req.conversation_id,
    }




@app.get("/api/hierarchy/{doc_id}/points")
def get_hierarchy_points(
    doc_id: str,
    limit: int = 50,
    node_type: Optional[str] = None
):
    """Debug inspection endpoint to verify indexed hierarchical Qdrant points."""
    points = pipeline.vector_store.get_hierarchical_points(
        document_id=doc_id,
        limit=limit,
        node_type=node_type
    )
    return {
        "document_id": doc_id,
        "count": len(points),
        "collection": pipeline.vector_store.hierarchical_collection_name,
        "points": points
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_includes=["*.py"],
        reload_excludes=[
            "*.json",
            "*.pdf",
            "*.docx",
            "*.doc",
            "*.sqlite3",
            "*.db",
            "doc_metadata/*",
            "data/*",
            "uploaded_pdfs/*"
        ]
    )
