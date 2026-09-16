import sys
import os
import uvicorn

if __name__ == "__main__":
    # Ensure stdout/stderr UTF-8 encoding on Windows
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    print("[Rajbhasha RAG] Starting Python FastAPI Backend on http://127.0.0.1:8000 ...")
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
