import sys
import re
from typing import Dict, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from vector_store import VectorStore
from query_analyzer import SemanticQueryAnalyzer
from retriever import HybridRetriever
from reranker import EvidenceReranker
from generator import OllamaGroundedGenerator
from document_extractor import detect_file_type, extract_document_pages
from extractor import structure_aware_chunking
from term_extractor import TerminologyExtractor
from pdf_segmenter import PDFSegmenter
from metadata_extractor import DocumentMetadataExtractor
from index_knowledge_layer import (
    StructuredIndexStore,
    StructuredIndexEngine,
    TOCExtractor
)
from table_extractor import TableExtractor, table_chunks
from table_store import StructuredTableStore
from table_query_engine import StructuredTableEngine
from structured_store import StructuredRecordStore, build_structured_records, FieldNormalizer


class RAGPipeline:

    def __init__(self):
        print("[RAGPipeline] Initializing...")

        self.vector_store = VectorStore()
        self.analyzer = SemanticQueryAnalyzer(embed_fn=self.vector_store.embed_fn)
        FieldNormalizer.initialize(
            self.vector_store.embed_fn,
            transliterate_fn=self.analyzer.algorithmic_transliterate
        )
        self.term_extractor = TerminologyExtractor()
        self.segmenter = PDFSegmenter()

        # Dynamic Structured Index Knowledge Layer
        self.index_store = StructuredIndexStore()
        self.index_engine = StructuredIndexEngine(self.index_store, embed_fn=self.vector_store.embed_fn)
        self.table_extractor = TableExtractor()
        self.table_store = StructuredTableStore()
        self.table_engine = StructuredTableEngine(self.table_store, embed_fn=self.vector_store.embed_fn)
        self.structured_store = StructuredRecordStore()

        self.retriever = HybridRetriever(
            self.vector_store,
            self.analyzer
        )
        self.reranker = EvidenceReranker()
        self.generator = OllamaGroundedGenerator()

        print("[RAGPipeline] Initialization complete.")

    # =========================================================
    # PDF INDEXING (DYNAMIC FOR ANY PDF)
    # =========================================================

    def index_document(
        self,
        file_path: str,
        document_id: str,
        filename: str,
        document_type: str = "magazine"
    ) -> Dict[str, Any]:

        print(f"[RAGPipeline] Indexing document: {filename} ({document_id}) [Type: {document_type}]")

        # -----------------------------------------
        file_type = detect_file_type(filename)
        # 1. Extract raw pages (preserving text)
        # -----------------------------------------
        pages = extract_document_pages(file_path)

        if not pages:
            raise ValueError(f"No readable content was extracted from the {file_type.upper()} document.")

        full_text = "\n\n".join(p["text"] for p in pages if p.get("text"))

        # Extract dynamic document metadata (year, quarter, report_period)
        doc_meta = DocumentMetadataExtractor.extract_metadata(pages, filename, document_type=document_type)
        extracted_year = doc_meta.get("year")
        extracted_doc_type = doc_meta.get("document_type", document_type)
        quarter = doc_meta.get("quarter")
        report_period = doc_meta.get("report_period")
        print(f"[RAGPipeline] Extracted metadata: Year={extracted_year}, Type={extracted_doc_type}, Quarter={quarter}, Period={report_period}")

        # Extract structured tables/forms independently from normal text.
        try:
            tables = self.table_extractor.extract(file_path, pages, document_id, filename, file_type)
            self.table_store.save_tables(document_id, filename, tables)
            structured_records = build_structured_records(document_id, pages, tables)
            self.structured_store.save_records(document_id, filename, structured_records)
            vector_table_chunks = table_chunks(
                tables,
                source_filename=filename,
                document_name=filename,
                document_type=extracted_doc_type,
                year=extracted_year
            )
            if vector_table_chunks:
                self.vector_store.add_chunks(vector_table_chunks, collection_type="table")
            print(f"[RAGPipeline] Tables: document={document_id}, extracted={len(tables)}, structured_records={len(structured_records)}, embedded={len(vector_table_chunks)}")
        except Exception as exc:
            safe_msg = str(exc).encode("ascii", errors="replace").decode("ascii")
            print(f"[RAGPipeline] Non-blocking table extraction notice: {safe_msg}")
            tables = []
            structured_records = []
            vector_table_chunks = []

        # -----------------------------------------
        # 3. Dynamic Structured TOC Extraction & Knowledge Layer
        # -----------------------------------------
        try:
            toc_records = TOCExtractor.extract_records_from_pages(
                pages_data=pages,
                document_id=document_id,
                filename=filename
            )
            self.index_store.save_records(document_id, filename, toc_records)
            print(f"[RAGPipeline] Created {len(toc_records)} structured TOC records in Index Knowledge Layer.")
        except Exception as e:
            print(f"[RAGPipeline] Notice extracting structured TOC records: {e}")

        # -----------------------------------------
        # 4. Dynamic PDF Structural Segmentation
        # -----------------------------------------
        index_pages, content_pages = self.segmenter.segment_pages(pages)

        # -----------------------------------------
        # 5. Structure-aware Chunking for Dual Regions
        # -----------------------------------------
        index_chunks = structure_aware_chunking(
            index_pages,
            document_id=document_id,
            source_filename=filename,
            document_name=filename,
            document_type=extracted_doc_type,
            year=extracted_year
        ) if index_pages else []

        content_chunks = structure_aware_chunking(
            content_pages,
            document_id=document_id,
            source_filename=filename,
            document_name=filename,
            document_type=extracted_doc_type,
            year=extracted_year
        ) if content_pages else []

        total_chunks = index_chunks + content_chunks
        if not total_chunks:
            raise ValueError("No chunks were generated from the PDF.")

        # -----------------------------------------
        # 6. Populate Qdrant Collections
        # -----------------------------------------
        if index_chunks:
            self.vector_store.add_chunks(index_chunks, collection_type="index")

        if content_chunks:
            self.vector_store.add_chunks(content_chunks, collection_type="content")

        # Attach source chunk references after IDs exist
        try:
            terminology_count = self.term_extractor.extract_and_register_from_text(
                full_text, document_id=document_id, source_filename=filename, chunks=total_chunks
            )
        except Exception as exc:
            print(f"[RAGPipeline] Non-blocking terminology extraction notice: {exc}")
            terminology_count = 0

        print(
            f"[RAGPipeline] Indexed '{filename}' -> "
            f"Pages={len(pages)} (Index={len(index_pages)}, Content={len(content_pages)}), "
            f"Chunks={len(total_chunks)} (Index={len(index_chunks)}, Content={len(content_chunks)}, Table={len(vector_table_chunks)}), "
            f"DocumentID={document_id}"
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "document_name": filename,
            "document_type": extracted_doc_type,
            "year": extracted_year,
            "quarter": quarter,
            "report_period": report_period,
            "file_type": file_type,
            "total_pages": len(pages),
            "index_pages": len(index_pages),
            "content_pages": len(content_pages),
            "total_chunks": len(total_chunks),
            "index_chunks": len(index_chunks),
            "content_chunks": len(content_chunks),
            "table_chunks": len(vector_table_chunks),
            "structured_toc_records": len(self.index_store.get_records(document_id)),
            "terminology_terms": terminology_count,
            "table_records": len(tables),
            "structured_records": len(structured_records),
            "status": "ready"
        }

    # Existing callers retain this API while new callers use the format-neutral name.
    def index_pdf(self, file_path: str, document_id: str, filename: str) -> Dict[str, Any]:
        return self.index_document(file_path, document_id, filename)

    # =========================================================
    # QUESTION ANSWERING
    # =========================================================

    def ask(
        self,
        query: str,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        year: Optional[int] = None,
        query_mode: str = "AUTO",
        conversation_article_id: Optional[str] = None
    ) -> Dict[str, Any]:

        if not query or not query.strip():
            return {
                "answer": "Please enter a question.",
                "evidence": [],
                "query_analysis": {
                    "query_mode": query_mode
                }
            }

        # If no specific document_id, search across all indexed documents
        if not document_id or document_id in ["all", "All Uploaded Documents", ""]:
            document_id = None

        # =====================================================
        # 1. UNDERSTAND USER QUESTION & ROUTE MODE
        # =====================================================
        analysis = self.analyzer.analyze(query, requested_mode=query_mode, document_id=document_id)
        resolved_mode = analysis.get("query_mode", "CONTENT_BASED")
        safe_q = str(query).encode("ascii", errors="replace").decode("ascii")
        print(f"[RAGPipeline] Query route: document={document_id}, mode={resolved_mode}, "
              f"intent={analysis.get('intent')}, script={analysis.get('detected_script')}, query={safe_q!r}")

        # =====================================================
        # 2. DYNAMIC STRUCTURED INDEX KNOWLEDGE LAYER EVALUATION
        # =====================================================
        slot_data = self.index_engine.classify_intent(query, document_id)
        index_intent = slot_data.get("intent")
        is_index_intent = index_intent in {
            "COUNT", "LIST", "FIRST_ITEM", "LAST_ITEM", "NTH_ITEM",
            "ARTICLE_BY_PAGE", "PAGE_BY_TITLE", "AUTHOR_BY_TITLE",
            "TITLE_BY_AUTHOR", "EXISTS", "ARTICLES_BY_SECTION"
        }
        req_mode_upper = (query_mode or "AUTO").upper().strip()
        has_magazine_or_index_subject = any(
            k in query.lower() for k in [
                "magazine", "patrika", "article", "articles", "lekh", "kavita",
                "poem", "technical document", "technical documents", "तकनीकी", "गैर-तकनीकी"
            ]
        )
        is_legal_or_issued = bool(re.search(
            r"(?<![\w\u0900-\u097F])(?:धारा\s*3|dhara\s*3|section\s*3|jaari|jari|जारी|issue|issues|issued|sachiv|सचिव|patra|patron|पत्र|पत्रों|file|files|faile|फाइल|फाइलें)(?![\w\u0900-\u097F])",
            query.lower(),
            re.IGNORECASE
        ))
        should_check_index = not is_legal_or_issued and (
            (resolved_mode == "INDEX_BASED" and has_magazine_or_index_subject) or (
                is_index_intent and (slot_data.get("section") is not None or has_magazine_or_index_subject)
            )
        )

        if should_check_index:
            index_res = self.index_engine.execute_index_query(
                query=query,
                document_id=document_id,
                query_mode=query_mode,
                conversation_article_id=conversation_article_id
            )
            if index_res and index_res.get("answer"):
                slot_data = index_res.get("slot_data", slot_data)
                return {
                    "answer": index_res["answer"],
                    "evidence": index_res["evidence"],
                    "query_analysis": {
                        "intent": slot_data.get("intent", analysis.get("intent", "INDEX_QUERY")),
                        "query_mode": "INDEX_BASED",
                        "requested_mode": query_mode,
                        "detected_script": analysis.get("detected_script", "neutral"),
                        "target_title": slot_data.get("entity_candidate") or analysis.get("target_title"),
                        "target_entity": slot_data.get("target_entity"),
                        "requested_field": slot_data.get("requested_field"),
                        "entity_resolution": slot_data.get("entity_resolution", {}),
                        "resolved_article_id": (index_res.get("matched_records") or [{}])[0].get("article_id") if index_res.get("matched_records") else None,
                        "entities": analysis.get("entities", []),
                        "normalized_variants": analysis.get("normalized_variants", []),
                        "resolved_entities": analysis.get("resolved_entities", [])
                    }
                }
            elif is_index_intent and should_check_index:
                # If an explicit index query (author, page, ordinal, count) finds no matching record,
                # do NOT fall through to Content RAG to hallucinate or return unrelated editorial chunks.
                is_hindi = bool(re.search(r"[\u0900-\u097F]|kaun|kis|hai|lekhak|likha|ke lekhak", query.lower()))
                entity = slot_data.get("entity_candidate") or ""
                if is_hindi:
                    msg = f"दस्तावेज़ के अनुक्रमणिका (TOC) में '{entity}' से संबंधित कोई रचना या जानकारी उपलब्ध नहीं है।" if entity else "दस्तावेज़ के अनुक्रमणिका (TOC) में इस प्रश्न से संबंधित कोई जानकारी उपलब्ध नहीं है।"
                else:
                    msg = f"No record or information for '{entity}' was found in the document index." if entity else "No relevant record was found in the document index."
                return {
                    "answer": msg,
                    "evidence": [],
                    "query_analysis": {
                        "intent": index_intent,
                        "query_mode": "INDEX_BASED",
                        "requested_mode": query_mode,
                        "detected_script": analysis.get("detected_script", "neutral"),
                        "target_title": entity,
                        "target_entity": slot_data.get("target_entity"),
                        "requested_field": slot_data.get("requested_field"),
                        "entity_resolution": {},
                        "resolved_article_id": None,
                        "entities": analysis.get("entities", []),
                        "normalized_variants": analysis.get("normalized_variants", []),
                        "resolved_entities": analysis.get("resolved_entities", [])
                    }
                }

        # =====================================================
        # 3. STRUCTURED LOOKUP. A canonical field is always resolved against the
        # persisted structured record store first. Article chunks are never evidence
        # for a metadata/form field question.
        # =====================================================
        if analysis.get("table_intent"):
            field = analysis.get("requested_field")
            if field:
                # For address, combine all records (office name line + street continuation).
                if field == "address":
                    all_addr = self.structured_store.find_all(document_id, field)
                    # De-duplicate values while preserving order (main address first)
                    seen_vals: set = set()
                    unique_parts: list = []
                    for r in all_addr:
                        v = str(r.get("value", "")).strip()
                        if v and v not in seen_vals:
                            seen_vals.add(v)
                            unique_parts.append((r, v))
                    if unique_parts:
                        combined_value = "\n".join(v for _, v in unique_parts)
                        first_r = unique_parts[0][0]
                        evidence = {
                            "chunk_id": f"structured:{first_r.get('table_id') or field}",
                            "document_id": first_r["document_id"],
                            "text": combined_value,
                            "page": first_r.get("page"),
                            "section": first_r.get("section"),
                            "score": 1.0,
                            "chunk_type": "structured",
                            "table_id": first_r.get("table_id"),
                            "field": field,
                            "label": first_r["label"],
                            "value": combined_value,
                            "extraction_confidence": first_r.get("confidence", 1.0),
                        }
                        return {
                            "answer": combined_value,
                            "evidence": [evidence],
                            "query_analysis": {**analysis, "query_mode": analysis.get("query_mode"),
                                               "structured_evidence": first_r},
                        }
                    return {"answer": "The requested structured field was not found in this document.",
                            "evidence": [],
                            "query_analysis": {**analysis, "query_mode": analysis.get("query_mode")}}

                record = self.structured_store.find(document_id, field)
                print("\n================ STRUCTURED QUERY DEBUG ================\n"
                      f"QUERY: {query}\nLANGUAGE: {analysis.get('detected_script')}\n"
                      f"INTENT: {analysis.get('intent')}\nFIELD: {field}\nDOCUMENT_ID: {document_id}\n"
                      f"MATCH: {record}\nROUTE: {analysis.get('query_mode')}\n========================================================")
                if record:
                    evidence = {"chunk_id": f"structured:{record.get('table_id') or field}",
                                "document_id": record["document_id"], "text": str(record["value"]),
                                "page": record.get("page"), "section": record.get("section"), "score": 1.0,
                                "chunk_type": "structured", "table_id": record.get("table_id"),
                                "field": record["field"], "label": record["label"], "value": record["value"],
                                "extraction_confidence": record.get("confidence", 1.0)}
                    return {"answer": self._structured_answer(query, record), "evidence": [evidence],
                            "query_analysis": {**analysis, "query_mode": analysis.get("query_mode"),
                                               "structured_evidence": record}}
                return {"answer": "The requested structured field was not found in this document.", "evidence": [],
                        "query_analysis": {**analysis, "query_mode": analysis.get("query_mode")}}

            # Queries requesting a form's field labels can use the table engine,
            # but still must not fall through to generic content retrieval.
            print(f"[RAGPipeline] Structured form lookup document={document_id}; searching persisted tables directly")
            table_result = self.table_engine.answer(
                query, document_id, (), (),
                query_variants=analysis.get("normalized_variants")
            )
            if table_result and not analysis.get("is_hybrid"):
                try:
                    print(f"[RAGPipeline] Structured answer: document={document_id}, "
                          f"details={table_result.get('details')}, evidence={table_result.get('evidence')}")
                except Exception:
                    pass
                return {
                    "answer": table_result["answer"], "evidence": table_result["evidence"],
                    "query_analysis": {**analysis, "query_mode": analysis.get("query_mode", "FORM_BASED"), "table_details": table_result["details"]}
                }
            # If table lookup did not find a matching cell, fall through to content RAG
            # so questions about magazine articles, technical documents, or text sections are answered.
            pass

        # =====================================================
        # 4. CONTENT-BASED RAG PIPELINE (VECTOR + RERANK + LLM)
        # =====================================================
        filter_doc_type = document_type or analysis.get("target_document_type")
        filter_year = year if year is not None else analysis.get("target_year")

        candidates = self.retriever.retrieve_candidates(
            query,
            document_id=document_id,
            document_type=filter_doc_type,
            year=filter_year,
            query_mode="CONTENT_BASED" if resolved_mode in {"TABLE_BASED", "FORM_BASED", "REPORT_METADATA"} else resolved_mode,
            top_k=20
        )

        if not candidates:
            return {
                "answer": "यह जानकारी दिए गए दस्तावेज़ में उपलब्ध नहीं है।",
                "evidence": [],
                "query_analysis": analysis
            }

        # Rerank Evidence
        best_evidence = self.reranker.rerank(
            analysis,
            candidates,
            top_n=5
        )

        if not best_evidence:
            return {
                "answer": "इस प्रश्न का उत्तर देने के लिए पर्याप्त प्रमाण दस्तावेज़ में नहीं मिला।",
                "evidence": [],
                "query_analysis": analysis
            }

        if not self._has_sufficient_content_evidence(query, best_evidence):
            return {
                "answer": "इस प्रश्न का उत्तर दिए गए दस्तावेज़ के प्रमाण में उपलब्ध नहीं है।",
                "evidence": [],
                "query_analysis": analysis
            }

        # Generate Grounded Answer with LLM
        verified_table_data = table_result.get("details") if analysis.get("table_intent") and table_result else None
        raw_answer = self.generator.generate_answer(query, best_evidence, verified_metadata=verified_table_data)

        # Format Evidence For Frontend
        formatted_evidence = []
        for chunk in best_evidence:
            metadata = chunk.get("metadata", {})
            raw_page = metadata.get("page_number", metadata.get("page"))
            formatted_evidence.append({
                "chunk_id": chunk.get("id"),
                "document_id": metadata.get("document_id"),
                "document_name": metadata.get("document_name") or metadata.get("document_id"),
                "document_type": metadata.get("document_type"),
                "year": metadata.get("year"),
                "text": chunk.get("text", ""),
                "page": raw_page if raw_page not in (0, "0") else None,
                "section": metadata.get("section", "General"),
                "score": chunk.get("final_score", chunk.get("score", 0.85)),
                "chunk_type": metadata.get("collection_type", "content")
            })

        if analysis.get("table_intent") and table_result:
            formatted_evidence = table_result["evidence"] + formatted_evidence
        return {
            "answer": raw_answer,
            "evidence": formatted_evidence,
            "query_analysis": {
                "intent": analysis.get("intent", "semantic_qa"),
                "query_mode": analysis.get("query_mode", "CONTENT_BASED"),
                "requested_mode": analysis.get("requested_mode", query_mode),
                "detected_script": analysis.get("detected_script", "neutral"),
                "target_title": analysis.get("target_title", None),
                "entities": analysis.get("entities", []),
                "normalized_variants": analysis.get("normalized_variants", []),
                "resolved_entities": analysis.get("resolved_entities", [])
            }
        }

    def delete_document(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)
        self.index_store.delete_document(document_id)
        self.term_extractor.memory.delete_document(document_id)
        self.table_store.delete_document(document_id)
        self.structured_store.delete_document(document_id)

    @staticmethod
    def _has_sufficient_content_evidence(query: str, evidence: list[Dict[str, Any]]) -> bool:
        """Return true only when retrieved text has direct lexical support.

        Semantic similarity is useful for finding candidates, but must not by
        itself authorize a generated answer.  A meaningful question term must
        occur in the retrieved evidence, and the highest reranker score must
        clear a modest quality floor.  This intentionally prefers an honest
        "not found" response over an unsupported answer.
        """
        stopwords = {
            "what", "which", "who", "when", "where", "why", "how", "is", "are", "was", "were",
            "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "does", "did", "do",
            "please", "tell", "me", "about", "document", "question", "kya", "ka", "ke", "ki",
            "hai", "hain", "tha", "thi", "mein", "me", "ko", "se", "aur", "ye", "woh",
            "क्या", "का", "के", "की", "है", "हैं", "था", "थी", "में", "को", "से", "और", "यह", "वह"
        }
        terms = {
            token for token in re.findall(r"[A-Za-z0-9\u0900-\u097F]+", query.lower())
            if len(token) > 1 and token not in stopwords
        }
        if not terms:
            # A query without a meaningful subject cannot be safely grounded.
            return False

        evidence_text = " ".join(chunk.get("text", "").lower() for chunk in evidence)
        has_term_support = any(term in evidence_text for term in terms)
        top_score = float(evidence[0].get("rerank_score", 0.0))
        return has_term_support and top_score >= 0.30

    @staticmethod
    def _structured_answer(query: str, record: Dict[str, Any]) -> str:
        """Ground answer strictly in the document's extracted label and value.

        No domain vocabulary or answer templates are hardcoded; the label and value
        are extracted directly from the user's uploaded document.
        """
        value = str(record.get("value", "")).strip()
        label = str(record.get("label", "")).strip()

        # If a valid label exists in the document, present it cleanly with the value
        if label and not label.lower().startswith("address continuation"):
            return f"{label}: {value}"
        return value

