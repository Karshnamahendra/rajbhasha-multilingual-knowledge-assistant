"""End-to-end RAG orchestrator coordinating document parsing, segmentation, Qdrant indexing, and query resolution."""
import sys
import re
from typing import Dict, Any, Optional, List, Tuple

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
from generator import OllamaGroundedGenerator, extract_factual_span
from extractor import (
    detect_file_type,
    extract_document_pages,
    structure_aware_chunking,
    DocumentMetadataExtractor
)
from term_extractor import TerminologyExtractor
from pdf_segmenter import PDFSegmenter
from index_knowledge_layer import (
    StructuredIndexStore,
    StructuredIndexEngine,
    TOCExtractor
)
from table_extractor import TableExtractor, table_chunks, StructuredTableStore
from table_query_engine import StructuredTableEngine
from structured_store import StructuredRecordStore, build_structured_records, FieldNormalizer
from hierarchical_model import HierarchicalStore
from hierarchical_builder import build_document_hierarchy
from docx_toc import extract_docx_toc_records


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
        self.hierarchical_store = HierarchicalStore()

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
        extracted_doc_type = doc_meta.get("document_type", "magazine")
        quarter = doc_meta.get("quarter")
        report_period = doc_meta.get("report_period")
        print(f"[RAGPipeline] Extracted metadata: Year={extracted_year}, Type={extracted_doc_type}, Quarter={quarter}, Period={report_period}")

        # IDEMPOTENCY: Clear any prior points for this document_id across all collections
        self.vector_store.delete_document(document_id)

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
            # DOCX magazines often keep the contents page as a Word table. Reading the
            # cells directly keeps title / author(s) / page / category apart, which the
            # text-based parser above merges ("title : : : author", no page).
            if file_type == "docx":
                table_toc = extract_docx_toc_records(file_path, document_id, filename, pages=pages)
                if table_toc and len(table_toc) >= max(5, int(0.8 * len(toc_records))):
                    print(f"[RAGPipeline] Using DOCX contents table: {len(table_toc)} entries "
                          f"(text parser found {len(toc_records)}).")
                    toc_records = table_toc
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
            year=extracted_year,
            content_type="index",
            chunk_prefix="idx"
        ) if index_pages else []

        content_chunks = structure_aware_chunking(
            content_pages,
            document_id=document_id,
            source_filename=filename,
            document_name=filename,
            document_type=extracted_doc_type,
            year=extracted_year,
            content_type="content",
            chunk_prefix="cnt"
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

        # -------------------------------------------------------------
        # 7. Universal Dynamic Hierarchical Document Representation
        # -------------------------------------------------------------
        total_hierarchy_nodes = 0
        total_hierarchical_points = 0
        try:
            doc_meta = {
                "document_id": document_id,
                "document_name": filename,
                "filename": filename,
                "document_type": extracted_doc_type,
                "year": extracted_year,
                "quarter": quarter,
                "report_period": report_period,
                "total_pages": len(pages),
            }
            doc_hierarchy = build_document_hierarchy(
                file_path=file_path,
                pages=pages,
                tables=tables,
                toc_records=self.index_store.get_records(document_id),
                doc_meta=doc_meta
            )
            self.hierarchical_store.save_hierarchy(document_id, filename, doc_hierarchy)
            total_hierarchy_nodes = doc_hierarchy.root.count_nodes()

            # Step 3: Index universal hierarchy into Qdrant hierarchical collection
            hier_points = doc_hierarchy.to_indexable_points(include_branches=True)
            total_hierarchical_points = self.vector_store.add_hierarchical_points(hier_points)
            print(
                f"[RAGPipeline] Created universal hierarchy for '{filename}': "
                f"{total_hierarchy_nodes} recursive nodes, "
                f"indexed {total_hierarchical_points} points into Qdrant '{self.vector_store.hierarchical_collection_name}'."
            )
        except Exception as exc:
            print(f"[RAGPipeline] Non-blocking universal hierarchy notice: {exc}")

        print(
            f"[RAGPipeline] Indexed '{filename}' -> "
            f"Pages={len(pages)} (Index={len(index_pages)}, Content={len(content_pages)}), "
            f"Chunks={len(total_chunks)} (Index={len(index_chunks)}, Content={len(content_chunks)}, Table={len(vector_table_chunks)}), "
            f"HierarchyNodes={total_hierarchy_nodes}, HierarchicalPoints={total_hierarchical_points}, "
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
            "hierarchy_nodes": total_hierarchy_nodes,
            "hierarchical_points": total_hierarchical_points,
            "status": "ready"
        }

    # Existing callers retain this API while new callers use the format-neutral name.
    def index_pdf(self, file_path: str, document_id: str, filename: str) -> Dict[str, Any]:
        return self.index_document(file_path, document_id, filename)

    # =========================================================
    # HIERARCHICAL BRANCH EXPANSION HELPER
    # =========================================================

    def _expand_branch_evidence(
        self,
        evidence: List[Dict[str, Any]],
        query: str,
        analysis: Dict[str, Any],
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generic branch node child expansion.

        After initial reranking, any branch node (node_type='branch', node_value=None)
        in the top-3 results is expanded by fetching its direct children from Qdrant
        via path-prefix matching. The expanded pool is then re-reranked.

        This allows the generator to answer:
          - "What is in section X?" (branch aggregation — fixes K/L)
          - "What page is article X on?" (fetches पृष्ठ संख्या child leaf — fixes I)
          - Any hierarchical "description" question where the relevant data is in children

        No document-specific logic. Works for any hierarchy shape or document type.
        """
        expanded = list(evidence)
        existing_ids = {e.get("id") for e in expanded}
        is_branch_enum = (
            analysis.get("intent") == "branch_enumeration"
            or analysis.get("is_branch_enumeration")
        )

        seen_prefixes = set()
        for cand in evidence[:3]:   # only expand top-3 to keep context focused
            branch_path = cand.get("hierarchy_path") or []
            if not branch_path:
                continue

            target_prefix = None
            if cand.get("node_type") == "branch" and cand.get("node_value") is None:
                target_prefix = branch_path
            elif is_branch_enum and len(branch_path) >= 2:
                # For branch enumeration on leaf candidates, expand siblings under common parent
                target_prefix = branch_path[:-1]

            if not target_prefix:
                continue

            doc_scope = cand.get("document_id") or cand.get("metadata", {}).get("document_id") or document_id
            print(f"[RAGPipeline:Debug] _expand_branch_evidence cand_id={cand.get('id')} key={cand.get('node_key')} path={branch_path} doc={doc_scope} target_prefix={target_prefix}")

            prefix_key = (doc_scope, " > ".join(target_prefix))
            if prefix_key in seen_prefixes:
                continue
            seen_prefixes.add(prefix_key)

            n_expand = 100 if is_branch_enum else 12
            children = self.vector_store.search_children_by_path_prefix(
                path_prefix=target_prefix,
                document_id=doc_scope,
                n_results=n_expand
            )
            print(f"[RAGPipeline:Debug] search_children_by_path_prefix prefix={target_prefix} doc={doc_scope} -> found {len(children)}: {[c.get('node_key') for c in children][:15]}")
            added = 0
            for child in children:
                if child.get("id") not in existing_ids:
                    existing_ids.add(child.get("id"))
                    expanded.append(child)
                    added += 1
            if added:
                print(
                    f"[RAGPipeline] Branch expanded: '{target_prefix}' "
                    f"→ +{added} child nodes (doc: {doc_scope})"
                )

        # Re-rerank the expanded pool so the most relevant nodes (including new
        # children) surface at the top. Cap at 25 for branch enumeration to preserve all siblings.
        top_cap = 25 if is_branch_enum else 8
        if len(expanded) > len(evidence):
            expanded = self.reranker.rerank(analysis, expanded, top_n=top_cap)

        return expanded

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

        # =========================================================================
        # 1.5 PRIMARY UNIFIED HIERARCHICAL RETRIEVAL (Step 4 Primary Search Branch)
        # Search rajbhasha_hierarchical_collection as unified knowledge representation
        # across all document formats (magazine TOC, quarterly reports, tables, forms).
        # =========================================================================
        hier_candidates = self.retriever.retrieve_hierarchical_candidates(
            query=query,
            document_id=document_id,
            document_type=document_type,
            year=year,
            top_k=35,
            analysis=analysis
        )

        if hier_candidates:
            hier_best_evidence = self.reranker.rerank(
                analysis,
                hier_candidates,
                top_n=5
            )

            # ── Branch Node Child Expansion (fixes K/L/I) ─────────────────────
            # For any branch node in top-3 results, fetch its child nodes from
            # Qdrant using a generic path-prefix scroll. This lets the generator
            # answer "what is in this section?" and "what page is article X on?"
            # by seeing the branch's concrete children (including पृष्ठ संख्या leaves).
            # No document-specific logic — works for any hierarchy shape.
            hier_best_evidence = self._expand_branch_evidence(
                hier_best_evidence,
                query=query,
                analysis=analysis,
                document_id=document_id
            )

            # Accept any hierarchical candidate that clears the reranker quality floor.
            # The threshold is intentionally permissive (>= 0.30) because the reranker
            # already applies multi-factor scoring; a lower score still means the node
            # is the best match available and should be attempted.
            if hier_best_evidence and hier_best_evidence[0].get("rerank_score", 0.0) >= 0.30:
                top_cand = hier_best_evidence[0]
                top_val = top_cand.get("node_value")
                top_key = top_cand.get("node_key")

                # Value-Type & Content Quality Gating
                exp_type = analysis.get("expected_value_type")
                if exp_type == "DURATION":
                    cand_dur_txt = (str(top_val or "") + " " + str(top_key or "") + " " + str(top_cand.get("hierarchy_path_text") or "")).lower()
                    if not re.search(r'(?<![\w\u0900-\u097F])(?:वर्ष|साल|माह|महीने|दिन|घंटे|years?|months?|days?|hours?)(?![\w\u0900-\u097F])', cand_dur_txt):
                        dur_cands = [
                            c for c in hier_best_evidence
                            if re.search(r'(?<![\w\u0900-\u097F])(?:वर्ष|साल|माह|महीने|दिन|घंटे|years?|months?|days?|hours?)(?![\w\u0900-\u097F])',
                                         (str(c.get("node_value") or "") + " " + str(c.get("node_key") or "") + " " + str(c.get("hierarchy_path_text") or "")).lower())
                        ]
                        if dur_cands:
                            top_cand = dur_cands[0]
                            top_val = top_cand.get("node_value")
                            top_key = top_cand.get("node_key")
                        else:
                            top_cand = None

                elif exp_type in {"PERSON", "DESCRIPTION"} or analysis.get("is_content_question"):
                    val_str = str(top_val or "").strip()
                    if top_cand and (re.fullmatch(r"[-+]?\d+(?:[,.]\d+)?%?", val_str) or len(val_str.split()) <= 2):
                        desc_cands = [
                            c for c in hier_best_evidence
                            if len(str(c.get("node_value") or "").strip().split()) >= 4
                        ]
                        if desc_cands:
                            top_cand = desc_cands[0]
                            top_val = top_cand.get("node_value")
                            top_key = top_cand.get("node_key")
                        else:
                            top_cand = None

                elif exp_type == "MONETARY":
                    cand_mon_txt = (str(top_val or "") + " " + str(top_key or "") + " " + str(top_cand.get("hierarchy_path_text") or "")).lower()
                    if not re.search(r'(?:रुपए|रुपये|रु\.|rs\.?|inr|नगद|पुरस्कार|cash|prize)', cand_mon_txt):
                        mon_cands = [
                            c for c in hier_best_evidence
                            if re.search(r'(?:रुपए|रुपये|रु\.|rs\.?|inr|नगद|पुरस्कार|cash|prize)',
                                         (str(c.get("node_value") or "") + " " + str(c.get("node_key") or "") + " " + str(c.get("hierarchy_path_text") or "")).lower())
                        ]
                        if mon_cands:
                            top_cand = mon_cands[0]
                            top_val = top_cand.get("node_value")
                            top_key = top_cand.get("node_key")
                        else:
                            top_cand = None

                elif exp_type == "DATE":
                    cand_date_txt = (str(top_val or "") + " " + str(top_key or "") + " " + str(top_cand.get("hierarchy_path_text") or "")).lower()
                    if not re.search(r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|(?:दिनांक|जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)', cand_date_txt):
                        date_cands = [
                            c for c in hier_best_evidence
                            if re.search(r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|(?:दिनांक|जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)',
                                         (str(c.get("node_value") or "") + " " + str(c.get("node_key") or "") + " " + str(c.get("hierarchy_path_text") or "")).lower())
                        ]
                        if date_cands:
                            top_cand = date_cands[0]
                            top_val = top_cand.get("node_value")
                            top_key = top_cand.get("node_key")
                        else:
                            top_cand = None

                if top_cand is not None and top_cand != hier_best_evidence[0]:
                    hier_best_evidence = [top_cand] + [c for c in hier_best_evidence if c != top_cand]

                if top_cand is None:
                    print("[RAGPipeline] Hierarchical candidate did not satisfy expected value type; falling through to content RAG.")
                else:
                    print(
                        f"[RAGPipeline] Unified Hierarchical Match: id={top_cand.get('id')} | "
                        f"score={top_cand.get('rerank_score')} | path={top_cand.get('hierarchy_path_text')}"
                    )
                    is_branch_enum = (
                        analysis.get("intent") == "branch_enumeration"
                        or analysis.get("is_branch_enumeration")
                    )

                    # Always inject verified_hier_meta for any leaf node that carries a concrete value.
                    # This gives the generator an explicit authoritative anchor regardless of rerank score.
                    verified_hier_meta = None
                    top_val = top_cand.get("node_value")
                    top_key = top_cand.get("node_key")

                    if is_branch_enum:
                        # 1. Generic structural target mapping: map each requested entity to its target representations
                        meta_words = {"section", "धारा", "dhara", "rule", "नियम", "act", "अधिनियम", "report", "रिपोर्ट"}
                        target_entities = [str(e).strip() for e in analysis.get("entities", []) if str(e).strip().lower() not in meta_words]

                        branch_targets: Dict[str, set] = {}
                        for ent in target_entities:
                            e_low = ent.lower()
                            t_set = {e_low}
                            if hasattr(self, "analyzer") and hasattr(self.analyzer, "algorithmic_transliterate"):
                                try:
                                    trans = self.analyzer.algorithmic_transliterate(ent)
                                    if trans:
                                        t_set.add(trans.lower().strip())
                                except Exception:
                                    pass
                            for v in analysis.get("normalized_variants", []):
                                v_c = str(v).strip().lower()
                                if len(v_c) <= 4 and (v_c in t_set or any(t in v_c for t in t_set)):
                                    t_set.add(v_c)
                            branch_targets[ent] = t_set

                        cat_terms = {"क्षेत्र", "region", "regions", "भाग", "श्रेणी", "category", "वर्ग"}
                        q_cats = {ct for ct in cat_terms if ct in query.lower() or any(ct in v.lower() for v in analysis.get("normalized_variants", []))}

                        # 2. Segregate parent hierarchy candidates by (doc_scope, parent_path_tuple)
                        parent_candidates: Dict[Tuple[str, tuple], List[Dict[str, Any]]] = {}
                        for c in hier_best_evidence:
                            h_path = c.get("hierarchy_path") or c.get("metadata", {}).get("hierarchy_path") or []
                            doc_chunk = str(c.get("document_id") or c.get("metadata", {}).get("document_id") or c.get("file_name") or c.get("metadata", {}).get("file_name") or "")
                            if len(h_path) >= 2:
                                p_tuple = tuple(h_path[:-1])
                                parent_candidates.setdefault((doc_chunk, p_tuple), []).append(c)
                            elif len(h_path) == 1 and c.get("node_type") == "branch":
                                p_tuple = tuple(h_path)
                                parent_candidates.setdefault((doc_chunk, p_tuple), []).append(c)

                        # 3. Score candidate parent paths based on count of matched target branch entities
                        best_parent_key = None
                        best_parent_score = -1.0
                        for (p_doc, p_tuple), sibs in parent_candidates.items():
                            matched_entities_for_parent = set()
                            desc_count = 0
                            sum_rerank = 0.0
                            for s in sibs:
                                s_path = s.get("hierarchy_path") or s.get("metadata", {}).get("hierarchy_path") or []
                                if len(s_path) > len(p_tuple):
                                    sk = str(s.get("node_key") or s_path[len(p_tuple)]).strip()
                                    clean_sk = re.sub(r'["\'“”‘’\(\)\[\]]', ' ', sk).strip()
                                    sk_tokens = set(t.lower() for t in re.findall(r'[\w\u0900-\u097F]+', clean_sk) if t)

                                    for ent_name, t_set in branch_targets.items():
                                        if sk_tokens.intersection(t_set):
                                            matched_entities_for_parent.add(ent_name)

                                sv = str(s.get("node_value") or "").strip()
                                if sv and not re.fullmatch(r"[-+]?\d+(?:[,.]\d+)?%?", sv):
                                    desc_count += 1
                                sum_rerank += s.get("rerank_score", 0.0)

                            p_score = (len(matched_entities_for_parent) * 10.0) + (desc_count * 2.0) + sum_rerank
                            if p_score > best_parent_score:
                                best_parent_score = p_score
                                best_parent_key = (p_doc, p_tuple)

                        if best_parent_key:
                            best_doc, best_parent_tuple = best_parent_key
                            parent_path = list(best_parent_tuple)

                            # 4. Gather direct child candidates under best_parent_tuple for best_doc
                            candidate_pool = list(hier_best_evidence)
                            if hasattr(self, "vector_store") and hasattr(self.vector_store, "search_children_by_path_prefix"):
                                try:
                                    extra_children = self.vector_store.search_children_by_path_prefix(
                                        path_prefix=parent_path,
                                        document_id=best_doc if best_doc else None,
                                        n_results=100
                                    )
                                    existing_pool_ids = {cp.get("id") for cp in candidate_pool}
                                    for ec in extra_children:
                                        if ec.get("id") not in existing_pool_ids:
                                            existing_pool_ids.add(ec.get("id"))
                                            candidate_pool.append(ec)
                                except Exception as e_pool:
                                    print(f"[RAGPipeline:Debug] extra_children fetch notice: {e_pool}")

                            # 5. Structurally match each requested entity to EXACTLY its corresponding child leaf
                            matched_by_entity: Dict[str, Dict[str, Any]] = {}
                            fallback_matching: List[Dict[str, Any]] = []

                            for s in candidate_pool:
                                s_doc = str(s.get("document_id") or s.get("metadata", {}).get("document_id") or s.get("file_name") or s.get("metadata", {}).get("file_name") or "")
                                if best_doc and s_doc and s_doc != best_doc:
                                    continue
                                s_path = s.get("hierarchy_path") or s.get("metadata", {}).get("hierarchy_path") or []
                                if tuple(s_path[:len(parent_path)]) != best_parent_tuple or len(s_path) <= len(parent_path):
                                    continue

                                child_key = str(s.get("node_key") or s_path[len(parent_path)]).strip()
                                clean_key = re.sub(r'["\'“”‘’\(\)\[\]]', ' ', child_key).strip()
                                key_tokens = set(t.lower() for t in re.findall(r'[\w\u0900-\u097F]+', clean_key) if t)

                                cat_aligned = not q_cats or any(ct in clean_key.lower() for ct in q_cats) or any(ct in p.lower() for p in parent_path for ct in q_cats)
                                if not cat_aligned:
                                    continue

                                if branch_targets:
                                    for ent_name, t_set in branch_targets.items():
                                        if key_tokens.intersection(t_set):
                                            if ent_name not in matched_by_entity:
                                                matched_by_entity[ent_name] = s
                                            else:
                                                # Prefer candidate with non-empty descriptive node_value
                                                curr_val = str(matched_by_entity[ent_name].get("node_value") or "").strip()
                                                cand_val = str(s.get("node_value") or "").strip()
                                                if cand_val and not curr_val:
                                                    matched_by_entity[ent_name] = s
                                else:
                                    fallback_matching.append(s)

                            # Final scoped sibling leaves: strictly 1 per requested entity
                            if matched_by_entity:
                                scoped_chunks = list(matched_by_entity.values())
                            else:
                                # Fallback if no specific entity letters in query: deduplicate by child key
                                seen_keys = set()
                                scoped_chunks = []
                                for s in fallback_matching:
                                    sk = str(s.get("node_key") or (s.get("hierarchy_path") or [])[len(parent_path)]).strip()
                                    if sk not in seen_keys:
                                        seen_keys.add(sk)
                                        scoped_chunks.append(s)

                            # Build verified_hier_meta and child lines from ONLY scoped_chunks
                            child_lines = []
                            grouped_siblings: Dict[str, Dict[str, Any]] = {}
                            for chunk in scoped_chunks:
                                ck = str(chunk.get("node_key") or (chunk.get("hierarchy_path") or [])[len(parent_path)]).strip()
                                cv = str(chunk.get("node_value") or "").strip()
                                grouped_siblings[ck] = {
                                    "chunk": chunk,
                                    "value": cv if cv else None,
                                    "path": chunk.get("hierarchy_path_text") or " > ".join(chunk.get("hierarchy_path", []))
                                }
                                det_str = f" = {cv}" if cv else ""
                                child_lines.append(f"{ck}{det_str}")

                            summary_str = "\n".join(child_lines)
                            parent_name = parent_path[-1] if parent_path else "Section"
                            verified_hier_meta = {
                                "Field": parent_name,
                                "Value": summary_str,
                                "Entries": summary_str,
                                "Hierarchy Path": " > ".join(parent_path)
                            }

                            # Strictly scope hier_best_evidence to ONLY the authoritative sibling leaves
                            # so that generator context and final sources contain ONLY the relevant branches
                            hier_best_evidence = scoped_chunks
                            grouped = grouped_siblings
                            print(f"[RAGPipeline:Debug] ask: is_branch_enum=True, selected doc={best_doc}, parent_path={parent_path}, scoped_siblings={list(grouped.keys())}")

                    if not verified_hier_meta:
                        if top_val is not None and str(top_val).strip() != "":
                            verified_hier_meta = {
                                "Field": top_key or "Value",
                                "Value": str(top_val).strip(),
                                "Hierarchy Path": top_cand.get("hierarchy_path_text") or " > ".join(top_cand.get("hierarchy_path", []))
                            }
                        else:
                            # Top candidate is a branch with no node_value.
                            # For page-intent queries, expose source_page as a locator field
                            # so the generator can answer "which page is X on?" using the
                            # document's source_page metadata (clearly distinguished from a
                            # printed article page number which would be a पृष्ठ संख्या leaf).
                            page_intent_tokens = {
                                "पृष्ठ", "page", "पन्ना", "पेज", "prushth", "prishtha"
                            }
                            query_lower_tokens = set(re.findall(r"[\w\u0900-\u097F]+", query.lower()))
                            if query_lower_tokens.intersection(page_intent_tokens):
                                sp = top_cand.get("source_page")
                                if sp is not None:
                                    verified_hier_meta = {
                                        "Field": "दस्तावेज़ में स्थान (source_page)",
                                        "Value": str(sp),
                                        "Hierarchy Path": top_cand.get("hierarchy_path_text", "")
                                    }

                            # For branch nodes, aggregate direct children and sub-attributes present in evidence
                            top_path = top_cand.get("hierarchy_path") or []
                            if not verified_hier_meta and top_path:
                                grouped = {}
                                for c in hier_best_evidence:
                                    c_path = c.get("hierarchy_path") or c.get("metadata", {}).get("hierarchy_path") or []
                                    if len(c_path) > len(top_path) and c_path[:len(top_path)] == top_path:
                                        c_key = c_path[len(top_path)]
                                        if c_key not in grouped:
                                            grouped[c_key] = {
                                                "value": None,
                                                "attributes": {},
                                                "path": c.get("hierarchy_path_text") or " > ".join(c_path)
                                            }
                                        if len(c_path) == len(top_path) + 1:
                                            val = c.get("node_value")
                                            if val is not None and str(val).strip() != "":
                                                grouped[c_key]["value"] = str(val).strip()
                                        elif len(c_path) > len(top_path) + 1:
                                            sub_attr = " > ".join(c_path[len(top_path)+1:])
                                            sub_val = c.get("node_value")
                                            if sub_val is not None and str(sub_val).strip() != "":
                                                grouped[c_key]["attributes"][sub_attr] = str(sub_val).strip()
                                            elif sub_attr:
                                                grouped[c_key]["attributes"][sub_attr] = ""

                                if grouped:
                                    child_lines = []
                                    child_names = []
                                    for ck, cv in grouped.items():
                                        child_names.append(ck)
                                        details = []
                                        if cv.get("value"):
                                            details.append(f"= {cv['value']}")
                                        for ak, av in cv.get("attributes", {}).items():
                                            if av:
                                                details.append(f"{ak}: {av}")
                                            else:
                                                details.append(ak)
                                        det_str = f" ({', '.join(details)})" if details else ""
                                        child_lines.append(f"{ck}{det_str}")

                                    summary_str = "; ".join(child_lines)
                                    if len(grouped) == 1 and list(grouped.values())[0].get("value"):
                                        val_str = list(grouped.values())[0]["value"]
                                    elif len(child_names) <= 3 and not any(cv.get("value") for cv in grouped.values()):
                                        val_str = ", ".join(child_names)
                                    else:
                                        val_str = summary_str
                                    verified_hier_meta = {
                                        "Field": top_key or "Section",
                                        "Value": val_str,
                                        "Entries": summary_str,
                                        "Hierarchy Path": top_cand.get("hierarchy_path_text", "")
                                    }

                    # Generate Grounded Answer via Llama with hierarchical tree context
                    raw_answer = self.generator.generate_answer(query, hier_best_evidence, verified_metadata=verified_hier_meta)

                    refusal_phrases = [
                        "जानकारी नहीं मिली",
                        "प्रासंगिक जानकारी नहीं",
                        "not found",
                    ]

                    is_refusal = bool(raw_answer) and any(
                        phrase.lower() in raw_answer.lower()
                        for phrase in refusal_phrases
                    )

                    def _has_sibling(sibling_key: str, text: str) -> bool:
                        clean_k = re.sub(r'["\'“”‘’]', '', sibling_key).strip().lower()
                        clean_t = re.sub(r'["\'“”‘’]', '', text or '').strip().lower()
                        if clean_k in clean_t:
                            return True
                        k_tokens = [t for t in re.findall(r'[\w\u0900-\u097F]+', clean_k) if len(t) > 0]
                        return bool(k_tokens and all(kt in clean_t for kt in k_tokens))

                    all_siblings_covered = (
                        all(_has_sibling(sk, raw_answer) for sk in grouped.keys())
                        if (is_branch_enum and grouped) else True
                    )

                    if not all_siblings_covered or not raw_answer or is_refusal:
                        if verified_hier_meta and verified_hier_meta.get("Value"):
                            raw_answer = str(verified_hier_meta.get("Value")).strip()
                        elif top_val is not None and str(top_val).strip() != "":
                            raw_answer = str(top_val).strip()

                    if raw_answer and not is_branch_enum:
                        raw_answer = extract_factual_span(query, raw_answer, analysis)

                    formatted_evidence = []
                    for chunk in hier_best_evidence:
                        meta = chunk.get("metadata", {})
                        raw_page = chunk.get("source_page") or meta.get("source_page") or meta.get("page")
                        formatted_evidence.append({
                            "chunk_id": chunk.get("id"),
                            "document_id": chunk.get("document_id") or meta.get("document_id"),
                            "document_name": chunk.get("file_name") or meta.get("file_name") or meta.get("document_name"),
                            "document_type": chunk.get("document_type") or meta.get("document_type"),
                            "year": chunk.get("year") or meta.get("year"),
                            "text": chunk.get("hierarchy_path_text") or chunk.get("text", ""),
                            "page": raw_page if raw_page not in (0, "0", None) else None,
                            "section": "Hierarchical",
                            "score": chunk.get("rerank_score", chunk.get("score", 0.85)),
                            "chunk_type": "hierarchical",
                            "node_key": chunk.get("node_key"),
                            "node_value": chunk.get("node_value"),
                            "hierarchy_path": chunk.get("hierarchy_path"),
                            "hierarchy_path_text": chunk.get("hierarchy_path_text")
                        })

                    return {
                        "answer": raw_answer,
                        "evidence": formatted_evidence,
                        "query_analysis": {
                            "intent": analysis.get("intent", "hierarchical_search"),
                            "query_mode": "HIERARCHICAL",
                            "requested_mode": query_mode,
                            "detected_script": analysis.get("detected_script", "neutral"),
                            "target_title": analysis.get("target_title"),
                            "entities": analysis.get("entities", []),
                            "normalized_variants": analysis.get("normalized_variants", []),
                            "resolved_entities": analysis.get("resolved_entities", [])
                        }
                    }

        print("[RAGPipeline] Hierarchical search did not find high-confidence candidate; falling back to legacy branches.")

        # =====================================================
        # 2. DYNAMIC STRUCTURED INDEX KNOWLEDGE LAYER EVALUATION (FALLBACK)
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
                query_variants=analysis.get("normalized_variants"),
                expected_value_type=analysis.get("expected_value_type"),
                target_languages=analysis.get("target_languages"),
                condition_languages=analysis.get("condition_languages")
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
        verified_table_data = (table_result.get("details") if (analysis.get("table_intent") and "table_result" in locals() and table_result) else None)
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
        self.hierarchical_store.delete_document(document_id)

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
            # Keep single Devanagari characters (e.g., 'क', 'ख', 'ग' as क्षेत्र identifiers)
            # and single digits; only drop single Latin letters that are not meaningful.
            if (
                (len(token) > 1 or re.match(r'^[\u0900-\u097F]$', token) or token.isdigit())
                and token not in stopwords
            )
        }
        if not terms:
            # A query without a meaningful subject cannot be safely grounded.
            return False

        evidence_text = " ".join(chunk.get("text", "").lower() for chunk in evidence)
        has_term_support = any(term in evidence_text for term in terms)
        top_score = float(evidence[0].get("rerank_score") if evidence[0].get("rerank_score") is not None else evidence[0].get("score", 0.0))
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