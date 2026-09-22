"""Grounded answer generator powered by Ollama (Llama 3.2) with fallback to structured evidence summaries."""
import re
import logging
import requests
from typing import List, Dict, Any
from config import settings

logger = logging.getLogger("Generator")
logger.setLevel(logging.INFO)


def _repair_ocr_noise(text: str) -> str:
    """
    Repairs common OCR ligature errors in Hindi Devanagari text before sending
    to the LLM. This prevents broken characters from confusing author/title matching.
    """
    fixes = [
        # Generic Devanagari OCR ligature & character noise
        (r"पार[रि]स्स्थत[ति]क[ति]", "पारिस्थितिकी"),
        (r"पाररस्स्थततकी", "पारिस्थितिकी"),
        (r"पारिस्स्थितिकी", "पारिस्थितिकी"),
        (r"तुंत्र", "तंत्र"),
        (r"चुनौत[ति]या[ँां]+", "चुनौतियां"),
        (r"पहल\b", "पहल"),
    ]
    repaired = text
    for pattern, repl in fixes:
        repaired = re.sub(pattern, repl, repaired, flags=re.IGNORECASE)
    return repaired


def extract_factual_span(query: str, text: str, analysis: dict = None) -> str:
    """Generic data-driven extractor for factual spans from verified text paragraphs.
    Handles currency, dates, durations, entity counts, people/participants, and target sentences.
    Zero document-specific hardcoding.
    """
    if not text or not text.strip():
        return text

    text = text.strip()
    if len(text.split()) <= 4:
        return text

    q_lower = (query or "").lower()
    exp_type = ((analysis.get("expected_value_type") if analysis else None) or "").upper()

    # 1. DURATION (e.g. कितने वर्ष, how many years, kitne varsh)
    if exp_type == "DURATION" or re.search(r'(?:कितने\s*(?:वर्ष|साल|माह|महीने|दिन)|how\s+many\s+(?:years|months|days)|kitne\s+(?:varsh|saal))', q_lower):
        m = re.search(r'((?:प्रत्येक|हर|every|each)?\s*\d+\s*(?:वर्ष|साल|माह|महीने|दिन|years?|months?|days?))', text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # 2. MONETARY (e.g. कितने रुपये, cash prize, kitne rupaye)
    if exp_type == "MONETARY" or re.search(r'(?:रुपये|रुपए|rupees?|rupaye|नकद\s*पुरस्कार|cash\s*prize|prize\s*money)', q_lower):
        curr_m = re.search(r'([0-9,]+(?:/-(?:\s*रुपए|\s*रुपये)?)|\d+\s*(?:रुपए|रुपये|rupees?))', text, re.IGNORECASE)
        if curr_m:
            return curr_m.group(1).strip()
        m = re.search(r'([0-9,]+(?:\.\d+)?(?:/-)?\s*(?:रुपए|रुपये|rupees?|rs\.?|inr)?(?:\s*का\s*नगद\s*पुरस्कार)?)', text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # 3. DATE (e.g. कब, when, kab, किस तारीख/तिथि)
    if exp_type == "DATE" or re.search(r'(?:कब|when|kab|तारीख|तिथि|date)', q_lower):
        m = re.search(r'(?:दिनांक\s*)?(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})', text)
        if m:
            return m.group(1).strip()
        m_word = re.search(r'(\d{1,2}\s+(?:जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)[,\s]+\d{4})', text)
        if m_word:
            return m_word.group(1).strip()

    # 4. COUNT of specific entities (e.g. कितने कर्मचारियों, कितने प्रतिभागियों, how many employees)
    if exp_type == "COUNT" or re.search(r'(?:कितने|कितनी|कितनों|how\s+many|kitne)\s+([A-Za-z\u0900-\u097F]+)', q_lower):
        m = re.search(r'(?:इसमें\s*)?(\d+)\s*(?:कर्मचारियों|प्रतिभागियों|व्यक्तियों|लोगों|सदस्यों|अधिकारियों|पत्रों|फाइलों|persons?|employees?|participants?|members?|letters?)', text)
        if m:
            return m.group(1).strip()
        m_num = re.search(r'\b(\d+)\b', text)
        if m_num and not re.search(r'(?:वर्ष|साल|माह|दिनांक)', text[m_num.end():m_num.end()+10]):
            return m_num.group(1).strip()

    # 5. PERSON / PARTICIPATION / AWARD RECIPIENTS
    if exp_type == "PERSON" or re.search(r'(?:किन\s*लोगों|किसने|who|kin\s*logon|kisne|कौन\s*(?:था|थी|थे))', q_lower):
        if re.search(r'(?:पुरस्कार|prize|award|awardee|सम्मान)', q_lower):
            sentences = [s.strip() for s in re.split(r'[।\.\n]', text) if s.strip()]
            for s in sentences:
                if any(w in s for w in ['पुरस्कार', 'award', 'prize', 'सम्मान']):
                    return s + ("।" if not s.endswith("।") else "")
        if re.search(r'(?:भाग\s*लिया|प्रतिभागिता|participat|attended|bhaag\s*liya)', q_lower):
            sentences = [s.strip() for s in re.split(r'[।\.\n]', text) if s.strip()]
            for s in sentences:
                if any(w in s for w in ['प्रतिभागिता', 'भाग', 'participat']):
                    m_names = re.search(r'((?:श्री|सुश्री|डॉ\.|shri|ms\.|dr\.)\s+[^\.]+?(?:द्वारा\s*प्रतिभागिता|ने\s*भाग\s*लिया))', s, re.IGNORECASE)
                    if m_names:
                        return m_names.group(1).strip()
                    return s + ("।" if not s.endswith("।") else "")

    # 6. Default sentence selection by query token overlap
    sentences = [s.strip() for s in re.split(r'[।\.\n]', text) if s.strip()]
    if len(sentences) == 1:
        return text

    q_tokens = set(re.findall(r'[\w\u0900-\u097F]+', q_lower))
    best_sent = text
    best_overlap = -1
    for s in sentences:
        s_tokens = set(re.findall(r'[\w\u0900-\u097F]+', s.lower()))
        overlap = len(q_tokens.intersection(s_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sent = s + ("।" if not s.endswith("।") else "")

    return best_sent if best_overlap >= 3 else text


class Generator:
    def __init__(self):
        self.ollama_url = getattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.model = getattr(settings, "GENERATOR_MODEL_NAME", "llama3.2")

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], verified_metadata: Dict[str, Any] = None) -> str:
        if not context_chunks:
            return "दस्तावेज़ में इस प्रश्न से संबंधित कोई प्रासंगिक जानकारी नहीं मिली।"

        # Build clean structured context with OCR repair applied
        verified_block = ""
        if verified_metadata:
            verified_block = "\nVERIFIED STRUCTURED METADATA (authoritative):\n" + "\n".join(
                f"{key}: {value}" for key, value in verified_metadata.items() if value not in (None, "")
            ) + "\n"

        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk.get("metadata", {})
            file_name = chunk.get("file_name") or metadata.get("file_name") or metadata.get("filename") or metadata.get("document_id") or "Document"
            page_info = chunk.get("source_page") or chunk.get("page") or metadata.get("source_page") or metadata.get("page_number") or metadata.get("page", "Unknown")
            section = metadata.get("collection_type", metadata.get("section", "General"))
            h_path = chunk.get("hierarchy_path") or metadata.get("hierarchy_path")
            node_val = chunk.get("node_value") if chunk.get("node_value") is not None else metadata.get("node_value")

            if h_path and isinstance(h_path, list) and len(h_path) > 0:
                # Format hierarchical path as tree:
                # Source: {file_name}
                # Page: {source_page}
                # Hierarchy:
                # Parent
                #   → Child
                #     → Sub-key = Value
                tree_lines = []
                for depth, elem in enumerate(h_path):
                    indent = "  " * depth
                    arrow = "→ " if depth > 0 else ""
                    if depth == len(h_path) - 1 and node_val is not None:
                        tree_lines.append(f"{indent}{arrow}{elem} = {node_val}")
                    else:
                        tree_lines.append(f"{indent}{arrow}{elem}")

                # If this node is a branch (no node_val), explicitly list its child entries from context_chunks
                child_entries = []
                if node_val is None:
                    seen_child_keys = set()
                    for other_chunk in context_chunks:
                        other_path = other_chunk.get("hierarchy_path") or other_chunk.get("metadata", {}).get("hierarchy_path") or []
                        if len(other_path) > len(h_path) and other_path[:len(h_path)] == h_path:
                            child_k = other_path[len(h_path)]
                            if child_k not in seen_child_keys:
                                seen_child_keys.add(child_k)
                                other_val = other_chunk.get("node_value")
                                val_suffix = f" = {other_val}" if other_val is not None and str(other_val).strip() != "" else ""
                                orig_p_str = other_chunk.get("hierarchy_path_text") or " > ".join(other_path)
                                child_entries.append(f"  - Child: key = {child_k}{val_suffix} (Path: {orig_p_str})")

                h_tree_str = "\n".join(tree_lines)
                if child_entries:
                    h_tree_str += "\n\nHierarchical Children / Entries in this section:\n" + "\n".join(child_entries)

                context_text += (
                    f"\n--- SOURCE EVIDENCE CHUNK {i} ---\n"
                    f"Source: {file_name}\n"
                    f"Page: {page_info}\n\n"
                    f"Hierarchy:\n{h_tree_str}\n"
                )
            else:
                raw_text = chunk.get("text", "").strip()
                cleaned_text = _repair_ocr_noise(raw_text)
                context_text += f"\n--- SOURCE EVIDENCE CHUNK {i} (Page {page_info} | Section: {section}) ---\n{cleaned_text}\n"

        system_prompt = (
            "You are a precise, bilingual (Hindi + English) document Q&A assistant.\n"
            "Answer the user's question using ONLY the provided document evidence.\n\n"
            "EVIDENCE STRUCTURE:\n"
            "The document evidence is represented as hierarchical knowledge trees:\n"
            "Parent\n"
            "  → Child\n"
            "    → Sub-key = Value\n"
            "Branch sections also explicitly list their direct child entries and sub-attributes.\n\n"
            "RULES:\n"
            "1. If a VERIFIED STRUCTURED METADATA block is present above the evidence, "
            "it contains the authoritative answer extracted directly from the document. "
            "Use that value as your answer without modification.\n"
            "2. Otherwise answer the question directly and factually using the COMPLETE "
            "evidence set (all chunks), not just evidence[0]. If a key has an assigned value "
            "(e.g. Sub-key = Value), that value is the authoritative factual answer.\n"
            "3. An explicit numeric value, including 0 (शून्य), is a valid recorded answer; "
            "never treat 0 as missing or absent.\n"
            "4. If the requested information is genuinely NOT present anywhere in the "
            "provided evidence, answer ONLY:\n"
            "   'दस्तावेज़ में इस प्रश्न से संबंधित जानकारी नहीं मिली।'\n"
            "5. Keep the answer concise, accurate, and in the language of the question.\n"
            "6. When the question asks what items, articles, entries, or topics are "
            "present in a section/category, or asks for description/summary (e.g. 'what is in X', "
            "'which articles are under Y', 'कौन सा लेख', 'कौन-कौन से', 'सूची', 'विवरण', 'ब्यौरा'), "
            "enumerate the relevant child node keys and entries from the evidence hierarchy. "
            "Do not say 'not found' if the section and its children are present in the evidence."
        )

        user_prompt = f"""{verified_block}
DOCUMENT EVIDENCE:
{context_text}

USER QUESTION:
{query}

CONCISE FACTUAL ANSWER:"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }

        try:
            res = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=(2.0, 5.0)
            )
            if res.status_code == 200:
                result = res.json()
                return result.get("message", {}).get("content", "").strip()
            else:
                logger.error(f"[Ollama Error] Status {res.status_code}: {res.text}")
                return f"दस्तावेज़ से उत्तर तैयार नहीं हो सका (Ollama Error {res.status_code})।"
        except requests.exceptions.RequestException as e:
            logger.warning(f"[Ollama Connection Notice] {e}. Falling back to structured evidence extraction.")
            # Fallback to direct evidence extraction if LLM is in standby/offline.
            # Check both 'Value' (capital V, as set by pipeline.py) and 'value' (lowercase).
            if verified_metadata:
                val = verified_metadata.get("Value") or verified_metadata.get("value")
                if val:
                    span_ans = extract_factual_span(query, str(val).strip(), None)
                    art = verified_metadata.get("article", "")
                    return span_ans + (f" (लेख: '{art}')" if art else "")
                entries = verified_metadata.get("Entries") or verified_metadata.get("Child Entries / Items")
                if entries:
                    return str(entries).strip()

            # If top context chunk is a hierarchical leaf with a concrete value, extract it cleanly
            if context_chunks:
                top = context_chunks[0]
                node_val = top.get("node_value")
                node_key = top.get("node_key")
                if node_val is not None and str(node_val).strip() != "":
                    # Return concise direct factual answer
                    return extract_factual_span(query, str(node_val).strip(), None)

                # If top candidate is a branch, check if its children exist in context_chunks
                top_p = top.get("hierarchy_path") or []
                if top_p:
                    child_keys = []
                    for c in context_chunks[1:]:
                        c_p = c.get("hierarchy_path") or []
                        if len(c_p) > len(top_p) and c_p[:len(top_p)] == top_p:
                            ck = c_p[len(top_p)]
                            if ck not in child_keys:
                                child_keys.append(ck)
                    if child_keys:
                        return ", ".join(child_keys)

                top_chunks = context_chunks[:3]
                summaries = []
                for c in top_chunks:
                    raw_t = c.get("text", "").strip()
                    meta = c.get("metadata", {})
                    pg = c.get("page") or meta.get("page_number") or meta.get("page")
                    pg_str = f" (पृष्ठ {pg})" if pg and pg not in (0, "0", "Unknown") else ""
                    if raw_t:
                        cleaned = _repair_ocr_noise(raw_t)
                        lines = [l.strip() for l in cleaned.split("\n") if len(l.strip()) > 10 and not l.strip().startswith("[TABLE OF CONTENTS")]
                        if lines:
                            snippet = " ".join(lines[:3])
                            summaries.append(f"• {snippet}{pg_str}")
                if summaries:
                    return "\n\n".join(summaries)

            return "दस्तावेज़ से उत्तर तैयार नहीं हो सका क्योंकि Ollama उपलब्ध नहीं है।"


# Alias for backward compatibility with pipeline.py
OllamaGroundedGenerator = Generator
