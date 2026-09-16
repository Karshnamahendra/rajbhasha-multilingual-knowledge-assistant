import re
import json
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
            page_info = chunk.get("page") or metadata.get("page_number") or metadata.get("page", "Unknown")
            section = metadata.get("collection_type", metadata.get("section", "General"))
            raw_text = chunk.get("text", "").strip()
            cleaned_text = _repair_ocr_noise(raw_text)
            context_text += f"\n--- SOURCE EVIDENCE CHUNK {i} (Page {page_info} | Section: {section}) ---\n{cleaned_text}\n"

        system_prompt = (

    "You are a precise, bilingual (Hindi + English) document Q&A assistant.\n\n"

    "YOUR ONLY JOB: Answer the user's question using ONLY the evidence provided below. No outside knowledge.\n\n"

    "DOCUMENT STRUCTURE AWARENESS:\n"

    "- Many documents contain a Table of Contents (TOC / विषय-सूची) where each line follows this pattern:\n"

    "    ArticleTitle ........ AuthorName ........ PageNumber\n"

    "  (dots '....' are called dot leaders — they are separators, not content)\n"

    "- In such lines, the AUTHOR is the person who WROTE the article, NOT the topic.\n\n"

    "STRICT EXTRACTION RULES:\n"

    "1. DISTINGUISH ROLES FROM AUTHORS: Do NOT confuse patrons, advisors, or editorial board members\n"

    "   (e.g., 'संरक्षक', 'सलाहकार', 'संपादक मंडल', 'Patron', 'Editor') with actual article authors.\n"

    "   These are organizational roles, not article authors.\n"

    "2. Use VERIFIED STRUCTURED METADATA when supplied; it is the only authority for metadata fields.\n"

    "3. Never infer an author, page, category, or relationship from weakly related chunks.\n"

    "4. REVERSE LOOKUP: When asked 'X ka article kaunsa hai?',\n"

    "   locate that author name in the TOC and return the exact article title paired with them.\n"

    "5. Answer in the SAME language/style as the question whenever possible.\n"

    "6. If the answer is NOT in verified metadata or evidence, say ONLY:\n"

    "   'दस्तावेज़ में इस प्रश्न से संबंधित जानकारी नहीं मिली।'\n"

    "7. Never hallucinate names, titles, or facts not present in the context.\n"

    "8. Keep your answer concise and direct. No preamble.\n\n"

    "STRUCTURED TABLE / FORM EVIDENCE:\n"

    "- Structured evidence (tables, form fields, reporting metadata) is authoritative over unrelated text chunks.\n"

    "- An explicit numeric value, including 0, is a valid recorded answer; never call it missing.\n"

    "- A blank field is not zero: state that no value was recorded. Preserve 'लागू नहीं' / 'not applicable' as recorded.\n"

    "- If asked what a form requests, return its labels/questions. If asked what was entered, return only the recorded value.\n"

    "- Never invent values or row/column relationships. Calculations must use verified structured values only.\n\n"


    "CONTENT-BASED QUESTIONS:\n"

    "- If the user asks a question about the actual content of an article, topic, paragraph, "
    "statement, fact, explanation, reason, process, conclusion, or information discussed in the document, "
    "answer using the relevant CONTENT/ARTICLE evidence provided below.\n"

    "- Do NOT answer content-based questions using only the TOC, index, author, page number, "
    "category, or other metadata.\n"

    "- For content-based questions, identify and use the content chunks that directly contain "
    "the information being asked about.\n"

    "- If multiple relevant content chunks are provided, use them together when they clearly refer "
    "to the same topic or article.\n"

    "- Do NOT assume information merely because it is related to the article title or author.\n"

    "- Do NOT generate an explanation from outside knowledge. The explanation must be supported "
    "by the retrieved document content.\n"

    "- If the requested information is not present in the provided content evidence, "
    "do not guess. Follow Rule 6 and return:\n"

    "  'दस्तावेज़ में इस प्रश्न से संबंधित जानकारी नहीं मिली।'\n\n"


    "INDEX / METADATA QUESTIONS:\n"

    "- Questions asking for an author, article title, page number, category, or article-author relationship "
    "should continue to follow the existing INDEX/TOC and VERIFIED STRUCTURED METADATA rules above.\n"

    "- Do NOT replace the existing index-based behavior with content-based reasoning when the user is asking "
    "for metadata.\n\n"


    "IMPORTANT:\n"

    "- INDEX/METADATA question → use verified metadata / TOC evidence.\n"

    "- CONTENT question → use actual article/content evidence.\n"

    "- Do not force a content question to be answered from the index.\n"

    "- Do not force an index question to be answered from unrelated article content.\n"

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
                "temperature": 0.05,
                "top_p": 0.85,
                "top_k": 20,
                "repeat_penalty": 1.1
            }
        }

        try:
            res = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=(3.0, 30.0)
            )
            if res.status_code == 200:
                result = res.json()
                return result.get("message", {}).get("content", "").strip()
            else:
                logger.error(f"[Ollama Error] Status {res.status_code}: {res.text}")
                return f"दस्तावेज़ से उत्तर तैयार नहीं हो सका (Ollama Error {res.status_code})।"
        except requests.exceptions.RequestException as e:
            logger.warning(f"[Ollama Connection Notice] {e}. Falling back to structured evidence extraction.")
            # Fallback to direct evidence extraction if LLM is in standby/offline
            if verified_metadata and verified_metadata.get("value"):
                field = verified_metadata.get("requested_field", "जानकारी")
                val = verified_metadata.get("value")
                art = verified_metadata.get("article", "")
                return f"{field.capitalize()}: {val}" + (f" (लेख: '{art}')" if art else "")

            if context_chunks:
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
