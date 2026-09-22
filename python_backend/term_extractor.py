"""Extracts technical, legal, and administrative bilingual terminology from ingested documents."""
import re
import json
import logging
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error

from config import settings
from terminology_memory import DynamicTerminologyMemory

logger = logging.getLogger("TermExtractor")

class TerminologyExtractor:
    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model_name: Optional[str] = None,
        storage_path: Optional[str] = None
    ):
        self.ollama_url = (ollama_url or getattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.model_name = model_name or getattr(settings, "GENERATOR_MODEL_NAME", "llama3.2")
        self.memory = DynamicTerminologyMemory(storage_path=storage_path)

    def _is_devanagari(self, text: str) -> bool:
        return bool(re.search(r"[\u0900-\u097F]", text))

    def extract_and_register_from_text(self, document_text: str, document_id: str = "", source_filename: str = "", chunks: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Scans document text, dynamically identifies high-value concepts/terminologies,
        and records them in the persistent memory without memorizing document-specific statements.
        """
        if not document_text or len(document_text.strip()) < 50:
            return 0

        # Sample high-density text portions (first 4500 characters)
        sample = document_text[:4500]

        prompt = (
            "You are a terminology extraction system for a multilingual knowledge base.\n"
            "Identify key technical, scientific, domain, organizational, or governance concepts and entities in the following text.\n"
            "For each concept, provide:\n"
            "1. 'canonical': Canonical English or standard name\n"
            "2. 'english': Common English aliases, abbreviations, or acronyms (e.g., ['AI', 'Artificial Intelligence'])\n"
            "3. 'hindi': Hindi (Devanagari script) translations or phonetic equivalents (e.g., ['आर्टिफिशियल इंटेलिजेंस', 'कृत्रिम बुद्धिमत्ता'])\n"
            "4. 'roman': Romanized Hindi or Hinglish spelling forms (e.g., ['artifishal intelijens', 'ai'])\n"
            "5. 'ocr_variants': Possible spelling or OCR misreadings (e.g., ['Artifical Intelligence'])\n\n"
            "STRICT RULES:\n"
            "- DO NOT extract facts, relational claims, author names, or document sentences (e.g. do NOT extract 'Author wrote X').\n"
            "- Extract ONLY reusable domain terminology, entities, and concepts.\n"
            "- Return ONLY a valid JSON array of objects.\n\n"
            "Example Format:\n"
            '[\n'
            '  {"canonical": "Artificial Intelligence", "english": ["AI"], "hindi": ["आर्टिफिशियल इंटेलिजेंस", "कृत्रिम बुद्धिमत्ता"], "roman": ["ai"], "ocr_variants": ["Artifical Intelligence"]},\n'
            '  {"canonical": "Semiconductor", "english": ["semiconductor"], "hindi": ["सेमीकंडक्टर"], "roman": ["semiconductor"], "ocr_variants": ["semicondutor"]}\n'
            ']\n\n'
            f"Text:\n{sample}\n\nJSON Output:"
        )

        extracted_count = 0
        try:
            req_data = json.dumps({
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1
                }
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=35) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw_response = result.get("response", "[]")

            terms_data = json.loads(raw_response)
            if isinstance(terms_data, dict):
                terms_data = terms_data.get("terms", terms_data.get("concepts", terms_data.get("entities", [])))

            if isinstance(terms_data, list):
                for item in terms_data:
                    canonical = item.get("canonical")
                    if canonical and isinstance(canonical, str):
                        registered = self.memory.register_term(
                            canonical=canonical,
                            english_aliases=item.get("english", []),
                            hindi_aliases=item.get("hindi", []),
                            roman_aliases=item.get("roman", []),
                            ocr_variants=item.get("ocr_variants", []),
                            source=source_filename or "document_extraction",
                            document_id=document_id or None,
                            source_chunk_id=self._source_chunk_id(canonical, chunks),
                            context=self._source_context(canonical, chunks)
                        )
                        if registered:
                            extracted_count += 1
                            
            logger.info(f"[TermExtractor] Extracted and registered {extracted_count} terminology concepts from '{source_filename}' via LLM.")
        except Exception as e:
            logger.warning(f"[TermExtractor] LLM terminology extraction fallback to heuristics: {e}")
            extracted_count = self._heuristic_extraction(sample, document_id=document_id, source_filename=source_filename, chunks=chunks)
        return extracted_count

    def _source_chunk_id(self, term: str, chunks: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        if not chunks:
            return None
        lowered = term.lower()
        for chunk in chunks:
            if lowered in chunk.get("text", "").lower():
                return chunk.get("id")
        return None

    def _source_context(self, term: str, chunks: Optional[List[Dict[str, Any]]]) -> Optional[str]:
        if not chunks:
            return None
        lowered = term.lower()
        for chunk in chunks:
            text = chunk.get("text", "")
            index = text.lower().find(lowered)
            if index >= 0:
                return text[max(0, index - 160):index + len(term) + 200]
        return None

    def _heuristic_extraction(self, text: str, document_id: str = "", source_filename: str = "", chunks: Optional[List[Dict[str, Any]]] = None) -> int:
        """Fallback rule-based extractor when LLM is unavailable."""
        count = 0
        # 1. Detect uppercase acronyms (2 to 6 characters, e.g., AI, ISRO, GPU, CPU, VLSI)
        acronym_pattern = re.compile(r"\b([A-Z]{2,6})\b")
        for acr in set(acronym_pattern.findall(text)):
            if acr not in {"THE", "AND", "FOR", "PDF", "PAGE", "VOL", "ISS"}:
                self.memory.register_term(
                    canonical=acr,
                    english_aliases=[acr],
                    roman_aliases=[acr.lower()],
                    source=source_filename or "heuristic_acronym", document_id=document_id or None,
                    source_chunk_id=self._source_chunk_id(acr, chunks), context=self._source_context(acr, chunks)
                )
                count += 1

        # 2. Detect capitalized multi-word technical concepts (e.g., Artificial Intelligence, Machine Learning)
        concept_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")
        for term in set(concept_pattern.findall(text)):
            # Filter out common stop-word combinations
            if not any(sw in term.lower().split() for sw in ["table", "contents", "index", "page", "editor", "editorial", "edition", "department"]):
                self.memory.register_term(
                    canonical=term,
                    english_aliases=[term],
                    roman_aliases=[term.lower()],
                    source=source_filename or "heuristic_concept", document_id=document_id or None,
                    source_chunk_id=self._source_chunk_id(term, chunks), context=self._source_context(term, chunks)
                )
                count += 1

        logger.info(f"[TermExtractor] Heuristics registered {count} candidate concepts from '{source_filename}'.")
        return count
