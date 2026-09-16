import json
import os
import re
import threading
import unicodedata
from typing import Dict, List, Optional, Set, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TerminologyMemory")

class DynamicTerminologyMemory:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DynamicTerminologyMemory, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, storage_path: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return
        
        if storage_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            storage_path = os.path.join(base_dir, "data", "terminology_store.json")
            
        self.storage_path = storage_path
        self._memory_lock = threading.RLock()
        
        # Structure: { canonical_key: { "canonical": str, "english": set(), "hindi": set(), "roman": set(), "ocr_variants": set(), "occurrences": int, "sources": set() } }
        self.terms: Dict[str, Dict[str, Any]] = {}
        self.alias_to_canonical: Dict[str, str] = {}
        
        self._ensure_storage_dir()
        self.load_from_disk()
        self._initialized = True

    def _ensure_storage_dir(self):
        directory = os.path.dirname(self.storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _normalize_key(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        return re.sub(r"[\s\-_]+", " ", text.strip().lower())

    def load_from_disk(self):
        with self._memory_lock:
            if not os.path.exists(self.storage_path):
                logger.info(f"[TerminologyMemory] No existing store at {self.storage_path}. Initialized with empty memory.")
                return
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key, entry in data.items():
                        self.terms[key] = {
                            "canonical": entry.get("canonical", key),
                            "category": entry.get("category", ""),
                            "english": set(entry.get("english", [])),
                            "hindi": set(entry.get("hindi", [])),
                            "roman": set(entry.get("roman", [])),
                            "ocr_variants": set(entry.get("ocr_variants", [])),
                            "occurrences": entry.get("occurrences", 1),
                            "sources": set(entry.get("sources", [])),
                            "document_ids": set(entry.get("document_ids", [])),
                            "source_chunk_ids": set(entry.get("source_chunk_ids", [])),
                            "contexts": list(entry.get("contexts", []))[:5]
                        }
                    self._rebuild_alias_index()
                logger.info(f"[TerminologyMemory] Loaded {len(self.terms)} canonical terms from {self.storage_path}")
            except Exception as e:
                logger.error(f"[TerminologyMemory] Failed to load store from {self.storage_path}: {e}")

    def save_to_disk(self):
        with self._memory_lock:
            try:
                serializable = {}
                for key, entry in self.terms.items():
                    serializable[key] = {
                        "canonical": entry["canonical"],
                        "category": entry.get("category", ""),
                        "english": sorted(list(entry["english"])),
                        "hindi": sorted(list(entry["hindi"])),
                        "roman": sorted(list(entry["roman"])),
                        "ocr_variants": sorted(list(entry["ocr_variants"])),
                        "occurrences": entry["occurrences"],
                        "sources": sorted(list(entry.get("sources", []))),
                        "document_ids": sorted(list(entry.get("document_ids", []))),
                        "source_chunk_ids": sorted(list(entry.get("source_chunk_ids", []))),
                        "contexts": entry.get("contexts", [])[:5]
                    }
                with open(self.storage_path, "w", encoding="utf-8") as f:
                    json.dump(serializable, f, ensure_ascii=False, indent=2)
                logger.info(f"[TerminologyMemory] Persisted {len(self.terms)} terms to {self.storage_path}")
            except Exception as e:
                logger.error(f"[TerminologyMemory] Error saving terminology memory: {e}")

    def _rebuild_alias_index(self):
        self.alias_to_canonical.clear()
        for key, entry in self.terms.items():
            all_forms = (
                {entry["canonical"]}
                | entry["english"]
                | entry["hindi"]
                | entry["roman"]
                | entry["ocr_variants"]
            )
            for form in all_forms:
                norm_form = self._normalize_key(form)
                if norm_form:
                    self.alias_to_canonical[norm_form] = key

    # Reject common relational words to prevent factual statement memorization
    FORBIDDEN_RELATIONAL_PATTERNS = re.compile(
        r"\b(by|written|author|wrote|published|page|editor|created|composed|द्वारा|लिखित|रचित|लेखक|पृष्ठ|लेखिका|कवि|कवयित्री)\b",
        re.IGNORECASE
    )

    def _is_valid_term(self, term: str) -> bool:
        if not term or not isinstance(term, str):
            return False
        cleaned = term.strip()
        words = cleaned.split()
        
        # Word limit 4 -> 10 ki aur char length 50 -> 100 ki
        if len(words) == 0 or len(words) > 10 or len(cleaned) > 100:
            return False

        # Colon (:) hata diya taaki subtitled titles reject na hon
        sentence_markers = [".", "?", "!", ";", "\n"]
        if any(marker in cleaned for marker in sentence_markers):
            return False

        if self.FORBIDDEN_RELATIONAL_PATTERNS.search(cleaned):
            return False

        return True

    def register_term(
        self,
        canonical: str,
        english_aliases: Optional[List[str]] = None,
        hindi_aliases: Optional[List[str]] = None,
        roman_aliases: Optional[List[str]] = None,
        ocr_variants: Optional[List[str]] = None,
        source: Optional[str] = None,
        document_id: Optional[str] = None,
        source_chunk_id: Optional[str] = None,
        context: Optional[str] = None,
        category: Optional[str] = None,
    ) -> bool:
        """
        Registers or enriches a conceptual terminology record.
        Strictly rejects sentences, relational claims, author bindings, or document-specific facts.
        """
        if not self._is_valid_term(canonical):
            return False

        canonical_key = self._normalize_key(canonical)
        # A concept may have different aliases in different documents. Its storage
        # key is therefore document-scoped; the displayed canonical remains clean.
        key = f"{document_id}::{canonical_key}" if document_id else canonical_key
        if not key:
            return False

        with self._memory_lock:
            if key not in self.terms:
                self.terms[key] = {
                    "canonical": canonical.strip(),
                    "category": (category or "").lower().strip(),
                    "english": set(),
                    "hindi": set(),
                    "roman": set(),
                    "ocr_variants": set(),
                    "occurrences": 0,
                    "sources": set(), "document_ids": set(), "source_chunk_ids": set(), "contexts": []
                }
            elif category:
                self.terms[key]["category"] = category.lower().strip()

            record = self.terms[key]
            record.setdefault("document_ids", set())
            record.setdefault("source_chunk_ids", set())
            record.setdefault("contexts", [])
            record["occurrences"] += 1
            if source:
                record["sources"].add(source)
            if document_id:
                record["document_ids"].add(document_id)
            if source_chunk_id:
                record["source_chunk_ids"].add(source_chunk_id)
            if context and context not in record["contexts"]:
                record["contexts"].append(context[:500])

            def filter_and_add(target_set: Set[str], candidates: Optional[List[str]]):
                if not candidates:
                    return
                for item in candidates:
                    if self._is_valid_term(item):
                        target_set.add(item.strip())

            filter_and_add(record["english"], english_aliases)
            filter_and_add(record["hindi"], hindi_aliases)
            filter_and_add(record["roman"], roman_aliases)
            filter_and_add(record["ocr_variants"], ocr_variants)

            self._rebuild_alias_index()
            self.save_to_disk()
            
            alias_count = len(record["english"] | record["hindi"] | record["roman"] | record["ocr_variants"])
            logger.info(f"[TerminologyMemory] Registered/Updated Canonical: '{record['canonical']}' | Total Aliases: {alias_count}")
            return True

    def resolve_term(self, term: str, document_id: Optional[str] = None) -> Optional[str]:
        """Returns the canonical term key if found in memory."""
        norm = self._normalize_key(term)
        if document_id:
            for key, record in self.terms.items():
                if document_id not in record.get("document_ids", set()):
                    continue
                forms = {record.get("canonical", "")} | record.get("english", set()) | record.get("hindi", set()) | record.get("roman", set()) | record.get("ocr_variants", set())
                if any(self._normalize_key(form) == norm for form in forms):
                    return key
            return None
        return self.alias_to_canonical.get(norm)

    def get_canonical_record(self, term: str) -> Optional[Dict[str, Any]]:
        """Returns the full dictionary record for a given term or alias."""
        canonical_key = self.resolve_term(term)
        if canonical_key and canonical_key in self.terms:
            rec = self.terms[canonical_key]
            return {
                "canonical": rec["canonical"],
                "english": list(rec["english"]),
                "hindi": list(rec["hindi"]),
                "roman": list(rec["roman"]),
                "ocr_variants": list(rec["ocr_variants"]),
                "occurrences": rec["occurrences"],
                "sources": list(rec.get("sources", []))
            }
        return None

    def get_all_aliases(self, term: str, document_id: Optional[str] = None) -> Set[str]:
        """Returns all recognized variations across languages and OCR variants."""
        canonical_key = self.resolve_term(term, document_id=document_id)
        if not canonical_key or canonical_key not in self.terms:
            return {term}
        record = self.terms[canonical_key]
        return (
            {record["canonical"]}
            | record["english"]
            | record["hindi"]
            | record["roman"]
            | record["ocr_variants"]
        )

    def get_all_terms(self) -> Dict[str, Any]:
        """Returns all registered canonical terms and their metadata."""
        with self._memory_lock:
            result = {}
            for k, rec in self.terms.items():
                result[k] = {
                    "canonical": rec["canonical"],
                    "english": sorted(list(rec["english"])),
                    "hindi": sorted(list(rec["hindi"])),
                    "roman": sorted(list(rec["roman"])),
                    "ocr_variants": sorted(list(rec["ocr_variants"])),
                    "occurrences": rec["occurrences"],
                    "sources": sorted(list(rec.get("sources", [])))
                }
            return result

    def get_terms_by_category(self, category: str) -> Set[str]:
        """Returns the set of canonical keys belonging to a specific category (e.g. 'language', 'role')."""
        with self._memory_lock:
            cat_norm = (category or "").lower().strip()
            return {
                key for key, entry in self.terms.items()
                if entry.get("category", "").lower().strip() == cat_norm
            }

    def detect_concepts(self, text: str) -> Dict[str, Set[str]]:
        """
        Dynamically detects canonical concepts present in text across all aliases,
        returning a map of {category: {canonical_keys}} and all {canonical_keys}.
        """
        if not text:
            return {"all": set()}
        lower = text.lower()
        found: Set[str] = set()
        with self._memory_lock:
            for alias, canonical_key in self.alias_to_canonical.items():
                if len(alias) < 2:
                    continue
                pat = r'(?<![\w\u0900-\u097F])' + re.escape(alias) + r'(?![\w\u0900-\u097F])'
                if re.search(pat, lower):
                    found.add(canonical_key)

            res: Dict[str, Set[str]] = {"all": found}
            for ckey in found:
                rec = self.terms.get(ckey, {})
                cat = rec.get("category", "general")
                res.setdefault(cat, set()).add(ckey)
            return res

    def delete_document(self, document_id: str) -> None:
        """Remove document associations without deleting shared concepts."""
        with self._memory_lock:
            for record in self.terms.values():
                record.setdefault("document_ids", set()).discard(document_id)
            self._rebuild_alias_index()
            self.save_to_disk()

    def _is_valid_term(self, term: str) -> bool:
        if not term or not isinstance(term, str):
            return False
        cleaned = term.strip()
        words = cleaned.split()
        
        # Word limit 4 -> 10 ki aur char length 50 -> 100 ki
        if len(words) == 0 or len(words) > 10 or len(cleaned) > 100:
            return False

        # Colon (:) hata diya taaki subtitled titles reject na hon
        sentence_markers = [".", "?", "!", ";", "\n"]
        if any(marker in cleaned for marker in sentence_markers):
            return False

        if self.FORBIDDEN_RELATIONAL_PATTERNS.search(cleaned):
            return False

        return True        
