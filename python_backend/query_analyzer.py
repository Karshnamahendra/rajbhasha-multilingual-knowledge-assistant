"""Analyzes user queries for intent, script detection (Devanagari/Roman/English), transliteration, and routing."""
import re
import unicodedata
import logging
from typing import Dict, Any, List, Set, Optional, Tuple
import numpy as np

from terminology_memory import DynamicTerminologyMemory
from structured_store import FieldNormalizer
from index_knowledge_layer import romanize_generic

logger = logging.getLogger("QueryAnalyzer")
logger.setLevel(logging.INFO)


class QueryNormalizer:
    """
    Multilingual script detector, entity normalizer, and multi-query variant generator
    leveraging the DynamicTerminologyMemory.
    Supports English, Hindi, Roman-Hindi, Hinglish, Mixed-script, and OCR noise.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.memory = DynamicTerminologyMemory(storage_path=storage_path)
        self.devanagari_pattern = re.compile(r"[\u0900-\u097F]")
        self.latin_pattern = re.compile(r"[a-zA-Z]")

        # Universal ISO/ITRANS phonetic mapping for Hindi transliteration
        self.consonants = {
            'ndr': 'न्द्र', 'ksh': 'क्ष', 'chh': 'छ', 'shh': 'ष',
            'kh': 'ख', 'gh': 'घ', 'ch': 'च', 'jh': 'झ',
            'th': 'थ', 'dh': 'ध', 'ph': 'फ', 'bh': 'भ', 'sh': 'श',
            'tr': 'त्र', 'gy': 'ज्ञ', 'gn': 'ज्ञ', 'dr': 'द्र',
            'pr': 'प्र', 'kr': 'क्र', 'gr': 'ग्र', 'br': 'ब्र',
            'k': 'क', 'g': 'ग', 'c': 'क', 'j': 'ज', 't': 'त',
            'd': 'द', 'n': 'न', 'p': 'प', 'f': 'फ़', 'b': 'ब',
            'm': 'म', 'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व',
            'w': 'व', 's': 'स', 'h': 'ह', 'z': 'ज़', 'q': 'क', 'x': 'क्स'
        }
        self.vowels_matra = {
            'aa': 'ा', 'ee': 'ी', 'oo': 'ू', 'ai': 'ै', 'au': 'ौ',
            'a': '', 'e': 'े', 'i': 'ि', 'o': 'ो', 'u': 'ु'
        }
        self.vowels_indep = {
            'aa': 'आ', 'ee': 'ई', 'oo': 'ऊ', 'ai': 'ऐ', 'au': 'औ',
            'a': 'अ', 'e': 'ए', 'i': 'इ', 'o': 'ओ', 'u': 'उ'
        }

    def detect_script(self, text: str) -> str:
        if not text:
            return "neutral"
        has_deva = bool(self.devanagari_pattern.search(text))
        has_latin = bool(self.latin_pattern.search(text))

        if has_deva and has_latin:
            return "mixed_hinglish"
        elif has_deva:
            return "devanagari_hindi"
        elif has_latin:
            return "latin_english_or_hinglish"
        return "neutral"

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        cleaned = re.sub(r"[\'\"`~@#$%^*()_+=\[\]{}|\\<>/]", " ", text)
        return re.sub(r"\s+", " ", cleaned).strip()

    def algorithmic_transliterate(self, text: str, to_script: Optional[str] = None, *args, **kwargs) -> str:
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        out_words = []
        for word in words:
            res = ""
            i = 0
            is_start = True
            while i < len(word):
                matched = False
                for clen in (3, 2, 1):
                    sub = word[i:i+clen]
                    if sub in self.consonants:
                        res += self.consonants[sub]
                        i += clen
                        matched = True
                        is_start = False
                        break
                if matched:
                    for vlen in (2, 1):
                        vsub = word[i:i+vlen]
                        if vsub in self.vowels_matra:
                            res += self.vowels_matra[vsub]
                            i += vlen
                            break
                    continue

                for vlen in (2, 1):
                    vsub = word[i:i+vlen]
                    if is_start and vsub in self.vowels_indep:
                        res += self.vowels_indep[vsub]
                        i += vlen
                        matched = True
                        is_start = False
                        break
                    elif vsub in self.vowels_matra:
                        res += self.vowels_matra[vsub]
                        i += vlen
                        matched = True
                        break

                if not matched:
                    i += 1
            if res:
                out_words.append(res)
        return " ".join(out_words)

    def extract_matched_spans(self, text: str, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        cleaned = self.clean_text(text)
        words = cleaned.split()
        matched_spans = []
        n = len(words)
        covered_indices = set()

        for i in range(n):
            if i in covered_indices:
                continue
            for length in range(min(4, n - i), 0, -1):
                phrase = " ".join(words[i:i+length])
                canonical_key = self.memory.resolve_term(phrase, document_id=document_id)
                if canonical_key and canonical_key in self.memory.terms:
                    record = self.memory.terms[canonical_key]
                    matched_spans.append({
                        "start": i,
                        "end": i + length,
                        "phrase": phrase,
                        "canonical": record["canonical"],
                        "record": record
                    })
                    for idx in range(i, i + length):
                        covered_indices.add(idx)
                    break

        return matched_spans

    def generate_normalized_variants(
        self,
        original_query: str, document_id: Optional[str] = None
    ) -> Tuple[List[str], Dict[str, Any]]:
        cleaned = self.clean_text(original_query)
        words = cleaned.split()
        script = self.detect_script(original_query)
        spans = self.extract_matched_spans(original_query, document_id=document_id)

        variants: Set[str] = set()
        variants.add(original_query.strip())
        if cleaned and cleaned != original_query.strip():
            variants.add(cleaned)

        resolved_traces = []

        if spans:
            canon_tokens = list(words)
            for sp in reversed(spans):
                canon_tokens[sp["start"]:sp["end"]] = [sp["canonical"]]
                resolved_traces.append({
                    "original": sp["phrase"],
                    "canonical": sp["canonical"],
                    "hindi_aliases": sorted(list(sp["record"]["hindi"])),
                    "english_aliases": sorted(list(sp["record"]["english"])),
                    "roman_aliases": sorted(list(sp["record"]["roman"])),
                    "ocr_variants": sorted(list(sp["record"]["ocr_variants"]))
                })
            variants.add(" ".join(canon_tokens))

            all_hindi_aliases = set()
            for sp in spans:
                all_hindi_aliases.update(sp["record"]["hindi"])

            for h_alias in all_hindi_aliases:
                deva_tokens = list(words)
                for sp in reversed(spans):
                    aliases = list(sp["record"]["hindi"])
                    alias_to_use = h_alias if h_alias in aliases else (aliases[0] if aliases else sp["canonical"])
                    deva_tokens[sp["start"]:sp["end"]] = [alias_to_use]
                variants.add(" ".join(deva_tokens))

            all_en_aliases = set()
            for sp in spans:
                all_en_aliases.update(sp["record"]["english"])

            for en_alias in all_en_aliases:
                latin_tokens = list(words)
                for sp in reversed(spans):
                    aliases = list(sp["record"]["english"])
                    alias_to_use = en_alias if en_alias in aliases else (aliases[0] if aliases else sp["canonical"])
                    latin_tokens[sp["start"]:sp["end"]] = [alias_to_use]
                variants.add(" ".join(latin_tokens))

        common_en = {"how", "many", "what", "which", "is", "are", "were", "was", "the", "in", "of", "to", "for", "from", "total", "issued", "received"}
        query_words = set(re.findall(r"\b[a-zA-Z]+\b", cleaned.lower()))
        is_english = bool(query_words and (len(query_words & common_en) / len(query_words) >= 0.3 or query_words & {"how", "what", "which", "issued", "received", "bilingual", "bilingually"}))

        if script in ["latin_english_or_hinglish", "mixed_hinglish"]:
            if not is_english:
                deva_trans = self.algorithmic_transliterate(cleaned)
                if deva_trans and deva_trans != cleaned:
                    variants.add(deva_trans)
                en_stops = {"what", "are", "the", "and", "is", "in", "of", "to", "for", "from", "how", "many", "under", "kaun", "se", "hain", "ke", "ki", "ka", "aur", "antargat"}
                content_words = [w for w in re.findall(r"\b[a-zA-Z]+\b", cleaned) if w.lower() not in en_stops]
                for cw in content_words:
                    tr = self.algorithmic_transliterate(cw)
                    if tr and tr.lower() != cw.lower() and re.search(r'[\u0900-\u097F]', tr):
                        variants.add(tr)

            # Universal cross-lingual vocabulary bridging for bilingual and Roman Hindi administrative documents
            bilingual_vocab = {
                # English terms
                "region": "क्षेत्र", "regions": "क्षेत्र", "section": "धारा", "sections": "धारा",
                "dhara": "धारा", "kshetra": "क्षेत्र",
                "poetry": "काव्य", "poem": "कविता", "poems": "कविताएं",
                "recitation": "पाठ", "competition": "प्रतियोगिता",
                "participated": "प्रतिभागिता", "participation": "प्रतिभागिता", "participants": "प्रतिभागी",
                "prize": "पुरस्कार", "award": "पुरस्कार", "awards": "पुरस्कार",
                "workshop": "कार्यशाला", "workshops": "कार्यशाला",
                "training": "प्रशिक्षण", "trained": "प्रशिक्षित",
                "meeting": "बैठक", "meetings": "बैठकें",
                "bilingual": "द्विभाषी", "bilingually": "द्विभाषी",
                "quarter": "तिमाही", "quarterly": "तिमाही",
                "officer": "अधिकारी", "officers": "अधिकारियों",
                "employee": "कर्मचारी", "employees": "कर्मचारियों",
                "personnel": "कार्मिक",
                "letter": "पत्र", "letters": "पत्रों",
                "phrases": "वाक्यांशों", "notes": "टिप्पणियों",
                "terminology": "शब्दावली", "administrative": "प्रशासनिक", "accounts": "लेखा",

                # Roman Hindi terms
                "kavya": "काव्य", "paath": "पाठ", "pratiyogita": "प्रतियोगिता", "puraskar": "पुरस्कार",
                "karyashala": "कार्यशाला", "karyashalayen": "कार्यशालाएं", "shabdavali": "शब्दावली",
                "aadharbhoot": "आधारभूत", "prashasnik": "प्रशासनिक", "lekha": "लेखा",
                "karmachariyon": "कर्मचारियों", "karmachari": "कर्मचारी", "karmik": "कार्मिक",
                "baithak": "बैठक", "baithakon": "बैठकों", "patra": "पत्र", "patron": "पत्रों",
                "tippaniyon": "टिप्पणियों", "tippani": "टिप्पणी", "vakyanshon": "वाक्यांशों", "vakyansh": "वाक्यांश",
                "prashikshan": "प्रशिक्षण", "aayojit": "आयोजित", "jankari": "जानकारी",
                "varsh": "वर्ष", "mahine": "महीने", "timahi": "तिमाही",
                "varishth": "वरिष्ठ", "adhikariyon": "अधिकारियों", "adhikari": "अधिकारी",
                "manak": "मानक", "maanak": "मानक", "dwibhashi": "द्विभाषी", "dvibhashi": "द्विभाषी",
                "sankalit": "संकलित", "pratibhadita": "प्रतिभागिता", "bhaag": "भाग",
            }
            en_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", cleaned)]
            for ew in en_words:
                if ew in bilingual_vocab:
                    variants.add(bilingual_vocab[ew])

        variant_list = [v for v in sorted(list(variants)) if len(v.strip()) > 1 or (len(v.strip()) == 1 and re.match(r'[\u0900-\u097F]', v.strip()))]

        metadata = {
            "original_query": original_query,
            "detected_script": script,
            "resolved_entities": resolved_traces,
            "variants_count": len(variant_list),
            "variants": variant_list
        }

        logger.info(f"[QueryNormalizer] Script: '{script}' | Generated {len(variant_list)} variants: {variant_list}")
        return variant_list, metadata


# Guard: meeting/committee-date queries must not be conflated with the quarterly
# reporting period date even if the embedding is similar.
_MEETING_DATE_EXCL_RE = re.compile(
    r'\b(?:baithak|meeting|samiti|committee|baith|बैठक|समिति|बाइठक)\b',
    re.IGNORECASE,
)

class SemanticQueryAnalyzer:
    def __init__(self, storage_path: Optional[str] = None, embed_fn: Optional[Any] = None):
        self.term_memory = DynamicTerminologyMemory(storage_path=storage_path)
        self.normalizer = QueryNormalizer(storage_path=storage_path)
        self.embed_fn = embed_fn
        self._meta_anchor_emb: Optional[np.ndarray] = None
        self._form_anchor_emb: Optional[np.ndarray] = None
        if self.embed_fn:
            self._init_anchors()

    def set_embed_fn(self, embed_fn: Any):
        self.embed_fn = embed_fn
        self._init_anchors()

    def _init_anchors(self):
        try:
            embs = self.embed_fn([
                "quarterly reporting period date office address phone email contact metadata",
                "form fields workshop training employees meetings documents administrative data"
            ])
            self._meta_anchor_emb = np.array(embs[0], dtype=np.float32)
            self._form_anchor_emb = np.array(embs[1], dtype=np.float32)
        except Exception:
            pass

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def algorithmic_transliterate(self, text: str, to_script: Optional[str] = None, *args, **kwargs) -> str:
        return self.normalizer.algorithmic_transliterate(text, to_script=to_script, *args, **kwargs)

    def extract_entities(self, query: str, document_id: Optional[str] = None) -> List[str]:
        # Strip unbalanced quotes and clean query
        cleaned_query = re.sub(r'[\'"]', '', query)
        
        capitalized = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', query)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', query)

        # Filter common English interrogatives/command words from capitalized
        english_stops = {"tell", "explain", "what", "who", "where", "when", "why", "how", "describe", "show", "list", "give", "write", "the", "and", "about", "are", "is", "in", "to", "of", "under"}
        clean_cap = [c for c in capitalized if c.lower() not in english_stops]
        clean_acr = [a for a in acronyms if a.lower() not in english_stops]

        hindi_stops = {"ने", "का", "के", "की", "में", "से", "पर", "है", "हैं", "कौन", "कितने", "सा", "किसने", "लेख", "कहा", "कववता", "कविता"}
        words = cleaned_query.split()
        hindi_entities = []
        current = []
        for w in words:
            if re.search(r'[\u0900-\u097F]', w) and w not in hindi_stops:
                current.append(w)
            else:
                if current:
                    hindi_entities.append(" ".join(current))
                    current = []
        if current:
            hindi_entities.append(" ".join(current))

        # Check terminology memory for multi-word or single-word matches
        matched_from_memory = []
        # Return unique non-empty entities
        seen = set()
        unique_entities = []
        for e in clean_cap + clean_acr + hindi_entities:
            e_clean = e.strip()
            if len(e_clean) >= 1 and e_clean.lower() not in english_stops and e_clean.lower() not in seen:
                seen.add(e_clean.lower())
                unique_entities.append(e_clean)

        return unique_entities

    def analyze(self, query: str, requested_mode: str = "AUTO", document_id: Optional[str] = None) -> Dict[str, Any]:
        cleaned_query = query.strip()
        lower_q = cleaned_query.lower()
        entities = self.extract_entities(cleaned_query, document_id=document_id)

        # Multi-variant normalization and script detection
        normalized_variants, norm_meta = self.normalizer.generate_normalized_variants(cleaned_query, document_id=document_id)
        detected_script = norm_meta.get("detected_script", "neutral")
        resolved_entities = norm_meta.get("resolved_entities", [])
        # This recognises English, Hindi, and Roman-Hindi field names before the
        # broader content/table heuristics decide a route.
        structured_field = FieldNormalizer.canonical(cleaned_query)

        # Build a combined lower-cased string from all script variants (original +
        # auto-transliterated Devanagari + Romanized Devanagari). Signals are evaluated against this so that
        # Roman-Hindi queries are matched via transliteration, not hardcoded word lists.
        variants = [lower_q]
        for v in normalized_variants:
            vl = v.lower()
            if vl not in variants:
                variants.append(vl)
        roman = romanize_generic(cleaned_query).lower()
        if roman and roman not in variants:
            variants.append(roman)
        all_variants_lower = " ".join(variants)

        # 1. Detect Content-Based Question Indicators (explanations, challenges, reasons, summaries, mechanisms, impact)
        is_content_question = (
            any(k in lower_q for k in [
                "challenge", "challenges", "chunauti", "chunautiyan", "चुनौती", "चुनौतियां", "चुनौतियाँ",
                "solution", "solutions", "samadhan", "समाधान",
                "explain", "describe", "summary", "summarize", "karan", "reason", "impact", "prabhav",
                "why", "how", "what is", "what are", "kya hai", "kya hain", "kaise", "kyun", "kyu",
                "importance", "benefits", "surveillance", "suraksha ki chunautiyan", "सुरक्षा की चुनौतियां",
                "प्रभाव", "असर", "परिणाम", "माध्यम", "किसके माध्यम", "किस माध्यम", "किस तरह", "तरीका", "उद्देश्य", "कारण", "विवरण", "प्रक्रिया", "विस्तार",
                "madhyam", "kiske madhyam", "tarika", "uddeshya", "vivaran"
            ]) and not any(k in lower_q for k in [
                "who wrote", "author", "writer", "kisne likha", "lekhak kaun", "लेखक",
                "how many", "total articles", "kitne lekh", "kul kitne", "कुल कितने",
                "list all", "page number", "kis page"
            ])
        )

        # 2. Detect COUNT Queries across English, Hindi, Roman Hindi, and Hinglish
        is_count = (
            any(k in lower_q for k in [
                "how many", "count", "kitne", "kitni", "kitnon", "kul kitne", "kitna",
                "कितने", "कितनों", "कुल कितने", "कुल", "संख्या", "गिनती"
            ])
            or bool(re.search(r'\btotal\s+(?:\w+\s+)*(?:articles?|lekh|kavita|poems?|stories|kahani|items?|records?)\b', lower_q))
            or bool(re.search(r'\b(?:total|kul)\s+(?:kitne|kitni|kitnon|count)\b', lower_q))
            or bool(re.search(r'\b(?:kitne|kitni|kitnon)\s+(?:\w+\s+)*(?:articles?|lekh|kavita|poems?|stories|kahani|items?|records?|documents?|docs?)\b', lower_q))
            or bool(re.search(r'\b(?:articles?|lekh|kavita|poems?|documents?)\s+(?:count|sankhya|kitne|kitni|kitnon|total|numbers?)\b', lower_q))
            or bool(re.search(r'\btotal\s+(?:articles?\s+|documents?\s+)?present\b', lower_q))
            or bool(re.search(r'\b(?:is\s+)?document\s+(?:mein|me)\s+(?:total\s+)?kitne\b', lower_q))
            # English: "no of X", "number of X", "total no of X", "total number of X"
            or bool(re.search(r'\b(?:total\s+)?(?:no\.?|number)\s+of\s+(?:\w+\s+)*(?:articles?|lekh|kavita|poems?|stories|kahani|items?|records?|documents?|docs?)\b', lower_q))
            or bool(re.search(r'\b(?:total\s+no\.?|total\s+number)\b', lower_q))
        )

        # Universal Expected Value-Type / Answer-Type Detection
        is_pct_query = bool(re.search(
            r'(?<![\w\u0900-\u097F])(?:प्रतिशत|प्रतिशतता|फीसदी|percent|percentage|pct|%)(?![\w\u0900-\u097F])',
            all_variants_lower, re.IGNORECASE
        ))
        is_duration_query = bool(re.search(
            r'(?<![\w\u0900-\u097F])(?:कितने\s*(?:वर्ष|साल|माह|महीने|दिन|घंटे)|कितनी\s*(?:अवधि)|how\s+many\s+(?:years?|months?|days?|hours?)|kitn[ei]\s+(?:saal|varsh|mahine)|duration|tenure|समय|अवधि)(?![\w\u0900-\u097F])',
            all_variants_lower, re.IGNORECASE
        ))
        is_date_query = bool(re.search(
            r'(?<![\w\u0900-\u097F])(?:कब|किस\s*(?:तारीख|तिथि|दिनांक|दिन)|what\s+date|which\s+date|when|kis\s*(?:tithi|tarikh|din)|date\s+of)(?![\w\u0900-\u097F])',
            all_variants_lower, re.IGNORECASE
        ))
        is_monetary_query = bool(re.search(
            r'(?<![\w\u0900-\u097F])(?:रुपये|रुपए|rupees?|rupaye|rs\.?|inr|राशि|amount|नकद\s*पुरस्कार|नकद\s*राशि|cash\s*prize|prize\s*money)(?![\w\u0900-\u097F])',
            all_variants_lower, re.IGNORECASE
        ))
        is_person_query = bool(re.search(
            r'(?<![\w\u0900-\u097F])(?:किन\s*लोगों|किसने|किसको|कौन|who|whom|kin\s*logon|kisne|kise)(?![\w\u0900-\u097F])',
            all_variants_lower, re.IGNORECASE
        ))

        expected_value_type = "TEXT"
        if is_pct_query:
            expected_value_type = "PERCENTAGE"
        elif is_person_query:
            expected_value_type = "PERSON"
        elif is_duration_query:
            expected_value_type = "DURATION"
        elif is_date_query:
            expected_value_type = "DATE"
        elif is_monetary_query:
            expected_value_type = "MONETARY"
        elif is_content_question:
            expected_value_type = "DESCRIPTION"
        elif is_count:
            expected_value_type = "COUNT"

        # Condition vs Target clause analysis (e.g. "X में से Y कितनों के ...")
        condition_clause = ""
        target_clause = lower_q
        split_match = re.search(r'(?<![\w\u0900-\u097F])(?:में\s*से|out\s+of|from\s+among)(?![\w\u0900-\u097F])', lower_q)
        if split_match:
            condition_clause = lower_q[:split_match.start()].strip()
            target_clause = lower_q[split_match.end():].strip()

        _LANG_PATTERNS = {
            "hindi": r'(?<![\w\u0900-\u097F])(?:हिन्दी|हिंदी|hindi)(?![\w\u0900-\u097F])',
            "english": r'(?<![\w\u0900-\u097F])(?:अंग्रेजी|अंग्रेज़ी|english)(?![\w\u0900-\u097F])',
            "bilingual": r'(?<![\w\u0900-\u097F])(?:द्विभाषी|द्विभाषीय|bilingual)(?![\w\u0900-\u097F])',
        }
        target_langs = set()
        condition_langs = set()
        for lang_name, lang_pat in _LANG_PATTERNS.items():
            if re.search(lang_pat, target_clause, re.IGNORECASE):
                target_langs.add(lang_name)
            if condition_clause and re.search(lang_pat, condition_clause, re.IGNORECASE):
                condition_langs.add(lang_name)

        if not split_match:
            for lang_name, lang_pat in _LANG_PATTERNS.items():
                if re.search(lang_pat, lower_q, re.IGNORECASE):
                    target_langs.add(lang_name)

        # 3. Detect LIST Queries
        is_list = any(k in lower_q for k in [
            "list", "name all", "name any", "sabke naam", "kaun kaun se", "kaun-kaun se",
            "नाम बताओ", "सूची", "कौन-कौन से लेख", "कौन से लेख", "which articles", "which poems", "कौन सी कविताएं", "बताओ",
            "show all", "list of all", "list all articles"
        ])

        # 3b. Detect BRANCH / CHILD ENUMERATION Queries across English, Hindi, and Hinglish
        # Generic: matches questions seeking enumeration of child branches, categories,
        # regions, divisions, or sub-items under a parent or across named siblings.
        is_branch_enumeration = (
            # Hindi Devanagari: "धारा 6 के अंतर्गत कौन-कौन से क्षेत्र हैं", "के अंतर्गत कौन से ...", "कौन-कौन से क्षेत्र"
            bool(re.search(
                r'(?<![\w\u0900-\u097F])(?:के\s*अंतर्गत|के\s*तहत|में)\s+(?:कौन-कौन\s*से|कौन\s*से|क्या-क्या)\b',
                lower_q
            ))
            or bool(re.search(
                r'(?<![\w\u0900-\u097F])(?:कौन-कौन\s*से|कौन\s*से)\s+(?:क्षेत्र|भाग|श्रेणियां|वर्ग|मद|विषय|प्रावधान|नियम)\b',
                lower_q
            ))
            # Roman / Hinglish: "ke antargat kaun kaun se", "under X kaun kaun se"
            or bool(re.search(
                r'\b(?:kaun\s*kaun\s*se|kon\s*kon\s*se|kya\s*kya)\s+(?:kshetra|regions?|categories|sections?|parts?|types?|items?|bhaag|shreni|varg)\b',
                lower_q
            ))
            or bool(re.search(
                r'\b(?:ke\s+antargat|ke\s+tahat|under|in)\s+.*?\b(?:kaun\s*kaun\s*se|kon\s*kon\s*se|kaun\s*se|kya\s*kya)\b',
                lower_q
            ))
            # English: "What are the regions under Section 6?", "which regions are under..."
            or bool(re.search(
                r'\b(?:what\s+are|which|list)\s+(?:the\s+)?(?:\w+\s+)*(?:regions?|categories|sections?|divisions?|zones?|groups?|parts?|types?|items?|provisions?|clauses?|rules?)\s+(?:under|in|of|within)\b',
                lower_q
            ))
            # English sibling enumeration: "What are the K, Kh and G regions?", "What are the X, Y and Z categories?"
            or bool(re.search(
                r'\bwhat\s+are\s+(?:the\s+)?(?:\b\w+\b\s*[,/]\s*)*\b\w+\b\s+(?:and|or)\s+\b\w+\b\s+(?:regions?|categories|sections?|divisions?|zones?|groups?|parts?|types?|items?|provisions?|clauses?)\b',
                lower_q
            ))
            # Generic "what is/are in [Section/Chapter/Parent]":
            or bool(re.search(
                r'\bwhat\s+(?:is|are|comes?|falls?)\s+(?:in|under|inside)\s+(?:section|dhara|chapter|part|rule|category|धारा)\b',
                lower_q
            ))
        )

        # 4. Detect ARTICLE_BY_AUTHOR Queries (asking which article was written by author X)
        is_title_by_author = (
            any(k in lower_q for k in [
                "which article was written by", "which poem was written by", "which story was written by", "which report was written by",
                "which article did", "which poem did", "which story did", "which report did",
                "what did", "article written by", "poem written by", "story written by", "report written by",
                "ka article kaunsa", "kaun sa lekh", "kaunsa lekh", "kaun si report", "kaunsi report",
                "kaun si kavita", "kaunsi kavita", "kaun sa article", "kaun si kahani", "kaunsi kahani",
                "kya likha", "kaunsa article likha", "kaun sa article likha", "kaunsa lekh likha", "kaun sa lekh likha",
                "कौन सा लेख लिखा", "कौन सी कविता लिखी", "ने कौन सा लेख", "ने कौन सी कविता", "ने क्या लिखा", "द्वारा लिखा गया"
            ])
            or bool(re.search(r'\bwhat\s+did\s+.+?\s+write\b', lower_q))
            or bool(re.search(r'\bwhich\s+(?:article|poem|story|report)\s+(?:was\s+written\s+by|did)\s+', lower_q))
            or bool(re.search(r'\bne\s+(?:kaun\s*(?:sa|si)|kya|likha|likhi)\b', lower_q))
            or bool(re.search(r'\bने\s+(?:कौन\s*(?:सा|सी)|क्या|लिखा|लिखी)', lower_q))
        )

        # 5. Detect AUTHOR_BY_ARTICLE Queries (asking who wrote article Y)
        is_author_by_article = (
            not is_title_by_author and (
                any(k in lower_q for k in [
                    "who wrote", "author", "writer", "written by", "kisne likha", "kisne likhi", "kisne likhe", "kisne write",
                    "kavita kisne", "lekhak kaun", "ke lekhak", "ka lekhak", "lekhak", "rachayita", "rachnakar", "kiski rachna",
                    "लेखक", "रचयिता", "रचनाकार", "किसने लिखा", "किसने लिखी", "किसने लिखे", "द्वारा लिखित", "की रचना", "किसका लेख", "किसकी कविता", "ने लिखा"
                ])
                or bool(re.search(r'\b(?:who\s+is\s+the\s+author|who\s+wrote)\b', lower_q))
                or bool(re.search(r'\b(?:kisne|kiski|kiska|kiske)\s+(?:likha|likhi|likhe|racha|rachi|rache)\b', lower_q))
            )
        )

        # 6. Detect PAGE, CATEGORY, STRUCTURE, ORDINAL, EXISTS
        is_page = any(k in lower_q for k in ["page", "page number", "start page", "पेज", "पृष्ठ", "kis page", "kis prishth", "किस पेज", "किस पृष्ठ", "which page"])
        is_legal_section = bool(re.search(r'\b(?:section|sec\.?|धारा)\s*\(?\d+', lower_q))
        is_category = not is_legal_section and any(k in lower_q for k in [
            "which category", "what category", "which section", "what section",
            "kis category", "kis section", "kis varg", "kis shreni",
            "किस श्रेणी", "किस वर्ग", "कौन सी श्रेणी", "कौन सा वर्ग",
            "category of", "section of", "shreni", "varg", "category"
        ])
        is_structure = any(k in lower_q for k in ["table of contents", "index", "front matter", "विषय सूची", "अनुक्रमणिका", "संपादक मंडल", "editorial"])
        
        # Ordinal words and numeric patterns (e.g. 1st, 2nd, 8th, eighth, aathva, etc.)
        # Ensure page numbers (e.g. page 8) are not treated as ordinals
        q_no_pg = re.sub(r'\b(?:page|prishth|पेज|पृष्ठ)\s*(?:no\.?|number|संख्या)?\s*\d+\b', ' ', lower_q)
        q_no_pg = re.sub(r'\b\d+\s*(?:page|prishth|पेज|पृष्ठ)\b', ' ', q_no_pg)
        is_ordinal = bool(re.search(r'\b\d+(?:st|nd|rd|th)\b', q_no_pg)) or any(k in q_no_pg for k in [
            "first", "1st", "pehla", "pahla", "पहला", "पहली", "प्रथम",
            "second", "2nd", "dusra", "doosra", "दूसरा", "दूसरी", "द्वितीय",
            "third", "3rd", "teesra", "tisra", "तीसरा", "तीसरी", "तृतीय",
            "fourth", "4th", "chautha", "चौथा", "चौथी", "चतुर्थ",
            "fifth", "5th", "panchva", "पांचवां", "पंचम",
            "sixth", "6th", "chhattha", "छठा", "छठी",
            "seventh", "7th", "saatva", "सातवां",
            "eighth", "8th", "aathva", "aathwa", "आठवां", "आठवीं", "अष्टम",
            "ninth", "9th", "nauva", "नौवां",
            "tenth", "10th", "dasva", "दसवां",
            "last", "aakhiri", "akhiri", "antim", "अंतिम", "आखिरी", "final"
        ])
        is_exists = any(k in lower_q for k in ["exists or not", "exist or not", "exists", "hai ya nahi", "hai kya", "मौजूद है", "उपलब्ध है"])
        is_issued_doc = bool(re.search(
            r'(?<![\w\u0900-\u097F])(?:jaari|jari|जारी|issue|issues|issued|bilingual|bilingually|dvibhashi|द्विभाषी|धारा\s*3|dhara\s*3|section\s*3|patra|patron|पत्र|पत्रों|sachiv|सचिव|baithak|बैठक|file|files|faile|फाइल|फाइलें|रिपोर्ट|report|reports|प्रपत्र|proforma|फॉर्म|form)(?![\w\u0900-\u097F])',
            lower_q,
            re.IGNORECASE
        ))
        index_subject = not is_issued_doc and bool(re.search(
            r"(?<![\w\u0900-\u097F])(?:article|articles|poem|poems|story|stories|essay|essays|magazine|magazines|toc|contents|author|writer|lekh|kavita|kahani|patrika|technical\s+article|तकनीकी\s+लेख|लेख|कविता|कहानी|अनुक्रमणिका|विषय\s*सूची|लेखक|पत्रिका|रचना|रचनाएं)(?![\w\u0900-\u097F])",
            lower_q,
            re.IGNORECASE
        ))
        is_index_question = not is_issued_doc and (
            is_title_by_author or is_author_by_article or is_page or is_category or
            is_structure or is_ordinal or is_exists or ((is_count or is_list) and index_subject)
        )

        # Generic table/form signals. This is intentionally not tied to any document,
        # schema, or business column. Exact column matching happens dynamically against
        # extracted headers in StructuredTableEngine.
        numeric_reference = bool(re.search(r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?\s*(?:%|₹|rs\.?|cr\.?|lakh|crore)\b", lower_q))
        calculation_signal = bool(re.search(r"(?<![\w\u0900-\u097F])(difference|diff|increase|decrease|average|mean|total|sum|highest|lowest|maximum|minimum|percentage|percent|अंतर|फर्क|औसत|योग|कुल|अधिकतम|न्यूनतम|प्रतिशत)(?![\w\u0900-\u097F])", lower_q))
        value_signal = bool(re.search(r"\b(how much|how many|what was|show (?:the )?data|kitna|kitne|kitni|कितना|कितने|संख्या|डेटा)\b", lower_q))
        # Clean regex signals containing ONLY canonical concept keywords in English.
        # Transliteration into all_variants_lower allows Roman-Hindi queries to match,
        # while embedding similarity dynamically matches cross-lingual Hindi queries
        # without hardcoded synonym lists.
        report_metadata_regex = bool(re.search(
            r"\b(date|dated|quarter|quarterly|period|reporting\s+period|office|address|phone|telephone|email)\b",
            all_variants_lower))
        form_request_regex = bool(re.search(
            r"(?<![\w\u0900-\u097F])(?:details?\s+(?:asked|required)|field|fields|form|forms|workshop|workshops|training|trainings|employees?|officers?|files?|meetings?|letters?|issued|issue|issues|jaari|jari|जारी|bilingual|bilingually|dvibhashi|द्विभाषी|sachiv|सचिव|patra|patron|पत्र|पत्रों)(?![\w\u0900-\u097F])",
            all_variants_lower,
            re.IGNORECASE
        ))
        if is_issued_doc:
            form_request_regex = True


        report_metadata_emb = False
        form_request_emb = False
        if self.embed_fn and self._meta_anchor_emb is not None and self._form_anchor_emb is not None:
            try:
                q_emb = np.array(self.embed_fn([cleaned_query])[0], dtype=np.float32)
                meta_sim = self._cosine_sim(q_emb, self._meta_anchor_emb)
                form_sim = self._cosine_sim(q_emb, self._form_anchor_emb)
                report_metadata_emb = meta_sim >= 0.45
                form_request_emb = form_sim >= 0.45
            except Exception:
                pass

        report_metadata_signal = report_metadata_regex or report_metadata_emb
        form_request_signal = form_request_regex or form_request_emb
        table_intent = (structured_field is not None
                        or (numeric_reference and (value_signal or calculation_signal or len(re.findall(r"\b(?:19|20)\d{2}\b", lower_q)) >= 2))
                        or calculation_signal or (is_count and not index_subject)
                        or report_metadata_signal or form_request_signal)
        # Content / explanation / prose questions should NEVER be hijacked by table intent
        if is_content_question and not (is_pct_query or is_count or is_monetary_query):
            table_intent = False
        is_hybrid = table_intent and is_content_question

        # Extract explicit quoted titles (e.g., "Nayi Subah" or 'Semiconductor Ecosystem')
        quoted_matches = re.findall(r'[\'\"\'\"\“\”\‘\’]([^\'\"\'\"\“\”\‘\’]+)[\'\"\'\"\“\”\‘\’]', query)
        target_title = quoted_matches[0].strip() if quoted_matches else None

        # Check for item indicators: "poem X" / "X poem" / "कविता X" / "X लेख" if not count/list
        if not target_title and not (is_count or is_list or is_title_by_author):
            pre_match = re.search(r'(?:poem|article|story|essay|कविता|लेख|कहानी)\s+([A-Za-z0-9\u0900-\u097F\s]{2,35})', query, re.IGNORECASE)
            post_match = re.search(r'(?:who\s+wrote\s+(?:the\s+)?|about\s+(?:the\s+)?|in\s+(?:the\s+)?|the\s+)?([A-Z][a-zA-Z0-9\u0900-\u097F\s]{2,35})\s+(?:poem|article|story|essay|कविता|लेख|कहानी)', query)
            
            cand = None
            if post_match:
                cand = post_match.group(1).strip()
            elif pre_match:
                cand = pre_match.group(1).strip()

            if cand:
                cleaned_cand = re.sub(r'^(?:the|a|an|about|in|on|who\s+wrote|what\s+is|का|के|की|में)\s+', '', cand, flags=re.IGNORECASE).strip()
                if len(cleaned_cand) >= 2 and cleaned_cand.lower() not in {"how many", "which", "all", "the", "many"}:
                    target_title = cleaned_cand

        # Fallback to multi-word capitalized entity if target_title is still None and not a count/list query
        if not target_title and not (is_count or is_list or is_title_by_author):
            for ent in entities:
                if len(ent.split()) >= 2:
                    target_title = ent
                    break

        requested_field = None
        operation = "SEMANTIC_QA"
        if table_intent and not is_index_question:
            # Words such as "quarter" and "office" are often qualifiers in a
            # numeric form question (for example, "workshops conducted during
            # the quarter"), not a request for the reporting-period or office
            # metadata itself.  Direct metadata lookup is safe only when the
            # user explicitly asks for that field and is not asking for a count
            # or an operational/form fact.  All other table questions go to the
            # document-scoped table matcher, which compares the question against
            # extracted headers, row labels, and values.
            metadata_field = (
                structured_field in {"reporting_period", "office_name", "address", "phone", "email"}
                and not (is_count or value_signal or form_request_signal or calculation_signal)
                and not (structured_field == "reporting_period"
                         and bool(_MEETING_DATE_EXCL_RE.search(lower_q)))
            )
            intent = "REPORT_METADATA_LOOKUP" if metadata_field else ("TABLE_CALCULATION" if calculation_signal else "FORM_FIELD_LOOKUP")
            operation = "STRUCTURED_LOOKUP"
            requested_field = structured_field if metadata_field else None
        elif is_count:
            intent = "count"
            operation = "COUNT"
        elif is_branch_enumeration:
            intent = "branch_enumeration"
            operation = "BRANCH_ENUMERATION"
            requested_field = "branch_children"
        elif is_list:
            intent = "list"
            operation = "LIST"
        elif is_title_by_author:
            intent = "find_title_by_author"
            operation = "ARTICLE_BY_AUTHOR"
            requested_field = "title"
        elif is_author_by_article:
            intent = "find_author"
            operation = "AUTHOR_BY_ARTICLE"
            requested_field = "author"
        elif is_page:
            intent = "find_page"
            operation = "PAGE"
            requested_field = "page"
        elif is_category:
            intent = "find_category"
            operation = "CATEGORY"
            requested_field = "category"
        elif is_ordinal:
            intent = "ordinal_lookup"
            operation = "ORDINAL"
        elif is_exists:
            intent = "exists_check"
            operation = "EXISTS"
        elif is_structure:
            intent = "structure_query"
            operation = "STRUCTURE"
        else:
            intent = "semantic_qa"
            operation = "SEMANTIC_QA"

        # Determine Query Mode (INDEX_BASED vs CONTENT_BASED)
        req_upper = (requested_mode or "AUTO").upper().strip()
        if "INDEX" in req_upper:
            # A UI selection is a preference, not permission to search the TOC
            # for an ordinary office/form question.
            resolved_mode = "INDEX_BASED" if is_index_question else (("REPORT_METADATA" if intent == "REPORT_METADATA_LOOKUP" else "FORM_BASED") if table_intent else "CONTENT_BASED")
        elif "CONTENT" in req_upper:
            resolved_mode = ("REPORT_METADATA" if intent == "REPORT_METADATA_LOOKUP" else "FORM_BASED") if table_intent else "CONTENT_BASED"
        else:
            # Automatic routing: If explicitly content-seeking question, route to CONTENT_BASED
            if table_intent and not is_index_question:
                resolved_mode = "REPORT_METADATA" if intent == "REPORT_METADATA_LOOKUP" else "FORM_BASED"
            elif is_content_question:
                resolved_mode = "CONTENT_BASED"
            # Index/Metadata questions routed to INDEX_BASED
            elif is_index_question:
                resolved_mode = "INDEX_BASED"
            else:
                resolved_mode = "CONTENT_BASED"

        # Enrich entities with all canonical terms and aliases resolved from Terminology Memory
        enriched_entities = list(entities)
        for res in resolved_entities:
            canon = res.get("canonical")
            if canon and canon not in enriched_entities:
                enriched_entities.append(canon)
            for h in res.get("hindi_aliases", []):
                if h and h not in enriched_entities:
                    enriched_entities.append(h)
            for en in res.get("english_aliases", []):
                if en and en not in enriched_entities:
                    enriched_entities.append(en)

        # Extract target year if query mentions a year (e.g. 2023, 2024, 2025)
        year_match = re.search(r'\b(19[89]\d|20[0-3]\d)\b', cleaned_query)
        target_year = int(year_match.group(1)) if year_match else None

        # Extract target document type if query mentions magazine or report
        target_doc_type = None
        if re.search(r'\b(?:patrika|magazine|पत्रिका|अंक|मैगजीन)\b', all_variants_lower, re.IGNORECASE):
            target_doc_type = "magazine"
        elif re.search(r'\b(?:report|reporting|रिपोर्ट|प्रतिवेदन|प्रपत्र|form|proforma)\b', all_variants_lower, re.IGNORECASE):
            target_doc_type = "report"

        return {
            "original_query": query,
            "cleaned_query": cleaned_query,
            "detected_script": detected_script,
            "intent": intent,
            "operation": operation,
            "query_mode": resolved_mode,
            "requested_mode": requested_mode,
            "entities": enriched_entities,
            "target_title": target_title,
            "target_entity_type": "article" if requested_field else None,
            "target_entity": target_title,
            "target_year": target_year,
            "target_document_type": target_doc_type,
            "requested_field": requested_field,
            "is_count_or_list": is_count or is_list or is_branch_enumeration or is_author_by_article or is_title_by_author or is_structure,
            "is_branch_enumeration": is_branch_enumeration,
            "is_index_question": is_index_question,
            "normalized_variants": normalized_variants,
            "resolved_entities": resolved_entities,
            "document_id": document_id,
            "table_intent": table_intent,
            "table_operation": "calculation" if calculation_signal else ("lookup" if table_intent else None),
            "is_hybrid": is_hybrid,
            "expected_value_type": expected_value_type,
            "target_languages": list(target_langs),
            "condition_languages": list(condition_langs),
        }

    def expand_query(self, analysis: Dict[str, Any]) -> List[str]:
        orig = analysis["original_query"]
        variants = set(analysis.get("normalized_variants", [orig]))
        variants.add(analysis["cleaned_query"])

        dev_trans = self.algorithmic_transliterate(analysis["cleaned_query"])
        if dev_trans and dev_trans != analysis["cleaned_query"]:
            variants.add(dev_trans)

        # 1. Expand query-level matching terms from Terminology Memory
        query_aliases = self.term_memory.get_all_aliases(analysis["cleaned_query"])
        for al in query_aliases:
            if al and al != analysis["cleaned_query"]:
                variants.add(al)

        # 2. Expand entity-level matching terms from Terminology Memory & transliteration
        for ent in analysis["entities"]:
            variants.add(ent)
            dev_ent = self.algorithmic_transliterate(ent)
            if dev_ent:
                variants.add(dev_ent)
            
            ent_aliases = self.term_memory.get_all_aliases(ent, document_id=analysis.get("document_id"))
            for al in ent_aliases:
                if al:
                    variants.add(al)

        # For author or entity queries, always search TOC and Poetry index
        if (analysis["intent"] == "find_author" or analysis["is_count_or_list"]) and analysis.get("intent") != "branch_enumeration":
            variants.add("Table of Contents Index विषय सूची अनुक्रमणिका कविताएं")

        return list(set([v for v in variants if len(v.strip()) > 1 or (len(v.strip()) == 1 and re.match(r'[\u0900-\u097F]', v.strip()))]))

QueryAnalyzer = SemanticQueryAnalyzer
