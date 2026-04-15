"""
name_memory.py — Step 4: Name detection + persistent memory system.

Features:
- Fuzzy string matching (RapidFuzz)
- Phonetic matching (Soundex via jellyfish)
- MongoDB-backed name dictionary
- Auto-insert new names with validation flag
- Name correction cache for the session
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import jellyfish
from rapidfuzz import fuzz, process as rf_process

from mongo import get_collection, name_memory_schema

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Similarity thresholds
# ─────────────────────────────────────────────

FUZZY_THRESHOLD = 85        # RapidFuzz ratio threshold (0–100)
PHONETIC_MATCH = True       # Also use Soundex phonetic matching


# ─────────────────────────────────────────────
# Name Memory Manager
# ─────────────────────────────────────────────

class NameMemoryManager:
    """
    Manages the persistent name dictionary in MongoDB.

    Responsibilities:
    - Given a detected name, find its canonical form (or flag as new)
    - Update frequency counts
    - Add new names with validated=False (queued for human review)
    - Provide the full names dict for LLM prompt injection
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.col = get_collection("name_memory", config_path)
        self._session_cache: dict[str, str] = {}   # name → canonical (per-run cache)

    # ── public API ────────────────────────────

    def resolve_names(self, detected_names: list[str]) -> dict:
        """
        Given a list of raw names detected by the LLM:
        1. Match each to a canonical form if found (fuzzy + phonetic)
        2. Return dict: { raw_name: canonical_name }
        3. Add unmatched names to DB (needs human validation)

        Returns: { "Jon": "John", "Arav": "Aarav", "NewName": "NewName" }
        """
        resolution = {}
        for raw in detected_names:
            if not raw or not raw.strip():
                continue
            raw_clean = raw.strip()
            canonical = self._resolve_single(raw_clean)
            resolution[raw_clean] = canonical
        return resolution

    def get_names_for_prompt(self) -> dict:
        """
        Return the full name memory as a compact dict for LLM prompt injection.
        Format: { "canonical": ["variant1", "variant2"], ... }
        """
        docs = list(self.col.find({}, {"canonical": 1, "variations": 1, "_id": 0}))
        return {d["canonical"]: d.get("variations", []) for d in docs}

    def add_variation(self, canonical: str, variation: str):
        """
        Add a new variation to an existing canonical name entry.
        """
        if canonical == variation:
            return
        self.col.update_one(
            {"canonical": canonical},
            {
                "$addToSet": {"variations": variation},
                "$inc": {"frequency": 1},
                "$set": {"updated_at": datetime.now(timezone.utc)},
            },
        )
        logger.debug(f"Added variation '{variation}' → '{canonical}'")

    def mark_validated(self, canonical: str):
        """Human has confirmed this canonical name entry."""
        self.col.update_one(
            {"canonical": canonical},
            {"$set": {"validated": True, "updated_at": datetime.now(timezone.utc)}},
        )
        logger.info(f"Name '{canonical}' marked as validated.")

    def get_unvalidated(self) -> list[dict]:
        """Return all name entries not yet validated by a human."""
        docs = list(self.col.find({"validated": False}))
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

    def update_canonical(self, old_canonical: str, new_canonical: str):
        """
        Human overrides the canonical form.
        Moves old canonical to variations list.
        """
        self.col.update_one(
            {"canonical": old_canonical},
            {
                "$set": {
                    "canonical": new_canonical,
                    "validated": True,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$addToSet": {"variations": old_canonical},
            },
        )
        # Invalidate cache
        self._session_cache = {
            k: (new_canonical if v == old_canonical else v)
            for k, v in self._session_cache.items()
        }
        logger.info(f"Canonical updated: '{old_canonical}' → '{new_canonical}'")

    # ── internal matching ─────────────────────

    def _resolve_single(self, raw: str) -> str:
        """
        Try to match raw name against existing DB entries.
        Falls through: session_cache → DB fuzzy → phonetic → insert new
        """
        # 1. Session cache hit
        if raw in self._session_cache:
            return self._session_cache[raw]

        # 2. Exact match in DB
        doc = self.col.find_one({"canonical": raw})
        if doc:
            self._bump_frequency(raw)
            self._session_cache[raw] = raw
            return raw

        # 3. Check if raw is already a known variation
        doc = self.col.find_one({"variations": raw})
        if doc:
            canonical = doc["canonical"]
            self._bump_frequency(canonical)
            self._session_cache[raw] = canonical
            logger.info(f"Name '{raw}' resolved via variation → '{canonical}'")
            return canonical

        # 4. Fuzzy match against all canonicals
        all_names = [d["canonical"] for d in self.col.find({}, {"canonical": 1})]
        if all_names:
            match, score, _ = rf_process.extractOne(
                raw, all_names, scorer=fuzz.ratio
            ) or (None, 0, None)

            if match and score >= FUZZY_THRESHOLD:
                self.add_variation(match, raw)
                self._session_cache[raw] = match
                logger.info(f"Name '{raw}' fuzzy-matched → '{match}' (score={score})")
                return match

            # 5. Phonetic (Soundex) match
            if PHONETIC_MATCH:
                raw_soundex = jellyfish.soundex(raw)
                for candidate in all_names:
                    if jellyfish.soundex(candidate) == raw_soundex:
                        self.add_variation(candidate, raw)
                        self._session_cache[raw] = candidate
                        logger.info(
                            f"Name '{raw}' phonetic-matched → '{candidate}' "
                            f"(Soundex={raw_soundex})"
                        )
                        return candidate

        # 6. Truly new name — insert, mark for human validation
        self._insert_new_name(raw)
        self._session_cache[raw] = raw
        logger.info(f"New name detected and inserted: '{raw}' (awaiting validation)")
        return raw

    def _insert_new_name(self, name: str):
        """Insert a brand-new name into the memory DB."""
        existing = self.col.find_one({"canonical": name})
        if existing:
            self._bump_frequency(name)
            return
        doc = name_memory_schema(canonical=name, validated=False)
        self.col.insert_one(doc)

    def _bump_frequency(self, canonical: str):
        self.col.update_one(
            {"canonical": canonical},
            {
                "$inc": {"frequency": 1},
                "$set": {"updated_at": datetime.now(timezone.utc)},
            },
        )
