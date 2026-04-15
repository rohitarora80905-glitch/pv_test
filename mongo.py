"""
mongo.py — MongoDB connection, collection access, and document schemas.

All schema definitions use plain dicts (no ODM) to keep dependencies minimal.
Indexes are created on first connection.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import yaml
from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.collection import Collection
from pymongo.database import Database

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# MongoDB connection (singleton pattern)
# ─────────────────────────────────────────────

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_db(config_path: str = "config.yaml") -> Database:
    global _client, _db
    if _db is None:
        cfg = load_config(config_path)
        uri = cfg["mongodb"]["uri"]
        db_name = cfg["mongodb"]["db_name"]
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _db = _client[db_name]
        _ensure_indexes(_db, cfg["mongodb"]["collections"])
        logger.info(f"Connected to MongoDB: {uri} / {db_name}")
    return _db


def get_collection(name: str, config_path: str = "config.yaml") -> Collection:
    db = get_db(config_path)
    cfg = load_config(config_path)
    col_name = cfg["mongodb"]["collections"].get(name, name)
    return db[col_name]


# ─────────────────────────────────────────────
# Index setup
# ─────────────────────────────────────────────

def _ensure_indexes(db: Database, collections: dict):
    # transcripts
    db[collections["transcripts"]].create_index(
        [("patient_id", ASCENDING), ("timestamp", ASCENDING)]
    )
    db[collections["transcripts"]].create_index([("status", ASCENDING)])

    # corrected_transcripts
    db[collections["corrected"]].create_index([("transcript_id", ASCENDING)])
    db[collections["corrected"]].create_index([("needs_review", ASCENDING)])

    # name_memory
    db[collections["name_memory"]].create_index(
        [("canonical", ASCENDING)], unique=True
    )

    # review_queue
    db[collections["review_queue"]].create_index([("status", ASCENDING)])
    db[collections["review_queue"]].create_index([("created_at", ASCENDING)])

    logger.info("MongoDB indexes ensured.")


# ─────────────────────────────────────────────
# SCHEMA HELPERS
# Each function returns a dict ready to insert.
# ─────────────────────────────────────────────

def transcript_schema(
    raw_text: str,
    patient_id: Optional[str] = None,
    source: str = "whisper",
    metadata: Optional[dict] = None,
) -> dict:
    """
    Schema for raw Whisper transcript documents.

    Collection: transcripts
    Status flow: pending → processing → done | error
    """
    return {
        "raw_text": raw_text,
        "patient_id": patient_id,
        "source": source,
        "status": "pending",           # pending | processing | done | error
        "timestamp": datetime.now(timezone.utc),
        "metadata": metadata or {},
    }


def corrected_transcript_schema(
    transcript_id: str,
    cleaned_text: str,
    reviewed_text: Optional[str] = None,
    name_corrections: Optional[dict] = None,
    uncertain_flags: Optional[list] = None,
    confidence_score: Optional[float] = None,
    needs_review: bool = False,
) -> dict:
    """
    Schema for LLM-corrected transcript output.

    Collection: corrected_transcripts
    """
    return {
        "transcript_id": transcript_id,     # References transcripts._id
        "cleaned_text": cleaned_text,
        "reviewed_text": reviewed_text,     # Set after human approval
        "name_corrections": name_corrections or {},
        "uncertain_flags": uncertain_flags or [],
        "confidence_score": confidence_score,
        "needs_review": needs_review,
        "review_status": "pending" if needs_review else "auto_approved",
        # review_status: pending | approved | rejected
        "created_at": datetime.now(timezone.utc),
        "reviewed_at": None,
    }


def name_memory_schema(
    canonical: str,
    variations: Optional[list] = None,
    frequency: int = 1,
    validated: bool = False,
) -> dict:
    """
    Schema for the persistent name memory dictionary.

    Collection: name_memory
    - canonical: the accepted correct spelling
    - variations: list of phonetic/spelling variants seen in transcripts
    - validated: True if a human has confirmed this mapping
    """
    return {
        "canonical": canonical,
        "variations": variations or [],
        "frequency": frequency,
        "validated": validated,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def review_queue_schema(
    transcript_id: str,
    corrected_id: str,
    review_type: str,           # "name" | "text"
    original: str,
    suggested: str,
    context: Optional[str] = None,
) -> dict:
    """
    Schema for items placed in the human review queue.

    Collection: review_queue
    review_type:
        "name"  → a single name correction needs approval
        "text"  → a full transcript chunk needs approval
    Status flow: pending → approved | rejected
    """
    return {
        "transcript_id": transcript_id,
        "corrected_id": corrected_id,
        "review_type": review_type,
        "original": original,
        "suggested": suggested,
        "context": context,
        "status": "pending",            # pending | approved | rejected
        "reviewer_note": None,
        "created_at": datetime.now(timezone.utc),
        "resolved_at": None,
    }
