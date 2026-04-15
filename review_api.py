"""
review_api.py — Step 5: Human-in-the-loop review system.

Two review types:
  A. NAME REVIEW   — approve/reject individual name corrections
  B. TEXT REVIEW   — approve/reject full corrected transcripts

All review items live in the review_queue collection.
This module provides:
  - ReviewQueueManager  (business logic)
  - FastAPI router       (HTTP endpoints — mounted in fastapi_server.py)
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from mongo import (
    get_collection,
    review_queue_schema,
    corrected_transcript_schema,
)
from name_memory import NameMemoryManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/review", tags=["Human Review"])


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _to_str_id(doc: dict) -> dict:
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


def _parse_oid(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except (InvalidId, Exception):
        raise HTTPException(status_code=400, detail=f"Invalid ID: {id_str}")


# ─────────────────────────────────────────────
# Review Queue Manager (business logic)
# ─────────────────────────────────────────────

class ReviewQueueManager:
    """
    Manages the human review queue in MongoDB.
    Called by the main pipeline to enqueue items,
    and by the API to fetch/resolve them.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.queue_col = get_collection("review_queue", config_path)
        self.corrected_col = get_collection("corrected", config_path)
        self.name_mgr = NameMemoryManager(config_path)

    def enqueue_text_review(
        self,
        transcript_id: str,
        corrected_id: str,
        original_text: str,
        cleaned_text: str,
        flags: Optional[list] = None,
    ) -> str:
        """Add a full-text review item to the queue. Returns queue item _id."""
        context = f"Flags: {', '.join(flags)}" if flags else None
        doc = review_queue_schema(
            transcript_id=transcript_id,
            corrected_id=corrected_id,
            review_type="text",
            original=original_text,
            suggested=cleaned_text,
            context=context,
        )
        result = self.queue_col.insert_one(doc)
        logger.info(f"Enqueued TEXT review for transcript {transcript_id}")
        return str(result.inserted_id)

    def enqueue_name_review(
        self,
        transcript_id: str,
        corrected_id: str,
        original_name: str,
        suggested_name: str,
        context_snippet: Optional[str] = None,
    ) -> str:
        """Add a name correction review item to the queue. Returns queue item _id."""
        doc = review_queue_schema(
            transcript_id=transcript_id,
            corrected_id=corrected_id,
            review_type="name",
            original=original_name,
            suggested=suggested_name,
            context=context_snippet,
        )
        result = self.queue_col.insert_one(doc)
        logger.info(
            f"Enqueued NAME review: '{original_name}' → '{suggested_name}'"
        )
        return str(result.inserted_id)

    def resolve(
        self,
        queue_item_id: str,
        approved: bool,
        reviewer_note: Optional[str] = None,
        override_value: Optional[str] = None,
    ) -> dict:
        """
        Resolve a review item.

        - approved=True + no override → accept suggested value as-is
        - approved=True + override     → use override_value instead
        - approved=False               → reject; keep original

        Side effects:
        - Updates review_queue status
        - For NAME type: updates name_memory
        - For TEXT type: sets reviewed_text in corrected_transcripts
        """
        oid = _parse_oid(queue_item_id)
        item = self.queue_col.find_one({"_id": oid})
        if not item:
            raise HTTPException(status_code=404, detail="Review item not found.")
        if item["status"] != "pending":
            raise HTTPException(
                status_code=409,
                detail=f"Item already resolved (status={item['status']}).",
            )

        final_value = (
            override_value if override_value
            else (item["suggested"] if approved else item["original"])
        )
        status = "approved" if approved else "rejected"

        # Persist resolution
        self.queue_col.update_one(
            {"_id": oid},
            {
                "$set": {
                    "status": status,
                    "reviewer_note": reviewer_note,
                    "resolved_value": final_value,
                    "resolved_at": datetime.now(timezone.utc),
                }
            },
        )

        # Side effects
        if item["review_type"] == "name":
            self._apply_name_resolution(
                item["original"], item["suggested"], final_value, approved
            )
        elif item["review_type"] == "text":
            self._apply_text_resolution(item["corrected_id"], final_value, approved)

        logger.info(
            f"Review {queue_item_id} resolved: "
            f"type={item['review_type']}, approved={approved}"
        )
        return {
            "queue_item_id": queue_item_id,
            "status": status,
            "final_value": final_value,
        }

    # ── private side-effect handlers ─────────

    def _apply_name_resolution(
        self, original: str, suggested: str, final: str, approved: bool
    ):
        if approved:
            if final != suggested:
                # Human overrode our suggestion
                self.name_mgr.update_canonical(suggested, final)
            else:
                self.name_mgr.mark_validated(suggested)
                if original != suggested:
                    self.name_mgr.add_variation(suggested, original)
        # If rejected, leave name_memory as-is (variation stays unvalidated)

    def _apply_text_resolution(self, corrected_id: str, final_text: str, approved: bool):
        try:
            oid = ObjectId(corrected_id)
        except Exception:
            return
        self.corrected_col.update_one(
            {"_id": oid},
            {
                "$set": {
                    "reviewed_text": final_text if approved else None,
                    "review_status": "approved" if approved else "rejected",
                    "reviewed_at": datetime.now(timezone.utc),
                }
            },
        )


# ─────────────────────────────────────────────
# Pydantic request/response models
# ─────────────────────────────────────────────

class ResolveRequest(BaseModel):
    approved: bool
    reviewer_note: Optional[str] = None
    override_value: Optional[str] = None


# ─────────────────────────────────────────────
# API Router
# ─────────────────────────────────────────────

_mgr: Optional[ReviewQueueManager] = None

def get_review_manager() -> ReviewQueueManager:
    global _mgr
    if _mgr is None:
        _mgr = ReviewQueueManager()
    return _mgr


@router.get("/queue", summary="List pending review items")
async def list_review_queue(
    review_type: Optional[str] = Query(None, description="name | text"),
    status: str = Query("pending", description="pending | approved | rejected"),
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
):
    """
    Fetch items from the human review queue.
    Default returns pending items only.
    """
    mgr = get_review_manager()
    query: dict = {"status": status}
    if review_type:
        query["review_type"] = review_type
    docs = list(
        mgr.queue_col.find(query)
        .skip(skip)
        .limit(limit)
        .sort("created_at", 1)
    )
    return [_to_str_id(d) for d in docs]


@router.get("/queue/{item_id}", summary="Get a single review item")
async def get_review_item(item_id: str):
    mgr = get_review_manager()
    oid = _parse_oid(item_id)
    doc = mgr.queue_col.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Review item not found.")
    return _to_str_id(doc)


@router.post("/queue/{item_id}/resolve", summary="Approve or reject a review item")
async def resolve_review_item(item_id: str, req: ResolveRequest):
    """
    Resolve a pending review item.

    - Set `approved: true` to accept the suggested correction.
    - Set `approved: false` to reject and keep the original.
    - Optionally provide `override_value` to use a custom correction.
    """
    mgr = get_review_manager()
    return mgr.resolve(
        queue_item_id=item_id,
        approved=req.approved,
        reviewer_note=req.reviewer_note,
        override_value=req.override_value,
    )


@router.get("/names/unvalidated", summary="List unvalidated name memory entries")
async def list_unvalidated_names():
    """Returns all name entries in name_memory that have not been human-validated."""
    mgr = get_review_manager()
    return mgr.name_mgr.get_unvalidated()


@router.post("/names/{canonical}/validate", summary="Validate a canonical name")
async def validate_name(canonical: str):
    """Mark a canonical name entry as human-validated."""
    mgr = get_review_manager()
    mgr.name_mgr.mark_validated(canonical)
    return {"success": True, "canonical": canonical, "validated": True}


@router.post("/names/{canonical}/update", summary="Override a canonical name spelling")
async def update_canonical_name(canonical: str, new_canonical: str):
    """Override the canonical spelling of a name entry."""
    mgr = get_review_manager()
    mgr.name_mgr.update_canonical(canonical, new_canonical)
    return {
        "success": True,
        "old_canonical": canonical,
        "new_canonical": new_canonical,
    }


@router.get("/stats", summary="Review queue statistics")
async def review_stats():
    """Return counts of pending/approved/rejected items by type."""
    mgr = get_review_manager()
    pipeline = [
        {
            "$group": {
                "_id": {"type": "$review_type", "status": "$status"},
                "count": {"$sum": 1},
            }
        }
    ]
    raw = list(mgr.queue_col.aggregate(pipeline))
    stats: dict = {}
    for entry in raw:
        rtype = entry["_id"]["type"]
        status = entry["_id"]["status"]
        stats.setdefault(rtype, {})[status] = entry["count"]
    return stats
