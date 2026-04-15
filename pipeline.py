"""
pipeline.py — Step 6: Full pipeline orchestration.

This is the main runner. It:
1. Fetches pending transcripts from MongoDB
2. Marks them as 'processing'
3. Loads name memory for prompt injection
4. Runs text correction (chunked LLM calls)
5. Resolves detected names via name memory
6. Saves corrected output to MongoDB
7. Enqueues items for human review if needed
8. Marks transcript as 'done' or 'error'

Can be run:
  - As a CLI batch job:    python pipeline.py
  - As a triggered runner: called from FastAPI /pipeline/run endpoint
  - As a loop daemon:      python pipeline.py --watch
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId

from mongo import (
    get_collection,
    corrected_transcript_schema,
    load_config,
)
from text_corrector import TextCorrectionPipeline
from name_memory import NameMemoryManager
from review_api import ReviewQueueManager

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def _setup_logging(cfg: dict):
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("log_file", "output/pipeline.log")
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(log_file))
    except Exception:
        pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────

class ClinicalPipeline:
    """
    Main pipeline orchestrator.

    Usage:
        pipeline = ClinicalPipeline()
        stats = pipeline.run_batch()
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.batch_size = self.cfg["pipeline"]["batch_size"]
        self.confidence_threshold = self.cfg["pipeline"]["confidence_threshold"]

        self.transcripts_col = get_collection("transcripts", config_path)
        self.corrected_col = get_collection("corrected", config_path)

        self.corrector = TextCorrectionPipeline(config_path)
        self.name_mgr = NameMemoryManager(config_path)
        self.review_mgr = ReviewQueueManager(config_path)

        logger.info("ClinicalPipeline initialised.")

    # ── public API ────────────────────────────

    def run_batch(self) -> dict:
        """
        Process one batch of pending transcripts.

        Returns stats dict:
        {
            "processed": int,
            "succeeded": int,
            "failed": int,
            "queued_for_review": int,
        }
        """
        pending = self._fetch_pending(self.batch_size)
        if not pending:
            logger.info("No pending transcripts — batch complete.")
            return {"processed": 0, "succeeded": 0, "failed": 0, "queued_for_review": 0}

        logger.info(f"Processing batch of {len(pending)} transcript(s)…")
        stats = {"processed": 0, "succeeded": 0, "failed": 0, "queued_for_review": 0}

        for doc in pending:
            transcript_id = str(doc["_id"])
            try:
                queued = self._process_one(doc)
                stats["succeeded"] += 1
                if queued:
                    stats["queued_for_review"] += 1
            except Exception as e:
                logger.error(f"[{transcript_id}] Pipeline error: {e}", exc_info=True)
                self._mark_status(transcript_id, "error")
                stats["failed"] += 1
            stats["processed"] += 1

        logger.info(
            f"Batch complete — processed={stats['processed']}, "
            f"succeeded={stats['succeeded']}, failed={stats['failed']}, "
            f"queued_for_review={stats['queued_for_review']}"
        )
        return stats

    def process_single(self, transcript_id: str) -> dict:
        """Process a specific transcript by its MongoDB ID."""
        doc = self.transcripts_col.find_one({"_id": ObjectId(transcript_id)})
        if not doc:
            raise ValueError(f"Transcript {transcript_id} not found.")
        queued = self._process_one(doc)
        return {
            "transcript_id": transcript_id,
            "success": True,
            "queued_for_review": queued,
        }

    # ── core processing logic ─────────────────

    def _process_one(self, doc: dict) -> bool:
        """
        Process a single transcript document end-to-end.
        Returns True if the output was queued for human review.
        """
        transcript_id = str(doc["_id"])
        raw_text = doc.get("raw_text", "")

        if not raw_text.strip():
            logger.warning(f"[{transcript_id}] Empty raw_text — skipping.")
            self._mark_status(transcript_id, "done")
            return False

        logger.info(f"[{transcript_id}] Starting — {len(raw_text.split())} words")

        # Mark as in-progress
        self._mark_status(transcript_id, "processing")

        # ── Step A: Fetch name memory for prompt context ──
        known_names = self.name_mgr.get_names_for_prompt()

        # ── Step B: LLM text correction ───────────────────
        correction = self.corrector.process(
            raw_text=raw_text,
            known_names=known_names,
            transcript_id=transcript_id,
        )

        # ── Step C: Resolve detected names via memory ─────
        name_resolutions = self.name_mgr.resolve_names(
            correction["all_detected_names"]
        )

        # Merge LLM-suggested corrections with memory resolutions
        merged_name_corrections = {
            **correction["all_corrections"],
            **name_resolutions,
        }

        # ── Step D: Persist corrected output ─────────────
        needs_review = correction["needs_review"]
        corrected_doc = corrected_transcript_schema(
            transcript_id=transcript_id,
            cleaned_text=correction["cleaned_text"],
            name_corrections=merged_name_corrections,
            uncertain_flags=correction["all_flags"],
            confidence_score=correction["confidence_score"],
            needs_review=needs_review,
        )
        result = self.corrected_col.insert_one(corrected_doc)
        corrected_id = str(result.inserted_id)
        logger.info(f"[{transcript_id}] Saved corrected doc: {corrected_id}")

        # ── Step E: Enqueue for human review if needed ────
        if needs_review:
            self._enqueue_reviews(
                transcript_id=transcript_id,
                corrected_id=corrected_id,
                raw_text=raw_text,
                cleaned_text=correction["cleaned_text"],
                flags=correction["all_flags"],
                name_corrections=merged_name_corrections,
            )

        # ── Step F: Mark transcript done ──────────────────
        self._mark_status(transcript_id, "done")
        logger.info(
            f"[{transcript_id}] Done — "
            f"confidence={correction['confidence_score']:.2f}, "
            f"review={'yes' if needs_review else 'no'}"
        )
        return needs_review

    def _enqueue_reviews(
        self,
        transcript_id: str,
        corrected_id: str,
        raw_text: str,
        cleaned_text: str,
        flags: list,
        name_corrections: dict,
    ):
        """Enqueue text + name review items."""
        # Full text review
        self.review_mgr.enqueue_text_review(
            transcript_id=transcript_id,
            corrected_id=corrected_id,
            original_text=raw_text,
            cleaned_text=cleaned_text,
            flags=flags,
        )

        # Individual name reviews (only for names that actually changed)
        for original, corrected in name_corrections.items():
            if original != corrected:
                self.review_mgr.enqueue_name_review(
                    transcript_id=transcript_id,
                    corrected_id=corrected_id,
                    original_name=original,
                    suggested_name=corrected,
                )

    # ── utilities ─────────────────────────────

    def _fetch_pending(self, limit: int) -> list[dict]:
        return list(
            self.transcripts_col.find({"status": "pending"})
            .limit(limit)
            .sort("timestamp", 1)
        )

    def _mark_status(self, transcript_id: str, status: str):
        self.transcripts_col.update_one(
            {"_id": ObjectId(transcript_id)},
            {"$set": {"status": status}},
        )


# ─────────────────────────────────────────────
# FastAPI trigger endpoint (imported in fastapi_server.py)
# ─────────────────────────────────────────────

from fastapi import APIRouter

pipeline_router = APIRouter(prefix="/pipeline", tags=["Pipeline"])
_pipeline_instance: Optional[ClinicalPipeline] = None


def get_pipeline() -> ClinicalPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = ClinicalPipeline()
    return _pipeline_instance


@pipeline_router.post("/run", summary="Trigger a batch pipeline run")
async def trigger_pipeline_run():
    """
    Trigger the pipeline to process the next batch of pending transcripts.
    Returns processing statistics.
    """
    pl = get_pipeline()
    stats = pl.run_batch()
    return {"success": True, "stats": stats}


@pipeline_router.post("/run/{transcript_id}", summary="Process a specific transcript")
async def trigger_single(transcript_id: str):
    """Force-process a specific transcript by its MongoDB ID."""
    pl = get_pipeline()
    return pl.process_single(transcript_id)


@pipeline_router.get("/status", summary="Pipeline configuration and status")
async def pipeline_status():
    """Return current pipeline configuration."""
    pl = get_pipeline()
    pending_count = pl.transcripts_col.count_documents({"status": "pending"})
    processing_count = pl.transcripts_col.count_documents({"status": "processing"})
    done_count = pl.transcripts_col.count_documents({"status": "done"})
    error_count = pl.transcripts_col.count_documents({"status": "error"})
    review_pending = pl.review_mgr.queue_col.count_documents({"status": "pending"})
    return {
        "transcripts": {
            "pending": pending_count,
            "processing": processing_count,
            "done": done_count,
            "error": error_count,
        },
        "review_queue_pending": review_pending,
        "config": {
            "batch_size": pl.batch_size,
            "confidence_threshold": pl.confidence_threshold,
            "ollama_model": pl.cfg["ollama"]["model"],
            "chunk_size_words": pl.cfg["pipeline"]["chunk_size_words"],
        },
    }


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Transcription Pipeline")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously, polling every 30 seconds",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Process a single transcript by MongoDB ID",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    _setup_logging(cfg)

    pl = ClinicalPipeline(config_path=args.config)

    if args.id:
        logger.info(f"Processing single transcript: {args.id}")
        result = pl.process_single(args.id)
        print(result)

    elif args.watch:
        logger.info("Watch mode active — polling every 30s…")
        while True:
            try:
                pl.run_batch()
            except KeyboardInterrupt:
                logger.info("Pipeline stopped by user.")
                break
            except Exception as e:
                logger.error(f"Batch error: {e}", exc_info=True)
            time.sleep(30)

    else:
        # Single batch run
        stats = pl.run_batch()
        print(stats)
