"""
text_corrector.py — Step 3: Text correction pipeline.

Responsibilities:
- Split transcript into word-boundary chunks with overlap
- Call OllamaClient for each chunk
- Merge corrected chunks back into a single document
- Compute aggregate confidence score
- Identify which chunks need human review
"""

import logging
import re
from typing import Optional

import yaml

from ollama_client import OllamaClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def _load_pipeline_cfg(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)["pipeline"]


# ─────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 30,
) -> list[str]:
    """
    Split text into overlapping word-boundary chunks.

    - chunk_size: target words per chunk
    - overlap: words shared between consecutive chunks (preserves context)

    Overlap words are included at the END of chunk N and the START of chunk N+1
    so the LLM has sentence context at boundaries. The combiner trims them.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap   # Step forward, keeping overlap

    logger.debug(f"Chunked text into {len(chunks)} chunk(s) "
                 f"(chunk_size={chunk_size}, overlap={overlap})")
    return chunks


def merge_chunks(corrected_chunks: list[str], overlap: int = 30) -> str:
    """
    Merge corrected chunks back into a single string.
    Trims the overlap prefix from each chunk after the first
    to avoid duplicated sentences.
    """
    if not corrected_chunks:
        return ""
    if len(corrected_chunks) == 1:
        return corrected_chunks[0].strip()

    merged = corrected_chunks[0].strip()
    for chunk in corrected_chunks[1:]:
        words = chunk.split()
        # Drop the first `overlap` words (they were duplicated for context)
        trimmed = " ".join(words[overlap:]) if len(words) > overlap else " ".join(words)
        merged = merged.rstrip() + " " + trimmed.strip()

    return merged.strip()


# ─────────────────────────────────────────────
# Main correction pipeline
# ─────────────────────────────────────────────

class TextCorrectionPipeline:
    """
    Orchestrates chunking → LLM correction → merging for a single transcript.

    Usage:
        pipeline = TextCorrectionPipeline()
        result = pipeline.process(raw_text, known_names)
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = _load_pipeline_cfg(config_path)
        self.chunk_size = cfg["chunk_size_words"]
        self.overlap = cfg["chunk_overlap_words"]
        self.confidence_threshold = cfg["confidence_threshold"]
        self.llm = OllamaClient(config_path)

    def process(
        self,
        raw_text: str,
        known_names: Optional[dict] = None,
        transcript_id: Optional[str] = None,
    ) -> dict:
        """
        Correct a full transcript.

        Returns:
        {
            "cleaned_text":       str,
            "all_detected_names": list[str],
            "all_corrections":    dict,
            "all_flags":          list[str],
            "confidence_score":   float,
            "needs_review":       bool,
            "chunk_results":      list[dict]  (per-chunk debug data)
        }
        """
        label = transcript_id or "unknown"
        logger.info(f"[{label}] Starting text correction — "
                    f"{len(raw_text.split())} words")

        chunks = chunk_text(raw_text, self.chunk_size, self.overlap)
        chunk_results = []
        corrected_texts = []
        all_detected_names: list[str] = []
        all_corrections: dict = {}
        all_flags: list[str] = []
        confidence_scores: list[float] = []

        for i, chunk in enumerate(chunks):
            logger.info(f"[{label}] Processing chunk {i + 1}/{len(chunks)}")
            result = self.llm.correct_chunk(chunk, known_names or {})

            corrected_texts.append(result["cleaned_text"])
            all_detected_names.extend(result["detected_names"])
            all_corrections.update(result["corrected_names"])
            all_flags.extend(result["uncertain_flags"])
            confidence_scores.append(result["confidence_score"])
            chunk_results.append({
                "chunk_index": i,
                "original": chunk,
                **result,
            })

        # Aggregate
        merged_text = merge_chunks(corrected_texts, self.overlap)
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores else 0.0
        )
        needs_review = (
            avg_confidence < self.confidence_threshold
            or len(all_flags) > 0
        )

        logger.info(
            f"[{label}] Correction complete — "
            f"confidence={avg_confidence:.2f}, "
            f"needs_review={needs_review}, "
            f"flags={len(all_flags)}"
        )

        return {
            "cleaned_text": merged_text,
            "all_detected_names": list(set(all_detected_names)),
            "all_corrections": all_corrections,
            "all_flags": all_flags,
            "confidence_score": round(avg_confidence, 4),
            "needs_review": needs_review,
            "chunk_results": chunk_results,
        }
