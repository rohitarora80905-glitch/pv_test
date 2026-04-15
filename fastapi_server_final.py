"""
fastapi_server.py — Main FastAPI application (fully integrated).

Mounts:
  /health, /transcripts/*        → core transcript ingestion & status
  /review/*                      → human-in-the-loop review API
  /pipeline/*                    → pipeline trigger & status
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bson import ObjectId
from bson.errors import InvalidId

from mongo import get_db, get_collection, transcript_schema, load_config
from review_api import router as review_router
from pipeline import pipeline_router

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def setup_logging(cfg: dict):
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("log_file", "output/pipeline.log")
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(log_file))
    except FileNotFoundError:
        pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )

logger = logging.getLogger(__name__)
cfg = load_config()
setup_logging(cfg)


# ─────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Clinical Pipeline API…")
    try:
        db = get_db()
        app.state.db = db
        logger.info("MongoDB connection established.")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
    yield
    logger.info("Shutting down Clinical Pipeline API.")


app = FastAPI(
    title="Clinical Transcription Pipeline",
    description=(
        "Production pipeline for correcting Whisper medical transcripts "
        "using a local LLM (GPT-OSS-20B via Ollama). "
        "Includes human-in-the-loop review and persistent name memory."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sub-routers
app.include_router(review_router)
app.include_router(pipeline_router)


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def to_str_id(doc: dict) -> dict:
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


def parse_object_id(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except (InvalidId, Exception):
        raise HTTPException(status_code=400, detail=f"Invalid ID: {id_str}")


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class IngestRequest(BaseModel):
    raw_text: str = Field(..., min_length=1)
    patient_id: Optional[str] = None
    source: str = "whisper"
    metadata: Optional[dict] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    success: bool
    transcript_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    mongodb: str
    api_version: str


# ─────────────────────────────────────────────
# Core routes
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    mongo_status = "connected"
    try:
        get_db().command("ping")
    except Exception as e:
        logger.warning(f"MongoDB health check failed: {e}")
        mongo_status = "error"
    return HealthResponse(
        status="ok" if mongo_status == "connected" else "degraded",
        mongodb=mongo_status,
        api_version=app.version,
    )


@app.post("/transcripts/ingest", response_model=IngestResponse, tags=["Transcripts"])
async def ingest_transcript(req: IngestRequest):
    """Insert a raw Whisper transcript. Sets status=pending for pipeline processing."""
    col = get_collection("transcripts")
    doc = transcript_schema(
        raw_text=req.raw_text,
        patient_id=req.patient_id,
        source=req.source,
        metadata=req.metadata,
    )
    result = col.insert_one(doc)
    tid = str(result.inserted_id)
    logger.info(f"Ingested transcript {tid}")
    return IngestResponse(
        success=True,
        transcript_id=tid,
        message="Ingested. Status: pending.",
    )


@app.get("/transcripts/{transcript_id}", tags=["Transcripts"])
async def get_transcript(transcript_id: str):
    col = get_collection("transcripts")
    doc = col.find_one({"_id": parse_object_id(transcript_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found.")
    return to_str_id(doc)


@app.get("/transcripts/{transcript_id}/result", tags=["Transcripts"])
async def get_corrected_result(transcript_id: str):
    """Fetch the corrected output for a given transcript ID."""
    col = get_collection("corrected")
    doc = col.find_one({"transcript_id": transcript_id})
    if not doc:
        raise HTTPException(status_code=404, detail="No corrected result yet.")
    return to_str_id(doc)


@app.get("/transcripts", tags=["Transcripts"])
async def list_transcripts(
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
):
    col = get_collection("transcripts")
    query = {"status": status} if status else {}
    docs = list(col.find(query).skip(skip).limit(limit).sort("timestamp", 1))
    return [to_str_id(d) for d in docs]


@app.patch("/transcripts/{transcript_id}/status", tags=["Transcripts"])
async def update_transcript_status(transcript_id: str, status: str):
    valid = {"pending", "processing", "done", "error"}
    if status not in valid:
        raise HTTPException(status_code=400, detail=f"Status must be one of: {valid}")
    col = get_collection("transcripts")
    result = col.update_one(
        {"_id": parse_object_id(transcript_id)},
        {"$set": {"status": status}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Not found.")
    return {"success": True, "transcript_id": transcript_id, "new_status": status}


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    server_cfg = cfg.get("server", {})
    uvicorn.run(
        "fastapi_server:app",
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 8000),
        reload=server_cfg.get("reload", False),
    )
