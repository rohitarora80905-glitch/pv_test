from fastapi import FastAPI
import requests
import json
import re

app = FastAPI()

OLLAMA_URL = "http://34.158.237.223:11434/api/generate"

@app.get("/test")
def test_pipeline():
    raw_text = "patient jon was givn paracetamol 500 mg yestarday he say he feel dizy"

    prompt = f"""
You are an expert medical transcription correction assistant.

Your task is to clean and correct the given transcript while STRICTLY preserving its original meaning.

-----------------------
CRITICAL RULES (MUST FOLLOW)
-----------------------

1. DO NOT change or hallucinate medical information.
2. DO NOT change medication names, dosages, or numerical values unless clearly misspelled.
3. If unsure about a correction, DO NOT guess — mark it as uncertain.
4. Preserve the original tense and meaning exactly.
5. Do NOT add new information.

-----------------------
TEXT CORRECTION RULES
-----------------------

- Fix grammar, spelling, and sentence structure.
- Convert broken phrases into clear, natural sentences.
- Maintain clinical tone.
- Keep all important details intact.

-----------------------
NAME HANDLING RULES
-----------------------

- Detect all person names (patients, doctors, nurses).
- Correct obvious spelling mistakes in names (e.g., "jon" → "John").
- Use phonetic reasoning for similar names.
- If unsure about a name:
  → keep original
  → add to "uncertain_flags"

-----------------------
MEDICATION SAFETY RULES
-----------------------

- NEVER incorrectly modify:
  - drug names
  - dosages (e.g., 500 mg, 10 units)
- If a drug name seems slightly misspelled:
  → correct only if HIGH confidence
  → otherwise flag as uncertain

-----------------------
OUTPUT FORMAT (STRICT JSON ONLY)
-----------------------

Return ONLY valid JSON. No extra text.

{{
  "cleaned_text": "...",
  "detected_names": ["..."],
  "corrected_names": {{
    "original_name": "corrected_name"
  }},
  "uncertain_flags": ["..."]
}}

-----------------------
INPUT TEXT
-----------------------

{raw_text}
"""

    # ── Call Ollama ──────────────────────────────────────────────────────────
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": "gpt-oss:20b", "prompt": prompt, "stream": False},
            timeout=600,                      # prevent hanging indefinitely
        )
        response.raise_for_status()          # surface HTTP errors early
    except requests.RequestException as e:
        return {"error": "Ollama request failed", "detail": str(e)}

    raw_output = response.json().get("response", "")
    print("RAW LLM OUTPUT:\n", raw_output)  # debug

    # ── Parse JSON from model output ─────────────────────────────────────────
    try:
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON block found in model output")
        parsed = json.loads(json_match.group())
    except Exception as e:
        parsed = {
            "error": "Invalid JSON from model",
            "detail": str(e),
            "raw_output": raw_output,
        }

    return parsed                            # ← must be inside the function
