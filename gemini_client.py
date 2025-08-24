import os, time, json, re, inspect, tempfile
from typing import Any, Dict, List, Optional, Tuple

from log_util import logger
from google import genai
from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

_client: Optional[genai.Client] = None

def initialize_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """
    New SDK: construct a Client instead of calling genai.configure().
    Reads GEMINI_API_KEY if api_key not provided.
    """
    global _client
    if _client is not None:
        return _client

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set and no api_key was provided.")

    _client = genai.Client(api_key=key)
    return _client

def build_request(custom_id: str, file_name: str, diff_text: str) -> dict:
    prompt = f"""
You are a senior code reviewer. Analyze the unified diff below and respond with ONLY a JSON object (no markdown, no code fences, no extra text).

Rules:
- Top-level keys: "summary" (<= 80 words), "impact" (<= 80 words), "suggestions" (array).
- Each suggestion MUST have: "file" (string), "line" (integer or null), "comment" (string).
- Optional per-suggestion: "severity" one of ["info","warning","critical"].
- Max 6 suggestions. Use [] if none are warranted.

Diff to review:
```diff
{diff_text}
```"""
    return {
        "custom_id": custom_id,
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }

# ---------- Batch helpers ----------

def _supports_inline_requests(client: genai.Client) -> bool:
    """Best-effort detection: does client.batches.create accept 'requests' kw?"""
    try:
        sig = inspect.signature(client.batches.create)  # type: ignore[attr-defined]
        return "requests" in sig.parameters
    except Exception:
        # If we can't introspect, we’ll still try and catch at runtime.
        return True

def _try_inline_batch(
    client: genai.Client, requests: List[dict], display_name: str
) -> Tuple[Optional[Any], Optional[Exception]]:
    try:
        job = client.batches.create(  # type: ignore[call-arg]
            model=MODEL_NAME,
            requests=requests,
            display_name=display_name,
        )
        logger.info(f"[batches] inline create ok: {getattr(job, 'name', '<no name>')}")
        return job, None
    except TypeError as e:
        # Classic symptom: "unexpected keyword argument 'requests'"
        logger.warning(f"[batches] inline not supported on this SDK: {e}")
        return None, e
    except Exception as e:
        logger.error(f"[batches] inline create failed: {e}")
        return None, e

def submit_batch_job(
    client: genai.Client,
    requests: List[dict],
    display_name: str = "pr-review-job",
):
    """
    Create a batch job if supported. Fallbacks:
      1) Inline 'requests=' when available.
      2) If env GEMINI_BATCH_INPUT_URI is set, use file-based batch (input_uri).
      3) Final fallback: run per-request loop (no batch), returning a pseudo 'job' dict.
    """
    logger.info(f"Submitting Batch job with {len(requests)} requests.")

    # (1) Try inline requests when available (or attempt & catch TypeError)
    if _supports_inline_requests(client):
        job, err = _try_inline_batch(client, requests, display_name)
        if job is not None:
            return job

    # (2) File-based batch (only if you provide an input URI)
    # Supply a GCS URI or public URL via env to enable this path.
    input_uri = os.getenv("GEMINI_BATCH_INPUT_URI")  # e.g. gs://bucket/reqs.jsonl
    if input_uri:
        try:
            job = client.batches.create(  # type: ignore[call-arg]
                model=MODEL_NAME,
                input_uri=input_uri,
                display_name=display_name,
            )
            logger.info(f"[batches] file-based create ok: {getattr(job, 'name', '<no name>')}")
            return job
        except Exception as e:
            logger.error(f"[batches] file-based create failed: {e}")

    # (3) Fallback: no batch—process each request in a loop and return a pseudo-job
    logger.warning("[batches] falling back to per-request loop (no batch API).")
    results = generate_many(client, requests)
    # Return a light pseudo object so calling code can be uniform-ish
    return {"_kind": "pseudo-batch", "results": results, "state": "JOB_STATE_SUCCEEDED"}

def wait_for_batch_and_collect(
    client: genai.Client,
    job_or_name: Any,
    timeout_s: int = 900,
    poll_s: int = 5,
) -> List[Dict[str, Any]]:
    """
    Collect results from:
      - real inline batch → job.inlined_responses
      - file-based batch → not handled (depends on output URIs; you can extend)
      - pseudo batch → already has results
    """
    # Pseudo-job path
    if isinstance(job_or_name, dict) and job_or_name.get("_kind") == "pseudo-batch":
        return job_or_name["results"]

    # Resolve job object (name or object)
    if isinstance(job_or_name, str):
        job_name = job_or_name
    else:
        job_name = getattr(job_or_name, "name", None)
    if not job_name:
        logger.error("wait_for_batch_and_collect: invalid job handle")
        return []

    start = time.time()
    while True:
        job = client.batches.get(name=job_name)
        state = getattr(job, "state", None)
        state_name = getattr(state, "name", str(state))
        logger.info(f"[Batch] {job_name} state={state_name}")

        if state_name in {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }:
            break

        if time.time() - start > timeout_s:
            logger.error(f"Timed out waiting for batch job after {timeout_s}s")
            return []

        time.sleep(poll_s)

    if state_name != "JOB_STATE_SUCCEEDED":
        logger.error(f"Batch finished but not successful: {state_name}")
        return []

    # Inline batch results available directly
    if getattr(job, "inlined_responses", None):
        results: List[Dict[str, Any]] = []
        for item in job.inlined_responses:
            try:
                text = ""
                if getattr(item, "response", None) and item.response.candidates:
                    part = item.response.candidates[0].content.parts[0]
                    text = getattr(part, "text", "") or ""
                results.append(_parse_json(text))
            except Exception as e:
                logger.error(f"Failed to process batch result item: {e}")
                results.append({"summary": "", "impact": "", "suggestions": []})
        return results

    # File-based batch: you’ll typically read from job.output_* fields (URIs).
    # You can extend this branch if your SDK version returns output_uri(s).
    logger.warning("Batch succeeded but no inlined_responses were found. "
                   "If you used input_uri, fetch outputs from the job's output URIs.")
    return []

# ---------- Non-batch fallback ----------

def generate_many(client: genai.Client, requests: List[dict]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for i, req in enumerate(requests, 1):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=req["contents"],
            )
            text = ""
            if resp and resp.candidates and resp.candidates[0].content.parts:
                part = resp.candidates[0].content.parts[0]
                text = getattr(part, "text", "") or ""
            parsed = _parse_json(text)
            results.append({
                "custom_id": req.get("custom_id"),
                "response": parsed or {"summary": "", "impact": "", "suggestions": []},
            })
        except Exception as e:
            logger.error(f"request {i} failed: {e}")
            results.append({
                "custom_id": req.get("custom_id"),
                "response": {"summary": "", "impact": "", "suggestions": []},
            })
    return results

# ---------- JSON parsing ----------

def _parse_json(raw: str) -> Dict[str, Any]:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}. Raw text: {raw[:200]}...")
        return {"summary": "", "impact": "", "suggestions": []}
    
