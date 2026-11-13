import os
import io
import json
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pypdf import PdfReader
from bs4 import BeautifulSoup
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from prompts import (
    COVER_LETTER_SYSTEM_INSTRUCTION,
    COVER_LETTER_USER_PROMPT,
    RESUME_SYSTEM_INSTRUCTION,
    RESUME_USER_PROMPT,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

# Prompt edits change these hashes, so stale cached outputs are not reused.
PROMPT_HASHES = {
    "resume": hashlib.sha256((RESUME_SYSTEM_INSTRUCTION + RESUME_USER_PROMPT).encode("utf-8")).hexdigest(),
    "cover-letter": hashlib.sha256(
        (COVER_LETTER_SYSTEM_INSTRUCTION + COVER_LETTER_USER_PROMPT).encode("utf-8")
    ).hexdigest(),
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):(5173|5174|5175|3000)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env file")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "auto").strip()
GEMINI_AUTO_MODELS = tuple(dict.fromkeys(
    model.strip()
    for model in os.getenv("GEMINI_AUTO_MODELS", "gemini-3.8-flash,gemini-3.6-flash").split(",")
    if model.strip()
))
if not GEMINI_MODEL or (GEMINI_MODEL == "auto" and (
    not GEMINI_AUTO_MODELS or "auto" in GEMINI_AUTO_MODELS
)):
    raise RuntimeError("Set GEMINI_MODEL to a model ID or auto with a non-empty list of model IDs in GEMINI_AUTO_MODELS.")
client = genai.Client(api_key=api_key)
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def sha256_digest(data: bytes | str) -> str:
    if isinstance(data, str):
        payload = data.encode("utf-8")
    else:
        payload = data
    return hashlib.sha256(payload).hexdigest()


def load_json_cache(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def save_json_cache(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def get_resume_cache_file(pdf_hash: str) -> Path:
    return CACHE_DIR / f"resume_{pdf_hash}.json"


def get_tailored_output_cache_file(pdf_hash: str, jd_hash: str, additional_context: str, output_type: str) -> Path:
    context_hash = sha256_digest(additional_context or "")
    prompt_hash = PROMPT_HASHES[output_type][:12]
    return CACHE_DIR / f"tailored_{output_type}_{pdf_hash}_{jd_hash}_{context_hash}_{prompt_hash}.json"


def get_or_parse_resume(pdf_bytes: bytes) -> tuple[str, bool]:
    pdf_hash = sha256_digest(pdf_bytes)
    cache_file = get_resume_cache_file(pdf_hash)
    cached_data = load_json_cache(cache_file)
    if cached_data and cached_data.get("parsed_text"):
        return cached_data["parsed_text"], True

    parsed_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not parsed_text:
        raise HTTPException(status_code=400, detail="Could not extract text from uploaded PDF.")

    save_json_cache(
        cache_file,
        {
            "pdf_hash": pdf_hash,
            "parsed_text": parsed_text,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return parsed_text, False


def generate_with_retry(model: str, contents: str, system_instruction: str):
    models = GEMINI_AUTO_MODELS if model == "auto" else (model,)
    for model_index, candidate in enumerate(models):
        for attempt in range(3):
            try:
                return client.models.generate_content(
                    model=candidate,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                    ),
                )
            except genai_errors.APIError as error:
                if error.code != 503:
                    raise

                if attempt == 2:
                    if model_index == len(models) - 1:
                        raise
                    logging.warning(
                        "Gemini model %s remains busy after 3 attempts; falling back to %s.",
                        candidate,
                        models[model_index + 1],
                    )
                    break

                wait_seconds = 2 ** attempt
                logging.warning(
                    "Gemini model %s busy (attempt %s/3), retrying in %ss.",
                    candidate,
                    attempt + 1,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

    raise RuntimeError("No Gemini models configured for generation.")


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extracts text layer directly from in-memory PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def fetch_job_description(jd_input: str) -> str:
    """Parses JD string — extracts clean text if input is a URL."""
    if jd_input.startswith("http://") or jd_input.startswith("https://"):
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        response = requests.get(jd_input, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        return "\n".join(chunk for chunk in lines if chunk)
    return jd_input.strip()


def build_target_context(jd_source: str, additional_context: str) -> str:
    jd_text = fetch_job_description(jd_source).strip() if jd_source and jd_source.strip() else ""
    extra_text = (additional_context or "").strip()

    if jd_text and extra_text:
        return f"TARGET JOB DESCRIPTION:\n---\n{jd_text}\n---\n\nADDITIONAL CONTEXT:\n---\n{extra_text}\n---"
    if jd_text:
        return f"TARGET JOB DESCRIPTION:\n---\n{jd_text}\n---"
    if extra_text:
        return f"TARGET JOB DESCRIPTION / ADDITIONAL CONTEXT:\n---\n{extra_text}\n---"
    return ""


def get_cached_tailored_result(pdf_hash: str, jd_text: str, additional_context: str, output_type: str) -> str | None:
    jd_hash = sha256_digest(jd_text)
    cache_file = get_tailored_output_cache_file(pdf_hash, jd_hash, additional_context, output_type)
    cached_data = load_json_cache(cache_file)
    if cached_data and cached_data.get("markdown"):
        return cached_data["markdown"]
    return None


def save_tailored_result(pdf_hash: str, jd_text: str, additional_context: str, markdown: str, output_type: str):
    jd_hash = sha256_digest(jd_text)
    cache_file = get_tailored_output_cache_file(pdf_hash, jd_hash, additional_context, output_type)
    save_json_cache(
        cache_file,
        {
            "output_type": output_type,
            "pdf_hash": pdf_hash,
            "jd_hash": jd_hash,
            "additional_context_hash": sha256_digest(additional_context or ""),
            "markdown": markdown,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.post("/api/tailor-resume")
async def tailor_resume(
        resume_file: UploadFile = File(...),
        jd_source: str = Form(""),
        additional_context: str = Form("")
):
    try:
        pdf_bytes = await resume_file.read()
        pdf_hash = sha256_digest(pdf_bytes)
        master_resume_text, _ = get_or_parse_resume(pdf_bytes)

        target_context = build_target_context(jd_source, additional_context)
        if not target_context:
            raise HTTPException(status_code=400, detail="Please provide a job description or additional context.")

        jd_text = fetch_job_description(jd_source).strip() if jd_source and jd_source.strip() else ""
        cached_markdown = get_cached_tailored_result(pdf_hash, jd_text, additional_context or "", "resume")
        if cached_markdown:
            return {"markdown": cached_markdown, "cached": True}

        response = generate_with_retry(
            model=GEMINI_MODEL,
            contents=RESUME_USER_PROMPT.format(
                master_resume_text=master_resume_text,
                target_context=target_context,
            ),
            system_instruction=RESUME_SYSTEM_INSTRUCTION,
        )

        save_tailored_result(pdf_hash, jd_text, additional_context or "", response.text, "resume")
        return {"markdown": response.text, "cached": False}

    except HTTPException:
        raise
    except Exception as e:
        logging.exception(
            "Resume tailoring request failed. resume_file_name=%s jd_source_length=%s additional_context_length=%s",
            getattr(resume_file, "filename", None),
            len(jd_source) if jd_source else 0,
            len(additional_context) if additional_context else 0,
        )
        status_code = 503 if isinstance(e, genai_errors.APIError) and e.code == 503 else 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e


@app.post("/api/tailor-cover-letter")
async def tailor_cover_letter(
        resume_file: UploadFile = File(...),
        jd_source: str = Form(""),
        additional_context: str = Form("")
):
    try:
        pdf_bytes = await resume_file.read()
        pdf_hash = sha256_digest(pdf_bytes)
        master_resume_text, _ = get_or_parse_resume(pdf_bytes)

        target_context = build_target_context(jd_source, additional_context)
        if not target_context:
            raise HTTPException(status_code=400, detail="Please provide a job description or additional context.")

        jd_text = fetch_job_description(jd_source).strip() if jd_source and jd_source.strip() else ""
        cached_markdown = get_cached_tailored_result(pdf_hash, jd_text, additional_context or "", "cover-letter")
        if cached_markdown:
            return {"markdown": cached_markdown, "cached": True}

        response = generate_with_retry(
            model=GEMINI_MODEL,
            contents=COVER_LETTER_USER_PROMPT.format(
                master_resume_text=master_resume_text,
                target_context=target_context,
            ),
            system_instruction=COVER_LETTER_SYSTEM_INSTRUCTION,
        )

        save_tailored_result(pdf_hash, jd_text, additional_context or "", response.text, "cover-letter")
        return {"markdown": response.text, "cached": False}

    except HTTPException:
        raise
    except Exception as e:
        logging.exception(
            "Cover letter request failed. resume_file_name=%s jd_source_length=%s additional_context_length=%s",
            getattr(resume_file, "filename", None),
            len(jd_source) if jd_source else 0,
            len(additional_context) if additional_context else 0,
        )
        status_code = 503 if isinstance(e, genai_errors.APIError) and e.code == 503 else 500
        raise HTTPException(status_code=status_code, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)