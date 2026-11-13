import os
import io
import logging
from pathlib import Path

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pypdf import PdfReader
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env file")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
client = genai.Client(api_key=api_key)


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


@app.post("/api/tailor-resume")
async def tailor_resume(
        resume_file: UploadFile = File(...),
        jd_source: str = Form(...)
):
    try:
        # 1. Read PDF file bytes from React upload
        pdf_bytes = await resume_file.read()
        master_resume_text = extract_text_from_pdf_bytes(pdf_bytes)

        if not master_resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from uploaded PDF.")

        # 2. Extract or fetch Job Description text
        jd_text = fetch_job_description(jd_source)

        # 3. System Instruction Definition
        system_instruction = """You are an executive resume writer, ATS strategist, and recruiter.
STRICT RULES:
1. NEVER invent companies, job titles, dates, certifications, tools, or metrics not explicitly present in the master resume.
2. Format experience bullet points strictly using the STAR framework (Action Verb + Context + Quantifiable Result).
3. Align keywords and technical terms from the Job Description only where factually supported by the master resume.
4. Output valid, clean Markdown only. Do not wrap the response in conversational text.
"""

        # 4. User Prompt Definition
        user_prompt = f"""MASTER RESUME:
---
{master_resume_text}
---

TARGET JOB DESCRIPTION:
---
{jd_text}
---

INSTRUCTIONS:
Generate a tailored Markdown resume optimized for the target job description. Include:
1. Name & Contact Info (top header)
2. Professional Summary (3 targeted sentences highlighting alignment)
3. Tailored Experience (STAR-format bullet points using keywords from JD where factually accurate)
4. Technical Skills (categorized by relevance to JD)
5. Education & Certifications
"""

        # 5. Call Gemini API
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )

        return {"markdown": response.text}

    except Exception as e:
        logging.exception(
            "Resume tailoring request failed. resume_file_name=%s jd_source_length=%s",
            getattr(resume_file, "filename", None),
            len(jd_source) if jd_source else 0,
        )
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)