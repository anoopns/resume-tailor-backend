# backend
# Resume & Cover Letter Tailor (Backend)

FastAPI backend that parses master resumes (PDF), extracts job descriptions from raw text or URLs, and generates ATS-optimized Markdown resumes and cover letters using the Google Gemini API.

## Prerequisites

* **Python 3.10+** (required for `bytes | str` union type hints in `main.py`)
* A valid **Google Gemini API Key**

## Installation steps
* deactivate 2>/dev/null || true
* rm -rf venv
* python3 -m venv venv
* source venv/bin/activate
* pip install --upgrade pip
* pip install fastapi uvicorn python-multipart python-dotenv requests beautifulsoup4 pypdf google-genai httpx
* pip freeze > requirements.txt
* Create your **.env** configuration file (.env inside backend/ (backend/.env))
* add GEMINI_API_KEY, GEMINI_MODEL(value set to auto) & GEMINI_AUTO_MODELS(available models) to env

## Project Structure

```text
backend/
├── main.py                   # Main FastAPI application (with caching & model fallback)
├── prompts.py                # System instructions and prompt templates
├── mainShell.py              # Legacy single-endpoint prototype
├── test_model_selection.py   # Unit tests for retry, fallback, and caching
├── .env                      # Environment variables (not committed to Git)
└── cache/                    # Auto-generated JSON cache for PDFs & tailored outputs
