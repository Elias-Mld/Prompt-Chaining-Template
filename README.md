# Prompt Chaining

A local Python application that orchestrates a multi-step AI pipeline using Gemini to transform raw notes into a final deliverable (e.g., a SaaS proposal).

## Features

- Offers built-in templates (e.g., `4C Engine`, `SaaS Proposal Engine`).
- Takes 4 input blocks (text + an optional document for each block).
- Executes a 5-step pipeline:
  - Steps 1 to 4: Analysis / Structuring / Generation / Audit
  - Step 5: Final synthesis and polishing
- Allows asking follow-up questions on the final output.
- Supports exporting the final proposal as a PDF.

## Tech Stack

- `streamlit` (Web interface)
- `google-generativeai` (Gemini API)
- `python-dotenv` (Environment variables / API key loading)
- `pypdf` (Reading imported PDFs)
- `reportlab` (PDF export)

## Installation

```bash
cd "/Users/eliasmld/Desktop/OPTI Prompt"
python3 -m pip install -r requirements.txt

## Lancement

```bash
python3 -m streamlit run app.py
```

## Fichiers principaux

- `app.py`: application Streamlit principale (templates, pipeline, Q&A, export PDF)
- `prompt_chaining_gemini.py`: version CLI du pipeline
- `requirements.txt`: dependances Python
