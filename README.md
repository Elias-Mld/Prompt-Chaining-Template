# Prompt Chaining

A local Python project that runs a multi-step Gemini workflow to turn raw notes
into a polished final deliverable (for example, a SaaS proposal).

## Overview

This project provides:

- Four prompt templates:
  - `4C Engine`
  - `SaaS Proposal Engine`
  - `TikTok Video Script Engine`
  - `Newsletter Engine`
- Four input blocks (text, with optional supporting documents)
- A five-call pipeline:
  - Calls 1-4: context framing, analytical extraction, generation, and review
  - Call 5: final synthesis and refinement
- Follow-up Q&A on the generated output
- PDF export for the final proposal

## Included 5-Step Templates

- `4C Engine`: general-purpose strategic and technical synthesis
- `SaaS Proposal Engine`: high-converting SaaS commercial proposal generation
- `TikTok Video Script Engine`: retention-focused TikTok script creation with hook, shot list, and CTA options
- `Newsletter Engine`: editorial newsletter generation with subject lines, preview text, and publication-ready body

## Tech Stack

- `streamlit` for the web interface
- `google-generativeai` for Gemini API access
- `python-dotenv` for loading `GEMINI_API_KEY` from `.env`
- `pypdf` for reading uploaded PDF files
- `reportlab` for PDF generation

## Installation

```bash
cd "/Users/eliasmld/Desktop/OPTI Prompt"
python3 -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

## Run the App

```bash
python3 -m streamlit run app.py
```

## Key Files

- `app.py`: Streamlit interface (templates, pipeline, Q&A, PDF export)
- `prompt_chaining_gemini.py`: command-line pipeline runner
- `requirements.txt`: project dependencies
