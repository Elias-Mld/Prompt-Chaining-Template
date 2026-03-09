# Prompt Chaining

Application Python locale qui orchestre un pipeline IA multi-etapes avec Gemini pour transformer des notes brutes en livrable final (ex: proposition SaaS).

## Ce que fait le projet

- Propose des templates (ex: `4C Engine`, `SaaS Proposal Engine`)
- Prend 4 blocs d'entree (texte + document optionnel par bloc)
- Execute un pipeline en 5 appels :
  - Steps 1 a 4: analyse/structuration/generation/audit
  - Step 5: synthese finale et polissage
- Permet de poser des questions de suivi sur l'output final
- Permet d'exporter la proposition en PDF

## Stack

- `streamlit` (interface web)
- `google-generativeai` (API Gemini)
- `python-dotenv` (chargement cle API)
- `pypdf` (lecture PDF importes)
- `reportlab` (export PDF)

## Installation

```bash
cd "/Users/eliasmld/Desktop/OPTI Prompt"
python3 -m pip install -r requirements.txt
```

Creer un fichier `.env`:

```env
GEMINI_API_KEY=your_key_here
```

## Lancement

```bash
python3 -m streamlit run app.py
```

## Fichiers principaux

- `app.py`: application Streamlit principale (templates, pipeline, Q&A, export PDF)
- `prompt_chaining_gemini.py`: version CLI du pipeline
- `requirements.txt`: dependances Python
