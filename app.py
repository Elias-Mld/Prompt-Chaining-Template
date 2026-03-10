#!/usr/bin/env python3
"""
Streamlit app for templates agentic pipeline.

Run:
    python3 -m streamlit run app.py

Dependencies:
    python3 -m pip install streamlit python-dotenv google-generativeai
"""

from __future__ import annotations

import os
from io import BytesIO

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

FLASH_MODEL_TARGET = "gemini-2.5-flash"
PRO_MODEL_TARGET = "gemini-3.1-pro"
STEP_WORD_LIMIT_RULE = ""
FOLLOWUP_QA_PROMPT = """
You are an expert assistant answering questions about the final generated proposal.
Use the proposal as the source of truth. If an answer is not present in the proposal,
state it clearly and suggest what additional information is needed.

Respond in this language: {final_language}

Final Proposal:
{final_output}

User Question:
{user_question}
""".strip()


# ----------------------------
# Template registry
# ----------------------------
# You can add more templates later by duplicating this structure.
TEMPLATES = {
    "4C Engine": {
        "boxes": {
            "context": {
                "label": "Context",
                "placeholder": "Project context, strategic angle, and high-level vision...",
            },
            "clarification": {
                "label": "Clarification",
                "placeholder": "Technical details, assumptions, and constraints...",
            },
            "creation": {
                "label": "Creation",
                "placeholder": "Expected output direction, style, and depth...",
            },
            "concerns": {
                "label": "Concerns",
                "placeholder": "Risks, objections, and critical watch points...",
            },
        },
        "prompts": {
            "step1": """
Analyze the following input. Identify the core domain (e.g., Economics, Tech,
Strategy), the implicit goals of the user, and the level of expertise required
to provide a professional response. Define the mental model best suited to
process this information.

Input:
{context_input}

{word_limit_rule}
""".strip(),
            "step2": """
Using the analysis from Step 1, break down the input into its fundamental
components. Identify key arguments, technical data, and logical gaps or
ambiguities that should be addressed. Build a structured knowledge map.

Context Analysis:
{output_step_1}

Clarification Input:
{clarification_input}

{word_limit_rule}
""".strip(),
            "step3": """
Using the structured knowledge map from Step 2, generate a high-value technical
draft with strong structure, precision, and clarity.

Knowledge Map:
{output_step_2}

Creation Input:
{creation_input}

{word_limit_rule}
""".strip(),
            "step4": """
Act as a skeptical technical auditor. Analyze the draft and extract:
- blindspots,
- missing or weak areas,
- debatable logic,
- risk flags to address.

Then provide an improved audited version.

Draft from Step 3:
{output_step_3}

Concerns Input:
{concerns_input}

{word_limit_rule}
""".strip(),
            "step5_final": """
Merge the four outputs below into one coherent final response.
Remove redundancy, keep only high-signal content, and preserve logical flow.
Then finalize for professional use:
- concise and impactful,
- technically rigorous,
- resilient to criticism,
- clean markdown structure.

Output ONLY the final polished response.
Final output language: {final_language}

Step 1 Output:
{output_step_1}

Step 2 Output:
{output_step_2}

Step 3 Output:
{output_step_3}

Step 4 Output:
{output_step_4}
""".strip(),
        },
    }
    ,
    "SaaS Proposal Engine": {
        "boxes": {
            "context": {
                "label": "MVP Scoping Notes",
                "placeholder": "Paste raw client meeting notes and core product context...",
            },
            "clarification": {
                "label": "Architecture & Tech Stack Context",
                "placeholder": "Add technical constraints, preferred stack, compliance needs...",
            },
            "creation": {
                "label": "Roadmap Constraints",
                "placeholder": "Add timing constraints, team size, delivery expectations...",
            },
            "concerns": {
                "label": "Financial Model Inputs",
                "placeholder": "Add budget constraints, pricing expectations, support scope...",
            },
        },
        "prompts": {
            "step1": """
Analyze these raw client meeting notes. Identify the core business problem.
Then, extract and list only the Must-Have features required to launch the first
version of the application (the MVP - Minimum Viable Product for Web or Mobile).
Exclude any nice-to-have or superfluous ideas for now to keep the scope realistic.

Client Meeting Notes:
{context_input}

{word_limit_rule}
""".strip(),
            "step2": """
Using the MVP scope defined in step 1, propose a modern and scalable Tech Stack
suitable for a B2B SaaS product. Clearly separate the Frontend, the Backend, and
the Database architecture.
Write a short, persuasive paragraph justifying these choices in terms of security,
performance, and long-term maintainability.

MVP Scope from Step 1:
{output_step_1}

Architecture Context:
{clarification_input}

{word_limit_rule}
""".strip(),
            "step3": """
Based on the MVP scope, create a clear and realistic Development Roadmap. Break
down the SaaS project into 4 chronological phases:
1) UI/UX Design & Wireframing
2) MVP Development
3) Testing (QA & Beta)
4) Deployment & Launch

Assign an estimated timeline (in weeks or months) to each phase to give the
client clear visibility.

MVP Scope from Step 1:
{output_step_1}

Roadmap Constraints:
{creation_input}

{word_limit_rule}
""".strip(),
            "step4": """
Structure the pricing proposal for this project. You must strictly separate the
financial investment into two distinct categories:
- Build phase (initial design and development fees, billed as a one-off)
- Run phase (recurring monthly fees for cloud hosting, server maintenance, and bug fixes)

Write this section in a transparent, reassuring, and professional tone.

Financial Inputs:
{concerns_input}

Context from Step 1 (MVP):
{output_step_1}

{word_limit_rule}
""".strip(),
            "step5_final": """
Act as an elite Chief Technology Officer (CTO) and Head of Sales. Take the
outputs from steps 1, 2, 3, and 4, and assemble them into a comprehensive,
high-converting commercial proposal for this SaaS project.

Use clean Markdown formatting (H1, H2, bullet points, bold text for key metrics).
Include a compelling Executive Summary at the very beginning and a strong
conclusion with a clear Call to Action (CTA) for the client to sign the proposal.
Final output language: {final_language}

Step 1 Output:
{output_step_1}

Step 2 Output:
{output_step_2}

Step 3 Output:
{output_step_3}

Step 4 Output:
{output_step_4}
""".strip(),
        },
    },
    "TikTok Video Script Engine": {
        "boxes": {
            "context": {
                "label": "Audience & Topic",
                "placeholder": "Topic, niche, audience profile, and desired angle...",
            },
            "clarification": {
                "label": "Brand Voice & Constraints",
                "placeholder": "Tone, banned claims, platform constraints, and key message...",
            },
            "creation": {
                "label": "Offer & CTA",
                "placeholder": "Product/service mention, call-to-action, links, and conversion goal...",
            },
            "concerns": {
                "label": "Risks & Optimization Notes",
                "placeholder": "Compliance risks, retention pain points, expected objections...",
            },
        },
        "prompts": {
            "step1": """
Analyze the audience and topic brief. Extract:
- target persona,
- desired transformation,
- emotional hook potential,
- likely watch-time triggers.

Then produce a short strategic positioning note for a high-performing TikTok video.

Audience and Topic Input:
{context_input}

{word_limit_rule}
""".strip(),
            "step2": """
Using Step 1 strategy, convert constraints into a creative execution framework.
Include:
- voice and pacing rules,
- do/don't guardrails,
- content claims to avoid,
- strongest framing options for the first 3 seconds.

Step 1 Strategy:
{output_step_1}

Brand and Constraint Input:
{clarification_input}

{word_limit_rule}
""".strip(),
            "step3": """
Draft a TikTok script optimized for retention and conversion.
Requirements:
- include a time-coded structure (Hook / Body / Proof / CTA),
- write short spoken lines suitable for mobile attention spans,
- add suggested on-screen text and b-roll cues.

Step 2 Framework:
{output_step_2}

Offer and CTA Input:
{creation_input}

{word_limit_rule}
""".strip(),
            "step4": """
Act as a content risk reviewer and performance editor.
Audit the draft for:
- weak hooks,
- retention drop-off moments,
- trust/compliance risk,
- unclear CTA wording.

Then rewrite an improved script version that fixes these issues.

Step 3 Draft:
{output_step_3}

Risks and Optimization Input:
{concerns_input}

{word_limit_rule}
""".strip(),
            "step5_final": """
Assemble a final creator-ready TikTok script package from Steps 1-4.
Output must include:
1) Final Script (clean, ready to record)
2) Shot List
3) On-Screen Text
4) Caption options (3 variants)
5) Hashtag set (balanced niche + broad)

Use concise markdown. Keep only high-impact content.
Final output language: {final_language}

Step 1 Output:
{output_step_1}

Step 2 Output:
{output_step_2}

Step 3 Output:
{output_step_3}

Step 4 Output:
{output_step_4}
""".strip(),
        },
    },
    "Newsletter Engine": {
        "boxes": {
            "context": {
                "label": "Topic & Reader Persona",
                "placeholder": "Main topic, subscriber profile, and desired outcome...",
            },
            "clarification": {
                "label": "Research Notes & Sources",
                "placeholder": "Facts, data points, references, examples, and links...",
            },
            "creation": {
                "label": "Angle & Editorial Direction",
                "placeholder": "Narrative angle, structure preferences, and CTA objective...",
            },
            "concerns": {
                "label": "Brand Guardrails",
                "placeholder": "Tone rules, legal sensitivities, taboo claims, and style constraints...",
            },
        },
        "prompts": {
            "step1": """
Analyze the newsletter brief and define:
- core reader problem,
- reader sophistication level,
- strongest editorial promise,
- recommended voice and narrative angle.

Provide an editorial strategy note.

Topic and Reader Input:
{context_input}

{word_limit_rule}
""".strip(),
            "step2": """
Using Step 1 strategy, process the research material into a reliable knowledge base.
Output:
- key claims to include,
- supporting evidence,
- insights worth highlighting,
- uncertainties that should be framed carefully.

Step 1 Strategy:
{output_step_1}

Research Input:
{clarification_input}

{word_limit_rule}
""".strip(),
            "step3": """
Write a full newsletter draft using the chosen editorial direction.
Structure:
- Subject line options (3)
- Preview text options (2)
- Introduction
- Main sections with clear subheadings
- Practical takeaways
- Call to action

Step 2 Knowledge Base:
{output_step_2}

Editorial Direction Input:
{creation_input}

{word_limit_rule}
""".strip(),
            "step4": """
Act as a senior editor and brand reviewer.
Audit the draft for:
- weak narrative flow,
- overclaims or unsupported statements,
- tone misalignment,
- low clarity or low actionability.

Then rewrite the draft to resolve all issues.

Step 3 Draft:
{output_step_3}

Brand Guardrails Input:
{concerns_input}

{word_limit_rule}
""".strip(),
            "step5_final": """
Produce the final publication-ready newsletter package from Steps 1-4.
Output format:
1) Final Subject Line (best option)
2) Final Preview Text
3) Final Newsletter Body
4) Alternate CTA options (2)
5) Repurposing ideas for social posts (3 bullets)

Ensure coherence, credibility, and strong readability.
Final output language: {final_language}

Step 1 Output:
{output_step_1}

Step 2 Output:
{output_step_2}

Step 3 Output:
{output_step_3}

Step 4 Output:
{output_step_4}
""".strip(),
        },
    },
}


def extract_text(response: object) -> str:
    """Extract text safely from Gemini response objects."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    try:
        candidates = getattr(response, "candidates", [])
        parts = candidates[0].content.parts if candidates else []
        merged = "".join(getattr(part, "text", "") for part in parts).strip()
        if merged:
            return merged
    except Exception:
        pass

    raise RuntimeError("Empty response from model.")


def call_model(model: genai.GenerativeModel, prompt: str) -> str:
    """Send one prompt to Gemini and return text output."""
    response = model.generate_content(prompt)
    return extract_text(response)


def extract_uploaded_text(uploaded_file: object) -> str:
    """
    Read uploaded document content as text.
    Supported: .txt, .md, .pdf
    """
    if uploaded_file is None:
        return ""

    filename = getattr(uploaded_file, "name", "document")
    raw_bytes = uploaded_file.getvalue()
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise RuntimeError(
                "PDF support requires `pypdf`. Install with: python3 -m pip install pypdf"
            ) from exc

        reader = PdfReader(BytesIO(raw_bytes))
        pages: list[str] = []
        for page in reader.pages:
            pages.append((page.extract_text() or "").strip())
        return "\n\n".join(p for p in pages if p).strip()

    # Text-like fallback for txt/md and similar files.
    return raw_bytes.decode("utf-8", errors="ignore").strip()


def merge_step_input(user_text: str, doc_text: str) -> str:
    """
    Merge manual text and optional uploaded document text.
    """
    user_text = user_text.strip()
    doc_text = doc_text.strip()
    if user_text and doc_text:
        return (
            f"Manual Notes:\n{user_text}\n\n"
            f"Document Content:\n{doc_text}"
        )
    return user_text or doc_text


def proposal_to_pdf_bytes(proposal_text: str) -> bytes:
    """
    Convert proposal text to PDF bytes.
    Requires reportlab: python3 -m pip install reportlab
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except Exception as exc:
        raise RuntimeError(
            "PDF export requires `reportlab`. Install with: python3 -m pip install reportlab"
        ) from exc

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    body_style = styles["BodyText"]

    elements = [Paragraph("Generated Proposal", title_style), Spacer(1, 12)]

    for raw_line in proposal_text.splitlines():
        line = raw_line.strip()
        if not line:
            elements.append(Spacer(1, 8))
            continue

        # Minimal markdown cleanup for better PDF readability.
        line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        line = line.replace("**", "")
        if line.startswith("# "):
            elements.append(Paragraph(line[2:].strip(), styles["Heading2"]))
        elif line.startswith("## "):
            elements.append(Paragraph(line[3:].strip(), styles["Heading3"]))
        elif line.startswith("- "):
            elements.append(Paragraph(f"• {line[2:].strip()}", body_style))
        else:
            elements.append(Paragraph(line, body_style))

    doc.build(elements)
    return buffer.getvalue()


def resolve_required_model_name(target_model: str, available_models: list[str]) -> str:
    """Resolve target model by exact match, then prefix match."""
    if target_model in available_models:
        return target_model
    for available in available_models:
        if available.startswith(target_model):
            return available
    raise RuntimeError(f"Required model not available for this API key: {target_model}")


def get_step_1_to_4_model_options(available_models: list[str]) -> dict[str, str]:
    """
    Build selectable options for steps 1-4.
    Returns a map: UI label -> concrete model name.
    """
    options: dict[str, str] = {}

    try:
        flash_name = resolve_required_model_name(FLASH_MODEL_TARGET, available_models)
        options["Gemini Flash 2.5 (default)"] = flash_name
    except RuntimeError:
        pass

    try:
        pro_name = resolve_required_model_name(PRO_MODEL_TARGET, available_models)
        options["Gemini Pro 3.1 (higher quality)"] = pro_name
    except RuntimeError:
        pass

    if not options:
        raise RuntimeError(
            "Neither Gemini Flash 2.5 nor Gemini Pro 3.1 is available for steps 1-4."
        )

    return options


@st.cache_data(show_spinner=False)
def get_available_models() -> list[str]:
    """Return all Gemini models that support generateContent."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found. Add it to your .env file.")

    genai.configure(api_key=api_key)
    available: list[str] = []

    for model_info in genai.list_models():
        methods = getattr(model_info, "supported_generation_methods", []) or []
        if "generateContent" not in methods:
            continue
        name = getattr(model_info, "name", "")
        if name.startswith("models/"):
            name = name[len("models/") :]
        if name:
            available.append(name)

    return sorted(set(available))


@st.cache_resource(show_spinner=False)
def get_pipeline_models(step_1_to_4_model_name: str) -> tuple[genai.GenerativeModel, genai.GenerativeModel]:
    """Configure SDK and return cached models for pipeline and final step."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found. Add it to your .env file.")

    genai.configure(api_key=api_key)
    available_models = get_available_models()
    if not available_models:
        raise RuntimeError("No Gemini models available for generateContent with this API key.")

    if step_1_to_4_model_name not in available_models:
        raise RuntimeError(f"Selected model is not available: {step_1_to_4_model_name}")

    pro_model_name = resolve_required_model_name(PRO_MODEL_TARGET, available_models)

    return (
        genai.GenerativeModel(model_name=step_1_to_4_model_name),
        genai.GenerativeModel(model_name=pro_model_name),
    )


def main() -> None:
    st.set_page_config(page_title="Prompt Chaining", layout="wide")
    st.title("Prompt Chaining")
    st.caption("Select a template, fill the boxes, then run the chaining pipeline.")

    selected_template_name = st.selectbox("Template", options=list(TEMPLATES.keys()), index=0)
    template = TEMPLATES[selected_template_name]
    box_cfg = template["boxes"]
    prompts = template["prompts"]
    context_label = box_cfg["context"]["label"]
    clarification_label = box_cfg["clarification"]["label"]
    creation_label = box_cfg["creation"]["label"]
    concerns_label = box_cfg["concerns"]["label"]

    try:
        available_models = get_available_models()
        step_model_options = get_step_1_to_4_model_options(available_models)
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        return

    step_model_labels = list(step_model_options.keys())
    default_idx = 0
    for idx, label in enumerate(step_model_labels):
        if "Flash" in label:
            default_idx = idx
            break

    selected_step_model_label = st.selectbox(
        "Model for Steps 1 to 4",
        options=step_model_labels,
        index=default_idx,
        help="Step 5 remains fixed on Gemini Pro 3.1.",
    )
    selected_step_model_name = step_model_options[selected_step_model_label]
    selected_output_language = st.selectbox(
        "Final Output Language",
        options=["English", "French"],
        index=0,
    )

    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None
    if "followup_history" not in st.session_state:
        st.session_state.followup_history = []

    st.markdown(f"### {selected_template_name} Inputs")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        context_input = st.text_area(
            box_cfg["context"]["label"],
            placeholder=box_cfg["context"]["placeholder"],
            height=240,
        )
        context_file = st.file_uploader(
            f"Document {box_cfg['context']['label']} (optional)",
            type=["txt", "md", "pdf"],
            key="context_file",
        )
    with col2:
        clarification_input = st.text_area(
            box_cfg["clarification"]["label"],
            placeholder=box_cfg["clarification"]["placeholder"],
            height=240,
        )
        clarification_file = st.file_uploader(
            f"Document {box_cfg['clarification']['label']} (optional)",
            type=["txt", "md", "pdf"],
            key="clarification_file",
        )
    with col3:
        creation_input = st.text_area(
            box_cfg["creation"]["label"],
            placeholder=box_cfg["creation"]["placeholder"],
            height=240,
        )
        creation_file = st.file_uploader(
            f"Document {box_cfg['creation']['label']} (optional)",
            type=["txt", "md", "pdf"],
            key="creation_file",
        )
    with col4:
        concerns_input = st.text_area(
            box_cfg["concerns"]["label"],
            placeholder=box_cfg["concerns"]["placeholder"],
            height=240,
        )
        concerns_file = st.file_uploader(
            f"Document {box_cfg['concerns']['label']} (optional)",
            type=["txt", "md", "pdf"],
            key="concerns_file",
        )

    run_clicked = st.button("Run Prompt Chaining", type="primary", use_container_width=True)

    if run_clicked:
        try:
            context_doc_text = extract_uploaded_text(context_file)
            clarification_doc_text = extract_uploaded_text(clarification_file)
            creation_doc_text = extract_uploaded_text(creation_file)
            concerns_doc_text = extract_uploaded_text(concerns_file)
        except Exception as exc:
            st.error(f"Document parsing error: {exc}")
            return

        context_payload = merge_step_input(context_input, context_doc_text)
        clarification_payload = merge_step_input(clarification_input, clarification_doc_text)
        creation_payload = merge_step_input(creation_input, creation_doc_text)
        concerns_payload = merge_step_input(concerns_input, concerns_doc_text)

        missing = [
            label
            for label, value in [
                (context_label, context_payload),
                (clarification_label, clarification_payload),
                (creation_label, creation_payload),
                (concerns_label, concerns_payload),
            ]
            if not value.strip()
        ]
        if missing:
            st.warning(f"Please fill all four boxes before running: {', '.join(missing)}.")
            return

        try:
            flash_model, pro_model = get_pipeline_models(selected_step_model_name)
            st.info(
                "Models in use: "
                f"`{flash_model.model_name}` for steps 1-4, "
                f"`{pro_model.model_name}` for final step."
            )
        except Exception as exc:
            st.error(f"Configuration error: {exc}")
            return

        try:
            with st.status("Running orchestration...", expanded=True) as status:
                status.write(f"Step 1/5: Analyzing {context_label}...")
                step1 = call_model(
                    flash_model,
                    prompts["step1"].format(
                        context_input=context_payload,
                        word_limit_rule=STEP_WORD_LIMIT_RULE,
                    ),
                )

                status.write(f"Step 2/5: Building {clarification_label} map...")
                step2 = call_model(
                    flash_model,
                    prompts["step2"].format(
                        clarification_input=clarification_payload,
                        output_step_1=step1,
                        word_limit_rule=STEP_WORD_LIMIT_RULE,
                    ),
                )

                status.write(f"Step 3/5: Generating {creation_label} draft...")
                step3 = call_model(
                    flash_model,
                    prompts["step3"].format(
                        creation_input=creation_payload,
                        output_step_1=step1,
                        output_step_2=step2,
                        word_limit_rule=STEP_WORD_LIMIT_RULE,
                    ),
                )

                status.write(f"Step 4/5: Auditing {concerns_label}...")
                step4 = call_model(
                    flash_model,
                    prompts["step4"].format(
                        concerns_input=concerns_payload,
                        output_step_1=step1,
                        output_step_3=step3,
                        word_limit_rule=STEP_WORD_LIMIT_RULE,
                    ),
                )

                status.write("Step 5/5: Global synthesis + final polish...")
                final_output = call_model(
                    pro_model,
                    prompts["step5_final"].format(
                        output_step_1=step1,
                        output_step_2=step2,
                        output_step_3=step3,
                        output_step_4=step4,
                        final_language=selected_output_language,
                    ),
                )

                status.update(label="Pipeline complete.", state="complete", expanded=False)

            st.session_state.pipeline_results = {
                "step1": step1,
                "step2": step2,
                "step3": step3,
                "step4": step4,
                "final_output": final_output,
                "final_language": selected_output_language,
                "step_model_name": selected_step_model_name,
            }
            st.session_state.followup_history = []
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            return

    results = st.session_state.pipeline_results
    if not results:
        return

    st.subheader("Final Output")
    with st.container(border=True):
        st.markdown(results["final_output"])

    try:
        pdf_bytes = proposal_to_pdf_bytes(results["final_output"])
        st.download_button(
            label="Download Proposal (PDF)",
            data=pdf_bytes,
            file_name="proposal_output.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:
        st.info(f"PDF download not available yet: {exc}")

    st.markdown("### Questions on Final Proposal")
    followup_question = st.text_area(
        "Ask a question about the generated proposal",
        placeholder="Example: Can you break down the budget by milestone?",
        height=120,
        key="followup_question",
    )
    ask_clicked = st.button("Ask Question", use_container_width=True)
    if ask_clicked:
        if not followup_question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                _, pro_model = get_pipeline_models(results["step_model_name"])
                with st.status("Analyzing final proposal and answering...", expanded=False):
                    answer = call_model(
                        pro_model,
                        FOLLOWUP_QA_PROMPT.format(
                            final_language=results["final_language"],
                            final_output=results["final_output"],
                            user_question=followup_question.strip(),
                        ),
                    )
                st.session_state.followup_history.append(
                    {"question": followup_question.strip(), "answer": answer}
                )
            except Exception as exc:
                st.error(f"Follow-up failed: {exc}")

    if st.session_state.followup_history:
        st.markdown("#### Q&A History")
        for idx, item in enumerate(reversed(st.session_state.followup_history), start=1):
            with st.expander(f"Question {idx}", expanded=(idx == 1)):
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(item["answer"])

    st.markdown("---")
    st.subheader("Debug Outputs")
    with st.expander(f"Step 1 - {context_label}", expanded=False):
        st.markdown(results["step1"])
    with st.expander(f"Step 2 - {clarification_label}", expanded=False):
        st.markdown(results["step2"])
    with st.expander(f"Step 3 - {creation_label}", expanded=False):
        st.markdown(results["step3"])
    with st.expander(f"Step 4 - {concerns_label} Audit", expanded=False):
        st.markdown(results["step4"])


if __name__ == "__main__":
    main()
