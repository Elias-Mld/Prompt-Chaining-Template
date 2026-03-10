#!/usr/bin/env python3
"""
Multi-Input Prompt Chaining Pipeline with Gemini Pro (google-generativeai SDK).

Install dependencies:
    pip install google-generativeai python-dotenv

Create a .env file in the same directory:
    GEMINI_API_KEY=your_api_key_here

Usage examples (interactive, recommended):
    python prompt_chaining_gemini.py

Usage examples (all inputs via CLI):
    python prompt_chaining_gemini.py --step1 "..." --step2 "..." --step3 "..." --step4 "..."

Usage examples (large inputs from files):
    python prompt_chaining_gemini.py --step1-file s1.txt --step2-file s2.txt --step3-file s3.txt --step4-file s4.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv


def call_gemini(model: genai.GenerativeModel, prompt: str) -> str:
    """
    Send one prompt to Gemini and return plain text output.
    Raises RuntimeError on an empty or malformed model response.
    """
    response = model.generate_content(prompt)
    text: Optional[str] = getattr(response, "text", None)

    if text and text.strip():
        return text.strip()

    # Some responses may come back in structured parts instead of response.text.
    try:
        parts = response.candidates[0].content.parts  # type: ignore[attr-defined]
        merged = "".join(getattr(part, "text", "") for part in parts).strip()
        if merged:
            return merged
    except Exception:
        pass

    raise RuntimeError("Gemini returned an empty response. Try re-running with clearer input.")


def step_1_context_framing(model: genai.GenerativeModel, step_input: str) -> str:
    prompt = f"""
Analyze the following input. Identify the core domain (e.g., Economics, Tech,
Strategy), the implicit goals of the user, and the level of expertise required
to provide a professional response. Define the "mental model" best suited to
process this information.

Input:
{step_input}
""".strip()
    return call_gemini(model, prompt)


def step_2_analytical_extraction(
    model: genai.GenerativeModel, user_input: str, output_step_1: str
) -> str:
    prompt = f"""
Using the analysis from Step 1, break down the input into its fundamental
components. Identify key arguments, technical data, and any logical gaps or
ambiguities that need to be addressed. Organize this into a structured
knowledge map.

Context Analysis:
{output_step_1}

Raw Input:
{user_input}
""".strip()
    return call_gemini(model, prompt)


def step_3_generative_synthesis(
    model: genai.GenerativeModel, user_input: str, output_step_2: str
) -> str:
    prompt = f"""
Analyze the Original User Input to determine which format is requested: a
GUIDE, a BLUEPRINT, or a CONVERSATION. Then, using the Structured Knowledge
Map, generate the final output by strictly adhering to that chosen format:

A GUIDE: (Educational/Procedural) Provide a step-by-step manual with clear
headings, 'Pro-Tips', and actionable advice.

A BLUEPRINT: (Technical/Structural) Provide a high-level architectural
framework, including components, dependencies, and a roadmap.

A CONVERSATION: (Exploratory/Dialectic) Provide a nuanced, deep-dive dialogue
exploring perspectives, trade-offs, and critical questions.

Original User Input:
{user_input}

Structured Knowledge Map:
{output_step_2}
""".strip()
    return call_gemini(model, prompt)


def step_4_critical_review(
    model: genai.GenerativeModel, user_input: str, output_step_3: str
) -> str:
    prompt = f"""
Act as a highly skeptical Auditor. Critically analyze the draft from Step 3 by
answering these three questions:

What are the blindspots? (What perspectives or data were ignored?)

What's missing or not right? (Identify inaccuracies or incomplete arguments.)

What holes can be poked? (Where is the logic weak or debatable?)

After identifying these flaws, rewrite the entire response to address them. The
final result must be a polished, "Senior Expert" level output that is resilient
to criticism. Output ONLY the final, refined response.

Original User Intent:
{user_input}

Draft to Audit:
{output_step_3}
""".strip()
    return call_gemini(model, prompt)


def final_synthesis_from_four_outputs(
    model: genai.GenerativeModel,
    output_step_1: str,
    output_step_2: str,
    output_step_3: str,
    output_step_4: str,
) -> str:
    prompt = f"""
You are a senior technical editor and synthesis specialist.
Merge the four analyses below into one final, polished, high-value technical
note for tech and data science professionals.

Rules:
- Keep the most rigorous and actionable insights.
- Remove redundancies and generic filler.
- Ensure a coherent structure and strong logical flow.
- Use precise technical language.
- Output only the final polished text.

Output from call 1:
{output_step_1}

Output from call 2:
{output_step_2}

Output from call 3:
{output_step_3}

Output from call 4:
{output_step_4}
""".strip()
    return call_gemini(model, prompt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-input 4-call pipeline, then a final synthesis call."
    )
    parser.add_argument(
        "--step1",
        dest="step1_input",
        help="Input text for call 1 (context and framing).",
    )
    parser.add_argument(
        "--step2",
        dest="step2_input",
        help="Input text for call 2 (analytical extraction).",
    )
    parser.add_argument(
        "--step1-file",
        dest="step1_file",
        help="Path to a text file for call 1 input.",
    )
    parser.add_argument(
        "--step2-file",
        dest="step2_file",
        help="Path to a text file for call 2 input.",
    )
    parser.add_argument(
        "--step3",
        dest="step3_input",
        help="Input text for call 3 (generative synthesis).",
    )
    parser.add_argument(
        "--step3-file",
        dest="step3_file",
        help="Path to a text file for call 3 input.",
    )
    parser.add_argument(
        "--step4",
        dest="step4_input",
        help="Input text for call 4 (critical review).",
    )
    parser.add_argument(
        "--step4-file",
        dest="step4_file",
        help="Path to a text file for call 4 input.",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        help="Optional Gemini model override (e.g. gemini-1.5-pro).",
    )
    return parser.parse_args()


def read_text_file(file_path: str, prompt_label: str) -> str:
    """
    Read one step input from a text file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()
    except OSError as exc:
        raise ValueError(f"{prompt_label}: unable to read file '{file_path}' ({exc}).") from exc

    if not text:
        raise ValueError(f"{prompt_label}: file '{file_path}' is empty.")
    return text


def resolve_step_input(cli_value: Optional[str], file_path: Optional[str], prompt_label: str) -> str:
    """
    Resolve input in this order: file > CLI value > interactive multiline paste.
    """
    if file_path and file_path.strip():
        return read_text_file(file_path.strip(), prompt_label)

    if cli_value and cli_value.strip():
        return cli_value.strip()

    print(
        f"{prompt_label}\n"
        "Paste multiline text, then type END on a new line and press Enter."
    )
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)

    user_value = "\n".join(lines).strip()
    if not user_value:
        raise ValueError(f"{prompt_label} cannot be empty.")
    return user_value


def resolve_model_name(model_override: Optional[str]) -> str:
    """
    Always use a fixed model.
    """
    _ = model_override  # Model override is intentionally ignored.
    return "gemini-3.1-pro"


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found. Add it to your .env file.")
        sys.exit(1)

    args = parse_args()

    genai.configure(api_key=api_key)

    selected_model_name = resolve_model_name(args.model_name)
    model = genai.GenerativeModel(model_name=selected_model_name)

    try:
        print(f"Using model: {selected_model_name}")
        print("Please provide four inputs (one per call).")
        step1_input = resolve_step_input(
            args.step1_input,
            args.step1_file,
            "Input 1 - Context and framing",
        )
        step2_input = resolve_step_input(
            args.step2_input,
            args.step2_file,
            "Input 2 - Analytical extraction",
        )
        step3_input = resolve_step_input(
            args.step3_input,
            args.step3_file,
            "Input 3 - Generative synthesis",
        )
        step4_input = resolve_step_input(
            args.step4_input,
            args.step4_file,
            "Input 4 - Critical review",
        )

        print("Running call 1/5: Context framing...")
        step1_output = step_1_context_framing(model, step1_input)

        print("Running call 2/5: Analytical extraction...")
        step2_output = step_2_analytical_extraction(model, step2_input, step1_output)

        print("Running call 3/5: Generative synthesis...")
        step3_output = step_3_generative_synthesis(model, step3_input, step2_output)

        print("Running call 4/5: Critical review...")
        step4_output = step_4_critical_review(model, step4_input, step3_output)

        print("Running call 5/5: Final synthesis from the four outputs...")
        final_output = final_synthesis_from_four_outputs(
            model,
            step1_output,
            step2_output,
            step3_output,
            step4_output,
        )
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("FINAL REFINED OUTPUT")
    print("=" * 80)
    print(final_output)


if __name__ == "__main__":
    main()
