"""
LLM Document Assistant
----------------------
Summarises a technical PDF, extracts key data as JSON, and answers questions about it
with a LOCAL language model running in LM Studio. No data leaves your computer.

Usage:
    python doc_assistant.py                              # menu: choose a PDF and a mode
    python doc_assistant.py spec.pdf                     # summary + key data
    python doc_assistant.py spec.pdf --mode summary      # summary only
    python doc_assistant.py spec.pdf --mode extract      # key data only (JSON)
    python doc_assistant.py spec.pdf --mode ask          # Q&A with page citations and highlighting
"""

import argparse
import json
import math
import os
import sys
import re
import subprocess  # runs Terminal commands; used to open the highlighted PDF automatically
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import pymupdf
from openai import OpenAI
from pypdf import PdfReader

# LM Studio offers an OpenAI-compatible server on this address by default.
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "google/gemma-4-e4b"
CHUNK_SIZE = 6000  # characters per chunk, small enough for local models




def choose_pdf() -> Path | None:
    """Open a file dialog to choose a PDF. Returns None if the user cancels."""
    root = tk.Tk()
    root.withdraw()                       # hide the empty main window, we only want the dialog
    root.attributes("-topmost", True)     # show the dialog in front of VS Code
    path = filedialog.askopenfilename(title="Choose a PDF",
                                      filetypes=[("PDF files", "*.pdf")])
    root.destroy()                        # close tkinter again
    return Path(path) if path else None

# ---------- 1. Read the PDF ----------
def read_pdf(path: Path) -> str:
    """Return the text of all pages in the PDF."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    if not text.strip():
        raise ValueError("No text found. The PDF may be a scanned image.")
    return text

def read_pages(path: Path) -> list[str]:
    """Return the text of each page separately, so we know the page numbers."""
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]


def split_into_chunks(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split long text into pieces, breaking at paragraph boundaries if possible."""
    chunks, current = [], ""
    for paragraph in text.split("\n"):
        if len(current) + len(paragraph) > size and current:
            chunks.append(current)
            current = ""
        current += paragraph + "\n"
    if current.strip():
        chunks.append(current)
    return chunks


# ---------- 2. Talk to the local model ----------
def ask_model(client: OpenAI, model: str, system: str, user: str) -> str:
    """Send one request to the local LLM and return its answer."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,  # low temperature = more factual, less creative
    )
    return response.choices[0].message.content.strip()


# ---------- 3. Summarise ----------
def summarise(client: OpenAI, model: str, text: str) -> str:
    """Summarise each chunk, then combine the partial summaries (map-reduce)."""
    system = ("You are a technical writer. Summarise technical documents "
              "precisely. Never invent facts that are not in the text.")
    chunks = split_into_chunks(text)
    partial = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"  Summarising part {i}/{len(chunks)} ...")
        partial.append(ask_model(client, model, system,
                                 f"Summarise this section in 3-5 bullet points:\n\n{chunk}"))
    if len(partial) == 1:
        return partial[0]
    combined = "\n\n".join(partial)
    return ask_model(client, model, system,
                     "Combine these partial summaries into one clear summary "
                     f"with at most 8 bullet points:\n\n{combined}")


# ---------- 4. Extract key data ----------
EXTRACT_PROMPT = """Extract the key data from this technical text.
Answer ONLY with valid JSON in exactly this structure:
{
  "document_type": "",
  "product_or_system": "",
  "key_parameters": [{"name": "", "value": "", "unit": "", "applies_to": "", "note": ""}],
  "standards": [],
  "safety_notes": []
}

Rules:
- key_parameters:
  - "value" is only the number or range, e.g. "16 to 27".
  - "unit" is only the unit, e.g. "mm", "deg C", "%", "Pa", "years". Put any extra words in "note".
  - If a value is given in two units, e.g. "minus 29 deg C (minus 20 deg F)",
    put the metric value in "value" and the imperial value in "note".
    Example: "value": "-29", "unit": "deg C", "note": "-20 deg F".
    Never combine two units into one range.
  - "applies_to" is the equipment or section the value belongs to, e.g. "Modular units (2.4)"
    or "Ceiling-mounted units (2.5)". Use the nearest section heading in the text.
- standards: one standard per entry. If the year is written in the text, write NAME NUMBER-YEAR,
  e.g. "NFPA 70-2007". If no year is written in the text, write only NAME NUMBER,
  e.g. "ASHRAE/IESNA 90.1". Never write the word "YEAR".
  If two standards are named together, list them separately.
  Do not include product features such as enclosure types.
- safety_notes: only requirements that protect people or equipment, such as alarms, interlocks,
  automatic shutdowns, fire or smoke detection, and electrical protection.
  Do not include general installation, testing or documentation requirements.
  Each entry must be a plain text string, not an object.
- Use empty strings or empty lists if something is not in the text. Do not guess.

TEXT:
"""


def parse_json(answer: str) -> dict:
    """Local models sometimes wrap JSON in ``` fences or add text. Find the JSON part."""
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    if not match:
        raise ValueError("The model did not return JSON.")
    return json.loads(match.group(0))


def extract_key_data(client: OpenAI, model: str, text: str) -> dict:
    """Extract structured data from each chunk and merge the results."""
    system = "You extract structured data from technical documents and reply only with JSON."
    result = {"document_type": "", "product_or_system": "",
              "key_parameters": [], "standards": [], "safety_notes": []}
    chunks = split_into_chunks(text)
    for i, chunk in enumerate(chunks, start=1):
        print(f"  Extracting data from part {i}/{len(chunks)} ...")
        try:
            data = parse_json(ask_model(client, model, system, EXTRACT_PROMPT + chunk))
        except (ValueError, json.JSONDecodeError) as err:
            print(f"  Skipped part {i}: {err}")
            continue
        for key in ("document_type", "product_or_system"):
            result[key] = result[key] or data.get(key, "")
        for key in ("key_parameters", "standards", "safety_notes"):
            for item in data.get(key, []):
                if item and str(item).lower() not in [str(x).lower() for x in result[key]]:
                    result[key].append(item)
    return result


# ---------- 5. Write the report ----------
def write_report(pdf: Path, summary: str | None, data: dict | None, out: Path) -> None:
    lines = [f"# Report: {pdf.name}", ""]
    if summary:
        lines += ["## Summary", "", summary, ""]
    if data:
        lines += ["## Key data", "",
                  f"- **Document type:** {data['document_type'] or '-'}",
                  f"- **Product / system:** {data['product_or_system'] or '-'}", "",
                  "| Parameter | Value | Unit |", "|---|---|---|"]
        for p in data["key_parameters"]:
            lines.append(f"| {p.get('name', '')} | {p.get('value', '')} | {p.get('unit', '')} |")
        lines += ["", "**Standards:** " + (", ".join(data["standards"]) or "-"), "",
                  "**Safety notes:**"] + [f"- {n}" for n in data["safety_notes"] or ["-"]]
    out.write_text("\n".join(lines), encoding="utf-8")


# ---------- 6. Ask questions about the PDF (with chat history) ----------

# Common words that say nothing about the topic. They are ignored in the search.
STOPWORDS = {"the", "and", "for", "what", "which", "how", "does", "must", "shall",
             "are", "is", "be", "it", "of", "to", "in", "on", "with", "this", "that",
             "there", "can", "you", "tell", "about", "required", "document"}

def get_keywords(question: str) -> set[str]:
    """Important words of the question: lowercase, no stopwords, simple plural removal."""
    words = set()
    for w in re.findall(r"[a-z0-9]+", question.lower()):
        if w in STOPWORDS or len(w) < 3:
            continue
        if w.endswith("s") and len(w) > 4:   # "compressors" -> "compressor"
            w = w[:-1]
        words.add(w)
    return words

def find_relevant_chunks(question: str, chunks: list[str], top_k: int = 2) -> list[int]:
    """Simple retrieval: score each chunk by how often the question's keywords appear."""

    words = get_keywords(question)

    lowered = [chunk.lower() for chunk in chunks]
    n = len(chunks)

    # IDF: words found in few chunks are strong clues, words found everywhere are weak.
    weights = {}
    for w in words:
        df = sum(1 for text in lowered if w in text)   # in how many chunks does w appear?
        weights[w] = math.log(n / df) if df else 0.0

    scores = []
    for i, text in enumerate(lowered):
        score = sum(text.count(w) * weights[w] for w in words)
        scores.append((score, i))

    scores.sort(reverse=True)                     # highest score first , sorts the pairs by score, highest first. Python sorts tuples by their first value.
    best = [i for score, i in scores[:top_k] if score > 0] # keeps the best top_k with a score above 0
    return sorted(best)                           # keep document order


def add_user_message(messages: list, text: str) -> None:
    """Add the user's question to the conversation history."""
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages: list, text: str) -> None:
    """Add the model's answer to the conversation history."""
    messages.append({"role": "assistant", "content": text})


def chat(client: OpenAI, model: str, messages: list) -> str:
    """Send the WHOLE conversation to the model and return its answer."""
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.2)
    return response.choices[0].message.content.strip()

def extract_quote(answer: str) -> str:
    """Find the line 'QUOTE: ...' in the model's answer and return the phrase."""
    match = re.search(r"QUOTE:\s*(.+)", answer)
    if not match:
        return ""
    return match.group(1).strip(' "*')

def ask_questions(client: OpenAI, model: str, pdf_path: Path) -> None:
    """Interactive Q&A. Each answer is highlighted in a copy of the PDF, which opens automatically."""
    pages = read_pages(pdf_path)
    out = Path(pdf_path.stem + "_highlighted.pdf")
    doc = pymupdf.open(pdf_path)          # a copy in memory; the original file is never changed
    has_highlights = False
    rules = ("Answer questions using only the document pages below. "
             "Mention the page number where you found the answer. "
             "If the answer is not in them, say so. "
             "At the very end, add one line starting with 'QUOTE:' followed by a short phrase "
             "(at most 12 words) copied exactly from the document that supports your answer.")
    history = []          # earlier questions and answers (without document text)
    last_question = ""

    print(f"Ask about the document ({len(pages)} pages). Type 'exit' to stop.")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        # 1. Retrieval: search with the question alone. Only if nothing is found, add the last question (follow-ups).
        best = find_relevant_chunks(question, pages, top_k=3)     # takes the best top_k.
        if not best and last_question:
            best = find_relevant_chunks(last_question + " " + question, pages, top_k=3)
        if not best:
            print("AI: I could not find anything about this in the document.\n")
            continue
        print(f"  (using pages {', '.join(str(i + 1) for i in best)})")
        context = "\n\n".join(f"[Page {i + 1}]\n{pages[i]}" for i in best)

        # 2. Build the request: rules + found pages, then recent history, then the new question.
        messages = [{"role": "system", "content": rules + "\n\nDOCUMENT PAGES:\n\n" + context}]
        messages += history[-6:]
        add_user_message(messages, question)

        # 3. Generation: the model answers using only the found pages.
        answer = chat(client, model, messages)
        quote = extract_quote(answer)
        shown = re.sub(r"QUOTE:.*", "", answer).strip()    # hide the QUOTE line from the answer

        add_user_message(history, question)
        add_assistant_message(history, shown)
        last_question = question
        print("AI:", shown)

        # 4. Keep or remove the highlights from earlier questions?
        removed = False
        if has_highlights:
            keep = input("  Keep the earlier highlights? (y/n, Enter = keep): ").strip().lower()
            if keep == "n":
                doc.close()
                doc = pymupdf.open(pdf_path)   # start again from a clean copy of the original
                has_highlights = False
                removed = True
                print("  Earlier highlights removed.")

        # 5. Highlight this answer: the exact quote first, the question's keywords as fallback.
        n = highlight_phrases(doc, best, [quote]) if quote else 0
        what = "the quote"
        if n == 0:
            n = highlight_phrases(doc, best, sorted(get_keywords(question)))
            what = "keywords (quote not found exactly)"
        if n:
            has_highlights = True
            print(f"  Highlighted {what}: {n} spot(s).")

        # 6. Save the copy and show it.
        if n or removed:
            doc.save(out)
            open_file(out)
            print(f"  Saved and opened {out}\n")
        else:
            print("  Nothing to highlight.\n")

    doc.close()


def open_file(path: Path) -> None:
    """Open a file with the default app on macOS, Windows or Linux."""
    if sys.platform == "darwin":                 # macOS
        subprocess.run(["open", str(path)])
    elif sys.platform == "win32":                # Windows
        os.startfile(path)
    else:                                        # Linux
        subprocess.run(["xdg-open", str(path)])

# ---------- 7. Highlight text in a copy of the PDF ----------
def highlight_phrases(doc, page_indices: list[int], phrases: list[str]) -> int:
    """Highlight the phrases on the given pages of an OPEN document (in memory). Returns the number of highlights."""
    count = 0
    for i in page_indices:
        page = doc[i]
        for phrase in phrases:
            rects = page.search_for(phrase)          # where is the phrase on this page?
            if rects:
                page.add_highlight_annot(rects)      # draw yellow highlight over those spots
                count += len(rects)
    return count


# ---------- 8. Run a mode / interactive menu ----------
def run_mode(client: OpenAI, model: str, pdf: Path, mode: str) -> None:
    """Run one mode on one PDF."""
    if mode == "ask":
        ask_questions(client, model, pdf)
        return
    print(f"Reading {pdf} ...")
    text = read_pdf(pdf)
    summary = data = None
    if mode in ("summary", "both"):
        summary = summarise(client, model, text)
    if mode in ("extract", "both"):
        data = extract_key_data(client, model, text)
        Path(pdf.stem + "_data.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    out = Path(pdf.stem + "_report.md")
    write_report(pdf, summary, data, out)
    print(f"Done. Report saved to {out}")


MENU = {"1": "summary", "2": "extract", "3": "both", "4": "ask"}


def interactive_menu(client: OpenAI, model: str) -> None:
    """Menu for running the tool without command-line arguments (e.g. with the Run button)."""
    print("Please choose a PDF in the window that opens ...")
    pdf = choose_pdf()
    if pdf is None:
        print("No file chosen. Goodbye.")
        return
    print(f"Chosen file: {pdf.name}")

    while True:
        print("\nWhat do you want to do?")
        print("  1 = summary   2 = extract   3 = both (summary and extract)   4 = ask questions   0 = exit")
        choice = input("Your choice: ").strip()
        if choice == "0":
            print("Goodbye.")
            break
        if choice not in MENU:
            print("Please type 0, 1, 2, 3 or 4.")
            continue
        run_mode(client, model, pdf, MENU[choice])

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarises a technical specification, extracts its key data, "
                    "or answers questions about it, using a local model in LM Studio.")
    parser.add_argument("pdf", type=Path, nargs="?",
                        help="Path to the PDF file (leave out to open the menu)")
    parser.add_argument("--mode", choices=["summary", "extract", "both", "ask"], default="both",
                        help="summary = short summary, extract = key data as JSON, "
                             "both = summary + extract, ask = interactive Q&A with highlighting "
                             "(default: %(default)s)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model name as shown in LM Studio (default: %(default)s)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="Address of the LM Studio server (default: %(default)s)")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="lm-studio")  # key is not checked locally
    if args.pdf is None:
        interactive_menu(client, args.model)      # Run button: no PDF given, so show the menu
    else:
        run_mode(client, args.model, args.pdf, args.mode)


if __name__ == "__main__":
    main()
