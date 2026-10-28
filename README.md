# LLM Document Assistant

A local AI assistant for technical documents. It summarises a PDF specification,
extracts key data as structured JSON, and answers questions about it, citing page
numbers and highlighting the source in the PDF.

Everything runs **locally** with a model in [LM Studio](https://lmstudio.ai/).
No document text leaves the computer, which matters for confidential project documents.

## Features

- **Summary** (`--mode summary`): map-reduce summarisation. Each text chunk is summarised
  separately, then the partial summaries are combined.
- **Key data extraction** (`--mode extract`): standards, parameters (value, unit, section,
  note) and safety requirements as JSON, merged across chunks with case-insensitive
  duplicate removal.
- **Interactive Q&A** (`--mode ask`):
  - **Retrieval:** keyword search with **IDF weighting** finds the 3 most relevant pages
    for each question (a simple form of retrieval-augmented generation).
  - **Follow-up questions:** the previous question is used for the search only when the
    current one finds nothing (e.g. "And how accurate must it be?").
  - **Source citation:** answers include the page number.
  - **PDF highlighting:** the supporting quote is highlighted in a copy of the PDF, which
    opens automatically. Earlier highlights can be kept or removed after each answer.
    The original PDF is never changed.
- **Interactive menu:** run the script without arguments to choose a PDF in a file dialog
  and run several modes one after another.

## Setup

Requirements: Python 3.10+, [LM Studio](https://lmstudio.ai/) with a model loaded and the
local server running (default `http://localhost:1234/v1`).

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Windows, use `python` instead of `python3`.

## Usage

**Easiest way:** run the script without arguments, for example with the Run button in
VS Code. A file dialog opens to choose a PDF, then a menu lets you pick a mode
(summary, extract, both, ask) and run several modes one after another.

```bash
python doc_assistant.py
```

**Command line:** choose the PDF and mode directly.

```bash
python doc_assistant.py spec.pdf --mode summary
python doc_assistant.py spec.pdf --mode extract
python doc_assistant.py spec.pdf --mode both
python doc_assistant.py spec.pdf --mode ask
python doc_assistant.py --help
```

The default model is `google/gemma-4-e4b`. To use another model, add
`--model "name-as-shown-in-LM-Studio"`. For another local server, use `--base-url`.

Outputs (saved in the current folder): `<name>_report.md`, `<name>_data.json`,
`<name>_highlighted.pdf`.

## Try it with the included example

The folder `examples/` contains a public specification and the tool's unedited output:

- `AHU_Specifications.pdf`: VA Master Specification 23 73 00, Indoor Central-Station
  Air-Handling Units (U.S. Department of Veterans Affairs, version 03-01-23),
  converted from the Word file published at https://www.cfm.va.gov/TIL/spec/237300.docx
- `AHU_Specifications_report.md` and `AHU_Specifications_data.json`: the output of
  `--mode both`

```bash
python doc_assistant.py examples/AHU_Specifications.pdf --mode ask
```

## Evaluation

Tested on a 13-page HVAC specification (computer-room air-conditioning units) with
`google/gemma-4-e4b`. The test document is not included because it is confidential.

**Retrieval (ask mode)**
- Keyword counting alone picked the wrong pages for "thermostat temperature range",
  because common words like "temperature" appear on many pages. Adding **IDF weighting**
  fixed this, and questions with no answer in the document now correctly return nothing.
- Using the previous question for every search added noise ("thermostat" pulled in the
  wrong page for a question about manufacturers). It is now used only as a fallback.
- When the right page was not retrieved, the model answered "not stated in the document"
  instead of inventing an answer.

**Extraction (several runs, prompt improved between runs)**
- All 12 referenced standards were found. All extracted numeric values matched the
  specification; no invented values were found.
- Prompt rules improved the results measurably: separate standards, a consistent
  `NAME NUMBER-YEAR` format, clean units with extra words in a `note` field, and the
  section each value belongs to (e.g. filter values for modular vs. ceiling units).
- Each rule also had side effects: the placeholder "YEAR" was copied literally,
  "600 by 1200 mm" was split as if it were two units, and in one run the ranges
  "16 to 27 deg C" and "35 to 65 %" were split in half. All were found through testing.

## Results on a public specification

Input: the VA specification in `examples/` (see above). The outputs are unedited.

- **Standards:** all 19 publications listed in section 1.3 were found, with their years.
- **Parameters:** 42 values were extracted. Each was checked against the PDF and found
  in the text; no invented values. Metric values with the imperial value in `note`
  worked consistently.
- **Summary:** accurate, with no invented facts.

Problems found:

- The standards list has 40 entries: the 19 correct ones, plus short duplicates from
  other chunks (e.g. "AHRI 430"), references to other specification sections
  (e.g. "Section 23 82 16") and a vague entry ("ASTM Standards").
- The document is a template. Spec writer notes and optional paragraphs (marked `//...//`)
  were partly extracted as if they were requirements, for example the 23,600 L/s capacity
  limit and the damper leakage rates.
- Some parameter names are imprecise (a 152 × 152 mm door window is called an
  "access opening").
- Safety notes mix real safety items (e.g. latch restraint, no torch cutting) with
  general requirements.

## Limitations

- **Results vary between runs.** Values found in one run can be missing in the next.
  The output supports a human reviewer and does not replace one.
- Keyword retrieval does not understand synonyms or meaning. Embedding-based search
  would be the next step.
- Duplicates are only removed when the wording is identical apart from capitalisation.
  Standard names are not normalised, so the same standard can appear in a long and a
  short form.
- Section context can be lost when a chunk starts in the middle of a section.
- If the model's quote does not match the PDF text exactly, the question's keywords are
  highlighted instead.
- Scanned PDFs and drawings (e.g. equipment schedules) have no text layer and are not
  supported. This would need OCR or a vision model.
- The file dialog needs a screen. On a server without a display, use the command-line
  options instead.
- Tested on macOS. Windows and Linux are supported in the code but not yet tested.
- On Linux, the file dialog may need the system package `python3-tk`.

## Why I built this

I worked as a technical office engineer for more than seven years. When pricing a new
project, it was important to make sure that the price covered all specifications and
requirements, so I spent a lot of time reading specifications.

This tool helps me summarise a specification and find the most important data quickly.

Now, as a Master's student in Data Science, I want to specialise in AI for technical data
and documentation. I built this project to practise.

I used a local model because cloud API services are expensive while I am still learning.
A local model also keeps confidential project documents on my own computer.

## How it was built

The base script was created with AI assistance. I then extended it step by step:
page-based retrieval with IDF weighting, follow-up handling, page citations,
PDF highlighting, an interactive menu, case-insensitive duplicate removal and the
extraction prompt rules. Each change was tested on a real specification, and the
evaluation above summarises the results.

## Tech stack

Python · LM Studio (OpenAI-compatible API) · `openai` · `pypdf` · `PyMuPDF` · `tkinter`

PyMuPDF is licensed under AGPL-3.0. Commercial use requires a commercial license from Artifex.