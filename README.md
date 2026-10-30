# College Knowledge Assistant (RAG)

A local Streamlit assistant that answers questions using a selectable knowledge base of college PDF and TXT files. Retrieval uses the same core pattern shown in the `DL TRAINING` reference (`RecursiveCharacterTextSplitter` → MiniLM embeddings → FAISS → Ollama); answers are constrained to retrieved passages, displayed with source evidence, and gated by a relevance threshold. The configured model is `gemma2:2b`.

> **Sample data warning:** Every bundled policy and contact detail is fictional and exists only for demonstrating and testing the application. Do not treat it as actual college information.

## Requirements and prerequisites

The `requirements.txt` lists the Python packages used by this app. Embeddings use FastEmbed with ONNX Runtime DirectML to run on compatible Windows GPUs, including AMD GPUs, without PyTorch. Ollama is a separately installed local runtime, not a Python package. Install Ollama for your operating system and make its command available in the terminal.

The embedding model `sentence-transformers/all-MiniLM-L6-v2` is downloaded by FastEmbed on first use; subsequent use is local. First startup therefore needs internet access, unless that model is already cached. Embeddings use the DirectML GPU provider. Afterward, both generation and document search run locally.

## Setup

Open a terminal in this project directory:

```bash
python -m venv .venv
```

Activate the environment (Windows PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Install the app dependencies and download the requested Ollama model:

```bash
python -m pip install -r requirements.txt
ollama pull gemma2:2b
```

Start Ollama if it is not already running, then launch Streamlit:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal (normally `http://localhost:8501`). The first run builds a local FAISS index and can take longer while the embedding model is downloaded. Scanned/image-only PDFs are not OCRed; use text-extractable PDFs or convert/OCR them first.

## How to use

1. In the sidebar, keep or select bundled sample documents. Five fictional college documents are included in `sample_docs/`.
2. Optionally add multiple `.pdf` and `.txt` files with the uploader. Newly uploaded files join the selected samples in the current session's knowledge base.
3. Adjust chunk size/overlap, retrieval `k`, and the minimum relevance threshold. The default threshold is a cautious starting point, not a universal value; tune it using known answerable and unanswerable questions for your document set.
4. Ask a question. The application displays the retrieved passages, relevance scores, source filename, and PDF page (where available).
5. If no passage reaches the threshold, the model is not called. It returns a fixed not-found answer. If the retrieved documents conflict, the prompt tells Gemma not to choose a winner and to report both source-specific claims.

For the conflict exercise, add both files from `sample_docs/conflict_demo/` through the uploader and ask: **“When does the Fall semester begin?”** The expected behavior is to report both dates with both filenames and say the source documents disagree. The sample files are deliberately not selected by default so the normal corpus is not contradictory.

## Architecture

```text
PDF/TXT files
    └─ extract text + source metadata (PDF page metadata retained)
       └─ RecursiveCharacterTextSplitter (size/overlap configurable)
          └─ FastEmbed all-MiniLM-L6-v2 embeddings (ONNX Runtime DirectML GPU)
             └─ FAISS vector index (rebuilt when files/settings change)
                └─ top-k similarity retrieval + minimum relevance gate
                   └─ grounded-only prompt → Ollama gemma2:2b
                      └─ answer + retrieved source passages and citations in Streamlit
```

`app.py` owns ingestion, Streamlit widgets, session state, FAISS construction, Ollama streaming, and evidence display. `rag_core.py` holds reusable chunking, relevance, citation, and prompt helpers. `sample_docs/` contains a fictional 5-document baseline plus an optional conflicting pair. `tests/` contains fast model-free checks; `test_questions.json` is the manual evaluation set.

### Why these design choices?

- **Reference-style interface:** The sidebar configuration, chunk controls, chat-history replay, streamed answer, retrieved-chunk expander, hardware status, and pipeline footer follow the organization of `DL TRAINING/app.py`, adapted to college PDFs/TXTs and source-grounded responses.
- **Chunking:** Recursive splitting prefers paragraph and line boundaries, with overlap to reduce lost context at chunk edges. The defaults are 500 characters and 200 characters of overlap, matching the reference application's initial slider values; evaluate alternatives on the unseen demonstration document.
- **Retrieval:** FAISS performs local vector search; `k` is exposed so students can observe the trade-off between recall and irrelevant context.
- **Abstention:** A configurable relevance cutoff blocks generation when retrieval appears weak. A vector score is only a heuristic—not proof of answer correctness—so false positives/negatives should be measured and the threshold validated on labeled questions.
- **Citations:** Source metadata is retained through splitting and shown beside each retrieved excerpt. The model is instructed to cite those exact labels, and the raw evidence is displayed so users can verify claims.
- **Conflicts:** Both conflicting files are injected together during the test; the prompt explicitly requires reporting each source claim instead of silently selecting one. This is a prompt-level safeguard, not a formal contradiction detector; inspect the displayed evidence and test the model output.
- **Local model:** `gemma2:2b` runs via Ollama. Temperature is zero to reduce variation. A small model can still miss nuance, so citations and retrieved context remain visible for human review.

## Tests and evaluation

Run the fast automated unit tests (no running Ollama server or embedding model required):

```bash
python -m unittest discover -s tests -v
```

The unit tests validate the retrieval helpers, prompt instructions, citation formatting, and sample fixtures. They do **not** measure semantic retrieval quality or guarantee LLM output. For end-to-end evaluation, run `streamlit run app.py` and work through all ten cases in `test_questions.json`. Compare the result with each expected outcome; record whether the right evidence was retrieved, whether citations match, and whether the abstention/conflict policy was followed. Manually add both conflict files for question 10. For the cross-document case, ask the compound question as written and confirm evidence from both files appears.

### Ten manual test questions and expected results

| # | Scenario / question | Expected result |
|---|---|---|
| 1 | What is the application fee and when is the Fall application due? | $40 fee; July 15; cite `admissions.txt`. |
| 2 | What GPA and paperwork are needed for the Merit Scholarship? | Cumulative GPA at least 3.5, form, and 500-word statement; cite `scholarships.txt`. |
| 3 | How late can I study in the library on a weekday? | Library closes at 10:00 p.m. Monday–Friday; cite `library.txt`. |
| 4 | How many classes can I miss before I fall below the attendance rule? | At least 75% attendance is expected; explain that absence count depends on course sessions; cite `attendance.txt`. |
| 5 | Where do I go for a health appointment, and what number can I call? | West Hall, Room 204; 555-0104; cite `campus_services.txt`. |
| 6 | Combine the attendance minimum and the health center's weekday opening time. | 75% attendance from `attendance.txt` and 9:00 a.m. weekday opening from `campus_services.txt`; cite both. |
| 7 | Paraphrase test: “What grades do I need to qualify for the merit award?” | Recognize the paraphrase and return GPA ≥ 3.5, citing `scholarships.txt`. |
| 8 | What is the college's official mascot's name? | Not found in the provided documents; abstain and do not guess. |
| 9 | What is the password for the student portal? | Not found; abstain and do not invent credentials. |
| 10 | (Upload both conflict notices) When does the Fall semester begin? | State that the documents conflict: Sep 3 in `calendar_notice_A.txt` and Sep 8 in `calendar_notice_B.txt`; do not pick one, advise confirmation with the Registrar. |

The sample facts are intentionally fictional. Replace them with approved institutional documents before any real deployment.

## Short demo script (about 2 minutes)

1. Start Ollama and `streamlit run app.py`; show the five selected sample documents and indexed chunk count.
2. Ask test #1; point to the answer and `admissions.txt` evidence.
3. Ask test #6; verify two source filenames appear in the retrieved evidence.
4. Ask test #8; show the fixed abstention path (no generation when all matches are below threshold).
5. Upload a new, unseen text/PDF document provided by the assessor and ask a fact in it. This demonstrates ingestion rather than memorized output.
6. Upload both conflict notices and ask test #10; discuss the cited disagreement.
7. Live modification: change the sidebar default `top_k` from `3` to `2`, rerun the cross-document question, and explain how retrieval count changes context/recall.

## Technical issue addressed

PDF page metadata is commonly zero-based in extraction libraries, but readers expect page numbers starting at 1. The core formatting helper converts zero-based PDF metadata to human-facing numbering while retaining the original source metadata on each chunk. The unit test checks that page index `0` is cited as page 1. Another practical limitation is retrieval score calibration: a similarity score is not a calibrated probability, so the threshold is exposed and the manual evaluation set includes answerable and absent questions.
