# Short live demo (approximately 2 minutes)

**Setup before presenting:** Start Ollama; run `ollama pull gemma2:2b` once, then launch `streamlit run app.py`. Keep the baseline five fictional documents selected.

1. **Show ingestion (20 sec):** Point to indexed document/chunk counts and explain that TXT/PDF files are split, embedded, and added to FAISS.
2. **Direct evidence (20 sec):** Ask “What is the application fee and when is the Fall application due?” Show $40, July 15, and the `admissions.txt` passage.
3. **Two sources + paraphrase (25 sec):** Ask “Combine the attendance minimum and the health center's weekday opening time.” Verify citations from `attendance.txt` and `campus_services.txt`. Then phrase a scholarship GPA question differently from the source.
4. **Abstention (15 sec):** Ask for the mascot. Show that no sufficiently relevant passage was retrieved and the app did not generate a guess.
5. **Unseen document (20 sec):** Upload the assessor's new PDF/TXT, ask one question whose answer is only in that file, and inspect the evidence.
6. **Conflict (20 sec):** Upload both files in `sample_docs/conflict_demo/` and ask the Fall start date. Expected: surface both dates and filenames, identify the disagreement, and recommend confirming with the Registrar.
7. **Live modification:** Change the default `top_k` in `app.py` from 3 to 2, rerun the compound question, and explain how fewer retrieved passages can reduce irrelevant context but risk missing evidence.

## Technical issue to explain

PDF extraction libraries commonly number pages from zero, while people number from one. `format_passages()` adds one when displaying PDF citations, and the unit test verifies that a stored page index of 0 displays as page 1. Retrieval similarity is also only a heuristic, not a calibrated confidence probability; the threshold should be measured on the evaluation questions rather than chosen blindly.
