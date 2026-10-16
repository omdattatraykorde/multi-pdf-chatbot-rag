# Research Test Plan

## Current model set

- Gemini 2.5 Flash — `gemini-2.5-flash`
- Gemini 2.5 Flash-Lite — `gemini-2.5-flash-lite`
- Groq GPT-OSS 120B — `openai/gpt-oss-120b`
- Groq GPT-OSS 20B — `openai/gpt-oss-20b`

## Test procedure

1. Upload the same PDFs for every model.
2. Process the PDFs once.
3. Use the same evaluation CSV for every model.
4. Keep retrieval K, hybrid retrieval settings, and temperature fixed.
5. Run the benchmark from the Research tab.
6. Compare latency, answer token F1, context overlap, source hit rate, and retrieval counts.
7. Interpret the results only from recorded runs; do not fabricate values.

## Retrieval behavior

The application indexes each page with semantic paragraph-aware chunks and
uses reciprocal-rank fusion of dense FAISS search and lexical matching. The
lexical path is especially useful for IDs, numbers, acronyms, and exact terms.
Query normalization adds a small set of domain-neutral synonyms without
rewriting the user's intent. Scanned pages are sent through OCR when the
optional Tesseract runtime is installed.

## Test dataset

Use `research/evaluation_test.csv`. It contains eight questions with concise expected answers and source/page references for controlled evaluation.

## Important research rule

A model failure must be recorded as a failure. The benchmark must not silently replace one selected model with another, because doing so would contaminate the model comparison. The interactive chat has a quota fallback for normal use, but benchmark errors remain associated with the selected model.
