import csv
import os
import re
import time
from datetime import datetime

from rag import _get_context
from llm_factory import LLMFactory


RESULT_FILE = os.path.join("research", "benchmark_results.csv")


def _tokens(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def token_f1(reference, prediction):
    ref = _tokens(reference)
    pred = _tokens(prediction)
    if not ref or not pred:
        return 0.0
    from collections import Counter
    r, p = Counter(ref), Counter(pred)
    common = sum((r & p).values())
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return round(2 * precision * recall / (precision + recall), 4)


def context_overlap(answer, context):
    a = set(_tokens(answer))
    c = set(_tokens(context))
    if not a:
        return 0.0
    return round(len(a & c) / len(a), 4)


def source_hit(docs, expected_source="", expected_page=""):
    if not expected_source:
        return ""
    expected = expected_source.lower().strip()
    expected_page_num = None
    if str(expected_page).strip():
        try:
            expected_page_num = int(float(expected_page))
        except ValueError:
            expected_page_num = None
    for doc in docs:
        source = str(doc.metadata.get("source", "")).lower()
        page = int(doc.metadata.get("page", 0)) + 1
        source_ok = expected in source or os.path.basename(expected) in source
        page_ok = expected_page_num is None or page == expected_page_num
        if source_ok and page_ok:
            return 1
    return 0


def _write_rows(rows):
    os.makedirs("research", exist_ok=True)
    headers = [
        "Timestamp", "Question ID", "Model", "Question", "Expected Answer",
        "Answer", "Response Time (sec)", "Chunks Retrieved", "Pages Retrieved",
        "Source Hit", "Answer Token F1", "Context Overlap"
    ]
    # A controlled benchmark is a new experiment: replace stale/incompatible results
    # so the Research dashboard always reads one consistent schema.
    with open(RESULT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def run_benchmark(vector_db, rows, models, k=5):
    output = []
    for item in rows:
        qid = str(item.get("id", ""))
        question = str(item.get("question", "")).strip()
        expected = str(item.get("expected_answer", "")).strip()
        expected_source = str(item.get("source", "")).strip()
        expected_page = str(item.get("page", "")).strip()
        if not question:
            continue

        context, docs = _get_context(vector_db, question, k=k)
        for model in models:
            prompt = f"""You are a careful document research assistant.
Answer ONLY from the supplied context. Cite factual claims with the exact
source marker format [Source: filename | Page: N]. Never invent citations.
If the answer is not supported by the context, say:
I couldn't find that information in the uploaded PDFs.

Context:
{context}

Question:
{question}

Answer:"""
            start = time.perf_counter()
            try:
                llm = LLMFactory.get_llm(model)
                answer = llm.invoke(prompt).content
                elapsed = time.perf_counter() - start
                pages = len({(d.metadata.get('source', ''), d.metadata.get('page', 0)) for d in docs})
                output.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Question ID": qid,
                    "Model": model,
                    "Question": question,
                    "Expected Answer": expected,
                    "Answer": answer,
                    "Response Time (sec)": round(elapsed, 3),
                    "Chunks Retrieved": len(docs),
                    "Pages Retrieved": pages,
                    "Source Hit": source_hit(docs, expected_source, expected_page),
                    "Answer Token F1": token_f1(expected, answer) if expected else "",
                    "Context Overlap": context_overlap(answer, context),
                })
            except Exception as exc:
                output.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Question ID": qid,
                    "Model": model,
                    "Question": question,
                    "Expected Answer": expected,
                    "Answer": f"ERROR: {exc}",
                    "Response Time (sec)": round(time.perf_counter() - start, 3),
                    "Chunks Retrieved": len(docs),
                    "Pages Retrieved": len({(d.metadata.get('source', ''), d.metadata.get('page', 0)) for d in docs}),
                    "Source Hit": "",
                    "Answer Token F1": "",
                    "Context Overlap": "",
                })
    _write_rows([
        [r[h] for h in ["Timestamp", "Question ID", "Model", "Question", "Expected Answer", "Answer", "Response Time (sec)", "Chunks Retrieved", "Pages Retrieved", "Source Hit", "Answer Token F1", "Context Overlap"]]
        for r in output
    ])
    return output


def read_csv(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8-sig")
    return list(csv.DictReader(text.splitlines()))
