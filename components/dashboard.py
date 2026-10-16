import os
import pandas as pd
import streamlit as st

from research_evaluator import read_csv, run_benchmark, RESULT_FILE


def _load_logs():
    path = "research_logs.csv"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _load_benchmark():
    if not os.path.exists(RESULT_FILE) or os.path.getsize(RESULT_FILE) == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(RESULT_FILE)
        # Older benchmark files may have different column names or an incomplete header.
        aliases = {
            "Response Time": "Response Time (sec)",
            "Response Time (s)": "Response Time (sec)",
            "Latency": "Response Time (sec)",
            "Answer F1": "Answer Token F1",
            "F1": "Answer Token F1",
            "Source Hit Rate": "Source Hit",
        }
        df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
        required = {
            "Model", "Question ID", "Question", "Expected Answer", "Answer",
            "Response Time (sec)", "Chunks Retrieved", "Pages Retrieved",
            "Source Hit", "Answer Token F1", "Context Overlap"
        }
        # Never crash the Research tab because of a stale/corrupt result file.
        if not required.intersection(set(df.columns)):
            return pd.DataFrame()
        for col in required:
            if col not in df.columns:
                df[col] = pd.NA
        return df
    except Exception:
        return pd.DataFrame()


def _comparison_table(df):
    if df.empty:
        return df
    out = df.copy()
    # Normalize columns so the dashboard remains compatible with prior result files.
    for col in ["Response Time (sec)", "Answer Token F1", "Context Overlap", "Source Hit", "Chunks Retrieved", "Pages Retrieved"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Model" not in out.columns:
        return pd.DataFrame()
    grouped = out.groupby("Model", dropna=False).agg(
        Runs=("Model", "size"),
        Avg_Latency=("Response Time (sec)", "mean"),
        Median_Latency=("Response Time (sec)", "median"),
        Avg_Chunks=("Chunks Retrieved", "mean"),
        Avg_Pages=("Pages Retrieved", "mean"),
        Avg_F1=("Answer Token F1", "mean"),
        Avg_Context=("Context Overlap", "mean"),
        Source_Hit_Rate=("Source Hit", "mean"),
    ).reset_index()
    grouped["Avg_Latency"] = grouped["Avg_Latency"].round(3)
    grouped["Median_Latency"] = grouped["Median_Latency"].round(3)
    grouped["Avg_Chunks"] = grouped["Avg_Chunks"].round(2)
    grouped["Avg_Pages"] = grouped["Avg_Pages"].round(2)
    grouped["Avg_F1"] = (grouped["Avg_F1"] * 100).round(1)
    grouped["Avg_Context"] = (grouped["Avg_Context"] * 100).round(1)
    grouped["Source_Hit_Rate"] = (grouped["Source_Hit_Rate"] * 100).round(1)
    return grouped.rename(columns={
        "Avg_Latency": "Avg latency (s)", "Median_Latency": "Median latency (s)",
        "Avg_Chunks": "Avg chunks", "Avg_Pages": "Avg pages", "Avg_F1": "Answer F1 (%)",
        "Avg_Context": "Context overlap (%)", "Source_Hit_Rate": "Source hit (%)"
    })


def render_dashboard(experiment_data=None):
    st.markdown("<div class='section-kicker'>EXPERIMENTS</div>", unsafe_allow_html=True)
    st.markdown("## Research & model comparison")
    st.caption("Run controlled experiments with identical questions, documents and retrieval settings, then compare the measured results.")

    if experiment_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", experiment_data["total_pdfs"])
        c2.metric("Pages", experiment_data["total_pages"])
        c3.metric("Text chunks", experiment_data["total_chunks"])
        c4.metric("Embedding time", f'{experiment_data["embedding_time"]:.2f}s')
        st.caption(
            f"Chunking: {experiment_data.get('chunking', 'page-aware')} · "
            f"Retrieval: {experiment_data.get('retrieval', 'hybrid')} · "
            f"Embeddings: {experiment_data.get('embedding_backend', 'local')} · "
            f"OCR pages recovered: {experiment_data.get('ocr_pages', 0)}"
        )

    st.markdown("### Controlled benchmark")
    st.caption("Use a CSV with question, expected_answer and optional source/page fields. Every selected model receives the same benchmark.")

    template = "id,question,expected_answer,source,page\nQ1,What is the main topic?,,,\n"
    st.download_button("Download benchmark template", template, "evaluation_template.csv", "text/csv")

    eval_file = st.file_uploader("Evaluation CSV", type=["csv"], key="evaluation_csv")
    models = st.multiselect(
        "Models",
        ["Gemini 2.5 Flash", "Gemini 2.5 Flash-Lite", "Groq GPT-OSS 120B", "Groq GPT-OSS 20B"],
        default=["Gemini 2.5 Flash", "Groq GPT-OSS 120B"],
    )
    k = st.slider("Retrieved chunks (K)", 3, 8, 5)

    if eval_file:
        rows = read_csv(eval_file)
        st.info(f"{len(rows)} benchmark question(s) loaded."
                " The same questions and retrieval configuration will be used for each selected model.")
        with st.expander("Preview benchmark", expanded=False):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if st.button("Run controlled comparison", type="primary", use_container_width=True, disabled=(experiment_data is None or not models)):
            with st.spinner("Running benchmark across selected models…"):
                results = run_benchmark(st.session_state.vector_db, rows, models, k=k)
            st.success(f"Completed {len(results)} model-question runs.")
            st.rerun()

    benchmark = _load_benchmark()
    if not benchmark.empty:
        st.markdown("### Results")
        summary = _comparison_table(benchmark)
        st.dataframe(summary, use_container_width=True, hide_index=True)

        numeric = benchmark.copy()
        numeric["Response Time (sec)"] = pd.to_numeric(numeric["Response Time (sec)"], errors="coerce")
        st.markdown("### Performance view")
        a, b = st.columns(2)
        with a:
            latency = numeric.groupby("Model", as_index=False)["Response Time (sec)"].mean()
            latency.columns = ["Model", "Average response time"]
            st.bar_chart(latency.set_index("Model"))
        with b:
            f1 = numeric.copy()
            f1["Answer Token F1"] = pd.to_numeric(f1["Answer Token F1"], errors="coerce")
            f1 = f1.dropna(subset=["Answer Token F1"]).groupby("Model", as_index=False)["Answer Token F1"].mean()
            if not f1.empty:
                f1["Answer Token F1"] = (f1["Answer Token F1"] * 100).round(1)
                st.bar_chart(f1.set_index("Model"))
                st.caption("Lexical similarity to the supplied reference answer; use it as an automated signal, not as a replacement for factual review.")
            else:
                st.info("No reference answers were supplied, so answer F1 is unavailable.")

        st.markdown("### Research interpretation")
        valid = summary.copy()
        fastest = valid.loc[valid["Avg latency (s)"].idxmin()]
        st.markdown(f"**Fastest:** {fastest['Model']} · {fastest['Avg latency (s)']} s average latency")
        if valid["Answer F1 (%)"].notna().any():
            scored = valid.dropna(subset=["Answer F1 (%)"])
            if not scored.empty:
                best = scored.loc[scored["Answer F1 (%)"].idxmax()]
                st.markdown(f"**Highest automated answer similarity:** {best['Model']} · {best['Answer F1 (%)']}%")
        st.caption("For a defensible conclusion, interpret answer quality, source hit rate and latency together. Keep documents, questions, K and temperature constant across models.")

        with st.expander("Raw benchmark data"):
            st.dataframe(benchmark, use_container_width=True, hide_index=True)
            st.download_button("Download benchmark results", benchmark.to_csv(index=False), "benchmark_results.csv", "text/csv")
    else:
        logs = _load_logs()
        if logs.empty:
            st.info("No research runs yet. Start chatting, or upload the benchmark CSV above.")
        else:
            st.markdown("### Existing chat-run comparison")
            logs["Response Time (sec)"] = pd.to_numeric(logs["Response Time (sec)"], errors="coerce")
            comparison = logs.groupby("Actual Model").agg(
                Runs=("Actual Model", "size"), Avg_Response=("Response Time (sec)", "mean"),
                Median_Response=("Response Time (sec)", "median"), Avg_Chunks=("Chunks Retrieved", "mean"),
                Avg_Pages=("Pages Retrieved", "mean"),
            ).reset_index()
            comparison = comparison.round(3)
            comparison.columns = ["Model", "Runs", "Avg response (s)", "Median response (s)", "Avg chunks", "Avg pages"]
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.caption("Operational performance only. Use the controlled benchmark for research-grade answer comparison.")
