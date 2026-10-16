import streamlit as st


def render_performance(metrics=None):

    st.subheader("📈 AI Performance")

    if metrics is None:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("🤖 Model", "-")
        c2.metric("⏱ Response", "-")
        c3.metric("🧩 Chunks", "-")
        c4.metric("📄 PDFs", "-")

        c5, c6, c7 = st.columns(3)

        c5.metric("📑 Pages", "-")
        c6.metric("❓ Question", "-")
        c7.metric("💬 Answer", "-")

        return

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🤖 Model",
        metrics["model"]
    )

    c2.metric(
        "⏱ Response",
        f'{metrics["response_time"]} sec'
    )

    c3.metric(
        "🧩 Chunks",
        metrics["chunks"]
    )

    c4.metric(
        "📄 PDFs",
        metrics["pdfs"]
    )

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "📑 Pages",
        metrics["pages"]
    )

    c6.metric(
        "❓ Question Length",
        metrics["question_length"]
    )

    c7.metric(
        "💬 Answer Length",
        metrics["answer_length"]
    )

    st.divider()

    with st.expander("📊 Research Summary", expanded=False):

        st.markdown(f"""
**Model Used:** {metrics["model"]}

**Response Time:** {metrics["response_time"]} sec

**Chunks Retrieved:** {metrics["chunks"]}

**PDFs Retrieved:** {metrics["pdfs"]}

**Pages Retrieved:** {metrics["pages"]}

**Question Length:** {metrics["question_length"]}

**Answer Length:** {metrics["answer_length"]}
""")