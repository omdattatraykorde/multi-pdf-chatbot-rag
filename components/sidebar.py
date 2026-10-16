import streamlit as st
from chat_history import delete_chat, load_chat_history


LOGO_SVG = """
<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect x="7" y="6" width="34" height="36" rx="10" fill="none" stroke="currentColor" stroke-width="2.6"/>
  <path d="M15 16h11.5c4.1 0 7.5 3.4 7.5 7.5S30.6 31 26.5 31H21l-6 5v-20Z" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round"/>
  <path d="M19 22h9M19 26h6" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>
</svg>
"""


def render_sidebar():
    st.sidebar.markdown(
        f'<div class="brand"><div class="brand-mark logo-mark">{LOGO_SVG}</div><div class="brand-title">Multi PDF ChatBot</div></div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<div class='side-section'>DOCUMENTS</div>", unsafe_allow_html=True)
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload one or more PDF documents for semantic search.",
    )

    if uploaded_files:
        st.sidebar.markdown(
            f"<div class='file-count'><strong>{len(uploaded_files)}</strong> document(s) selected</div>",
            unsafe_allow_html=True,
        )
        for pdf in uploaded_files:
            size_kb = max(1, round(pdf.size / 1024))
            st.sidebar.markdown(
                f"<div class='file-row'><span class='file-icon'>PDF</span><span class='file-info'><b>{pdf.name}</b><small>{size_kb} KB</small></span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.markdown(
            "<div class='muted-box'><b>Upload your PDFs</b><br><span>Ask questions across one or multiple documents.</span></div>",
            unsafe_allow_html=True,
        )

    process_clicked = st.sidebar.button(
        "Re-process documents" if st.session_state.get("vector_db") is not None else "Process documents",
        use_container_width=True,
        type="primary",
        disabled=not uploaded_files,
    )

    if st.session_state.get("messages"):
        if st.sidebar.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_metrics = None
            st.rerun()

    with st.sidebar.expander("More", expanded=False):
        if st.button("Generate document summary", use_container_width=True, key="sidebar_summary"):
            st.session_state.generate_summary_requested = True
            st.rerun()

    with st.sidebar.expander("History", expanded=False):
        history = load_chat_history()
        if not history:
            st.caption("No conversations yet.")
        else:
            for chat_id, timestamp, model, question, answer in history[:12]:
                label = question.strip().replace("\n", " ")
                if len(label) > 54:
                    label = f"{label[:51]}..."
                row_text, open_col, delete_col = st.columns([5.8, 1, 1], vertical_alignment="center")
                with row_text:
                    st.markdown(
                        f"<div class='history-row'><b>{label}</b><small>{timestamp} · {model}</small></div>",
                        unsafe_allow_html=True,
                    )
                with open_col:
                    if st.button("↗", key=f"open_history_{chat_id}", help="Open this chat"):
                        st.session_state.messages = [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer, "model": model, "docs": []},
                        ]
                        st.rerun()
                with delete_col:
                    if st.button("×", key=f"delete_history_{chat_id}", help="Delete this chat"):
                        delete_chat(chat_id)
                        st.rerun()

    st.sidebar.markdown("<div class='side-section'>WORKSPACE</div>", unsafe_allow_html=True)
    ready = st.session_state.get("vector_db") is not None
    if ready:
        st.sidebar.markdown("<div class='status-card ready'><div class='status-title'><span class='dot green'></span>Ready</div><div class='status-copy'>Your processed PDFs are searchable.</div></div>", unsafe_allow_html=True)
    elif uploaded_files:
        st.sidebar.markdown("<div class='status-card pending'><div class='status-title'><span class='dot amber'></span>Ready to process</div><div class='status-copy'>Build the index to start chatting.</div></div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<div class='status-card'><div class='status-title'><span class='dot red'></span>No documents</div><div class='status-copy'>Upload PDFs to start.</div></div>", unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-footer">Focused document chat · v5.0</div>', unsafe_allow_html=True)
    selected_model = st.session_state.get("composer_model")
    generate_summary = st.session_state.pop("generate_summary_requested", False)
    return uploaded_files, process_clicked, generate_summary, selected_model
