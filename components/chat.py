import re

import streamlit as st

from llm_factory import LLMFactory
from rag import ask_question
from voice_input import transcribe_audio, audio_hash


def _sources(docs):
    shown = set()
    rows = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown PDF")
        page = int(doc.metadata.get("page", 0)) + 1
        key = (source, page)
        if key not in shown:
            shown.add(key)
            rows.append((source, page))
    return rows


def _display_answer(answer):
    """Keep source details in the expandable source panel, not twice on screen."""
    return re.sub(r"\s*\[Source:[^\]]+\]", "", answer or "").strip()


def _handle_question(question, model):
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        answer, docs, used_model, metrics = ask_question(st.session_state.vector_db, question, model)
        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "model": used_model,
            "docs": docs, "metrics": metrics,
        })
        st.session_state.last_metrics = metrics
    except Exception as exc:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "I couldn't generate an answer right now. "
                "Please check your model/API configuration and try again.\n\n"
                f"Details: `{exc}`"
            ),
            "model": model, "error": True,
        })


def _submit_question(question, model):
    if st.session_state.get("is_processing"):
        return
    st.session_state.is_processing = True
    try:
        with st.spinner("Searching your documents and generating an answer…"):
            _handle_question(question, model)
    finally:
        st.session_state.is_processing = False


def render_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = None
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []
    if "question_widget_version" not in st.session_state:
        st.session_state.question_widget_version = 0
    if "voice_hash" not in st.session_state:
        st.session_state.voice_hash = None
    if "voice_widget_version" not in st.session_state:
        st.session_state.voice_widget_version = 0
    if "pending_transcript" not in st.session_state:
        st.session_state.pending_transcript = ""
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "chat_paused" not in st.session_state:
        st.session_state.chat_paused = False

    models = LLMFactory.get_available_models()
    if "composer_model" not in st.session_state or st.session_state.composer_model not in models:
        st.session_state.composer_model = models[0]

    ready = st.session_state.vector_db is not None
    voice_key = f"voice_input_{st.session_state.voice_widget_version}"
    question_key = f"question_input_{st.session_state.question_widget_version}"
    transcript_default = st.session_state.pop("pending_transcript", "")

    if not ready:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-logo">
                    <div class="empty-logo-page"></div>
                    <div class="empty-logo-bubble"><span></span><span></span><span></span></div>
                </div>
                <h3>Chat with your PDFs</h3>
                <p>Upload one or more documents, process them, and ask questions naturally. The chatbot retrieves relevant passages before answering.</p>
                <div class="empty-points"><span>Grounded answers</span><span>Page sources</span><span>Voice input</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if not st.session_state.messages:
            st.markdown('<div class="suggest-label">TRY ASKING</div>', unsafe_allow_html=True)
            suggestions = st.session_state.suggested_questions or [
                "What is the main objective?",
                "What are the most important points?",
                "What key details should I know?",
            ]
            cols = st.columns(min(len(suggestions), 4))
            for i, suggestion in enumerate(suggestions):
                if cols[i].button(
                    suggestion,
                    key=f"suggestion_{i}",
                    use_container_width=True,
                    disabled=st.session_state.is_processing or st.session_state.chat_paused,
                ):
                    _submit_question(suggestion, st.session_state.composer_model)
                    st.rerun()

        for message in st.session_state.messages:
            role = message["role"]
            avatar = ":material/person:" if role == "user" else ":material/auto_awesome:"
            with st.chat_message(role, avatar=avatar):
                st.markdown(
                    _display_answer(message["content"])
                    if role == "assistant"
                    else message["content"]
                )
                if role == "assistant" and not message.get("error"):
                    docs = message.get("docs", [])
                    if docs:
                        with st.expander(f"Sources · {len(_sources(docs))} page(s)", expanded=False):
                            for source, page in _sources(docs):
                                st.markdown(f"**{source}**  ·  page {page}")

    st.markdown("<div class='composer-spacer'></div>", unsafe_allow_html=True)
    composer = st.container(key="composer")
    with composer:
        processing = st.session_state.is_processing
        c1, c2, c3, c4 = st.columns([1.7, 7.4, 0.7, 0.7], vertical_alignment="center")
        with c1:
            st.selectbox("Model", models, key="composer_model", label_visibility="collapsed", disabled=processing)
        with c2:
            st.text_input(
                "Ask your documents",
                value=transcript_default,
                placeholder="Message Multi PDF ChatBot…",
                key=question_key,
                label_visibility="collapsed",
                disabled=processing or st.session_state.chat_paused,
            )
        with c3:
            audio = st.audio_input(
                "Voice",
                sample_rate=16000,
                key=voice_key,
                label_visibility="collapsed",
                help="Record a short question. It will be transcribed into the text box.",
                disabled=processing or st.session_state.chat_paused,
            )
        with c4:
            action_label = "Ⅱ" if processing else ("▶" if st.session_state.chat_paused else "↑")
            submitted = st.button(
                action_label,
                key="send_question",
                use_container_width=True,
                disabled=False,
                help=(
                    "Pause response" if processing
                    else ("Resume chat" if st.session_state.chat_paused else "Send question")
                ),
            )
            if submitted and processing:
                st.session_state.chat_paused = True
                st.rerun()
            if submitted and st.session_state.chat_paused and not processing:
                st.session_state.chat_paused = False
                st.rerun()

    if audio and audio_hash(audio) != st.session_state.get("voice_hash"):
        st.session_state.voice_hash = audio_hash(audio)
        try:
            with st.spinner("Transcribing…"):
                transcript = transcribe_audio(audio)
            if transcript:
                st.session_state.pending_transcript = transcript
                st.session_state.question_widget_version += 1
                st.session_state.voice_widget_version += 1
                st.rerun()
            st.warning("I couldn't detect any speech. Please try again.")
        except Exception as exc:
            st.error(f"Voice transcription unavailable: {exc}")

    if submitted and ready:
        question = st.session_state.get(question_key, "").strip()

        if not question:
            st.warning("Type a question or record one with the microphone.")
        else:
            _submit_question(question, st.session_state.composer_model)
            # Widget values cannot be assigned after creation. New keys clear
            # both composer controls safely on the next rerun.
            st.session_state.pending_transcript = ""
            st.session_state.question_widget_version += 1
            st.session_state.voice_widget_version += 1
            st.rerun()
    elif submitted and not ready:
        st.warning("Process at least one PDF before asking a question.")
