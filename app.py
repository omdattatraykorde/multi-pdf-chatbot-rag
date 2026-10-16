import streamlit as st

from ingest import create_vector_db
from rag import generate_summary
from components.sidebar import render_sidebar
from components.dashboard import render_dashboard
from components.chat import render_chat
from components.navbar import render_navbar
from chat_history import initialize_database
from suggestions import generate_suggested_questions

st.set_page_config(
    page_title="Multi PDF ChatBot",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_database()

for key, default in {
    "vector_db": None,
    "experiment_data": None,
    "summary": None,
    "composer_model": None,
    "messages": [],
    "last_metrics": None,
    "suggested_questions": [],
    "question_widget_version": 0,
    "voice_widget_version": 0,
    "pending_transcript": "",
    "voice_hash": None,
    "upload_signature": None,
    "generate_summary_requested": False,
    "is_processing": False,
    "chat_paused": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility:hidden;}
    [data-testid="stHeader"] {background:transparent;}
    [data-testid="stAppViewContainer"] {background:#090b0f;}
    [data-testid="stSidebar"] {background:#0d1016;border-right:1px solid #20242d;}
    [data-testid="stSidebarContent"] {position:relative;padding:0;overflow-x:hidden;}
    [data-testid="stSidebarUserContent"] {padding:0 18px 18px;}
    .block-container {max-width:1420px;padding:24px 42px 170px;}

    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.brand) {position:sticky;top:76px;z-index:1000;background:#0d1016;}
    [data-testid="stSidebar"] .brand {position:relative;z-index:1000;width:auto;box-sizing:border-box;display:flex;gap:10px;align-items:center;padding:16px 2px 13px;margin:0 0 18px;background:#0d1016;border-bottom:1px solid #232731;}
    [data-testid="stSidebar"][aria-expanded="false"] .brand {display:none;}
    .brand-mark {width:36px;height:36px;border-radius:11px;background:#f3f5f8;color:#11151c;display:flex;align-items:center;justify-content:center;box-shadow:0 6px 18px rgba(0,0,0,.2);}
    .logo-mark svg {width:27px;height:27px;display:block;}
    .brand-title {font-size:14px;font-weight:750;color:#f2f4f8;letter-spacing:-.2px;white-space:nowrap;}
    .brand-sub {font-size:10px;color:#7f8796;margin-top:3px;}
    .side-section {font-size:9px;letter-spacing:1.5px;color:#6f7888;font-weight:750;margin:20px 0 9px;}
    .file-count {font-size:11px;color:#9ca5b4;margin:8px 0 10px;}
    .file-count strong {color:#e4e8ee;}
    .file-row {display:flex;align-items:center;gap:9px;padding:9px 10px;background:#141821;border:1px solid #242934;border-radius:9px;margin:6px 0;color:#cfd5df;overflow:hidden;transition:border-color .15s,background .15s;}.file-row:hover{background:#181d27;border-color:#3a4352;}
    .file-icon {font-size:8px;font-weight:800;letter-spacing:.5px;color:#aeb7c5;border:1px solid #414856;border-radius:5px;padding:3px 4px;flex:0 0 auto;}
    .file-info {min-width:0;display:flex;flex-direction:column;line-height:1.25;}
    .file-info b {font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#d6dbe4;font-weight:600;}
    .file-info small {font-size:9px;color:#70798a;margin-top:3px;}
    .muted-box {padding:13px;border:1px dashed #303641;border-radius:10px;color:#a0a8b6;font-size:11px;line-height:1.5;background:#10131a;}
    .muted-box span {color:#727b8b;}
    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]{background:#10131a!important;border:1px dashed #303641!important;border-radius:10px!important;padding:10px!important;}
    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] button{background:#1b202a!important;color:#e5e9f0!important;border:1px solid #424b5b!important;border-radius:9px!important;font-size:12px!important;}
    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] small{color:#737e8f!important;}
    .status-card {padding:11px 12px;border:1px solid #272d38;border-radius:10px;background:#12151c;}
    .status-card.ready {border-color:#244536;background:#101a15;}.status-card.pending {border-color:#4a3c22;background:#18150f;}
    .status-title {font-size:11px;color:#e0e5ec;font-weight:650;display:flex;align-items:center;gap:8px;}
    .status-copy {font-size:9px;color:#77808f;line-height:1.45;margin:6px 0 0 15px;}
    .history-row {padding:9px 10px;margin:5px 0;border:1px solid #252c38;border-radius:8px;background:#11151d;overflow:hidden;transition:border-color .15s,background .15s;}.history-row:hover{background:#181d27;border-color:#3a4352;}
    .history-row b {display:block;color:#dce2eb;font-size:10px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    .history-row small {display:block;color:#707b8b;font-size:8px;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    [data-testid="stSidebar"] .history-row + div button {min-height:24px;padding:0 4px;background:transparent;border:0;color:#7f8b9e;font-size:12px;}
    [data-testid="stSidebar"] .history-row + div button:hover {background:#202936;color:#edf2f8;border:0;}
    .dot {width:7px;height:7px;border-radius:50%;display:inline-block;flex:0 0 auto;}.green{background:#4bd18e}.red{background:#e26b78}.amber{background:#e1ad58}
    .sidebar-footer {color:#5e6776;font-size:9px;margin-top:22px;padding-top:14px;border-top:1px solid #232731;}

    .workspace-mini {display:flex;justify-content:space-between;align-items:center;margin:0 0 8px;padding:2px 0 10px;border-bottom:1px solid #20242d;}
    .workspace-mini-label {font-size:9px;letter-spacing:1.7px;color:#758094;font-weight:800;}
    .workspace-title-block {min-width:0;}.eyebrow{font-size:9px;letter-spacing:1.7px;color:#758094;font-weight:800;margin-bottom:6px;}
    .workspace-title {font-size:29px;font-weight:780;color:#f2f4f8;letter-spacing:-.9px;line-height:1.1;}
    .workspace-subtitle {font-size:11px;color:#7f8898;margin-top:7px;}
    .workspace-badge {display:none;}
    .badge-dot,.context-dot {display:inline-block;width:6px;height:6px;border-radius:50%;background:#55d59a;vertical-align:middle;margin-right:5px;}

    [data-testid="stTabs"] {margin-top:0;}.stTabs [data-baseweb="tab-list"]{gap:24px;border-bottom:1px solid #20242d;}.stTabs [data-baseweb="tab"]{height:42px;padding:0;color:#828b9b;font-size:12px;font-weight:650;}.stTabs [aria-selected="true"]{color:#f0f3f7;}.stTabs [data-baseweb="tab-highlight"]{height:2px;background:#f0f2f5;}
    .chat-context {display:flex;align-items:center;gap:7px;margin:18px 0 16px;color:#8f98a8;font-size:10px;}.chat-context.muted{color:#737c8c;}.chat-context .context-divider{color:#4c5360;}
    .empty-state {margin:88px auto 28px;max-width:640px;text-align:center;padding:38px 38px;border:1px solid #252a34;border-radius:18px;background:linear-gradient(180deg,#10131a,#0e1117);}
    .empty-logo {width:54px;height:54px;margin:0 auto 17px;border-radius:16px;background:#f3f5f8;position:relative;box-shadow:0 8px 24px rgba(0,0,0,.2);}
    .empty-logo-page {position:absolute;width:22px;height:27px;left:13px;top:11px;border:2px solid #161a22;border-radius:5px;}
    .empty-logo-page:after {content:"";position:absolute;left:5px;right:5px;top:7px;height:2px;background:#161a22;box-shadow:0 6px 0 #161a22;}
    .empty-logo-bubble {position:absolute;right:8px;bottom:8px;width:18px;height:14px;border:2px solid #161a22;border-radius:7px;background:#f3f5f8;display:flex;align-items:center;justify-content:center;gap:2px;}
    .empty-logo-bubble:after {content:"";position:absolute;left:2px;bottom:-5px;border-width:4px 4px 0 0;border-style:solid;border-color:#161a22 transparent transparent transparent;}
    .empty-logo-bubble span {width:2px;height:2px;background:#161a22;border-radius:50%;}
    .empty-state h3{color:#eef1f5;font-size:21px;margin:0 0 8px;}.empty-state p{color:#7f8898;font-size:12px;line-height:1.6;margin:0 auto;max-width:490px;}.empty-points{display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-top:18px;}.empty-points span{font-size:9px;color:#a1a9b7;border:1px solid #2a303a;background:#151922;padding:5px 8px;border-radius:999px;}
    .suggest-label{font-size:9px;letter-spacing:1.4px;color:#70798a;font-weight:800;margin:18px 0 8px;}.stButton button{border-radius:10px;}
    [data-testid="stChatMessage"]{background:transparent;border:0;padding:8px 0;}.stChatMessage [data-testid="stChatMessageContent"]{font-size:13px;line-height:1.7;}.stChatMessage [data-testid="stChatMessageAvatar"]{background:#1a202b;border:1px solid #3a4658;color:#aebbd0;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,.18);}.composer-spacer{height:92px;}

    .st-key-composer{position:fixed;z-index:1000;left:calc(50% + 12.5vw);transform:translateX(-50%);bottom:12px;width:min(900px,calc(75vw - 24px));background:rgba(15,18,24,.99);border:1px solid #343a46;border-radius:18px;padding:7px 9px 6px;box-shadow:0 14px 42px rgba(0,0,0,.48);backdrop-filter:blur(18px);}
    .st-key-composer [data-testid="stForm"]{border:0!important;padding:0!important;background:transparent!important;}
    .st-key-composer .stSelectbox>div,.st-key-composer .stTextInput [data-testid="stTextInputRootElement"]{background:#171b23!important;border:1px solid #303847!important;border-radius:12px!important;transition:border-color .15s,box-shadow .15s;}
    .st-key-composer .stTextInput [data-testid="stTextInputRootElement"]:focus-within{border-color:#66748b!important;box-shadow:0 0 0 3px rgba(105,120,145,.14);}
    .st-key-composer .stSelectbox>div>div{min-height:36px;}.st-key-composer input{color:#eef1f5;font-size:11px;height:36px;padding-left:13px;}.st-key-composer input::placeholder{color:#7c8798;opacity:1;font-size:11px;}
    .st-key-composer .stButton button{border-radius:10px;background:#e8edf5;color:#151a23;border:1px solid #c8d1df;font-weight:650;}.stButton button:hover{background:#d8e1ee;border-color:#aebbd0;}
    .stButton button[kind="primary"]{background:#e8edf5;color:#151a23;border-color:#c8d1df;}
    [data-testid="stSidebar"] button[kind="primary"]{background:#e8edf5!important;color:#151a23!important;border-color:#c8d1df!important;}
    [data-baseweb="tag"]{background:#dbe3ef!important;color:#1a2230!important;}
    [data-baseweb="tag"] svg{fill:#526176!important;}
    .st-key-composer button[kind="secondaryFormSubmit"],.st-key-composer .stButton button{height:36px;border-radius:10px;background:#eef1f6;color:#11151c;border:0;font-size:16px;font-weight:750;}
    .st-key-composer [data-testid="stToggle"]{font-size:10px;color:#7c8798;margin:0 0 5px 2px;}
    .st-key-composer [data-testid="stToggle"] label{display:none;}
    .st-key-composer [data-testid="stAudioInput"] button{min-height:36px;height:36px;width:36px;padding:0;border-radius:50%;background:transparent;color:#dbe5f3;border:0;box-shadow:none;transition:background .15s,color .15s,transform .15s;}.st-key-composer [data-testid="stAudioInput"] button:hover{background:#263144;color:#fff;transform:translateY(-1px);}
    .st-key-composer [data-testid="stAudioInput"]{margin:0!important;overflow:visible;background:transparent!important;border:0!important;box-shadow:none!important;}.st-key-composer [data-testid="stAudioInput"] small{display:none;}
    .composer-meta{display:none;}
    [data-testid="stChatMessage"] + [data-testid="stChatMessage"]{border-top:1px solid #1d2430;margin-top:14px;padding-top:20px;}

    [data-testid="stMetric"]{background:#12161d;border:1px solid #252b35;border-radius:11px;padding:11px;}.section-kicker{font-size:9px;letter-spacing:1.5px;color:#727c8c;font-weight:800;margin-top:8px;}div[data-testid="stDataFrame"]{border:1px solid #272d37;border-radius:10px;overflow:hidden;}
    @media (max-width: 900px){.block-container{padding-left:20px;padding-right:20px}.workspace-badge{display:none}.st-key-composer{left:calc(50% + 12.5vw);width:calc(75vw - 24px)}.workspace-title{font-size:23px}}
    @media (max-width: 620px){.st-key-composer{bottom:8px;padding:8px}.st-key-composer [data-testid="stForm"] [data-testid="column"]{padding:0 3px}.composer-meta{display:none}.composer-spacer{height:70px}}
    </style>
    """,
    unsafe_allow_html=True,
)

uploaded_files, process_clicked, generate_summary_clicked, selected_model = render_sidebar()

# A changed upload selection invalidates the old index and old starter questions.
current_signature = tuple(sorted((pdf.name, pdf.size) for pdf in (uploaded_files or [])))
if current_signature != st.session_state.upload_signature:
    if st.session_state.upload_signature is not None:
        st.session_state.vector_db = None
        st.session_state.experiment_data = None
        st.session_state.messages = []
        st.session_state.summary = None
        st.session_state.suggested_questions = []
        st.session_state.last_metrics = None
    st.session_state.upload_signature = current_signature

if process_clicked:
    if not uploaded_files:
        st.error("Please upload at least one PDF before processing.")
    else:
        try:
            with st.spinner("Processing documents and building the searchable index…"):
                # Pass the selected provider to ingestion so Gemini can
                # optionally describe extracted visuals.  Ingestion remains
                # text/OCR-only when the provider is unavailable.
                ingest_model = selected_model or "Gemini 2.5 Flash"
                vector_db, experiment_data = create_vector_db(
                    uploaded_files,
                    model_name=ingest_model,
                )
            st.session_state.vector_db = vector_db
            st.session_state.experiment_data = experiment_data
            st.session_state.messages = []
            st.session_state.summary = None
            selected = st.session_state.get("composer_model") or "Gemini 2.5 Flash"
            with st.spinner("Preparing document-specific starter questions…"):
                st.session_state.suggested_questions = generate_suggested_questions(vector_db, selected, limit=4)
            st.success(f"Workspace ready · {experiment_data['total_pdfs']} PDF(s) · {experiment_data['total_pages']} page(s) indexed.")
        except Exception as exc:
            st.error(f"Document processing failed: {exc}")

chat_tab, research_tab = render_navbar()

with chat_tab:
    render_chat()

with research_tab:
    render_dashboard(st.session_state.experiment_data)
    if generate_summary_clicked and st.session_state.vector_db is not None:
        with st.spinner("Generating document summary…"):
            st.session_state.summary = generate_summary(
                st.session_state.vector_db,
                st.session_state.get("composer_model") or "Gemini 2.5 Flash",
            )
    if st.session_state.summary:
        st.markdown("### AI-generated summary")
        st.markdown(st.session_state.summary)
