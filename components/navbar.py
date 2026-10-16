import streamlit as st


def render_navbar():
    st.markdown(
        """
        <div class="workspace-mini">
            <span class="workspace-mini-label">DOCUMENT CHAT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.tabs(["Chat", "Research"])
