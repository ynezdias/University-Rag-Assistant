import logging
from html import escape
import streamlit as st
from src.rag import ask
from src.knowledge import DATA_DIR
import time

st.set_page_config(
    page_title="QuackQuery",
    page_icon="🎓",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0b1120; color: #e8e4d9; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

[data-testid="stSidebar"] { background: #0d1527 !important; border-right: 1px solid #1e2d4a; }
[data-testid="stSidebar"] * { color: #a0aec0 !important; }

.rag-header {
    background: linear-gradient(135deg, #0d1527 0%, #0b1120 60%, #13203a 100%);
    border-bottom: 1px solid #1e2d4a;
    padding: 2.5rem 3rem 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    position: relative;
    overflow: hidden;
}
.rag-header::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(200,155,60,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.header-crest {
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #c89b3c, #e8c56a);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem;
    flex-shrink: 0;
    box-shadow: 0 4px 24px rgba(200,155,60,0.3);
}
.header-text h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #f0ead6;
    margin: 0 0 0.15rem;
    letter-spacing: -0.02em;
}
.header-text p { font-size: 0.82rem; color: #6b7fa3; margin: 0; letter-spacing: 0.04em; text-transform: uppercase; }
.header-badge {
    margin-left: auto;
    background: rgba(200,155,60,0.1);
    border: 1px solid rgba(200,155,60,0.3);
    color: #c89b3c;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.query-panel {
    background: #0d1527;
    border-bottom: 1px solid #1e2d4a;
    padding: 2rem 3rem;
}
.query-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #6b7fa3;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.stTextInput > div > div {
    background: #111c33 !important;
    border: 1px solid #2a3a5c !important;
    border-radius: 10px !important;
    color: #e8e4d9 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
}
.stTextInput > div > div:focus-within {
    border-color: #c89b3c !important;
    box-shadow: 0 0 0 3px rgba(200,155,60,0.12) !important;
}
.stTextInput input { color: #e8e4d9 !important; }
.stTextInput input::placeholder { color: #3d5078 !important; }

.stButton > button {
    background: linear-gradient(135deg, #c89b3c, #d4a843) !important;
    color: #0b1120 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    box-shadow: 0 4px 16px rgba(200,155,60,0.25) !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

.panel-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #4a5e80;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.panel-label::after { content: ''; flex: 1; height: 1px; background: #1e2d4a; }

.answer-card {
    background: #111c33;
    border: 1px solid #1e2d4a;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    font-size: 0.97rem;
    line-height: 1.8;
    color: #d4cfc4;
}

.source-chip {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    background: #111c33;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.7rem;
    transition: border-color 0.15s;
}
.source-chip:hover { border-color: #2e4268; }
.source-chip-num {
    background: rgba(200,155,60,0.15);
    color: #c89b3c;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    width: 22px; height: 22px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.source-chip-file { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #8b9fc4; margin-bottom: 0.2rem; word-break: break-all; }
.source-chip-page { font-size: 0.7rem; color: #4a5e80; }

.chunk-card {
    background: #0d1527;
    border: 1px solid #1a2740;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    line-height: 1.7;
    color: #5a7098;
}

.placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 40vh;
    color: #2a3a5c;
    text-align: center;
    gap: 1rem;
}
.placeholder-icon { font-size: 3rem; opacity: 0.4; }
.placeholder-text { font-size: 0.9rem; line-height: 1.6; max-width: 280px; }

.stSpinner > div { border-top-color: #c89b3c !important; }
details { border: 1px solid #1e2d4a !important; border-radius: 10px !important; background: #0a1020 !important; margin-top: 1.5rem; }
summary { color: #6b7fa3 !important; font-size: 0.8rem !important; padding: 0.7rem 1rem !important; cursor: pointer; }
</style>
""", unsafe_allow_html=True)


# ── Answer renderer ────────────────────────────────────────────────────────────

st.title("QuackQuery")
st.caption("University document intelligence with inspectable evidence")

with st.sidebar:
    corpus = st.selectbox("Knowledge base", ["synthetic", "unverified"],
        format_func=lambda x: {"synthetic": "Synthetic demo", "unverified": "Original documents (unverified)"}[x])
    st.info("Fictional test documents, not official university guidance." if corpus == "synthetic"
            else "These documents have not been verified against official university sources.")
    st.caption("Conversations stay in this browser session. Each knowledge base has its own history.")
    if st.button("Clear conversation"):
        st.session_state.setdefault("conversations", {})[corpus] = []
    st.caption("Citations are checked for valid sources and matching quotations. This does not guarantee that every claim is correct.")

conversations = st.session_state.setdefault("conversations", {})
messages = conversations.setdefault(corpus, [])


def render_result(result, turn):
    response = result["response"]
    if response["status"] == "answered":
        for number, claim in enumerate(response["claims"], 1):
            st.markdown(claim["text"])
            for evidence_number, evidence in enumerate(claim["evidence"], 1):
                source = result["chunks"][evidence["source_id"] - 1]
                meta = source["metadata"]
                with st.expander(f"Source {evidence['source_id']}: {meta['filename']} - {meta.get('locator', 'Document')}"):
                    st.text(evidence["quote"])
                    st.caption(f"Type: {meta.get('source_type', 'unverified')} | Published: {meta.get('publication_date', 'unknown')}")
                    st.text(source["text"])
                    path = (DATA_DIR / meta.get("document_id", "")).resolve()
                    if path.is_relative_to(DATA_DIR.resolve()) and path.is_file() and path.suffix.lower() in (".pdf", ".docx"):
                        st.download_button("Download source", path.read_bytes(), file_name=path.name,
                            key=f"download_{corpus}_{turn}_{number}_{evidence_number}")
    else:
        st.write(response["message"])
    st.caption(f"Response time: {result['seconds']:.1f}s")


for index, message in enumerate(messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            render_result(message["result"], index)

question = st.chat_input("Ask about requirements, courses, or a specific academic year", max_chars=1000)
if question:
    now = time.monotonic()
    recent = [timestamp for timestamp in st.session_state.get("requests", []) if now - timestamp < 60]
    if len(recent) >= 6:
        st.warning("Please wait a moment. This demo allows six questions per minute per session.")
    else:
        st.session_state["requests"] = recent + [now]
        history = [{"role": m["role"], "content": m["content"]} for m in messages[-6:]]
        with st.spinner("Finding evidence..."):
            try:
                result = ask(question, corpus=corpus, history=history)
            except Exception:
                logging.exception("QuackQuery request failed")
                st.error("The answer service is unavailable. Please try again shortly.")
            else:
                messages.extend([{"role": "user", "content": question},
                                 {"role": "assistant", "content": result["answer"], "result": result}])
                conversations[corpus] = messages[-20:]
                st.rerun()
