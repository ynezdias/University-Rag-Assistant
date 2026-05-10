import streamlit as st
from src.rag import ask_university_bot

st.set_page_config(
    page_title="Stevens RAG Assistant",
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

def render_answer(text: str):
    lines = text.strip().split("\n")
    preferred_lines, conflict_lines, body_lines = [], [], []
    mode = "body"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("✓"):
            mode = "preferred"
        elif stripped.startswith("⚠"):
            mode = "conflict"
        elif stripped == "" and mode in ("preferred", "conflict"):
            if mode == "preferred":
                preferred_lines.append(line)
            else:
                conflict_lines.append(line)
            continue

        if mode == "preferred":
            preferred_lines.append(line)
        elif mode == "conflict":
            conflict_lines.append(line)
        else:
            body_lines.append(line)

    if preferred_lines:
        content = "\n".join(preferred_lines).strip()
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(40,160,90,0.10),rgba(20,120,60,0.06));
                    border:1px solid rgba(40,160,90,0.35);border-left:3px solid #28a05a;
                    border-radius:10px;padding:1rem 1.2rem;margin-bottom:1rem;
                    font-size:0.9rem;color:#7ddba0;line-height:1.75;">
            <strong style="color:#4cc87a;font-size:0.72rem;letter-spacing:0.08em;
                           text-transform:uppercase;display:block;margin-bottom:0.4rem;">
                ✓ Preferred Answer (most recent source)
            </strong>{content}</div>""", unsafe_allow_html=True)

    if conflict_lines:
        content = "\n".join(conflict_lines).strip()
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(220,80,50,0.10),rgba(180,50,30,0.06));
                    border:1px solid rgba(220,80,50,0.35);border-left:3px solid #dc5032;
                    border-radius:10px;padding:1rem 1.2rem;margin-bottom:1rem;
                    font-size:0.88rem;color:#f0a090;line-height:1.75;">
            <strong style="color:#e8705a;font-size:0.72rem;letter-spacing:0.08em;
                           text-transform:uppercase;display:block;margin-bottom:0.4rem;">
                ⚠ Conflict Detected
            </strong>{content}</div>""", unsafe_allow_html=True)

    body = "\n".join(body_lines).strip()
    if body:
        st.markdown(f'<div class="answer-card">{body}</div>', unsafe_allow_html=True)

    if not preferred_lines and not conflict_lines and not body:
        st.markdown('<div class="answer-card">I don\'t know based on the university documents.</div>', unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
  <div class="header-crest">🎓</div>
  <div class="header-text">
    <h1>University RAG Assistant</h1>
    <p>Stevens Institute of Technology · Document Intelligence</p>
  </div>
  <div class="header-badge">AI-Powered</div>
</div>
""", unsafe_allow_html=True)


# ── Query bar ──────────────────────────────────────────────────────────────────
st.markdown('<div class="query-panel">', unsafe_allow_html=True)
st.markdown('<div class="query-label">Ask a question about Stevens documents</div>', unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        label="question",
        label_visibility="collapsed",
        placeholder="e.g. What is the graduate fall priority deadline?",
        key="question_input",
    )
with col_btn:
    ask = st.button("Ask →", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Results ────────────────────────────────────────────────────────────────────
col_ans, col_src = st.columns([1, 1])

if ask and question.strip():
    with st.spinner("Searching documents..."):
        answer, chunks = ask_university_bot(question.strip())

    with col_ans:
        st.markdown('<div class="panel-label">Answer</div>', unsafe_allow_html=True)
        render_answer(answer)

    with col_src:
        st.markdown('<div class="panel-label">Sources</div>', unsafe_allow_html=True)
        for i, chunk in enumerate(chunks, 1):
            meta  = chunk["metadata"]
            fname = meta.get("filename", "unknown")
            page  = meta.get("page_number", "?")
            chunk_num = meta.get("chunk_number", "?")
            st.markdown(f"""
            <div class="source-chip">
              <div class="source-chip-num">{i}</div>
              <div>
                <div class="source-chip-file">{fname}</div>
                <div class="source-chip-page">Page {page} · Chunk {chunk_num}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        with st.expander(f"View {len(chunks)} retrieved chunks"):
            for i, chunk in enumerate(chunks, 1):
                meta = chunk["metadata"]
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<span style="color:#c89b3c;font-weight:600">Chunk {i}</span>'
                    f' · {meta.get("filename","?")} '
                    f'p.{meta.get("page_number","?")} '
                    f'c.{meta.get("chunk_number","?")}<br><br>'
                    f'{chunk["text"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

else:
    with col_ans:
        st.markdown('<div class="panel-label">Answer</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="placeholder">
          <div class="placeholder-icon">◈</div>
          <div class="placeholder-text">
            Type a question about admissions, tuition, deadlines,
            courses, or international student policies.
          </div>
        </div>""", unsafe_allow_html=True)

    with col_src:
        st.markdown('<div class="panel-label">Sources</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="placeholder">
          <div class="placeholder-icon">◇</div>
          <div class="placeholder-text">
            Retrieved document chunks will appear here after you ask a question.
          </div>
        </div>""", unsafe_allow_html=True)