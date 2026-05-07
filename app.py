import streamlit as st

from src.rag import ask_university_bot


st.set_page_config(page_title="University RAG Assistant")

st.title("🎓 University RAG Assistant")

question = st.text_input(
    "Ask a university-related question:"
)

if st.button("Ask"):
    with st.spinner("Searching documents..."):

        answer, chunks = ask_university_bot(question)

    st.subheader("Answer")

    st.write(answer)

    st.subheader("Sources")

    for chunk in chunks:
        meta = chunk["metadata"]

        st.write(
            f"{meta['filename']} "
            f"(Page {meta['page_number']})"
        )

    with st.expander("Retrieved Chunks"):
        for chunk in chunks:
            st.write(chunk["text"])
            st.divider()