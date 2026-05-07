import streamlit as st
from src.rag import ask_university_bot


st.set_page_config(
    page_title="University RAG Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 University RAG Assistant")
st.write("Ask questions about admissions, courses, tuition, deadlines, policies, and international student FAQs.")

question = st.text_input("Enter your question:")

top_k = st.slider("Number of chunks to retrieve", min_value=3, max_value=8, value=5)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching university documents..."):
            answer, sources, chunks = ask_university_bot(question, top_k=top_k)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for source in sources:
            st.write(
                f"- **{source['filename']}**, "
                f"page {source['page_number']}, "
                f"chunk {source['chunk_number']}"
            )

        with st.expander("View retrieved chunks"):
            for i, chunk in enumerate(chunks, start=1):
                metadata = chunk["metadata"]
                st.markdown(
                    f"### Chunk {i}: {metadata['filename']} | Page {metadata['page_number']}"
                )
                st.write(chunk["text"])