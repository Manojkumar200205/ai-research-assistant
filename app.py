import streamlit as st
from src.rag_engine import ResearchAssistant
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("📄 AI Research Assistant")
st.write("Upload research paper PDFs and ask questions across them.")

# Check API key exists
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Initialize assistant in session state
if "assistant" not in st.session_state:
    st.session_state.assistant = ResearchAssistant(groq_key)
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# --- Upload PDFs (multiple) ---
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("Process PDF(s)"):
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.processed_files:
                st.info(f"{uploaded_file.name} already processed, skipping.")
                continue

            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    num_chunks = st.session_state.assistant.process_pdf(
                        tmp_path, source_name=uploaded_file.name
                    )
                    st.session_state.processed_files.append(uploaded_file.name)
                    st.success(f"{uploaded_file.name}: {num_chunks} chunks created.")

                except ValueError as e:
                    st.error(f"{uploaded_file.name}: {str(e)}")
                except Exception as e:
                    st.error(f"{uploaded_file.name}: Unexpected error - {str(e)}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

if st.session_state.processed_files:
    st.caption(f"Processed documents: {', '.join(st.session_state.processed_files)}")

# --- Ask Questions ---
if st.session_state.assistant.vectorstore is not None:
    st.divider()
    query = st.text_input("Ask a question about the uploaded paper(s):")

    if query:
        if not query.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Searching and generating answer..."):
                try:
                    answer, results = st.session_state.assistant.ask(query)
                    st.subheader("Answer")
                    st.write(answer)

                    with st.expander("View retrieved chunks (sources)"):
                        for i, r in enumerate(results):
                            source = r.metadata.get("source", "unknown")
                            st.markdown(f"**Chunk {i+1}** — *source: {source}*")
                            st.write(r.page_content[:300] + "...")
                            st.divider()
                except Exception as e:
                    st.error(f"Something went wrong while generating the answer: {str(e)}")
else:
    st.info("Upload and process at least one PDF to start asking questions.")