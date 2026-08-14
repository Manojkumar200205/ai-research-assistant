from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import re
import os


class ResearchAssistant:
    def __init__(self, groq_api_key):
        self.client = Groq(api_key=groq_api_key)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None

    def extract_text(self, pdf_path):
        """Extract and clean text from a PDF file."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # Remove references section
        text = re.split(r'\n\s*REFERENCES?\s*\n', text, flags=re.IGNORECASE)[0]
        # Clean PDF artifacts
        text = re.sub(r'/\d+\s*/\d+', '', text)
        text = re.sub(r'-\s+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def chunk_text(self, text):
        """Split text into overlapping chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)

    def process_pdf(self, pdf_path, source_name="document"):
        """Full pipeline: extract, chunk, embed, and add to vectorstore."""
        text = self.extract_text(pdf_path)
        if not text.strip():
            raise ValueError("No text could be extracted from this PDF. It may be scanned or image-based.")

        chunks = self.chunk_text(text)
        # Tag each chunk with its source document
        docs_with_metadata = [(chunk, {"source": source_name}) for chunk in chunks]
        texts = [d[0] for d in docs_with_metadata]
        metadatas = [d[1] for d in docs_with_metadata]

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            self.vectorstore.add_texts(texts, metadatas=metadatas)

        return len(chunks)

    def ask(self, query, k=10):
        """Retrieve relevant chunks and generate an answer."""
        if self.vectorstore is None:
            raise ValueError("No documents have been processed yet.")

        results = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([r.page_content for r in results])

        prompt = f"""Answer the question using only the context below.
        If the context doesn't contain the answer, say so clearly.

        Context:
        {context}

        Question: {query}
        """

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content, results