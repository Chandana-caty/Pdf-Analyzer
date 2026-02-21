# import os

# # 🔹 Fix OpenMP duplicate runtime issue
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import streamlit as st
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv

# import google.generativeai as genai
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS  # ✅ updated import
# # from langchain.chains.question_answering import load_qa_chain
# from langchain.chains import load_qa_chain
# from langchain.prompts import PromptTemplate

# # ------------------ CONFIG ------------------ #

# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")

# if not api_key:
#     raise ValueError("GOOGLE_API_KEY is not set in .env file")

# genai.configure(api_key=api_key)

# EMBEDDING_MODEL = "models/text-embedding-004"
# CHAT_MODEL = "gemini-2.5-flash"
# FAISS_INDEX_PATH = "faiss_index"


# # ------------------ PDF HANDLING ------------------ #

# def get_pdf_text(pdf_docs):
#     """Extract text from uploaded PDF files."""
#     text = ""
#     if not pdf_docs:
#         return text

#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             page_text = page.extract_text() or ""
#             text += page_text
#     return text


# def get_text_chunks(text: str):
#     """Split large text into overlapping chunks."""
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=10000,
#         chunk_overlap=1000
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# # ------------------ VECTOR STORE ------------------ #

# def get_vector_store(text_chunks):
#     """Create and save a FAISS vector store from text chunks."""
#     if not text_chunks:
#         raise ValueError("No text chunks to index.")

#     embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local(FAISS_INDEX_PATH)
#     return vector_store


# def load_vector_store():
#     """Load FAISS vector store from disk if it exists."""
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
#         vector_store = FAISS.load_local(
#             FAISS_INDEX_PATH,
#             embeddings,
#             allow_dangerous_deserialization=True
#         )
#         return vector_store
#     except Exception:
#         return None


# # ------------------ QA CHAIN ------------------ #

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. 
#     Make sure to provide all the details. 
#     If the answer is not in the provided context, just say:
#     "answer is not available in the context"
#     and don't provide a wrong answer.

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(
#         model=CHAT_MODEL,
#     )

#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"]
#     )

#     chain = load_qa_chain(
#         model,
#         chain_type="stuff",
#         prompt=prompt
#     )
#     return chain


# # ------------------ USER QUERY HANDLING ------------------ #

# def user_input(user_question: str):
#     vector_store = load_vector_store()
#     if vector_store is None:
#         st.error("⚠️ Please upload PDFs and click 'Submit & Process' before asking questions.")
#         return

#     embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

#     new_db = FAISS.load_local(
#         FAISS_INDEX_PATH,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()
#     response = chain(
#         {"input_documents": docs, "question": user_question},
#         return_only_outputs=True
#     )

#     st.write("**Your question:**")
#     st.write(user_question)
#     st.write("**Answer:**")
#     st.write(response["output_text"])


# # ------------------ STREAMLIT APP ------------------ #

# def main():
#     st.set_page_config("Chat with PDFs", page_icon="📂")

#     # Header
#     st.header(":rainbow[Chat with PDFs :material/docs:]")
#     # User Input
#     user_question = st.text_input("🟧 Ask a question from the uploaded PDF files:")
#     if user_question:
#         user_input(user_question)

#     # Sidebar
#     with st.sidebar:
#         st.subheader("📗 Upload Your Documents")
#         st.title("📘 Menu")

#         pdf_docs = st.file_uploader(
#             "📕 Upload your PDF files and click on the 'Submit & Process' button",
#             accept_multiple_files=True
#         )

#         if st.button("Submit & Process"):
#             if not pdf_docs:
#                 st.error("Please upload at least one PDF file.")
#             else:
#                 with st.spinner("Processing PDFs..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     if not raw_text.strip():
#                         st.error("No text could be extracted from the PDFs.")
#                     else:
#                         text_chunks = get_text_chunks(raw_text)
#                         get_vector_store(text_chunks)
#                         st.success("✅ Processing complete! You can now ask questions.")


# if __name__ == "__main__":
#     main()


import os
import pickle
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import faiss

import google.generativeai as genai

# ------------------ CONFIG ------------------ #
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in .env file")

genai.configure(api_key=api_key)

EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"

INDEX_PATH = "faiss.index"
META_PATH = "faiss_meta.pkl"


# ------------------ HELPERS ------------------ #
def read_pdfs(pdf_files) -> str:
    text = ""
    for f in pdf_files:
        reader = PdfReader(f)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    return text



def split_text(text: str, chunk_size: int = 2000, overlap: int = 200):
    if not text.strip():
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break
    return chunks


def embed_texts(texts):
    # google-generativeai embed_content supports batching via list input in many setups,
    # but to be safest on Streamlit Cloud, do one-by-one.
    vectors = []
    for t in texts:
        resp = genai.embed_content(model=EMBED_MODEL, content=t)
        vec = np.array(resp["embedding"], dtype="float32")
        vectors.append(vec)
    return np.vstack(vectors)


def build_faiss_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (works well if normalized)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index


def save_index(index, chunks):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": chunks}, f)


def load_index():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta["chunks"]


def retrieve_chunks(index, chunks, query: str, k: int = 4):
    qvec = genai.embed_content(model=EMBED_MODEL, content=query)["embedding"]
    qvec = np.array(qvec, dtype="float32")[None, :]
    faiss.normalize_L2(qvec)

    scores, ids = index.search(qvec, k)
    results = []
    for i in ids[0]:
        if i == -1:
            continue
        results.append(chunks[i])
    return results


def ask_gemini(context_chunks, question: str):
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say exactly: "answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""
    model = genai.GenerativeModel(CHAT_MODEL)
    resp = model.generate_content(prompt)
    return resp.text


# ------------------ STREAMLIT APP ------------------ #
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📂")
    st.header("📂 Chat with PDFs")

    user_question = st.text_input("Ask a question from the uploaded PDF files:")

    with st.sidebar:
        st.subheader("Upload Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF files, then click Submit & Process",
            accept_multiple_files=True,
            type=["pdf","txt"],
        )
     

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
                return

            with st.spinner("Reading PDFs..."):
                text = read_pdfs(pdf_docs)
               

            if not text.strip():
                st.error("No text could be extracted from the PDFs.")
                return

            chunks = split_text(text, chunk_size=2000, overlap=200)

            with st.spinner("Creating embeddings + FAISS index..."):
                vectors = embed_texts(chunks)
                index = build_faiss_index(vectors)
                save_index(index, chunks)

            st.success("✅ Processing complete! You can now ask questions.")

    if user_question:
        index, chunks = load_index()
        if index is None:
            st.error("⚠️ Upload PDFs and click 'Submit & Process' first.")
            return

        with st.spinner("Searching + generating answer..."):
            top_chunks = retrieve_chunks(index, chunks, user_question, k=4)
            answer = ask_gemini(top_chunks, user_question)

        st.write("**Answer:**")
        st.write(answer)


if __name__ == "__main__":
    main()
