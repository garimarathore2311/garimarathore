import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import openai

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "0f436efbf1ed4315822b1c680290b982"  # Replace with your actual key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Load and process PDF only once
@st.cache_resource
def load_vector_store(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([pdf_text])
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

st.title("PDF Chatbot UI")

pdf_path = st.text_input("Enter PDF path", "C:/Users/garim/Downloads/The+48+Laws+Of+Power.pdf")

if pdf_path:
    try:
        vector_store = load_vector_store(pdf_path)
        user_question = st.text_input("Ask a question about the book:")
        if user_question:
            docs = vector_store.similarity_search(user_question, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Answer the question using only the context below."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_question}"}
                ]
            )
            st.markdown("**Answer:**")
            st.write(response.choices[0].message["content"])
    except Exception as e:
        st.error(f"Error: {e}")
