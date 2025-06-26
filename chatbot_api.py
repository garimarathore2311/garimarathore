from fastapi import FastAPI, Request
from pydantic import BaseModel
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import openai

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = ""  # Replace with your actual key
openai.api_key = os.environ["OPENAI_API_KEY"]

app = FastAPI()

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Load and process PDF only once (global cache)
pdf_path = "C:/Users/garim/Downloads/The+48+Laws+Of+Power.pdf"  # Change as needed
pdf_text = extract_text_from_pdf(pdf_path)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents([pdf_text])
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QuestionRequest):
    user_question = req.question
    docs = vector_store.similarity_search(user_question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Answer the question using only the context below."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_question}"}
        ]
    )
    return {"answer": response.choices[0].message["content"]}

# To run: uvicorn chatbot_api:app --reload
# Then POST to http://127.0.0.1:8000/ask with JSON: {"question": "Your question here"}
