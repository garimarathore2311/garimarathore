from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
import requests
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# === INIT ===
print("🔧 Initializing FastAPI app...")
app = FastAPI()
templates = Jinja2Templates(directory="templates")

print("🔐 Setting up OpenAI client...")
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === CONFIG ===
PDF_URL = "https://drive.google.com/uc?export=download&id=1-bL9VRo-9FjxA1OZ5rGk7o_lhlW0lhlr"

# === STEP 1: Load and process PDF ===
def extract_chunks_from_drive(pdf_url, chunk_size=1500, overlap=200):
    try:
        print("📄 Downloading PDF from Google Drive...")
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf = fitz.open(stream=response.content, filetype="pdf")
        full_text = "\n".join([page.get_text() for page in pdf])
        print("✅ PDF downloaded and read successfully.")
    except Exception as e:
        print(f"❌ Failed to read PDF: {e}")
        return [], ""

    print(full_text[:2000])  # Print the first 2000 characters

    print("✂️ Splitting text into chunks...")
    chunks = []
    for i in range(0, len(full_text), chunk_size - overlap):
        chunks.append(full_text[i:i + chunk_size])
    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks, full_text

print("🔍 Extracting and preparing data from PDF...")
chunks, full_text = extract_chunks_from_drive(PDF_URL)
laws = re.split(r'LAW \d+:', full_text)

# === DEBUGGING: Inspecting Chunks ===
for i, chunk in enumerate(chunks):
    if "15" in chunk or "fifteen" in chunk.lower():
        print(f"Chunk {i}:\n{chunk[:500]}\n{'-'*40}")

if "15" in full_text or "fifteen" in full_text.lower():
    print("Found '15' or 'fifteen' in the extracted text!")
else:
    print("Could not find '15' or 'fifteen' in the extracted text.")

# Highlighting the specific section for LAW 15
matches = list(re.finditer(r"LAW\s*15", full_text, re.IGNORECASE))
for match in matches:
    start = max(match.start() - 200, 0)
    end = match.end() + 800
    print(full_text[start:end])
    print("-" * 40)

# === STEP 2: Generate Embeddings ===
def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

print("🧠 Generating embeddings for all chunks (semantic search)...")
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
if None in chunk_embeddings:
    print("❌ Embedding generation failed for one or more chunks.")
    exit()

# === STEP 3: Web Interface ===
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    print("📥 GET / - Rendering form.")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    print(f"🧠 Received question: {question}")

    try:
        print("📌 Embedding user question...")
        question_embedding = get_embedding(question)
        if question_embedding is None:
            raise ValueError("Failed to embed question.")

        print("🔍 Computing similarities...")
        similarities = cosine_similarity(
            [question_embedding],
            chunk_embeddings
        )[0]
        top_indices = similarities.argsort()[-12:][::-1]
        top_context = "\n\n".join([chunks[i] for i in top_indices])
        print(f"📚 Selected top {len(top_indices)} relevant chunks.")

        print("---- Top Chunks for Debugging ----")
        for i, idx in enumerate(top_indices):
            print(f"Chunk {i+1} (index {idx}):")
            print(chunks[idx][:500])  # print first 500 chars
            print("-------------------------------")

        print("💬 Sending prompt to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert on the book 'The 48 Laws of Power.' Use the provided context if possible, but always answer the user's question as helpfully and completely as possible, even if the context does not contain the answer."},
                {"role": "user", "content": f"Context:\n{top_context}\n\nQuestion:\n{question}"}
            ]
        )
        answer = response.choices[0].message.content.strip()
        print("✅ Response received.")
    except Exception as e:
        print(f"❌ Error during answer generation: {e}")
        answer = "Sorry, something went wrong while processing your question."

    return templates.TemplateResponse("index.html", {
        "request": request,
        "question": question,
        "answer": answer
    })
