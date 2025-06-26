from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import fitz  # PyMuPDF
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

# === CONFIG ===
PDF_PATH = "C:/Users/garim/Downloads/The+48+Laws+Of+Power.pdf"

# === INIT ===
print("🔧 Initializing FastAPI app...")
app = FastAPI()
templates = Jinja2Templates(directory="templates")

print("🔐 Setting up OpenAI client...")
openai.api_key = os.getenv("OPENAI_API_KEY")  # ✅ Uses environment variable

# === STEP 1: Load and process PDF ===
def extract_chunks_from_pdf(file_path, chunk_size=500):
    print(f"📄 Opening PDF from: {file_path}")
    try:
        with fitz.open(file_path) as doc:
            full_text = "\n".join([page.get_text() for page in doc])
        print("✅ PDF read successfully.")
    except Exception as e:
        print(f"❌ Failed to read PDF: {e}")
        return []

    print("✂️ Splitting text into chunks...")
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks

print("🔍 Extracting and preparing data from PDF...")
chunks = extract_chunks_from_pdf(PDF_PATH)

if chunks:
    print("🔡 Creating TF-IDF vectors for chunks...")
    vectorizer = TfidfVectorizer().fit(chunks)
    chunk_vectors = vectorizer.transform(chunks)
    print("✅ Vectorization complete.")
else:
    print("❌ No chunks to vectorize. Exiting.")
    exit()

# === STEP 2: Web Interface ===
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    print("📥 GET / - Rendering form.")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    print(f"🧠 Received question: {question}")

    print("🔍 Searching for most relevant chunks...")
    question_vec = vectorizer.transform([question])
    scores = cosine_similarity(question_vec, chunk_vectors).flatten()
    top_indices = scores.argsort()[-4:][::-1]
    top_context = "\n\n".join([chunks[i] for i in top_indices])
    print(f"📚 Context selected from top {len(top_indices)} chunks.")

    print("🧾 Sending prompt to OpenAI...")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Answer using the provided context only."},
                {"role": "user", "content": f"Context:\n{top_context}\n\nQuestion:\n{question}"}
            ]
        )
        answer = response.choices[0].message.content
        print("✅ Response received from OpenAI.")
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        answer = "Error fetching answer from OpenAI."

    return templates.TemplateResponse("index.html", {
        "request": request,
        "question": question,
        "answer": answer
    })
