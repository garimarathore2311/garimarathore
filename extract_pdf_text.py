import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

print(">>> Script started!")

# === STEP 1: Extract text from PDF ===
def extract_text_from_pdf(file_path):
    print("Opening PDF...")
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return None
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    print("Finished reading PDF.")
    return text

pdf_path = "C:/Users/garim/Downloads/The+48+Laws+Of+Power.pdf"
print("Extracting text from:", pdf_path)

pdf_text = extract_text_from_pdf(pdf_path)

# === STEP 2: Chunk the text ===
if pdf_text:
    print("Text length:", len(pdf_text))
    print("Preview:\n", pdf_text[:1000])

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([pdf_text])

    print("Total chunks created:", len(chunks))
    print("Sample chunk:\n", chunks[0].page_content)
else:
    print("❌ Failed to extract text from PDF.")
    exit()

# === STEP 3: Create embeddings and store in FAISS ===
os.environ["OPENAI_API_KEY"] = ""  # 🔐 Replace with your actual key
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

print("✅ Chunks embedded and stored in FAISS vector DB.")

# === STEP 4: User Q&A ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

user_question = input("\n❓ Ask a question about the book: ")

# Search relevant chunks
docs = vector_store.similarity_search(user_question, k=4)
context = "\n\n".join([doc.page_content for doc in docs])

# Ask GPT-4
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Answer the question using only the context below."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_question}"}
    ]
)

print("\n💬 Answer:\n", response.choices[0].message.content)
