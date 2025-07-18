import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load local embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load local LLM (example: vicuna-like or llama2)
tokenizer = AutoTokenizer.from_pretrained('your-local-llm-model')
model = AutoModelForCausalLM.from_pretrained('your-local-llm-model', device_map='auto', torch_dtype=torch.float16)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, chunk_size=500):
    # Simple whitespace chunking, or use tokenizer for better chunking
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# Build embeddings and FAISS index
texts = chunk_text(extracted_text)
embeddings = embed_model.encode(texts, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Query phase
def query_llm(question, index, texts, embed_model, tokenizer, model):
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, k=3)
    context = "\n\n".join([texts[i] for i in I[0]])

    prompt = f"Context:\n{context}\n\nRewrite this question to be clearer and more challenging:\n{question}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

