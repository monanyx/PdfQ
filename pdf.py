import streamlit as st
import fitz  # PyMuPDF

import openai
import faiss
import numpy as np
import tiktoken
import os

# Securely load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["openai_api_key"]
    except Exception:
        api_key = ""
openai.api_key = ""  # <-- put your real OpenAI API key here

# Constants
EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 500  # tokens approx

# Tokenizer for splitting text
tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, max_tokens=CHUNK_SIZE):
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    for word in words:
        tokens = len(tokenizer.encode(word))
        if current_tokens + tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = tokens
        else:
            current_chunk.append(word)
            current_tokens += tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# Batch embedding for efficiency
def get_embeddings(texts):
    try:
        response = openai.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        return np.array([r.embedding for r in response.data]).astype("float32")
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def get_embedding(text):
    embs = get_embeddings([text])
    if embs is not None:
        return embs[0]
    else:
        return np.zeros((1536,), dtype="float32")  # fallback, adjust dim as needed

def build_faiss_index(embeddings):
    try:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index
    except Exception as e:
        st.error(f"FAISS index error: {e}")
        return None

def search_index(index, query_embedding, k=5):
    try:
        D, I = index.search(query_embedding, k)
        return I[0]
    except Exception as e:
        st.error(f"FAISS search error: {e}")
        return []

def gpt_rewrite_question(context_chunks, user_question):
    prompt = (
        "Based on the following context excerpts:\n\n"
        + "\n\n---\n\n".join(context_chunks)
        + f"\n\nPlease rewrite this question to be clearer and more challenging:\n{user_question}\n"
    )
    try:
        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI completion error: {e}")
        return ""

# Streamlit UI
st.title("PDF Q&A & Question Rewriter")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.text_chunks = []

if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        try:
            with st.spinner(f"Reading {file.name}..."):
                pdf = fitz.open(stream=file.read(), filetype="pdf")
                text = ""
                for page in pdf:
                    text += page.get_text()
                all_text += text + "\n\n"
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")

    st.write("Extracted text from PDFs")

    with st.spinner("Chunking and embedding text..."):
        chunks = chunk_text(all_text)
        # Batch embeddings for all chunks
        embeddings = get_embeddings(chunks)
        if embeddings is None:
            st.stop()

    # Build FAISS index and store chunks
    st.session_state.faiss_index = build_faiss_index(embeddings)
    st.session_state.text_chunks = chunks
    st.success(f"Indexed {len(chunks)} text chunks!")

if st.session_state.faiss_index:
    user_q = st.text_input("Enter question to rewrite")

    if user_q:
        with st.spinner("Embedding and searching context..."):
            q_emb = get_embedding(user_q).reshape(1, -1)
            idxs = search_index(st.session_state.faiss_index, q_emb, k=3)
            context = [st.session_state.text_chunks[i] for i in idxs if i < len(st.session_state.text_chunks)]
        rewritten_q = gpt_rewrite_question(context, user_q)
        st.markdown("### Rewritten Question:")
        st.write(rewritten_q)
