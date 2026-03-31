# Loading the Document
import os
from dotenv import load_dotenv
import requests
from fastapi import HTTPException
from groq import Groq
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

GITHUB_RAW_URL = "https://raw.githubusercontent.com/tnahddisttud/sample-doc/refs/heads/main/atliqai_hr_policies.txt"

def load_document(url: str) -> str:
    """Fetch a plain-text file from a raw GitHub URL."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text

raw_text = load_document(GITHUB_RAW_URL)
print(f"Loaded {len(raw_text):,} characters")
print(raw_text[:400])  # Sanity check

# PRE PROCESSING DATA  - CHUNCKING.
CHUNK_SIZE = 50

def parse_word_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[dict]:
    # Strip markdown heading symbols and blank lines
    clean_lines = []
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            clean_lines.append(line)

    # Join everything into one word list and slice
    words = " ".join(clean_lines).split()

    chunks = []
    for i in range(0, len(words), chunk_size):
        content = " ".join(words[i : i + chunk_size])
        chunks.append({
            "chunk_index": len(chunks),
            "content": content,
        })

    return chunks

chunks = parse_word_chunks(raw_text)
print(f"Total chunks: {len(chunks)}")

# Inspect a chunk
for chunk in chunks[:3]:
    print("─" * 55)
    print(f"Content : {chunk['content'][:200]}…")


def build_chunk_text(chunk: dict) -> str:
    return chunk["content"]

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Extract Chunk Texts
chunk_texts = [build_chunk_text(c) for c in chunks]

print(f"Embedding {len(chunk_texts)} chunks …")
embeddings = embedder.encode(chunk_texts, show_progress_bar=True)

print(f"Shape: {embeddings.shape}")

# "path" = no server needed for demos
# Production use: 
q_client = QdrantClient(url="http://localhost:6333")
#client = QdrantClient(path="/tmp/langchain_qdrant")

COLLECTION_NAME = "docs"
DIM = embedder.get_sentence_embedding_dimension()

def create_collection(name: str, distance: models.Distance):
    try:
        if not q_client.collection_exists(collection_name= COLLECTION_NAME):
            q_client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    distance=distance,
                    size=DIM
                )
            )
        else:
            print(f"collection '{name}' created successfully")
    except Exception:
        print(f"Exception raised during collection creation.")

create_collection( COLLECTION_NAME, models.Distance.COSINE)

# Creating Points
points = [
    PointStruct(
        id=idx,
        vector=embedding.tolist(),
        payload={
            "content": chunk["content"],
        },
    )
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
]

result = q_client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True,   # Block until indexing completes before returning
)
print(f"Indexed {len(points)} points — status: {result.status}")

info = q_client.get_collection(COLLECTION_NAME)
print(f"Points     : {info.points_count}")
print(f"Dimensions : {info.config.params.vectors.size}")

def retrieve(
    query: str,
    top_k: int = 5
) -> list[dict]:
    """
    Embed the query and return the top-k most similar chunks.

    Args:
        query          : User's question.
        top_k          : Number of chunks to return.
        section_filter : Optional H2 heading to restrict the search scope.
    """
    query_vector = embedder.encode(query).tolist()

    hits = q_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [{**hit.payload, "score": round(hit.score, 4)} for hit in hits.points]

results = retrieve("What is the leave policy", top_k=3)
for r in results:
    print(f"[score={r['score']}]")
    print(f"  {r['content'][:200]}…\n")

# Retrieval process.
SYSTEM_PROMPT = """You are a helpful HR assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so — do not make things up.
Always cite the section name when referencing specific information."""

def build_context(retrieved_chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(f"[Source {i}]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"==========>GROQ_API_KEY  : {GROQ_API_KEY}")

if not GROQ_API_KEY:
    raise HTTPException(status_code=500, detail="Missing api_key in env file")
else:
    print(f"GROQ_API_KEY : {GROQ_API_KEY}")

GROQ_MODEL = os.getenv("GROQ_MODEL")    
print(f"GROQ_MODEL : {GROQ_MODEL}")

client = Groq()
#GROQ_MODEL  = "openai/gpt-oss-safeguard-20b"



def rag(query: str, top_k: int = 5):
    """
    End-to-end RAG pipeline:
      1. Retrieve relevant chunks from Qdrant
      2. Format them as a context block
      3. Send context + query to Groq and return the answer
    """
    # Step 1 — Retrieve
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return "No relevant content found in the document."

    # Step 2 — Build context
    context = build_context(chunks)

    # Step 3 — Generate
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,   # Low = factual;  High = creative
    )
    return response.choices[0].message.content, context

llm_response = rag(SYSTEM_PROMPT)
print(f"llm response : {llm_response}")
