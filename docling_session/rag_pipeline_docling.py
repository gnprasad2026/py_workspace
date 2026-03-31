import os
from dotenv import load_dotenv

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker
from hierarchical.postprocessor import ResultPostprocessor

from fastapi import HTTPException
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from groq import Groq

SOURCE = "https://raw.githubusercontent.com/tnahddisttud/sample-doc/refs/heads/main/AtliqAI_HR_Policies.pdf"

load_dotenv()

def load_document(source: str):
    """
    Parse a PDF using Docling.
    Returns a DoclingDocument object — not a plain string.
    """
    converter = DocumentConverter()
    result = converter.convert(source)
    ResultPostprocessor(result).process()
    return result.document

doc = load_document(SOURCE)
print(f"Document loaded: {doc.name}")

markdown_doc = doc.export_to_markdown()
print(markdown_doc[:100])

# CHUNKING DATA.

chunker   = HierarchicalChunker()
doc_chunks = list(chunker.chunk(doc))

print(f"Total chunks: {len(doc_chunks)}")

# Inspect a raw DocChunk
# sample = doc_chunks[10]
# print(f" sample chunk : {sample} ")
# #print(f"headings : {sample.meta.headings}")
# #print(f"text     : {sample.text[:200]}…")

def convert_chunk(doc_chunk) -> dict:
    """
    Convert a Docling DocChunk into a plain dict.

    headings   → list preserved as-is
    content    → paragraph text
    chunk_text → breadcrumb + content  (what gets embedded)
    """
    headings   = doc_chunk.meta.headings or []
    content    = doc_chunk.text.strip()
    breadcrumb = " > ".join(headings)
    chunk_text = f"{breadcrumb}\n\n{content}" if breadcrumb else content

    return {
        "headings":   headings,
        "content":    content,
        "chunk_text": chunk_text,
    }

chunks = [convert_chunk(c) for c in doc_chunks]

# for chunk in chunks[:3]:
#     print("─" * 100)
#     print(f"headings        : {chunk['headings']}")
#     print(f"content         : {chunk['content'][:200]}....")
#     print(f"content-text    : {chunk['chunk_text'][:200]}....")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

chunk_texts = [c["chunk_text"] for c in chunks]

print(f"Embedding {len(chunk_texts)} chunks …")
embeddings = embedder.encode(chunk_texts, show_progress_bar=True)

print(f"Shape: {embeddings.shape}")   # → (N, 384)


from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

# "path" = no server needed for demos
q_client = QdrantClient(url="http://localhost:6333")
#client = QdrantClient(path="/tmp/my_qdrant")

COLLECTION_NAME = "docs"
DIM = embedder.get_sentence_embedding_dimension()

def create_collection(name: str, distance: models.Distance):
    try:
        if not q_client.collection_exists(collection_name=name):
            q_client.create_collection(
                collection_name=name,
                vectors_config = models.VectorParams(
                    distance=distance,
                    size=DIM
                    )
            )
            print(f"collection '{name}' created successfully")
        else:
            print(f"Collection '{name}' is not created. since it is already existed.")
    except Exception:
        print(f"Exception raise : create method level.")

create_collection(COLLECTION_NAME, models.Distance.COSINE)
print("Collection created.")


# Creating Points
points = [
    PointStruct(
        id=idx,
        vector=embedding.tolist(),
        payload={
            "headings":   chunk["headings"],   # stored as a JSON array
            "content":    chunk["content"],
            "chunk_text": chunk["chunk_text"],
        },
    )
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
]

#result = q_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True,)
#print(f"Indexed {len(points)} points — status: {result.status}")


def retrieve(query: str, top_k: int = 5 ) -> list[dict]:
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

results = retrieve("What is the leave policy?", top_k=3)
for r in results:
    print(f"[{r['score']}]  {r['headings']}")
    print(f"  {r['content']}…\n")

SYSTEM_PROMPT = """You are a helpful HR assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so — do not make things up.
Always cite the section name when referencing specific information."""

def build_context(retrieved_chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(f"[Source {i}]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)

GROQ_MODEL = os.getenv("GROQ_MODEL")    
print(f"==========> GROQ_MODEL : {GROQ_MODEL}")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"==========> GROQ_API_KEY  : {GROQ_API_KEY}")

llm_model = Groq()

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

    response = llm_model.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1, #Low = factual, #High = creative
    )
    return response.choices[0].message.content, context

#answer, context = rag("How many casual leaves am I entitled to?")
#answer, context = rag("what are the employee health benifits ?")
#answer, context = rag("Is employees can work from home when ever he wants ?")
#answer, context = rag("Becuase of project tight deadlines, any employee worked more time than expected. did he get any advantages ?")

answer, context = rag("you are employee of AtliqAI employee with 34L/year package. and 1.2LK/month is the base salary as part of the componsation. And you have worked 5 years. Please tell me how much gratuity you will get after 65months ?")

print(f"\n\nSOURCES:\n {context}")
print(f"{100*'='}")
print(f" Final response from model : {answer}")

