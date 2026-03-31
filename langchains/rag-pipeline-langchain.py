import os
import traceback
from pathlib import Path
from dotenv import load_dotenv
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling_core.transforms.chunker import HierarchicalChunker
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
#from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL")    
print(f"==========> GROQ_MODEL : {GROQ_MODEL}")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"==========> GROQ_API_KEY  : {GROQ_API_KEY}")
QDRANT_PATH = os.getenv("QDRANT_PATH")
print(f"==========> QDRANT_PATH  : {QDRANT_PATH}")


# llm_model = ChatGroq(
#     model=GROQ_MODEL,
#     temperature=0,
#     max_tokens=None,
#     reasoning_format="parsed",
#     timeout=None,
#     max_retries=2,
# )


SOURCE = "https://raw.githubusercontent.com/tnahddisttud/sample-doc/refs/heads/main/AtliqAI_HR_Policies.pdf"



documents = DoclingLoader(
    file_path=SOURCE,
    export_type=ExportType.DOC_CHUNKS,
    chunker=HierarchicalChunker(),
).load()

print(f"size of chunks : {len(documents)}")

if len(documents) > 0:
    print(documents[0])

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

try:
    vectorstore = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="delete_collection",
    )
except Exception:
    print(traceback.format_exc())

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the context below. Cite section names. Say 'I don't know' if unsure."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        dl_meta  = doc.metadata.get("dl_meta", {})
        headings = dl_meta.get("headings", [])
        source   = " > ".join(headings) if headings else "Unknown"
        parts.append(f"[{i}] {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def rag(query: str) -> str:
    docs         = retriever.invoke(query)
    context      = format_docs(docs)
    prompt_value = RAG_PROMPT.invoke({"context": context, "question": query})
    # response     = llm_model.invoke(prompt_value)
    # return response.content



# llm_model = ChatGroq(
#     model="openai/gpt-oss-20b",
#     temperature=0,
#     max_tokens=None,
#     reasoning_format="parsed",
#     timeout=None,
#     max_retries=2,
# )
for q in [
        "How many casual leaves am I entitled to?",
        "What is the notice period for Band 4 employees?",
        "How long is the probation period?",
    ]:
        print(f"\nQ: {q}\nA: {rag(q)}\n")
