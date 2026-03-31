import os
import traceback
from pathlib import Path
import itertools
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

import os

def check_path_type_os(path_location): 
    if os.path.isfile(path_location):
        return "FILE"
    elif os.path.isdir(path_location):
        return "DIR"
    else:
        print(f"'{path_location}' does not exist or is another type of file system object.")

GROQ_MODEL = os.getenv("GROQ_MODEL")    
print(f"==========> GROQ_MODEL : {GROQ_MODEL}")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"==========> GROQ_API_KEY  : {GROQ_API_KEY}")


# llm_model = ChatGroq(
#     model=GROQ_MODEL,
#     temperature=0,
#     max_tokens=None,
#     reasoning_format="parsed",
#     timeout=None,
#     max_retries=2,
# )


#SOURCE = "https://raw.githubusercontent.com/tnahddisttud/sample-doc/refs/heads/main/AtliqAI_HR_Policies.pdf"
SOURCE =Path("/home/prasad/ai_bootcamp/session_6/assignment-1/data/marketing/")

if check_path_type_os(SOURCE) == 'DIR':
    files_list = [os.path.join(SOURCE, filename) for filename in os.listdir(SOURCE)]
else:
    files_list = SOURCE

for filename in files_list:
    documents = DoclingLoader(
        file_path=filename,
        export_type=ExportType.DOC_CHUNKS,
        chunker=HierarchicalChunker(),
    ).load()

documents = [DoclingLoader( file_path=filename, export_type=ExportType.DOC_CHUNKS, chunker=HierarchicalChunker(), ).load() for filename in files_list]

final_docs = list(itertools.chain.from_iterable(documents))

print(f"size of chunks : {len(final_docs)}")

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)



try:
    vectorstore = QdrantVectorStore.from_documents(
        documents=final_docs,
        embedding=embeddings,
        collection_name="delete_collection",
    )
except Exception:
    print(traceback.format_exc())
