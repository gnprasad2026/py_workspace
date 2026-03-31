import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

load_dotenv()
                                    
print(f" port details : {os.getenv("QDRANT_HOST_PORT")}")
print(f" host details : {os.getenv("QDRANT_HOST")}")

q_client =QdrantClient(host=os.getenv("QDRANT_HOST"),
                       port=os.getenv("QDRANT_HOST_PORT"))

model = SentenceTransformer("all-MiniLM-L6-V2")

#READ Operation.
vector_records_list = q_client.retrieve(collection_name="sample_collection",
                    ids=[0],
                    with_payload = True,
                    with_vectors = True
                    )

print(f"vector count : {vector_records_list.count}")
      
for record in vector_records_list:
    print(f"======================================")
    print(f"record payload : {record.payload}")
    print(f"record vector : {record.vector}")
    print(f"record vector : {record.id}")
    print(f"======================================")