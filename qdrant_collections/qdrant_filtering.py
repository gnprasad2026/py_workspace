import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter,FieldCondition, MatchValue

load_dotenv()

COLLECTION_NAME = "sample_collection"
                                    
print(f" port details : {os.getenv("QDRANT_HOST_PORT")}")
print(f" host details : {os.getenv("QDRANT_HOST")}")

q_client =QdrantClient(host=os.getenv("QDRANT_HOST"),
                       port=os.getenv("QDRANT_HOST_PORT"))

model = SentenceTransformer("all-MiniLM-L6-V2")

# Performing semantic search.
#query = "what animals make good pets ?"
#query ="suggest about good life style"
#query = "which animal is reffered to speed"
query = "technology"
#query = "which animal communicate using sound ?"
#query = "How to protective sensitive data"
#query = "what is the big animal on the earth"
#query = "which animals are used to travel from one place to another place"

query_vector = model.encode(query).tolist()

query_info = q_client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter= Filter(
        must = [
            FieldCondition(key="category", match=MatchValue(value="business")),
            FieldCondition(key="role", match=MatchValue(value="private")),
        ]
    ),
    #with_payload=False,,
    #with_vectors=False,
    #limit=3,
    #score_threshold=0.35
)
print(f"query info : {query_info}")

for point in query_info.points:
    print(f" point score  : { point.score}")
    print(f" point payload.text : {point.payload["text"]}")

