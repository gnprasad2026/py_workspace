import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

print(f" port details : {os.getenv("QDRANT_URL")}")
print(f" host details : {os.getenv("QDRANT_API_KEY")}")

print(f" port details : {os.getenv("QDRANT_HOST")}")
print(f" host details : {os.getenv("QDRANT_HOST_PORT")}")


qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    )

print(qdrant_client.get_collections())

#sentence-transformers
