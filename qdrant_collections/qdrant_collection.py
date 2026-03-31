import os
import traceback
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


load_dotenv()
                                    
print(f" port details : {os.getenv("QDRANT_HOST_PORT")}")
print(f" host details : {os.getenv("QDRANT_HOST")}")

q_client =QdrantClient(host=os.getenv("QDRANT_HOST"),
                       port=os.getenv("QDRANT_HOST_PORT"))

model = SentenceTransformer("all-MiniLM-L6-V2")

DIM = model.get_sentence_embedding_dimension()
print(f"LLM Dimension : {DIM}")

def create_collection(name: str, distance: models.Distance):
    try:
        if not q_client.collection_exists(collection_name=name):
            q_client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    distance=distance,
                    size=DIM
                    )
            )
            print(f"collection '{name}' created successfully")
        else:
            print(f"Collection '{name}' is not created. since it is already existed.")
    except Exception:
        print(f"Exception raise : create method level.")

try:              
    create_collection("sample_collection",models.Distance.COSINE)
    
    collections_list = q_client.get_collections()

    for collection in collections_list.collections:
        colInfo = q_client.get_collection(collection.name)
        print(f"collection INFO : {colInfo}")

    new_text = "Elephants are the largest land animal on Earth."
    new_vector = model.encode(new_text).tolist()

    print(f"about to perform upsert operation with client.")
    #create embeddings.
    q_client.upsert(
        collection_name="sample_collection",
        points = [
            models.PointStruct(
                id = 0,
                vector = new_vector,
                payload = {"text": new_text,"categroy":"animal","role":"public"}
            )
            
        ]
    ) 
    print(f"new_vector : {new_vector}.")
    print(f"New_vector is added to collection.")


    #READ Operation.
    vector_records_list = q_client.retrieve(collection_name="sample_collection",
                      ids=[0],
                      with_payload = True,
                      with_vectors = True
                      )
    for record in vector_records_list:
        print(f"======================================")
        print(f"record payload : {record.payload}")
        print(f"record vector : {record.vector}")
        print(f"record vector : {record.id}")
        print(f"======================================")
        

except Exception:
    traceback.print_exc()
    print(f"Exception raised : call level...!")

