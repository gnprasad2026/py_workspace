import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient,models

load_dotenv()

COLLECTION_NAME = "sample_collection"
                                    
print(f" port details : {os.getenv("QDRANT_HOST_PORT")}")
print(f" host details : {os.getenv("QDRANT_HOST")}")

q_client =QdrantClient(host=os.getenv("QDRANT_HOST"),
                       port=os.getenv("QDRANT_HOST_PORT"))

model = SentenceTransformer("all-MiniLM-L6-V2")

documents =[
    {"id":2, "text":"Cats are independent pets that enjoy quiet environments.", "category":"animal", "role":"public"},
    {"id":3, "text":"Elephants are the largest land animals and have excellent memory.", "category":"animal", "role":"public"},
    {"id":4, "text":"Tigers are powerful predators known for their striped fur.", "category":"animal", "role":"public"},
    {"id":5, "text":"Birds can fly using their wings and have lightweight bones.", "category":"animal", "role":"public"},
    {"id":6, "text":"Fish live in water and breathe through gills.", "category":"animal", "role":"public"},
    {"id":7, "text":"Dolphins are intelligent marine mammals that communicate using sounds.", "category":"animal", "role":"public"},
    {"id":8, "text":"Lions are known as the king of the jungle and live in groups called prides.", "category":"animal", "role":"public"},
    {"id":9, "text":"Rabbits are small mammals that are known for their long ears.", "category":"animal", "role":"public"},
    {"id":10, "text":"Horses are strong animals used for riding and transportation.", "category":"animal", "role":"public"},
    {"id":11, "text":"Pandas primarily eat bamboo and are native to China.", "category":"animal", "role":"public"},
    {"id":12, "text":"Artificial Intelligence is transforming industries by automating complex tasks.", "category":"technology", "role":"public"},
    {"id":13, "text":"Regular exercise helps maintain physical and mental health.", "category":"health", "role":"public"},
    {"id":14, "text":"Stock markets fluctuate based on economic conditions and investor sentiment.", "category":"finance", "role":"public"},
    {"id":15, "text":"The company’s quarterly revenue exceeded internal expectations.", "category":"business", "role":"private"},
    {"id":16, "text":"Cloud computing enables scalable and on-demand access to computing resources.", "category":"technology", "role":"public"},
    {"id":17, "text":"A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals.", "category":"health", "role":"public"},
    {"id":18, "text":"The new product launch strategy is confidential until next quarter.", "category":"business", "role":"private"},
    {"id":19, "text":"Cryptocurrencies operate on decentralized blockchain technology.", "category":"finance", "role":"public"},
    {"id":20, "text":"Employee performance reviews are restricted to management access only.", "category":"hr", "role":"private"},
    {"id":21, "text":"Cybersecurity measures are essential to protect sensitive data from breaches.", "category":"technology", "role":"public"}, 
]

docs_text_list =  [doc["text"] for doc in documents]
docs_text_list_vector = model.encode(docs_text_list)
DIM = model.get_sentence_embedding_dimension()

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

def add_multi_lines_to_vectordb(colectin_name:str, documents:list[dict]):

    #creating collection if not exists:
    create_collection(COLLECTION_NAME,models.Distance.COSINE)
    #upsert multiple statements - Operation.
    vector_records_list = q_client.upsert(
        collection_name=COLLECTION_NAME,
        points = [
            models.PointStruct(
                id = doc["id"],
                vector = docs_text_list_vector[i].tolist(),
                payload = {
                    "text": doc["text"],
                    "category" : doc["category"],
                    "role" : doc["role"],
                }
            )
            for i, doc in enumerate(documents)
        ],
        wait=True,
    )
    



add_multi_lines_to_vectordb(COLLECTION_NAME,documents)

collections_list = q_client.get_collections()

for collection in collections_list.collections:
    colInfo = q_client.get_collection(collection.name)
    print(f"collection INFO : {colInfo}")




      
