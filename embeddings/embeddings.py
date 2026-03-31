from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Example Documents
docs = [
    "Dogs are loyal and friendly domestic animals.",                  # text_1
    "Cats are independent and curious creatures.",                    # text_2
    "The Milky Way galaxy contains over 200 billion stars."           # text_3
    ]

# Creating Embedding
embedding = embedder.encode(docs)
embedding.shape

# Creating Embedding of the "Query" text.
query = "What animals make good pets?"
query_embedding = embedder.encode(query)

# Calculate "Similarity" between text_1 & query
score_1 = embedder.similarity(query_embedding, embedding[0])
print(f"score_1 : {score_1}")

# Calculate "Similarity" between text_2 & query
score_2 = embedder.similarity(query_embedding, embedding[1])
print(f"score_2 : {score_2}")

# Calculate "Similarity" between text_3 & query
score_3 = embedder.similarity(query_embedding, embedding[2])
print(f"score_3 : {score_3}")

