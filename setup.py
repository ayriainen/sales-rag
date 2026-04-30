"""
This is database setup that you only need to run once: "python setup.py". Run prep.py before this.
Converts the chunks from chunks.json into vector embeddings.
The sentence-transformer used is all-MiniLM-L6-v2.
Outputs chroma_db to be used by rag.py.
"""
import json
import chromadb
from chromadb.utils import embedding_functions

with open("chunks.json") as f:
    chunks = json.load(f)

client = chromadb.PersistentClient(path="./chroma_db")

emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(
    name="sales_data",
    embedding_function=emb_fn,
    metadata={"hnsw:space": "cosine"},
)

def ingest(doc_type, chunks, tags, batch_size=500):
    """Ingests/loads the chunks and prints status"""
    ids = [f"{doc_type}_{i}" for i in range(len(chunks))]
    for start in range(0, len(chunks), batch_size):
        collection.add(
            documents=chunks[start:start+batch_size],
            ids=ids[start:start+batch_size],
            metadatas=tags[start:start+batch_size],
        )
        print(f"Ingested {min(start+batch_size, len(chunks))}/{len(chunks)} {doc_type}")

for doc_type, payload in chunks.items():
    ingest(doc_type, payload["chunks"], payload["tags"])

print(f"Total docs: {collection.count()}")
