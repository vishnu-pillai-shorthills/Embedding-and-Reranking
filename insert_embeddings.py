import json
import os
import numpy as np
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
import torch

# MongoDB connection
connection_string = os.getenv("MONGO_URL")
client = MongoClient(connection_string)
db = client["embedding_database3"]
collection = db["embeddings"]

def generate_embeddings(text, model_name="thenlper/gte-base"):
    """Generate embeddings for a given text using a specified model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Generate embeddings with the model
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)  # Mean pooling
    
    return embeddings.squeeze(0).numpy()

def store_text_and_embeddings_to_mongo(json_path, collection):
    """Process and store only text and embeddings from predicted_chunks.json into MongoDB."""
    with open(json_path, "r") as file:
        data = json.load(file)  # Load JSON file
    
    # Iterate through the JSON entries
    for item in data:
        for chunk_text in item.get("chunks", []):
            # Skip short chunks
            if len(chunk_text) < 100:
                continue
            
            # Generate embedding for the chunk
            embedding = generate_embeddings(chunk_text)
            
            # Prepare the document with only text and embedding
            document = {
                "text": chunk_text,
                "embedding": embedding.tolist(),
            }
            
            # Insert the document into MongoDB
            result = collection.insert_one(document)
            print(f"Inserted chunk into MongoDB with ID: {result.inserted_id}")

    print("All chunks have been processed and stored in MongoDB.")

# Example usage
if __name__ == "__main__":
    store_text_and_embeddings_to_mongo("data/chunks_who.json", collection)

