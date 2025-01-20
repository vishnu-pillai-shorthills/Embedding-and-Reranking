# from langchain_groq import ChatGroq
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from pymongo import MongoClient
import torch
import pymongo
import pdfplumber
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from torch.nn.functional import softmax

load_dotenv()

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def split_into_chunks(text, max_tokens=512):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i + max_tokens])

connection_string = os.getenv("MONGO_URL")
client = MongoClient(connection_string)
# print("client : ", client)

# Access the database and collection
db = client["embedding_database"]
collection = db["embeddings"]

# def generate_embeddings(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
# # def generate_embeddings(text, model_name="sentence-transformers/thenlper/gte-base"):
#     """Generate embeddings for a given text using a Hugging Face model."""
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
    
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         # Use the CLS token embeddings for simplicity
#         embeddings = outputs.last_hidden_state[:, 0, :].squeeze(0)
#     return embeddings.numpy()

# Function to generate embeddings using a transformer model
# def generate_embeddings(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
    
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         # Use the CLS token embeddings for simplicity
#         embeddings = outputs.last_hidden_state[:, 0, :].squeeze(0)
#     return embeddings.numpy()

def generate_embeddings(text, model_name="thenlper/gte-base"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Generate embeddings with the model
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the mean pooling strategy for sentence embeddings
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    
    return embeddings.squeeze(0).numpy()

def recursive_split_with_overlap(text, chunk_size=512, overlap_size=50):
    # Initialize the text splitter with overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap_size, length_function=len
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    return chunks

# def store_embeddings_in_mongo(pdf_path, collection):
#     """Extract text, generate embeddings, and store in MongoDB."""
#     # Extract text from PDF
#     text = extract_text_from_pdf(pdf_path)
#     print(f"Extracted text from {pdf_path}.")
    
#     # Generate embeddings
#     embeddings = generate_embeddings(text)
#     print(f"Generated embeddings of shape: {embeddings.shape}")
    
#     # Insert into MongoDB
#     document = {
#         "file_name": os.path.basename(pdf_path),
#         "embedding": embeddings.tolist(),  # Convert to list for JSON serialization
#         "text": text,
#     }
#     result = collection.insert_one(document)
#     print(f"Inserted document with ID: {result.inserted_id}")

# Function to store embeddings in MongoDB in chunks
# def store_embeddings_in_mongo(pdf_path, collection, chunk_size=512):
#     # Extract text from PDF
#     text = extract_text_from_pdf(pdf_path)
#     print(f"Extracted text from {pdf_path}.")
    
#     # Split text into chunks
#     chunks = list(split_into_chunks(text, chunk_size))
#     print(f"Split text into {len(chunks)} chunks.")
    
#     # Loop through the chunks, generate embeddings, and insert into MongoDB
#     for idx, chunk in enumerate(chunks):
#         # Generate embeddings for the chunk
#         embeddings = generate_embeddings(chunk)
#         print(f"Generated embeddings for chunk {idx} of size {embeddings.shape}.")
        
#         # Create the document to insert
#         document = {
#             "file_name": os.path.basename(pdf_path),
#             "chunk_index": idx,
#             "text": chunk,
#             "embedding": embeddings.tolist(),  # Convert embeddings to a list for MongoDB
#         }
        
#         # Insert the chunk into MongoDB
#         result = collection.insert_one(document)
#         print(f"Inserted chunk {idx} into MongoDB with ID: {result.inserted_id}")

# Function to recursively split text with overlap
def recursive_split_with_overlap(text, chunk_size=512, overlap_size=50):
    # Initialize the text splitter with overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap_size, length_function=len
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    return chunks

# Function to store embeddings in MongoDB in chunks
def store_embeddings_in_mongo(pdf_path, collection, chunk_size=512, overlap_size=50):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted text from {pdf_path}.")
    
    # Recursively split text into chunks with overlap
    chunks = recursive_split_with_overlap(text, chunk_size, overlap_size)
    print(f"Split text into {len(chunks)} chunks.")
    
    # Loop through the chunks, generate embeddings, and insert into MongoDB
    for idx, chunk in enumerate(chunks):
        # Generate embeddings for the chunk
        embeddings = generate_embeddings(chunk)
        print(f"Generated embeddings for chunk {idx} of size {embeddings.shape}.")
        
        # Create the document to insert
        document = {
            "file_name": os.path.basename(pdf_path),
            "chunk_index": idx,
            "text": chunk,
            "embedding": embeddings.tolist(),  # Convert embeddings to a list for MongoDB
        }
        
        # Insert the chunk into MongoDB
        result = collection.insert_one(document)
        print(f"Inserted chunk {idx} into MongoDB with ID: {result.inserted_id}")

def fetch_candidates_from_mongo(collection):
    """Fetch all documents and their embeddings from MongoDB."""
    documents = collection.find({}, {"embedding": 1, "text": 1, "_id": 0})
    candidates = []
    for doc in documents:
        candidates.append({
            "embedding": np.array(doc["embedding"]),  # Convert list back to NumPy array
            "text": doc["text"],
        })
    return candidates

def initial_ranking(query, collection, top_k=5):
    """Perform initial ranking using cosine similarity."""
    # Generate query embedding
    query_embedding = generate_embeddings(query)

    # Fetch all candidates
    candidates = fetch_candidates_from_mongo(collection)

    # Compute cosine similarity
    similarities = [
        (candidate["text"], cosine_similarity(query_embedding.reshape(1, -1), candidate["embedding"].reshape(1, -1))[0, 0])
        for candidate in candidates
    ]

    # Sort by similarity score in descending order
    ranked_candidates = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Return top-k results
    return ranked_candidates[:top_k]


def save_to_file(data, file_path):
    """Save data to a text file."""
    with open(file_path, "w") as file:
        for item in data:
            if isinstance(item, tuple):
                file.write(f"{item[0]} - Score: {item[1]}\n")
            else:
                file.write(f"{item}\n")
    print(f"Saved results to {file_path}")

def rerank_candidates(query, candidates, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Rerank candidates using a cross-encoder."""
    # Save initial results to a file
    save_to_file(candidates, "before_reranking.txt")

    # Load reranking model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    reranked_results = []
    for candidate_text, _ in candidates:  # Ignore the initial similarity score
        # Tokenize query and candidate text
        inputs = tokenizer(query, candidate_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits  # Access logits for classification tasks
        score = logits.squeeze().item()  # Convert logits to a scalar score
        reranked_results.append((candidate_text, score))

    # Sort by score in descending order
    reranked_results.sort(key=lambda x: x[1], reverse=True)

    # Save reranked results to a file
    save_to_file(reranked_results, "after_reranking.txt")

    return reranked_results


def compare_results(initial_results, reranked_results):
    """Compare initial ranking and reranking results."""
    print("\nInitial Results:")
    for rank, (text, score) in enumerate(initial_results, start=1):
        print(f"Rank {rank}: Score = {score:.4f}, Text = {text}")

    print("\nReranked Results:")
    for rank, (text, score) in enumerate(reranked_results, start=1):
        print(f"Rank {rank}: Score = {score:.4f}, Text = {text}")




if __name__ == "__main__":

    #  PDF path
    # pdf_path = "/home/shtlp_0068/Documents/reranking/assets/cricket_tutorial.pdf"  # Path to your PDF file

    # store_embeddings_in_mongo(pdf_path, collection)
    # Store embeddings for the PDF
    # pdf_path = "/home/shtlp_0068/Documents/reranking/assets/Gen AI.pdf"
    # store_embeddings_in_mongo(pdf_path, collection)

    # Query for initial ranking
    query_text = "Which country is the current World Champion in cricket (as of the provided text)?"
    top_k = 5

    # Perform initial ranking
    initial_results = initial_ranking(query_text, collection, top_k)
    # print("\nTop-K Initial Ranking Results:")
    # for rank, (text, score) in enumerate(initial_results, start=1):
    #     print(f"Rank {rank}: Score = {score:.4f}, Text = {text}")

    # # Perform reranking
    reranked_results = rerank_candidates(query_text, initial_results)
    # print("\nTop-K Reranked Results:")
    # for rank, (text, score) in enumerate(reranked_results, start=1):
    #     print(f"Rank {rank}: Score = {score:.4f}, Text = {text}")

    # Compare initial and reranked results
    # compare_results(initial_results, reranked_results)


