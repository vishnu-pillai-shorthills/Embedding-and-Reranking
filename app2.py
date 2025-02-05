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
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
import json

load_dotenv()

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text



def split(markdown_text):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)
    return md_header_splits

def export_to_markdown(pdf_path):
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()

# def split_into_chunks(text, max_tokens=512):
    # words = text.split()
    # for i in range(0, len(words), max_tokens):
    #     yield " ".join(words[i:i + max_tokens])
def split_chunks(path):   
    markdown_text = export_to_markdown(pdf_path=path)
    docs = split(markdown_text=markdown_text)
    # print(docs)
    chunks = []
    
    for i, doc in enumerate(docs):
            
        try:
        
            chunk = doc.page_content
            if len(chunk) < 100:
                continue
            chunks.append(chunk)
        except:
            pass
    # print(chunks)
    return chunks

connection_string = os.getenv("MONGO_URL")
client = MongoClient(connection_string)
# print("client : ", client)

# Access the database and collection
db = client["embedding_database4"]
collection = db["embeddings"]


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

# Function to store embeddings in MongoDB in chunks
def store_embeddings_in_mongo(pdf_path, collection, chunk_size=512, overlap_size=50):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted text from {pdf_path}.")
    
    # Recursively split text into chunks with overlap
    # chunks = recursive_split_with_overlap(text, chunk_size, overlap_size)
    chunks = split_chunks(pdf_path)
    print(f"Split text into {len(chunks)} chunks.")

    # print(chunks)
    
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
    return ranked_candidates[:20]
    # return ranked_candidates[:top_k]





def save_to_json(data, file_path):
    formatted_data = []

    # Iterate through each query's data and format it
    for query_data in data:
        query = query_data["query"]  # Assuming each entry in data has a "query" key
        chunks = [
            {"text": item["text"], "score": item["score"]} for item in query_data["chunks"]
        ]
        formatted_data.append({"query": query, "chunks": chunks})

    # Save to the JSON file
    with open(file_path, "w") as file:
        json.dump(formatted_data, file, indent=4)

    print(f"Saved results to {file_path}")


# def rerank_candidates(query, candidates, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
def rerank_candidates(query, candidates, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Rerank candidates using a cross-encoder."""
    # Save initial results to a file
    # save_to_file(candidates, "before_reranking.txt")

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
    # save_to_file(reranked_results, "after_reranking.txt")

    return reranked_results[:5]




def process_queries(queries, collection, top_k=5):
    before_reranking_results = []
    after_reranking_results = []
    
    for idx, query in enumerate(queries):
        print(f"\nProcessing Query {idx + 1}: {query}")
        
        # Perform initial ranking
        initial_results = initial_ranking(query, collection, top_k)
        before_reranking = initial_results[:top_k]
        
        # Store initial results (before reranking)
        before_reranking_results.append({
            "query": query,
            "chunks": [{"text": chunk, "score": score} for chunk, score in before_reranking]
            # "chunks": [{"text": chunk, "score": score} for chunk, score in initial_results]
        })
        
        # Perform reranking
        reranked_results = rerank_candidates(query, initial_results)
        
        # Store reranked results (after reranking)
        after_reranking_results.append({
            "query": query,
            "chunks": [{"text": chunk, "score": score} for chunk, score in reranked_results]
        })
        
        # Compare initial and reranked results
        # compare_results(initial_results, reranked_results)
    
    # Save both before and after reranking results to JSON files
    save_to_json(before_reranking_results, "before_reranking_diksha.json")
    save_to_json(after_reranking_results, "reranker1/after_reranking_diksha.json")

def load_queries_from_file(file_path):
    """
    Load queries from a text file. Each line in the file represents a single query.
    :param file_path: Path to the text file containing queries.
    :return: List of queries.
    """
    try:
        with open(file_path, 'r') as file:
            queries = [line.strip() for line in file.readlines() if line.strip()]
        print(f"Loaded {len(queries)} queries from {file_path}.")
        return queries
    except Exception as e:
        print(f"Error loading queries from file: {e}")
        return []




if __name__ == "__main__":

    #  PDF path
    # pdf_path = "/home/shtlp_0068/Documents/reranking/assets/FinancialManagement.pdf"  # Path to your PDF file

    # store_embeddings_in_mongo(pdf_path, collection)
    # Store embeddings for the PDF
    # pdf_path = "/home/shtlp_0068/Documents/reranking/assets/Gen AI.pdf"
    # store_embeddings_in_mongo(pdf_path, collection)

    # Query for initial ranking
    # queries = ["Which country is the current World Champion in cricket (as of the provided text)?"]
    # top_k = 5

    
    
    

    queries = [
            "What is the transfer time for an inactive caller in Auto Attendant?",
            "Default admin password for Poly soundpoint or soundstation factory reset?",
            "System requirements for Nextiva legacy app?",
            "Default admin password for Poly Trio factory reset?",
            "Default PIN for deregistering Yealink W56H?",
            "Limitation of Grandstream GXW 4200 ATAs regarding volume?",
            "What to do if phone payment fails?",
            "Where do call center threshold alerts show?",
            "How to enable pairing mode on Voyager 5200?",
            "How to access voicemail messages on Nextiva?",
            "Default admin password for Poly Trio reset?",
            "What to do if customer's PIN is unknown or not authorized?",
            "Default PIN for Yealink W56H deregistering from W70B?",
            "What happens with Directed Call Pickup and Barge-In?",
            "Purpose of the Panasonic Landing Page?",
            "What's the document context on bandwidth?",
            "Preferred method to set line mirroring on Panasonic KX-HDV-230?",
            "Guide for static IP setup on Cisco SPA phones?",
            "First step for static IP on Cisco CP device?",
            "Default password for static IP on Poly device via WebUI?",
            "How to deregister a handset from Panasonic TGP 600?",
            "What feature does Orbital Call Parking provide?",
            "How to reset a user's password on Nextiva?",
            "Default admin password for Poly Trio reset?",
            "How to factory reset Grandstream GXW 4200?",
            "Steps to verify a customer?",
            "Default admin password for Poly VVX factory reset?",
            "Default admin password for resetting Poly soundpoint?",
            "First step for Yealink headsets setup by Nextiva?",
            "Purpose of ACD_LOGIN custom tag on Panasonic KX-HDV230?",
            "Default PIN to deregister Yealink W56H?",
            "How to send SMS/MMS on NextivaONE app?",
            "Which Cisco phones support WiFi on Nextiva?",
            "What if Call2Teams setup person isn't a Global Administrator?",
            "Steps to add a user to Akixi portal?",
            "Process to enable international dialing on Nextiva?",
            "What to do if auto attendant plays the wrong name?",
            "How to deregister handset from Panasonic TGP 600?",
            "Default admin password for Poly Edge reset?",
            "How to send logs to Nextiva support via mobile app?",
            "Setup for Panasonic KX-HDV-230 for Pactolus accounts?",
            "Default admin password for Poly VVX factory reset?",
            "How to enable headset memory on Voyager 4310/4320?",
            "How to update credit card info in vfax billing portal?",
            "What's required for Bluetooth use with T46U phone and Voyager 5200?",
            "Default username/password for Snom C520 provisioning?",
            "How to factory reset Poly CCX 500/600?",
            "What to do with “Service usage limit is reached” error?",
            "What should be included in park user ID for 3.0 account?",
            "Login credentials for Panasonic TGP 600 web interface?",
            "Required firmware version for Panasonic KX-HDV-230 setup?",
            "Port for SFTP connections to Nextiva?",
            "Options for notifications when attendees join/leave a conference call?",
            "Who can set up Call2Teams account?",
            "Factory reset steps for Panasonic UTG 200B/300B?",
            "Can NextivaONE write to Outlook or Google calendars?",
            "Requirements to connect Poly Voyager Focus 2 without USB-A?",
            "Default admin password for Poly CCX factory reset?",
            "How to disable auto-answer on Panasonic UTG phones?",
            "BLF programming tag for Panasonic KX-HDV230?",
            "Call recording options in Nextiva Unity client?",
            "Default password for Poly device advanced settings?",
            "Custom tags for speed dial setup on Panasonic KX-HDV-230?",
            "Methods for verifying fax-only customer with Nextiva?",
            "What to do if calls drop sporadically across phones?",
            "What happens to calls outside business hours or holidays?",
            "How to reset Nextiva X-650 cordless phone via base unit?",
            "Solution for calls dropping between 10-11 minutes?",
            "Default username/password for Yealink W70B web interface?",
            "What happens when connecting Voyager 4310/4320 via USB?",
            "Max file size for attachments in Message Pro on NextivaONE?",
            "What topics are in Nextiva's support queues 'Platform' category?",
            "Two methods to factory reset Poly soundpoint or soundstation?",
            "How to set a custom date range for Nextiva Analytics report?",
            "What to do if customer prefers not to call for feature issues?",
            "License required for controlling call center settings in NextivaONE?",
            "Purpose of KMPIQ feature in Unity for call centers?",
            "Tool for accessing customer payments via phone transactions?",
            "How to view available voice task templates in Inference Portal?",
            "Correct value for SBC_OBP tag on Panasonic KX-HDV-230?",
            "Compatible headset brands with NextivaONE for call control?",
            "Minimum system requirements for NextivaONE macOS desktop app?",
            "Max participants in a single conference call using a conference bridge?",
            "Default admin password for Poly VVX factory reset?",
            "How to add a new integration in NextivaONE?",
            "How to deregister handset from Panasonic TGP 500/550?",
            "Default admin password for Poly CCX during factory reset?",
            "Requirements to set up Panasonic KX-HDV-230 for Pactolus accounts?",
            "Which Cisco phones have spec sheets on the landing page?",
            "What to do before configuring Yealink headsets by Nextiva?",
            "License required for controlling call center in NextivaONE?",
            "Procedure for requesting port push or port cancel?",
            "Supported hardware version for GXW 4200 by Nextiva?",
            "How to disable SIP ALG on Ubiquiti EdgeMAX 5.5+?",
            "Default admin password for Poly VVX reset?",
            "Information required to verify customer as per article?",
            "What are agent thresholds in call centers?",
            "How to set CommPilot Express status on Nextiva Unity?",
            "Custom tags for Shared Call Appearance on Panasonic KX-HDV-230?",
            "Factory reset steps for Grandstream GXW 4200?"
        ]

    process_queries(queries, collection, top_k=5)

    # Perform initial ranking
    # initial_results = initial_ranking(query_text, collection, top_k)
    
    # reranked_results = rerank_candidates(query_text, initial_results)
   


