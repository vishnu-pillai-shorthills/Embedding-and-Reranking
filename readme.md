# Embedding and Reranking System
 
## Overview
 
This project focuses on embedding-based search and reranking using MongoDB for storing embeddings. It processes JSONL files, extracts textual data, generates embeddings, and performs semantic search with reranking.
 
## Features
 
- Extracts and processes text from JSONL files.
- Splits documents into meaningful chunks.
- Generates embeddings using SentenceTransformers.
- Stores embeddings in MongoDB.
- Performs initial semantic search using cosine similarity.
- Re-ranks top results using a cross-encoder model.
- Saves search results to JSON files.
 
## Technologies Used
- Python
- MongoDB
- Transformers
- CrossEncoder (MS Marco)
- tqdm
- NumPy
## Installation
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up MongoDB:
   - Ensure MongoDB is running.
   - Add the MongoDB connection string to `.env`:
     ```sh
     MONGO_URL=<your_mongodb_connection_string>
 
## Prerequisites
 
Ensure you have the following installed:
 
- Python (>= 3.8)
- Git
- Virtual Environment (`venv`)
- Required Python libraries (listed in `requirements.txt`)
 
### Install Dependencies
 
```sh
pip install -r requirements.txt
```

### insert_embeddings.py

This file is ran to insert the chunks of the dataset in the mongoDb database. 
```sh
python insert_embeddings.py
```

### app2.py

This file is used to convert the queries into embeddings and then those queries are used to do the similarity search with the chunks in the mongoDB database and the top 5 chunks are taken before reranking and saved in a json file. A reranker is used to apply on the top 20 similar chunks to generate even more accurate results and the top 5 chunks are stored in a separate json file.
```sh
python app2.py
```

### matching.py

This file is used to check if the top 5 chunks for a particular query match with the query chunk pair in the original dataset. Both the chunks are normalized and then compared. The percentage of the number of queries which get the chunk required is printed.

```sh
python matching.py
```