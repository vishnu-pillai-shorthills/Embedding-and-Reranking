import json

def normalize_text(text):
    """Normalize text by removing extra spaces, converting to lowercase, and handling newlines."""
    if isinstance(text, str):  # Ensure that the text is a string before normalizing
        return " ".join(text.lower().replace("\n", " ").split())
    return ""

def calculate_matching_queries(before_file, predicted_file):
    
    with open(before_file, 'r') as before_f, open(predicted_file, 'r') as predicted_f:
        before_data = json.load(before_f)
        predicted_data = json.load(predicted_f)

    matching_count = 0
    total_queries = len(before_data)

    for query_data in before_data:
        query = query_data['query']
        before_chunks = [
            normalize_text(chunk['text'] if isinstance(chunk, dict) and 'text' in chunk else chunk) 
            for chunk in query_data['chunks']
        ]
        predicted_query_data = next((item for item in predicted_data if item['query'] == query), None)
        
        if predicted_query_data:
            # Handle cases where 'chunks' is a list of strings (as seen in predicted_data)
            predicted_chunks = [
                normalize_text(chunk)  # Here we are directly normalizing the chunk since it's a string
                for chunk in predicted_query_data['chunks']
                if isinstance(chunk, str)
            ]
            
            # Check if any predicted chunk matches any before chunk (using substring match)
            if any(predicted_chunk in before_chunk for before_chunk in before_chunks for predicted_chunk in predicted_chunks):
                matching_count += 1
            else:
                print(query)
        else:
            print(f"Warning: Query '{query}' not found in predicted data.")

    return matching_count, total_queries

# Paths to the JSON files
before_file = "after_reranking_who.json"
predicted_file = "data/chunks_who.json"
# predicted_file = "data/predict_chunks.json"

# Calculate matching queries
matches, total = calculate_matching_queries(before_file, predicted_file)

# Display the result
print(f"Total Queries: {total}")
print(f"Matching Queries: {matches}")
print(f"Percentage Match: {matches / total * 100:.2f}%")
