import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import pandas as pd
import os
from sklearn.preprocessing import normalize

# File paths
EMBEDDINGS_FILE = "embeddings.npy"
INDEX_FILE = "faiss.index"
RECORDS_FILE = "records.npy"

# Step 1: Load data from Excel
file_path = "test_data/ra_1.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")

# Get column names
column_names = df.columns.tolist()

# Convert the DataFrame into a list of records (rows)
records = df.values.tolist()

# Step 2: Embed the records with column headers
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Check if embeddings exist
if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(INDEX_FILE) and os.path.exists(RECORDS_FILE):
    print("Loading precomputed embeddings and FAISS index...")

    # Load embeddings
    embeddings = np.load(EMBEDDINGS_FILE)
    records = np.load(RECORDS_FILE, allow_pickle=True).tolist()

    # Load FAISS index
    index = faiss.read_index(INDEX_FILE)

else:
    print("Generating new embeddings...")

    embedded_records = []
    embeddings = []

    for record in records:
        row_text = " | ".join([f"{col}: {val}" for col, val in zip(column_names, record)])
        print(f'{row_text}\n')
        embedding = model.encode(row_text, normalize_embeddings=True)
        
        embedded_records.append({
            "record": record,
            "row_text": row_text,
            "embedding": embedding
        })
        embeddings.append(embedding)

    # Convert to NumPy array
    embeddings = np.array(embeddings, dtype='float32')
    embeddings = normalize(embeddings, axis=1, norm='l2')

    # Save embeddings and records
    np.save(EMBEDDINGS_FILE, embeddings)
    np.save(RECORDS_FILE, np.array(records, dtype=object))

    # Initialize and save FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)

print("Embeddings and FAISS index ready.")

# Search function
def search(query_text, k=10):
    query_embedding = model.encode(query_text, normalize_embeddings=True)
    query_embedding = normalize(query_embedding.reshape(1, -1), axis=1, norm='l2').astype('float32')

    distances, indices = index.search(query_embedding, k)
    similar_records = []

    # Filter out results with low similarity scores
    threshold = 0.5 
    for i, idx in enumerate(indices[0]):
      if idx == -1 or distances[0][i] < threshold:
        continue
      similar_records.append({
        # "record": records[idx]["record"],
        # "formatted_text": records[idx]["row_text"],  # Include formatted row with headers
        # "distance": distances[0][i]
        "record": records[idx],
        "distance": distances[0][i]
      })

    return similar_records

# searchy = search("What are all the vulnerabilities associated with threat id M3.")
# print(searchy)

# Step 5: Set up the Llama model
llm = Llama(
    model_path="models/O1-OPEN.OpenO1-Qwen-7B-v0.1.Q2_K.gguf",
    n_ctx=4096,
    chat_format="qwen",
    main_gpu=1,
)

# Step 6: Query and augment the input to Llama
def augmented_query_to_llm(query_text, k=5):
    # Step 6.1: Perform the search to get the top-k similar records
    similar_records = search(query_text, k)

    print(f"Top {k} similar records: {similar_records}")

    system_prompt = """You are an AI assistant helping with risk analysis of threats and vulnerabilities in a mission critical system. 
    I will provide different scenarios and I want you to query your knowledge base to answer.
    Give the output in this format:
    - threat ID, 
    - vulnerability ID,
    - countermeasure ID
    - Your reasoning about the scenario.
    """
    
    # Step 6.2: Combine the similar records into a single string (to augment the query)
    augmented_query = f"""
      Below is the query:

      ### User Query:
      {query_text}

      ### Relevant Context:

      Based on this, provide an accurate response.
      """ # Start with the original query
    for record in similar_records:
        augmented_query += f"Related context: {' '.join(map(str, record['record']))}\n"
        augmented_query += f"Similarity score: {record['distance']:.4f}\n\n"

    # Step 6.3: Pass the augmented query to the Llama model for a response
    try:
      response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ],
        stream=True  # Enables streaming response
      )

      # Print output token-by-token
      for chunk in response:
          print(chunk['choices'][0]['delta'].get('content', ''), end='', flush=True)


    except Exception as e:
      print(f"An error occurred: {e}")
      response = None
    
    return response

# Example query
# query = input("Enter your query: ")
query = " What thread is associated to an ID of M3?" "
augmented_query_to_llm(query, k=3)

# print("Llama Response:", response)


