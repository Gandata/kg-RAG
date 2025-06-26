import os
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from openai import OpenAI # For vLLM
# from transformers import pipeline # If using local transformers pipeline

# --- Configuration from Environment Variables ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT_INTERNAL = os.getenv("DB_PORT_INTERNAL", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_SERVED_NAME = os.getenv("VLLM_MODEL_SERVED_NAME", "mistralai/Mistral-7B-Instruct-v0.1") # Default if not set

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- Initialize OpenAI client for vLLM ---
vllm_client = OpenAI(
    base_url=VLLM_API_BASE_URL,
    api_key="dummy_key_vllm_doesnt_need_auth_by_default" # vLLM doesn't require an API key by default
)

# --- Database Connection ---
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT_INTERNAL,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        register_vector(conn) # Register pgvector type
        print("Successfully connected to PostgreSQL via Docker Compose!")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

# --- Embedding Model ---
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding_model = None

# --- (Your embed_documents, retrieve_documents functions) ---
# Ensure they use the `get_db_connection()` and `embedding_model`

def generate_answer_vllm(query, context_docs):
    if not context_docs:
        return "I couldn't find relevant information to answer your question."
    if not vllm_client:
        return "vLLM client not initialized."

    context_str = "\n\n".join(context_docs)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question based ONLY on the provided context. If the context doesn't provide the answer, say so."},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
    ]
    
    print(f"\n--- Sending to vLLM (Model: {VLLM_MODEL_SERVED_NAME}) ---")
    print(f"User Message: {messages[-1]['content']}")
    print("--- End of vLLM Message ---\n")

    try:
        response = vllm_client.chat.completions.create(
            model=VLLM_MODEL_SERVED_NAME, # This must match the model vLLM is serving
            messages=messages,
            max_tokens=200,
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error during vLLM QA generation: {e}")
        return f"Sorry, I encountered an error with vLLM: {e}"

# --- Your main RAG logic (adapted from previous examples) ---
def embed_and_store_documents(docs_with_paths): # docs_with_paths = [(content, path), ...]
    if not embedding_model:
        print("Embedding model not loaded. Skipping embedding.")
        return
    conn = get_db_connection()
    if not conn: return
    cursor = conn.cursor()
    
    # Create table if not exists (idempotent)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        path TEXT UNIQUE,
        embedding VECTOR(384) -- Dimension of all-MiniLM-L6-v2
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embedding ON documents USING HNSW (embedding vector_l2_ops);") # Or IVFFLAT
    conn.commit()

    embeddings_to_insert = []
    print("Generating embeddings...")
    for content, path in docs_with_paths:
        # Check if document already exists by path
        cursor.execute("SELECT id FROM documents WHERE path = %s", (path,))
        if cursor.fetchone():
            print(f"Document '{path}' already exists. Skipping.")
            continue
        try:
            doc_embedding = embedding_model.encode(content).tolist()
            embeddings_to_insert.append((content, path, doc_embedding))
            print(f"Embedded: {path[:50]}...")
        except Exception as e:
            print(f"Error embedding document {path}: {e}")
    
    if embeddings_to_insert:
        print(f"Inserting {len(embeddings_to_insert)} new documents into PostgreSQL...")
        execute_values(cursor, 
                       "INSERT INTO documents (content, path, embedding) VALUES %s",
                       embeddings_to_insert)
        conn.commit()
        print(f"Successfully inserted {len(embeddings_to_insert)} documents.")
    else:
        print("No new documents to insert.")
        
    cursor.close()
    conn.close()

def retrieve_relevant_documents(query_text, top_k=3):
    if not embedding_model:
        print("Embedding model not available for retrieval.")
        return []
    conn = get_db_connection()
    if not conn: return []
    cursor = conn.cursor()

    query_embedding = embedding_model.encode(query_text).tolist()
    
    # Retrieve using cosine similarity (1 - L2 distance for normalized vectors)
    # Or use <-> for L2 distance, <#> for inner product (negative for similarity)
    cursor.execute("""
        SELECT content, path FROM documents
        ORDER BY embedding <-> %s::vector
        LIMIT %s;
    """, (query_embedding, top_k)) # <-> is L2 distance
    
    results = cursor.fetchall()
    retrieved_docs_content = [res[0] for res in results] # just content
    
    print(f"\nRetrieved {len(retrieved_docs_content)} documents:")
    for i, res in enumerate(results):
        print(f"  Doc {i+1} ({res[1]}): {res[0][:100]}...")

    cursor.close()
    conn.close()
    return retrieved_docs_content

def main_interactive_rag():
    # Example: Create some dummy documents in rag_app/data if they don't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    doc_files_info = [
        ("doc1.txt", "The capital of France is Paris. Paris is known for the Eiffel Tower and the Louvre Museum."),
        ("doc2.txt", "Berlin is the capital of Germany. The Brandenburg Gate is a famous landmark in Berlin."),
        ("doc3.txt", "The Python programming language is versatile and widely used in data science and web development.")
    ]
    docs_to_embed = []
    for fname, content in doc_files_info:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                f.write(content)
            print(f"Created dummy document: {fpath}")
        docs_to_embed.append((content, fpath)) # Embed the content along with its path

    # Embed documents (idempotent due to path check)
    print("\n--- Embedding Documents ---")
    embed_and_store_documents(docs_to_embed)

    print("\n--- RAG System Ready ---")
    while True:
        user_query = input("\nAsk a question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        if not user_query.strip():
            continue

        print(f"\nProcessing query: '{user_query}'")
        
        # 1. Retrieve relevant documents
        retrieved_docs = retrieve_relevant_documents(user_query, top_k=2)
        
        if not retrieved_docs:
            print("No relevant documents found to answer the query.")
            continue

        # 2. Generate answer using vLLM
        print("\nGenerating answer using vLLM...")
        answer = generate_answer_vllm(user_query, retrieved_docs)
        
        print(f"\nLLM Answer: {answer}")

if __name__ == "__main__":
    if not embedding_model:
        print("Critical error: Embedding model could not be loaded. Exiting.")
    else:
        main_interactive_rag()