import psycopg2  # Imports the Postgree sql Library.
import os        # Provides access to the OS.
from dotenv import load_dotenv  # Loads variables from a .env file into the environment.

load_dotenv()  # Reads the .env file.


def search(query_embedding, top_k=5): # query_embeding (vector representation of the users question) and top_k (returns the 5 most smilar chunks) are the parameters.
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )  # Creates connection to PostgreSQL and os.getnev() gets the Locally running PostgreSQL details.

    cur = conn.cursor()  # cursor executes SQL statements.
    cur.execute(
        """
        SELECT content,
               1 - (embedding <=> %s::vector) AS similarity  
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_embedding, query_embedding, top_k),
    )  # Runs this sql inside the PostgreSQL.
       # <=> : pgvector operator computes vector distance.
       # 1 - (embedding <=> %s::vector) : Distance between the vectors is converted to similarity.
       # OREDR BY embedding <=> %s::vector : This part of the query sorts chunks by vector distance.
       # LIMIT %S : Returns onl y top matches.
    rows = cur.fetchall()  # retrieves all rows returned by the SQL.
    
    for content, similarity in rows:
        print(f"Similarity: {similarity:.4f}")
        print(content[:150])
        print()
        
    MIN_SIMILARITY = 0.70

    filtered_rows = [
        (content, similarity) for content, similarity in rows if similarity >= MIN_SIMILARITY
    ]
    
    cur.close() # Releases the cursor.
    conn.close() # Disconnects from the PostgreSQL.
    return filtered_rows  # Retrieved chunks have been recieved.