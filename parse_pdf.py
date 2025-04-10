import pymupdf
from pathlib import Path
import psycopg
import json
from sentence_transformers import SentenceTransformer

# Config
pdf_path = "./documents/dafi36-2903.pdf"
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract pdf data into chunks
def extract_chunks(pdf_path):
    doc = pymupdf.open(pdf_path)
    chunks = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        # Split page into paragraphs
        for para in page_text.split("\n\n"):
            cleaned = para.strip()
            if len(cleaned) > 30:  # Filter out noise
                chunks.append({
                    "content": cleaned,
                    "metadata": {
                        "source": Path(pdf_path).name,
                        "page": page_num + 1
                    }
                })

    return chunks

# connect to db and insert
chunks = extract_chunks(pdf_path)

with psycopg.connect("dbname=ragdb user=michael host=localhost port=5432") as conn:
    # Open a cursor to perform database operations
    with conn.cursor() as cur:
        for chunk in chunks:
            content = chunk["content"]
            metadata = chunk["metadata"]
            embedding = model.encode(content).tolist()

            cur.execute(
                """
                INSERT INTO documents (content, embedding, metadata)
                VALUES (%s, %s, %s)
                """,
                (content, embedding, json.dumps(metadata))
            )

    conn.commit()
