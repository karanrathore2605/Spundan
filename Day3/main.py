import os
import re
import faiss
from sentence_transformers import SentenceTransformer


# ========================================
# CONFIGURATION
# ========================================

DOCUMENT_FOLDER = "documents"

SENTENCES_PER_CHUNK = 3
OVERLAP_SENTENCES = 1


# ========================================
# LOAD EMBEDDING MODEL
# ========================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ========================================
# LOAD DOCUMENTS
# ========================================

def load_documents():

    documents = []

    for filename in os.listdir(DOCUMENT_FOLDER):

        if filename.endswith(".txt"):

            file_path = os.path.join(
                DOCUMENT_FOLDER,
                filename
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read()

            documents.append({
                "source": filename,
                "text": text
            })

    return documents


# ========================================
# CREATE MEANINGFUL CHUNKS
# ========================================

def create_chunks(
    text,
    sentences_per_chunk=2,
    overlap_sentences=1
):

    # Split text into sentences
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    chunks = []

    start = 0

    while start < len(sentences):

        # Select sentences for current chunk
        end = start + sentences_per_chunk

        chunk_sentences = sentences[start:end]

        # Join sentences together
        chunk = " ".join(chunk_sentences)

        chunks.append(chunk)

        # Move forward while keeping overlap
        start = end - overlap_sentences

    return chunks


# ========================================
# MAIN
# ========================================

def main():

    # ====================================
    # 1. LOAD DOCUMENTS
    # ====================================

    documents = load_documents()

    print(f"\nTotal documents: {len(documents)}")


    # ====================================
    # 2. CREATE CHUNKS
    # ====================================

    chunks = []

    for document in documents:

        document_chunks = create_chunks(
            document["text"],
            sentences_per_chunk=SENTENCES_PER_CHUNK,
            overlap_sentences=OVERLAP_SENTENCES
        )

        for chunk in document_chunks:

            chunks.append({
                "text": chunk,
                "source": document["source"]
            })


    # ====================================
    # 3. DISPLAY CHUNKS
    # ====================================

    print("\n========================================")
    print("        DAY 3: EMBEDDINGS + CHUNKING")
    print("========================================")

    print(f"\nTotal chunks: {len(chunks)}")


    print("\n----------------------------------------")
    print("CHUNKS")
    print("----------------------------------------")

    for i, chunk in enumerate(chunks):

        print(f"\nChunk {i}:")
        print(f"Source: {chunk['source']}")
        print(f"Text: {chunk['text']}")


    # ====================================
    # 4. EXTRACT CHUNK TEXT
    # ====================================

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]


    # ====================================
    # 5. GENERATE EMBEDDINGS
    # ====================================

    print("\n\n========================================")
    print("        EMBEDDING GENERATION")
    print("========================================")

    embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True
    )

    print(f"\nEmbedding shape: {embeddings.shape}")


    # ====================================
    # 6. CREATE FAISS INDEX
    # ====================================

    print("\n\n========================================")
    print("        VECTOR DATABASE")
    print("========================================")

    # Number of dimensions in each vector
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to FAISS
    index.add(embeddings)

    print(
        f"\nVectors stored in FAISS: {index.ntotal}"
    )


    # ====================================
    # 7. USER QUERY
    # ====================================

    query = "What is gradient descent?"


    print("\n\n========================================")
    print("        SEMANTIC SEARCH")
    print("========================================")

    print("\nQuery:")
    print(query)


    # ====================================
    # 8. GENERATE QUERY EMBEDDING
    # ====================================

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )


    # ====================================
    # 9. SEARCH TOP 3
    # ====================================

    distances, indices = index.search(
        query_embedding,
        3
    )


    # ====================================
    # 10. DISPLAY RESULTS
    # ====================================

    print("\nTop 3 relevant chunks:")

    for rank, chunk_index in enumerate(
        indices[0],
        start=1
    ):

        result = chunks[chunk_index]

        print("\n----------------------------------------")

        print(f"Result {rank}")

        print(f"Chunk {chunk_index}")

        print(f"Source: {result['source']}")

        print(f"Distance: {distances[0][rank - 1]:.4f}")

        print(f"Text: {result['text']}")


    # ====================================
    # 11. COMPLETED
    # ====================================

    print("\n========================================")
    print("        SEARCH COMPLETED")
    print("========================================")


# ========================================
# PROGRAM ENTRY POINT
# ========================================

if __name__ == "__main__":
    main()