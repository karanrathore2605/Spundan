import faiss
import numpy as np


def create_index(embeddings):
    """
    Create a FAISS index from embeddings.
    """

    # Number of dimensions in each vector
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Convert to float32
    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    # Add embeddings to FAISS
    index.add(embeddings)

    return index


def search(index, query_embedding, top_k=3):
    """
    Search for the most similar vectors.
    """

    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    return distances[0], indices[0]