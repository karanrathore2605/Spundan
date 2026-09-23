from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts):
    """
    Convert a list of texts into embedding vectors.
    """

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings