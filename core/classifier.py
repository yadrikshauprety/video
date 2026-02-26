from .embedder import encode_text
import numpy as np

UCF_LABELS = [
    "playing basketball",
    "playing guitar",
    "boxing",
    "horse riding",
    "diving",
    "running",
]

# Cache for text embeddings
text_embeddings_cache = None


def get_text_embeddings():
    global text_embeddings_cache
    if text_embeddings_cache is None:
        text_embeddings_cache = [
            encode_text(label) for label in UCF_LABELS
        ]
    return text_embeddings_cache


def classify_video(video_embedding):
    text_embeddings = get_text_embeddings()

    similarities = [
        np.dot(video_embedding, text_emb)
        for text_emb in text_embeddings
    ]

    best_index = np.argmax(similarities)
    return UCF_LABELS[best_index]