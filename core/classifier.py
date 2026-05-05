import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

# Lazy loading to prevent hanging on imports
_model = None
_processor = None
model_id = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_clip_model():
    global _model, _processor
    if _model is None:
        print(f"Loading CLIP model on {device}...")
        _model = CLIPModel.from_pretrained(model_id).to(device)
        _processor = CLIPProcessor.from_pretrained(model_id)
    return _model, _processor

LABELS = [
    "person doing sign language",
    "person gesturing with hands",
    "person walking",
    "person running",
    "person sitting",
    "person talking",
    "person standing",
    "person dancing",
    "person typing on a laptop",
    "person cooking in a kitchen",
    "person exercising or working out",
    "person driving a car",
    "reading a book",
    "eating or drinking",
    "playing a musical instrument"
]

def classify_video(video_embedding):
    """
    Classifies a video based on its average CLIP embedding using zero-shot classification.
    """
    model, processor = get_clip_model()
    
    inputs = processor(text=LABELS, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    
    if isinstance(video_embedding, np.ndarray):
        video_embedding = torch.from_numpy(video_embedding).to(device).float()
    
    similarity = (100.0 * video_embedding @ text_features.T).softmax(dim=-1)
    top_idx = similarity.argmax().item()
    return LABELS[top_idx]