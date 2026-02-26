import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.eval()


# -------------------------
# IMAGE ENCODING (MANUAL + SAFE)
# -------------------------
def encode_image(frame):
    image = Image.fromarray(frame)

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        image_features = model.visual_projection(pooled_output)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy()[0]


# -------------------------
# TEXT ENCODING (MANUAL + SAFE)
# -------------------------
def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = text_outputs.pooler_output
        text_features = model.text_projection(pooled_output)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()[0]


# -------------------------
# VIDEO EMBEDDING
# -------------------------
def generate_video_embedding(frames):
    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    embeddings = [encode_image(frame) for frame in frames]

    video_embedding = np.mean(embeddings, axis=0)

    video_embedding = video_embedding / np.linalg.norm(video_embedding)

    return video_embedding