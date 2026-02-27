from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
import shutil
import os

from core.video_processor import extract_frames
from core.embedder import generate_video_embedding, encode_text
from core.vector_store import add_vector, search_vector
from core.classifier import classify_video
from core.database import init_db, add_video, get_video_by_index, DB_PATH # Added DB_PATH
import sqlite3 # Added to fetch all videos

app = FastAPI()
init_db()

UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/stream", StaticFiles(directory=UPLOAD_DIR), name="stream")

@app.get("/videos")
async def get_all_videos():
    """Fetches all videos from the database for the gallery view."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT path, label FROM videos")
    results = cursor.fetchall()
    conn.close()
    return {"videos": results}

@app.post("/index-video")
async def index_video(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    frames = extract_frames(file_path)
    video_embedding = generate_video_embedding(frames)
    label = classify_video(video_embedding)

    add_vector(video_embedding)
    add_video(file_path, label)

    return {"message": "Video indexed", "label": label}

@app.post("/search")
async def search(query: str):
    query_embedding = encode_text(query)
    scores, indices = search_vector(query_embedding)

    results = []
    for idx in indices:
        video = get_video_by_index(idx)
        if video:
            results.append(video)
    return {"results": results}