from fastapi import FastAPI, UploadFile, Query
from fastapi.staticfiles import StaticFiles
import shutil, os, sqlite3, json

# Importing your core logic modules
from core.video_processor import extract_frames
from core.face_processor import process_and_link_faces
from core.embedder import generate_video_embedding, encode_text
from core.vector_store import add_vector, search_vector
from core.classifier import classify_video
from core.database import init_db, add_video, get_video_by_index, DB_PATH

app = FastAPI()

# Ensure database tables are created on startup
init_db()

# Define and create necessary directories
UPLOAD_DIR = "uploaded_videos"
THUMB_DIR = "static/face_thumbnails"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)

# Static mounting for video streaming and face thumbnails
# This allows the frontend to access files via URL
app.mount("/stream", StaticFiles(directory=UPLOAD_DIR), name="stream")
app.mount("/faces", StaticFiles(directory=THUMB_DIR), name="faces")

@app.post("/index-video")
async def index_video(file: UploadFile):
    """
    Handles video upload, scene-aware frame extraction, 
    semantic action indexing, and face identity linking.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Extract Frames (using Scene Detection logic)
    frames = extract_frames(file_path)
    
    # 2. Action Semantic Indexing (CLIP)
    v_emb = generate_video_embedding(frames)
    label = classify_video(v_emb)
    add_vector(v_emb) # Adds to FAISS vector store
    
    # 3. Save to SQLite and Process Face Identities
    video_id = add_video(file_path, label)
    process_and_link_faces(frames, video_id)
    
    return {"message": "Success", "label": label}

@app.post("/search")
async def search(query: str, threshold: float = 0.25):
    """
    Performs semantic search using CLIP embeddings.
    Filters results based on the confidence threshold.
    """
    # Prompt Engineering: wrapping query for better CLIP performance
    refined_query = f"a video of a person {query}"
    q_emb = encode_text(refined_query)
    scores, indices = search_vector(q_emb)

    results = []
    for score, idx in zip(scores, indices):
        if float(score) >= threshold:
            # get_video_by_index retrieves the file path from DB using FAISS index
            video = get_video_by_index(idx)
            if video:
                results.append({
                    "path": video[0], 
                    "label": video[1], 
                    "score": float(score)
                })
    return {"results": results}

@app.get("/videos")
async def get_all_videos():
    """Returns all indexed videos for the Gallery view."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT path, label FROM videos")
    res = [{"path": r[0], "label": r[1]} for r in cursor.fetchall()]
    conn.close()
    return {"videos": res}

@app.get("/all-persons")
async def get_all_persons():
    """Returns unique identified persons for the Identity Manager."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, thumbnail FROM persons")
    res = [{"id": r[0], "name": r[1], "thumbnail": r[2]} for r in cursor.fetchall()]
    conn.close()
    return {"persons": res}

@app.post("/name-person/{p_id}")
async def name_person(p_id: int, name: str = Query(...)):
    """Updates the name of a specific person identity."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE persons SET name = ? WHERE id = ?", (name, p_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/person-videos/{p_id}")
async def get_person_videos(p_id: int):
    """Retrieves all videos where a specific person appears."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT v.path, v.label 
        FROM videos v JOIN faces f ON v.id = f.video_id 
        WHERE f.person_id = ?
    """, (p_id,))
    res = [{"path": r[0], "label": r[1]} for r in cursor.fetchall()]
    conn.close()
    return {"videos": res}