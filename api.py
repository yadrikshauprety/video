from fastapi import FastAPI, UploadFile, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil, os, sqlite3, json, time, threading, traceback
import numpy as np

from core.video_processor import extract_frames
from core.face_processor import process_and_link_faces, cluster_all_faces, remove_duplicate_faces
from core.embedder import generate_video_embedding, encode_text
from core.vector_store import add_vector, search_vector
from core.classifier import classify_video
from core.database import init_db, add_video, get_video_by_index, DB_PATH, link_face_to_person

app = FastAPI()

job_progress = {
    "bulk_index": {"status": "idle", "current": 0, "total": 0, "eta": 0, "message": ""},
    "clustering": {"status": "idle", "current": 0, "total": 0, "message": ""},
}
stop_flags = {"bulk_index": False, "clustering": False}

init_db()

UPLOAD_DIR = "uploaded_videos"
THUMB_DIR = "static/face_thumbnails"
AUDIO_DIR = "static/audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

app.mount("/stream", StaticFiles(directory=UPLOAD_DIR), name="stream")
app.mount("/faces", StaticFiles(directory=THUMB_DIR), name="faces")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

def process_single_video(file_path):
    print(f"DEBUG: Processing {file_path}")
    frames = extract_frames(file_path)
    if not frames:
        print(f"DEBUG: No frames extracted from {file_path}")
        return "unknown"
    v_emb = generate_video_embedding(frames)
    label = classify_video(v_emb)
    add_vector(v_emb)
    video_id = add_video(file_path, label, v_emb)
    process_and_link_faces(frames, video_id)
    return label

@app.post("/index-video")
async def index_video(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    label = process_single_video(file_path)
    return {"message": "Success", "label": label}

def run_bulk_index_task(directory_path: str):
    global job_progress, stop_flags
    try:
        stop_flags["bulk_index"] = False
        directory_path = os.path.normpath(directory_path.strip().replace('"', '').replace("'", ""))
        extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
        
        if not os.path.isdir(directory_path):
            job_progress["bulk_index"] = {"status": "error", "message": f"Not a directory: {directory_path}"}
            return

        files = [f for f in os.listdir(directory_path) if f.lower().endswith(extensions)]
        print(f"DEBUG: Found {len(files)} files in {directory_path}")
        
        job_progress["bulk_index"] = {
            "status": "processing", 
            "current": 0, 
            "total": len(files), 
            "eta": 0, 
            "start_time": time.time(), 
            "message": f"Indexing {len(files)} videos..."
        }
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM videos")
        indexed_paths = {os.path.normpath(r[0]) for r in cursor.fetchall()}
        conn.close()

        for i, filename in enumerate(files):
            if stop_flags["bulk_index"]:
                job_progress["bulk_index"]["status"] = "cancelled"
                job_progress["bulk_index"]["message"] = "Cancelled by user."
                return

            file_path = os.path.normpath(os.path.join(directory_path, filename))
            dest_path = os.path.normpath(os.path.join(UPLOAD_DIR, filename))
            
            if dest_path in indexed_paths or file_path in indexed_paths:
                print(f"DEBUG: Skipping {filename}, already indexed.")
            else:
                try:
                    if not os.path.exists(dest_path):
                        shutil.copy(file_path, dest_path)
                    process_single_video(dest_path)
                except Exception as e:
                    print(f"DEBUG: Error processing {filename}: {str(e)}")
                    print(traceback.format_exc())
                
            job_progress["bulk_index"]["current"] = i + 1
            elapsed = time.time() - job_progress["bulk_index"]["start_time"]
            avg_time = elapsed / (i + 1)
            job_progress["bulk_index"]["eta"] = avg_time * (len(files) - (i + 1))
            
        job_progress["bulk_index"]["status"] = "completed"
        job_progress["bulk_index"]["message"] = f"Finished indexing {len(files)} videos."
    except Exception as e:
        print(f"DEBUG: Bulk task failed: {str(e)}")
        print(traceback.format_exc())
        job_progress["bulk_index"]["status"] = "error"
        job_progress["bulk_index"]["message"] = str(e)

@app.post("/index-bulk")
async def index_bulk(directory_path: str, background_tasks: BackgroundTasks):
    normalized_path = os.path.normpath(directory_path.strip().replace('"', '').replace("'", ""))
    if not os.path.exists(normalized_path): 
        return {"error": f"Directory not found: {normalized_path}"}
    background_tasks.add_task(run_bulk_index_task, normalized_path)
    return {"message": "Started"}

# --- OTHER ENDPOINTS ---
@app.get("/job-status")
async def get_job_status(): return job_progress

@app.post("/cancel-job/{job_type}")
async def cancel_job(job_type: str):
    if job_type in stop_flags: stop_flags[job_type] = True; return {"message": "Cancelled"}
    return {"error": "Invalid"}

@app.post("/clear-jobs")
async def clear_jobs():
    global job_progress
    for job in job_progress: job_progress[job] = {"status": "idle", "current": 0, "total": 0, "message": ""}
    return {"message": "Cleared"}

@app.post("/search")
async def search(query: str, threshold: float = 0.15):
    prompts = [f"a video of {query}", f"a person {query}", f"a scene showing {query}", f"someone {query}", f"a close up of {query}", query]
    if any(x in query.lower() for x in ["sign", "gesture", "hand"]):
        prompts.append(f"a person using their hands to {query}"); prompts.append(f"manual communication or {query}")
    embs = [encode_text(p) for p in prompts]
    q_emb = np.mean(embs, axis=0); q_emb = q_emb / np.linalg.norm(q_emb)
    scores, indices = search_vector(q_emb); results = []
    for score, idx in zip(scores, indices):
        if float(score) >= threshold:
            video = get_video_by_index(idx)
            if video: results.append({"path": video[0], "label": video[1], "score": float(score)})
    return {"results": results}

@app.get("/videos")
async def get_all_videos():
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT path, label, created_at FROM videos"); res = [{"path": r[0], "label": r[1], "created_at": r[2]} for r in cursor.fetchall()]
    conn.close(); return {"videos": res}

@app.post("/extract-audio")
async def extract_audio_task(video_path: str):
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        filename = os.path.basename(video_path); audio_filename = f"{os.path.splitext(filename)[0]}.mp3"; audio_path = os.path.join(AUDIO_DIR, audio_filename)
        if not os.path.exists(audio_path):
            video = VideoFileClip(video_path)
            if video.audio: video.audio.write_audiofile(audio_path, logger=None); video.close()
            else: video.close(); return {"error": "No audio"}
        return {"audio_url": f"/audio/{audio_filename}"}
    except Exception as e: return {"error": str(e)}

@app.get("/all-persons")
async def get_all_persons():
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT p.id, p.name, p.thumbnail, COUNT(f.id) FROM persons p LEFT JOIN faces f ON p.id = f.person_id GROUP BY p.id")
    res = [{"id": r[0], "name": r[1], "thumbnail": r[2], "count": r[3]} for r in cursor.fetchall()]
    conn.close(); return {"persons": res}

@app.post("/rebuild-index")
async def rebuild_index():
    from core.vector_store import INDEX_PATH, DIM; import faiss
    if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
    new_index = faiss.IndexFlatIP(DIM); conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT embedding FROM videos ORDER BY id ASC")
    for r in cursor.fetchall():
        if r[0]: emb = np.array(json.loads(r[0])).astype("float32"); new_index.add(np.array([emb]))
    faiss.write_index(new_index, INDEX_PATH); import core.vector_store; core.vector_store.index = new_index
    conn.close(); return {"message": "Rebuilt"}

@app.delete("/remove-duplicates")
async def cleanup_duplicates(): return {"removed_count": remove_duplicate_faces()}

@app.get("/face-stats")
async def get_face_stats():
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM persons"); p = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM faces"); f = cursor.fetchone()[0]
    cursor.execute("SELECT p.id, COUNT(f.id) FROM persons p LEFT JOIN faces f ON p.id = f.person_id GROUP BY p.id")
    dist = {str(r[0]): r[1] for r in cursor.fetchall()}
    cursor.execute("SELECT confidence FROM faces"); confs = [r[0] for r in cursor.fetchall()]
    cursor.execute("SELECT video_id, COUNT(*) FROM faces GROUP BY video_id")
    v_dist = {str(r[0]): r[1] for r in cursor.fetchall()}
    conn.close(); return {"total_people": p, "total_faces": f, "distribution": dist, "confidences": confs, "video_distribution": v_dist}

@app.get("/face-gallery")
async def get_face_gallery():
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT f.person_id, f.thumbnail_path, f.confidence, v.path FROM faces f JOIN videos v ON f.video_id = v.id ORDER BY f.person_id")
    gallery = {}
    for r in cursor.fetchall():
        p_id = str(r[0]); gallery.setdefault(p_id, []).append({"thumbnail": r[1], "confidence": r[2], "video": os.path.basename(r[3])})
    conn.close(); return {"gallery": gallery}

@app.post("/name-person/{p_id}")
async def name_person(p_id: int, name: str = Query(...)):
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor(); cursor.execute("UPDATE persons SET name = ? WHERE id = ?", (name, p_id))
    conn.commit(); conn.close(); return {"status": "success"}

@app.get("/person-videos/{p_id}")
async def get_person_videos(p_id: int):
    conn = sqlite3.connect(DB_PATH); cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT v.path, v.label FROM videos v JOIN faces f ON v.id = f.video_id WHERE f.person_id = ?", (p_id,))
    res = [{"path": r[0], "label": r[1]} for r in cursor.fetchall()]
    conn.close(); return {"videos": res}