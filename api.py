from fastapi import FastAPI, UploadFile, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
import shutil, os, sqlite3, json, time
import numpy as np

# Importing your core logic modules
from core.video_processor import extract_frames
from core.face_processor import process_and_link_faces, cluster_all_faces, remove_duplicate_faces
from core.embedder import generate_video_embedding, encode_text
from core.vector_store import add_vector, search_vector
from core.classifier import classify_video
from core.database import init_db, add_video, get_video_by_index, DB_PATH, link_face_to_person

app = FastAPI()

# Enhanced Global state for all background jobs
job_progress = {
    "bulk_index": {"status": "idle", "current": 0, "total": 0, "eta": 0, "message": ""},
    "clustering": {"status": "idle", "current": 0, "total": 0, "message": ""}
}

# Ensure database tables are created on startup
init_db()

# Define and create necessary directories
UPLOAD_DIR = "uploaded_videos"
THUMB_DIR = "static/face_thumbnails"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)

app.mount("/stream", StaticFiles(directory=UPLOAD_DIR), name="stream")
app.mount("/faces", StaticFiles(directory=THUMB_DIR), name="faces")

def process_single_video(file_path):
    """Internal helper to process a video from path."""
    frames = extract_frames(file_path)
    v_emb = generate_video_embedding(frames)
    label = classify_video(v_emb)
    add_vector(v_emb)
    video_id = add_video(file_path, label, v_emb) # Storing embedding in DB
    process_and_link_faces(frames, video_id)
    return label

@app.post("/index-video")
async def index_video(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    label = process_single_video(file_path)
    return {"message": "Success", "label": label}

def run_bulk_index_task(directory_path: str):
    global job_progress
    try:
        directory_path = os.path.normpath(directory_path.strip().replace('"', '').replace("'", ""))
        extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
        files = [f for f in os.listdir(directory_path) if f.lower().endswith(extensions)]
        
        job_progress["bulk_index"] = {"status": "processing", "current": 0, "total": len(files), "eta": 0, "start_time": time.time(), "message": "Indexing videos..."}
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM videos")
        indexed_paths = {r[0] for r in cursor.fetchall()}
        conn.close()

        for i, filename in enumerate(files):
            try:
                file_path = os.path.join(directory_path, filename)
                dest_path = os.path.join(UPLOAD_DIR, filename)
                if dest_path in indexed_paths or file_path in indexed_paths:
                    pass
                else:
                    if not os.path.exists(dest_path):
                        shutil.copy(file_path, dest_path)
                    process_single_video(dest_path)
                
                job_progress["bulk_index"]["current"] = i + 1
                elapsed = time.time() - job_progress["bulk_index"]["start_time"]
                avg_time = elapsed / (i + 1)
                job_progress["bulk_index"]["eta"] = avg_time * (len(files) - (i + 1))
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
        job_progress["bulk_index"]["status"] = "completed"
    except Exception as e:
        job_progress["bulk_index"]["status"] = "error"
        job_progress["bulk_index"]["message"] = str(e)

@app.post("/index-bulk")
async def index_bulk(directory_path: str, background_tasks: BackgroundTasks):
    normalized_path = os.path.normpath(directory_path.strip().replace('"', '').replace("'", ""))
    if not os.path.exists(normalized_path):
        return {"error": f"Directory not found: {normalized_path}"}
    background_tasks.add_task(run_bulk_index_task, normalized_path)
    return {"message": "Bulk processing started"}

def run_clustering_task():
    global job_progress
    try:
        from core.face_processor import clustering_progress
        job_progress["clustering"] = {"status": "processing", "current": 0, "total": 100, "message": "Initializing..."}
        
        # We need to run cluster_all_faces in a way that we can see its progress
        # Since it updates its own global clustering_progress, we'll poll it
        import threading
        t = threading.Thread(target=cluster_all_faces)
        t.start()
        
        while t.is_alive():
            from core.face_processor import clustering_progress as cp
            job_progress["clustering"].update(cp)
            time.sleep(0.5)
            
        job_progress["clustering"]["status"] = "completed"
        job_progress["clustering"]["message"] = "Clustering complete!"
    except Exception as e:
        job_progress["clustering"]["status"] = "error"
        job_progress["clustering"]["message"] = str(e)

@app.post("/cluster-faces")
async def start_clustering(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_clustering_task)
    return {"message": "Clustering started"}

@app.get("/job-status")
async def get_job_status():
    return job_progress

# --- REST OF API ENDPOINTS ---
@app.post("/search")
async def search(query: str, threshold: float = 0.15):
    prompts = [f"a video of {query}", f"a person {query}", f"a scene showing {query}", query]
    embs = [encode_text(p) for p in prompts]
    q_emb = np.mean(embs, axis=0)
    q_emb = q_emb / np.linalg.norm(q_emb)
    scores, indices = search_vector(q_emb)
    results = []
    for score, idx in zip(scores, indices):
        if float(score) >= threshold:
            video = get_video_by_index(idx)
            if video:
                results.append({"path": video[0], "label": video[1], "score": float(score)})
    return {"results": results}

@app.get("/videos")
async def get_all_videos():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT path, label, created_at FROM videos")
    res = [{"path": r[0], "label": r[1], "created_at": r[2]} for r in cursor.fetchall()]
    conn.close()
    return {"videos": res}

@app.delete("/videos/delete-by-time")
async def delete_videos_by_time(hours: float = Query(...)):
    """Deletes videos indexed within the last X hours."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Find videos to delete
    cursor.execute("SELECT id, path FROM videos WHERE created_at >= datetime('now', '-' || ? || ' hours')", (hours,))
    to_delete = cursor.fetchall()
    
    deleted_count = 0
    for v_id, path in to_delete:
        # Delete from disk (optional, but requested 'remove videos')
        if os.path.exists(path):
            try: os.remove(path)
            except: pass
            
        # Delete from DB
        cursor.execute("DELETE FROM videos WHERE id = ?", (v_id,))
        # Delete associated faces
        cursor.execute("SELECT thumbnail_path FROM faces WHERE video_id = ?", (v_id,))
        face_thumbs = cursor.fetchall()
        for f in face_thumbs:
            f_path = os.path.join(THUMB_DIR, f[0])
            if os.path.exists(f_path):
                try: os.remove(f_path)
                except: pass
        cursor.execute("DELETE FROM faces WHERE video_id = ?", (v_id,))
        deleted_count += 1
    
    conn.commit()
    conn.close()
    
    # Also need to rebuild FAISS after deletion to stay in sync
    await rebuild_index()
    
    return {"message": f"Deleted {deleted_count} videos and cleaned up associated data.", "count": deleted_count}

@app.get("/all-persons")
async def get_all_persons():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.id, p.name, p.thumbnail, COUNT(f.id) as face_count 
        FROM persons p 
        LEFT JOIN faces f ON p.id = f.person_id 
        GROUP BY p.id
    """)
    res = [{"id": r[0], "name": r[1], "thumbnail": r[2], "count": r[3]} for r in cursor.fetchall()]
    conn.close()
    return {"persons": res}

@app.post("/rebuild-index")
async def rebuild_index():
    """Wipes the FAISS index and rebuilds it from the database."""
    from core.vector_store import INDEX_PATH, DIM
    import faiss
    
    # 1. Clear FAISS index
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    
    new_index = faiss.IndexFlatIP(DIM)
    
    # 2. Fetch all videos with embeddings
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT embedding FROM videos ORDER BY id ASC")
    rows = cursor.fetchall()
    
    added_count = 0
    for r in rows:
        if r[0]:
            emb = np.array(json.loads(r[0])).astype("float32")
            new_index.add(np.array([emb]))
            added_count += 1
    
    # 3. Save new index
    faiss.write_index(new_index, INDEX_PATH)
    
    # 4. Update the global index object in vector_store (it will reload on next add)
    import core.vector_store
    core.vector_store.index = new_index
    
    conn.close()
    return {"message": "Index rebuilt successfully", "vectors_added": added_count}

@app.delete("/remove-duplicates")
async def cleanup_duplicates():
    count = remove_duplicate_faces()
    return {"message": "Cleanup complete", "removed_count": count}

@app.get("/face-stats")
async def get_face_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM persons")
    total_people = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM faces")
    total_faces = cursor.fetchone()[0]
    cursor.execute("SELECT p.id, COUNT(f.id) FROM persons p LEFT JOIN faces f ON p.id = f.person_id GROUP BY p.id")
    distribution = {str(r[0]): r[1] for r in cursor.fetchall()}
    cursor.execute("SELECT confidence FROM faces")
    confidences = [r[0] for r in cursor.fetchall()]
    cursor.execute("SELECT video_id, COUNT(*) FROM faces GROUP BY video_id")
    video_dist = {str(r[0]): r[1] for r in cursor.fetchall()}
    conn.close()
    return {"total_people": total_people, "total_faces": total_faces, "distribution": distribution, "confidences": confidences, "video_distribution": video_dist}

@app.get("/face-gallery")
async def get_face_gallery():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT f.person_id, f.thumbnail_path, f.confidence, v.path as video_path
        FROM faces f JOIN videos v ON f.video_id = v.id
        ORDER BY f.person_id
    """)
    rows = cursor.fetchall()
    gallery = {}
    for r in rows:
        p_id = str(r[0])
        if p_id not in gallery: gallery[p_id] = []
        gallery[p_id].append({"thumbnail": r[1], "confidence": r[2], "video": os.path.basename(r[3])})
    conn.close()
    return {"gallery": gallery}

@app.post("/name-person/{p_id}")
async def name_person(p_id: int, name: str = Query(...)):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE persons SET name = ? WHERE id = ?", (name, p_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/person-videos/{p_id}")
async def get_person_videos(p_id: int):
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