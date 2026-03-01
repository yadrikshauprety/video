import sqlite3
import json
import os

DB_PATH = "data/videos.db"
os.makedirs("data", exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Videos Table
    cursor.execute("CREATE TABLE IF NOT EXISTS videos (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT, label TEXT)")
    
    # Persons Table
    cursor.execute("CREATE TABLE IF NOT EXISTS persons (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, thumbnail TEXT)")
    
    # Faces Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            person_id INTEGER,
            embedding TEXT,
            thumbnail_path TEXT,
            FOREIGN KEY(video_id) REFERENCES videos(id),
            FOREIGN KEY(person_id) REFERENCES persons(id)
        )
    """)
    conn.commit()
    conn.close()

def add_video(path, label):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO videos (path, label) VALUES (?, ?)", (path, label))
    v_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return v_id

# --- THE MISSING FUNCTION ---
def get_video_by_index(index_id):
    """Retrieves video details based on the vector index (FAISS index)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Note: SQLite IDs usually start at 1, FAISS indices start at 0
    cursor.execute("SELECT path, label FROM videos WHERE id=?", (int(index_id) + 1,))
    result = cursor.fetchone()
    conn.close()
    return result

def link_face_to_person(video_id, person_id, embedding, thumb):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO faces (video_id, person_id, embedding, thumbnail_path) VALUES (?, ?, ?, ?)",
        (video_id, person_id, json.dumps(embedding), thumb)
    )
    conn.commit()
    conn.close()