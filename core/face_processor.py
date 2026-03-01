import cv2
import os
import uuid
import numpy as np
import json
import sqlite3
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# 1. Global Initialization
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

FACES_DIR = "static/face_thumbnails"
os.makedirs(FACES_DIR, exist_ok=True)
DB_PATH = "data/videos.db"

def get_known_people():
    """Fetches unique identities and their embeddings from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # We group by person_id to get one 'representative' face for each identity
    cursor.execute("SELECT person_id, embedding FROM faces GROUP BY person_id")
    data = cursor.fetchall()
    conn.close()
    return [(row[0], np.array(json.loads(row[1]))) for row in data]

def process_and_link_faces(frames, video_id):
    """Detects faces, groups them by identity, and links them to the database."""
    # Load people already in the DB at the start of the process
    known_people = get_known_people()
    
    for frame in frames:
        faces = face_app.get(frame)
        for face in faces:
            new_emb = face.normed_embedding
            matched_person_id = None
            
            # 2. Similarity Check (Grouping Logic)
            # Threshold 0.45: Lower = stricter, Higher = looser
            for p_id, p_emb in known_people:
                score = cosine(new_emb, p_emb)
                if score < 0.45:  # This person is already in the system
                    matched_person_id = p_id
                    break
            
            # 3. Crop and Save Thumbnail
            bbox = face.bbox.astype(int)
            # Ensure crop is within frame boundaries to avoid errors
            y1, y2, x1, x2 = max(0, bbox[1]), bbox[3], max(0, bbox[0]), bbox[2]
            face_img = frame[y1:y2, x1:x2]
            
            thumb_name = f"{uuid.uuid4()}.jpg"
            if face_img.size > 0:
                cv2.imwrite(os.path.join(FACES_DIR, thumb_name), face_img)

            # 4. Handle New Identities
            if matched_person_id is None:
                # Create a new person record if no match was found
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute("INSERT INTO persons (name, thumbnail) VALUES (?, ?)", (None, thumb_name))
                matched_person_id = cur.lastrowid
                conn.commit()
                conn.close()
                
                # CRITICAL: Update local list so subsequent frames in THIS video
                # recognize this person instead of creating another duplicate.
                known_people.append((matched_person_id, new_emb))

            # 5. Link this specific detection to the identified person
            # We import here to avoid potential circular import issues
            from core.database import link_face_to_person
            link_face_to_person(video_id, matched_person_id, new_emb.tolist(), thumb_name)