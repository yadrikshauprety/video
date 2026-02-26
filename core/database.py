import sqlite3

DB_PATH = "data/videos.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT,
            label TEXT
        )
    """)

    conn.commit()
    conn.close()


def add_video(path, label):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("INSERT INTO videos (path, label) VALUES (?, ?)", (path, label))

    conn.commit()
    conn.close()


def get_video_by_index(index_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT path, label FROM videos WHERE id=?", (index_id+1,))
    result = cursor.fetchone()

    conn.close()
    return result
