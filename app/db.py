import sqlite3
import os

def create_connection(db_file="db/corpus.sqlite"):
    os.makedirs(os.path.dirname(db_file),exist_ok=True)
    conn=sqlite3.connect(db_file)
    return conn

def create_tables(conn):
    cursor=conn.cursor()
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS file_index(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    sha256 TEXT
    );
        
    CREATE TABLE IF NOT EXISTS text_lines(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER,
        page_no INTEGER,
        line_no INTEGER,
        text TEXT,
        length INTEGER
    );
        
    CREATE TABLE IF NOT EXISTS text_similarity(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        line_id_a INTEGER,
        line_id_b INTEGER,
        lev_ratio REAL,
        jaro REAL,
        dice REAL,
        avg_score REAL
        );
        """)
    conn.commit()
    print("Table created successfully")

if __name__=="__main__":
    conn=create_connection()
    create_tables(conn)
    conn.close()


    

