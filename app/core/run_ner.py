import sqlite3
from app.core.paths import DB_PATH
from app.core.ner_ops import run_ner

def main():
    conn = sqlite3.connect(str(DB_PATH))
    try:
        run_ner(conn, model_name="xx_ent_wiki_sm")  # model adı
    finally:
        conn.close()

if __name__ == "__main__":
    main()
