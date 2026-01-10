import sqlite3
from app.core.paths import DB_PATH
from app.core.ner_ops import run_ner
import spacy
nlp = spacy.load("xx_ent_wiki_sm")
doc = nlp("Magnus Carlsen dünya şampiyonudur. Garry Kasparov efsanedir.")
print([(e.text, e.label_) for e in doc.ents])


def main():
    conn = sqlite3.connect(str(DB_PATH))
    try:
        run_ner(conn, model_name="xx_ent_wiki_sm")  # model adı
    finally:
        conn.close()

if __name__ == "__main__":
    main()
