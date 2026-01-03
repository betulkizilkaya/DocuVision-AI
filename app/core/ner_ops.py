import re
import sqlite3
import unicodedata
from typing import Iterable, List, Tuple, Optional
from app.core.paths import DB_PATH
print("[NER] DB_PATH =", DB_PATH.resolve())

conn = sqlite3.connect(str(DB_PATH))
import sqlite3
from app.core.paths import DB_PATH

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

for t in ["file_index", "text_lines", "entities_raw", "persons", "person_mentions"]:
    cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", (t,))
    exists = cur.fetchone()[0]
    print(t, "exists?" , bool(exists))

cur.execute("SELECT COUNT(*) FROM text_lines")
print("text_lines count =", cur.fetchone()[0])

cur.execute("SELECT COUNT(*) FROM entities_raw")
print("entities_raw count =", cur.fetchone()[0])

cur.execute("SELECT COUNT(*) FROM persons")
print("persons count =", cur.fetchone()[0])

cur.execute("SELECT COUNT(*) FROM person_mentions")
print("person_mentions count =", cur.fetchone()[0])

conn.close()

_space = re.compile(r"\s+")
_punct_edges = re.compile(r"^[\W_]+|[\W_]+$")

def normalize_name(s: str) -> str:
    """
    Kişi isimlerini tekilleştirmek için basit normalizasyon.
    text_similarity’deki mantığa benzer (NFKC + whitespace + casefold).
    """
    s = unicodedata.normalize("NFKC", s)
    s = _space.sub(" ", s.strip())
    s = s.casefold()
    # Türkçe i̇ problemi
    s = s.replace("i̇", "i")
    # baş/son noktalama temizle
    s = _punct_edges.sub("", s)
    return s

def ensure_ner_indexes(conn: sqlite3.Connection) -> None:
    """
    Aynı satır-entity tekrar yazılmasın diye indeksler.
    (create_tables içinde yoksa bile burada garanti ediyoruz.)
    """
    cur = conn.cursor()
    cur.executescript("""
    CREATE UNIQUE INDEX IF NOT EXISTS ux_entities_raw
      ON entities_raw(line_id, ent_text, ent_type);

    CREATE UNIQUE INDEX IF NOT EXISTS ux_person_mentions
      ON person_mentions(person_id, line_id, raw_name);
    """)
    conn.commit()

def get_lines(conn: sqlite3.Connection, file_id: Optional[int] = None,
             min_len: int = 20, max_len: int = 300) -> List[Tuple[int, int, str]]:
    """
    text_lines’dan satırları çeker.
    return: [(line_id, file_id, text), ...]
    """
    cur = conn.cursor()
    if file_id is None:
        cur.execute("""
            SELECT id, file_id, text
            FROM text_lines
            WHERE length BETWEEN ? AND ?
        """, (min_len, max_len))
    else:
        cur.execute("""
            SELECT id, file_id, text
            FROM text_lines
            WHERE file_id = ?
              AND length BETWEEN ? AND ?
        """, (file_id, min_len, max_len))
    return cur.fetchall()

def insert_entity_raw(conn: sqlite3.Connection, line_id: int, ent_text: str, ent_type: str) -> None:
    cur = conn.cursor()
    norm = normalize_name(ent_text)
    cur.execute("""
        INSERT OR IGNORE INTO entities_raw(line_id, ent_text, ent_type, norm_text)
        VALUES (?, ?, ?, ?)
    """, (line_id, ent_text, ent_type, norm))

def upsert_person(conn: sqlite3.Connection, norm_name: str) -> int:
    """
    persons tablosunda norm_name yoksa ekler, varsa var olan id’yi döner.
    """
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO persons(norm_name) VALUES (?)", (norm_name,))
    cur.execute("SELECT id FROM persons WHERE norm_name = ?", (norm_name,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError("persons upsert başarısız oldu.")
    return int(row[0])

def insert_person_mention(conn: sqlite3.Connection, person_id: int, line_id: int, raw_name: str) -> None:
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO person_mentions(person_id, line_id, raw_name)
        VALUES (?, ?, ?)
    """, (person_id, line_id, raw_name))

# -----------------------------
# NER ENGINE
# -----------------------------

def load_spacy_model(model_name: str):
    """
    spaCy modelini yükler. Model yoksa None döndürür.
    """
    try:
        import spacy
        return spacy.load(model_name)
    except Exception:
        return None

def regex_fallback_persons(text: str) -> List[str]:
    """
    Model yoksa minimum işlev için kaba bir PERSON çıkarımı:
    Yan yana gelen büyük harfli kelimeleri ad-soyad gibi yakalar.
    (Türkçede %100 değildir, ama sistemin çökmesini engeller.)
    """
    # Örn: "Magnus Carlsen", "Garry Kasparov"
    # Türkçe için: "Ahmet Yılmaz"
    pattern = re.compile(r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+){1,3})\b")
    return [m.group(1).strip() for m in pattern.finditer(text)]

def extract_entities(text: str, nlp=None) -> List[Tuple[str, str]]:
    """
    return: [(ent_text, ent_type), ...]
    """
    if nlp is None:
        persons = regex_fallback_persons(text)
        return [(p, "PERSON") for p in persons]

    doc = nlp(text)
    out = []
    for ent in doc.ents:
        # spaCy etiketleri: PERSON, ORG, GPE, LOC, DATE...
        out.append((ent.text, ent.label_))
    return out

def run_ner(conn: sqlite3.Connection,
            model_name: str = "xx_ent_wiki_sm",
            file_id: Optional[int] = None,
            commit_every: int = 500) -> None:
    """
    text_lines -> entities_raw (+ person tablolarına bağlama)
    """
    ensure_ner_indexes(conn)

    nlp = load_spacy_model(model_name)
    if nlp is None:
        print(f"[NER] Uyarı: spaCy model yüklenemedi ({model_name}). Regex fallback kullanılacak.")
    else:
        print(f"[NER] spaCy model yüklendi: {model_name}")

    rows = get_lines(conn, file_id=file_id)
    print(f"[NER] İşlenecek satır: {len(rows)}")

    cur = conn.cursor()
    processed = 0
    ent_count = 0
    person_mentions = 0

    for line_id, _fid, text in rows:
        ents = extract_entities(text, nlp=nlp)

        for ent_text, ent_type in ents:
            if not ent_text.strip():
                continue

            insert_entity_raw(conn, line_id, ent_text, ent_type)
            ent_count += 1

            if ent_type == "PERSON":
                norm = normalize_name(ent_text)
                if not norm:
                    continue
                pid = upsert_person(conn, norm)
                insert_person_mention(conn, pid, line_id, ent_text)
                person_mentions += 1

        processed += 1
        if processed % commit_every == 0:
            conn.commit()
            print(f"[NER] {processed} satır işlendi | entity≈{ent_count} | person_mention≈{person_mentions}")

    conn.commit()
    print(f"[NER] Bitti. Satır={processed} | entity≈{ent_count} | person_mention≈{person_mentions}")
