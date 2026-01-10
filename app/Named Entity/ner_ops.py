import re
import sqlite3
import unicodedata
from typing import Iterable, List, Tuple, Optional
from app.core.paths import DB_PATH

conn = sqlite3.connect(str(DB_PATH))

_space = re.compile(r"\s+")
_punct_edges = re.compile(r"^[\W_]+|[\W_]+$")

def normalize_name(s: str) -> str:
    _TITLES = {
        "dr", "dr.", "prof", "prof.", "doç", "doç.", "doc", "doc.",
        "asst", "asst.", "assist", "assist.", "av", "av."
    }
    """
    Kişi isimlerini tekilleştirmek için basit normalizasyon.
    text_similarity’deki mantığa benzer (NFKC + whitespace + casefold).
    """
    s = unicodedata.normalize("NFKC", s)
    s = _space.sub(" ", s.strip())
    s = s.casefold()
    # Ünvanları temizle (başta/arada)
    parts = [p for p in s.split(" ") if p not in _TITLES]
    s = " ".join(parts)

    # Türkçe i̇ problemi
    s = s.replace("i̇", "i")
    # baş/son noktalama temizle
    s = _punct_edges.sub("", s)
    return s

_BAD_PERSON_TAIL = {
    # TR kurum/kuruluş bitişleri
    "üniversitesi", "universitesi", "fakültesi", "fakultesi", "bölümü", "bolumu",
    "müdürlüğü", "mudurlugu", "bakanlığı", "bakanligi", "kurumu", "kuruluşu", "kurulusu",
    # şirket ekleri
    "a.ş", "a.s", "ltd", "limited", "inc", "corp", "co"
}

_BAD_PERSON_WORDS = {
    # satranç/kitap domaininde PERSON zannedilen yaygın kelimeler
    "chess", "department", "exercises", "beginner", "beginners", "champion",
    "game", "attack", "opening", "lines", "black", "white", "king", "queen", "rook",
    "bishop", "knight", "pawn", "mate"
}

def is_valid_person(ent_text: str) -> bool:
    t = ent_text.strip()
    if not t:
        return False

    # rakam içeriyorsa (Black 93, Chapter 2 vb.) ele
    if any(ch.isdigit() for ch in t):
        return False

    # tek kelime PERSON genelde hatalı (White, Deadly, Cornered vb.)
    parts = [p for p in t.split() if p]
    if len(parts) < 2:
        return False

    # çok kısa token ele
    if any(len(p) < 2 for p in parts):
        return False

    # tamamen büyük harfli kısaltmalar (PDF, SQL vb.)
    letters = [ch for ch in t if ch.isalpha()]
    if letters and all(ch.isupper() for ch in letters):
        return False

    # normalize üzerinden son kelime kontrolü
    norm = normalize_name(t)
    if not norm:
        return False
    norm_parts = norm.split()

    # ünvanları normalize_name zaten siliyor; yine de güvenlik
    if len(norm_parts) < 2:
        return False

    last = norm_parts[-1]
    if last in _BAD_PERSON_TAIL:
        return False

    # “tamamı domain kelimesi” ise ele (örn: "chess department")
    # (en az 2 kelime ama ikisi de bad list ise)
    if all(p in _BAD_PERSON_WORDS for p in norm_parts):
        return False

    return True

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
    pattern = re.compile(r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}){1,3})\b")
    return [m.group(1).strip() for m in pattern.finditer(text)]

def extract_entities(text: str, nlp=None) -> List[Tuple[str, str]]:
    LABEL_MAP = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "ORG": "ORG",
        "LOC": "LOC",
        "GPE": "GPE"
    }

    if nlp is None:
        persons = regex_fallback_persons(text)
        return [(p, "PERSON") for p in persons]

    doc = nlp(text)
    out = []
    for ent in doc.ents:
        lab = ent.label_
        lab = LABEL_MAP.get(lab, lab)
        out.append((ent.text, lab))

    return out

def run_ner(conn: sqlite3.Connection,
            model_name: str = "en_core_web_sm",
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
                if not is_valid_person(ent_text):
                    continue
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
