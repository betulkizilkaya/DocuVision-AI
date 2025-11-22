import sqlite3
from pathlib import Path
import unicodedata
import re

from rapidfuzz.fuzz import ratio as lev_ratio
import textdistance as td
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Proje yolları
APP_DIR  = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DB_PATH  = ROOT_DIR / "db" / "corpus.sqlite"

MIN_LEN = 20
MAX_LEN = 300 # Satır filtresi

# Bu ikisini gevşetirsek daha çok eşleşme buluruz ama yavaşlar.
LENGTH_TOL = 10 #Uzunluk toleransı (|len(a)-len(b)| ≤ 10) Metinlerin uzunluğu farklıysa ama anlam aynıysa (ör. küçük ekler) toleransı artırabilirsin
PREFIX_LEN = 2 #Aynı ilk N harf (bloklama). Düşürürsek hız düşer, recall artar; yükseltirsek hız artar, recall düşer.

THRESH = 0.90 # %90 eşik
COMPARE_WITHIN_FILE = True # aynı dosya içi karşılaştırma
COMPARE_ACROSS_FILES = True # farklı dosyalar arası karşılaştırma


_norm_space = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s) #Saçma kodlamaları düzeltir
    s = _norm_space.sub(" ", s.strip()) #boşlukları tek boşluğa indirir
    s = s.casefold()            # Türkçe için en doğru lowercase
    s = s.replace("i̇", "i")     # Türkçe I/İ hatası düzeltme
    return s

def ensure_schema(conn: sqlite3.Connection):#kontrol amaçlı. şemayı garantilemek için
    cur = conn.cursor()
    # text_similarity yoksa oluştur
    cur.execute("PRAGMA table_info(text_similarity)")
    cols = {r[1] for r in cur.fetchall()}
    if "text_similarity" not in {t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}:
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS text_similarity(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          line_id_a INTEGER NOT NULL,
          line_id_b INTEGER NOT NULL,
          lev_ratio REAL,
          jaro REAL,
          dice REAL,
          tfidf_cosine REAL,
          avg_score REAL,
          passed_threshold INTEGER
        );
        """)
    else:
        # Eksik kolonları ekle
        to_add = []
        if "tfidf_cosine" not in cols:    to_add.append("ALTER TABLE text_similarity ADD COLUMN tfidf_cosine REAL;")
        if "passed_threshold" not in cols:to_add.append("ALTER TABLE text_similarity ADD COLUMN passed_threshold INTEGER;")
        for stmt in to_add:
            cur.execute(stmt)

    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_txtsim ON text_similarity(line_id_a, line_id_b)")# UNIQUE(line_id_a, line_id_b) → aynı çifti iki kere yazmaz.line_id_a < line_id_b kuralıyla yönsüz tekil çift saklanır.
    cur.execute("CREATE INDEX IF NOT EXISTS ix_txtsim_avg ON text_similarity(avg_score)")
    conn.commit()

def load_lines(conn: sqlite3.Connection):#Veritabanından çektik
    cur = conn.cursor()
    cur.execute("""
        SELECT id, file_id, length, text
        FROM text_lines
        WHERE length BETWEEN ? AND ?
    """, (MIN_LEN, MAX_LEN))
    rows = cur.fetchall()
    # Veritabanından çekip normalize ediyoruz; (line_id, file_id, len, text_norm) listesi hazır.
    items = []
    for (lid, fid, ln, txt) in rows:
        norm = normalize_text(txt)
        if norm:
            items.append((lid, fid, len(norm), norm))
    return items

def build_buckets(items):
    # Milyonlarca alakasız karşılaştırmayı erken elemek için
    buckets = {}
    for lid, fid, ln, txt in items:
        prefix = txt[:PREFIX_LEN] if len(txt) >= PREFIX_LEN else txt
        key = (ln // 10, prefix) #10 karakterlik uzunluk dilimleri + ilk 2 harf aynı olanlar bir arada.
        #Böylece yüzbinlerce “alakasız” karşılaştırma yapılmaz.
        buckets.setdefault(key, []).append((lid, fid, ln, txt))
    return buckets

def compute_tfidf(texts):
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b\w+\b"  # Türkçe karakterleri destekler
    )

    X = vec.fit_transform(texts)  # sparse
    return vec, X

def cosine_from_sparse_row(X, i, j):#iki satırın anlamsal yakınlığı ölçülür.
    # Seyrek matris üzerinde hızlı nokta çarpımı
    v = X[i].multiply(X[j]).sum()
    ni = np.sqrt(X[i].multiply(X[i]).sum())
    nj = np.sqrt(X[j].multiply(X[j]).sum())
    if ni == 0 or nj == 0:
        return 0.0
    return float(v / (ni * nj))

def generate_candidates(bucket):
    # küçük optimizasyon: uzunluk toleransı + aynı prefix
    N = len(bucket)
    for idx_a in range(N):
        lid_a, fid_a, len_a, txt_a = bucket[idx_a]
        for idx_b in range(idx_a + 1, N):
            lid_b, fid_b, len_b, txt_b = bucket[idx_b]
            if abs(len_a - len_b) > LENGTH_TOL:
                continue
            # aynı dosya içinde? / dosyalar arası?
            if fid_a == fid_b and not COMPARE_WITHIN_FILE:
                continue
            if fid_a != fid_b and not COMPARE_ACROSS_FILES:
                continue
            yield (idx_a, idx_b)

def main():
    conn = sqlite3.connect(str(DB_PATH))
    ensure_schema(conn)

    print("→ Satırları yükleniyor…")
    items = load_lines(conn)   # (line_id, file_id, length, text_norm)
    if not items:
        print("❗ text_lines boş görünüyor.")
        return

    # id → sıra index
    id2idx = {lid: i for i, (lid, _, _, _) in enumerate(items)}
    texts = [t[3] for t in items]

    print("→ TF-IDF vektörleri hesaplanıyor…")
    _, X = compute_tfidf(texts)

    print("→ Bucket'lar oluşturuluyor…")
    buckets = build_buckets(items)

    cur = conn.cursor()
    inserted = 0
    checked_pairs = 0

    for key, bucket in buckets.items():
        if len(bucket) < 2:
            continue
        # bucket içindeki global indeksleri çıkart
        global_idx = [id2idx[lid] for (lid, _, _, _) in bucket]

        for ia, ib in generate_candidates(bucket):
            checked_pairs += 1

            lid_a, fid_a, len_a, txt_a = bucket[ia]
            lid_b, fid_b, len_b, txt_b = bucket[ib]

            g_a = global_idx[ia]
            g_b = global_idx[ib]

            lev = lev_ratio(txt_a, txt_b) / 100.0            #Levenshtein: edit mesafesi tabanlı oran (yazım farklarına duyarlı).
            jaro = td.jaro_winkler(txt_a, txt_b)             #Jaro-Winkler: yazım hatalarına esnek, kısa stringlerde iyi.
            dice = td.dice.distance(txt_a, txt_b)            #Dice: karakter/kelime çifti benzerliği (yapısal).
            dice = 1.0 - dice
            tfidf = cosine_from_sparse_row(X, g_a, g_b)      #TF-IDF cosine: anlam temelli (kelime önemine bakar).

            avg = (lev + jaro + dice + tfidf) / 4.0          #avg_score: hepsinin ortalaması → tek güvenilir skor.
            passed = 1 if avg >= THRESH else 0

            if passed: #Yalnızca %90+ olanlar yazılır.
                a, b = (lid_a, lid_b) if lid_a < lid_b else (lid_b, lid_a)
                try:
                    cur.execute("""
                        INSERT INTO text_similarity
                        (line_id_a, line_id_b, lev_ratio, jaro, dice, tfidf_cosine, avg_score, passed_threshold)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (a, b, lev, jaro, dice, tfidf, avg, passed))
                    inserted += 1
                except sqlite3.IntegrityError:
                    #Tekillik ihlalinde IntegrityError yakalanır ve tekrar yazılmaz.
                    pass

        conn.commit()

    print(f"✓ Tamamlandı. Kontrol edilen aday çift: {checked_pairs:,} | Kaydedilen ≥{int(THRESH*100)}%: {inserted:,}")
    conn.close()

if __name__ == "__main__":
    main()
