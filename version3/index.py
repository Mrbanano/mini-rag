"""
POC — BM25 + Query Expansion via OpenAI
Comparacion justa vs versiones anteriores (mismos datos, mismas queries)

pip install rank-bm25 openai psutil
"""

import sqlite3
import time
import psutil
import os
import json
from openai import OpenAI
from rank_bm25 import BM25Okapi

# ── Config ─────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-...")
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Monitor ────────────────────────────────────────────────────────────────────

proc = psutil.Process(os.getpid())

def ram_mb() -> float:
    return proc.memory_info().rss / 1024 / 1024

class Step:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        proc.cpu_percent(interval=None)
        self._ram_before = ram_mb()
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        ms  = (time.perf_counter() - self._t) * 1000
        ram = ram_mb()
        cpu = proc.cpu_percent(interval=None)
        d   = ram - self._ram_before
        print(
            f"  ⏱ {ms:7.1f} ms | CPU {cpu:5.1f}% | RAM {ram:6.1f} MB ({d:+.1f} MB)"
            f"  ← {self.label}"
        )

def header(t):
    print(f"\n{'─'*60}\n  {t}  (RAM: {ram_mb():.1f} MB)\n{'─'*60}")

# ── Setup ──────────────────────────────────────────────────────────────────────

DB_PATH = "memory_bm25.db"

header("INICIO — BM25 + Query Expansion (OpenAI)")

with Step("inicializar sqlite3"):
    db = sqlite3.connect(DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS memory_meta (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            content    TEXT    NOT NULL,
            type       TEXT    NOT NULL DEFAULT 'episodic',
            created_at TEXT    NOT NULL DEFAULT (datetime('now')),
            expires_at TEXT
        )
    """)
    db.commit()

# BM25 vive en RAM pero es tiny — solo listas de tokens
bm25_corpus = []   # lista de documentos tokenizados
bm25_ids    = []   # IDs correspondientes
bm25_model  = None # se reconstruye al agregar docs

# ── Helpers ────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Tokenizacion simple: lowercase + split. Sin deps extra."""
    return text.lower().split()

def rebuild_bm25():
    global bm25_model
    if bm25_corpus:
        bm25_model = BM25Okapi(bm25_corpus)

# Cargar index en RAM si la DB ya tenia datos
for row in db.execute("SELECT id, content FROM memory_meta").fetchall():
    bm25_ids.append(row[0])
    bm25_corpus.append(tokenize(row[1]))
rebuild_bm25()

def save(content: str, type: str = "episodic", expires_days=30):
    label = f"save [{type}] '{content[:40]}'" + ("..." if len(content) > 40 else "")
    with Step(label):
        existing = db.execute(
            "SELECT id FROM memory_meta WHERE content = ?", [content]
        ).fetchone()
        if existing:
            return existing[0]

        expires = f"datetime('now', '+{expires_days} days')" if expires_days else "NULL"
        cursor = db.execute(
            f"INSERT INTO memory_meta(content, type, expires_at) VALUES (?, ?, {expires})",
            [content, type],
        )
        mid = cursor.lastrowid
        db.commit()

        # Agregar al corpus BM25
        bm25_corpus.append(tokenize(content))
        bm25_ids.append(mid)
        rebuild_bm25()
        return mid

def expand_query(query: str) -> list[str]:
    """Llama al LLM para expandir la query con sinonimos y terminos relacionados."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"Dame 6 sinonimos o terminos relacionados con '{query}'. "
                    f"Responde SOLO con JSON: {{\"terms\": [\"term1\", \"term2\", ...]}}. "
                    f"Sin explicacion, sin markdown."
                )
            }],
            max_tokens=100,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        terms = data.get("terms", [])
        return [query] + terms
    except Exception as e:
        print(f"    [Warning] Expansion falló ({e}). Usando query original.")
        return [query]  # fallback: solo la query original

def recall_bm25(query: str, k: int = 3, use_expansion: bool = True) -> list[dict]:
    if not bm25_model:
        return []

    if use_expansion:
        terms = expand_query(query)
        # BM25 con todos los terminos expandidos concatenados
        tokens = tokenize(" ".join(terms))
    else:
        tokens = tokenize(query)

    scores = bm25_model.get_scores(tokens)

    # Top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        mid = bm25_ids[idx]
        row = db.execute(
            "SELECT content, type FROM memory_meta WHERE id = ?", [mid]
        ).fetchone()
        if row:
            results.append({
                "content": row[0],
                "type":    row[1],
                "score":   round(float(scores[idx]), 3),
            })
    return results

def search(query: str, use_expansion: bool = True):
    mode = "con expansion" if use_expansion else "sin expansion"
    print(f"\n  🔍 '{query}' ({mode})")
    with Step(f"recall k=3"):
        results = recall_bm25(query, use_expansion=use_expansion)
    if not results:
        print("    (sin resultados)")
        return
    max_score = max(r["score"] for r in results) or 1
    for i, r in enumerate(results, 1):
        bar = "█" * int((r["score"] / max_score) * 20)
        print(f"    {i}. [{r['type']:12s}] {r['content']}")
        print(f"       {bar:<20} score={r['score']}")

# ── Mismos datos que versiones anteriores ─────────────────────────────────────

header("GUARDANDO RECUERDOS (mismos datos que nomic-Q)")

save("La tia de Alvaro se casa en marzo en Puebla",             "episodic")
save("Alvaro menciono que su perro se llama Bruno",             "episodic")
save("Hoy Alvaro tuvo una reunion con un cliente del gobierno", "episodic")
save("Alvaro fue a Oaxaca de vacaciones la semana pasada",      "episodic")
save("Alvaro prefiere respuestas cortas y directas",            "user_profile", expires_days=None)
save("Alvaro trabaja con TypeScript y Next.js principalmente",  "user_profile", expires_days=None)
save("Alvaro es CEO de una empresa de software en Puebla",      "user_profile", expires_days=None)
save("El proyecto Banabot corre en una Raspberry Pi 4",         "fact", expires_days=180)
save("La ventana de contexto del agente esta limitada a 8192 tokens", "fact", expires_days=180)
save("Se decidio usar usearch para la memoria semantica",       "fact", expires_days=180)

save("Alvaro loves drinking black coffee in the morning",       "episodic")
save("The new server architecture will use Kubernetes",         "fact")
save("La couleur preferee d'Alvaro est le bleu",                "user_profile")
save("Nous devons acheter du pain et du lait demain",           "episodic")
save("Alvaro hat letztes Jahr ein neues Auto gekauft",          "episodic")
save("Das Buro befindet sich im Stadtzentrum",                  "fact")
save("Alvaro esta aprendendo a tocar violao",                   "episodic")
save("O Brasil e o maior pais da America do Sul",               "fact")

total = db.execute("SELECT COUNT(*) FROM memory_meta").fetchone()[0]
print(f"\n  {total} recuerdos guardados")

# ── Mismas queries que nomic-Q ─────────────────────────────────────────────────

header("BUSQUEDAS RELEVANTES — SIN expansion (BM25 puro)")
search("boda familiar",                           use_expansion=False)
search("mascotas o animales",                     use_expansion=False)
search("stack tecnologico del desarrollador",     use_expansion=False)
search("hardware donde corre el bot",             use_expansion=False)
search("viaje o vacaciones recientes",            use_expansion=False)
search("restricciones del agente de IA",          use_expansion=False)

header("BUSQUEDAS RELEVANTES — CON expansion LLM")
search("boda familiar")
search("mascotas o animales")
search("stack tecnologico del desarrollador")
search("hardware donde corre el bot")
search("viaje o vacaciones recientes")
search("restricciones del agente de IA")

header("MULTILENGUAJE — CON expansion")
search("family wedding")
search("animaux de compagnie")
search("urlaub")
search("onde o bot funciona?")

header("RUIDO — CON expansion (score deberia ser 0 o muy bajo)")
search("receta de tamales")
search("precio del dolar hoy")
search("como jugar ajedrez")

header("RESUMEN FINAL")
stats = db.execute("SELECT type, COUNT(*) FROM memory_meta GROUP BY type").fetchall()
for t, c in stats:
    print(f"  {t:15s}: {c} recuerdos")
print(f"\n  RAM final : {ram_mb():.1f} MB")
print(f"  DB disco  : {os.path.getsize(DB_PATH)/1024:.1f} KB")
print(f"\n  Comparativa RAM:")
print(f"    bge-small-en (ingles)     : ~250 MB  calidad español: mala")
print(f"    multilingual-MiniLM-L12   : ~670 MB  calidad: buena, no cabe")
print(f"    nomic-Q                   : ~312 MB  calidad español: mala")
print(f"    BM25 + expansion LLM      : ~{ram_mb():.0f} MB  calidad: ???")