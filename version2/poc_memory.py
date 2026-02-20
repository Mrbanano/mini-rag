"""
POC — Memoria Semántica
Stack: fastembed + usearch + sqlite3 (stdlib)

pip install fastembed usearch
Sin compilación manual. Wheels nativos ARM64 / aarch64.
"""

import sqlite3
import numpy as np
import time
import psutil
import os
from fastembed import TextEmbedding
from usearch.index import Index

embedding_model = "nomic-ai/nomic-embed-text-v1.5-Q"
DIMS = 768  # era 384

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

DB_PATH    = "memory.db"
INDEX_PATH = "memory.usearch"
DIMS       = DIMS

header("INICIO")

with Step(f"cargar modelo ({embedding_model})"):
    model = TextEmbedding(embedding_model)

with Step("inicializar usearch index"):
    index = Index(ndim=DIMS, metric="cos", dtype="f32")
    if os.path.exists(INDEX_PATH):
        index.load(INDEX_PATH)

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
    db.execute("CREATE INDEX IF NOT EXISTS idx_type ON memory_meta(type)")
    db.commit()

# ── Helpers ────────────────────────────────────────────────────────────────────

def embed(text: str) -> np.ndarray:
    return np.array(list(model.embed([text]))[0], dtype=np.float32)

def save(content: str, type: str = "episodic", expires_days=30):
    label = f"save [{type}] '{content[:40]}'" + ("..." if len(content) > 40 else "")
    with Step(label):
        # Idempotente: no duplicar si ya existe el mismo contenido
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
        index.add(mid, embed(content))
        return mid

def recall(query: str, k: int = 3) -> list:
    matches = index.search(embed(query), k)
    if not len(matches):
        return []
    ids      = [int(m.key) for m in matches]
    dist_map = {int(m.key): round(float(m.distance), 4) for m in matches}
    ph       = ",".join("?" * len(ids))
    rows     = db.execute(
        f"SELECT id, content, type FROM memory_meta WHERE id IN ({ph})", ids
    ).fetchall()
    row_map = {r[0]: r for r in rows}
    return [
        {
            "content": row_map[i][1],
            "type":    row_map[i][2],
            "dist":    dist_map[i],
            "score":   round(1 - dist_map[i], 3),
        }
        for i in ids if i in row_map
    ]

def search(query: str):
    print(f"\n  🔍 '{query}'")
    with Step("recall k=3"):
        results = recall(query)
    for i, r in enumerate(results, 1):
        bar = "█" * int(r["score"] * 20)
        print(f"    {i}. [{r['type']:12s}] {r['content']}")
        print(f"       {bar:<20} score={r['score']}  dist={r['dist']}")

# ── Datos de prueba ────────────────────────────────────────────────────────────

header("GUARDANDO RECUERDOS")

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

with Step("guardar indice usearch a disco"):
    index.save(INDEX_PATH)

total = db.execute("SELECT COUNT(*) FROM memory_meta").fetchone()[0]
print(f"\n  {total} recuerdos guardados")

# ── Busquedas ─────────────────────────────────────────────────────────────────

header("BUSQUEDAS RELEVANTES")
search("boda familiar")
search("mascotas o animales")
search("stack tecnologico del desarrollador")
search("hardware donde corre el bot")
search("viaje o vacaciones recientes")
search("restricciones del agente de IA")

header("RUIDO (score deberia ser bajo)")
search("receta de tamales")
search("precio del dolar hoy")
search("como jugar ajedrez")

header("RESUMEN FINAL")
stats = db.execute("SELECT type, COUNT(*) FROM memory_meta GROUP BY type").fetchall()
for t, c in stats:
    print(f"  {t:15s}: {c} recuerdos")
size_kb = os.path.getsize(INDEX_PATH) / 1024 if os.path.exists(INDEX_PATH) else 0
print(f"\n  RAM final      : {ram_mb():.1f} MB")
print(f"  Index en disco : {size_kb:.1f} KB")
print(f"  DB en disco    : {os.path.getsize(DB_PATH) / 1024:.1f} KB")

# NOTA sobre borrado en usearch:
# usearch.remove() marca el vector pero no libera espacio real.
# En produccion, reconstruir el indice periodicamente con solo los IDs activos
# que existan en memory_meta (despues de borrar los expirados de sqlite).