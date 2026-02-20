"""
POC — Memoria Semántica con sqlite-vec + fastembed
pip install sqlite-vec fastembed psutil
"""

import sqlite3
import sqlite_vec
import numpy as np
import time
import psutil
import os
from fastembed import TextEmbedding

store = "poc.db"

# ── Monitor ────────────────────────────────────────────────────────────────────

proc = psutil.Process(os.getpid())

def ram_mb() -> float:
    return proc.memory_info().rss / 1024 / 1024

class Step:
    """Context manager que mide tiempo, RAM y CPU de cada operación."""
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        proc.cpu_percent(interval=None)  # reset contador CPU
        self._ram_before = ram_mb()
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed_ms = (time.perf_counter() - self._t) * 1000
        ram_now    = ram_mb()
        cpu        = proc.cpu_percent(interval=None)
        delta      = ram_now - self._ram_before
        sign       = "+" if delta >= 0 else ""
        print(
            f"  ⏱  {elapsed_ms:7.1f} ms  "
            f"| CPU {cpu:5.1f}%  "
            f"| RAM {ram_now:6.1f} MB ({sign}{delta:.1f} MB)"
            f"  ← {self.label}"
        )

def header(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}  (RAM base: {ram_mb():.1f} MB)")
    print(f"{'─'*60}")

# ── Setup ──────────────────────────────────────────────────────────────────────

header("INICIO")

with Step("cargar modelo ONNX (bge-small-en-v1.5)"):
    model = TextEmbedding("BAAI/bge-small-en-v1.5")

with Step("inicializar sqlite + sqlite-vec"):
    db = sqlite3.connect(store)  # cambiar a "poc.db" para persistir
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.executescript("""
        CREATE VIRTUAL TABLE memories USING vec0(
            id INTEGER PRIMARY KEY,
            embedding FLOAT[384]
        );
        CREATE TABLE memory_meta (
            id      INTEGER PRIMARY KEY,
            content TEXT,
            type    TEXT
        );
    """)

# ── Helpers ────────────────────────────────────────────────────────────────────

def embed(text: str) -> bytes:
    vec = list(model.embed([text]))[0]
    return np.array(vec, dtype=np.float32).tobytes()

def save(content: str, type: str = "episodic"):
    label = f"save [{type}] '{content[:45]}...'" if len(content) > 45 else f"save [{type}] '{content}'"
    with Step(label):
        cursor = db.execute(
            "INSERT INTO memory_meta(content, type) VALUES (?, ?)", [content, type]
        )
        mid = cursor.lastrowid
        db.execute("INSERT INTO memories(id, embedding) VALUES (?, ?)", [mid, embed(content)])
        db.commit()

def recall(query: str, k: int = 3) -> list[dict]:
    rows = db.execute(
        """
        SELECT m.content, m.type, v.distance
        FROM memories v
        JOIN memory_meta m ON v.id = m.id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance ASC
        """,
        [embed(query), k],
    ).fetchall()
    return [{"content": r[0], "type": r[1], "dist": round(r[2], 4)} for r in rows]

def search(query: str):
    print(f"\n  🔍 '{query}'")
    with Step("recall k=3"):
        results = recall(query)
    for i, r in enumerate(results, 1):
        score = 1 / (1 + r["dist"])
        bar   = "█" * int(score * 20)
        print(f"    {i}. [{r['type']:12s}] {r['content']}")
        print(f"       {bar:<20} score={score:.3f}  dist={r['dist']}")

# ── Datos de prueba ────────────────────────────────────────────────────────────

header("GUARDANDO RECUERDOS")

save("La tía de Alvaro se casa en marzo en Puebla",            "episodic")
save("Alvaro mencionó que su perro se llama Bruno",            "episodic")
save("Hoy Alvaro tuvo una reunión con un cliente del gobierno", "episodic")
save("Alvaro fue a Oaxaca de vacaciones la semana pasada",     "episodic")
save("Alvaro prefiere respuestas cortas y directas",           "user_profile")
save("Alvaro trabaja con TypeScript y Next.js principalmente", "user_profile")
save("Alvaro es CEO de una empresa de software en Puebla",     "user_profile")
save("El proyecto Banabot corre en una Raspberry Pi 4",        "fact")
save("La ventana de contexto del agente está limitada a 8192 tokens", "fact")
save("Se decidió usar sqlite-vec para la memoria semántica",   "fact")

print(f"\n  → {db.execute('SELECT COUNT(*) FROM memory_meta').fetchone()[0]} recuerdos en DB")

# ── Búsquedas ─────────────────────────────────────────────────────────────────

header("BÚSQUEDAS RELEVANTES")
search("boda familiar")
search("mascotas o animales")
search("stack tecnológico del desarrollador")
search("hardware donde corre el bot")
search("viaje o vacaciones recientes")
search("restricciones del agente de IA")

header("BÚSQUEDAS SIN RELACIÓN (ruido)")
search("receta de tamales")
search("precio del dólar hoy")
search("cómo jugar ajedrez")

header("RESUMEN FINAL")
print(f"  RAM final: {ram_mb():.1f} MB")