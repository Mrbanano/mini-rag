# POC Memory (sqlite-vec + fastembed o usearch + sqlite3)

Este proyecto es una Prueba de Concepto (POC) para la memoria semántica en Python administrado mediante `uv` (o `pip`).

## Estructura
- `version1/`: Utiliza `sqlite-vec`. Contiene la prueba de concepto original.
- `version2/`: Utiliza `usearch` + `sqlite3` (stdlib). Orientada a zero compilación y compatibilidad ARM64 nativa. Esta versión es ideal porque `sqlite3` viene con Python nativo y `usearch` cuenta con wheels precompilados.

## Instalación de `uv` (Ubuntu/Debian)
Si estás en Ubuntu o Debian y prefieres usar `apt` para instalar dependencias base, asegúrate de tener curl instalado para bajar el script de `uv`:

```bash
sudo apt update
sudo apt install curl -y
curl -LsSf https://astral.sh/uv/install.sh | sh
```

*(Nota: `uv` no está en los repositorios oficiales de `apt`, el método recomendado es mediante su script).*

## Instalación del Proyecto

### Opción 1: Instalación rápida con `uv` (Recomendada)
Si tienes [`uv`](https://docs.astral.sh/uv/) instalado, todas las dependencias (`fastembed`, `usearch`, etc.) ya están listadas en el archivo `pyproject.toml`. 

1. Clona el proyecto y entra:
```bash
git clone <url-del-repo>
cd poc-mini-rag
```

2. Instala el entorno automáticamente:
```bash
uv sync
```

### Opción 2: Instalación clásica con `pip`
Para instalar en tu entorno virtual clásico:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Nota: También puedes agregar las dependencias de la v2 de forma manual mediante `pip install fastembed usearch` o `uv add fastembed usearch`)*.

## Cómo ejecutar

### Versión 2: `usearch` + `sqlite3` stdlib (Nueva recomendación)
Esta versión escribirá dos archivos en tu directorio: `memory.db` (sqlite con el texto temporal) y `memory.usearch` (índice vectorial). Si lo corres de nuevo, los reutiliza. 

```bash
uv run python version2/poc_memory.py
```
*O con pip puro:* `python version2/poc_memory.py`

**¿Por qué este stack no da guerra?**
- **sqlite3**: stdlib de Python, zero install.
- **usearch**: Mantenido con wheels precompilados para aarch64, x86_64, armv7l. No necesita compilación.
- **fastembed**: Descarga el modelo y corre eficientemente en dispositivos como la Raspberry Pi 4 / DietPi.

> **Importante**: `usearch` no borra vectores de forma física inmediatamente. Solo los marca como borrados (`usearch.remove()`). Para uso en producción se aconseja reconstruir este índice periódicamente leyendo los IDs activos de la tabla `memory_meta` cuando existan registros expirados.

---

### Versión 1: `sqlite-vec` (Anterior)
```bash
uv run python version1/poc_memory.py
```
