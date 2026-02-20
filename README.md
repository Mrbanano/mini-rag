# POC Memory (sqlite-vec + fastembed)

Este proyecto es una Prueba de Concepto (POC) para memoria semántica en Python utilizando:
- `sqlite-vec` para búsqueda vectorial
- `fastembed` para generación de embeddings locales (modelo ligero `bge-small-en-v1.5`)
- Administrado mediante el gestor de paquetes y entornos `uv` (o `pip`)

## Instalación de `uv` (Ubuntu/Debian)
Si estás en Ubuntu o Debian y prefieres usar `apt` para instalar dependencias base, primero asegúrate de tener curl instalado para bajar el script de `uv`:

```bash
sudo apt update
sudo apt install curl -y
curl -LsSf https://astral.sh/uv/install.sh | sh
```

*(Nota: `uv` no está actualmente en los repositorios oficiales de `apt`, por lo que el método recomendado por sus creadores es mediante su script de instalación directo).*

## Instalación del Proyecto

### Opción 1: Instalación rápida con `uv` (Recomendada)
Si tienes [`uv`](https://docs.astral.sh/uv/) instalado, todas las dependencias ya están listadas en el archivo `pyproject.toml`. 

1. Clona el proyecto y entra en la carpeta:
```bash
git clone <url-del-repo>
cd poc-mini-rag
```

2. Instala el entorno de forma automática:
```bash
uv sync
```

### Opción 2: Instalación clásica con `pip`
Si prefieres no usar `uv` o quieres instalarlo en tu entorno virtual clásico, se incluye un archivo de dependencias listo para usarse.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Cómo ejecutar

Una vez instaladas las dependencias, corre el script principal. 

Si instalaste usando `uv`:
```bash
uv run python poc_memory.py
```

Si usaste el entorno virtual clásico (`pip`):
```bash
python poc_memory.py
```
