(.venv) alvarocastillo@MacBook-Pro poc-mini-rag % uv run python poc_memory.py
Using CPython 3.12.9
Removed virtual environment at: .venv
Creating virtual environment at: .venv
Installed 40 packages in 132ms

────────────────────────────────────────────────────────────
  INICIO  (RAM base: 93.8 MB)
────────────────────────────────────────────────────────────
  ⏱    108.0 ms  | CPU 181.5%  | RAM  324.0 MB (+230.2 MB)  ← cargar modelo ONNX (bge-small-en-v1.5)
  ⏱    187.3 ms  | CPU   1.7%  | RAM  325.1 MB (+1.1 MB)  ← inicializar sqlite + sqlite-vec

────────────────────────────────────────────────────────────
  GUARDANDO RECUERDOS  (RAM base: 325.1 MB)
────────────────────────────────────────────────────────────
  ⏱      8.1 ms  | CPU 510.4%  | RAM  328.3 MB (+3.2 MB)  ← save [episodic] 'La tía de Alvaro se casa en marzo en Puebla'
  ⏱      4.1 ms  | CPU 579.6%  | RAM  328.3 MB (+0.0 MB)  ← save [episodic] 'Alvaro mencionó que su perro se llama Bruno'
  ⏱      5.2 ms  | CPU 529.3%  | RAM  328.4 MB (+0.1 MB)  ← save [episodic] 'Hoy Alvaro tuvo una reunión con un cliente de...'
  ⏱      3.8 ms  | CPU 597.8%  | RAM  328.6 MB (+0.2 MB)  ← save [episodic] 'Alvaro fue a Oaxaca de vacaciones la semana p...'
  ⏱      4.0 ms  | CPU 588.6%  | RAM  328.7 MB (+0.1 MB)  ← save [user_profile] 'Alvaro prefiere respuestas cortas y directas'
  ⏱      3.4 ms  | CPU 456.9%  | RAM  328.7 MB (+0.0 MB)  ← save [user_profile] 'Alvaro trabaja con TypeScript y Next.js princ...'
  ⏱      3.3 ms  | CPU 815.2%  | RAM  328.7 MB (+0.0 MB)  ← save [user_profile] 'Alvaro es CEO de una empresa de software en P...'
  ⏱      4.7 ms  | CPU 536.2%  | RAM  328.7 MB (+0.0 MB)  ← save [fact] 'El proyecto Banabot corre en una Raspberry Pi...'
  ⏱      3.9 ms  | CPU 591.0%  | RAM  328.7 MB (+0.0 MB)  ← save [fact] 'La ventana de contexto del agente está limita...'
  ⏱      5.0 ms  | CPU 623.3%  | RAM  328.7 MB (+0.0 MB)  ← save [fact] 'Se decidió usar sqlite-vec para la memoria se...'

  → 10 recuerdos en DB

────────────────────────────────────────────────────────────
  BÚSQUEDAS RELEVANTES  (RAM base: 328.7 MB)
────────────────────────────────────────────────────────────

  🔍 'boda familiar'
  ⏱      2.4 ms  | CPU 433.7%  | RAM  328.8 MB (+0.1 MB)  ← recall k=3
    1. [fact        ] Se decidió usar sqlite-vec para la memoria semántica
       ██████████           score=0.524  dist=0.9089
    2. [fact        ] La ventana de contexto del agente está limitada a 8192 tokens
       ██████████           score=0.515  dist=0.9407
    3. [episodic    ] Alvaro mencionó que su perro se llama Bruno
       ██████████           score=0.515  dist=0.9433

  🔍 'mascotas o animales'
  ⏱      2.4 ms  | CPU 775.8%  | RAM  328.8 MB (+0.0 MB)  ← recall k=3
    1. [fact        ] La ventana de contexto del agente está limitada a 8192 tokens
       ██████████           score=0.529  dist=0.8905
    2. [episodic    ] Alvaro mencionó que su perro se llama Bruno
       ██████████           score=0.525  dist=0.903
    3. [episodic    ] Alvaro fue a Oaxaca de vacaciones la semana pasada
       ██████████           score=0.525  dist=0.9031

  🔍 'stack tecnológico del desarrollador'
  ⏱      4.0 ms  | CPU 377.2%  | RAM  328.8 MB (+0.0 MB)  ← recall k=3
    1. [fact        ] Se decidió usar sqlite-vec para la memoria semántica
       ███████████          score=0.587  dist=0.7038
    2. [user_profile] Alvaro trabaja con TypeScript y Next.js principalmente
       ███████████          score=0.584  dist=0.7112
    3. [fact        ] El proyecto Banabot corre en una Raspberry Pi 4
       ███████████          score=0.577  dist=0.7334

  🔍 'hardware donde corre el bot'
  ⏱      2.8 ms  | CPU 646.3%  | RAM  328.8 MB (+0.0 MB)  ← recall k=3
    1. [fact        ] El proyecto Banabot corre en una Raspberry Pi 4
       ████████████         score=0.612  dist=0.635
    2. [fact        ] La ventana de contexto del agente está limitada a 8192 tokens
       ███████████          score=0.559  dist=0.7899
    3. [user_profile] Alvaro trabaja con TypeScript y Next.js principalmente
       ███████████          score=0.557  dist=0.795

  🔍 'viaje o vacaciones recientes'
  ⏱      4.9 ms  | CPU 582.8%  | RAM  328.8 MB (+0.0 MB)  ← recall k=3
    1. [episodic    ] Alvaro fue a Oaxaca de vacaciones la semana pasada
       ███████████          score=0.564  dist=0.7715
    2. [user_profile] Alvaro prefiere respuestas cortas y directas
       ███████████          score=0.558  dist=0.7923
    3. [episodic    ] Alvaro mencionó que su perro se llama Bruno
       ███████████          score=0.554  dist=0.8062

  🔍 'restricciones del agente de IA'
  ⏱      3.4 ms  | CPU 307.5%  | RAM  328.8 MB (+0.0 MB)  ← recall k=3
    1. [fact        ] La ventana de contexto del agente está limitada a 8192 tokens
       ███████████          score=0.582  dist=0.7186
    2. [fact        ] Se decidió usar sqlite-vec para la memoria semántica
       ███████████          score=0.560  dist=0.7847
    3. [user_profile] Alvaro es CEO de una empresa de software en Puebla
       ███████████          score=0.552  dist=0.812

────────────────────────────────────────────────────────────
  BÚSQUEDAS SIN RELACIÓN (ruido)  (RAM base: 328.8 MB)
────────────────────────────────────────────────────────────

  🔍 'receta de tamales'
  ⏱      2.3 ms  | CPU 838.0%  | RAM  328.8 MB (+0.0 MB)  ← recall k=3
    1. [user_profile] Alvaro prefiere respuestas cortas y directas
       ██████████           score=0.541  dist=0.8483
    2. [fact        ] Se decidió usar sqlite-vec para la memoria semántica
       ██████████           score=0.537  dist=0.8618
    3. [episodic    ] La tía de Alvaro se casa en marzo en Puebla
       ██████████           score=0.531  dist=0.8828

  🔍 'precio del dólar hoy'
  ⏱      3.2 ms  | CPU 626.7%  | RAM  328.8 MB (+0.0 MB)  ← recall k=3
    1. [episodic    ] Alvaro mencionó que su perro se llama Bruno
       ██████████           score=0.542  dist=0.8442
    2. [episodic    ] Hoy Alvaro tuvo una reunión con un cliente del gobierno
       ██████████           score=0.540  dist=0.8516
    3. [user_profile] Alvaro prefiere respuestas cortas y directas
       ██████████           score=0.539  dist=0.857

  🔍 'cómo jugar ajedrez'
  ⏱      2.9 ms  | CPU 396.6%  | RAM  328.8 MB (+0.0 MB)  ← recall k=3
    1. [user_profile] Alvaro prefiere respuestas cortas y directas
       ███████████          score=0.553  dist=0.8068
    2. [episodic    ] Alvaro mencionó que su perro se llama Bruno
       ███████████          score=0.551  dist=0.8142
    3. [fact        ] La ventana de contexto del agente está limitada a 8192 tokens
       ██████████           score=0.545  dist=0.8365

────────────────────────────────────────────────────────────
  RESUMEN FINAL  (RAM base: 328.8 MB)
────────────────────────────────────────────────────────────
  RAM final: 328.8 MB
(.venv) alvarocastillo@MacBook-Pro poc-mini-rag % 