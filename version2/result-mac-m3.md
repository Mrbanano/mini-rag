────────────────────────────────────────────────────────────
  INICIO  (RAM: 77.1 MB)
────────────────────────────────────────────────────────────
  ⏱   151.9 ms | CPU 124.3% | RAM  311.4 MB (+234.3 MB)  ← cargar modelo (bge-small-en-v1.5)
  ⏱     0.1 ms | CPU  99.2% | RAM  311.4 MB (+0.0 MB)  ← inicializar usearch index
  ⏱     3.3 ms | CPU  57.5% | RAM  312.3 MB (+0.9 MB)  ← inicializar sqlite3

────────────────────────────────────────────────────────────
  GUARDANDO RECUERDOS  (RAM: 312.3 MB)
────────────────────────────────────────────────────────────
  ⏱     7.9 ms | CPU 259.3% | RAM  313.8 MB (+1.5 MB)  ← save [episodic] 'La tia de Alvaro se casa en marzo en Pue'...
  ⏱     3.4 ms | CPU 608.1% | RAM  313.8 MB (+0.0 MB)  ← save [episodic] 'Alvaro menciono que su perro se llama Br'...
  ⏱     3.9 ms | CPU 565.2% | RAM  314.0 MB (+0.2 MB)  ← save [episodic] 'Hoy Alvaro tuvo una reunion con un clien'...
  ⏱     3.1 ms | CPU 267.6% | RAM  314.0 MB (+0.0 MB)  ← save [episodic] 'Alvaro fue a Oaxaca de vacaciones la sem'...
  ⏱     4.7 ms | CPU 738.2% | RAM  314.0 MB (+0.1 MB)  ← save [user_profile] 'Alvaro prefiere respuestas cortas y dire'...
  ⏱     3.2 ms | CPU 434.6% | RAM  314.0 MB (+0.0 MB)  ← save [user_profile] 'Alvaro trabaja con TypeScript y Next.js '...
  ⏱     3.9 ms | CPU 769.7% | RAM  314.0 MB (+0.0 MB)  ← save [user_profile] 'Alvaro es CEO de una empresa de software'...
  ⏱     3.8 ms | CPU 438.3% | RAM  314.0 MB (+0.0 MB)  ← save [fact] 'El proyecto Banabot corre en una Raspber'...
  ⏱     4.1 ms | CPU 839.3% | RAM  314.1 MB (+0.0 MB)  ← save [fact] 'La ventana de contexto del agente esta l'...
  ⏱     3.9 ms | CPU 399.2% | RAM  314.1 MB (+0.0 MB)  ← save [fact] 'Se decidio usar usearch para la memoria '...
  ⏱     0.6 ms | CPU 921.1% | RAM  314.1 MB (+0.0 MB)  ← guardar indice usearch a disco

  10 recuerdos guardados

────────────────────────────────────────────────────────────
  BUSQUEDAS RELEVANTES  (RAM: 314.1 MB)
────────────────────────────────────────────────────────────

  🔍 'boda familiar'
  ⏱     1.8 ms | CPU 454.3% | RAM  314.1 MB (+0.0 MB)  ← recall k=3
    1. [fact        ] Se decidio usar usearch para la memoria semantica
       ████████████         score=0.607  dist=0.3931
    2. [fact        ] La ventana de contexto del agente esta limitada a 8192 tokens
       ███████████          score=0.558  dist=0.4424
    3. [episodic    ] Alvaro menciono que su perro se llama Bruno
       ███████████          score=0.555  dist=0.4449

  🔍 'mascotas o animales'
  ⏱     2.2 ms | CPU 824.3% | RAM  314.1 MB (+0.0 MB)  ← recall k=3
    1. [fact        ] La ventana de contexto del agente esta limitada a 8192 tokens
       ████████████         score=0.603  dist=0.3965
    2. [fact        ] Se decidio usar usearch para la memoria semantica
       ████████████         score=0.6  dist=0.4
    3. [episodic    ] Alvaro menciono que su perro se llama Bruno
       ███████████          score=0.592  dist=0.4077

  🔍 'stack tecnologico del desarrollador'
  ⏱     4.1 ms | CPU 487.7% | RAM  314.1 MB (+0.0 MB)  ← recall k=3
    1. [user_profile] Alvaro trabaja con TypeScript y Next.js principalmente
       ██████████████       score=0.747  dist=0.2529
    2. [fact        ] Se decidio usar usearch para la memoria semantica
       ██████████████       score=0.746  dist=0.2538
    3. [fact        ] El proyecto Banabot corre en una Raspberry Pi 4
       ██████████████       score=0.731  dist=0.269

  🔍 'hardware donde corre el bot'
  ⏱     2.3 ms | CPU 168.0% | RAM  314.1 MB (+0.0 MB)  ← recall k=3
    1. [fact        ] El proyecto Banabot corre en una Raspberry Pi 4
       ███████████████      score=0.798  dist=0.2016
    2. [fact        ] Se decidio usar usearch para la memoria semantica
       ██████████████       score=0.701  dist=0.2993
    3. [fact        ] La ventana de contexto del agente esta limitada a 8192 tokens
       █████████████        score=0.688  dist=0.312

  🔍 'viaje o vacaciones recientes'
  ⏱     2.5 ms | CPU 1068.7% | RAM  314.2 MB (+0.0 MB)  ← recall k=3
    1. [fact        ] Se decidio usar usearch para la memoria semantica
       ██████████████       score=0.705  dist=0.2947
    2. [episodic    ] Alvaro fue a Oaxaca de vacaciones la semana pasada
       ██████████████       score=0.702  dist=0.2976
    3. [user_profile] Alvaro prefiere respuestas cortas y directas
       █████████████        score=0.686  dist=0.3139

  🔍 'restricciones del agente de IA'
  ⏱     3.9 ms | CPU 493.1% | RAM  314.2 MB (+0.0 MB)  ← recall k=3
    1. [fact        ] La ventana de contexto del agente esta limitada a 8192 tokens
       ██████████████       score=0.742  dist=0.2582
    2. [fact        ] Se decidio usar usearch para la memoria semantica
       ██████████████       score=0.707  dist=0.2928
    3. [user_profile] Alvaro es CEO de una empresa de software en Puebla
       █████████████        score=0.67  dist=0.3297

────────────────────────────────────────────────────────────
  RUIDO (score deberia ser bajo)  (RAM: 314.2 MB)
────────────────────────────────────────────────────────────

  🔍 'receta de tamales'
  ⏱     2.0 ms | CPU 532.2% | RAM  314.2 MB (+0.0 MB)  ← recall k=3
    1. [user_profile] Alvaro prefiere respuestas cortas y directas
       ████████████         score=0.64  dist=0.3598
    2. [fact        ] Se decidio usar usearch para la memoria semantica
       ████████████         score=0.634  dist=0.3663
    3. [episodic    ] La tia de Alvaro se casa en marzo en Puebla
       ████████████         score=0.61  dist=0.3897

  🔍 'precio del dolar hoy'
  ⏱     2.0 ms | CPU 370.4% | RAM  314.2 MB (+0.0 MB)  ← recall k=3
    1. [episodic    ] Alvaro menciono que su perro se llama Bruno
       ████████████         score=0.644  dist=0.3563
    2. [episodic    ] Hoy Alvaro tuvo una reunion con un cliente del gobierno
       ████████████         score=0.637  dist=0.3626
    3. [user_profile] Alvaro prefiere respuestas cortas y directas
       ████████████         score=0.633  dist=0.3672

  🔍 'como jugar ajedrez'
  ⏱     3.2 ms | CPU 739.7% | RAM  314.2 MB (+0.0 MB)  ← recall k=3
    1. [user_profile] Alvaro prefiere respuestas cortas y directas
       █████████████        score=0.674  dist=0.3255
    2. [fact        ] Se decidio usar usearch para la memoria semantica
       █████████████        score=0.674  dist=0.3263
    3. [episodic    ] Alvaro menciono que su perro se llama Bruno
       █████████████        score=0.669  dist=0.3314

────────────────────────────────────────────────────────────
  RESUMEN FINAL  (RAM: 314.2 MB)
────────────────────────────────────────────────────────────
  episodic       : 4 recuerdos
  fact           : 3 recuerdos
  user_profile   : 3 recuerdos

  RAM final      : 314.2 MB
  Index en disco : 16.5 KB
  DB en disco    : 16.0 KB