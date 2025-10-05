# ¿Qué Estamos Probando y Cómo?

Esta es una guía de referencia sobre los modelos y las tareas incluidas en el benchmark.

---

## Las 9 Arquitecturas Bajo Examen

Hemos implementado una selección completa que abarca tres familias principales de arquitecturas para responder a tu pregunta de investigación de manera rigurosa.

### Grupo 1: RNNs Clásicos (La Línea de Base)
Son los modelos secuenciales tradicionales. Esperamos que sufran en tareas de memoria a largo plazo.
1.  **`LSTM`**: El modelo de memoria por excelencia de la "vieja escuela".
2.  **`GRU`**: Una versión más simple y rápida del LSTM.

### Grupo 2: Transformers (El Campeón a Vencer)
Son los modelos que dominan el NLP actual, basados en el mecanismo de atención.
3.  **`Vanilla Transformer`**: El modelo original, para tener una referencia pura.
4.  **`RoPE Transformer`**: Una variante moderna que usa "Rotary Position Embeddings" para entender mejor las posiciones, muy común en modelos como Llama.

### Grupo 3: RNNs Lineales Modernos (Los Nuevos Contendientes)
Esta es la parte más interesante de la investigación. Son arquitecturas recientes que intentan ser tan potentes como los Transformers, pero mucho más eficientes (complejidad lineal en lugar de cuadrática).
5.  **`DeltaNet`**: Un RNN lineal de alto rendimiento basado en la "delta rule".
6.  **`Gated DeltaNet`**: Una versión mejorada con "compuertas" (gating), inspirada en Mamba.
7.  **`RWKV-7`**: Un híbrido que combina las fortalezas de los RNNs (eficiencia) y los Transformers (potencia).
8.  **`Retention (RetNet)`**: Propone un nuevo mecanismo llamado "retención" como alternativa a la "atención" de los Transformers.
9.  **`GLA (Gated Linear Attention)`**: Usa una forma de atención lineal con compuertas para ser más expresivo.

---

## El Método de Prueba: 3 Desafíos Clave

No usamos texto normal. Usamos 3 "tareas de diagnóstico" sintéticas, diseñadas para poner a prueba habilidades muy específicas de los modelos.

### Desafío 1: `MQAR` (Memoria Asociativa)
-   **La Prueba:** El modelo ve una lista de pares clave-valor (ej: "la clave 5 es 'manzana', la clave 8 es 'pera'"). Luego, se le pregunta por una de las claves (ej: "¿qué es la clave 5?").
-   **Qué Mide:** La capacidad de **memorizar y recuperar información precisa**. Es una prueba de fuego para la memoria a corto plazo. Los Transformers deberían brillar aquí.

### Desafío 2: `Flip-Flop` (Seguimiento de Estado a Largo Plazo)
-   **La Prueba:** El modelo lee una secuencia larga donde un símbolo especial "invierte" un estado interno (como un interruptor de luz). Al final, debe decir cuál es el estado final.
-   **Qué Mide:** La capacidad de **mantener y actualizar un estado a lo largo de una secuencia muy larga**. Mide la memoria a largo plazo y la capacidad de evitar "olvidar". Aquí es donde los RNNs clásicos suelen fallar.

### Desafío 3: `Knapsack` (Razonamiento Algorítmico)
-   **La Prueba:** Se le presenta al modelo una versión simplificada del "problema de la mochila", un clásico problema de optimización.
-   **Qué Mide:** Si el modelo puede ir más allá de memorizar y empezar a mostrar signos de **razonamiento algorítmico abstracto**. Esta es la tarea más difícil y donde esperamos que la mayoría de los modelos fallen, pero cualquier éxito, por pequeño que sea, es muy significativo.

---

## El Proceso del Benchmark (Paso a Paso)

El script `run_benchmark.py` automatiza todo el experimento:

1.  **Itera:** Para cada una de las **9 arquitecturas**...
2.  **Entrena:** ...entrena un modelo desde cero en cada una de las **3 tareas**. Esto da un total de 27 experimentos (9 modelos x 3 tareas).
3.  **Mide:** En cada experimento, registra 4 métricas clave:
    -   **`Accuracy` (Precisión):** ¿Qué porcentaje de las respuestas son correctas?
    -   **`Perplexity` (Perplejidad):** ¿Qué tan "seguro" o "confundido" está el modelo? (Menor es mejor).
    -   **`Loss` (Pérdida):** La métrica de error durante el entrenamiento.
    -   **`Training Time` (Tiempo):** ¿Qué tan rápido aprende el modelo? (Mide la eficiencia).
4.  **Compara:** Al final, guarda todos los resultados en `results/` y genera gráficos que te permiten ver de un vistazo qué arquitectura es la mejor para cada desafío.
