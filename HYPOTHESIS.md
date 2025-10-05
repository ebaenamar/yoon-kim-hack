# Hipótesis del Benchmark y Mapeo de Métricas

Este documento describe las hipótesis científicas que estamos probando y cómo las métricas que recopilamos servirán para confirmarlas o refutarlas.

---

## 1. La Hipótesis Central de la Investigación

Nuestra pregunta de investigación se puede formular como una hipótesis principal, audaz y falsable:

> **Hipótesis Principal:** Las arquitecturas modernas de RNNs Lineales (como DeltaNet, RetNet, GLA) pueden igualar o superar el rendimiento de los Transformers en tareas que requieren memoria y seguimiento de estado, mientras son significativamente más eficientes computacionalmente.

Para probar esto, lo desglosamos en sub-hipótesis, una para cada una de nuestras tareas de diagnóstico.

---

## 2. Cómo Cada Métrica Prueba una Parte de la Hipótesis

Cada tarea está diseñada para probar una capacidad específica, y cada métrica nos da una pieza del puzzle.

### Tarea 1: `MQAR` (Memoria Asociativa)

-   **Sub-Hipótesis:** "Los Transformers, gracias a su mecanismo de atención, serán significativamente superiores a todas las arquitecturas recurrentes (clásicas y modernas) en la recuperación de información asociativa."
-   **Métricas que lo Prueban:**
    -   **`Accuracy` (Precisión):** **Esta es la métrica clave aquí.** Esperamos ver una brecha enorme. Si los Transformers alcanzan >95% de accuracy y los RNNs se quedan en <50%, la hipótesis se confirma. La **desviación estándar** nos dirá si esta diferencia es estadísticamente significativa.
    -   **`Perplexity` (Perplejidad):** Nos dirá qué tan "confundido" está el modelo. Un Transformer seguro tendrá una perplejidad muy baja.

### Tarea 2: `Flip-Flop` (Seguimiento de Estado a Largo Plazo)

-   **Sub-Hipótesis:** "Los RNNs Lineales Modernos superarán a los RNNs Clásicos (LSTM/GRU) y serán competitivos con los Transformers, demostrando que han solucionado el problema del 'olvido' a largo plazo."
-   **Métricas que lo Prueban:**
    -   **`Accuracy`:** La métrica principal. Esperamos ver a los RNNs Clásicos fallar, y a los RNNs Lineales competir directamente con los Transformers.
    -   **`Training Time` (Tiempo de Entrenamiento):** **Esta es la segunda métrica clave.** Si un RNN Lineal iguala la accuracy de un Transformer pero en la mitad de tiempo, probamos la parte de la "eficiencia" de nuestra hipótesis principal.

### Tarea 3: `Knapsack` (Razonamiento Algorítmico)

-   **Sub-Hipótesis:** "La mayoría de las arquitecturas fallarán en aprender el algoritmo subyacente, pero cualquier modelo que logre una accuracy consistentemente por encima del azar demostrará una capacidad emergente para el razonamiento abstracto."
-   **Métricas que lo Prueban:**
    -   **`Accuracy`:** Aquí no buscamos un 100%. El "éxito" sería encontrar una arquitectura que, de manera consistente (baja desviación estándar), logre un 20-30% de accuracy, mientras que las demás se quedan en <5%.
    -   **Análisis de Errores (Cualitativo):** Para el modelo ganador (si lo hay), un análisis posterior de sus errores nos daría pistas sobre *cómo* está intentando razonar.

---

## 3. La Visualización de Resultados: Contando la Historia

Nuestros gráficos están diseñados para contar esta historia de forma visual e impactante:

1.  **Gráfico de Barras de Accuracy (con barras de error):** Será la prueba visual más potente para comparar el rendimiento y la consistencia.

2.  **Gráfico de Eficiencia (Accuracy vs. Tiempo):** Este gráfico es crucial para probar la hipótesis principal. Buscaremos modelos en la **esquina superior izquierda**: alta accuracy y bajo tiempo de entrenamiento.

3.  **Tabla de Resumen Numérico:** La tabla final nos dará los números exactos, con medias y desviaciones estándar, para poder hacer afirmaciones cuantitativas y rigurosas en un futuro paper.
