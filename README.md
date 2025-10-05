# Informe Final: El Pentatlón de Arquitecturas

## 1. Resumen Ejecutivo

Este informe detalla una serie de experimentos controlados para evaluar el rendimiento comparativo de tres familias de arquitecturas de redes neuronales (RNN, Transformers y SSMs) en un conjunto de tareas de diagnóstico. El objetivo principal era identificar perfiles de especialización y comprender cómo los sesgos inductivos de cada arquitectura se alinean con diferentes desafíos cognitivos, como la memoria asociativa y el razonamiento algorítmico.

**Hallazgos Clave:**

1.  **La Especialización es Clave:** No existe una arquitectura universalmente superior. El rendimiento está intrínsecamente ligado a la naturaleza de la tarea, lo que demuestra que la elección de la arquitectura debe basarse en el problema a resolver.
2.  **SSMs Dominan la Memoria Asociativa:** Los modelos basados en recurrencia lineal (ej. `deltanet`) mostraron una superioridad decisiva en la tarea de memoria asociativa (`MQAR`), confirmando su eficacia para la recuperación de información en contexto.
3.  **Los RNNs Retienen una Ventaja Algorítmica:** El `lstm`, un RNN clásico, superó a todas las demás arquitecturas en la tarea de optimización (`Knapsack`), lo que sugiere que su naturaleza secuencial sigue siendo beneficiosa para problemas que requieren un razonamiento por pasos.
4.  **Los Transformers Necesitan Escala o Aumentos:** Con una configuración base modesta, los Transformers estándar (`rope`, `vanilla_transformer`) no lograron especializarse en ninguna tarea, lo que indica que su fortaleza puede depender de una escala masiva o de aumentos arquitectónicos específicos que no se probaron aquí.

## 2. Metodología Experimental

### 2.1. Arquitecturas Bajo Prueba

Seleccionamos modelos representativos de tres familias dominantes para la comparativa:
*   **Redes Neuronales Recurrentes (RNNs):** `lstm`, `gru`. Procesan secuencias de forma inherentemente secuencial, manteniendo un estado oculto comprimido.
*   **Modelos Basados en Atención (Transformers):** `vanilla_transformer`, `rope`. Procesan la secuencia de forma paralela, con un coste computacional cuadrático.
*   **Modelos de Espacio de Estados (SSMs):** `deltanet`, `gla`. Buscan combinar la eficiencia inferencial de los RNNs con el entrenamiento paralelizable de los Transformers.

### 2.2. Suite de Tareas de Diagnóstico ("El Pentatlón")

Diseñamos un conjunto de tareas para probar capacidades cognitivas distintas:
*   **`MQAR` (Memoria Asociativa):** Mide la capacidad de almacenar y recuperar hechos (pares clave-valor) de forma precisa.
*   **`Flip-Flop` (Seguimiento de Estado):** Mide la memoria a largo plazo y la capacidad de actualizar estados internos a lo largo de una secuencia.
*   **`Knapsack` (Razonamiento Algorítmico):** Mide la capacidad de aprender un procedimiento de optimización implícito para resolver un problema NP-duro.

### 2.3. Configuración de Entrenamiento Base

Para garantizar una comparación justa y atribuir las diferencias de rendimiento a la arquitectura en sí, todos los modelos se entrenaron bajo restricciones idénticas:

```python
config = {
    "d_model": 256,         # Dimensión del modelo modesta
    "n_layers": 4,          # Profundidad moderada
    "num_heads": 4,         # Para modelos de atención
    "num_epochs": 30,       # Entrenamiento corto para evaluar la eficiencia de aprendizaje
    "batch_size": 32,
    "optimizer": "AdamW",
    "learning_rate": 3e-4
}
```
Esta configuración intencionadamente restringida nos permite observar los sesgos inductivos fundamentales de cada modelo en un régimen de recursos limitados.

## 3. Análisis de Resultados

### 3.1. Rendimiento Relativo: El Mapa de Especialización

Para visualizar las fortalezas *relativas*, normalizamos las puntuaciones de precisión por tarea. Un valor de 1.0 (amarillo brillante) denota la mejor arquitectura para esa tarea específica.

![Relative Performance Heatmap](results/plots/relative_performance_heatmap.png)

**Interpretación del Mapa de Calor:**
Este gráfico ilustra de forma contundente nuestras conclusiones. Se observan "islas" de alto rendimiento bien definidas:
*   `deltanet` es el campeón indiscutible en `mqar`.
*   `lstm` es el claro ganador en `knapsack`.
*   `rope` muestra una sorprendente fortaleza relativa en `flip_flop`.

Esto demuestra visualmente que no hay un ganador global, sino especialistas claros para cada dominio del problema.

### 3.2. Rendimiento Absoluto por Tarea

El siguiente gráfico muestra la precisión absoluta, que fundamenta el análisis relativo.

![Accuracy by Task](results/plots/accuracy_comparison.png)

*   **Análisis de `MQAR`:** Solo `deltanet` logró un rendimiento superior al azar. Esto confirma que su mecanismo de actualización de estado es inherentemente más adecuado para la memorización asociativa que el estado oculto de un RNN o la atención global de un Transformer en esta escala.

*   **Análisis de `Flip-Flop`:** Todos los modelos convergieron a una precisión baja (~10%). Esto sugiere que la tarea, en su diseño actual, puede presentar un cuello de botella de capacidad para modelos de esta escala, o que requiere hiperparámetros muy específicos para ser resuelta.

*   **Análisis de `Knapsack`:** El `lstm` fue el único que mostró un aprendizaje significativo (aunque todavía bajo en términos absolutos). Este es un hallazgo clave: para tareas que se asemejan a un algoritmo clásico, la naturaleza secuencial de los RNN parece proporcionar un sesgo inductivo útil que los modelos más paralelos no poseen.

## 4. Conclusiones y Próximos Pasos

Los resultados de este "Pentatlón" confirman que el rendimiento del modelo es altamente específico de la tarea y refleja los sesgos inductivos de la arquitectura. **DeltaNet** se consolida como un aprendiz asociativo fuerte, los **RNNs** retienen una ventaja en el razonamiento procedimental, y los **Transformers** parecen requerir una escala mayor o aumentos para ser competitivos en estas tareas de diagnóstico.

**Próximos Pasos Planificados:**
1.  **Completar el Pentatlón:** Depurar y ejecutar con éxito la tarea `Hidden-Mode` para tener un conjunto de resultados completo.
2.  **Búsqueda de Hiperparámetros (Grid Search):** Realizar una búsqueda exhaustiva para los modelos más prometedores en sus tareas respectivas para determinar su rendimiento óptimo.
    *   **DeltaNet en `MQAR`:** Explorar `d_model` y `n_layers` para medir la escalabilidad de su memoria.
    *   **LSTM en `Knapsack`:** Ajustar `d_model` y `n_layers` para potenciar su capacidad de razonamiento.
3.  **Expandir el Conjunto de Arquitecturas:** Integrar modelos más modernos como **Mamba-2** y **DeltaFormer** para validar si las conclusiones se mantienen frente a arquitecturas de última generación.
