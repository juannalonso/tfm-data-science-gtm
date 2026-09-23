# Entrega 4 — Diseño del análisis y estrategia de modelado

## 1. Problema que se busca resolver

El proyecto aborda un problema real del ámbito comercial B2B: la **cualificación y priorización de leads** en estrategias de outbound. Un comercial que vende productos o servicios a restaurantes dispone habitualmente de un universo grande de empresas potencialmente contactables, pero solo una fracción encaja realmente con su Perfil de Cliente Ideal (ICP). Actualmente, este proceso se resuelve de dos formas: mediante scrapers con filtros manuales rígidos (fuera de escala y sin ranking), o mediante criterio humano subjetivo (poco reproducible y difícil de escalar).

El resultado deseado es un sistema que, dado un pequeño conjunto de restaurantes-referencia aportados por el comercial (típicamente 3 clientes que ya funcionan bien), aprenda automáticamente el patrón que los caracteriza y devuelva un ranking priorizado de todos los restaurantes del universo por similitud con ese ICP, acompañado de una explicación individualizada para cada resultado.

El usuario final del sistema es el equipo comercial de una agencia Go-to-Market especializada en outbound B2B. La decisión que apoya el sistema es **a qué restaurantes contactar primero** dentro de una campaña de outbound, optimizando el tiempo del comercial al reducir el ruido y priorizar leads con mayor probabilidad de encaje. Para considerarse útil, el sistema debe superar de forma medible al filtrado manual como baseline, ofrecer resultados explicables y funcionar con muy pocos ejemplos (few-shot learning).

## 2. Análisis de datos planteado y utilidad esperada

Antes del modelado se han realizado varios análisis descriptivos sobre la capa gold que han condicionado directamente las decisiones técnicas del proyecto:

**Análisis de cobertura y calidad de fuentes:** se validó empíricamente la cobertura de OpenStreetMap respecto al censo oficial del Ayuntamiento de Madrid (96,2%) y el porcentaje de restaurantes enriquecidos con datos oficiales tras entity resolution (80,4%). Este análisis fundamenta la arquitectura híbrida de fuentes.

**Análisis de distribución de features numéricas:** se examinó la distribución de renta media (rango 13.000–37.000 € tras limpieza), competencia geográfica (media de 209 restaurantes en 500m, con máximos en Sol de 350) y cobertura de campos operativos (tiene_web 42%, tiene_terraza 10%, ofrece_delivery <1%). Esto ha permitido descartar variables con baja varianza (delivery, takeaway) del modelo por su nula capacidad discriminativa.

**Análisis crítico del enriquecimiento LLM:** sobre las 794 primeras respuestas de Gemini se analizó la distribución de las variables cualitativas generadas. Se detectó un sesgo del modelo hacia categorías centrales en dos variables: `posicionamiento_precio` (89,9% "medio") y `tipo_clientela` (83,8% "mixta"). En cambio, `ambiente` (casual 50%, tradicional 37%, moderno 11%) y `presencia_digital` (nula 44%, básica 31%, activa 25%) presentan distribuciones equilibradas. Esta observación empírica ha condicionado directamente qué variables entran al modelo y cuáles se mantienen únicamente como campos descriptivos en la interfaz.

**Análisis geográfico:** se calcularon las variables derivadas de competencia (restaurantes en radios de 500m y 1km) y se cruzó la sección censal con la renta media del INE, comprobando la variabilidad intra-barrio (secciones del mismo barrio pueden diferir hasta 15.000 € en renta anual por persona). Este análisis fundamenta la inclusión de la renta como feature discriminativa.

Estos análisis alimentan tanto el modelado como el MVP: en la aplicación Streamlit, el usuario podrá consultar las mismas variables analizadas (competencia, renta, categorización LLM) tanto para los seeds como para los leads devueltos, favoreciendo la interpretación humana del resultado.

## 3. Tipo de modelos que se van a plantear

La tarea es un problema de **recomendación con muy pocos ejemplos positivos** (few-shot learning), sin datos históricos etiquetados de conversión. No es clasificación supervisada tradicional (no existe una variable objetivo binaria "convirtió/no convirtió"), ni tampoco clustering puro (el ICP viene dado por el usuario, no se descubre). Se plantea abordarlo mediante técnicas de **similarity learning** complementadas con una alternativa de clasificación supervisada.

| Alternativa | Tipo | Por qué se plantea | Limitación principal |
|---|---|---|---|
| **Baseline** | Filtro por reglas manuales | Simula el enfoque actual del comercial: filtros duros por barrio, cocina, presencia digital y posicionamiento. Proporciona la referencia mínima que el modelo debe superar. | No aprende del ICP, solo aplica reglas fijas. No devuelve ranking, solo lista binaria. |
| **Modelo 1: Similitud coseno con vector combinado** | Similarity learning (interpretable) | Representa cada restaurante como un vector que combina features numéricas normalizadas, codificación one-hot de variables categóricas y el embedding semántico del resumen generado por LLM (sentence-transformers, 384 dimensiones). Calcula la similitud coseno con el centroide de los seeds. Es robusto con pocos ejemplos, no requiere entrenamiento, y las contribuciones por feature son directamente interpretables. | Trata todas las dimensiones con el mismo peso salvo por la varianza natural del centroide. No modela interacciones complejas entre variables. |
| **Modelo 2: XGBoost con positive-unlabeled learning** | Clasificación (avanzado, comparación) | Se plantea como comparación para explorar si un modelo más flexible captura patrones no evidentes. Los seeds se etiquetan como positivos y se genera una muestra aleatoria de negativos hipotéticos del resto del universo. Se entrena un clasificador binario y se usa la probabilidad como score. | Con solo 3-5 positivos reales, el riesgo de sobreajuste es alto. Los "negativos" no son verdaderos negativos (son unlabeled). La interpretación requiere SHAP. |

El criterio de selección final privilegiará el **modelo más adecuado para el caso de uso**, no necesariamente el que obtenga la mejor métrica: un modelo ligeramente menos preciso pero más estable, explicable y coherente con el volumen real de datos (pocos seeds) puede ser preferible. Se contempla explícitamente la posibilidad de descartar el Modelo 2 si su rendimiento no supera al Modelo 1 o si su comportamiento resulta inestable ante distintos conjuntos de seeds, dejando su implementación mencionada como línea futura de trabajo.

## 4. Datos de entrada del análisis y los modelos

La entrada de los modelos es la capa gold consolidada, ya descrita en la Entrega 3 y actualmente cerrada con 1.688 restaurantes y 33 columnas.

**Dataset principal:** `data/gold/gold_restaurantes_madrid.parquet`.
**Granularidad:** una fila por restaurante del distrito Centro de Madrid.
**Clave principal:** `restaurante_id` (identificador sintético reproducible construido por hash sobre nombre + código postal + coordenadas).

| Entrada | Descripción | Granularidad / tipo | Uso en el análisis o modelo |
|---|---|---|---|
| `gold_restaurantes_madrid.parquet` | Dataset consolidado final de la Entrega 3. | Una fila por restaurante (1.688 filas × 33 columnas). | Fuente única de features y descriptivos. |
| `renta_media` | Renta neta media por persona/año de la sección censal (INE 2023). | Numérica continua (€). | Feature socioeconómica del entorno. |
| `competencia_500m`, `competencia_1km` | Número de restaurantes en un radio geográfico. | Numérica entera. | Features derivadas geoespacialmente. |
| `tiene_web`, `tiene_telefono`, `tiene_horario`, `tiene_terraza`, `es_accesible` | Indicadores operativos binarios (presencia de dato en OSM). | Booleana. | Features estructurales del restaurante. |
| `barrio`, `cocina`, `ambiente`, `presencia_digital` | Variables categóricas (5-30 valores cada una). | Categórica. | Features one-hot en el vector del modelo. |
| `resumen_llm` | Texto de 1-2 frases generado por Gemini con descripción cualitativa del restaurante. | Texto libre. | Se convierte en embedding semántico de 384 dimensiones mediante sentence-transformers (modelo `all-MiniLM-L6-v2`). |
| `icps_demos.json` | Fichero auxiliar con los 6 restaurantes-ejemplo de las 2 demos preseleccionadas para la app. | JSON estructurado. | Presets del sistema en Streamlit. |

**Variables excluidas del modelo:**
- `ofrece_delivery` y `ofrece_takeaway`: excluidas por muy baja varianza (<2% de valores True), no aportan discriminación.
- `posicionamiento_precio` y `tipo_clientela`: excluidas del scoring por el sesgo del LLM detectado (baja varianza tras análisis empírico), pero se mantienen visibles en la app como campos descriptivos.
- Identificadores no explicativos (`restaurante_id`, `fecha_construccion`, `fuente_principal`).
- Metadatos técnicos que no describen al restaurante (`metodo_enriquecimiento_censo`, `renta_imputada` como flag).

**Disponibilidad temporal:** todas las variables están disponibles en el momento del scoring (no hay riesgo de data leakage temporal, ya que no se predice ningún evento futuro).

## 5. Datos de salida y forma de consumo

La salida principal del sistema es un **ranking priorizado** de los restaurantes del universo respecto a los seeds aportados, con explicación individual para cada resultado.

| Campo de salida | Descripción | Tipo | Uso posterior |
|---|---|---|---|
| `restaurante_id` | Identificador del restaurante rankeado. | string | Trazabilidad y unión con el resto de datos del sistema. |
| `nombre` | Nombre del restaurante. | string | Visualización en la app. |
| `similitud_score` | Score de similitud coseno con el centroide del ICP (rango 0-1, valores más altos = más similar). | float | Ranking principal y filtrado en la app. |
| `posicion_ranking` | Posición en el ranking (1 = más similar). | int | Ordenación visual. |
| `explicacion` | Detalle de las features en las que este restaurante coincide más con el promedio del ICP (por ejemplo: barrio, ambiente, rango de renta similar). | dict / string | Justificación mostrada al usuario en el detalle del lead. |
| `fecha_ejecucion` | Momento en que se generó el ranking. | datetime | Trazabilidad y reproducibilidad. |

**Formato de consumo:**

- **En la aplicación Streamlit:** el ranking se muestra como tabla interactiva ordenada por score, con selección de top-K (por defecto 20). El usuario puede filtrar por barrio, ambiente, presencia digital, etc. Al clicar un restaurante se abre una vista de detalle con toda su información (incluido resumen LLM) y la explicación del score.
- **Descarga:** botón para exportar el ranking a CSV, para integrarlo con herramientas de outbound (Clay, Smartlead, HeyReach).
- **Visualización geográfica:** mapa de Madrid con los seeds marcados en verde y los top-20 leads en azul, para facilitar la interpretación espacial del ranking.

**Explicación y confianza:** para cada lead del ranking se muestra por qué ha entrado en el top, indicando las features en las que se parece más al ICP promedio. En próximas iteraciones se podrá añadir SHAP para explicar formalmente la contribución de cada feature al score.

## 6. Estrategia para diseñar y seleccionar el modelo

**Preparación del dataset de modelado:** partiendo de la capa gold, se construye una matriz de features `X` de dimensiones 1.688 × ~440 (contando el embedding del resumen LLM). Las variables numéricas se normalizan con `StandardScaler`, las categóricas se convierten a one-hot con `pd.get_dummies`, y el texto del `resumen_llm` se convierte en embedding con `sentence-transformers` (`all-MiniLM-L6-v2`).

**Definición de la salida:** no hay variable objetivo estrictamente hablando. En el Modelo 1 (similitud coseno), la salida es directamente el score de similitud calculado. En el Modelo 2 (XGBoost), la salida es la probabilidad de ser positivo (pertenecer al ICP), donde los 3 seeds se etiquetan como positivos y se genera una muestra negativa aleatoria del resto.

**Construcción del baseline:** filtro por reglas duras que un comercial podría escribir a mano: por ejemplo, para el escenario premium, "restaurantes con `tiene_web = True` AND `presencia_digital in ['activa', 'sofisticada']` AND `barrio in ['JUSTICIA', 'PALACIO', 'CORTES']`". Devuelve todos los restaurantes que cumplen, sin ranking.

**Comparación entre alternativas:**
- **Modelo 1 (similitud coseno):** rápido, sin entrenamiento, interpretable. Se toma como candidato principal.
- **Modelo 2 (XGBoost):** requiere entrenar con positivos + negativos sintéticos. Se compara su ranking con el del Modelo 1 y con el baseline.

**Criterios de comparación:**
- **Calidad predictiva** (precision@k, lift vs baseline).
- **Estabilidad:** ¿el ranking cambia mucho si se cambia un seed? Un modelo demasiado sensible al conjunto exacto de seeds no es robusto.
- **Interpretabilidad:** capacidad de explicar por qué un restaurante entra en el top.
- **Coste computacional:** el sistema debe responder en tiempo interactivo desde la app.
- **Utilidad práctica para el MVP.**

**Regla de decisión final:** se seleccionará el modelo que (i) supere al baseline en precision@k con margen estadísticamente significativo, (ii) devuelva rankings coherentes y estables entre distintos conjuntos de seeds, (iii) sea explicable al usuario final. Si ningún modelo cumple los tres criterios, se optará por el Modelo 1 (similitud coseno) por su robustez y simplicidad, y se documentará el resultado del Modelo 2 como línea de mejora futura.

## 7. Estrategia de validación y evaluación

Dada la ausencia de ground truth histórico (no hay datos de qué leads convirtieron y cuáles no), la validación se plantea mediante técnicas alternativas coherentes con el enfoque de few-shot learning:

**Separación de datos: recuperación de seeds held-out.**
Para cada conjunto de seeds del ICP, se aparta uno de los seeds (leave-one-out), se entrena el modelo con los 2 restantes, y se comprueba si el seed apartado aparece en el top-K del ranking devuelto. Se repite iterativamente y se calcula la tasa de recuperación. Es la técnica estándar en recomendadores sin ground truth explícito.

**Baseline sintético con ICPs generados.**
Se generan conjuntos de seeds sintéticos con reglas conocidas (por ejemplo, "3 restaurantes premium con ambiente moderno en Justicia") y se verifica si el sistema recupera restaurantes que cumplen esas mismas reglas en el top-K.

**Validación cualitativa por el usuario.**
Se revisa manualmente el top-20 de las 2 demos preparadas (software de reservas premium y distribuidor asiático) y se valora la coherencia comercial: ¿los restaurantes devueltos son leads plausibles para el ICP? Esta validación cualitativa complementa las métricas cuantitativas.

| Elemento | Decisión prevista | Justificación |
|---|---|---|
| **Separación de datos** | Leave-one-out sobre los seeds del ICP + generación de ICPs sintéticos para validación adicional. | No hay ground truth; la recuperación de seeds held-out es la técnica estándar. |
| **Métrica principal** | Precision@K (K=10, 20) y Lift vs baseline. | Precision@K refleja directamente la utilidad para el comercial (calidad de los primeros leads). Lift mide la ganancia real sobre el filtro manual. |
| **Baseline** | Filtro manual por reglas duras + selección aleatoria. | Permite medir la mejora real del modelo sobre lo que ya se hace hoy y sobre el azar. |
| **Criterio de aceptación** | Precision@20 al menos 30% superior al baseline manual. Recuperación de seeds held-out ≥70%. | Umbrales razonables que justifican el uso del sistema frente a alternativas más simples. |

**Análisis por segmentos:** se estudiará el comportamiento del sistema para distintos perfiles de ICP (premium vs asiático, homogéneo vs heterogéneo) para detectar en qué casos el modelo funciona mejor o peor. Es información valiosa tanto para la memoria como para la defensa.

**Resultado mínimo aceptable:** si ningún modelo alcanza los umbrales establecidos, se profundizará en el análisis de errores (¿qué tipo de seeds funciona mal?), se documentará la limitación honestamente y se planteará como línea futura de trabajo la ampliación del dataset con datos históricos reales de conversión (que no están disponibles en el contexto del TFM).

## 8. Riesgos y alternativas

**Ausencia de variable objetivo real.**
El proyecto no dispone de datos históricos de conversión (qué leads acabaron siendo clientes reales). La validación se realiza mediante recuperación de seeds y evaluación cualitativa, no mediante etiquetas ground truth. Es una limitación reconocida y honesta del enfoque, propia del caso de uso (few-shot learning sin historial). Se documentará explícitamente en la memoria y la defensa. La alternativa sería un entorno productivo con feedback loop de conversión real, algo fuera del alcance del TFM.

**Riesgo de data leakage.**
Bajo. Todos los datos utilizados están disponibles en el momento del scoring y no hay componente temporal en las predicciones. El único riesgo residual es que los seeds coincidan con restaurantes del universo (evidente), pero se resuelve excluyéndolos del ranking devuelto.

**Volumen y calidad de datos.**
Volumen: 1.688 restaurantes es suficiente para el alcance del MVP (distrito Centro de Madrid). Calidad: la capa gold está bien caracterizada, con un 80,4% de restaurantes cruzados con datos oficiales del censo y 99,9% de enriquecimiento LLM. Las limitaciones conocidas están documentadas en la Entrega 3.

**Sesgo del enriquecimiento LLM.**
El análisis de las 794 primeras respuestas de Gemini reveló un sesgo hacia categorías centrales en `posicionamiento_precio` y `tipo_clientela`. Este sesgo se ha gestionado excluyendo esas variables del modelo (baja capacidad discriminativa) y manteniéndolas solo como descriptivos en la app. Es un ejemplo de análisis crítico de una fuente de datos generada por IA.

**Segmentos con pocos datos.**
Algunas cocinas están sobrerrepresentadas (regional, spanish, chinese) y otras muy poco (vietnamese, korean con menos de 15 registros). Esto puede afectar al modelo cuando el ICP se concentra en cocinas poco representadas. Se gestionará ampliando el conjunto de seeds en esos casos o documentando la limitación.

**Incertidumbre principal.**
La parte que más incertidumbre genera es la **evaluación cuantitativa sin ground truth**: aunque técnicas como leave-one-out sobre seeds son estándar, no reemplazan a un ground truth real. La calidad del sistema se demostrará también con la validación cualitativa de las 2 demos preparadas y con la coherencia de los rankings observada por el usuario.

**Alternativas si el enfoque falla.**
Si el Modelo 1 no supera al baseline manual, la alternativa inmediata es enriquecer la representación vectorial (más features, embedding de mayor dimensión, o normalización distinta). Si el Modelo 2 (XGBoost) resulta inestable con solo 3 seeds, se descarta y se justifica su exclusión en la memoria: se documentaría como línea futura de trabajo para escenarios con más datos históricos disponibles. En el caso extremo de que ningún modelo funcione con rigor, el sistema seguiría siendo defendible como una herramienta de análisis y exploración cualitativa de leads apoyada en enriquecimiento de datos y LLM, sin componente predictivo formal.
