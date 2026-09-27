# Entrega 4 — Diseño del análisis y estrategia de modelado

## 1. Problema que se busca resolver

El proyecto aborda un problema real del ámbito comercial B2B: la **cualificación y priorización de leads** en estrategias de outbound. Un comercial que vende productos o servicios a restaurantes dispone habitualmente de un universo grande de empresas potencialmente contactables, pero solo una fracción encaja realmente con su Perfil de Cliente Ideal (ICP). Actualmente, este proceso se resuelve de dos formas: mediante scrapers con filtros manuales rígidos (fuera de escala y sin ranking), o mediante criterio humano subjetivo (poco reproducible y difícil de escalar).

El resultado deseado es un sistema que, dado un pequeño conjunto de restaurantes-referencia aportados por el comercial (típicamente 3-5 clientes que ya funcionan bien), aprenda automáticamente el patrón que los caracteriza y devuelva un ranking priorizado de todos los restaurantes del universo por similitud con ese ICP, acompañado de una explicación individualizada para cada resultado.

El usuario final del sistema es el equipo comercial de una agencia Go-to-Market especializada en outbound B2B. La decisión que apoya el sistema es **a qué restaurantes contactar primero** dentro de una campaña de outbound, optimizando el tiempo del comercial al reducir el ruido y priorizar leads con mayor probabilidad de encaje. Para considerarse útil, el sistema debe superar de forma medible al filtrado manual como baseline, ofrecer resultados explicables y funcionar con muy pocos ejemplos (few-shot learning).

## 2. Análisis de datos planteado y utilidad esperada

Antes del modelado se han realizado varios análisis descriptivos sobre la capa gold que han condicionado directamente las decisiones técnicas del proyecto:

**Análisis de cobertura y calidad de fuentes:** se validó empíricamente la cobertura de OpenStreetMap respecto al censo oficial del Ayuntamiento de Madrid (96,2%) y el porcentaje de restaurantes enriquecidos con datos oficiales tras entity resolution (80,4%). Este análisis fundamenta la arquitectura híbrida de fuentes.

**Análisis de distribución de features numéricas:** se examinó la distribución de renta media (rango 13.000–37.000 € tras limpieza), competencia geográfica (media de 209 restaurantes en 500m, con máximos en Sol de 350) y cobertura de campos operativos (tiene_web 42%, tiene_terraza 10%, ofrece_delivery <1%). Esto ha permitido descartar variables con baja varianza (delivery, takeaway) del modelo por su nula capacidad discriminativa.

**Análisis crítico del enriquecimiento LLM:** sobre las 794 primeras respuestas de Gemini se analizó la distribución de las variables cualitativas generadas. Se detectó un sesgo del modelo hacia categorías centrales en dos variables: `posicionamiento_precio` (89,9% "medio") y `tipo_clientela` (83,8% "mixta"). En cambio, `ambiente` (casual 50%, tradicional 37%, moderno 11%) y `presencia_digital` (nula 44%, básica 31%, activa 25%) presentan distribuciones equilibradas. Esta observación empírica ha condicionado directamente qué variables entran al modelo y cuáles se mantienen únicamente como campos descriptivos en la interfaz.

**Análisis geográfico:** se calcularon las variables derivadas de competencia (restaurantes en radios de 500m y 1km) y se cruzó la sección censal con la renta media del INE, comprobando la variabilidad intra-barrio (secciones del mismo barrio pueden diferir hasta 15.000 € en renta anual por persona). Este análisis fundamenta la inclusión de la renta como feature discriminativa.

Estos análisis alimentan tanto el modelado como el MVP: en la aplicación Streamlit, el usuario podrá consultar las mismas variables analizadas (competencia, renta, categorización LLM) tanto para los seeds como para los leads devueltos, favoreciendo la interpretación humana del resultado.

## 3. Tipo de modelos planteados y resultados

La tarea es un problema de **recomendación con muy pocos ejemplos positivos** (few-shot learning), sin datos históricos etiquetados de conversión. No es clasificación supervisada tradicional (no existe una variable objetivo binaria "convirtió/no convirtió"), ni tampoco clustering puro (el ICP viene dado por el usuario, no se descubre). Se abordó mediante técnicas de **similarity learning** complementadas con una alternativa de clasificación supervisada, comparando ambas contra un baseline manual.

| Alternativa | Tipo | Implementación | Resultado LOO (Recall@100) |
|---|---|---|---|
| **Baseline manual** | Filtro por reglas duras | `posicionamiento_precio ∈ {premium, alta_gama}` AND `tiene_web = True` AND `presencia_digital = activa`. Devuelve 43 restaurantes sin ranking. | **1.000** (espejismo — ver §6) |
| **Modelo 1: Similitud coseno con vector combinado (551 dims)** | Similarity learning (interpretable) | Vector combinado de 167 features estructuradas (numéricas normalizadas + booleanas + one-hot de categóricas) más embedding semántico de 384 dimensiones del `resumen_llm` generado por Gemini, calculado con `sentence-transformers/all-MiniLM-L6-v2`. Score = similitud coseno con el centroide de los seeds, rango [-1, 1]. Determinista, sin entrenamiento. | **0.600** |
| **Modelo 2: XGBoost con Positive-Unlabeled Learning** | Clasificación (benchmark) | 5 seeds como positivos, 200 negativos aleatorios del resto del universo. Clasificador binario, se usa la probabilidad como score. Introduce estocasticidad por el muestreo de negativos. | **0.400** |

**Decisión final:** se selecciona la **similitud coseno con vector combinado (Modelo 1)** como modelo en producción. Justificación detallada en §6, tras la interpretación de resultados.

## 4. Datos de entrada del análisis y los modelos

La entrada de los modelos es la capa gold consolidada, descrita en la Entrega 3 y actualmente cerrada con 1.688 restaurantes y 33 columnas.

**Dataset principal:** `data/gold/gold_restaurantes_madrid.parquet`.
**Granularidad:** una fila por restaurante del distrito Centro de Madrid.
**Clave principal:** `restaurante_id` (identificador sintético reproducible construido por hash sobre nombre + código postal + coordenadas).

| Entrada | Descripción | Granularidad / tipo | Uso en el análisis o modelo |
|---|---|---|---|
| `gold_restaurantes_madrid.parquet` | Dataset consolidado final de la Entrega 3. | Una fila por restaurante (1.688 filas × 33 columnas). | Fuente única de features y descriptivos. |
| `renta_media` | Renta neta media por persona/año de la sección censal (INE 2023). | Numérica continua (€). | Feature socioeconómica del entorno. |
| `competencia_500m`, `competencia_1km` | Número de restaurantes en un radio geográfico. | Numérica entera. | Features derivadas geoespacialmente. |
| `tiene_web`, `tiene_telefono`, `tiene_horario`, `tiene_terraza`, `es_accesible`, `ofrece_delivery`, `ofrece_takeaway` | Indicadores operativos binarios (presencia de dato en OSM). | Booleana. | Features estructurales del restaurante (7 flags). |
| `barrio`, `cocina`, `ambiente`, `presencia_digital`, `epigrafe_oficial` | Variables categóricas (5-30 valores cada una). | Categórica. | Features one-hot en el vector del modelo (~157 dims tras codificación). |
| `resumen_llm` | Texto de 1-2 frases generado por Gemini con descripción cualitativa del restaurante. | Texto libre. | **Embedding semántico de 384 dimensiones** integrado en el vector del modelo (`sentence-transformers/all-MiniLM-L6-v2`), cacheado en `data/processed/embeddings_resumen_llm.parquet`. |
| Presets de ICP | Configuración de los 5 seeds premium (Sagardi, Sandó, Dray Martina, AskuaBarra, Gioia) más los 3 seeds del caso asiático (Okashi Sanda, Hunan Restaurant, Tuk Tuk), identificados por `restaurante_id`. | Configuración en `app/app.py`. | Presets de la app Streamlit para reproducibilidad. |

**Vector final del modelo:** 551 dimensiones por restaurante (3 numéricas + 7 booleanas + ~157 one-hot categóricas + 384 embedding).

**Variables excluidas del modelo:**
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
| `similitud_score` | Score de similitud coseno con el centroide del ICP. Rango teórico **[-1, 1]** (1 = idéntico, -1 = opuesto). En la práctica, los scores observados caen en **[0.40, 0.85]** debido a la naturaleza mayoritariamente no-negativa del embedding. | float | Ranking principal y filtrado en la app. |
| `posicion_ranking` | Posición en el ranking (1 = más similar). | int | Ordenación visual. |
| `bloques_del_score` | Desglose del score en (a) similitud estructural (167 features) y (b) similitud semántica (embedding). | dict | Explicabilidad mostrada al usuario. |
| `fecha_ejecucion` | Momento en que se generó el ranking. | datetime | Trazabilidad y reproducibilidad. |

**Formato de consumo:**

- **En la aplicación Streamlit:** el ranking se muestra como tabla interactiva ordenada por score, con selección de top-K (por defecto 20). El usuario puede filtrar por barrio, ambiente, presencia digital, etc. Al clicar un restaurante se abre una vista de detalle con toda su información (incluido resumen LLM) y la descomposición del score por bloques.
- **Descarga:** botón para exportar el ranking a CSV, para integrarlo con herramientas de outbound (Clay, Smartlead, HeyReach).
- **Visualización geográfica:** mapa de Madrid con los seeds marcados en verde y los top-20 leads en azul, para facilitar la interpretación espacial del ranking.

## 6. Estrategia de modelado, evaluación y decisión final

### 6.1. Preparación del dataset de modelado

Partiendo de la capa gold, se construye una matriz de features `X` de dimensiones 1.688 × 551. Las variables numéricas se normalizan con `StandardScaler`, las categóricas se convierten a one-hot con `pd.get_dummies`, y el texto del `resumen_llm` se convierte en embedding con `sentence-transformers` (`all-MiniLM-L6-v2`). Todo el pipeline vive en `src/similitud.py`, la misma función `construir_vectores` que usa la app Streamlit en producción — **el código y la documentación están alineados**.

### 6.2. Metodología de evaluación: Leave-One-Out (LOO)

Dada la ausencia de ground truth histórico (no hay datos de qué leads convirtieron), la validación se realiza mediante **recuperación de seeds held-out**: para cada seed del ICP, se aparta, se calcula el ranking con los otros 4, y se mide en qué posición aparece el seed apartado (Recall@K).

- **Recall@20:** ¿aparece el seed apartado en el top 1.2% del universo (20 de 1683)?
- **Recall@100:** ¿aparece en el top 6% (100 de 1683)?

### 6.3. Resultados del experimento

Los seeds premium utilizados son los 5 restaurantes configurados como Demo 1 en la app: **Sagardi**, **Sandó**, **Dray Martina**, **AskuaBarra** y **Gioia** (identificados por `restaurante_id` para evitar colisiones con homónimos en el dataset).

| Modelo | Recall@20 | Recall@100 | Interpretación |
|---|---|---|---|
| Baseline manual | **0.600** | **1.000** | Espejismo — ver más abajo |
| Coseno + embedding (551 feat) | 0.000 | **0.600** | 3/5 seeds en top 6% del universo |
| XGBoost PU | **0.200** | 0.400 | 1 seed en top 1 absoluto |

### 6.4. Por qué el baseline "gana" pero es un espejismo

El baseline manual devuelve un **conjunto fijo de 43 restaurantes** que cumplen las 3 reglas (premium/alta_gama + web + digital activa). Los 5 seeds cumplen esas reglas por diseño, así que están automáticamente dentro del filtro. Los puestos [6, 7, 19, 22, 25] son artificiales: derivan del orden por índice dentro de los 43 restaurantes empatados a score=1.

**Recall@100 = 1.0 no significa que el baseline rankee bien**, significa que "los 5 seeds están dentro de mis 43 restaurantes filtrados". No aporta capacidad de descubrimiento: cualquier restaurante fuera del filtro (con web pero posicionamiento `medio`, o con presencia digital `básica`) queda invisible aunque semánticamente encaje con el ICP.

### 6.5. Por qué el coseno gana en utilidad práctica

**Coseno + embedding: Recall@100 = 0.60** — rankea los 1683 restaurantes del universo entero. Cuando se dejan fuera del centroide, las seeds aparecen en:

- **Sandó** en posición 36
- **Dray Martina** en posición 39
- **AskuaBarra** en posición 85

Estas 3 seeds caen dentro del **top 6% del universo**, lo que confirma que el modelo captura la firma vectorial del ICP más allá de las 3 reglas duras del baseline. Además, el ranking cualitativo (top 10) devuelve restaurantes coherentes como La Taberna del Alabardero, Café de Oriente y Zest Almagro, todos reconocibles como perfil premium.

### 6.6. XGBoost como benchmark

XGBoost PU consigue Recall@20 = 0.20 metiendo una seed en el TOP 1 absoluto (posición 1 de 1683), lo cual es notable. Sin embargo, en Recall@100 (0.40) queda por debajo del coseno. XGBoost es más "confiado" con los positivos claros pero menos exhaustivo con la exploración semántica. Además, introduce estocasticidad por el muestreo de negativos sintéticos y requiere SHAP para interpretabilidad.

### 6.7. Sobre el embedding de resumen_llm

El embedding se implementó siguiendo el feedback del tutor, que señaló una incoherencia entre documentación y código en la iteración anterior. Comparado con el mismo modelo **sin embedding** (167 features estructuradas), Recall@100 apenas cambia (~0.60). El embedding **no aporta ganancia cuantitativa medible con 5 seeds premium tan homogéneas estructuralmente**, pero queda integrado en el sistema para:

- Escalar a 20-50 seeds futuras, donde el ruido estructural aumentará y el embedding podrá aportar
- Habilitar explicabilidad semántica en la app (descomposición del score por bloques)

Esta es una **conclusión honesta**: el ejercicio se hizo, se midió, y el resultado se cuenta tal cual, tanto en el notebook como en esta memoria.

### 6.8. Decisión final: coseno con embedding en producción

**Ventajas frente al baseline manual:**
- Rankea el universo entero (1683), no solo un subconjunto filtrado (43)
- Score continuo interpretable ("este lead se parece al X% a tu ICP")
- Descubre leads fuera de las reglas duras que semánticamente encajan
- Escalable sin reentrenar (basta con actualizar el centroide al añadir seeds)

**Ventajas frente a XGBoost:**
- No requiere muestreo de negativos sintéticos (sesgo eliminado)
- Determinista: mismo input, mismo output
- Interpretable mediante descomposición del score por bloques (estructural vs semántico)
- Recall@100 superior (0.60 vs 0.40)

**XGBoost queda como benchmark comparativo, no como modelo principal.** El baseline manual queda como referencia de mínimos (lo que cualquier comercial haría sin herramienta), pero se ha demostrado que su alto Recall@100 es artefacto del propio diseño del filtro y no una medida de utilidad práctica.

## 7. Riesgos y limitaciones reconocidas

**Ausencia de variable objetivo real.**
El proyecto no dispone de datos históricos de conversión (qué leads acabaron siendo clientes reales). La validación se realiza mediante recuperación de seeds y evaluación cualitativa, no mediante etiquetas ground truth. Es una limitación reconocida y honesta del enfoque, propia del caso de uso (few-shot learning sin historial). La alternativa sería un entorno productivo con feedback loop de conversión real, algo fuera del alcance del TFM.

**Muestra pequeña de seeds (5).**
Con 5 seeds y un universo de 1.683 restaurantes, el LOO es una prueba de estrés muy exigente: Recall@20 pide que el seed apartado quede en el top 1.2% cuando el centroide se calcula con solo 4 puntos. Se documenta como limitación intrínseca; con 20-50 seeds las métricas serían más discriminativas y el embedding aportaría más valor.

**Riesgo de data leakage.**
Bajo. Todos los datos utilizados están disponibles en el momento del scoring y no hay componente temporal en las predicciones. El único riesgo residual es que los seeds coincidan con restaurantes del universo (evidente), pero se resuelve excluyéndolos del ranking devuelto.

**Sesgo del enriquecimiento LLM.**
El análisis de las 794 primeras respuestas de Gemini reveló un sesgo hacia categorías centrales en `posicionamiento_precio` y `tipo_clientela`. Este sesgo se ha gestionado excluyendo esas variables del modelo (baja capacidad discriminativa) y manteniéndolas solo como descriptivos en la app. Es un ejemplo de análisis crítico de una fuente de datos generada por IA.

**Segmentos con pocos datos.**
Algunas cocinas están sobrerrepresentadas (regional, spanish, chinese) y otras muy poco (vietnamese, korean con menos de 15 registros). Esto puede afectar al modelo cuando el ICP se concentra en cocinas poco representadas. Se gestiona ampliando el conjunto de seeds en esos casos o documentando la limitación.

**Baseline como espejismo.**
Como se ha demostrado en §6.4, un baseline binario con reglas alineadas con las características de los seeds puede parecer superior en métricas de recall pero no aporta valor operativo (no rankea, no descubre). Este hallazgo es en sí mismo una contribución del proyecto: obliga a mirar las métricas con criterio y no confiar ciegamente en un número.

## 8. Líneas futuras de trabajo

- **Ampliación del conjunto de seeds** a 20-50 restaurantes por ICP para validar cuantitativamente el aporte del embedding y estabilizar las métricas LOO.
- **Feedback loop de conversión real**: capturar qué leads del ranking acaban siendo contactados y qué porcentaje convierten, para reentrenar los pesos del vector o entrenar un modelo supervisado con datos reales.
- **SHAP sobre XGBoost** para explicabilidad avanzada (actualmente resuelta mediante descomposición por bloques del score coseno, más sencilla y suficiente para el MVP).
- **Ampliación geográfica** más allá del distrito Centro de Madrid, replicando el pipeline en otros distritos o ciudades.
- **Otros verticales B2B** (farmacéuticas industriales, laboratorios, clínicas) donde el mismo patrón de few-shot learning aplicaría.
