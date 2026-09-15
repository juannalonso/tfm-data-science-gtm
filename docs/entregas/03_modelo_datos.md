# Entrega 3 — Diseño del modelo de datos y capa gold del proyecto

- **Autor:** Juan Alonso
- **Tutor:** Julio Valero

---

## 1. Resumen de la idea y datos del proyecto

**Problema y solución.** El proyecto aborda el cuello de botella recurrente en las estrategias de outbound B2B: la construcción y priorización de listas de empresas-objetivo cualificadas. Las herramientas comerciales del mercado (Apollo, ZoomInfo, Google Maps Scrapers) resuelven bien la recolección, pero devuelven listas planas, con duplicados y sin criterio de priorización. La solución propuesta es un sistema que, a partir de un pequeño conjunto de clientes-referencia aportados por el usuario (10 a 20 empresas), aprende automáticamente el perfil ideal de cliente (ICP) mediante técnicas de Machine Learning y prioriza el universo restante por similitud, con explicación individualizada de cada score.

**Refinamiento del alcance tras la Entrega 2.** Siguiendo la recomendación del profesor de acotar el MVP a un único sector y territorio para validar empíricamente la viabilidad de las fuentes antes de diseñar el clasificador, el proyecto se ha concretado en:

- **Sector de aplicación:** restauración (restaurantes propiamente dichos, incluyendo bares-restaurante y restaurantes de comida rápida).
- **Territorio piloto de validación empírica:** distrito Centro de Madrid, con arquitectura extensible al resto de España como trabajo futuro.
- **Estrategia de enriquecimiento:** híbrida, combinando fuentes estructuradas (OpenStreetMap y datos abiertos del Ayuntamiento de Madrid) con un agente de IA generativa para variables cualitativas no disponibles en fuentes estructuradas.

**Fuentes y aportación de cada una.** El sistema se construye sobre cuatro fuentes complementarias:

- **OpenStreetMap (vía API Overpass):** universo de trabajo principal. Aporta identificación del establecimiento, coordenadas, dirección, tipo de cocina, horario, presencia web, características de accesibilidad y servicios (delivery, takeaway, outdoor_seating). Volumen validado: 1.688 restaurantes en el distrito Centro de Madrid.
- **Censo de Locales y Actividades del Ayuntamiento de Madrid:** capa de enriquecimiento oficial. Aporta clasificación normalizada del epígrafe (RESTAURANTE, BAR RESTAURANTE, comida rápida), barrio, sección censal, situación administrativa y trazabilidad con datos socioeconómicos posteriores. Volumen: 225.628 locales totales, 1.626 restaurantes filtrados en el Centro tras aplicar los criterios del proyecto.
- **INE — Datos por sección censal:** capa de enriquecimiento socioeconómico. Permite asociar a cada restaurante variables del entorno (renta, población, edad media) a través de la sección censal aportada por el censo municipal. Uso planificado para la fase de modelado.
- **Agente LLM (Gemini de Google):** capa de enriquecimiento cualitativo. Aporta variables no disponibles en fuentes estructuradas: tipo de clientela objetivo, posicionamiento (premium/medio/low-cost), ambiente, presencia digital sintetizada, valoración global.

---

## 2. Tecnología o formato de almacenamiento elegido

La elección tecnológica del proyecto responde a tres criterios: **reproducibilidad** (que el pipeline sea ejecutable por cualquiera con el repositorio), **eficiencia** (que los datasets quepan cómodamente en memoria y se lean rápido) y **simplicidad** (que no requiera infraestructura compleja).

Sobre esa base, se ha adoptado una combinación de dos formatos:

**Parquet como formato principal.** Todos los datasets consolidados (raw estabilizada, processed, gold) se almacenan como ficheros Parquet. La justificación es doble: por un lado, Parquet es un formato columnar comprimido que reduce el espacio significativamente respecto a CSV (en las pruebas realizadas sobre el dataset de restaurantes, Parquet ocupa 110,7 KB frente a los 197,7 KB del CSV equivalente); por otro, preserva los tipos de datos, lo que evita reprocesados innecesarios y errores de conversión entre sesiones.

**CSV como formato complementario.** Cada tabla en Parquet se guarda también en CSV. Esto permite que los datasets sean inspeccionables directamente desde cualquier editor o herramienta ofimática (útil para validación manual y para la revisión académica del profesor), sin que ello incremente sustancialmente el tamaño del repositorio dada la escala del proyecto.

Se ha descartado el uso de una base de datos relacional (SQLite, PostgreSQL) por dos razones: la escala del proyecto (unas decenas de miles de registros como máximo) no lo justifica, y añadiría complejidad de configuración innecesaria en un proyecto que debe poder ejecutarse de forma reproducible por cualquier evaluador. Se ha descartado también JSON y Excel por eficiencia inferior a Parquet en el caso de datos tabulares estructurados.

---

## 3. Estructura de capas de datos

El proyecto adopta la estructura estándar en tres capas raw → processed → gold, tal como sugiere el enunciado. Esta arquitectura es apropiada al proyecto porque separa claramente responsabilidades y permite trazar de dónde viene cada campo del dataset final.

```
data/
├── raw/          — Datos originales tal como se descargan de la fuente
├── processed/    — Datos limpios, normalizados y con cruces iniciales
└── gold/         — Dataset final preparado para consumo del modelo ML
```

**Capa raw.** Contiene las descargas originales sin apenas modificación. Se guarda tanto el formato de descarga original de cada fuente como una versión estabilizada en Parquet. Los ficheros actualmente presentes son:

- `restaurantes_centro_madrid_osm.parquet` y `.csv` (1.688 registros): descarga inicial de OpenStreetMap vía API Overpass, con las etiquetas relevantes extraídas.
- `200085-5-censo-locales.csv` (225.628 registros): fichero original del Ayuntamiento de Madrid en su formato de publicación oficial.

**Capa processed.** Contiene los datasets con las transformaciones intermedias aplicadas: normalización de nombres de calle y números, filtrado por epígrafes de restauración, filtrado geográfico al distrito Centro, filtrado por situación administrativa (Abierto), conversión de coordenadas UTM a lat/lon, y cruces iniciales entre OSM y el censo. Actualmente contiene:

- `censo_restaurantes_madrid.parquet` (1.626 registros): censo filtrado a los epígrafes RESTAURANTE, BAR RESTAURANTE y RESTAURANTES DE COMIDA RÁPIDA en situación "Abierto" en el distrito Centro.
- `restaurantes_enriquecidos_direccion.parquet` (818 registros): resultado del cruce por dirección normalizada entre OSM y el censo.
- `restaurantes_enriquecidos_proximidad.parquet` (737 registros): resultado del cruce por proximidad geográfica (umbral 30 metros) para los restaurantes de OSM que no cruzaron por dirección.

**Capa gold.** Contiene el dataset final consolidado que alimenta directamente al modelo. Se detalla en la sección 4 y actualmente está pendiente de construcción como último paso del pipeline de datos (previsto para la siguiente iteración del proyecto). Su diseño está ya especificado y validado empíricamente.

---

## 4. Definición de la capa gold

La capa gold constituye el contrato de datos entre la fase de ingeniería de datos y la fase de modelado. Contiene el dataset final consolidado sobre el cual se aplicarán los algoritmos de cualificación de leads. Su diseño es el resultado del análisis empírico realizado sobre las fuentes disponibles y responde a las necesidades específicas del modelo predictivo posterior.

### 4.1 Estructura general

La capa gold del proyecto se compone de **una tabla principal única** consolidada por restaurante, denominada `gold_restaurantes_madrid.parquet`. Esta decisión responde a que el modelo de cualificación opera sobre entidades individuales (restaurantes) y no requiere estructura relacional en su input.

- **Nombre del dataset:** `gold_restaurantes_madrid`.
- **Granularidad:** un registro por restaurante identificado.
- **Volumen esperado:** aproximadamente 1.688 registros para el territorio piloto (distrito Centro de Madrid), escalable al orden de decenas de miles al ampliar a la ciudad completa o a otras ciudades.
- **Formato de almacenamiento:** Parquet como principal, CSV complementario.
- **Uso previsto:** input directo del pipeline de Machine Learning (few-shot ICP + explicabilidad SHAP + evaluación cuantitativa).

### 4.2 Clave primaria

La clave primaria de la tabla es el campo `restaurante_id`, un identificador sintético generado por el pipeline. Se opta por un identificador propio en lugar de reutilizar el `osm_id` de OpenStreetMap por dos razones: primero, el `osm_id` puede cambiar si un editor de OSM elimina y recrea un nodo (algo que ocurre ocasionalmente); segundo, un identificador propio permite trazar restaurantes que en el futuro provengan de fuentes distintas a OSM sin depender de la existencia de un `osm_id` para todos.

El `restaurante_id` se construye como un hash reproducible basado en la combinación de `nombre normalizado + código postal + coordenadas redondeadas`, lo que garantiza que el mismo restaurante genera siempre el mismo identificador entre ejecuciones del pipeline.

### 4.3 Campos y organización lógica

Los campos de la capa gold se organizan en siete bloques temáticos, cada uno correspondiente a un origen o naturaleza de la información. Esta organización facilita la comprensión del dataset y permite una implementación modular del pipeline.

**Bloque 1 — Identificación y localización básica.**

Contiene los campos que identifican y ubican geográficamente cada restaurante. Todos son de obligado cumplimiento salvo casos residuales de restaurantes con información incompleta.

- `restaurante_id` (string, PK): identificador único generado por el pipeline.
- `nombre` (string): nombre comercial del restaurante.
- `nombre_normalizado` (string): nombre en mayúsculas, sin acentos y sin caracteres especiales, utilizado para procesos internos de matching.
- `latitud` (float): coordenada geográfica (WGS84).
- `longitud` (float): coordenada geográfica (WGS84).
- `direccion_completa` (string): dirección completa formateada.
- `codigo_postal` (string): código postal.
- `barrio` (string): barrio administrativo según el censo municipal.
- `distrito` (string): distrito administrativo según el censo municipal.
- `seccion_censal` (string): sección censal, usada para cruce posterior con datos del INE.

**Bloque 2 — Clasificación del establecimiento.**

Contiene la clasificación del tipo de restaurante, combinando la información de OpenStreetMap y del censo oficial del Ayuntamiento.

- `epigrafe_oficial` (categórico): clasificación oficial según el Ayuntamiento (RESTAURANTE, BAR RESTAURANTE, RESTAURANTES DE COMIDA RÁPIDA).
- `cocina_osm` (categórico): tipo de cocina según OpenStreetMap (italiana, japonesa, tapas, mediterránea, etc.) cuando está informado.
- `tipo_acceso` (categórico): tipo de acceso al local (Puerta Calle, Local Interior, etc.) según el censo.

**Bloque 3 — Características operativas.**

Contiene información sobre el funcionamiento del restaurante, útil para caracterizar el modelo de negocio.

- `tiene_web` (bool): true si el restaurante tiene una web propia registrada.
- `url_web` (string, nullable): URL del restaurante cuando existe.
- `tiene_telefono` (bool): true si tiene teléfono registrado.
- `tiene_horario` (bool): true si el horario de apertura está registrado.
- `tiene_terraza` (bool): true si tiene servicio de terraza registrado.
- `ofrece_delivery` (bool): true si ofrece delivery.
- `ofrece_takeaway` (bool): true si ofrece comida para llevar.
- `es_accesible` (bool): true si el establecimiento está declarado accesible.

**Bloque 4 — Enriquecimiento socioeconómico.**

Contiene variables del entorno del restaurante, obtenidas del INE mediante el cruce por sección censal. Estas variables son fundamentales para el modelo, ya que caracterizan el contexto en el que opera el negocio.

- `poblacion_barrio` (int, nullable): población residente del barrio.
- `renta_media_barrio` (float, nullable): renta media disponible por hogar en euros.
- `edad_media_barrio` (float, nullable): edad media de la población del barrio.
- `densidad_poblacion_barrio` (float, nullable): habitantes por km².

**Bloque 5 — Enriquecimiento geográfico derivado.**

Contiene variables calculadas por el pipeline a partir de las coordenadas de los restaurantes. Estas variables capturan el contexto competitivo y geográfico de cada establecimiento.

- `competencia_500m` (int): número de otros restaurantes en un radio de 500 metros.
- `competencia_1km` (int): número de otros restaurantes en un radio de 1 kilómetro.
- `zona_predominante` (categórico): clasificación derivada de la zona (turística, residencial, oficinas, mixta) según la densidad y tipo de puntos de interés del entorno.

**Bloque 6 — Enriquecimiento cualitativo vía LLM.**

Contiene variables cualitativas generadas por un agente de IA generativa (Gemini). Estas variables capturan dimensiones no disponibles en las fuentes estructuradas: percepción del establecimiento, tipo de clientela, posicionamiento comercial.

- `posicionamiento_precio` (categórico): posicionamiento estimado (low_cost, medio, premium, alta_gama).
- `tipo_clientela` (categórico): clientela principal identificada (turista, profesional, residente, mixta).
- `ambiente` (categórico): ambiente predominante (moderno, tradicional, casual, formal, familiar).
- `presencia_digital` (categórico): grado de presencia digital estimado (nula, básica, activa, sofisticada).
- `resumen_llm` (string): resumen textual de una a dos frases generado por el LLM caracterizando al establecimiento.

**Bloque 7 — Metadatos del registro.**

Contiene información sobre la trazabilidad y calidad del enriquecimiento realizado en cada registro. Estos campos permiten al modelo ponderar la fiabilidad de cada fila y al analista auditar el pipeline.

- `fuente_principal` (categórico): fuente principal del registro (siempre OSM en la iteración actual).
- `metodo_enriquecimiento_censo` (categórico): método por el cual se cruzó con el censo (direccion, proximidad_30m, no_matcheado).
- `distancia_match_metros` (float, nullable): distancia geográfica al match del censo cuando el cruce fue por proximidad.
- `enriquecido_llm` (bool): indica si el registro ha sido enriquecido con LLM.
- `fecha_ultima_actualizacion` (datetime): fecha del último procesamiento del registro.

### 4.4 Resumen cuantitativo de campos

El dataset gold en su diseño completo consta de **35 campos** organizados en los siete bloques descritos. Esta cantidad es apropiada para el problema: suficiente para que el modelo de ML disponga de features discriminatorias variadas (numéricas, categóricas, booleanas, geográficas y cualitativas), y contenida para que el análisis siga siendo interpretable y la explicabilidad SHAP tenga sentido práctico.

### 4.5 Uso posterior de la capa gold

La capa gold se ha diseñado como input directo del modelo de cualificación, sin necesidad de transformaciones adicionales relevantes. En la fase de modelado (Entrega 4) sobre esta tabla se aplicarán:

- **Codificación de variables categóricas** (one-hot encoding, target encoding según el algoritmo).
- **Escalado de variables numéricas** cuando el algoritmo lo requiera.
- **Cálculo de embeddings del bloque cualitativo** para técnicas de similarity learning.
- **Segmentación en universo objetivo (todos los restaurantes) y seed set (ejemplos ICP aportados por el usuario).**

El dataset gold no incluye ningún campo objetivo (`target`) porque el problema se aborda como aprendizaje no supervisado con seeds, no como clasificación supervisada. La cualificación se calcula en el momento de aplicar el modelo, en función de los ejemplos ICP que el usuario proporcione.

---

## 5. Relaciones entre datos

El proyecto utiliza varias fuentes de datos que se combinan durante el proceso de construcción de la capa gold. Sin embargo, **el dataset final consumido por el modelo es una única tabla desnormalizada**, no un modelo relacional con varias tablas interconectadas en tiempo de consumo. Esta decisión se justifica a continuación, junto con la descripción de las relaciones que sí existen durante el proceso de construcción.

### 5.1 Relaciones durante la construcción del dataset

Aunque la salida sea una tabla única, durante el pipeline sí se producen cruces entre fuentes. Las relaciones son las siguientes:

**OpenStreetMap ↔ Censo del Ayuntamiento (relación N:1 aproximada).**
Cada restaurante de OpenStreetMap se asocia a un registro del censo municipal. La relación es conceptualmente 1:1 (un restaurante físico corresponde a un local censado), pero en la práctica se comporta como N:1 en algunos casos, ya que un mismo local del censo puede quedar como candidato de match para varios registros de OSM cercanos. El pipeline resuelve estos casos asignando cada restaurante de OSM a su match más cercano y evitando reasignaciones múltiples.

La relación se establece mediante dos claves de cruce complementarias:

- **Clave de dirección normalizada:** combinación de `nombre_calle_normalizado + numero`, aplicada como cruce principal.
- **Clave de proximidad geográfica:** distancia euclídea entre coordenadas, aplicada como cruce secundario para los registros que no cruzan por dirección (umbral de 30 metros).

**Restaurante ↔ Datos del INE (relación N:1).**
Cada restaurante se relaciona con los datos socioeconómicos de su sección censal. Muchos restaurantes comparten la misma sección censal (relación N:1), por lo que varios registros heredan los mismos valores de renta, población y edad media del entorno. La clave de cruce es el campo `seccion_censal`, obtenido del censo municipal.

**Restaurante ↔ Enriquecimiento LLM (relación 1:1).**
Cada restaurante se enriquece individualmente mediante una consulta al agente LLM. La relación es estrictamente 1:1: una consulta por restaurante, un conjunto de variables cualitativas por restaurante.

### 5.2 Justificación de la desnormalización

Se ha optado por consolidar toda la información en una única tabla desnormalizada por tres motivos:

- **El modelo consume entidades individuales.** El algoritmo de cualificación opera restaurante a restaurante; no necesita realizar joins en tiempo de inferencia. Tener la información pre-consolidada simplifica y acelera el modelado.
- **La escala no penaliza la desnormalización.** Con un universo del orden de miles de registros, la redundancia de datos (por ejemplo, la renta del barrio repetida en todos los restaurantes de una misma sección censal) tiene un coste de almacenamiento despreciable.
- **Favorece la reproducibilidad y la auditabilidad.** Una tabla única y autocontenida es más fácil de inspeccionar, versionar y compartir que un conjunto de tablas relacionadas.

### 5.3 Problemas anticipados al combinar fuentes

La combinación de fuentes heterogéneas presenta retos conocidos, detectados empíricamente durante la validación:

- **Inconsistencia en direcciones.** Los nombres de vía difieren entre OSM (con prefijos: "Calle de Fuencarral") y el censo (sin prefijos: "FUENCARRAL"). Se resuelve mediante una función de normalización que elimina prefijos y estandariza el formato. Esta normalización recupera un 65% de los cruces por dirección.
- **Cobertura parcial del cruce.** No todos los restaurantes de OSM tienen dirección completa (74,6%) ni todos cruzan con éxito. El cruce combinado (dirección + proximidad) alcanza un 92,1% de enriquecimiento, dejando un 7,9% de registros sin datos del censo.
- **Discrepancias de clasificación.** Un mismo establecimiento puede clasificarse de forma distinta en cada fuente (por ejemplo, un café histórico etiquetado como "restaurante de comida rápida" en el censo). Se conserva la clasificación de ambas fuentes como campos separados, permitiendo al modelo y al analista contrastar.
- **Coordenadas erróneas en el censo.** Un subconjunto de registros del censo presenta coordenadas inválidas (fuera de Madrid). Se filtran mediante validación de rango geográfico antes del cruce por proximidad.

---

## 6. Diccionario de datos inicial

Se presenta el diccionario de los campos principales de la capa gold. Por concisión se documentan los campos más relevantes para el análisis y el modelo; el diccionario completo de los 35 campos se mantendrá versionado en el repositorio junto al dataset.

| Campo | Descripción | Tipo | Fuente | Obligatorio | Observaciones |
|---|---|---|---|---|---|
| `restaurante_id` | Identificador único sintético | string | Pipeline | Sí | Hash reproducible de nombre + CP + coordenadas |
| `nombre` | Nombre comercial | string | OSM | Sí | — |
| `latitud` | Coordenada geográfica | float | OSM | Sí | Sistema WGS84 |
| `longitud` | Coordenada geográfica | float | OSM | Sí | Sistema WGS84 |
| `codigo_postal` | Código postal | string | OSM / Censo | No | Cobertura ~69% en OSM, completado con censo |
| `barrio` | Barrio administrativo | string | Censo | No | Presente en registros enriquecidos |
| `seccion_censal` | Sección censal | string | Censo | No | Clave de cruce con INE |
| `epigrafe_oficial` | Clasificación oficial del local | categórico | Censo | No | RESTAURANTE / BAR RESTAURANTE / COMIDA RÁPIDA |
| `cocina_osm` | Tipo de cocina | categórico | OSM | No | Cobertura ~54% de restaurantes |
| `tiene_web` | Presencia de web propia | bool | OSM | Sí | Derivado de la existencia de URL |
| `tiene_terraza` | Servicio de terraza | bool | OSM / Censo | Sí | — |
| `renta_media_barrio` | Renta media del barrio | float | INE | No | Heredada por sección censal |
| `poblacion_barrio` | Población del barrio | int | INE | No | Heredada por sección censal |
| `competencia_500m` | Restaurantes en 500m | int | Pipeline | Sí | Calculado sobre coordenadas |
| `posicionamiento_precio` | Posicionamiento estimado | categórico | LLM | No | low_cost / medio / premium / alta_gama |
| `tipo_clientela` | Clientela principal | categórico | LLM | No | turista / profesional / residente / mixta |
| `ambiente` | Ambiente predominante | categórico | LLM | No | moderno / tradicional / casual / formal / familiar |
| `metodo_enriquecimiento_censo` | Método de cruce con censo | categórico | Pipeline | Sí | direccion / proximidad_30m / no_matcheado |

---

## 7. Problemas de calidad esperados

A partir de la exploración empírica de las fuentes, se han identificado los siguientes problemas de calidad concretos, aterrizados al caso del proyecto:

**Valores nulos en campos de OpenStreetMap.** La cobertura de campos en OSM es desigual. Sobre los 1.688 restaurantes del distrito Centro se ha medido: nombre 98,6%, dirección 78,8%, código postal 69,1%, teléfono 45,6%, web 41,8% y tipo de cocina 54,1%. Los campos de contacto y cocina, por tanto, presentan una proporción relevante de nulos que el pipeline debe gestionar.

**Registros sin enriquecimiento del censo.** Un 7,9% de los restaurantes de OSM no logra cruzarse con el censo municipal ni por dirección ni por proximidad. Estos registros carecen de barrio, sección censal y epígrafe oficial, lo que impide su enriquecimiento socioeconómico posterior vía INE.

**Coordenadas inválidas en el censo.** Un subconjunto de registros del censo municipal presenta coordenadas fuera del rango geográfico de Madrid (latitud 0, longitud en rango portugués). Se han detectado y filtrado 91 registros de este tipo en el subconjunto de restaurantes del Centro.

**Inconsistencia en nombres de vía entre fuentes.** OSM incluye prefijos de tipo de vía ("Calle de", "Plaza de") mientras que el censo almacena solo el nombre. Sin normalización, el cruce directo por dirección fallaría casi por completo.

**Discrepancias de clasificación entre fuentes.** Un mismo establecimiento puede tener clasificaciones distintas en OSM y en el censo. Se ha observado, por ejemplo, un café histórico clasificado como "restaurante de comida rápida" en el censo. No es un error a corregir, sino una discrepancia a conservar como información.

**Duplicidad potencial de establecimientos.** El mismo restaurante puede aparecer en OSM con ligeras variaciones (nodo y way, o dos nodos cercanos). Requiere deduplicación en la fase de entity resolution.

**Variabilidad e imprecisión del enriquecimiento LLM.** Las variables generadas por el agente LLM pueden presentar inconsistencias entre ejecuciones y, ocasionalmente, información no verificable (alucinaciones). Es un riesgo intrínseco de la fuente que debe gestionarse.

**Desactualización relativa de OSM.** Al ser una fuente colaborativa, OSM puede contener restaurantes ya cerrados o no reflejar aperturas recientes. La comparación con el censo (1.688 en OSM vs 1.626 en censo, 96,2% de cobertura) sugiere una desviación moderada y asumible.

---

## 8. Decisiones de limpieza y transformación previstas

Se definen las siguientes decisiones iniciales de preparación de datos. Son hipótesis de trabajo que podrán ajustarse durante el desarrollo:

**Tratamiento de valores nulos.**
- Campos booleanos derivados de existencia (`tiene_web`, `tiene_telefono`, `tiene_terraza`): un nulo se interpreta como ausencia (valor `false`), no como dato perdido.
- Campos categóricos (`cocina_osm`, `epigrafe_oficial`): los nulos se codifican como categoría explícita "desconocido" en lugar de eliminarse, para no perder registros.
- Campos socioeconómicos del INE (`renta_media_barrio`, etc.): los nulos derivados de registros sin sección censal se marcarán para posible imputación por la mediana del distrito, evaluando su impacto en el modelo.

**Gestión de duplicados.** Se aplicará deduplicación en la fase de entity resolution combinando similitud de nombre (embeddings + distancia de cadenas) y proximidad geográfica. Dos registros a menos de una distancia umbral con nombres suficientemente similares se consolidarán en uno.

**Normalización de texto.** Los nombres de vía se normalizan eliminando prefijos de tipo de vía y estandarizando a mayúsculas sin acentos. Los nombres de restaurante se normalizan para el matching, conservando siempre el nombre original para presentación.

**Normalización de números de portal.** Los números de portal del censo (formato "000005") se limpian extrayendo la parte numérica ("5"), descartando sufijos de letra o bis cuando impiden el cruce.

**Conversión de coordenadas.** Las coordenadas UTM del censo (EPSG:25830) se convierten a latitud/longitud WGS84 (EPSG:4326) para homogeneizar con OSM antes de cualquier cruce geográfico.

**Variables derivadas a construir.**
- Variables de competencia (`competencia_500m`, `competencia_1km`) calculadas a partir de las coordenadas de todo el universo.
- Variable de zona predominante derivada del entorno.
- Variables cualitativas del bloque LLM.

**Criterios de validez de un registro.** Se considerará válido para el modelo todo restaurante que disponga, como mínimo, de identificador, nombre y coordenadas geográficas. Los registros sin coordenadas (que impiden el enriquecimiento geográfico) se marcarán como incompletos y se evaluará su exclusión.

**Datos que se descartan.** Se descartan del universo los locales del censo que no son restaurantes (bares sin cocina, cafeterías puras, otros epígrafes de hostelería), los locales en situación distinta de "Abierto", y los registros con coordenadas inválidas no recuperables.

---

## 9. Riesgos del modelo de datos

**Parte del modelo de datos más clara.** La capa de ingesta y la estructura de capas (raw → processed → gold) están validadas empíricamente y no presentan incertidumbre relevante. La obtención de restaurantes desde OSM y su enriquecimiento con el censo municipal funcionan con tasas de cobertura medidas y satisfactorias (92,1% de enriquecimiento).

**Parte que genera más incertidumbre.** El enriquecimiento cualitativo vía LLM es el componente con mayor incertidumbre, por tres motivos: su reproducibilidad es menor que la de las fuentes estructuradas, su calidad debe validarse (riesgo de alucinaciones), y su coste, aunque bajo, es variable. Se mitigará fijando temperatura cero, cacheando las respuestas y validando una muestra manualmente.

**Fuente que puede dar más problemas.** El cruce con el INE por sección censal es el eslabón menos probado hasta la fecha. Aunque la sección censal está disponible en el censo municipal, la obtención y cruce de los datos socioeconómicos del INE a nivel de sección censal requiere trabajo adicional y puede presentar problemas de correspondencia de códigos entre años.

**Qué ocurriría si no se puede construir la capa gold como se ha definido.** El diseño es modular: si un bloque de enriquecimiento fallara, el resto de la capa gold seguiría siendo funcional. El modelo puede entrenarse con un subconjunto de features. En el peor caso, con solo los bloques 1, 2, 3 y 5 (identificación, clasificación, operativa y geográfico derivado), el dataset ya sería suficiente para un modelo de cualificación básico.

**Alternativa de simplificación.** Si el modelo de datos completo resultara inviable en el tiempo disponible, la simplificación prevista es la siguiente: prescindir del bloque socioeconómico del INE (el más costoso de integrar) y del enriquecimiento LLM, conservando el enriquecimiento con el censo municipal, que ya aporta barrio, distrito, epígrafe oficial y sección censal, más las variables derivadas geográficamente. Esta versión reducida es plenamente funcional y está ya prácticamente construida a día de hoy.

---
