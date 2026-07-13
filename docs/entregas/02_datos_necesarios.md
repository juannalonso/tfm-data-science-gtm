# Entrega 2 — Selección de idea de proyecto y análisis de datos necesarios

- **Autor:** Juan Alonso
- **Tutor:** Julio Valero
- **Fecha:** 12-07-26

---

## 1. Idea seleccionada

**Título:** Generador inteligente de listas B2B locales con cualificación ML para outbound

### Problema que resuelve

En cualquier estrategia de Go-to-Market B2B basada en outbound (cold email, LinkedIn, secuencias multicanal), uno de los principales cuellos de botella es la **construcción de listas de empresas-objetivo bien cualificadas**. Los equipos comerciales y las agencias de GTM invierten un tiempo desproporcionado en compilar y depurar universos de empresas por sector, geografía y perfil, y en decidir cuáles priorizar. Las herramientas SaaS del mercado (Apollo, ZoomInfo, Google Maps Scraper) resuelven la parte de recolección pero devuelven listas planas, ruidosas, con muchos duplicados y sin criterio de priorización. El resultado es que muchas campañas se lanzan sobre universos mal cualificados, con bajas tasas de respuesta y con un coste de oportunidad elevado, tanto para pequeñas empresas que arrancan su motor comercial como para agencias que gestionan varios clientes en paralelo.

### Solución planteada

El proyecto propone construir un sistema que, a partir de **fuentes de datos abiertas y públicas** (INE, datos.gob.es, OpenStreetMap, registros mercantiles), sea capaz de generar listas de empresas-objetivo filtradas por criterios geográficos (municipio, provincia, código postal), sectoriales (CNAE, categoría de actividad) y de tamaño (número de empleados, facturación cuando esté disponible), y de **cualificarlas automáticamente mediante técnicas de Machine Learning**. En concreto: un módulo de **entity resolution** para deduplicar entidades procedentes de fuentes heterogéneas, un módulo de **enriquecimiento** que asocie a cada empresa información complementaria (web, tecnología detectada, indicios de actividad), y un **clasificador ICP** de tipo *few-shot* que, dado un pequeño conjunto de clientes-ejemplo del usuario, aprenda a puntuar y priorizar el resto del universo según similitud.

### MVP del proyecto final

El producto mínimo viable será una aplicación interactiva en **Streamlit** en la que el usuario podrá: (1) definir los filtros del universo (por ejemplo: fabricantes de plástico en Cataluña con entre 20 y 100 empleados), (2) subir un pequeño CSV con clientes-referencia que representen su ICP, y (3) obtener como resultado una lista descargable en CSV, deduplicada, enriquecida y ordenada por *score* de similitud al ICP, con un desglose explicable de por qué cada empresa ha sido puntuada así (usando **SHAP**). La demo incluirá también métricas de evaluación del modelo (*precision@k*, *lift sobre baseline aleatorio*) y una visualización geoespacial de la distribución de las empresas priorizadas sobre un mapa.

---

## 2. Datos necesarios

### Variables o campos requeridos

Para cada empresa del universo de trabajo, se requieren los siguientes campos:

- **Identificación**: nombre o razón social, NIF o identificador único cuando esté disponible.
- **Localización**: dirección, municipio, provincia, código postal, coordenadas geográficas (latitud/longitud).
- **Clasificación sectorial**: código CNAE (Clasificación Nacional de Actividades Económicas) o categoría equivalente (por ejemplo, la etiqueta `amenity`/`shop`/`office` de OpenStreetMap).
- **Tamaño**: número de empleados en tramos (según DIRCE del INE) o valor exacto cuando esté disponible.
- **Datos de contacto**: web corporativa, teléfono (cuando aplique).
- **Antigüedad y estado**: fecha de constitución, situación activa/inactiva.

### Granularidad

La granularidad de trabajo es a **nivel de empresa individual** (una fila = una empresa). Los agregados (por municipio, provincia o CNAE) se calculan on-the-fly desde ese nivel base. Esta granularidad es imprescindible porque el objetivo del sistema es puntuar y priorizar cada empresa individualmente para su uso en campañas outbound.

### Profundidad histórica

El proyecto no requiere una serie temporal profunda porque el caso de uso es **de estado actual**: qué empresas existen ahora mismo con determinado perfil. Se trabajará con la **última versión disponible** de cada fuente (DIRCE actualizado anualmente por el INE, exports actuales de OpenStreetMap, últimas altas del Registro Mercantil). Sí se contempla mantener una copia con fecha de descarga para trazabilidad y para poder detectar cambios entre versiones en el futuro.

### Volumen aproximado

Se estima trabajar con un universo de entre **50.000 y 500.000 empresas** dependiendo del ámbito geográfico y sectorial escogido para las pruebas. Como referencia, el DIRCE del INE contiene aproximadamente 3,4 millones de empresas activas en España, y OpenStreetMap cubre varios cientos de miles de establecimientos con etiquetas de negocio en el país. Un subconjunto de 100.000 empresas es suficiente para entrenar y evaluar los modelos con solidez, y cabe cómodamente en memoria para el desarrollo.

### Datos imprescindibles vs deseables

**Imprescindibles:**

- Nombre de la empresa.
- Localización geográfica (mínimo municipio o coordenadas).
- Clasificación sectorial (CNAE o equivalente).
- Tramo de tamaño (número de empleados).

**Deseables pero no obligatorios:**

- Web corporativa (permite enriquecimiento posterior con tech stack, contenido web, etc.).
- Datos económicos (facturación, resultado): útiles para segmentación fina pero no accesibles de forma abierta para todas las empresas.
- Fecha de constitución y trayectoria histórica.
- Datos de decisores clave (nombres, cargos): quedan fuera del alcance por consideraciones de privacidad (ver sección 4).

---

## 3. Fuentes de datos previstas

### Fuentes concretas

| Fuente | Tipo de dato | Formato | Enlace | Histórico |
|--------|--------------|---------|--------|-----------|
| **INE — DIRCE** (Directorio Central de Empresas) | Universo estadístico de empresas por CNAE, provincia y tramo de empleo | CSV / API | https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736160707 | Actualización anual, series desde 1999 |
| **datos.gob.es** | Catálogo de datasets abiertos del sector público (padrones de actividades económicas, licencias municipales, censos locales) | CSV, JSON, XLSX | https://datos.gob.es | Variable según dataset |
| **OpenStreetMap** (via Overpass API o Geofabrik) | Establecimientos con etiquetas geolocalizadas de negocio | JSON / OSM / PBF | https://overpass-turbo.eu · https://download.geofabrik.de | Snapshot semanal, histórico disponible |
| **Registro Mercantil / BORME** | Actos mercantiles (constituciones, ceses, cambios de administradores) | XML (via API BORME) / PDF | https://www.boe.es/diario_borme/ | Diario, desde 2009 |
| **Datos abiertos municipales** (ayuntamientos de grandes ciudades) | Licencias de actividad, terrazas, censos comerciales locales | CSV, GeoJSON | Portales de datos abiertos por ayuntamiento | Variable |

### Accesibilidad

Todas las fuentes previstas son **públicas, abiertas y accesibles sin coste**. No requieren autenticación de pago ni permisos especiales. INE y datos.gob.es publican bajo licencias abiertas compatibles con uso académico y comercial. OpenStreetMap se distribuye bajo licencia ODbL (Open Database License), que exige atribución y mantiene el carácter abierto de los trabajos derivados. El BORME es de acceso público por su naturaleza legal.

### Formato esperado

Los formatos principales son **CSV** (para INE y datasets públicos tabulares), **JSON/OSM/PBF** (para OpenStreetMap) y **XML** (para el BORME). El proyecto contempla una capa de ingesta que normaliza todos los orígenes a un esquema tabular común en Parquet, sobre el cual trabajan las capas posteriores.

### Estabilidad y mantenimiento de las fuentes

Las fuentes son de **carácter institucional o comunitario** y se consideran estables y mantenidas:

- El INE es un organismo público con obligación legal de publicación periódica.
- datos.gob.es es el portal oficial de datos abiertos del Gobierno de España.
- OpenStreetMap tiene una comunidad activa y global con exports semanales.
- El BORME es un boletín oficial diario.

No dependemos de scrapers frágiles a la web ni de APIs privadas sujetas a cambios comerciales o de precio.

### Riesgos detectados

- **Datos incompletos:** el DIRCE ofrece agregados por tramos (no el número exacto de empleados por empresa). La resolución fina exige cruzar con otras fuentes.
- **Cambios en API/web:** Overpass y los portales municipales pueden reestructurar endpoints; se mitiga cacheando descargas versionadas.
- **Límites de descarga:** Overpass tiene *rate limits* razonables; se trabajará con descargas incrementales por provincia.
- **Falta de actualización:** algunos datasets municipales están desactualizados. Se prioriza INE y OSM como espinas dorsales.
- **Calidad heterogénea:** los nombres de empresa varían entre fuentes ("Acme S.L." vs "ACME SL" vs "Acme Sociedad Limitada"). Este es precisamente uno de los retos técnicos centrales del proyecto (entity resolution).
- **Documentación:** el BORME es voluminoso y su parseo requiere trabajo; se contempla usar librerías existentes como `mbormeparser` o similares.

---

## 4. Consideraciones de privacidad y protección de datos

### Información personal identificable (PII)

El proyecto trabaja con **datos corporativos**, no con datos personales de individuos. Las empresas son personas jurídicas y su información básica (nombre, dirección, CIF, sector) no está sujeta al RGPD en el mismo sentido que los datos de personas físicas. Aun así, hay dos zonas grises que se manejan con cuidado: los nombres de administradores publicados en el BORME (personas físicas) y los establecimientos autónomos donde la razón social puede coincidir con el nombre de una persona.

### Necesidad de anonimización o agregación

Para minimizar riesgos, el proyecto **excluye del pipeline** los campos de nombre de administrador provenientes del BORME. Solo se conserva el hecho societario (constitución, cese) a nivel de empresa. En el caso de autónomos, se filtra por CNAE para trabajar exclusivamente con actividades típicamente ejercidas bajo denominación comercial distinta del nombre personal.

### Uso seguro en un proyecto académico

Los datos utilizados son **públicos por naturaleza y de libre acceso** para cualquier ciudadano. Su uso en un TFM público en GitHub es plenamente compatible con las licencias de origen, cumpliendo con la obligación de atribución en el caso de OpenStreetMap.

### Riesgos éticos o legales

- **RGPD:** minimizado al trabajar con datos corporativos y excluir información de personas físicas identificables.
- **Uso comercial derivado:** las licencias de INE, datos.gob.es y OSM permiten uso comercial. El proyecto se enmarca en un contexto académico, pero no se está infringiendo ninguna restricción de origen.
- **Sesgos:** el DIRCE cubre bien empresas registradas pero infrarrepresenta economía informal y pequeñas actividades no censadas. Este sesgo se documentará como limitación del proyecto.
- **Uso responsable del output:** el sistema puede facilitar campañas de outbound; se recomienda su uso con cumplimiento de RGPD para el envío efectivo de comunicaciones (base legítima, opt-out, etc.), pero el envío queda fuera del alcance del TFM.

### Datos evitados por motivos de privacidad

Se ha decidido **no incluir**:

- Datos de contacto personal de decisores (emails o teléfonos individuales de personas identificables).
- Perfiles de LinkedIn ni scraping de redes sociales.
- Cualquier fuente que requiera vulnerar términos de servicio de terceros.

Estas exclusiones son deliberadas: mantener el proyecto en el terreno seguro de las fuentes institucionales abiertas.

---

## 5. Viabilidad inicial del proyecto

### ¿Es viable obtener los datos necesarios?

**Sí**, con alta seguridad. Todas las fuentes previstas son públicas, están documentadas y son accesibles sin coste ni permisos especiales. Se han identificado enlaces concretos para cada una y se conoce el formato de descarga. No existen dependencias de terceros comerciales que puedan bloquear el proyecto.

### ¿La información disponible tiene suficiente calidad, granularidad y profundidad histórica?

**Sí** para el caso de uso planteado. La granularidad a nivel de empresa individual es suficiente y está disponible en OpenStreetMap y en los portales municipales. El DIRCE aporta el marco de referencia estadístico. La profundidad histórica no es crítica porque el problema es de estado actual, no de forecasting.

La única limitación real es que **el número exacto de empleados por empresa** no siempre está disponible en fuentes abiertas (el DIRCE lo publica por tramos). Esto no invalida el proyecto pero obliga a trabajar con tramos como *proxy*.

### ¿La idea puede desarrollarse de forma realista durante el curso?

**Sí**. El alcance está bien delimitado. Los módulos principales (ingesta, entity resolution, clasificador ICP, dashboard) son abordables individualmente con las técnicas cubiertas en el máster (Python avanzado, estadística, ML, visualización). No se requiere infraestructura pesada: todo el pipeline puede ejecutarse en un portátil.

### Parte del proyecto más arriesgada en este momento

El principal riesgo técnico es el **entity resolution** — la deduplicación de empresas que aparecen con nombres, direcciones o identificadores ligeramente distintos en varias fuentes. Es un problema clásico de NLP y *string matching* con soluciones bien conocidas (embeddings, Levenshtein, Jaro-Winkler, aprendizaje supervisado sobre pares etiquetados), pero conseguir una precisión y *recall* razonables sobre un universo grande y ruidoso es donde se juega buena parte del valor del proyecto.

El riesgo académico principal es garantizar que el **clasificador ICP** aporta un *lift* medible sobre un baseline aleatorio, dado que el ICP se aprende a partir de muy pocas muestras positivas (*few-shot*). Se abordará con técnicas de *similarity learning* y *positive-unlabeled learning*.

### Alternativa si la fuente principal de datos no funciona

Si el INE cambiara su política de publicación o si OpenStreetMap no cubriera con suficiente detalle un sector específico, existen varias alternativas:

- **Escenario A (fuente institucional falla):** replegarse a **OpenStreetMap como espina dorsal única**, complementado con los portales municipales de ayuntamientos con datos abiertos activos (Madrid, Barcelona, Valencia, Zaragoza, entre otros).
- **Escenario B (OSM insuficiente en un sector):** utilizar **Overture Maps Foundation** (proyecto abierto de Amazon, Meta, Microsoft y TomTom con datos de puntos de interés bajo licencia CDLA) como alternativa moderna a OSM.
- **Escenario C (necesidad de más volumen):** usar **datasets de HuggingFace** con listas empresariales anonimizadas para pruebas y validación adicional.

En cualquiera de los escenarios, el proyecto puede continuar sin cambios en su arquitectura ni en su MVP.

---
