# TFM — Cualificación de leads B2B para outbound mediante Machine Learning

**Máster en Data Science e Inteligencia Artificial**

- **Autor:** Juan Alonso
- **Curso académico:** 2025-2026
- **Tutor:** Julio Valero

---

## Contexto del proyecto

Trabajo Fin de Máster centrado en la aplicación de técnicas de Data Science, Machine Learning e Inteligencia Artificial al problema de la **cualificación y priorización de leads B2B** para estrategias de outbound (cold email, LinkedIn, secuencias multicanal).

El sistema, a partir de un pequeño conjunto de empresas-referencia aportadas por el usuario (su perfil ideal de cliente o ICP), aprende automáticamente qué caracteriza a esos clientes mediante técnicas de *few-shot learning*, prioriza el universo restante de empresas por similitud, y explica cada puntuación de forma individualizada mediante SHAP.

El proyecto se desarrolla en el marco de las prácticas del autor como **GTM Engineer** en una agencia/consultora Go-to-Market especializada en proyectos complejos de estrategia y ventas asistidos por IA.

## Alcance actual

- **Sector de aplicación:** restauración (restaurantes, bares-restaurante y restaurantes de comida rápida).
- **Territorio piloto de validación:** distrito Centro de Madrid, con arquitectura extensible al resto de España.
- **Estrategia de enriquecimiento:** híbrida, combinando fuentes de datos abiertas (OpenStreetMap, censo del Ayuntamiento de Madrid, INE) con un agente de IA generativa para variables cualitativas.

## Estructura del repositorio

```
.
├── README.md
├── docs/
│   ├── entregas/
│   │   ├── 01_ideas_producto.md      — Ideas iniciales de producto
│   │   ├── 02_datos_necesarios.md    — Idea seleccionada y análisis de datos
│   │   └── 03_modelo_datos.md        — Modelo de datos y capa gold
│   └── apuntes/
│       └── apuntes_tfm.md            — Notas de trabajo y preparación de defensa
├── data/
│   ├── raw/                          — Datos originales de las fuentes
│   ├── processed/                    — Datos limpios y cruzados
│   └── gold/                         — Dataset final para el modelo
├── notebooks/                        — Análisis exploratorio y pruebas
└── src/                              — Código fuente del proyecto
```

## Entregas

Documentación de las entregas incrementales del proyecto:

- **[Entrega 1 — Ideas de producto](docs/entregas/01_ideas_producto.md):** exploración inicial de ideas de producto de Data Science / IA.
- **[Entrega 2 — Selección de idea y datos necesarios](docs/entregas/02_datos_necesarios.md):** idea seleccionada, análisis de los datos requeridos, fuentes previstas, consideraciones de privacidad y valoración de viabilidad.
- **[Entrega 3 — Modelo de datos y capa gold](docs/entregas/03_modelo_datos.md):** tecnología de almacenamiento, estructura de capas, definición de la capa gold, relaciones entre datos, diccionario, problemas de calidad y riesgos.

## Fuentes de datos

| Fuente | Aportación | Acceso |
|--------|-----------|--------|
| **OpenStreetMap** (API Overpass) | Universo de restaurantes: nombre, coordenadas, dirección, cocina, servicios | Abierto |
| **Censo de Locales del Ayuntamiento de Madrid** | Clasificación oficial, barrio, sección censal, situación administrativa | Abierto |
| **INE** (por sección censal) | Variables socioeconómicas del entorno (renta, población, edad) | Abierto |
| **Agente LLM (Gemini)** | Variables cualitativas: posicionamiento, clientela, ambiente | API con tier gratuito |

## Stack tecnológico

- **Lenguaje:** Python
- **Análisis y modelado:** pandas, numpy, scikit-learn, XGBoost / LightGBM
- **Datos geográficos:** geopandas, shapely, pyproj
- **Explicabilidad:** SHAP
- **Enriquecimiento cualitativo:** API de Gemini
- **Visualización y dashboard:** Streamlit
- **Almacenamiento:** Parquet (principal) + CSV (complementario)

## Notas sobre datos y privacidad

Este repositorio es público. Ningún dato interno de la empresa de prácticas ni de sus clientes se versiona en él. Los datos utilizados proceden de fuentes abiertas y públicas (OpenStreetMap, datos.madrid.es, INE), trabajando exclusivamente con información corporativa de establecimientos, sin datos personales identificables.

---

*Repositorio en desarrollo activo.*
