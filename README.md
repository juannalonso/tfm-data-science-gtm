# TFM — Cualificador de leads B2B para outbound

Sistema de cualificación de leads B2B mediante técnicas de *few-shot learning* y enriquecimiento con IA generativa, aplicado al sector de la restauración en el distrito Centro de Madrid.

- **Autor:** Juan Alonso
- **Tutor:** Julio Valero
- **Máster:** Data Science e Inteligencia Artificial
- **Curso:** 2025-2026

---

## Qué es

A partir de **3 restaurantes-ejemplo** que representan el perfil de cliente ideal (ICP) de un comercial B2B, el sistema aprende automáticamente el patrón que los caracteriza y devuelve un **ranking de restaurantes similares** en el universo de 1.688 restaurantes del distrito Centro de Madrid, con explicación individual por lead y perfil enriquecido listo para pegar en HubSpot o Notion.

## Cómo usarlo

```bash
# Clonar el repositorio
git clone https://github.com/juannalonso/tfm-data-science-gtm.git
cd tfm-data-science-gtm

# Crear entorno virtual e instalar dependencias
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Lanzar la aplicación
streamlit run app/app.py
```

La app se abre en `http://localhost:8501` con dos casos de demo preconfigurados: **software de reservas premium** y **distribuidor de producto asiático**.

## Entregas

Las entregas son incrementales; cada una refleja el estado del proyecto en su momento.

- [Entrega 1 — Ideas de producto](docs/entregas/01_ideas_producto.md)
- [Entrega 2 — Selección de idea y datos necesarios](docs/entregas/02_datos_necesarios.md)
- [Entrega 3 — Modelo de datos y capa gold](docs/entregas/03_modelo_datos.md)
- [Entrega 4 — Diseño del análisis y estrategia de modelado](docs/entregas/04_analisis_modelado.md)
- [Entrega 5 — Diseño del frontal y experiencia de usuario](docs/entregas/05_frontal_experiencia_usuario.md)

## Fuentes de datos

| Fuente | Aportación | Cobertura |
|---|---|---|
| **OpenStreetMap** (API Overpass) | Universo de restaurantes, ubicación, cocina, servicios | 1.688 restaurantes |
| **Censo Ayuntamiento Madrid** | Clasificación oficial, barrio, sección censal | 80,4% cruzado |
| **INE** (sección censal) | Renta media del entorno del establecimiento | 80,4% enriquecido |
| **Gemini LLM** | Ambiente, presencia digital, posicionamiento, resumen cualitativo | 99,9% enriquecido |

## Arquitectura del sistema

- **Capa gold** (`data/gold/gold_restaurantes_madrid.parquet`): 1.688 restaurantes × 33 columnas consolidadas de las 4 fuentes.
- **Modelo de similitud** (`src/similitud.py`): vectorización combinada (numéricas escaladas + one-hot + booleanas) y cálculo de similitud coseno con el centroide del ICP.
- **App Streamlit** (`app/app.py`): interfaz operativa con tres pestañas (Explorar, Cualificar, Sobre el proyecto).
- **Notebooks de análisis** (`notebooks/`): construcción de la capa gold, enriquecimiento LLM y comparación de modelos ML.

## Stack tecnológico

- **Lenguaje:** Python 3.9
- **Análisis:** pandas, scikit-learn, XGBoost
- **Datos geográficos:** shapely, pyproj, geopandas
- **Enriquecimiento cualitativo:** Google Gemini (`google-genai`)
- **Frontal:** Streamlit
- **Almacenamiento:** Parquet + CSV

## Estructura del repositorio

```
tfm-data-science-gtm/
├── README.md
├── docs/entregas/                      # Entregas académicas 1-5
├── data/
│   ├── raw/                            # Datos originales de las fuentes
│   ├── processed/                      # Datos limpios y cruzados
│   └── gold/                           # Capa gold final
├── notebooks/
│   ├── 01_prueba_fuentes_osm.ipynb
│   ├── 02_construccion_gold.ipynb
│   ├── 03_enriquecimiento_llm.ipynb
│   ├── 04a_modelos_ml_premium.ipynb
│   └── 04b_modelos_ml_asiatico.ipynb
├── src/
│   └── similitud.py                    # Motor de similitud coseno
└── app/
    └── app.py                          # Aplicación Streamlit
```

## Notas sobre datos y privacidad

Este repositorio es público. Los datos utilizados proceden exclusivamente de fuentes abiertas (OpenStreetMap, Ayuntamiento de Madrid, INE) y se refieren a información corporativa de establecimientos, sin datos personales identificables. No se incluyen datos de clientes ni de la empresa de prácticas del autor.

---

*Repositorio activo hasta el 4 de octubre de 2026.*
