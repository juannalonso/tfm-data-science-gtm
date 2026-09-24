"""
similitud.py — Modelo de cualificación por similitud coseno
============================================================
Vectoriza la capa gold y devuelve el score de similitud coseno de cada
restaurante frente al centroide del ICP (seeds).

Decisiones (v0, sin embedding aún):
- Numéricas: competencia_500m, competencia_1km, renta_media (si existe).
- Booleanas: banderas operativas de OSM.
- Categóricas (one-hot): cocina, epigrafe_oficial, barrio, ambiente,
  presencia_digital. Estas dos últimas provienen del enriquecimiento LLM
  y tienen varianza útil.
- EXCLUIDAS del modelo por baja varianza (según análisis de sesgo LLM):
  posicionamiento_precio (90% "medio") y tipo_clientela (84% "mixta").
  Se muestran en la app para el usuario, pero no entran en el vector.
- Embedding de resumen_llm: pendiente (siguiente iteración).

El código es defensivo: usa sólo las columnas que existan en el DataFrame,
para que funcione tanto con la gold "reducida" como con la "completa".

Autor: Juan Alonso — TFM Data Science e IA
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Candidatas: se usarán solo las que estén presentes en el DataFrame.
NUMERICAS_CANDIDATAS: list[str] = [
    "competencia_500m",
    "competencia_1km",
    "renta_media",
]
BOOLEANAS_CANDIDATAS: list[str] = [
    "tiene_web",
    "tiene_telefono",
    "tiene_horario",
    "tiene_terraza",
    "ofrece_delivery",
    "ofrece_takeaway",
    "es_accesible",
]
CATEGORICAS_CANDIDATAS: list[str] = [
    "cocina",
    "cocina_osm",       # nombre alternativo si un día se regenera así
    "epigrafe_oficial",
    "barrio",
    "ambiente",
    "presencia_digital",
]

# Documentado explícitamente: excluidas del modelo por baja varianza.
EXCLUIDAS_POR_SESGO: list[str] = ["posicionamiento_precio", "tipo_clientela"]


@dataclass
class ModeloSimilitud:
    X: np.ndarray                    # matriz (n_restaurantes, n_features)
    ids: pd.Index                    # restaurante_id en el mismo orden
    feature_names: list[str]
    columnas_usadas: dict[str, list[str]]  # traza de qué se usó


def construir_vectores(df: pd.DataFrame) -> ModeloSimilitud:
    df = df.copy()

    numericas = [c for c in NUMERICAS_CANDIDATAS if c in df.columns]
    booleanas = [c for c in BOOLEANAS_CANDIDATAS if c in df.columns]
    categoricas = [c for c in CATEGORICAS_CANDIDATAS if c in df.columns]

    if not (numericas or booleanas or categoricas):
        raise ValueError("La gold no contiene ninguna de las columnas esperadas.")

    # Booleanas -> 0/1
    for c in booleanas:
        df[c] = df[c].fillna(False).astype(int)

    # Categóricas: rellenamos y one-hot
    for c in categoricas:
        df[c] = df[c].astype("string").fillna("desconocido")
    df_cat = (
        pd.get_dummies(df[categoricas], prefix=categoricas).astype(int)
        if categoricas else pd.DataFrame(index=df.index)
    )

    # Numéricas: nulos con la mediana
    df_num = df[numericas].copy() if numericas else pd.DataFrame(index=df.index)
    for c in numericas:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
        df_num[c] = df_num[c].fillna(df_num[c].median())

    X = pd.concat([df_num, df[booleanas], df_cat], axis=1)
    feature_names = X.columns.tolist()
    X_scaled = StandardScaler().fit_transform(X.values)

    return ModeloSimilitud(
        X=X_scaled,
        ids=df["restaurante_id"].reset_index(drop=True),
        feature_names=feature_names,
        columnas_usadas={
            "numericas": numericas,
            "booleanas": booleanas,
            "categoricas": categoricas,
            "excluidas_por_sesgo": [c for c in EXCLUIDAS_POR_SESGO if c in df.columns],
        },
    )


def _coseno(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = b / (np.linalg.norm(b) + 1e-12)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    return (A @ b) / A_norm.ravel()


def calcular_scores(modelo: ModeloSimilitud, ids_seeds: list[str]) -> pd.Series:
    """Devuelve una Serie con el score de similitud por restaurante_id.

    - Centroide del ICP = media de los vectores de las seeds.
    - Score en [-1, 1]. Cuanto mayor, más parecido al ICP.
    - Las propias seeds quedan con score = NaN (no se recomiendan).
    """
    mask = modelo.ids.isin(ids_seeds).values
    if mask.sum() == 0:
        raise ValueError("Ninguno de los IDs indicados existe en el modelo.")

    centroide = modelo.X[mask].mean(axis=0)
    scores = _coseno(modelo.X, centroide)

    s = pd.Series(scores, index=modelo.ids, name="score_similitud")
    s.loc[list(ids_seeds)] = np.nan
    return s
