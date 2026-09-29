"""
similitud.py — Modelo de cualificación por similitud coseno
============================================================
Vectoriza la capa gold y devuelve el score de similitud coseno de cada
restaurante frente al centroide del ICP (seeds).

Vector combinado (v1 con embedding):
- Numéricas escaladas: competencia_500m, competencia_1km, renta_media.
- Booleanas: banderas operativas de OSM.
- Categóricas one-hot: cocina, epigrafe_oficial, barrio, ambiente,
  presencia_digital.
- Embedding del resumen_llm (sentence-transformers/all-MiniLM-L6-v2, 384 dim).

Excluidas del modelo por baja varianza (análisis de sesgo LLM):
- posicionamiento_precio (90% "medio").
- tipo_clientela (84% "mixta").
Se muestran en la app como campos descriptivos, pero no entran al vector.

Los embeddings se computan una vez y se cachean en
`data/processed/embeddings_resumen_llm.parquet` para arranques posteriores
rápidos. El score de similitud coseno está teóricamente en el rango [-1, 1].

Autor: Juan Alonso — TFM Data Science e IA
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
    "cocina_osm",
    "epigrafe_oficial",
    "barrio",
    "ambiente",
    "presencia_digital",
]

EXCLUIDAS_POR_SESGO: list[str] = ["posicionamiento_precio", "tipo_clientela"]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Ruta del cache de embeddings (relativa a la raíz del repo)
CACHE_EMB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "processed" / "embeddings_resumen_llm.parquet"
)


@dataclass
class ModeloSimilitud:
    X: np.ndarray                    # matriz (n_restaurantes, n_features)
    ids: pd.Index                    # restaurante_id en el mismo orden
    feature_names: list[str]
    columnas_usadas: dict[str, list[str]]


def _computar_embeddings(textos: list[str]) -> np.ndarray:
    """Descarga (si es primera vez) el modelo sentence-transformers y computa
    los embeddings para la lista de textos. Devuelve matriz (n, 384)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    emb = model.encode(textos, show_progress_bar=False, batch_size=32)
    return np.asarray(emb, dtype=np.float32)


def _cargar_o_computar_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Devuelve embeddings alineados con df (por restaurante_id).
    Usa cache en disco si existe y cubre todos los ids necesarios."""
    textos = df["resumen_llm"].fillna("").astype(str).tolist()
    ids = df["restaurante_id"].tolist()

    if CACHE_EMB_PATH.exists():
        cache = pd.read_parquet(CACHE_EMB_PATH).set_index("restaurante_id")
        if all(i in cache.index for i in ids):
            emb_cols = [c for c in cache.columns if c.startswith("emb_")]
            emb = cache.loc[ids, emb_cols].values.astype(np.float32)
            return emb

    # Cache inexistente o incompleta: computar desde cero
    print(f"Computando embeddings de {len(textos)} resúmenes (primera vez, ~10-30s)...")
    emb = _computar_embeddings(textos)

    cols = [f"emb_{i}" for i in range(emb.shape[1])]
    cache_df = pd.DataFrame(emb, columns=cols)
    cache_df["restaurante_id"] = ids
    CACHE_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_parquet(CACHE_EMB_PATH, index=False)
    return emb


def construir_vectores(df: pd.DataFrame, use_embedding: bool = True) -> ModeloSimilitud:
    df = df.copy()

    numericas = [c for c in NUMERICAS_CANDIDATAS if c in df.columns]
    booleanas = [c for c in BOOLEANAS_CANDIDATAS if c in df.columns]
    categoricas = [c for c in CATEGORICAS_CANDIDATAS if c in df.columns]

    if not (numericas or booleanas or categoricas):
        raise ValueError("La gold no contiene ninguna de las columnas esperadas.")

    for c in booleanas:
        df[c] = df[c].fillna(False).astype(int)
    for c in categoricas:
        df[c] = df[c].astype("string").fillna("desconocido")

    df_num = df[numericas].copy() if numericas else pd.DataFrame(index=df.index)
    for c in numericas:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
        df_num[c] = df_num[c].fillna(df_num[c].median())

    df_cat = (
        pd.get_dummies(df[categoricas], prefix=categoricas).astype(int)
        if categoricas else pd.DataFrame(index=df.index)
    )

    X_est = pd.concat([df_num, df[booleanas], df_cat], axis=1)
    feature_names_est = X_est.columns.tolist()
    X_est_scaled = StandardScaler().fit_transform(X_est.values).astype(np.float32)

    # Embedding del resumen_llm — se concatena al vector estructurado
    incluye_emb = use_embedding and "resumen_llm" in df.columns
    if incluye_emb:
        emb = _cargar_o_computar_embeddings(df)
        X = np.hstack([X_est_scaled, emb])
        feature_names = feature_names_est + [f"emb_{i}" for i in range(emb.shape[1])]
    else:
        X = X_est_scaled
        feature_names = feature_names_est

    return ModeloSimilitud(
        X=X,
        ids=df["restaurante_id"].reset_index(drop=True),
        feature_names=feature_names,
        columnas_usadas={
            "numericas": numericas,
            "booleanas": booleanas,
            "categoricas": categoricas,
            "excluidas_por_sesgo": [c for c in EXCLUIDAS_POR_SESGO if c in df.columns],
            "embedding": ["resumen_llm (384 dim)"] if incluye_emb else [],
        },
    )


def _coseno(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = b / (np.linalg.norm(b) + 1e-12)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    return (A @ b) / A_norm.ravel()


def calcular_scores(modelo: ModeloSimilitud, ids_seeds: list[str]) -> pd.Series:
    """Devuelve una Serie con el score de similitud por restaurante_id.

    - Centroide del ICP = media de los vectores de las seeds.
    - Score en [-1, 1] teórico; en la práctica cae en un rango más estrecho
      por la naturaleza de los vectores (features no negativas + embedding).
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


# ============================================================
# Explicabilidad: descomposición del score por bloques
# ============================================================

def descomponer_score(
    modelo: ModeloSimilitud,
    ids_seeds: list[str],
    restaurante_id: str,
    df_gold: pd.DataFrame,
) -> dict:
    """Descompone el score de un restaurante contra el centroide del ICP en:

    - score_total: coseno global del vector combinado (551 dims).
    - score_estructural: coseno solo sobre las features estructuradas.
    - score_semantico: coseno solo sobre el embedding LLM.
    - peso_estructural / peso_semantico: contribución de cada bloque al
      numerador del coseno global (suman 1). Da una intuición de qué bloque
      ha pesado más en el score final.
    - coincidencias: lista de variables categóricas donde el lead comparte
      valor con al menos una de las seeds (barrio, ambiente, presencia
      digital, epígrafe).
    """
    # Máscara de seeds e índice del restaurante
    mask_seeds = modelo.ids.isin(ids_seeds).values
    if mask_seeds.sum() == 0:
        raise ValueError("Ninguna seed encontrada en el modelo.")

    idx_arr = modelo.ids[modelo.ids == restaurante_id].index
    if len(idx_arr) == 0:
        raise ValueError(f"restaurante_id {restaurante_id} no está en el modelo.")
    idx_r = int(idx_arr[0])

    # Split del vector en bloque estructural vs bloque embedding
    n_est = sum(1 for f in modelo.feature_names if not f.startswith("emb_"))
    X_est_all = modelo.X[:, :n_est]
    X_emb_all = modelo.X[:, n_est:]

    v_total = modelo.X[idx_r]
    v_est = X_est_all[idx_r]
    v_emb = X_emb_all[idx_r]

    c_total = modelo.X[mask_seeds].mean(axis=0)
    c_est = X_est_all[mask_seeds].mean(axis=0)
    c_emb = X_emb_all[mask_seeds].mean(axis=0) if X_emb_all.shape[1] > 0 else np.array([])

    def _cos1(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / denom)

    score_total = _cos1(v_total, c_total)
    score_estructural = _cos1(v_est, c_est)
    score_semantico = _cos1(v_emb, c_emb) if X_emb_all.shape[1] > 0 else 0.0

    # Contribución de cada bloque al numerador del coseno global
    contrib_est = float(np.dot(v_est, c_est))
    contrib_emb = float(np.dot(v_emb, c_emb)) if X_emb_all.shape[1] > 0 else 0.0
    contrib_total = contrib_est + contrib_emb
    if abs(contrib_total) > 1e-9:
        peso_est = contrib_est / contrib_total
        peso_emb = contrib_emb / contrib_total
    else:
        peso_est = 0.5
        peso_emb = 0.5

    # Coincidencias categóricas con las seeds
    cats_check = ["barrio", "ambiente", "presencia_digital", "epigrafe_oficial"]
    seeds_rows = df_gold[df_gold["restaurante_id"].isin(ids_seeds)]
    lead_row = df_gold[df_gold["restaurante_id"] == restaurante_id]
    if len(lead_row) == 0:
        lead_row_dict = {}
    else:
        lead_row_dict = lead_row.iloc[0].to_dict()

    coincidencias = []
    for cat in cats_check:
        if cat not in df_gold.columns:
            continue
        seed_vals = seeds_rows[cat].dropna().astype(str).str.strip().unique().tolist()
        lead_val = lead_row_dict.get(cat)
        lead_val_str = str(lead_val).strip() if pd.notna(lead_val) else None
        coincide = lead_val_str in seed_vals if lead_val_str else False
        coincidencias.append({
            "variable": cat,
            "lead_valor": lead_val_str if lead_val_str else "—",
            "seed_valores": seed_vals,
            "coincide": coincide,
        })

    return {
        "score_total": score_total,
        "score_estructural": score_estructural,
        "score_semantico": score_semantico,
        "peso_estructural": peso_est,
        "peso_semantico": peso_emb,
        "coincidencias": coincidencias,
    }
