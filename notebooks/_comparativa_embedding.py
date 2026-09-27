import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.similitud import construir_vectores
from xgboost import XGBClassifier

gold = pd.read_parquet("data/gold/gold_restaurantes_madrid.parquet")
SEEDS_5 = ["b4c626a9f914", "5ee59f48bde1", "1d4eb6981e34", "99601101fac4", "8f0ebbb4291d"]

modelo_emb = construir_vectores(gold, use_embedding=True)
X = modelo_emb.X
idx_seeds = [gold.index[gold["restaurante_id"] == s].tolist()[0] for s in SEEDS_5]

# ============================================================
# Baseline manual: filtro por reglas ICP
# ============================================================
manual = gold[
    (gold["posicionamiento_precio"].isin(["premium", "alta_gama"])) &
    (gold["tiene_web"] == True) &
    (gold["presencia_digital"] == "activa")
].index.tolist()

print(f"Baseline manual: {len(manual)} restaurantes cumplen ICP duro")

# ============================================================
# LOO Recall para coseno y XGBoost
# ============================================================

def loo_coseno(X, seeds_idx, k):
    hits = 0
    for i, sid in enumerate(seeds_idx):
        otras = [s for j, s in enumerate(seeds_idx) if j != i]
        c = X[otras].mean(axis=0, keepdims=True)
        sims = cosine_similarity(c, X)[0]
        mask = np.ones(len(sims), dtype=bool)
        for o in otras: mask[o] = False
        ranked = np.argsort(-sims[mask])[:k]
        if sid in np.where(mask)[0][ranked]:
            hits += 1
    return hits / len(seeds_idx)

def loo_xgboost(X, seeds_idx, k, n_neg=200):
    hits = 0
    rng = np.random.RandomState(42)
    all_idx = np.arange(X.shape[0])
    for i, sid in enumerate(seeds_idx):
        otras = [s for j, s in enumerate(seeds_idx) if j != i]
        # Positivos: los 4 seeds restantes
        # Negativos sintéticos: muestras aleatorias no-seed
        no_seeds = np.setdiff1d(all_idx, seeds_idx)
        neg_idx = rng.choice(no_seeds, size=n_neg, replace=False)
        X_train = np.vstack([X[otras], X[neg_idx]])
        y_train = np.array([1]*len(otras) + [0]*n_neg)
        clf = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", use_label_encoder=False, verbosity=0
        )
        clf.fit(X_train, y_train)
        # Predicción sobre todo el universo (excluyendo los 4 seeds usados como entrenamiento)
        mask = np.ones(X.shape[0], dtype=bool)
        for o in otras: mask[o] = False
        probs = clf.predict_proba(X[mask])[:, 1]
        ranked = np.argsort(-probs)[:k]
        pos = np.where(mask)[0][ranked]
        if sid in pos:
            hits += 1
    return hits / len(seeds_idx)

def baseline_manual_recall(seeds_idx, manual_idx, k):
    hits = 0
    for sid in seeds_idx:
        others_ranked = [i for i in manual_idx if i != sid][:k]
        if sid in others_ranked:
            hits += 1
    return hits / len(seeds_idx)


print("\n" + "="*60)
print("COMPARATIVA DE MODELOS (5 seeds)")
print("="*60)

print("\n1. BASELINE MANUAL (filtro por reglas)")
r20 = baseline_manual_recall(idx_seeds, manual, 20)
r100 = baseline_manual_recall(idx_seeds, manual, 100)
print(f"  Recall@20:  {r20:.3f}")
print(f"  Recall@100: {r100:.3f}")

print("\n2. COSENO CON EMBEDDING (551 features)")
r20 = loo_coseno(X, idx_seeds, 20)
r100 = loo_coseno(X, idx_seeds, 100)
print(f"  Recall@20:  {r20:.3f}")
print(f"  Recall@100: {r100:.3f}")

print("\n3. XGBOOST PU (positivos + negativos sinteticos)")
r20 = loo_xgboost(X, idx_seeds, 20)
r100 = loo_xgboost(X, idx_seeds, 100)
print(f"  Recall@20:  {r20:.3f}")
print(f"  Recall@100: {r100:.3f}")
