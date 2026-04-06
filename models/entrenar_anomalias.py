"""
entrenar_anomalias.py
=====================
Script de entrenamiento — Detección de Anomalías en Consumo Eléctrico
Modelo elegido: XGBoost con selección por importancia (threshold='mean')

Artefactos generados:
  - modelo_anomalias.pkl   → XGBClassifier entrenado con features seleccionadas
  - selector_anomalias.pkl → SelectFromModel ajustado (define qué features usar)
  - features_anomalias.pkl → lista con los nombres de las features seleccionadas

Uso:
  python entrenar_anomalias.py
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

# ── Configuración ──────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
DATA_PATH      = Path(__file__).parent.parent / "train_models" / "data" / "datos_limpios_anomalias.csv"
OUTPUT_DIR     = Path(__file__).parent
TARGET_COL     = "Abnormal_Usage"

# Hiperparámetros exactos encontrados por Grid Search en la notebook
XGB_PARAMS = {
    "n_estimators"    : 100,
    "max_depth"       : 7,
    "learning_rate"   : 0.1,
    "subsample"       : 1.0,
    "colsample_bytree": 1.0,
    "random_state"    : RANDOM_STATE,
    "eval_metric"     : "logloss",
}

# ── 1. Cargar datos ────────────────────────────────────────────────────────────
print("=" * 60)
print("ENTRENAMIENTO — Detección de Anomalías (XGBoost)")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Datos cargados: {df.shape[0]:,} filas, {df.shape[1]} columnas")

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

print(f"    Features disponibles ({len(X.columns)}): {X.columns.tolist()}")
print(f"    Distribución de clases:\n{y.value_counts().to_string()}")

# ── 2. Split train/test ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"\n[2] Split: {len(X_train):,} train / {len(X_test):,} test")

# ── 3. Entrenar XGBoost base (necesario para SelectFromModel) ─────────────────
print("\n[3] Entrenando XGBoost base para calcular importancias...")
xgb_base = XGBClassifier(**XGB_PARAMS)
xgb_base.fit(X_train, y_train)   # XGBoost no requiere escalado

# ── 4. Selección de features por importancia (threshold = mean) ───────────────
selector = SelectFromModel(
    estimator=xgb_base,
    threshold="mean",  # conserva features con importancia >= promedio
    prefit=True        # xgb_base ya está entrenado
)

features_seleccionadas = X.columns[selector.get_support()].tolist()
features_descartadas   = X.columns[~selector.get_support()].tolist()
umbral = xgb_base.feature_importances_.mean()

print(f"\n[4] Selección por importancia (umbral = mean = {umbral:.4f})")
print(f"    Features SELECCIONADAS ({len(features_seleccionadas)}): {features_seleccionadas}")
print(f"    Features DESCARTADAS   ({len(features_descartadas)}): {features_descartadas}")

# ── 5. Reentrenar XGBoost solo con features seleccionadas ─────────────────────
X_train_fs = selector.transform(X_train)
X_test_fs  = selector.transform(X_test)

print(f"\n[5] Reentrenando XGBoost con {len(features_seleccionadas)} features...")
xgb_final = XGBClassifier(**XGB_PARAMS)
xgb_final.fit(X_train_fs, y_train)

# ── 6. Evaluación rápida en test ───────────────────────────────────────────────
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

y_pred      = xgb_final.predict(X_test_fs)
y_pred_prob = xgb_final.predict_proba(X_test_fs)[:, 1]

print("\n[6] Métricas en test set:")
print(f"    Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"    F1-Score : {f1_score(y_test, y_pred):.4f}")
print(f"    ROC-AUC  : {roc_auc_score(y_test, y_pred_prob):.4f}")

# ── 7. Guardar artefactos ──────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ruta_modelo    = OUTPUT_DIR / "modelo_anomalias.pkl"
ruta_selector  = OUTPUT_DIR / "selector_anomalias.pkl"
ruta_features  = OUTPUT_DIR / "features_anomalias.pkl"

with open(ruta_modelo,   "wb") as f: pickle.dump(xgb_final,             f)
with open(ruta_selector, "wb") as f: pickle.dump(selector,              f)
with open(ruta_features, "wb") as f: pickle.dump(features_seleccionadas, f)

print("\n[7] Artefactos guardados:")
print(f"    {ruta_modelo}")
print(f"    {ruta_selector}")
print(f"    {ruta_features}")
print("\n✓ Entrenamiento completado.\n")
