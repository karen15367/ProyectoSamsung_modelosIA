"""
entrenar_smartlighting.py
=========================
Script de entrenamiento — Predicción de Consumo Energético (Smart Lighting)
Modelo elegido: Regresión Lineal con selección por |coeficiente| (threshold='mean')

Artefactos generados:
  - modelo_smartlighting.pkl   → LinearRegression entrenado con features seleccionadas
  - scaler_smartlighting.pkl   → StandardScaler ajustado con X_train (SIEMPRE se guarda)
  - selector_smartlighting.pkl → SelectFromModel ajustado (define qué features usar)
  - features_smartlighting.pkl → lista con los nombres de las features seleccionadas

Uso:
  python entrenar_smartlighting.py
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Configuración ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
DATA_PATH    = Path(__file__).parent.parent / "train_models" / "data" / "datos_limpios_SmartLighting.csv"
OUTPUT_DIR   = Path(__file__).parent
TARGET_COL   = "energy_consumption_kwh"

# Columnas a excluir además del target
# (lighting_action_class es una variable derivada de clasificación, no es un predictor)
COLS_EXCLUIR = ["lighting_action_class"]

# ── 1. Cargar datos ────────────────────────────────────────────────────────────
print("=" * 60)
print("ENTRENAMIENTO — Consumo Smart Lighting (Regresión Lineal)")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Datos cargados: {df.shape[0]:,} filas, {df.shape[1]} columnas")

X = df.drop([TARGET_COL] + COLS_EXCLUIR, axis=1)
y = df[TARGET_COL]

print(f"    Features disponibles ({len(X.columns)}): {X.columns.tolist()}")
print(f"    Target — estadísticas básicas:")
print(f"      min={y.min():.4f}  max={y.max():.4f}  mean={y.mean():.4f}")

# ── 2. Split train/test ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
    # Sin stratify porque es regresión
)
print(f"\n[2] Split: {len(X_train):,} train / {len(X_test):,} test")

# ── 3. Escalado ────────────────────────────────────────────────────────────────
# IMPORTANTE: el scaler se ajusta SOLO con X_train (evitar data leakage)
print("\n[3] Escalando features con StandardScaler...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X.columns,
    index=X_test.index
)

# ── 4. Entrenar Regresión Lineal base (para calcular |coeficientes|) ───────────
print("\n[4] Entrenando Regresión Lineal base para calcular coeficientes...")
lr_base = LinearRegression()
lr_base.fit(X_train_scaled, y_train)

# ── 5. Selección de features por |coeficiente| (threshold = mean) ─────────────
selector = SelectFromModel(
    estimator=lr_base,
    threshold="mean",  # conserva features con |coef| >= promedio
    prefit=True        # lr_base ya está entrenado
)

features_seleccionadas = X.columns[selector.get_support()].tolist()
features_descartadas   = X.columns[~selector.get_support()].tolist()
umbral = np.abs(lr_base.coef_).mean()

print(f"\n[5] Selección por |coef| (umbral = mean = {umbral:.4f})")
print(f"    Features SELECCIONADAS ({len(features_seleccionadas)}): {features_seleccionadas}")
print(f"    Features DESCARTADAS   ({len(features_descartadas)}): {features_descartadas}")

# ── 6. Reentrenar Regresión Lineal solo con features seleccionadas ─────────────
X_train_fs = selector.transform(X_train_scaled)
X_test_fs  = selector.transform(X_test_scaled)

print(f"\n[6] Reentrenando Regresión Lineal con {len(features_seleccionadas)} features...")
lr_final = LinearRegression()
lr_final.fit(X_train_fs, y_train)

# ── 7. Evaluación rápida en test ───────────────────────────────────────────────
y_pred = lr_final.predict(X_test_fs)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n[7] Métricas en test set:")
print(f"    MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"    RMSE : {rmse:.4f}")
print(f"    R²   : {r2_score(y_test, y_pred):.4f}")

# ── 8. Guardar artefactos ──────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ruta_modelo    = OUTPUT_DIR / "modelo_smartlighting.pkl"
ruta_scaler    = OUTPUT_DIR / "scaler_smartlighting.pkl"
ruta_selector  = OUTPUT_DIR / "selector_smartlighting.pkl"
ruta_features  = OUTPUT_DIR / "features_smartlighting.pkl"

with open(ruta_modelo,   "wb") as f: pickle.dump(lr_final,              f)
with open(ruta_scaler,   "wb") as f: pickle.dump(scaler,                f)
with open(ruta_selector, "wb") as f: pickle.dump(selector,              f)
with open(ruta_features, "wb") as f: pickle.dump(features_seleccionadas, f)

print("\n[8] Artefactos guardados:")
print(f"    {ruta_modelo}")
print(f"    {ruta_scaler}")
print(f"    {ruta_selector}")
print(f"    {ruta_features}")
print("\n✓ Entrenamiento completado.\n")
