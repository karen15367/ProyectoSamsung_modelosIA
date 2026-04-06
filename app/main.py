"""
main.py
=======
API con FastAPI — ProyectoSamsung ModelosIA
Expone dos endpoints de predicción:
  POST /predecir-anomalia      → Detección de anomalías (XGBoost Clasificación)
  POST /predecir-consumo       → Predicción de consumo energético (Regresión Lineal)

Para correr el servidor:
  uv run uvicorn main:app --reload
  (o: python -m uvicorn main:app --reload)
"""

import pickle
import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── Rutas de artefactos ────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR.parent / "models"

# ── Cargar modelos al iniciar la app ───────────────────────────────────────────
def cargar_pkl(ruta: Path):
    """Carga un archivo .pkl con manejo de error claro."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"Artefacto no encontrado: {ruta}\n"
            f"Ejecuta primero los scripts de entrenamiento en train_models/"
        )
    with open(ruta, "rb") as f:
        return pickle.load(f)

print("Cargando modelos...")

modelo_anomalias    = cargar_pkl(MODELS_DIR / "modelo_anomalias.pkl")
selector_anomalias  = cargar_pkl(MODELS_DIR / "selector_anomalias.pkl")
features_anomalias  = cargar_pkl(MODELS_DIR / "features_anomalias.pkl")

modelo_smartlighting   = cargar_pkl(MODELS_DIR / "modelo_smartlighting.pkl")
scaler_smartlighting   = cargar_pkl(MODELS_DIR / "scaler_smartlighting.pkl")
selector_smartlighting = cargar_pkl(MODELS_DIR / "selector_smartlighting.pkl")
features_smartlighting = cargar_pkl(MODELS_DIR / "features_smartlighting.pkl")

print(f"  ✓ Anomalías   — features: {features_anomalias}")
print(f"  ✓ SmartLighting — features: {features_smartlighting}")

# ── Inicializar FastAPI ────────────────────────────────────────────────────────
app = FastAPI(
    title="ProyectoSamsung — API de Modelos IA",
    description="Endpoints para detección de anomalías y predicción de consumo energético",
    version="1.0.0",
)

# Permitir que el frontend (archivos HTML/JS) pueda llamar a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # En producción reemplazar con el dominio real
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir el frontend estático desde la carpeta front-end/
FRONTEND_DIR = BASE_DIR.parent / "front-end"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Esquemas de entrada y salida (Pydantic) ────────────────────────────────────

class EntradaAnomalia(BaseModel):
    """
    Datos necesarios para detectar si un consumo es anómalo.
    Corresponden a las 2 features seleccionadas del modelo XGBoost.
    """
    actual_energy_kwh: float = Field(
        ...,
        description="Consumo energético real registrado (kWh)",
        example=3.5
    )
    usage_deviation_pct: float = Field(
        ...,
        description="Desviación porcentual del consumo respecto al esperado (%)",
        example=15.2
    )

class ResultadoAnomalia(BaseModel):
    prediccion: int            # 0 = Normal, 1 = Anómalo
    etiqueta: str              # "Normal" o "Anómalo"
    probabilidad_anomalia: float   # probabilidad de ser anómalo (0–1)
    features_usadas: list[str]


class EntradaConsumo(BaseModel):
    """
    Datos necesarios para predecir el consumo energético en Smart Lighting.
    Corresponden a las 2 features seleccionadas del modelo de Regresión Lineal.
    """
    occupancy_count: float = Field(
        ...,
        description="Número de personas en el espacio iluminado",
        example=12
    )
    adjusted_light_intensity: float = Field(
        ...,
        description="Intensidad de luz ajustada (valor normalizado)",
        example=0.75
    )

class ResultadoConsumo(BaseModel):
    consumo_predicho_kwh: float    # predicción de consumo energético
    features_usadas: list[str]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def raiz():
    """Sirve la homepage si existe el frontend estático."""
    index = FRONTEND_DIR / "homepage.html"
    if index.exists():
        return FileResponse(str(index))
    return {"mensaje": "API funcionando. Documentación en /docs"}


@app.get("/health")
def health():
    """Verificación de estado de la API."""
    return {
        "estado": "ok",
        "modelos_cargados": {
            "anomalias"     : True,
            "smartlighting" : True,
        }
    }


@app.post("/predecir-anomalia", response_model=ResultadoAnomalia)
def predecir_anomalia(datos: EntradaAnomalia):
    """
    Detecta si un registro de consumo eléctrico es anómalo o normal.

    - **actual_energy_kwh**: consumo real registrado en kWh
    - **usage_deviation_pct**: desviación porcentual del consumo esperado

    Devuelve la clase predicha (0=Normal, 1=Anómalo) y la probabilidad.
    """
    try:
        # Construir el vector de entrada en el orden que espera el selector
        # El selector_anomalias sabe qué columnas tomar del DataFrame original
        # Aquí recreamos ese vector manualmente porque solo recibimos 2 valores
        X_input = np.array([[
            datos.actual_energy_kwh,
            datos.usage_deviation_pct
        ]])

        # El modelo ya fue entrenado directamente con estas 2 features,
        # así que no necesitamos pasar por el selector en producción.
        prediccion  = int(modelo_anomalias.predict(X_input)[0])
        probabilidades = modelo_anomalias.predict_proba(X_input)[0]
        prob_anomalo   = float(probabilidades[1])

        return ResultadoAnomalia(
            prediccion=prediccion,
            etiqueta="Anómalo" if prediccion == 1 else "Normal",
            probabilidad_anomalia=round(prob_anomalo, 4),
            features_usadas=features_anomalias,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/predecir-consumo", response_model=ResultadoConsumo)
def predecir_consumo(datos: EntradaConsumo):
    """
    Predice el consumo energético en kWh para un escenario de iluminación inteligente.

    - **occupancy_count**: número de personas en el espacio
    - **adjusted_light_intensity**: intensidad de luz ajustada (0–1 típicamente)

    Devuelve el consumo predicho en kWh.
    """
    try:
        # El modelo de regresión lineal fue entrenado con datos escalados.
        # El scaler se ajustó con TODAS las features originales (21 columnas),
        # por lo que necesitamos escalar en el mismo espacio y luego seleccionar.
        # Para producción, dado que solo tenemos 2 features, usamos el enfoque
        # directo: reconstruir el escalado manual con los parámetros del scaler.

        # El scaler fue ajustado con las 21 features originales.
        # Cargamos features_smartlighting para saber el orden original
        # y extraemos media/std solo de las 2 que necesitamos.
        todas_las_features = list(scaler_smartlighting.feature_names_in_) \
            if hasattr(scaler_smartlighting, 'feature_names_in_') \
            else features_smartlighting

        valores_input = {
            "occupancy_count"         : datos.occupancy_count,
            "adjusted_light_intensity": datos.adjusted_light_intensity,
        }

        valores_escalados = []
        for feat in features_smartlighting:
            idx = todas_las_features.index(feat)
            val_escalado = (valores_input[feat] - scaler_smartlighting.mean_[idx]) \
                           / scaler_smartlighting.scale_[idx]
            valores_escalados.append(val_escalado)

        X_input = np.array([valores_escalados])

        consumo_predicho = float(modelo_smartlighting.predict(X_input)[0])

        return ResultadoConsumo(
            consumo_predicho_kwh=round(consumo_predicho, 4),
            features_usadas=features_smartlighting,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
