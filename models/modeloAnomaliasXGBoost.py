# Librerías básicas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

# Preprocesamiento
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)

#Modelo
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

# Configuración
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")
#%matplotlib inline

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 1. CARGAR DATOS LIMPIOS (DATASET)
df = pd.read_csv('C:/Users/ariza/ProyectoSamsung_modelosIA/train_models/data/dataset_limpio.csv')

print("Datos cargados:")
print(f"Dimensiones: {df.shape}")
print(f"\nColumnas:{df.columns.tolist()}")

# Verificar valores faltantes
print("Valores faltantes:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No hay valores faltantes")
else:
    print(missing[missing > 0])

# Distribución de la variable objetivo
print("Distribución de la variable objetivo (Abnormal_Usage):")
print(df['Abnormal_Usage'].value_counts())
print(f"\nPorcentajes:")
print(df['Abnormal_Usage'].value_counts(normalize=True) * 100)

#:::::::::::::::::::::::::::::::::::::::::::
# 2. PREPARACIÓN DE DATOS
#:::::::::::::::::::::::::::::::::::::::::::

# Separar features y target
X = df.drop('Abnormal_Usage', axis=1)
y = df['Abnormal_Usage']

print(f"Shape de X (features): {X.shape}")
print(f"Shape de y (target): {y.shape}")

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y  # Mantener la proporción de clases
)

print("División de datos:")
print(f"  - Entrenamiento: {X_train.shape[0]} muestras ({(len(X_train)/len(X))*100:.1f}%)")
print(f"  - Prueba: {X_test.shape[0]} muestras ({(len(X_test)/len(X))*100:.1f}%)")

print(f"\nDistribución de clases en entrenamiento:")
print(y_train.value_counts())
print(f"\nDistribución de clases en prueba:")
print(y_test.value_counts())

#XGBoost no necesita escalado de datos. Por lo tanto, se utilizan:
# X_train y X_test directamente

#:::::::::::::::::::::::::::::::::::::::::::
## 3. MODELO XGBOOST
#:::::::::::::::::::::::::::::::::::::::::::
print("\n" + "="*60)
print("MODELO BASE XGBOOST")
print("="*60)

#Sin optimizar
# Entrenar XGBoost con parámetros por defecto
xgb_base = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
xgb_base.fit(X_train, y_train)  # XGBoost no requiere escalado

# Predicciones
y_pred_xgb_base = xgb_base.predict(X_test)
y_pred_proba_xgb_base = xgb_base.predict_proba(X_test)[:, 1]

# Métricas
print("RESULTADOS XGBOOST BASE")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb_base):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb_base):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb_base):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_xgb_base):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_xgb_base):.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_xgb_base, target_names=['Normal', 'Anómalo']))

#:::::::::::::::::::::::::::::::::::::::::::
# 4. OPTIMIZACIÓN CON GRID SEARCH
#:::::::::::::::::::::::::::::::::::::::::::
print("\n" + "="*60)
print("OPTIMIZACIÓN XGBOOST — GRID SEARCH")
print("="*60)

param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

print("Grilla de hiperparámetros para XGBoost:")
print(param_grid_xgb)
print(f"\nTotal de combinaciones: {len(param_grid_xgb['n_estimators']) * len(param_grid_xgb['max_depth']) * len(param_grid_xgb['learning_rate']) * len(param_grid_xgb['subsample']) * len(param_grid_xgb['colsample_bytree'])}")

cv_strategy = 5 #Es para usar 5-fold CV, aunque se puede ajustar a 3 para mayor velocidad si es necesario
# Configurar Grid Search para XGBoost
grid_search_xgb = GridSearchCV(
    estimator=XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
    param_grid=param_grid_xgb,
    cv=cv_strategy,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

# Usar la misma muestra que con SVM
X_train_xgb_sample, _, y_train_xgb_sample, _ = train_test_split(
    X_train, y_train,
    train_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y_train
)

# Grid Search con muestra
grid_search_xgb.fit(X_train_xgb_sample, y_train_xgb_sample)
print("\nGrid Search completado")

# Mejores hiperparámetros encontrados
print("MEJORES HIPERPARÁMETROS XGBOOST")
print(f"Mejores parámetros: {grid_search_xgb.best_params_}")
print(f"Mejor F1-score (validación cruzada): {grid_search_xgb.best_score_:.4f}")

# Modelo final con todos los datos
xgb_best = XGBClassifier(
    n_estimators=grid_search_xgb.best_params_['n_estimators'],
    max_depth=grid_search_xgb.best_params_['max_depth'],
    learning_rate=grid_search_xgb.best_params_['learning_rate'],
    subsample=grid_search_xgb.best_params_['subsample'],
    colsample_bytree=grid_search_xgb.best_params_['colsample_bytree'],
    random_state=RANDOM_STATE,
    eval_metric='logloss'
)

xgb_best.fit(X_train, y_train)

# Evaluar el mejor modelo en el conjunto de prueba
y_pred_xgb_best = xgb_best.predict(X_test)
y_pred_proba_xgb_best = xgb_best.predict_proba(X_test)[:, 1]

print("RESULTADOS XGBOOST OPTIMIZADO (TEST SET)")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb_best):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb_best):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb_best):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_xgb_best):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_xgb_best):.4f}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_xgb_best, target_names=['Normal', 'Anómalo']))

# Matriz de confusión
#...

#:::::::::::::::::::::::::::::::::::::::::::
## 6. SELECCIÓN DE CARACTERÍSTICAS POR IMPORTANCIA (SelectFromModel)
#:::::::::::::::::::::::::::::::::::::::::::

print("\n" + "="*60)
print("SELECCIÓN DE FEATURES — XGBOOST (umbral=mean)")
print("="*60)

# Obtener importancia de features
feature_importance_xgb = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_best.feature_importances_
}).sort_values('importance', ascending=False)
print("Importancia de features (XGBoost):")
print(feature_importance_xgb)

# SelectFromModel con threshold='mean':
# conserva features con importancia >= promedio de todas las importancias
selector_xgb = SelectFromModel(
    estimator = xgb_best,
    threshold = 'mean',   # umbral = importancia promedio
    prefit    = True      # xgb_best ya está entrenado
)

# Aplicar selección a train y test
X_train_xgb_fs = selector_xgb.transform(X_train)
X_test_xgb_fs  = selector_xgb.transform(X_test)

# Identificar features seleccionadas y descartadas
features_seleccionadas_xgb = X.columns[selector_xgb.get_support()].tolist()
features_descartadas_xgb   = X.columns[~selector_xgb.get_support()].tolist()

umbral = xgb_best.feature_importances_.mean()
print(f"Umbral de importancia (promedio): {umbral:.4f}")

print(f"\nFeatures SELECCIONADAS ({len(features_seleccionadas_xgb)}):")
for f in features_seleccionadas_xgb:
    imp = xgb_best.feature_importances_[list(X.columns).index(f)]
    print(f" {f:35s}  importancia: {imp:.4f}")

print(f"\nFeatures DESCARTADAS ({len(features_descartadas_xgb)}):")
for f in features_descartadas_xgb:
    imp = xgb_best.feature_importances_[list(X.columns).index(f)]
    print(f" {f:35s}  importancia: {imp:.4f}")

# Reentrenar XGBoost con los hiperparámetros ya optimizados y las features seleccionadas
xgb_fs = XGBClassifier(
    n_estimators     = grid_search_xgb.best_params_['n_estimators'],
    max_depth        = grid_search_xgb.best_params_['max_depth'],
    learning_rate    = grid_search_xgb.best_params_['learning_rate'],
    subsample        = grid_search_xgb.best_params_['subsample'],
    colsample_bytree = grid_search_xgb.best_params_['colsample_bytree'],
    random_state     = RANDOM_STATE,
    eval_metric      = 'logloss'
)
xgb_fs.fit(X_train_xgb_fs, y_train)

# Predicciones
y_pred_xgb_fs = xgb_fs.predict(X_test_xgb_fs)
y_pred_proba_xgb_fs = xgb_fs.predict_proba(X_test_xgb_fs)[:, 1]

n_features_total = X_train.shape[1]
print("RESULTADOS XGBOOST — CON SELECCIÓN POR IMPORTANCIA")
print(f"Features usadas : {len(features_seleccionadas_xgb)} de {n_features_total}")
print(f"Accuracy  : {accuracy_score(y_test, y_pred_xgb_fs):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_xgb_fs):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_xgb_fs):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred_xgb_fs):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_pred_proba_xgb_fs):.4f}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_xgb_fs, target_names=['Normal', 'Anómalo']))

# Comparación: XGBoost completo vs XGBoost con selección por importancia
comparacion_xgb_fs = pd.DataFrame({
    'Modelo'      : ['XGBoost Optimizado (todas las features)',
                     f'XGBoost + Importancia ({len(features_seleccionadas_xgb)} features, umbral=mean)'],
    'N° Features' : [n_features_total, len(features_seleccionadas_xgb)],
    'Accuracy'    : [accuracy_score(y_test, y_pred_xgb_best),  accuracy_score(y_test, y_pred_xgb_fs)],
    'Precision'   : [precision_score(y_test, y_pred_xgb_best), precision_score(y_test, y_pred_xgb_fs)],
    'Recall'      : [recall_score(y_test, y_pred_xgb_best),    recall_score(y_test, y_pred_xgb_fs)],
    'F1-Score'    : [f1_score(y_test, y_pred_xgb_best),        f1_score(y_test, y_pred_xgb_fs)],
    'ROC-AUC'     : [roc_auc_score(y_test, y_pred_proba_xgb_best), roc_auc_score(y_test, y_pred_proba_xgb_fs)]
})

print("Comparación XGBoost — todas las features vs Selección por Importancia:")
print(comparacion_xgb_fs.round(4).to_string(index=False))

#::::::::::::::::::::::::::::::::
## 6. VALIZACIÓN CRUZADA
#::::::::::::::::::::::::::::::::

print("\n" + "="*60)
print("VALIDACIÓN CRUZADA — XGBOOST (5-fold)")
print("="*60) 

t0 = time.time()
cv_5fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring_metrics = {
    'accuracy' : 'accuracy',
    'precision': 'precision',
    'recall'   : 'recall',
    'f1'       : 'f1',
    'roc_auc'  : 'roc_auc'
}

cv_results_xgb_completo = cross_validate(
    xgb_best,
    X_train, y_train,          # XGBoost no requiere datos escalados
    cv=cv_5fold,
    scoring=scoring_metrics,
    n_jobs=-1,
    return_train_score=True    # Permite detectar overfitting comparando train vs validación
)

print("\nRESULTADOS VALIDACIÓN CRUZADA — XGBOOST TODAS LAS FEATURES")
for metrica in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    val_scores   = cv_results_xgb_completo[f'test_{metrica}']
    train_scores = cv_results_xgb_completo[f'train_{metrica}']
    print(f"  {metrica:10s} | Val: {val_scores.mean():.4f} ± {val_scores.std():.4f} "
          f"| Train: {train_scores.mean():.4f} ± {train_scores.std():.4f}")

print(f"\nTiempo: {time.time()-t0:.1f}s")

##  Validación cruzada: Selección por Importancia
#t0 = time.time()
# Se realiza validación cruzada con el modelo XGBoost entrenado solo con las features
# seleccionadas por importancia (umbral = mean), para evaluar su rendimiento.
cv_results_xgb_fs = cross_validate(
    xgb_fs,
    X_train_xgb_fs, y_train,   # Subconjunto ya transformado por SelectFromModel
    cv=cv_5fold,
    scoring=scoring_metrics,
    n_jobs=-1,
    return_train_score=True
)

print("\nRESULTADOS VALIDACIÓN CRUZADA — XGBOOST CON SELECCIÓN POR IMPORTANCIA")
for metrica in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    val_scores   = cv_results_xgb_fs[f'test_{metrica}']
    train_scores = cv_results_xgb_fs[f'train_{metrica}']
    print(f"  {metrica:10s} | Val: {val_scores.mean():.4f} ± {val_scores.std():.4f} "
          f"| Train: {train_scores.mean():.4f} ± {train_scores.std():.4f}")

print(f"\nTiempo: {time.time()-t0:.1f}s")

# comparación de validación cruzada
metricas_cv = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

resumen_cv_xgb = pd.DataFrame({
    'Métrica'              : metricas_cv,
    'XGB_Completo_Media'   : [cv_results_xgb_completo[f'test_{m}'].mean() for m in metricas_cv],
    'XGB_Completo_Std'     : [cv_results_xgb_completo[f'test_{m}'].std()  for m in metricas_cv],
    'XGB_FS_Media'         : [cv_results_xgb_fs[f'test_{m}'].mean()       for m in metricas_cv],
    'XGB_FS_Std'           : [cv_results_xgb_fs[f'test_{m}'].std()        for m in metricas_cv],
})

print("COMPARACIÓN VALIDACIÓN CRUZADA — XGBoost Completo vs XGBoost + Importancia")
print(resumen_cv_xgb.round(4).to_string(index=False))