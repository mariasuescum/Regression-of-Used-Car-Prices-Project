import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint
import joblib
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

# 1. Cargar datos
df = pd.read_csv("data/final_dataset.csv")
features = [
    'age', 'milage', 'accident', 'engine_hp',
    'brand_id', 'model_id', 'fuel_type_id', 'transmission_norm'
]
X = df[features]
y = df['price']

# 2. Dividir train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Definir modelo y espacio de búsqueda
rf = RandomForestRegressor(random_state=30)
param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 25),
    'min_samples_leaf': randint(1, 20),
}

# 4. Tuning con RandomizedSearchCV
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='r2',
    verbose=1,
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)

# Modelo
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=12,
    min_samples_split=23,
    random_state=30
)

# Entrenar el modelo
model.fit(X_train, y_train)

# Predicciones
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Métricas
mse = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
overfit_diff = r2_train - r2_test

# MODELO BASE
print("---------MODELO BASE--------------")
print(f"R² Entrenamiento: {r2_train:.4f}")
print(f"R² Test: {r2_test:.4f}")
print(f"MSE Test: {mse:.2f}")
print(f"Overfitting (diferencia): {overfit_diff:.4f}")

print("---------MODELO MEJORADO-----------")

# MODELO MEJORADO 
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Mejores hiperparámetros:", random_search.best_params_)
print(f"R² Test: {r2_score(y_test, y_pred):.4f}")
print(f"MSE Test: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Overfitting (diferencia): {overfit_diff:.4f}")

# Validación cruzada K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=kf, scoring='r2')
print("-------VALIDACIÓN CRUZADA K-Fold:-------")
print("R² por fold:", cv_scores)
print(f"R² promedio: {np.mean(cv_scores):.4f}")
print(f"Desviación estándar: {np.std(cv_scores):.4f}")

# 6. Guardar modelo entrenado
joblib.dump(best_model, "models/random_forest_best.pkl")
