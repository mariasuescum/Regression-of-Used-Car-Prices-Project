import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint
import joblib

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

# Calcular R² y Overfitting
y_train_pred = random_search.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_pred)
overfit_diff = r2_train - r2_test


# 5. Evaluación
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Mejores hiperparámetros:", random_search.best_params_)
print(f"R² Test: {r2_score(y_test, y_pred):.4f}")
print(f"MSE Test: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Overfitting: {overfit_diff:.4f}")

# 6. Guardar modelo entrenado
joblib.dump(best_model, "models/random_forest_best.pkl")
