import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from scipy.stats import randint

# Cargar el dataset
df = pd.read_csv("data/final_dataset.csv")

# Features y target
features = [
    'age', 'milage', 'accident', 'engine_hp',
    'brand_id', 'model_id', 'fuel_type_id', 'transmission_norm'
]
X = df[features]
y = df['price']

# División en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir modelo base
rf = RandomForestRegressor(random_state=30)

# Definir espacio de búsqueda
param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 25),
    'min_samples_leaf': randint(1, 20),
}

# Randomized Search
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

# Entrenar
random_search.fit(X_train, y_train)

# Mejor modelo
best_model = random_search.best_estimator_
print("Mejores hiperparámetros:", random_search.best_params_)

# Evaluación
y_pred = best_model.predict(X_test)
print(f"R² Test: {r2_score(y_test, y_pred):.4f}")
print(f"MSE Test: {mean_squared_error(y_test, y_pred):.2f}")
