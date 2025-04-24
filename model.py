import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Cargar el dataset
df = pd.read_csv("data/final_dataset.csv")

# Selección de características numéricas
features = [
    'age', 'milage', 'accident', 'engine_hp',
    'brand_id', 'model_id', 'fuel_type_id', 'transmission_norm'
]

# Target
target = 'price'

# Filtrar solo las columnas necesarias
X = df[features]
y = df[target]

# División train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

print(f"Mean Squared Error (Test): {mse:.2f}")
print(f"R² Entrenamiento: {r2_train:.4f}")
print(f"R² Test: {r2_test:.4f}")
print(f"Overfitting (diferencia): {overfit_diff:.4f}")