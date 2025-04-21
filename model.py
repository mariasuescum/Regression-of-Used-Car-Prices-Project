import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Cargar el dataset
df = pd.read_csv("data\cleaned_dataset.csv")

# Selección de características numéricas
features = [
    'model_year', 'milage', 'accident', 'engine_hp',
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
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")