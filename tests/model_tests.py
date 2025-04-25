import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class TestMachineLearningModel(unittest.TestCase):
    # Test para la validación del preprocesamiento de datos
    def test_feature_columns(self):
        df = pd.read_csv("data/final_dataset.csv")
        expected_columns = [
            'age', 'milage', 'accident', 'engine_hp', 
            'brand_id', 'model_id', 'fuel_type_id', 'transmission_norm'
        ]
        # Verificar que las columnas esperadas estén en el dataframe
        for col in expected_columns:
            self.assertIn(col, df.columns)
    
    def test_target_column(self):
        df = pd.read_csv("data/final_dataset.csv")
        # Verificar que la columna objetivo 'price' esté presente
        self.assertIn('price', df.columns)
    
    # Test para asegurar que R² sea mayor que 0.70
    def test_r2_score(self):
        df = pd.read_csv("data/final_dataset.csv")
        X = df[['age', 'milage', 'accident', 'engine_hp', 'brand_id', 'model_id', 'fuel_type_id', 'transmission_norm']]
        y = df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=30)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # calcular R²
        r2_test = r2_score(y_test, y_pred)
        
        # Verificar que R² sea mayor que 0.7 (umbral mínimo que pensamos como razonable)
        self.assertGreater(r2_test, 0.7, f"R² score is too low: {r2_test:.4f}")
    
    # Test para asegurar que el MSE sea menor que 170000000
    def test_mse(self):
        df = pd.read_csv("data/final_dataset.csv")
        X = df[['age', 'milage', 'accident', 'engine_hp', 'brand_id', 'model_id', 'fuel_type_id', 'transmission_norm']]
        y = df['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=30)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Calcular MSE
        mse = mean_squared_error(y_test, y_pred)
        
        # Verificar que el MSE no sea mayor que un umbral razonable
        self.assertLess(mse, 170000000, f"MSE is too high: {mse:.2f}")
    
    # Test para verificar que el modelo se guarda correctamente
    def test_model_save(self):
        model = RandomForestRegressor(n_estimators=100, random_state=30)
        df = pd.read_csv("data/final_dataset.csv")
        X = df[['age', 'milage', 'accident', 'engine_hp', 'brand_id', 'model_id', 'fuel_type_id', 'transmission_norm']]
        y = df['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        model.fit(X_train, y_train)
        
        # Guardar el modelo
        joblib.dump(model, "models/random_forest_test.pkl")
        
        # Verificar que el archivo del modelo existe
        self.assertTrue(os.path.exists("models/random_forest_test.pkl"))

if __name__ == "__main__":
    unittest.main()