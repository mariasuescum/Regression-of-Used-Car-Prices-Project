import joblib
import pandas as pd
from utils import encode_brand, encode_fuel_type, encode_transmission, encode_model

# 1. Cargamos el modelo desde el archivo .pkl UNA sola vez.
model = joblib.load("models/random_forest_best.pkl")


def predict_price(inputs: dict) -> float:
    """
    Recibe un diccionario con los datos del coche
    y devuelve el precio estimado usando el modelo real.
    """

    # 2. Codificamos los valores que el modelo necesita como números
    brand_id = encode_brand(inputs["brand"])
    fuel_id = encode_fuel_type(inputs["fuel_type"])
    trans_id = encode_transmission(inputs["transmission"])
    model_id = encode_model(inputs["model"])

    # 3. Preparamos los datos como un array en el mismo orden que se entrenó el modelo
    feature_names = [
    "age", "milage", "accident", "engine_hp",
    "brand_id", "model_id", "fuel_type_id", "transmission_norm"
]

    feature_values = [[
    inputs["age"],
    inputs["milage"],
    inputs["accident"],
    inputs["engine_hp"],
    encode_brand(inputs["brand"]),
    encode_model(inputs["model"]),
    encode_fuel_type(inputs["fuel_type"]),
    encode_transmission(inputs["transmission"])
]]

    features_df = pd.DataFrame(feature_values, columns=feature_names)
    # 4. Predecimos con el modelo cargado
    prediction = model.predict(features_df)

    # 5. Devolvemos el resultado como número flotante
    return float(prediction[0])