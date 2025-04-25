import joblib
import pandas as pd
import numpy as np
from utils import (
    encode_brand,
    encode_model,
    encode_fuel_type,
    encode_transmission
)

# Cargamos el modelo entrenado con log(price)
model = joblib.load("models/random_forest_best.pkl")

def predict_price(inputs: dict) -> float:
    """
    Recibe un diccionario con los datos del coche,
    aplica codificación, crea un DataFrame con nombres de columnas,
    predice el logaritmo del precio y lo revierte con np.exp().
    """

    # Codificar valores
    brand_id = encode_brand(inputs["brand"])
    model_id = encode_model(inputs["model"])
    fuel_id = encode_fuel_type(inputs["fuel_type"])
    trans_id = encode_transmission(inputs["transmission"])

    # Crear DataFrame con nombres de columna
    feature_names = [
        "age", "milage", "accident", "engine_hp",
        "brand_id", "model_id", "fuel_type_id", "transmission_norm"
    ]

    feature_values = [[
        inputs["age"],
        inputs["milage"],
        inputs["accident"],
        inputs["engine_hp"],
        brand_id,
        model_id,
        fuel_id,
        trans_id
    ]]

    features_df = pd.DataFrame(feature_values, columns=feature_names)

    # Predecir en log-escala y revertir con np.exp()
    pred_log = model.predict(features_df)
    pred_real = np.exp(pred_log)

    return float(pred_real[0])
