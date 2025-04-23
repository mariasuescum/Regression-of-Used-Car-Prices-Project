import joblib
import numpy as np
from utils import encode_brand, encode_fuel_type, encode_transmission

# 1. Cargamos el modelo desde el archivo .pkl UNA sola vez (¡eficiencia!)
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

    # 3. Preparamos los datos como un array en el mismo orden que se entrenó el modelo
    features = np.array(
        [
            [
                inputs["age"],  # edad del coche
                inputs["milage"],  # kilometraje
                inputs["accident"],  # 0 o 1
                inputs["engine_hp"],  # potencia del motor
                brand_id,  # marca codificada
                0,  # model_id (de momento lo ignoramos, ponemos 0)
                fuel_id,  # tipo de combustible codificado
                trans_id,  # transmisión codificada
            ]
        ]
    )

    # 4. Predecimos con el modelo cargado
    prediction = model.predict(features)

    # 5. Devolvemos el resultado como número flotante
    return float(prediction[0])
