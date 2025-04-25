import os
import pandas as pd
from datetime import datetime

def save_prediction_to_csv(inputs: dict, predicted_price: float):
    registro = inputs.copy()
    registro["predicted_price"] = predicted_price
    registro["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_registro = pd.DataFrame([registro])

    if not os.path.exists("data/logs.csv"):
        df_registro.to_csv("data/logs.csv", index=False)
    else:
        df_registro.to_csv("data/logs.csv", mode="a", header=False, index=False)
