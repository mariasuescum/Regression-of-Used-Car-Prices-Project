import gradio as gr
import pandas as pd
import datetime
from logger import save_prediction_to_csv


# Cargamos el modelo real de predicción
from rf_best_prediction import predict_price

# Listas visibles del formulario (sin codificar)
from config import brands, fuel_types, transmissions

#  Cargamos el archivo CSV que contiene la relación modelo / marca
model_brand_df = pd.read_csv("data/model_brand_mapping.csv")

# 2 Creamos dos diccionarios auxiliares:
# - model_brand_map: dado un modelo, nos da la marca correspondiente
# - brand_models_map: dada una marca, devuelve la lista de modelos asociados
model_brand_map = dict(model_brand_df[['model', 'brand']].values)
brand_models_map = model_brand_df.groupby('brand')['model'].apply(list).to_dict()

# Esta función se ejecuta al pulsar el botón y lanza la predicción
def estimate_price_interface(brand, model, fuel_type, transmission, year, milage, engine_hp, accident):
    current_year = datetime.datetime.now().year
    age = current_year - year

    inputs = {
        "brand": brand,
        "model": model,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "age": age,
        "milage": milage,
        "engine_hp": engine_hp,
        "accident": accident,
    }

    predicted_price = predict_price(inputs)
    # 💾 Guardar predicción en logs.csv
    save_prediction_to_csv(inputs, predicted_price)
    price_buy = predicted_price - 500
    price_sell = predicted_price + 500

    return (
        f"💰 Precio estimado: {predicted_price:.2f} €\n\n"
        f"🛒 Si compras, ofrece ≈ {price_buy:.2f} €\n"
        f"🏷️ Si vendes, sugiere ≈ {price_sell:.2f} €"
    )

# Función auxiliar: actualiza el dropdown de modelos al cambiar la marca
def update_models(brand):
    modelos = brand_models_map.get(brand, [])
    return gr.update(choices=modelos, value=modelos[0] if modelos else None)

# Función auxiliar: autocompleta la marca al seleccionar un modelo
def update_brand(model):
    marca = model_brand_map.get(model, "")
    return gr.update(value=marca)

# Construcción de la interfaz visual con Gradio
with gr.Blocks(title="Tasador Inteligente de Vehículos") as demo:
    gr.Markdown("# 🚗 Tasador de coches de segunda mano")
    gr.Markdown(
        "Introduce las características del vehículo para estimar su precio en el mercado actual."
    )

    # Marca y modelo (con sincronización dinámica)
    with gr.Row():
        brand = gr.Dropdown(label="Marca", choices=brands)
        model = gr.Dropdown(label="Modelo", choices=[])

    # Tipo de combustible y transmisión
    with gr.Row():
        fuel_type = gr.Dropdown(label="Tipo de combustible", choices=fuel_types)
        transmission = gr.Radio(label="Transmisión", choices=transmissions)

    # Año, kilómetros y potencia
    with gr.Row():
        year = gr.Slider(
            label="Año de matriculación", minimum=2000, maximum=2025, step=1, value=2015
        )
        milage = gr.Number(label="Kilómetros (en miles)", value=60)
        engine_hp = gr.Number(label="Potencia del motor (HP)", value=150)

    # Accidentes
    accident = gr.Radio(label="¿Ha tenido accidentes?", choices=[0, 1], value=0)

    # Resultado final
    output = gr.Textbox(label="Resultado de tasación", lines=5)

    # Botón para lanzar la predicción
    btn = gr.Button("📊 Estimar precio")
    btn.click(
        fn=estimate_price_interface,
        inputs=[brand, model, fuel_type, transmission, year, milage, engine_hp, accident],
        outputs=output,
    )

    # Vinculamos los cambios dinámicos entre campos (sincronización marca ↔ modelo)
    brand.change(fn=update_models, inputs=brand, outputs=model)
    model.change(fn=update_brand, inputs=model, outputs=brand)

# Lanzamos la app en local
if __name__ == "__main__":
    demo.launch(favicon_path="/images/12311044.png", server_name="0.0.0.0", server_port=8081)
