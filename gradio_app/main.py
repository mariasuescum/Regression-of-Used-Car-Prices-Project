import gradio as gr
from rf_best_prediction import predict_price
from config import brands, models, fuel_types, transmissions
import datetime


# Esta función recoge los valores del usuario, prepara el dict y llama al modelo
def estimate_price_interface(
    brand, model, fuel_type, transmission, year, milage, engine_hp, accident
):
    current_year = datetime.datetime.now().year
    age = current_year - year

    # Creamos el diccionario que espera el modelo
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
    price_buy = predicted_price - 500
    price_sell = predicted_price + 500

    return (
        f"💰 Precio estimado: {predicted_price:.2f} €\n\n"
        f"🛒 Si compras, ofrece ≈ {price_buy:.2f} €\n"
        f"🏷️ Si vendes, sugiere ≈ {price_sell:.2f} €"
    )


# Interfaz Gradio usando 'Blocks' (más flexible que Interface)
with gr.Blocks(title="Tasador Inteligente de Vehículos") as demo:
    gr.Markdown("# 🚗 Tasador de coches de segunda mano")
    gr.Markdown(
        "Introduce los datos del vehículo para estimar su precio en el mercado actual."
    )

    # Fila de selección
    with gr.Row():
        brand = gr.Dropdown(label="Marca", choices=brands)
        model = gr.Dropdown(label="Modelo", choices=models)

    with gr.Row():
        fuel_type = gr.Dropdown(label="Tipo de combustible", choices=fuel_types)
        transmission = gr.Radio(label="Transmisión", choices=transmissions)

    # Fila de valores numéricos
    with gr.Row():
        year = gr.Slider(
            label="Año de matriculación", minimum=2000, maximum=2025, step=1, value=2015
        )
        milage = gr.Number(label="Kilómetros (en miles)", value=60)
        engine_hp = gr.Number(label="Potencia del motor (HP)", value=150)

    # Campo adicional
    accident = gr.Radio(label="¿Ha tenido accidentes?", choices=[0, 1], value=0)

    # Resultado
    output = gr.Textbox(label="Resultado de tasación", lines=5)

    # Botón y conexión con la función
    btn = gr.Button("📊 Estimar precio")
    btn.click(
        fn=estimate_price_interface,
        inputs=[
            brand,
            model,
            fuel_type,
            transmission,
            year,
            milage,
            engine_hp,
            accident,
        ],
        outputs=output,
    )


# Lanza la app localmente (http://localhost:7860)
if __name__ == "__main__":
    demo.launch()
