
# Informe de Rendimiento - Modelo Random Forest

Evaluado con `test_size=0.2`, los resultados fueron:

- **MAE**: `7376.45`
- **RMSE**: `12864.22`
- **R²**: `0.8114`

## Importancia de las Variables
![Importancia de las variables](outputs/feature_importance.png)

## Distribución de los residuos
![Distribución de los residuos](outputs/residuals_distribution.png)

## Predicción vs Real
![Predicción vs Real](outputs/predicted_vs_real.png)

## Validación Cruzada

- R² por fold: `[0.8106673  0.8223831  0.8102396  0.75141838 0.82542534]`
- R² promedio: `0.8040`
- Desviación estándar: `0.0270`

---

> Modelo guardado como `random_forest_best.pkl` en la carpeta `models/`.
