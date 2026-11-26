# Predicción de fallecidos en siniestros de tránsito en Uruguay

Trabajo final de Machine Learning (Universidad Católica del Uruguay).
Integrantes: Bianca Villalba y Victoria Felser.

Tema elegido: **series temporales**. Se predice la cantidad semanal de personas fallecidas en
siniestros de tránsito en Uruguay (2013–2019) y se comparan modelos base, SARIMAX y LightGBM.

## Datos

Conjunto «Fallecidos en siniestros de tránsito 2013–2019», publicado por la Unidad Nacional de
Seguridad Vial (UNASEV) en el Catálogo Nacional de Datos Abiertos. Los 7 archivos CSV (uno por año) y
la descripción de las columnas están incluidos en la carpeta `datos/`.

Unidad Nacional de Seguridad Vial. (2019). *Fallecidos en siniestros de tránsito 2013–2019* [Conjunto de
datos]. Catálogo Nacional de Datos Abiertos. https://catalogodatos.gub.uy/dataset/unasev-fallecidos_siniestros_transito

## Estructura

```
.
├── main.py            # todo el proceso: limpieza, serie semanal, análisis exploratorio y modelos
├── requirements.txt   # librerías necesarias
├── datos/             # CSV originales de la UNASEV (2013–2019) y metadatos
└── resultados/        # gráficos, tablas y predicciones generados por main.py
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python main.py
```

En macOS, LightGBM necesita OpenMP: `brew install libomp`.

## Qué genera (`resultados/`)

| Archivo | Contenido |
|---|---|
| `fallecidos_limpio.csv` | Base limpia (3.472 fallecidos) |
| `serie_semanal.csv` | Serie semanal (364 semanas) |
| `01_serie_semanal.png` | Serie y tendencia (media móvil de 52 semanas) |
| `02_estacionalidad_mes.png` | Promedio semanal de fallecidos según el mes |
| `03_dia_semana.png` | Fallecidos según el día de la semana |
| `04_descomposicion.png` | Descomposición en tendencia, estacionalidad y ruido |
| `05_predicciones_2019.png` | SARIMAX vs. valores reales en 2019 |
| `06_comparacion_modelos.png` | SARIMAX vs. LightGBM vs. valores reales en 2019 |
| `sarimax_comparacion_ordenes.csv` | AIC de las configuraciones (p, d, q) probadas |
| `lightgbm_busqueda_grilla.csv` | Error de validación de las 36 combinaciones |
| `comparacion_mae.csv` | MAE en 2019 de todos los modelos |
| `sarimax_ljung_box.csv` | Prueba de Ljung-Box sobre los residuos de SARIMAX |
| `diebold_mariano.csv` | Prueba de Diebold-Mariano entre modelos |
| `mae_por_tipo_de_semana.csv` | MAE en semanas de fiestas/Turismo vs. el resto |
| `predicciones_todos.csv` | Predicciones semanales de 2019 de todos los modelos |

## Resultados (MAE en 2019)

| Modelo | MAE |
|---|---|
| SARIMAX (1, 1, 1) + variables de calendario | 3,00 |
| LightGBM (3 hojas, 50 árboles, 20 semanas por hoja) | 3,20 |
| Promedio histórico | 3,46 |
| Persistencia | 3,77 |
| Persistencia estacional | 4,48 |

La prueba de Diebold-Mariano indica que SARIMAX y LightGBM son significativamente mejores que el promedio
histórico (p = 0,002 y p = 0,005), pero la diferencia entre ellos no es significativa (p = 0,087).
