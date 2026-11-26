# Trabajo final - Machine Learning (UCU)
# Tema: Series temporales
# Caso: Fallecidos en siniestros de tránsito en Uruguay (2013-2019)
# Integrantes: Bianca Villalba y Victoria Felser
#
# Cómo ejecutarlo (desde la carpeta del proyecto):
#     pip install -r requirements.txt
#     python main.py
# Los CSV de la UNASEV deben estar en la carpeta "datos/".
# Todos los gráficos y tablas se guardan en la carpeta "resultados/".

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # guarda los gráficos en archivos sin abrir ventanas
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from dateutil.easter import easter
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# Carpetas del proyecto (relativas a este archivo, así funciona en cualquier computadora)
CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_DATOS = CARPETA_PROYECTO / "datos"
CARPETA_RESULTADOS = CARPETA_PROYECTO / "resultados"
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Nombres que les ponemos a las columnas (mismo orden que en los archivos).
# Así todos los años quedan con los mismos nombres.
NOMBRES_COLUMNAS = [
    "fecha_hora",
    "departamento",
    "localidad",
    "jurisdiccion",
    "tipo_siniestro",
    "vehiculo",
    "rol",
    "edad",
    "sexo",
    "dias_hasta_fallecer",
    "otro_vehiculo",
    "lugar",
    "x",
    "y",
]

# Buscamos TODOS los archivos .csv que hay en la carpeta datos
archivos = sorted(list(CARPETA_DATOS.glob("datos-fallecidos-*.csv")) + list(CARPETA_DATOS.glob("fallecidos-nueva-*.csv")))
print("Archivos encontrados:", len(archivos))
for a in archivos:
    print(" - ", a.name)
if len(archivos) != 7:
    raise FileNotFoundError(f"Se esperaban 7 archivos CSV (2013-2019) en {CARPETA_DATOS} y se encontraron {len(archivos)}.")

# Leemos cada archivo y lo guardamos en una lista
tablas = []
for archivo in archivos:
    tabla = pd.read_csv(archivo, encoding="latin-1")
    tabla.columns = NOMBRES_COLUMNAS
    print(archivo.name, "->", len(tabla), "filas")
    tablas.append(tabla)

# Juntamos todas las tablas en una sola, una abajo de la otra
datos = pd.concat(tablas, ignore_index=True)
print("Total de filas:", len(datos))
print("Columnas:", list(datos.columns))

# Convertir la columna de fecha (de texto a fecha)
# Hay dos formatos distintos:
#   2013-2018: "01/01/2013 23:15:00"  (día/mes/año)
#   2019:      "2019-01-02 13:30:00"  (año-mes-día)
texto_fecha = datos["fecha_hora"]   # guardamos el texto original

# 1) Probamos el formato de 2013-2018
fechas = pd.to_datetime(texto_fecha, format="%d/%m/%Y %H:%M:%S", errors="coerce")

# 2) Las que quedaron vacías, las probamos con el formato de 2019
vacias = fechas.isna()
fechas.loc[vacias] = pd.to_datetime(texto_fecha[vacias], format="%Y-%m-%d %H:%M:%S", errors="coerce")

datos["fecha_hora"] = fechas
print("Fechas que no se pudieron leer:", datos["fecha_hora"].isna().sum())

# Corregir las fechas con año 2020
# Son 6 filas que vienen del archivo de 2019, pero dicen año 2020 (error de tipeo).
es_2020 = datos["fecha_hora"].dt.year == 2020
print("Filas con año 2020:")
print(datos[es_2020][["fecha_hora", "departamento", "lugar"]])

# Les restamos 1 año: pasan de diciembre 2020 a diciembre 2019
datos.loc[es_2020, "fecha_hora"] = datos.loc[es_2020, "fecha_hora"] - pd.DateOffset(years=1)

# ¿Cuántos fallecidos hay en cada año?
print("Fallecidos por año:")
print(datos["fecha_hora"].dt.year.value_counts().sort_index())

# Buscar filas repetidas
repetidas = datos.duplicated(keep=False)
print("Filas repetidas:")
print(datos[repetidas][["fecha_hora", "departamento", "vehiculo", "rol", "edad", "sexo"]])

# Eliminamos las copias (se queda una de cada pareja)
print("Filas antes:", len(datos))
datos = datos.drop_duplicates().reset_index(drop=True)
print("Filas después de eliminar repetidas:", len(datos))

# Revisar las categorías de algunas columnas
for columna in ["jurisdiccion", "tipo_siniestro", "vehiculo", "rol", "sexo"]:
    print("\n---", columna, "---")
    print(datos[columna].value_counts(dropna=False))

# Valores faltantes (se mantienen porque no afectan el conteo de fallecidos)
print("\nRegistros sin sexo:", datos["sexo"].isna().sum())
print("Registros sin edad:", datos["edad"].isna().sum())

# Unificamos una categoría escrita de dos formas distintas
datos["vehiculo"] = datos["vehiculo"].replace({"CUADRICICLO": "CUATRICICLO"})

print(datos["vehiculo"].value_counts())

# Ordenamos por fecha (en series temporales el ORDEN importa)
datos = datos.sort_values("fecha_hora").reset_index(drop=True)

# Resumen final de los datos limpios
print("\n===== RESUMEN DE LOS DATOS LIMPIOS =====")
print("Total de fallecidos:", len(datos))
print("Desde:", datos["fecha_hora"].min())
print("Hasta:", datos["fecha_hora"].max())
print("Fallecidos por año:")
print(datos["fecha_hora"].dt.year.value_counts().sort_index())

# Guardamos la tabla limpia en un archivo nuevo en la carpeta de resultados
datos.to_csv(CARPETA_RESULTADOS / "fallecidos_limpio.csv", index=False)
print("Datos limpios guardados en:", CARPETA_RESULTADOS / "fallecidos_limpio.csv")

# ARMAR LA SERIE SEMANAL
# Contamos cuántos fallecidos hubo en cada semana (de lunes a domingo; cada semana se identifica por su domingo).
serie_semanal = datos.set_index("fecha_hora").resample("W").size()

# La primera y la última semana están incompletas
# (2013 empezó un martes y 2019 terminó un martes), así que las sacamos.
serie_semanal = serie_semanal.iloc[1:-1]

print("\n SERIE SEMANAL")
print("Cantidad de semanas:", len(serie_semanal))
print("Primeras semanas:")
print(serie_semanal.head())
print("Últimas semanas:")
print(serie_semanal.tail())

# Guardamos la serie semanal en un archivo
serie_semanal.to_csv(CARPETA_RESULTADOS / "serie_semanal.csv", header=["fallecidos"])
print("Serie semanal guardada en:", CARPETA_RESULTADOS / "serie_semanal.csv")

# ANÁLISIS EXPLORATORIO
# Gráfico 1: serie semanal y tendencia (media móvil centrada de 52 semanas)
plt.figure(figsize=(12, 5))
plt.plot(serie_semanal, label="Fallecidos por semana", color="steelblue", linewidth=1)
tendencia = serie_semanal.rolling(window=52, center=True).mean()
print("Tendencia - primer valor:", round(tendencia.dropna().iloc[0], 1))
print("Tendencia - último valor:", round(tendencia.dropna().iloc[-1], 1))
plt.plot(tendencia, label="Tendencia (promedio móvil de 52 semanas)", color="red", linewidth=2.5)
plt.title("Fallecidos en siniestros de tránsito por semana (Uruguay, 2013-2019)")
plt.xlabel("Semana")
plt.ylabel("Cantidad de fallecidos")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CARPETA_RESULTADOS / "01_serie_semanal.png", dpi=150)
plt.close()
print("Gráfico guardado en:", CARPETA_RESULTADOS / "01_serie_semanal.png")

# Gráfico 2: estacionalidad - promedio de fallecidos por semana según el mes
promedio_por_mes = serie_semanal.groupby(serie_semanal.index.month).mean()
print("\nPromedio de fallecidos por semana, según el mes:")
print(promedio_por_mes.round(1))

nombres_meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]

plt.figure(figsize=(10, 5))
plt.bar(nombres_meses, promedio_por_mes, color="steelblue")
plt.axhline(serie_semanal.mean(), color="red", linestyle="--", label="Promedio general")
plt.title("Promedio de fallecidos por semana según el mes (2013-2019)")
plt.xlabel("Mes")
plt.ylabel("Fallecidos por semana (promedio)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(CARPETA_RESULTADOS / "02_estacionalidad_mes.png", dpi=150)
plt.close()
print("Gráfico guardado en:", CARPETA_RESULTADOS / "02_estacionalidad_mes.png")

# Gráfico 3: fallecidos según el día de la semana
por_dia = datos["fecha_hora"].dt.dayofweek.value_counts().sort_index()
print("\nFallecidos según el día de la semana (total 2013-2019):")
print(por_dia)

nombres_dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

plt.figure(figsize=(9, 5))
plt.bar(nombres_dias, por_dia, color="steelblue")
plt.axhline(por_dia.mean(), color="red", linestyle="--", label="Promedio")
plt.title("Fallecidos en siniestros de tránsito según el día de la semana (2013-2019)")
plt.xlabel("Día de la semana")
plt.ylabel("Cantidad de fallecidos (total)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(CARPETA_RESULTADOS / "03_dia_semana.png", dpi=150)
plt.close()
print("Gráfico guardado en:", CARPETA_RESULTADOS / "03_dia_semana.png")

# Gráfico 4: descomposición de la serie
descomposicion = seasonal_decompose(serie_semanal, model="additive", period=52)

fig, ejes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

ejes[0].plot(serie_semanal, color="steelblue")
ejes[0].set_title("Serie original (fallecidos por semana)")

ejes[1].plot(descomposicion.trend, color="red")
ejes[1].set_title("Tendencia")

ejes[2].plot(descomposicion.seasonal, color="green")
ejes[2].set_title("Estacionalidad (se repite cada 52 semanas)")

ejes[3].scatter(descomposicion.resid.index, descomposicion.resid, color="gray", s=10)
ejes[3].axhline(0, color="black", linewidth=1)
ejes[3].set_title("Ruido (lo que queda sin explicar)")

for eje in ejes:
    eje.grid(alpha=0.3)

plt.suptitle("Descomposición de la serie semanal de fallecidos (2013-2019)", fontsize=14)
plt.tight_layout()
plt.savefig(CARPETA_RESULTADOS / "04_descomposicion.png", dpi=150)
plt.close()
print("Gráfico guardado en:", CARPETA_RESULTADOS / "04_descomposicion.png")

print("\nCuánto varía cada componente (desvío estándar):")
print("  Tendencia:      ", round(descomposicion.trend.std(), 2))
print("  Estacionalidad: ", round(descomposicion.seasonal.std(), 2))
print("  Ruido:          ", round(descomposicion.resid.std(), 2))

# MODELOS
entrenamiento = serie_semanal[serie_semanal.index.year <= 2018]
prueba = serie_semanal[serie_semanal.index.year == 2019]

print("\nDIVISIÓN ENTRENAMIENTO / PRUEBA ")
print("Entrenamiento:", len(entrenamiento), "semanas, desde", entrenamiento.index.min().date(), "hasta", entrenamiento.index.max().date())
print("Prueba:       ", len(prueba), "semanas, desde", prueba.index.min().date(), "hasta", prueba.index.max().date())

# MODELOS BASE
pred_persistencia = serie_semanal.shift(1)[prueba.index]
pred_persistencia_estacional = serie_semanal.shift(52)[prueba.index]
pred_promedio = pd.Series(entrenamiento.mean(), index=prueba.index)

def calcular_mae(real, prediccion):
    return (real - prediccion).abs().mean()

print("\nMODELOS BASE: error en 2019 (MAE)")
print("Persistencia:            ", round(calcular_mae(prueba, pred_persistencia), 2))
print("Persistencia estacional: ", round(calcular_mae(prueba, pred_persistencia_estacional), 2))
print("Promedio:                ", round(calcular_mae(prueba, pred_promedio), 2))

# SARIMAX
def crear_variables_calendario(semanas):
    variables = pd.DataFrame(index=semanas)
    dia_del_anio = semanas.dayofyear
    variables["seno_anual"] = np.sin(2 * np.pi * dia_del_anio / 365.25)
    variables["coseno_anual"] = np.cos(2 * np.pi * dia_del_anio / 365.25)

    semana_fiestas = []
    semana_turismo = []
    for domingo in semanas:
        lunes = domingo - pd.Timedelta(days=6)
        navidad = pd.Timestamp(domingo.year, 12, 25)
        anio_nuevo = pd.Timestamp(domingo.year, 1, 1)
        tiene_fiestas = (lunes <= navidad <= domingo) or (lunes <= anio_nuevo <= domingo)
        semana_fiestas.append(int(tiene_fiestas))

        # La Semana de Turismo va de lunes a domingo y termina el domingo de Pascua
        pascua = pd.Timestamp(easter(domingo.year))
        semana_turismo.append(int(domingo == pascua))

    variables["semana_fiestas"] = semana_fiestas
    variables["semana_turismo"] = semana_turismo
    return variables

variables_x = crear_variables_calendario(serie_semanal.index)
print("\nVariables de calendario (primeras filas):")
print(variables_x.head())
print("Semanas con fiestas:", variables_x["semana_fiestas"].sum())
print("Semanas de Turismo: ", variables_x["semana_turismo"].sum())

x_entrenamiento = variables_x.loc[entrenamiento.index]
x_prueba = variables_x.loc[prueba.index]

# Comparamos varias configuraciones (p, d, q) usando SOLO los datos de entrenamiento.
# Usamos el AIC (cuanto más bajo, mejor): mide qué tan bien se ajusta el modelo y
# penaliza la cantidad de parámetros. Así no usamos 2019 para elegir el modelo.
print("\nSARIMAX: comparación de configuraciones (solo entrenamiento 2013-2018)")
comparacion_ordenes = []
for orden in [(0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1), (1, 1, 2), (1, 0, 1)]:
    ajuste = SARIMAX(entrenamiento, exog=x_entrenamiento, order=orden).fit(disp=False)
    comparacion_ordenes.append({"orden": str(orden), "AIC": round(ajuste.aic, 1)})
comparacion_ordenes = pd.DataFrame(comparacion_ordenes).sort_values("AIC")
print(comparacion_ordenes.to_string(index=False))
comparacion_ordenes.to_csv(CARPETA_RESULTADOS / "sarimax_comparacion_ordenes.csv", index=False)

modelo_sarimax = SARIMAX(entrenamiento, exog=x_entrenamiento, order=(1, 1, 1))
resultado_sarimax = modelo_sarimax.fit(disp=False)

# Predicción a un paso: agregamos 2019 al modelo SIN volver a entrenarlo (refit=False),
# así cada semana se predice con los datos reales hasta la semana anterior.
resultado_2019 = resultado_sarimax.append(prueba, exog=x_prueba, refit=False)
pred_sarimax = resultado_2019.predict(start=prueba.index[0], end=prueba.index[-1])

print("\n SARIMAX: error en 2019 (MAE)")
print("SARIMAX:", round(calcular_mae(prueba, pred_sarimax), 2))

print("\nEfecto de cada variable (coeficientes aprendidos) y su p-valor:")
print(pd.DataFrame({"coeficiente": resultado_sarimax.params, "p_valor": resultado_sarimax.pvalues}).round(3))

# Gráfico 5
plt.figure(figsize=(12, 5))
plt.plot(prueba, label="Real", color="black", linewidth=1.5, marker="o", markersize=3)
plt.plot(pred_sarimax, label=f"SARIMAX (MAE {calcular_mae(prueba, pred_sarimax):.2f})", color="red", linewidth=2)
plt.plot(pred_promedio, label=f"Promedio (MAE {calcular_mae(prueba, pred_promedio):.2f})", color="gray", linestyle="--")
plt.title("Fallecidos por semana en 2019: valores reales vs. predicciones")
plt.xlabel("Semana")
plt.ylabel("Cantidad de fallecidos")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CARPETA_RESULTADOS / "05_predicciones_2019.png", dpi=150)
plt.close()
print("Gráfico guardado en:", CARPETA_RESULTADOS / "05_predicciones_2019.png")

predicciones = pd.DataFrame({
    "real": prueba,
    "persistencia": pred_persistencia,
    "persistencia_estacional": pred_persistencia_estacional,
    "promedio": pred_promedio,
    "sarimax": pred_sarimax,
})
predicciones.to_csv(CARPETA_RESULTADOS / "predicciones_parte1.csv")
print("Predicciones guardadas en:", CARPETA_RESULTADOS / "predicciones_parte1.csv")

# PARTE 2: LIGHTGBM (BOOSTING)
# Transformamos la serie en una tabla: cada fila es una semana y la columna lag_1 tiene
# los fallecidos de la semana anterior. Así LightGBM recibe lo que pasó antes
# para poder predecir los fallecidos de esta semana.

variables = pd.DataFrame({"fallecidos": serie_semanal})
variables["lag_1"] = serie_semanal.shift(1)
print(variables.head())

calendario = crear_variables_calendario(serie_semanal.index)
variables = variables.join(calendario)
pd.set_option("display.max_columns", None)
print(variables.head())
print(variables.shape)

# Separamos entrenamiento y prueba

columnas_x = ["lag_1", "seno_anual", "coseno_anual", "semana_fiestas", "semana_turismo"]
datos_entrenamiento = variables[variables.index.year <= 2018].dropna()
datos_prueba = variables[variables.index.year == 2019]

x_entrenamiento_lgb = datos_entrenamiento[columnas_x]
y_entrenamiento_lgb = datos_entrenamiento["fallecidos"]
x_prueba_lgb = datos_prueba[columnas_x]

print("Entrenamiento:", len(datos_entrenamiento), "semanas")
print("Prueba:       ", len(datos_prueba), "semanas")
print("Valores vacíos en prueba:", x_prueba_lgb.isna().sum().sum())

# Elegimos los hiperparámetros de LightGBM con una búsqueda en grilla: se prueban todas
# las combinaciones de la lista con validación cruzada temporal y se elige la de menor error.

grilla = {
    "num_leaves": [3, 7, 15, 31],
    "n_estimators": [50, 100, 200],
    "min_child_samples": [5, 10, 20],
}

cortes = TimeSeriesSplit(n_splits=4)
print("\nCortes de la validación cruzada temporal:")
for numero, (indices_entr, indices_val) in enumerate(cortes.split(x_entrenamiento_lgb), start=1):
    fechas_entr = x_entrenamiento_lgb.index[indices_entr]
    fechas_val = x_entrenamiento_lgb.index[indices_val]
    print(f"  Corte {numero}: entrenamiento {fechas_entr[0].date()} a {fechas_entr[-1].date()} ({len(indices_entr)} semanas)"
          f" | validación {fechas_val[0].date()} a {fechas_val[-1].date()} ({len(indices_val)} semanas)")

busqueda = GridSearchCV(
    lgb.LGBMRegressor(learning_rate=0.05, random_state=42, verbose=-1),
    param_grid=grilla,
    cv=cortes,
    scoring="neg_mean_absolute_error",
)
busqueda.fit(x_entrenamiento_lgb, y_entrenamiento_lgb)

print("Mejores hiperparámetros:", busqueda.best_params_)
print("MAE de validación:", round(-busqueda.best_score_, 2))

# Guardamos el error de validación de las 36 combinaciones
resultados_grilla = pd.DataFrame(busqueda.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
resultados_grilla["mae_validacion"] = -resultados_grilla["mean_test_score"]
resultados_grilla.sort_values("rank_test_score").to_csv(CARPETA_RESULTADOS / "lightgbm_busqueda_grilla.csv", index=False)

# GridSearchCV vuelve a entrenar el mejor modelo con TODOS los datos de 2013-2018
modelo_lgb = busqueda.best_estimator_


# Ahora predecimos 2019 con los valores de la semana anterior:
pred_lgb = pd.Series(modelo_lgb.predict(x_prueba_lgb), index=datos_prueba.index)

print("\nLIGHTGBM: error en 2019 (MAE)")
print("LightGBM:", round(calcular_mae(prueba, pred_lgb), 2))


# Comparamos todos los modelos en una sola tabla:
print("\nCOMPARACIÓN DE TODOS LOS MODELOS (MAE en 2019)")
tabla_mae = pd.Series({
    "Persistencia": calcular_mae(prueba, pred_persistencia),
    "Persistencia estacional": calcular_mae(prueba, pred_persistencia_estacional),
    "Promedio": calcular_mae(prueba, pred_promedio),
    "SARIMAX": calcular_mae(prueba, pred_sarimax),
    "LightGBM": calcular_mae(prueba, pred_lgb),
}).sort_values()
print(tabla_mae.round(2))
tabla_mae.round(2).to_csv(CARPETA_RESULTADOS / "comparacion_mae.csv", header=["MAE_2019"])

# Vemos qué variables usaron más los árboles para predecir y cuántas veces:

importancia = pd.Series(modelo_lgb.feature_importances_, index=columnas_x).sort_values(ascending=False)
print("\nImportancia de cada variable (cuántas veces la usaron los árboles):")
print(importancia)

# Para saber qué tan importante es cada variable usamos la importancia por permutación
# (sobre los datos de 2019): se mezcla una columna y se mide cuánto empeora el MAE.
resultado = permutation_importance(
    modelo_lgb, x_prueba_lgb, prueba,
    scoring="neg_mean_absolute_error",
    n_repeats=30, random_state=42,
)
importancia_perm = pd.Series(resultado.importances_mean, index=columnas_x).sort_values(ascending=False)
print("\nCuánto empeora el MAE si se mezcla cada variable:")
print(importancia_perm.round(3))


# Gráfico 6: lo que pasó en 2019 vs. lo que predijeron los modelos
plt.figure(figsize=(12, 5))
plt.plot(prueba, label="Real", color="black", linewidth=1.5, marker="o", markersize=3)
plt.plot(pred_sarimax, label=f"SARIMAX (MAE {calcular_mae(prueba, pred_sarimax):.2f})", color="red", linewidth=2)
plt.plot(pred_lgb, label=f"LightGBM (MAE {calcular_mae(prueba, pred_lgb):.2f})", color="green", linewidth=2)
plt.plot(pred_promedio, label=f"Promedio (MAE {calcular_mae(prueba, pred_promedio):.2f})", color="gray", linestyle="--")
plt.title("Fallecidos por semana en 2019: SARIMAX vs. LightGBM")
plt.xlabel("Semana")
plt.ylabel("Cantidad de fallecidos")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CARPETA_RESULTADOS / "06_comparacion_modelos.png", dpi=150)
plt.close()
print("Gráfico guardado en:", CARPETA_RESULTADOS / "06_comparacion_modelos.png")


# PRUEBAS ADICIONALES

# 1) Prueba de Ljung-Box sobre los residuos de SARIMAX (entrenamiento).
# Pregunta: ¿quedó en los errores del modelo algún patrón que se podría haber aprovechado?
# Si el p-valor es mayor que 0,05, los residuos se comportan como ruido (no queda patrón).
# Se descarta el primer residuo porque, al diferenciar la serie, la primera semana no tiene anterior.
residuos_sarimax = resultado_sarimax.resid.iloc[1:]
ljung_box = acorr_ljungbox(residuos_sarimax, lags=[4, 8, 13, 26, 52], model_df=2)
ljung_box.index.name = "rezago"
print("\nPrueba de Ljung-Box sobre los residuos de SARIMAX:")
print(ljung_box.round(3))
ljung_box.round(4).to_csv(CARPETA_RESULTADOS / "sarimax_ljung_box.csv")


# 2) Prueba de Diebold-Mariano: ¿la diferencia de error entre SARIMAX y LightGBM es real o puede ser azar?
# Se compara, semana a semana, el error absoluto de cada modelo. Hipótesis nula: ambos predicen igual de bien.
# Se usa la corrección de Harvey, Leybourne y Newbold (1997) para muestras chicas (predicción a un paso).
def prueba_diebold_mariano(real, pred_a, pred_b):
    diferencia = (real - pred_a).abs() - (real - pred_b).abs()
    n = len(diferencia)
    estadistico = diferencia.mean() / np.sqrt(diferencia.var(ddof=0) / n)
    estadistico = estadistico * np.sqrt((n - 1) / n)
    p_valor = 2 * stats.t.sf(abs(estadistico), df=n - 1)
    return diferencia.mean(), estadistico, p_valor

print("\nPrueba de Diebold-Mariano (error absoluto, 2019):")
filas_dm = []
for nombre_a, pred_a, nombre_b, pred_b in [
    ("SARIMAX", pred_sarimax, "LightGBM", pred_lgb),
    ("SARIMAX", pred_sarimax, "Promedio", pred_promedio),
    ("LightGBM", pred_lgb, "Promedio", pred_promedio),
]:
    dif, est, pval = prueba_diebold_mariano(prueba, pred_a, pred_b)
    filas_dm.append({"comparacion": f"{nombre_a} vs. {nombre_b}", "diferencia_mae": round(dif, 3),
                     "estadistico": round(est, 3), "p_valor": round(pval, 3)})
tabla_dm = pd.DataFrame(filas_dm)
print(tabla_dm.to_string(index=False))
tabla_dm.to_csv(CARPETA_RESULTADOS / "diebold_mariano.csv", index=False)


# 3) Error según el tipo de semana: semanas especiales (fiestas o Turismo) vs. el resto.
especiales = (x_prueba["semana_fiestas"] == 1) | (x_prueba["semana_turismo"] == 1)
print("\nSemanas especiales en 2019:", list(prueba.index[especiales].date))
filas_tipo = []
for tipo, mascara in [("Semanas de fiestas o Turismo", especiales), ("Resto de las semanas", ~especiales)]:
    fila = {"tipo_de_semana": tipo, "cantidad": int(mascara.sum()), "promedio_real": round(prueba[mascara].mean(), 2)}
    for nombre, pred in [("SARIMAX", pred_sarimax), ("LightGBM", pred_lgb), ("Promedio", pred_promedio)]:
        fila[f"MAE_{nombre}"] = round(calcular_mae(prueba[mascara], pred[mascara]), 2)
    filas_tipo.append(fila)
tabla_tipo = pd.DataFrame(filas_tipo)
print("\nError según el tipo de semana (2019):")
print(tabla_tipo.to_string(index=False))
tabla_tipo.to_csv(CARPETA_RESULTADOS / "mae_por_tipo_de_semana.csv", index=False)


# Guardamos las predicciones de todos los modelos juntas
predicciones["lightgbm"] = pred_lgb
predicciones.to_csv(CARPETA_RESULTADOS / "predicciones_todos.csv")
print("Predicciones guardadas en:", CARPETA_RESULTADOS / "predicciones_todos.csv")
