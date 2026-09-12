# TFM - Gravedad de lesiones en accidentes de Madrid

Proyecto inicial para el TFM del perfil Data Scientist. El objetivo es estimar la
probabilidad de lesion grave o mortal para una persona implicada en un accidente de
trafico registrado por la Policia Municipal de Madrid.

## Alcance del proyecto

- Descarga reproducible de los CSV oficiales de 2019 a 2025.
- Comprobacion del esquema comun desde 2019.
- Normalizacion de fechas, horas y codigos de lesividad.
- Creacion documentada de la variable objetivo `lesion_grave`.
- Primer informe de calidad y distribucion de la clase objetivo.
- Comparacion temporal de modelos lineales, bagging, boosting y SVM.
- Seleccion de un XGBoost calibrado y sin la variable sexo.
- Evaluacion final sobre 2025 y serializacion del artefacto.
- Aplicacion Streamlit para prediccion individual y por CSV.

La version actual incluye el modelo final, la evaluacion temporal y una interfaz
para utilizar el artefacto sin ejecutar los notebooks.

## Resultado de la validacion inicial

Ejecucion realizada con los recursos oficiales disponibles el 25 de agosto de 2026:

- 322.317 personas implicadas.
- 137.328 expedientes de accidente.
- 3.796 lesiones graves o fallecimientos.
- Prevalencia de la clase positiva: 1,18 %.
- 13 registros con lesividad desconocida.
- 13.514 filas redundantes bajo una comparacion de todas las columnas.

Las filas aparentemente duplicadas no se eliminan en esta fase. El dataset no
incluye un identificador de persona y dos implicados distintos pueden compartir
todos los atributos publicados. Esta limitacion se analizara y documentara en el
EDA.

## Fuente de datos

- Portal de Datos Abiertos del Ayuntamiento de Madrid.
- Dataset: `300228-0-accidentes-trafico-detalle`.
- Licencia: Creative Commons Attribution 4.0 (CC BY 4.0).
- Unidad de observacion: persona implicada en un accidente.

Los datos no se redistribuyen en este repositorio. El script consulta la API oficial
y descarga los recursos vigentes.

## Preparacion del entorno

Desde la raiz del proyecto:

```bash
python -m venv .venv
```

En Windows:

```powershell
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Linux o macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Descargar y preparar los datos

```bash
python scripts/download_data.py
```

El comando crea:

- `data/raw/accidentes_madrid_YYYY.csv`: copia de cada recurso oficial.
- `data/processed/accidentes_madrid_2019_2025.csv.gz`: tabla unificada y
  preparada para el EDA.
- `data/processed/resumen_calidad.csv`: resumen por variable.
- `data/processed/resumen_inicial.json`: volumen, duplicidad y prevalencia.
- `data/processed/metadata_descarga.json`: procedencia de cada fichero.

Para volver a descargar ficheros existentes:

```bash
python scripts/download_data.py --overwrite
```

## Notebook inicial

Abrir `notebooks/01_ingesta_calidad_objetivo.ipynb` y ejecutar las celdas en orden.

El segundo notebook, `notebooks/02_eda_seleccion_variables.ipynb`, estudia el
desbalance, la unidad de observacion, los valores ausentes y las tasas de gravedad
por grupos. Termina con una propuesta explicita de variables incluidas y excluidas,
ademas de la division temporal de entrenamiento, validacion y test.

El tercero, `notebooks/03_regresion_logistica_baseline.ipynb`, construye un baseline
con regresion logistica, compara ponderacion de clases frente a un modelo trivial,
evalua el resultado en 2024 y selecciona provisionalmente un umbral usando solo esa
validacion. El conjunto de 2025 queda reservado para la comparacion final.

El cuarto, `notebooks/04_modelizacion_avanzada.ipynb`, compara la logistica con
feature engineering, SVM lineal y RBF, Random Forest, HistGradientBoosting y
XGBoost mediante validacion temporal expansiva. Incluye seleccion de
hiperparametros, blend probabilistico, intervalos bootstrap, importancia por
permutacion, auditoria por sexo y edad y calibracion probabilistica. El modelo
seleccionado es un XGBoost sin la variable sexo.

El quinto, `notebooks/05_evaluacion_final_productivizacion_calibrado.ipynb`,
reentrena el modelo con 2019 a 2024, aplica la calibracion congelada y realiza una
unica evaluacion final sobre 2025. Finalmente crea
`models/modelo_final.joblib` y comprueba que la serializacion conserva las
predicciones.

## Aplicacion Streamlit

Guarda el artefacto generado por el notebook 05 en:

```text
models/modelo_final.joblib
```

Desde la raiz del proyecto ejecuta:

```bash
python -m streamlit run app.py
```

La aplicacion ofrece:

- Formulario para estimar una observacion individual.
- Probabilidad calibrada y clasificacion con el umbral congelado.
- Carga de archivos CSV con hasta 50.000 observaciones.
- Descarga de las predicciones generadas.
- Plantilla con las columnas requeridas.
- Informacion sobre configuracion, calibracion y versiones del modelo.

La plantilla tambien se encuentra en `plantilla_prediccion.csv`. La codificacion
de `dia_semana` utiliza cero para lunes y seis para domingo.

El archivo `joblib` solo debe cargarse si procede de una fuente de confianza. Para
evitar incompatibilidades, `requirements.txt` fija las versiones principales con
las que se genero el artefacto final.

## Pruebas

Desde la raiz del proyecto:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Definicion inicial del objetivo

Segun el diccionario oficial:

- `03`: ingreso superior a 24 horas (grave).
- `04`: fallecido en 24 horas.
- `01`, `02`, `05`, `06`, `07`: lesion leve.
- `14` o codigo vacio: sin asistencia sanitaria.
- `77`: se desconoce.

La variable binaria se define como:

```text
lesion_grave = 1 si cod_lesividad pertenece a {03, 04}
lesion_grave = 0 para el resto de codigos conocidos
lesion_grave = NA si cod_lesividad es 77
```

Los registros desconocidos no se emplearan para entrenar ni evaluar modelos. El
identificador `num_expediente` se conservara para impedir que personas del mismo
accidente queden en particiones distintas, pero nunca se utilizara como predictor.
