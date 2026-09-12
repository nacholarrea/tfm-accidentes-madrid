"""Aplicación Streamlit para el modelo de gravedad de lesiones."""

from __future__ import annotations

from importlib.metadata import version
from io import BytesIO
import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tfm_accidentes.app_support import (
    ACCIDENT_TYPES,
    AGE_RANGES,
    DAYS,
    DISTRICTS,
    MONTHS,
    PERSON_TYPES,
    SEX_VALUES,
    VEHICLE_TYPES,
    WEATHER_STATES,
    build_manual_observation,
    dataframe_to_csv_bytes,
    make_prediction_template,
    predict_batch,
)
from tfm_accidentes.production import load_artifact, predict_accident_severity


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "modelo_final.joblib"
MODEL_PATH = Path(os.environ.get("TFM_MODEL_PATH", DEFAULT_MODEL_PATH))


st.set_page_config(
    page_title="Gravedad de lesiones en Madrid",
    page_icon="🚦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando el modelo final")
def load_cached_artifact(path: str, modified_time: int):
    """Carga una sola vez cada versión local del artefacto."""

    del modified_time
    return load_artifact(path)


def read_uploaded_csv(contents: bytes) -> pd.DataFrame:
    """Lee CSV con separador habitual o detectado automáticamente."""

    try:
        return pd.read_csv(BytesIO(contents), sep=None, engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(
            BytesIO(contents),
            sep=None,
            engine="python",
            encoding="latin-1",
        )


def show_prediction(probability: float, prediction: int, threshold: float) -> None:
    first, second, third = st.columns(3)
    first.metric("Probabilidad calibrada", f"{probability:.2%}")
    second.metric("Umbral operativo", f"{threshold:.2%}")
    third.metric("Clasificación", "Riesgo elevado" if prediction else "Riesgo menor")

    if prediction:
        st.warning(
            "La probabilidad estimada alcanza el umbral operativo y el caso queda señalado."
        )
    else:
        st.success(
            "La probabilidad estimada no alcanza el umbral operativo del modelo."
        )


st.title("Estimación de gravedad en accidentes de Madrid")
st.caption(
    "Prototipo de apoyo analítico. La predicción no sustituye la valoración de los servicios de emergencia."
)

if not MODEL_PATH.is_file():
    st.error(f"No se encuentra el artefacto en {MODEL_PATH}")
    st.info("Guarda modelo_final.joblib dentro de la carpeta models y reinicia la aplicación.")
    st.stop()

try:
    artifact = load_cached_artifact(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
except Exception as error:
    st.error("No se pudo cargar el modelo final")
    st.info(
        "Comprueba que instalaste requirements.txt en el mismo entorno desde el que ejecutas Streamlit."
    )
    with st.expander("Detalle técnico"):
        st.exception(error)
    st.stop()

required_columns = list(artifact["required_columns"])
threshold = float(artifact["threshold"])
model_configuration = artifact.get("model_configuration", {})
calibration_configuration = artifact.get("calibration") or {}

with st.sidebar:
    st.header("Modelo desplegado")
    st.write(f"Familia: `{model_configuration.get('family', 'No consta')}`")
    st.write(f"Entrenamiento: {artifact.get('training_period', 'No consta')}")
    st.write(f"Umbral calibrado: {threshold:.2%}")
    st.write(f"Calibración: `{calibration_configuration.get('method', 'No aplicada')}`")
    st.caption(f"Artefacto versión {artifact.get('artifact_version', 'No consta')}")

manual_tab, batch_tab, information_tab = st.tabs(
    ["Predicción individual", "Predicción por CSV", "Información del modelo"]
)

with manual_tab:
    st.subheader("Introducir una persona implicada")
    st.write("Completa los datos conocidos en el momento de realizar la estimación.")

    with st.form("manual_prediction"):
        first, second, third = st.columns(3)
        with first:
            distrito = st.selectbox("Distrito", DISTRICTS, index=DISTRICTS.index("CENTRO"))
            tipo_accidente = st.selectbox("Tipo de accidente", ACCIDENT_TYPES)
            estado_meteorologico = st.selectbox("Meteorología", WEATHER_STATES)
        with second:
            tipo_vehiculo = st.selectbox(
                "Tipo de vehículo", VEHICLE_TYPES, index=VEHICLE_TYPES.index("Turismo")
            )
            tipo_persona = st.selectbox("Tipo de persona", PERSON_TYPES)
            rango_edad = st.selectbox(
                "Rango de edad", AGE_RANGES, index=AGE_RANGES.index("De 30 a 34 años")
            )
        with third:
            hora_dia = st.slider("Hora", min_value=0, max_value=23, value=12)
            dia_semana = st.selectbox(
                "Día de la semana",
                list(DAYS),
                format_func=DAYS.get,
            )
            mes = st.selectbox("Mes", list(MONTHS), format_func=MONTHS.get)
            sexo = None
            if "sexo" in required_columns:
                sexo = st.selectbox("Sexo", SEX_VALUES)

        submitted = st.form_submit_button(
            "Calcular estimación",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        values = {
            "distrito": distrito,
            "tipo_accidente": tipo_accidente,
            "estado_meteorologico": estado_meteorologico,
            "tipo_vehiculo": tipo_vehiculo,
            "tipo_persona": tipo_persona,
            "rango_edad": rango_edad,
            "hora_dia": hora_dia,
            "dia_semana": dia_semana,
            "mes": mes,
        }
        if sexo is not None:
            values["sexo"] = sexo

        try:
            observation = build_manual_observation(values, required_columns)
            result = predict_accident_severity(observation, artifact).iloc[0]
        except Exception as error:
            st.error(f"No se pudo calcular la predicción: {error}")
        else:
            show_prediction(
                float(result["probabilidad_lesion_grave"]),
                int(result["prediccion_lesion_grave"]),
                threshold,
            )
            with st.expander("Datos enviados al modelo"):
                st.dataframe(observation, use_container_width=True, hide_index=True)

with batch_tab:
    st.subheader("Procesar varias observaciones")
    st.write(
        "Descarga la plantilla, conserva los nombres de las columnas y añade una fila por persona implicada."
    )
    template = make_prediction_template(required_columns)
    st.download_button(
        "Descargar plantilla CSV",
        data=dataframe_to_csv_bytes(template),
        file_name="plantilla_prediccion.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Cargar CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_data = read_uploaded_csv(uploaded_file.getvalue())
            batch_result = predict_batch(uploaded_data, artifact)
        except Exception as error:
            st.error(f"No se pudo procesar el archivo: {error}")
        else:
            flagged = int(batch_result["prediccion_lesion_grave"].sum())
            first, second, third = st.columns(3)
            first.metric("Observaciones", f"{len(batch_result):,}")
            second.metric("Casos señalados", f"{flagged:,}")
            third.metric(
                "Probabilidad media",
                f"{batch_result['probabilidad_lesion_grave'].mean():.2%}",
            )
            st.dataframe(batch_result.head(200), use_container_width=True)
            if len(batch_result) > 200:
                st.caption("La vista previa muestra las primeras 200 filas.")
            st.download_button(
                "Descargar predicciones",
                data=dataframe_to_csv_bytes(batch_result),
                file_name="predicciones_lesion_grave.csv",
                mime="text/csv",
                type="primary",
            )

with information_tab:
    st.subheader("Información técnica")
    first, second, third = st.columns(3)
    first.metric("Filas de entrenamiento", f"{artifact.get('training_rows', 0):,}")
    second.metric("Periodo", artifact.get("training_period", "No consta"))
    third.metric("Umbral", f"{threshold:.2%}")

    st.markdown("#### Variables requeridas")
    st.code("\n".join(required_columns), language=None)

    st.markdown("#### Configuración del modelo")
    st.json(model_configuration)

    st.markdown("#### Calibración")
    st.json(calibration_configuration)

    st.markdown("#### Versiones de entrenamiento")
    st.json(artifact.get("library_versions", {}))
    runtime_sklearn = version("scikit-learn")
    trained_sklearn = artifact.get("library_versions", {}).get("scikit_learn")
    if trained_sklearn and runtime_sklearn != trained_sklearn:
        st.warning(
            f"El artefacto se entrenó con scikit learn {trained_sklearn} y se está ejecutando con {runtime_sklearn}."
        )

    st.markdown("#### Limitaciones")
    st.write(
        "El modelo se entrenó con accidentes registrados por la Policía Municipal de Madrid. "
        "Sus asociaciones no tienen una interpretación causal y su rendimiento puede deteriorarse "
        "si cambian la población, el registro de datos o las condiciones de circulación."
    )
