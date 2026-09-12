"""Funciones independientes de Streamlit para la aplicación predictiva."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .production import predict_accident_severity


DISTRICTS = [
    "ARGANZUELA",
    "BARAJAS",
    "CARABANCHEL",
    "CENTRO",
    "CHAMARTÍN",
    "CHAMBERÍ",
    "CIUDAD LINEAL",
    "FUENCARRAL-EL PARDO",
    "HORTALEZA",
    "LATINA",
    "MONCLOA-ARAVACA",
    "MORATALAZ",
    "PUENTE DE VALLECAS",
    "RETIRO",
    "SALAMANCA",
    "SAN BLAS-CANILLEJAS",
    "TETUÁN",
    "USERA",
    "VICÁLVARO",
    "VILLA DE VALLECAS",
    "VILLAVERDE",
]

ACCIDENT_TYPES = [
    "Alcance",
    "Atropello a animal",
    "Atropello a persona",
    "Caída",
    "Choque contra obstáculo fijo",
    "Colisión frontal",
    "Colisión fronto-lateral",
    "Colisión lateral",
    "Colisión múltiple",
    "Despeñamiento",
    "Otro",
    "Solo salida de la vía",
    "Vuelco",
]

WEATHER_STATES = [
    "Despejado",
    "Granizando",
    "LLuvia intensa",
    "Lluvia débil",
    "Nevando",
    "Nublado",
    "Se desconoce",
]

VEHICLE_TYPES = [
    "Ambulancia SAMUR",
    "Autobus EMT",
    "Autobús",
    "Autobús articulado",
    "Autobús articulado EMT",
    "Autocaravana",
    "Bicicleta",
    "Bicicleta EPAC (pedaleo asistido)",
    "Camión de bomberos",
    "Camión rígido",
    "Caravana",
    "Ciclo",
    "Ciclo de motor L1e-A",
    "Ciclomotor",
    "Ciclomotor de dos ruedas L1e-B",
    "Ciclomotor de tres ruedas",
    "Cuadriciclo ligero",
    "Cuadriciclo no ligero",
    "Furgoneta",
    "Maquinaria agrícola",
    "Maquinaria de obras",
    "Microbús <= 17 plazas",
    "Moto de tres ruedas > 125cc",
    "Moto de tres ruedas hasta 125cc",
    "Motocicleta > 125cc",
    "Motocicleta hasta 125cc",
    "Otros vehículos con motor",
    "Otros vehículos sin motor",
    "Patinete",
    "Patinete no eléctrico",
    "Remolque",
    "Semiremolque",
    "Sin especificar",
    "Todo terreno",
    "Tractocamión",
    "Tranvía",
    "Tren/metro",
    "Turismo",
    "VMU eléctrico",
    "Vehículo articulado",
]

PERSON_TYPES = ["Conductor", "Pasajero", "Peatón", "Peatón (atropello sc)"]

AGE_RANGES = [
    "Menor de 5 años",
    "De 6 a 9 años",
    "De 10 a 14 años",
    "De 15 a 17 años",
    "De 18 a 20 años",
    "De 21 a 24 años",
    "De 25 a 29 años",
    "De 30 a 34 años",
    "De 35 a 39 años",
    "De 40 a 44 años",
    "De 45 a 49 años",
    "De 50 a 54 años",
    "De 55 a 59 años",
    "De 60 a 64 años",
    "De 65 a 69 años",
    "De 70 a 74 años",
    "Más de 74 años",
    "Desconocido",
]

SEX_VALUES = ["Hombre", "Mujer", "Se desconoce"]

DAYS = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}

MONTHS = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}

PREDICTION_COLUMNS = [
    "probabilidad_lesion_grave",
    "prediccion_lesion_grave",
]


def build_manual_observation(
    values: dict[str, Any],
    required_columns: list[str],
) -> pd.DataFrame:
    """Construye una observación con el orden exigido por el artefacto."""

    missing = set(required_columns).difference(values)
    if missing:
        raise ValueError(f"Faltan valores para estas columnas: {sorted(missing)}")
    return pd.DataFrame([{column: values[column] for column in required_columns}])


def make_prediction_template(required_columns: list[str]) -> pd.DataFrame:
    """Crea una fila válida que documenta el contrato del CSV."""

    example = {
        "distrito": "CENTRO",
        "tipo_accidente": "Atropello a persona",
        "estado_meteorologico": "Despejado",
        "tipo_vehiculo": "Turismo",
        "tipo_persona": "Peatón",
        "rango_edad": "De 30 a 34 años",
        "sexo": "Mujer",
        "hora_dia": 18,
        "dia_semana": 2,
        "mes": 9,
    }
    return build_manual_observation(example, required_columns)


def predict_batch(
    dataframe: pd.DataFrame,
    artifact: dict[str, Any],
    maximum_rows: int = 50_000,
) -> pd.DataFrame:
    """Valida un lote y añade sus probabilidades y decisiones."""

    if len(dataframe) > maximum_rows:
        raise ValueError(
            f"El archivo supera el máximo de {maximum_rows:,} observaciones"
        )
    base = dataframe.drop(columns=PREDICTION_COLUMNS, errors="ignore").copy()
    predictions = predict_accident_severity(base, artifact)
    return pd.concat([base, predictions], axis=1)


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Genera un CSV UTF 8 compatible con hojas de cálculo."""

    return dataframe.to_csv(index=False).encode("utf-8-sig")
