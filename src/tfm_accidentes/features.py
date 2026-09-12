"""Creacion reproducible de variables predictoras para los modelos."""

from __future__ import annotations

import numpy as np
import pandas as pd


CYCLICAL_SOURCE_COLUMNS = {"hora_dia", "dia_semana", "mes"}

AGE_MIDPOINTS = {
    "Menor de 5 años": 2.0,
    "De 6 a 9 años": 7.5,
    "De 10 a 14 años": 12.0,
    "De 15 a 17 años": 16.0,
    "De 18 a 20 años": 19.0,
    "De 21 a 24 años": 22.5,
    "De 25 a 29 años": 27.0,
    "De 30 a 34 años": 32.0,
    "De 35 a 39 años": 37.0,
    "De 40 a 44 años": 42.0,
    "De 45 a 49 años": 47.0,
    "De 50 a 54 años": 52.0,
    "De 55 a 59 años": 57.0,
    "De 60 a 64 años": 62.0,
    "De 65 a 69 años": 67.0,
    "De 70 a 74 años": 72.0,
    "Más de 74 años": 80.0,
}


def add_cyclical_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Representa hora, dia de la semana y mes mediante seno y coseno.

    Esta transformacion no aprende parametros de los datos, por lo que puede
    aplicarse antes de realizar la division temporal sin producir fuga de
    informacion.
    """

    missing_columns = CYCLICAL_SOURCE_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas para crear variables ciclicas: {missing}")

    result = dataframe.copy()
    result["hora_sin"] = np.sin(2 * np.pi * result["hora_dia"] / 24)
    result["hora_cos"] = np.cos(2 * np.pi * result["hora_dia"] / 24)
    result["dia_sin"] = np.sin(2 * np.pi * result["dia_semana"] / 7)
    result["dia_cos"] = np.cos(2 * np.pi * result["dia_semana"] / 7)
    result["mes_sin"] = np.sin(2 * np.pi * (result["mes"] - 1) / 12)
    result["mes_cos"] = np.cos(2 * np.pi * (result["mes"] - 1) / 12)
    return result


def _vehicle_group(value: object) -> str:
    """Agrupa tipos de vehículo poco frecuentes conservando su vulnerabilidad."""

    if pd.isna(value):
        return "No consta"

    text = str(value).casefold()
    if "moto" in text or "ciclomotor" in text:
        return "Motocicleta/ciclomotor"
    if "bicicleta" in text or text == "ciclo":
        return "Bicicleta/ciclo"
    if "vmu" in text or "patinete" in text:
        return "Movilidad personal"
    if any(
        token in text
        for token in ("camión", "tracto", "autobús", "autobus", "articulado", "maquinaria")
    ):
        return "Vehículo pesado"
    if "furgoneta" in text:
        return "Furgoneta"
    if "turismo" in text or "todo terreno" in text:
        return "Turismo"
    return "Otros"


def add_advanced_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Añade variables interpretables disponibles en el momento del accidente.

    No se utilizan lesividad, alcohol o drogas. Las interacciones se construyen
    con variables conocidas tras el accidente y antes de conocer su gravedad.
    """

    required = {
        "hora_dia",
        "dia_semana",
        "mes",
        "rango_edad",
        "tipo_vehiculo",
        "tipo_persona",
        "tipo_accidente",
        "estado_meteorologico",
    }
    missing_columns = required.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas para la ingeniería avanzada: {missing}")

    result = add_cyclical_features(dataframe)
    result["edad_aproximada"] = result["rango_edad"].map(AGE_MIDPOINTS).astype("float64")
    result["edad_desconocida"] = result["edad_aproximada"].isna().astype("int8")
    result["grupo_vehiculo"] = result["tipo_vehiculo"].map(_vehicle_group).astype("string")

    person = result["tipo_persona"].fillna("No consta").astype("string")
    accident = result["tipo_accidente"].fillna("No consta").astype("string")
    weather = result["estado_meteorologico"].fillna("No consta").astype("string")

    vulnerable_groups = {"Motocicleta/ciclomotor", "Bicicleta/ciclo", "Movilidad personal"}
    result["usuario_vulnerable"] = (
        person.str.startswith("Peatón") | result["grupo_vehiculo"].isin(vulnerable_groups)
    ).astype("int8")
    result["meteorologia_adversa"] = weather.str.contains(
        "lluvia|nevando|granizando", case=False, regex=True
    ).astype("int8")
    result["contexto_persona_accidente"] = (person + " | " + accident).astype("string")
    result["contexto_usuario_entorno"] = (
        result["grupo_vehiculo"] + " | " + weather
    ).astype("string")
    return result
