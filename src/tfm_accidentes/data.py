"""Carga, validacion y preparacion inicial de los datos de accidentes."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    EXPECTED_COLUMNS,
    SERIOUS_INJURY_CODES,
    UNKNOWN_INJURY_CODES,
    VALID_INJURY_CODES,
)


def _normalize_column_name(name: object) -> str:
    """Convierte un nombre de columna a snake_case ASCII."""

    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def read_accidents_csv(path: str | Path) -> pd.DataFrame:
    """Lee un CSV oficial y comprueba que contenga el esquema desde 2019."""

    path = Path(path)
    dataframe = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        dtype="string",
        keep_default_na=True,
        na_values=["NULL", "null"],
        low_memory=False,
    )
    dataframe.columns = [_normalize_column_name(column) for column in dataframe.columns]

    missing_columns = EXPECTED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"El fichero {path.name} no contiene estas columnas: {missing}")

    return dataframe


def load_raw_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Carga y concatena ficheros anuales, conservando el anio de origen."""

    frames: list[pd.DataFrame] = []
    for path_like in sorted(Path(path) for path in paths):
        match = re.search(r"(20\d{2})", path_like.name)
        if match is None:
            raise ValueError(f"No se puede inferir el anio desde {path_like.name}")

        frame = read_accidents_csv(path_like)
        frame["anio_fichero"] = int(match.group(1))
        frames.append(frame)

    if not frames:
        raise ValueError("No se ha proporcionado ningun fichero de accidentes")

    return pd.concat(frames, ignore_index=True)


def read_analysis_table(path: str | Path) -> pd.DataFrame:
    """Lee la tabla procesada recuperando los tipos relevantes para el analisis."""

    dataframe = pd.read_csv(
        path,
        compression="infer",
        parse_dates=["fecha", "fecha_hora"],
        dtype={
            "num_expediente": "string",
            "cod_distrito": "string",
            "cod_lesividad": "string",
        },
        low_memory=False,
    )

    integer_columns = [
        "anio_fichero",
        "anio",
        "mes",
        "dia_semana",
        "es_fin_semana",
        "hora_dia",
        "es_noche",
        "lesion_grave",
    ]
    for column in integer_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").astype("Int64")

    return dataframe


def normalize_injury_code(series: pd.Series) -> pd.Series:
    """Normaliza los codigos de lesividad a dos digitos.

    El diccionario oficial establece que un codigo vacio equivale a ausencia de
    asistencia sanitaria, por lo que se transforma en el codigo 14.
    """

    normalized = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    normalized = normalized.mask(normalized.eq(""))
    normalized = normalized.fillna("14")
    numeric_mask = normalized.str.fullmatch(r"\d+")
    normalized.loc[numeric_mask] = normalized.loc[numeric_mask].str.zfill(2)
    return normalized


def build_analysis_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Crea una tabla tipada con el objetivo y variables temporales basicas."""

    result = dataframe.copy()
    result["cod_lesividad"] = normalize_injury_code(result["cod_lesividad"])

    unexpected_codes = sorted(
        set(result["cod_lesividad"].dropna().unique()).difference(VALID_INJURY_CODES)
    )
    if unexpected_codes:
        raise ValueError(
            "Se han encontrado codigos de lesividad no documentados: "
            f"{unexpected_codes}"
        )

    result["fecha"] = pd.to_datetime(result["fecha"], dayfirst=True, errors="coerce")
    hour_as_timedelta = pd.to_timedelta(result["hora"], errors="coerce")
    result["fecha_hora"] = result["fecha"] + hour_as_timedelta

    result["anio"] = result["fecha"].dt.year.astype("Int64")
    result["mes"] = result["fecha"].dt.month.astype("Int64")
    result["dia_semana"] = result["fecha"].dt.dayofweek.astype("Int64")
    result["es_fin_semana"] = result["dia_semana"].isin([5, 6]).astype("Int8")
    result["hora_dia"] = result["fecha_hora"].dt.hour.astype("Int64")
    result["es_noche"] = result["hora_dia"].isin([0, 1, 2, 3, 4, 5]).astype("Int8")

    result["coordenada_x_utm"] = pd.to_numeric(
        result["coordenada_x_utm"].str.replace(",", ".", regex=False),
        errors="coerce",
    )
    result["coordenada_y_utm"] = pd.to_numeric(
        result["coordenada_y_utm"].str.replace(",", ".", regex=False),
        errors="coerce",
    )

    result["lesividad_conocida"] = ~result["cod_lesividad"].isin(UNKNOWN_INJURY_CODES)
    result["lesion_grave"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    known_mask = result["lesividad_conocida"]
    result.loc[known_mask, "lesion_grave"] = (
        result.loc[known_mask, "cod_lesividad"].isin(SERIOUS_INJURY_CODES).astype("int8")
    )

    # El expediente se conserva como variable de agrupacion, no como predictor.
    result["num_expediente"] = result["num_expediente"].astype("string").str.strip()

    return result


def build_quality_report(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Resume tipo, ausencias, cardinalidad y ejemplos de cada variable."""

    rows: list[dict[str, object]] = []
    total = len(dataframe)
    for column in dataframe.columns:
        series = dataframe[column]
        non_null_examples = series.dropna().astype(str).drop_duplicates().head(3).tolist()
        rows.append(
            {
                "variable": column,
                "tipo": str(series.dtype),
                "n_nulos": int(series.isna().sum()),
                "pct_nulos": float(series.isna().mean() * 100) if total else np.nan,
                "n_unicos": int(series.nunique(dropna=True)),
                "ejemplos": " | ".join(non_null_examples),
            }
        )

    return pd.DataFrame(rows).sort_values(["pct_nulos", "variable"], ascending=[False, True])
