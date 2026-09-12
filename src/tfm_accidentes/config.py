"""Configuracion comun del proyecto."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATASET_ID = "300228-0-accidentes-trafico-detalle"
CKAN_PACKAGE_URL = (
    "https://datos.madrid.es/api/3/action/package_show"
    f"?id={DATASET_ID}"
)

START_YEAR = 2019
END_YEAR = 2025

EXPECTED_COLUMNS = {
    "num_expediente",
    "fecha",
    "hora",
    "localizacion",
    "numero",
    "cod_distrito",
    "distrito",
    "tipo_accidente",
    "estado_meteorologico",
    "tipo_vehiculo",
    "tipo_persona",
    "rango_edad",
    "sexo",
    "cod_lesividad",
    "lesividad",
    "coordenada_x_utm",
    "coordenada_y_utm",
    "positiva_alcohol",
    "positiva_droga",
}

SERIOUS_INJURY_CODES = {"03", "04"}
UNKNOWN_INJURY_CODES = {"77"}
VALID_INJURY_CODES = {
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "14",
    "77",
}
