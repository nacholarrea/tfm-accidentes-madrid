"""Construccion, serializacion y uso del modelo final."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC

from .features import add_advanced_features


RANDOM_STATE = 42
TARGET = "lesion_grave"

BASIC_CATEGORICAL_FEATURES = [
    "distrito",
    "tipo_accidente",
    "estado_meteorologico",
    "tipo_vehiculo",
    "tipo_persona",
    "rango_edad",
    "sexo",
]

BASIC_NUMERIC_FEATURES = [
    "hora_sin",
    "hora_cos",
    "dia_sin",
    "dia_cos",
    "mes_sin",
    "mes_cos",
    "es_fin_semana",
    "es_noche",
]

ADVANCED_CATEGORICAL_FEATURES = [
    "distrito",
    "tipo_accidente",
    "estado_meteorologico",
    "tipo_vehiculo",
    "tipo_persona",
    "rango_edad",
    "sexo",
    "grupo_vehiculo",
    "contexto_persona_accidente",
    "contexto_usuario_entorno",
]

ADVANCED_NUMERIC_FEATURES = [
    "hora_dia",
    "dia_semana",
    "mes",
    "es_fin_semana",
    "es_noche",
    "hora_sin",
    "hora_cos",
    "dia_sin",
    "dia_cos",
    "mes_sin",
    "mes_cos",
    "edad_aproximada",
    "edad_desconocida",
    "usuario_vulnerable",
    "meteorologia_adversa",
]

BASE_INPUT_COLUMNS = [
    "distrito",
    "tipo_accidente",
    "estado_meteorologico",
    "tipo_vehiculo",
    "tipo_persona",
    "rango_edad",
    "sexo",
    "hora_dia",
    "dia_semana",
    "mes",
]

SUPPORTED_FAMILIES = {
    "logistic_basic",
    "logistic_advanced",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
}

SUPPORTED_CALIBRATION_METHODS = {"sigmoid_logit"}


def _add_model_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Crea todas las variables necesarias desde las columnas de entrada."""

    result = dataframe.copy()
    result["es_fin_semana"] = result["dia_semana"].isin([5, 6]).astype("int8")
    result["es_noche"] = result["hora_dia"].isin([0, 1, 2, 3, 4, 5]).astype("int8")
    return add_advanced_features(result)


def _selected_categorical_features(include_sex: bool) -> list[str]:
    return [
        column
        for column in ADVANCED_CATEGORICAL_FEATURES
        if include_sex or column != "sexo"
    ]


def required_input_columns(model_configuration: dict[str, Any]) -> list[str]:
    """Devuelve la union ordenada de columnas necesarias para predecir."""

    if model_configuration.get("kind", "single") == "blend":
        first = required_input_columns(model_configuration["first_model"])
        second = required_input_columns(model_configuration["second_model"])
        return list(dict.fromkeys(first + second))

    columns = list(BASE_INPUT_COLUMNS)
    if not model_configuration.get("include_sex", True):
        columns.remove("sexo")
    return columns


def _normalize_parameters(parameters: dict[str, Any] | None) -> dict[str, Any]:
    normalized = deepcopy(parameters or {})
    class_weight = normalized.get("class_weight")
    if isinstance(class_weight, dict):
        normalized["class_weight"] = {
            int(key) if str(key).isdigit() else key: value
            for key, value in class_weight.items()
        }
    return normalized


def _validate_calibration_configuration(
    calibration_configuration: dict[str, Any] | None,
) -> None:
    if calibration_configuration is None:
        return
    if not isinstance(calibration_configuration, dict):
        raise ValueError("La configuracion de calibracion debe ser un diccionario")

    method = calibration_configuration.get("method")
    if method not in SUPPORTED_CALIBRATION_METHODS:
        raise ValueError(f"Metodo de calibracion no soportado: {method}")

    for name in ("coefficient", "intercept"):
        value = calibration_configuration.get(name)
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"Falta un valor valido para calibration.{name}")

    if float(calibration_configuration["coefficient"]) <= 0:
        raise ValueError("El coeficiente de calibracion debe ser positivo")


def apply_probability_calibration(
    probabilities,
    calibration_configuration: dict[str, Any] | None,
) -> np.ndarray:
    """Aplica la transformacion probabilistica congelada en validacion."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1:
        raise ValueError("Las probabilidades deben formar un vector")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Las probabilidades deben ser finitas y estar entre cero y uno")
    if calibration_configuration is None:
        return values.copy()

    _validate_calibration_configuration(calibration_configuration)
    epsilon = 1e-6
    clipped = np.clip(values, epsilon, 1 - epsilon)
    logits = np.log(clipped / (1 - clipped))
    linear_predictor = (
        float(calibration_configuration["coefficient"]) * logits
        + float(calibration_configuration["intercept"])
    )

    calibrated = np.empty_like(linear_predictor)
    nonnegative = linear_predictor >= 0
    calibrated[nonnegative] = 1 / (1 + np.exp(-linear_predictor[nonnegative]))
    exponential = np.exp(linear_predictor[~nonnegative])
    calibrated[~nonnegative] = exponential / (1 + exponential)
    return calibrated


def _one_hot_preprocessor(include_sex: bool, drop: str | None) -> ColumnTransformer:
    categorical = _selected_categorical_features(include_sex)
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="No consta"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=100,
                                drop=drop,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ADVANCED_NUMERIC_FEATURES,
            ),
        ]
    )


def _basic_preprocessor(include_sex: bool) -> ColumnTransformer:
    categorical = [
        column
        for column in BASIC_CATEGORICAL_FEATURES
        if include_sex or column != "sexo"
    ]
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="No consta"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=100,
                                drop="first",
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                BASIC_NUMERIC_FEATURES,
            ),
        ]
    )


def _ordinal_preprocessor(include_sex: bool) -> ColumnTransformer:
    categorical = _selected_categorical_features(include_sex)
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="No consta"),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                ADVANCED_NUMERIC_FEATURES,
            ),
        ]
    )


def _feature_step() -> FunctionTransformer:
    return FunctionTransformer(_add_model_features, validate=False)


def build_single_estimator(model_configuration: dict[str, Any]) -> BaseEstimator:
    """Construye una familia individual con los parametros ya seleccionados."""

    family = model_configuration["family"]
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Familia no soportada: {family}")

    include_sex = bool(model_configuration.get("include_sex", True))
    parameters = _normalize_parameters(model_configuration.get("parameters"))

    if family in {"logistic_basic", "logistic_advanced"}:
        defaults = {
            "max_iter": 1500,
            "solver": "lbfgs",
            "random_state": RANDOM_STATE,
        }
        defaults.update(parameters)
        preprocessor = (
            _basic_preprocessor(include_sex)
            if family == "logistic_basic"
            else _one_hot_preprocessor(include_sex, drop="first")
        )
        model = LogisticRegression(**defaults)

    elif family == "linear_svm":
        calibration_cv = int(parameters.pop("calibration_cv", 3))
        calibration_ensemble = parameters.pop("calibration_ensemble", False)
        defaults = {
            "dual": "auto",
            "max_iter": 5000,
            "random_state": RANDOM_STATE,
        }
        defaults.update(parameters)
        base_pipeline = Pipeline(
            [
                ("features", _feature_step()),
                ("preprocessor", _one_hot_preprocessor(include_sex, drop="first")),
                ("model", LinearSVC(**defaults)),
            ]
        )
        return CalibratedClassifierCV(
            estimator=base_pipeline,
            method="sigmoid",
            cv=calibration_cv,
            ensemble=calibration_ensemble,
        )

    elif family == "rbf_svm":
        calibration_cv = int(parameters.pop("calibration_cv", 2))
        calibration_ensemble = parameters.pop("calibration_ensemble", True)
        defaults = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": None,
            "probability": False,
            "cache_size": 2000,
        }
        defaults.update(parameters)
        base_pipeline = Pipeline(
            [
                ("features", _feature_step()),
                ("preprocessor", _one_hot_preprocessor(include_sex, drop=None)),
                ("model", SVC(**defaults)),
            ]
        )
        return CalibratedClassifierCV(
            estimator=base_pipeline,
            method="sigmoid",
            cv=calibration_cv,
            ensemble=calibration_ensemble,
        )

    elif family == "random_forest":
        defaults = {
            "bootstrap": True,
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        }
        defaults.update(parameters)
        preprocessor = _one_hot_preprocessor(include_sex, drop=None)
        model = RandomForestClassifier(**defaults)

    elif family == "hist_gradient_boosting":
        categorical = _selected_categorical_features(include_sex)
        defaults = {
            "categorical_features": list(range(len(categorical))),
            "early_stopping": False,
            "random_state": RANDOM_STATE,
        }
        defaults.update(parameters)
        preprocessor = _ordinal_preprocessor(include_sex)
        model = HistGradientBoostingClassifier(**defaults)

    else:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost no esta instalado. Ejecuta la instalacion de requirements.txt"
            ) from exc

        defaults = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "verbosity": 0,
        }
        defaults.update(parameters)
        preprocessor = _one_hot_preprocessor(include_sex, drop=None)
        model = XGBClassifier(**defaults)

    return Pipeline(
        [
            ("features", _feature_step()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


class ProbabilityBlendClassifier(ClassifierMixin, BaseEstimator):
    """Promedia probabilidades de dos clasificadores ya especificados."""

    def __init__(
        self,
        first_estimator: BaseEstimator,
        second_estimator: BaseEstimator,
        first_weight: float = 0.5,
    ):
        self.first_estimator = first_estimator
        self.second_estimator = second_estimator
        self.first_weight = first_weight

    def fit(self, X, y):
        if not 0 <= self.first_weight <= 1:
            raise ValueError("first_weight debe estar entre cero y uno")
        self.first_estimator_ = clone(self.first_estimator).fit(X, y)
        self.second_estimator_ = clone(self.second_estimator).fit(X, y)
        self.classes_ = np.asarray(self.first_estimator_.classes_)
        if not np.array_equal(self.classes_, self.second_estimator_.classes_):
            raise ValueError("Los modelos del blend no comparten las mismas clases")
        return self

    def predict_proba(self, X):
        first = self.first_estimator_.predict_proba(X)
        second = self.second_estimator_.predict_proba(X)
        return self.first_weight * first + (1 - self.first_weight) * second

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def build_final_estimator(model_configuration: dict[str, Any]) -> BaseEstimator:
    """Construye un modelo individual o un blend previamente congelado."""

    kind = model_configuration.get("kind", "single")
    if kind == "single":
        return build_single_estimator(model_configuration)
    if kind != "blend":
        raise ValueError(f"Tipo de modelo no soportado: {kind}")

    return ProbabilityBlendClassifier(
        first_estimator=build_final_estimator(model_configuration["first_model"]),
        second_estimator=build_final_estimator(model_configuration["second_model"]),
        first_weight=float(model_configuration["first_weight"]),
    )


def validate_final_configuration(configuration: dict[str, Any]) -> None:
    """Impide abrir el test con una configuracion incompleta."""

    if configuration.get("confirmed") is not True:
        raise RuntimeError(
            "Confirma primero el modelo y el umbral obtenidos en el notebook 04"
        )

    threshold = configuration.get("threshold")
    if threshold is None or not np.isfinite(float(threshold)):
        raise ValueError("Debe indicarse el umbral congelado del notebook 04")
    if not 0 < float(threshold) < 1:
        raise ValueError("El umbral debe estar entre cero y uno")

    model_configuration = configuration.get("model")
    if not isinstance(model_configuration, dict):
        raise ValueError("Falta la configuracion del modelo final")

    required_input_columns(model_configuration)
    build_final_estimator(model_configuration)
    _validate_calibration_configuration(configuration.get("calibration"))


def create_artifact(
    estimator: BaseEstimator,
    configuration: dict[str, Any],
    training_rows: int,
    library_versions: dict[str, str],
) -> dict[str, Any]:
    """Agrupa el modelo y su informacion operativa en un unico objeto."""

    model_configuration = deepcopy(configuration["model"])
    return {
        "artifact_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "estimator": estimator,
        "threshold": float(configuration["threshold"]),
        "calibration": deepcopy(configuration.get("calibration")),
        "model_configuration": model_configuration,
        "required_columns": required_input_columns(model_configuration),
        "target": TARGET,
        "positive_class": 1,
        "training_period": "2019 a 2024",
        "test_year": 2025,
        "training_rows": int(training_rows),
        "library_versions": dict(library_versions),
    }


def save_artifact(artifact: dict[str, Any], path: str | Path) -> Path:
    """Guarda un artefacto en una ruta local."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return output_path


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Carga un artefacto de confianza y comprueba su estructura minima."""

    artifact = joblib.load(Path(path))
    required_keys = {
        "artifact_version",
        "estimator",
        "threshold",
        "required_columns",
        "positive_class",
    }
    missing = required_keys.difference(artifact)
    if missing:
        raise ValueError(f"El artefacto no contiene estas claves: {sorted(missing)}")
    return artifact


def _validate_prediction_input(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("La entrada debe ser un DataFrame de pandas")
    if dataframe.empty:
        raise ValueError("La entrada no contiene observaciones")

    missing = set(required_columns).difference(dataframe.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    selected = dataframe.loc[:, required_columns].copy()
    ranges = {
        "hora_dia": (0, 23),
        "dia_semana": (0, 6),
        "mes": (1, 12),
    }
    for column, (minimum, maximum) in ranges.items():
        values = pd.to_numeric(selected[column], errors="coerce")
        invalid = values.notna() & ~values.between(minimum, maximum)
        if invalid.any():
            raise ValueError(
                f"La columna {column} contiene valores fuera del intervalo esperado"
            )
    return selected


def predict_accident_severity(
    dataframe: pd.DataFrame,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    """Devuelve probabilidad y decision para observaciones nuevas."""

    required_columns = list(artifact["required_columns"])
    selected = _validate_prediction_input(dataframe, required_columns)
    estimator = artifact["estimator"]
    classes = np.asarray(estimator.classes_)
    positive_class = artifact["positive_class"]
    positive_positions = np.flatnonzero(classes == positive_class)
    if len(positive_positions) != 1:
        raise ValueError("No se encuentra una unica clase positiva en el modelo")

    raw_probabilities = estimator.predict_proba(selected)[:, positive_positions[0]]
    probabilities = apply_probability_calibration(
        raw_probabilities,
        artifact.get("calibration"),
    )
    predictions = (probabilities >= float(artifact["threshold"])).astype("int8")
    return pd.DataFrame(
        {
            "probabilidad_lesion_grave": probabilities,
            "prediccion_lesion_grave": predictions,
        },
        index=dataframe.index,
    )
