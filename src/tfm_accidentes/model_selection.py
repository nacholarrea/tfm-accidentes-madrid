"""Utilidades de evaluación temporal para clasificación desbalanceada."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import BaseCrossValidator


class ExpandingYearSplit(BaseCrossValidator):
    """Validación temporal: entrena con años anteriores y valida en uno posterior."""

    def __init__(self, validation_years: Iterable[int]):
        self.validation_years = tuple(int(year) for year in validation_years)
        if not self.validation_years:
            raise ValueError("Debe indicarse al menos un año de validación")

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.validation_years)

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("ExpandingYearSplit necesita los años en groups")

        years = np.asarray(groups)
        for validation_year in self.validation_years:
            train_indices = np.flatnonzero(years < validation_year)
            validation_indices = np.flatnonzero(years == validation_year)
            if not len(train_indices) or not len(validation_indices):
                raise ValueError(
                    f"No hay observaciones suficientes para validar {validation_year}"
                )
            yield train_indices, validation_indices


def binary_classification_metrics(
    y_true, probabilities, threshold: float = 0.5
) -> dict[str, float | int]:
    """Calcula métricas probabilísticas y métricas para un umbral concreto."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype("int8")
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "predicted_positives": int(predictions.sum()),
    }


def select_threshold_for_recall(
    y_true, probabilities, minimum_recall: float = 0.75
) -> float:
    """Maximiza precisión entre los umbrales que alcanzan el recall mínimo."""

    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall debe estar en el intervalo (0, 1]")

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    eligible = np.flatnonzero(recall[:-1] >= minimum_recall)
    if not len(eligible):
        raise ValueError("Ningún umbral alcanza el recall mínimo")
    best_position = eligible[np.argmax(precision[:-1][eligible])]
    return float(thresholds[best_position])


def optimize_blend_weight(
    y_true,
    first_probabilities,
    second_probabilities,
    weights: Iterable[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Busca por PR-AUC el peso del primer modelo en un promedio probabilístico."""

    if weights is None:
        weights = np.linspace(0, 1, 101)
    rows = []
    for weight in weights:
        blended = weight * np.asarray(first_probabilities) + (1 - weight) * np.asarray(
            second_probabilities
        )
        rows.append(
            {"first_weight": float(weight), "pr_auc": average_precision_score(y_true, blended)}
        )
    results = pd.DataFrame(rows)
    best_index = results["pr_auc"].idxmax()
    return float(results.loc[best_index, "first_weight"]), results


def paired_bootstrap_pr_auc_difference(
    y_true,
    challenger_probabilities,
    baseline_probabilities,
    n_resamples: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Intervalo estratificado para la diferencia PR-AUC(challenger - baseline)."""

    y_true = np.asarray(y_true)
    challenger = np.asarray(challenger_probabilities)
    baseline = np.asarray(baseline_probabilities)
    positive_indices = np.flatnonzero(y_true == 1)
    negative_indices = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(random_state)
    differences = np.empty(n_resamples)

    for position in range(n_resamples):
        sample = np.concatenate(
            [
                rng.choice(positive_indices, len(positive_indices), replace=True),
                rng.choice(negative_indices, len(negative_indices), replace=True),
            ]
        )
        differences[position] = average_precision_score(
            y_true[sample], challenger[sample]
        ) - average_precision_score(y_true[sample], baseline[sample])

    alpha = 1 - confidence_level
    return {
        "difference": float(
            average_precision_score(y_true, challenger)
            - average_precision_score(y_true, baseline)
        ),
        "lower": float(np.quantile(differences, alpha / 2)),
        "upper": float(np.quantile(differences, 1 - alpha / 2)),
    }
