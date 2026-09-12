import unittest

import numpy as np
import pandas as pd

from tfm_accidentes.app_support import (
    PREDICTION_COLUMNS,
    build_manual_observation,
    dataframe_to_csv_bytes,
    make_prediction_template,
    predict_batch,
)


class ConstantEstimator:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, dataframe):
        positive = np.full(len(dataframe), 0.4)
        return np.column_stack([1 - positive, positive])


class StreamlitSupportTests(unittest.TestCase):
    def setUp(self):
        self.required_columns = [
            "distrito",
            "tipo_accidente",
            "estado_meteorologico",
            "tipo_vehiculo",
            "tipo_persona",
            "rango_edad",
            "hora_dia",
            "dia_semana",
            "mes",
        ]
        self.artifact = {
            "estimator": ConstantEstimator(),
            "threshold": 0.2,
            "calibration": None,
            "required_columns": self.required_columns,
            "positive_class": 1,
        }

    def test_template_matches_required_columns(self):
        template = make_prediction_template(self.required_columns)

        self.assertEqual(template.columns.tolist(), self.required_columns)
        self.assertNotIn("sexo", template.columns)

    def test_manual_observation_rejects_missing_values(self):
        with self.assertRaises(ValueError):
            build_manual_observation({"distrito": "CENTRO"}, self.required_columns)

    def test_batch_preserves_input_and_adds_predictions(self):
        template = make_prediction_template(self.required_columns)

        result = predict_batch(template, self.artifact)

        self.assertEqual(result.index.tolist(), template.index.tolist())
        for column in PREDICTION_COLUMNS:
            self.assertIn(column, result.columns)
        np.testing.assert_allclose(result["probabilidad_lesion_grave"], 0.4)
        np.testing.assert_array_equal(result["prediccion_lesion_grave"], 1)

    def test_batch_rejects_excessive_size(self):
        template = make_prediction_template(self.required_columns)
        excessive = pd.concat([template] * 3, ignore_index=True)

        with self.assertRaises(ValueError):
            predict_batch(excessive, self.artifact, maximum_rows=2)

    def test_csv_export_uses_utf8_bom(self):
        template = make_prediction_template(self.required_columns)

        contents = dataframe_to_csv_bytes(template)

        self.assertTrue(contents.startswith(b"\xef\xbb\xbf"))
        self.assertIn("Peatón".encode("utf-8"), contents)


if __name__ == "__main__":
    unittest.main()
