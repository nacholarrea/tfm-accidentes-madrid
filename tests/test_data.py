"""Pruebas minimas de las transformaciones que definen el objetivo."""

import sys
import unittest
from pathlib import Path

import pandas as pd
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tfm_accidentes.data import (  # noqa: E402
    build_analysis_table,
    normalize_injury_code,
    read_analysis_table,
)


class InjuryCodeTests(unittest.TestCase):
    def test_normalize_injury_code(self) -> None:
        values = pd.Series(["3", "04", "14.0", "", None, "77"], dtype="string")
        result = normalize_injury_code(values)
        self.assertEqual(result.tolist(), ["03", "04", "14", "14", "14", "77"])

    def test_binary_target_keeps_unknown_as_missing(self) -> None:
        frame = pd.DataFrame(
            {
                "cod_lesividad": pd.Series(["3", "4", "1", "14", "77"], dtype="string"),
                "fecha": pd.Series(["01/01/2025"] * 5, dtype="string"),
                "hora": pd.Series(["02:30:00"] * 5, dtype="string"),
                "coordenada_x_utm": pd.Series(["440000"] * 5, dtype="string"),
                "coordenada_y_utm": pd.Series(["4470000"] * 5, dtype="string"),
                "num_expediente": pd.Series([f"2025S{i:06d}" for i in range(5)], dtype="string"),
            }
        )
        result = build_analysis_table(frame)
        self.assertEqual(result["lesion_grave"].tolist()[:4], [1, 1, 0, 0])
        self.assertTrue(pd.isna(result.loc[4, "lesion_grave"]))
        self.assertEqual(result.loc[0, "es_noche"], 1)

    def test_unexpected_injury_code_is_not_silently_labelled(self) -> None:
        frame = pd.DataFrame(
            {
                "cod_lesividad": pd.Series(["99"], dtype="string"),
                "fecha": pd.Series(["01/01/2025"], dtype="string"),
                "hora": pd.Series(["12:00:00"], dtype="string"),
                "coordenada_x_utm": pd.Series(["440000"], dtype="string"),
                "coordenada_y_utm": pd.Series(["4470000"], dtype="string"),
                "num_expediente": pd.Series(["2025S000001"], dtype="string"),
            }
        )
        with self.assertRaisesRegex(ValueError, "no documentados"):
            build_analysis_table(frame)

    def test_read_analysis_table_recovers_key_types(self) -> None:
        frame = pd.DataFrame(
            {
                "fecha": ["2025-01-01"],
                "fecha_hora": ["2025-01-01 12:00:00"],
                "num_expediente": ["2025S000001"],
                "cod_distrito": ["01"],
                "cod_lesividad": ["03"],
                "anio_fichero": [2025],
                "anio": [2025],
                "mes": [1],
                "dia_semana": [2],
                "es_fin_semana": [0],
                "hora_dia": [12],
                "es_noche": [0],
                "lesion_grave": [1],
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.csv.gz"
            frame.to_csv(path, index=False, compression="gzip")
            result = read_analysis_table(path)

        self.assertEqual(result.loc[0, "cod_lesividad"], "03")
        self.assertEqual(str(result["lesion_grave"].dtype), "Int64")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["fecha_hora"]))


if __name__ == "__main__":
    unittest.main()
