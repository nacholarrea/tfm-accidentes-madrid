import unittest

import pandas as pd

from tfm_accidentes.features import add_advanced_features, add_cyclical_features


class CyclicalFeaturesTests(unittest.TestCase):
    def test_expected_reference_values_and_input_is_unchanged(self):
        data = pd.DataFrame(
            {"hora_dia": [0, 6], "dia_semana": [0, 0], "mes": [1, 1]}
        )

        result = add_cyclical_features(data)

        self.assertAlmostEqual(result.loc[0, "hora_sin"], 0.0)
        self.assertAlmostEqual(result.loc[0, "hora_cos"], 1.0)
        self.assertAlmostEqual(result.loc[1, "hora_sin"], 1.0)
        self.assertAlmostEqual(result.loc[1, "hora_cos"], 0.0, places=12)
        self.assertAlmostEqual(result.loc[0, "dia_sin"], 0.0)
        self.assertAlmostEqual(result.loc[0, "dia_cos"], 1.0)
        self.assertAlmostEqual(result.loc[0, "mes_sin"], 0.0)
        self.assertAlmostEqual(result.loc[0, "mes_cos"], 1.0)
        self.assertNotIn("hora_sin", data.columns)

    def test_missing_source_column_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "mes"):
            add_cyclical_features(
                pd.DataFrame({"hora_dia": [12], "dia_semana": [3]})
            )

    def test_advanced_features_are_deterministic_and_interpretable(self):
        data = pd.DataFrame(
            {
                "hora_dia": [8, 23],
                "dia_semana": [1, 6],
                "mes": [2, 8],
                "rango_edad": ["De 25 a 29 años", "Desconocido"],
                "tipo_vehiculo": ["Motocicleta > 125cc", "Turismo"],
                "tipo_persona": ["Conductor", "Pasajero"],
                "tipo_accidente": ["Caída", "Alcance"],
                "estado_meteorologico": ["Lluvia débil", "Despejado"],
            }
        )

        result = add_advanced_features(data)

        self.assertEqual(result.loc[0, "edad_aproximada"], 27.0)
        self.assertEqual(result.loc[1, "edad_desconocida"], 1)
        self.assertEqual(result.loc[0, "grupo_vehiculo"], "Motocicleta/ciclomotor")
        self.assertEqual(result.loc[0, "usuario_vulnerable"], 1)
        self.assertEqual(result.loc[0, "meteorologia_adversa"], 1)
        self.assertEqual(result.loc[1, "meteorologia_adversa"], 0)
        self.assertEqual(result.loc[0, "contexto_persona_accidente"], "Conductor | Caída")


if __name__ == "__main__":
    unittest.main()
