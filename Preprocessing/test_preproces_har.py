"""Comprobaciones de integridad, trazabilidad y JSON opcionales.

Desde TinyML3/: python -m unittest discover -s Preprocessing -p test_preproces_har.py
"""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("preproces_har", Path(__file__).with_name("Preproces_HAR.py"))
har = importlib.util.module_from_spec(spec)
spec.loader.exec_module(har)


class PreprocesHARTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.datos = self.root / "datos"
        self.datos.mkdir()

    def csv(self, nombre="P1_caminar_001.csv", n=600, x=None, subcarpeta=""):
        t = np.arange(n, dtype=np.float64) / 100
        if x is None:
            x = np.column_stack([10 * c + t for c in range(6)])
        tabla = pd.DataFrame(np.column_stack([t, x]), columns=["t_s", *har.CANALES])
        ruta = self.datos / subcarpeta / nombre
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(ruta, index=False)
        return ruta, tabla

    def args(self, *extras):
        return har.crear_parser().parse_args([
            "--entrada", str(self.datos), "--salida", str(self.root / "salida.npz"),
            "--reporte", str(self.root / "reporte.json"), "--sin-zscore", *extras,
        ])

    def ejecutar(self, *extras):
        with contextlib.redirect_stdout(io.StringIO()):
            return har.main(self.args(*extras))

    def test_csv_sin_json_conserva_valores_y_trazabilidad(self):
        ruta, tabla = self.csv(n=700)
        informe = self.ejecutar()
        with np.load(self.root / "salida.npz", allow_pickle=False) as ds:
            np.testing.assert_array_equal(ds["X"][0], tabla[har.CANALES].to_numpy(dtype=np.float32)[200:400])
            np.testing.assert_array_equal(ds["X"][1], tabla[har.CANALES].to_numpy(dtype=np.float32)[300:500])
            np.testing.assert_array_equal(ds["inicio_muestra"], [200, 300])
            np.testing.assert_array_equal(ds["fin_muestra_exclusivo"], [400, 500])
            np.testing.assert_array_equal(ds["t_inicio_s"], [2.0, 3.0])
            self.assertEqual(ds["normalizacion"].item(), "ninguna")
            self.assertEqual(ds["archivo"].tolist(), [ruta.name] * 2)
            for campo in ds.files:
                self.assertNotEqual(ds[campo].dtype.kind, "O")
        self.assertEqual(informe["dataset"]["metadata_adquisicion"]["ausente"], 1)
        self.assertIsNone(informe["tomas"][0]["metadata_adquisicion"]["datos"])

    def test_json_invalido_es_opcional(self):
        ruta, _ = self.csv()
        for contenido in ("{", "[]", '{"huecos": NaN}', '{"huecos": 1e999}'):
            with self.subTest(contenido=contenido):
                ruta.with_suffix(".json").write_text(contenido, encoding="utf-8")
                informe = self.ejecutar("--sobrescribir")
                self.assertEqual(informe["dataset"]["total_tomas"], 1)
                self.assertEqual(informe["dataset"]["metadata_adquisicion"]["invalido"], 1)
                self.assertTrue(informe["avisos"])

    def test_csv_corruptos_y_tiempos_invalidos_se_rechazan(self):
        ruta, original = self.csv()
        casos = {}
        for nombre, columna, valor in (
            ("nan", "ax_g", np.nan), ("inf", "gx_dps", np.inf),
            ("fuera_float32", "ax_g", 1e100), ("tiempo_duplicado", "t_s", 0.0),
            ("retroceso", "t_s", -1.0),
        ):
            tabla = original.copy()
            tabla.loc[1, columna] = valor
            casos[nombre] = tabla
        casos["columna_faltante"] = original.drop(columns="ax_g")
        casos["columna_extra"] = original.assign(extra=1)
        casos["otra_frecuencia"] = original.assign(t_s=original["t_s"] * 2)
        casos["no_numerico"] = original.astype({"ax_g": "object"})
        casos["no_numerico"].loc[10, "ax_g"] = "error"
        casos["sin_filas"] = original.iloc[:0]
        for nombre, tabla in casos.items():
            with self.subTest(nombre=nombre):
                tabla.to_csv(ruta, index=False)
                with self.assertRaises((ValueError, pd.errors.ParserError)):
                    har.leer_csv(ruta, 100)
        # Pandas puede interpretar silenciosamente el campo adicional como índice.
        ruta.write_text(
            "t_s," + ",".join(har.CANALES) + "\n"
            + "0,0,1,2,3,4,5,6\n1,0.01,1,2,3,4,5,6\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "más campos"):
            har.leer_csv(ruta, 100)

    def test_longitud_insuficiente_se_documenta_sin_ventanas_vacias(self):
        self.csv()
        self.csv("P1_caminar_002.csv", n=599)
        informe = self.ejecutar()
        self.assertEqual(informe["dataset"]["total_tomas"], 1)
        self.assertEqual(informe["dataset"]["total_ventanas"], 1)
        self.assertIn("Longitud insuficiente", informe["archivos_omitidos"][0]["motivo"])

    def test_indicadores_antes_y_despues_no_descartan_la_toma(self):
        x = np.zeros((600, 6))
        x[:, 0] = 1
        x[:, 3] = np.arange(600)
        x[0, 0], x[-1, 0] = 4, -4
        self.csv(x=x)
        toma = self.ejecutar()["tomas"][0]
        self.assertEqual(toma["calidad_antes_recorte"]["muestras_cerca_limite_acel"], 2)
        despues = toma["calidad_despues_recorte"]
        self.assertEqual(despues["muestras_cerca_limite_acel"], 0)
        self.assertEqual(despues["repeticiones"]["acelerometro"]["racha_maxima_muestras_iguales"], 200)
        self.assertEqual(despues["repeticiones"]["giroscopio"]["pares_consecutivos_iguales"], 0)
        self.assertEqual(toma["ventanas"], 1)

    def test_parametros_invalidos(self):
        casos = [
            ["--recorte", "-1"], ["--ventana", "0"], ["--solape", "1"],
            ["--solape", "-0.1"], ["--solape", "0.333"], ["--fs-hz", "0"],
            ["--fs-hz", "nan"], ["--ventana", "inf"], ["--recorte", "0.005"],
            ["--epsilon-zscore", "0"], ["--epsilon-zscore", "nan"],
            ["--semilla", "-1"], ["--tomas-validacion", "0"],
        ]
        for opciones in casos:
            with self.subTest(opciones=opciones), self.assertRaises(ValueError):
                har.configurar(self.args(*opciones))

    def test_sin_recorte_y_sin_solape(self):
        _, tabla = self.csv()
        self.ejecutar("--recorte", "0", "--solape", "0")
        with np.load(self.root / "salida.npz") as ds:
            self.assertEqual(ds["X"].shape, (3, 200, 6))
            np.testing.assert_array_equal(ds["X"].reshape(600, 6), tabla[har.CANALES].to_numpy(dtype=np.float32))

    def test_marcas_de_revision_tienen_opcion_explicita(self):
        self.csv()
        self.csv("P1_caminar_002.csv", subcarpeta="revisar")
        marcada, _ = self.csv("P1_caminar_003.csv")
        marcada.with_suffix(".json").write_text('{"calidad_revisar": true}', encoding="utf-8")
        informe = self.ejecutar()
        self.assertEqual(informe["dataset"]["total_tomas"], 1)
        self.assertEqual(informe["dataset"]["csv_omitidos"], 2)
        informe = self.ejecutar("--incluir-revisar", "--sobrescribir")
        self.assertEqual(informe["dataset"]["total_tomas"], 3)

    def test_ventanas_no_cruzan_grabaciones(self):
        self.csv(n=700, x=np.ones((700, 6)), subcarpeta="Persona1/Caminar")
        self.csv("P2_correr_001.csv", n=700, x=np.full((700, 6), 9), subcarpeta="Persona2/Correr")
        self.ejecutar()
        with np.load(self.root / "salida.npz") as ds:
            np.testing.assert_array_equal(ds["y"], [1, 1, 2, 2])
            np.testing.assert_array_equal(ds["persona"], ["Persona1"] * 2 + ["Persona2"] * 2)
            self.assertTrue((ds["X"][:2] == 1).all())
            self.assertTrue((ds["X"][2:] == 9).all())

    def test_no_sobrescribe_sin_opcion(self):
        self.csv()
        self.ejecutar()
        salida = self.root / "salida.npz"
        anterior = salida.read_bytes()
        with self.assertRaisesRegex(ValueError, "ya existe"):
            self.ejecutar()
        self.assertEqual(salida.read_bytes(), anterior)

    def test_todas_las_tomas_invalidas_no_generan_npz(self):
        self.csv(n=200)
        with self.assertRaisesRegex(ValueError, "No hay tomas válidas"):
            self.ejecutar()
        self.assertFalse((self.root / "salida.npz").exists())

    def test_json_discrepante_genera_aviso_sin_reetiquetar(self):
        ruta, _ = self.csv()
        ruta.with_suffix(".json").write_text(json.dumps({
            "persona": "P2", "actividad": "correr", "numero_muestras": 3000, "fs_hz": 50,
        }), encoding="utf-8")
        informe = self.ejecutar()
        self.assertEqual(informe["tomas"][0]["actividad"], "caminar")
        self.assertEqual(len(informe["avisos"]), 4)


class ZScoreLOSOTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        personas, tomas, etiquetas = [], [], []
        ventanas = []
        for persona in range(1, 4):
            for clase in range(2):
                for toma in range(3):
                    for _ in range(2):
                        x = rng.normal(persona * 10 + clase * 2 + toma, 2, (8, 6))
                        x[:, 5] = 3  # Canal constante para comprobar el divisor.
                        ventanas.append(x)
                        personas.append(f"Persona{persona}")
                        tomas.append(f"Persona{persona}/clase{clase}/toma{toma}")
                        etiquetas.append(clase)
        self.X = np.asarray(ventanas, dtype=np.float32)
        self.y = np.asarray(etiquetas, dtype=np.int64)
        self.personas = np.asarray(personas)
        self.tomas = np.asarray(tomas)
        self.config = har.configurar(har.crear_parser().parse_args([]))

    def preparar(self, X=None):
        return har.preparar_loso_zscore(
            self.X if X is None else X, self.y, self.personas, self.tomas, self.config
        )

    def test_estadisticas_se_ajustan_solo_con_train(self):
        campos, _ = self.preparar()
        idx = campos["fold_1_indices_train"]
        esperado = self.X[idx].astype(np.float64)
        np.testing.assert_array_equal(campos["fold_1_media"], esperado.mean(axis=(0, 1)))
        np.testing.assert_array_equal(campos["fold_1_desviacion"], esperado.std(axis=(0, 1)))
        train = campos["fold_1_X_train"]
        np.testing.assert_allclose(train.mean(axis=(0, 1)), 0, atol=2e-7)
        np.testing.assert_allclose(train[:, :, :5].std(axis=(0, 1)), 1, atol=2e-7)

    def test_cambiar_val_test_no_cambia_el_ajuste(self):
        original, _ = self.preparar()
        cambiado = self.X.copy()
        for conjunto in ("val", "test"):
            cambiado[original[f"fold_1_indices_{conjunto}"]] += 1000
        nuevo, _ = self.preparar(cambiado)
        for nombre in ("media", "desviacion", "escala", "X_train"):
            np.testing.assert_array_equal(original[f"fold_1_{nombre}"], nuevo[f"fold_1_{nombre}"])
        for conjunto in ("val", "test"):
            idx = nuevo[f"fold_1_indices_{conjunto}"]
            esperado = ((cambiado[idx].astype(np.float64) - original["fold_1_media"]) / original["fold_1_escala"]).astype(np.float32)
            np.testing.assert_array_equal(nuevo[f"fold_1_X_{conjunto}"], esperado)

    def test_grupos_disjuntos_y_cobertura_de_todas_las_ventanas(self):
        campos, _ = self.preparar()
        for fold in campos["folds_loso"]:
            indices = [campos[f"{fold}_indices_{c}"] for c in ("train", "val", "test")]
            np.testing.assert_array_equal(np.sort(np.concatenate(indices)), np.arange(len(self.X)))
            grupos = [set(self.tomas[idx].tolist()) for idx in indices]
            self.assertFalse(grupos[0] & grupos[1] or grupos[0] & grupos[2] or grupos[1] & grupos[2])
            test = campos[f"{fold}_persona_test"].item()
            self.assertTrue((self.personas[indices[2]] == test).all())
            self.assertFalse((self.personas[indices[0]] == test).any())
            self.assertFalse((self.personas[indices[1]] == test).any())

    def test_canal_constante_no_produce_nan_y_guarda_escala(self):
        campos, _ = self.preparar()
        self.assertEqual(campos["fold_1_desviacion"][5], 0)
        self.assertEqual(campos["fold_1_escala"][5], self.config["epsilon_zscore"])
        for conjunto in ("train", "val", "test"):
            x = campos[f"fold_1_X_{conjunto}"]
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue((x[:, :, 5] == 0).all())
            self.assertEqual(x.dtype, np.float32)

    def test_particiones_reproducibles_y_escaladores_por_fold(self):
        a, _ = self.preparar()
        b, _ = self.preparar()
        for nombre in a:
            np.testing.assert_array_equal(a[nombre], b[nombre])
        self.assertFalse(np.array_equal(a["fold_1_media"], a["fold_3_media"]))

    def test_datos_insuficientes_no_se_dividen_por_ventana(self):
        with self.assertRaisesRegex(ValueError, "tres personas"):
            har.preparar_loso_zscore(self.X[:12], self.y[:12], self.personas[:12], self.tomas[:12], self.config)
        self.config["tomas_validacion_por_persona_actividad"] = 3
        with self.assertRaisesRegex(ValueError, "al menos 4 tomas"):
            self.preparar()

    def test_main_guarda_zscore_por_defecto_en_el_npz(self):
        with tempfile.TemporaryDirectory() as carpeta:
            root = Path(carpeta)
            for persona in range(1, 4):
                destino = root / "datos" / f"Persona{persona}" / "Caminar"
                destino.mkdir(parents=True)
                for toma in range(2):
                    t = np.arange(600) / 100
                    x = np.column_stack([t + persona * 10 + toma + c for c in range(6)])
                    pd.DataFrame(np.column_stack([t, x]), columns=["t_s", *har.CANALES]).to_csv(
                        destino / f"P{persona}_caminar_{toma:03d}.csv", index=False
                    )
            args = har.crear_parser().parse_args([
                "--entrada", str(root / "datos"), "--salida", str(root / "normalizado.npz"),
            ])
            with contextlib.redirect_stdout(io.StringIO()):
                reporte = har.main(args)
            with np.load(root / "normalizado.npz", allow_pickle=False) as ds:
                self.assertEqual(ds["folds_loso"].tolist(), ["fold_1", "fold_2", "fold_3"])
                self.assertEqual(ds["normalizacion"].item(), "ninguna")  # Se refiere a X base.
                self.assertEqual(ds["normalizacion_loso"].item(), "zscore_por_canal_ajustado_solo_train")
                for fold in ds["folds_loso"]:
                    train = ds[f"{fold}_X_train"]
                    np.testing.assert_allclose(train.mean(axis=(0, 1)), 0, atol=2e-7)
                    for campo in ("X_train", "X_val", "X_test", "media", "desviacion", "escala"):
                        self.assertTrue(np.isfinite(ds[f"{fold}_{campo}"]).all())
            self.assertIsNotNone(reporte["normalizacion_loso"])


if __name__ == "__main__":
    unittest.main()
