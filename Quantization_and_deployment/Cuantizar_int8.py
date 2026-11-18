"""
Cuantización INT8 (post-training) del modelo Keras para el Arduino Nano 33 BLE Sense.

Pasos:
  1. Carga el modelo Keras y los datos del fold elegido (ya normalizados con z-score).
  2. Convierte a TFLite INT8 completo (entrada y salida int8) usando X_train como
     conjunto representativo para calibrar los rangos.
  3. Compara float vs INT8 en validación y prueba (accuracy, F1 macro, concordancia).
  4. Genera:
       modelo_fold2_int8.tflite
       modelo_har_data.h            (modelo como arreglo C + constantes de normalización,
                                     cuantización de entrada/salida y nombres de clases)
       cuantizacion_resultados.json (métricas, tamaños, operadores usados)

Uso:
  python Cuantizar_int8.py --modelo modelo_fold2.keras --npz dataset_har_preprocess_v2.npz --fold 2 --salida cuantizacion_int8
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score


def convertir_int8(model: tf.keras.Model, X_rep: np.ndarray) -> bytes:
    def dataset_representativo():
        for i in range(len(X_rep)):
            yield [X_rep[i : i + 1].astype(np.float32)]

    def configurar(conv):
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = dataset_representativo
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        return conv

    try:
        return configurar(tf.lite.TFLiteConverter.from_keras_model(model)).convert()
    except Exception as e:  # algunas versiones de Keras 3 / TF necesitan pasar por SavedModel
        print(f"from_keras_model falló ({e!r}); intentando vía SavedModel...")
        with tempfile.TemporaryDirectory() as tmp:
            model.export(tmp)
            return configurar(tf.lite.TFLiteConverter.from_saved_model(tmp)).convert()


class ModeloTFLite:
    """Envoltorio para evaluar el modelo INT8 ventana por ventana."""

    def __init__(self, contenido: bytes):
        self.interp = tf.lite.Interpreter(model_content=contenido)
        self.interp.allocate_tensors()
        self.entrada = self.interp.get_input_details()[0]
        self.salida = self.interp.get_output_details()[0]
        self.in_scale, self.in_zp = self.entrada["quantization"]
        self.out_scale, self.out_zp = self.salida["quantization"]

    def operadores(self):
        try:
            return sorted({d["op_name"] for d in self.interp._get_ops_details()})
        except Exception:
            return []

    def predecir(self, X: np.ndarray) -> np.ndarray:
        pred = []
        for i in range(len(X)):
            q = np.round(X[i : i + 1] / self.in_scale + self.in_zp)
            q = np.clip(q, -128, 127).astype(np.int8)
            self.interp.set_tensor(self.entrada["index"], q)
            self.interp.invoke()
            out = self.interp.get_tensor(self.salida["index"])[0].astype(np.float32)
            pred.append(int(np.argmax((out - self.out_zp) * self.out_scale)))
        return np.array(pred)


def escribir_header(ruta: Path, tflite: bytes, media, desv, tfm: ModeloTFLite, clases, forma):
    def lista(v):
        return ", ".join(f"{float(x):.9g}f" for x in v)

    filas = []
    for i in range(0, len(tflite), 12):
        filas.append("  " + ", ".join(f"0x{b:02x}" for b in tflite[i : i + 12]))
    cuerpo = ",\n".join(filas)
    nombres = ", ".join(f'"{c}"' for c in clases)

    with open(ruta, "w", encoding="utf-8") as f:
        f.write("// Generado por cuantizar_int8.py. No editar a mano.\n#pragma once\n\n")
        f.write(f"#define HAR_N_VENTANA  {forma[0]}\n")
        f.write(f"#define HAR_N_CANALES  {forma[1]}\n")
        f.write(f"#define HAR_N_CLASES   {len(clases)}\n\n")
        f.write("// Normalización z-score por canal (ax ay az en g; gx gy gz en °/s), calculada solo con entrenamiento\n")
        f.write(f"const float HAR_MEDIA[HAR_N_CANALES] = {{{lista(media)}}};\n")
        f.write(f"const float HAR_DESV[HAR_N_CANALES]  = {{{lista(desv)}}};\n\n")
        f.write("// Cuantización de la entrada y la salida del modelo INT8\n")
        f.write(f"const float HAR_INPUT_SCALE  = {float(tfm.in_scale):.9g}f;\n")
        f.write(f"const int   HAR_INPUT_ZERO_POINT  = {int(tfm.in_zp)};\n")
        f.write(f"const float HAR_OUTPUT_SCALE = {float(tfm.out_scale):.9g}f;\n")
        f.write(f"const int   HAR_OUTPUT_ZERO_POINT = {int(tfm.out_zp)};\n\n")
        f.write(f"const char* const HAR_CLASES[HAR_N_CLASES] = {{{nombres}}};\n\n")
        f.write("alignas(8) const unsigned char modelo_har_tflite[] = {\n")
        f.write(cuerpo + "\n};\n")
        f.write(f"const unsigned int modelo_har_tflite_len = {len(tflite)};\n")


def main(args):
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    k = args.fold

    d = np.load(args.npz)
    g = lambda n: d[f"fold_{k}_{n}"]
    X_train = g("X_train").astype(np.float32)
    X_val, y_val = g("X_val").astype(np.float32), g("y_val")
    X_test, y_test = g("X_test").astype(np.float32), g("y_test")
    media = np.asarray(g("media"), dtype=np.float64).reshape(-1)
    desv = np.asarray(g("desviacion"), dtype=np.float64).reshape(-1)
    clases = [str(c) for c in d["clases"]]
    forma = X_train.shape[1:]
    assert media.size == forma[1] and desv.size == forma[1], "constantes de normalización inesperadas"

    model = tf.keras.models.load_model(args.modelo)
    print(f"Modelo cargado: {args.modelo}  (parámetros: {model.count_params()})")

    tflite = convertir_int8(model, X_train)
    ruta_tflite = salida / f"modelo_fold{k}_int8.tflite"
    ruta_tflite.write_bytes(tflite)

    tfm = ModeloTFLite(tflite)
    ops = tfm.operadores()
    print(f"Tamaño .tflite: {len(tflite) / 1024:.1f} KB")
    print(f"Entrada: scale={tfm.in_scale:.6g}, zero_point={tfm.in_zp}, dtype={tfm.entrada['dtype'].__name__}")
    print(f"Salida:  scale={tfm.out_scale:.6g}, zero_point={tfm.out_zp}, dtype={tfm.salida['dtype'].__name__}")
    print(f"Operadores del modelo (regístralos en el Arduino): {ops}")

    resultados = {}
    for nombre, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        p_float = np.argmax(model.predict(X, verbose=0), axis=1)
        p_int8 = tfm.predecir(X)
        resultados[nombre] = {
            "accuracy_float": float(accuracy_score(y, p_float)),
            "accuracy_int8": float(accuracy_score(y, p_int8)),
            "f1_macro_float": float(f1_score(y, p_float, average="macro")),
            "f1_macro_int8": float(f1_score(y, p_int8, average="macro")),
            "concordancia_float_int8": float(np.mean(p_float == p_int8)),
        }
        r = resultados[nombre]
        print(
            f"{nombre:5s} | acc float {r['accuracy_float']:.3f} -> int8 {r['accuracy_int8']:.3f} | "
            f"F1 float {r['f1_macro_float']:.3f} -> int8 {r['f1_macro_int8']:.3f} | "
            f"concordancia {r['concordancia_float_int8']:.3f}"
        )

    ruta_header = salida / "modelo_har_data.h"
    escribir_header(ruta_header, tflite, media, desv, tfm, clases, forma)

    with open(salida / "cuantizacion_resultados.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "modelo": str(args.modelo),
                "fold": k,
                "tamano_tflite_bytes": len(tflite),
                "entrada": {"escala": float(tfm.in_scale), "zero_point": int(tfm.in_zp)},
                "salida": {"escala": float(tfm.out_scale), "zero_point": int(tfm.out_zp)},
                "operadores": ops,
                "normalizacion": {"media": media.tolist(), "desviacion": desv.tolist()},
                "evaluacion": resultados,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nGenerado: {ruta_tflite}\n          {ruta_header}\n          {salida / 'cuantizacion_resultados.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cuantización INT8 para el Arduino Nano 33 BLE Sense")
    p.add_argument("--modelo", default="resultados_loso/modelo_fold2.keras")
    p.add_argument("--npz", default="dataset_har_preprocess_v2.npz")
    p.add_argument("--fold", type=int, default=2)
    p.add_argument("--salida", default="cuantizacion_int8")
    main(p.parse_args())