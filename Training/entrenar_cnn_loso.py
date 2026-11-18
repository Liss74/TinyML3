"""
Entrenamiento LOSO (leave-one-subject-out) de la CNN 1D para HAR.

Usa los folds ya preparados en dataset_har_preprocess_v2.npz:
  fold_k_X_train / X_val / X_test  (ya normalizados con z-score de train, NO volver a normalizar)
  fold_k_y_*                       (etiquetas enteras 0..4)

Para cada fold:
  - Entrena con early stopping (paciencia 12) monitorizando el F1 macro de validación
    y restaura los mejores pesos.
  - Evalúa UNA sola vez en la persona de prueba (test).
Al final reporta media ± desviación entre folds y guarda todo en JSON.

Uso:
  python entrenar_cnn_loso.py --npz dataset_har_preprocess_v2.npz --salida resultados_loso
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

SEED = 42
EPOCAS = 100
BATCH = 32
LR = 0.001
PACIENCIA = 12


def construir_modelo(n_ventana: int, n_canales: int, n_clases: int) -> tf.keras.Model:
    # Conv1D (pasos de tiempo, canales) = (200, 6).
    input_layer = tf.keras.Input(shape=(n_ventana, n_canales))
    x = tf.keras.layers.Conv1D(16, 5, padding="valid", activation="relu")(input_layer)
    x = tf.keras.layers.AveragePooling1D(2, padding="valid")(x)
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.AveragePooling1D(2, padding="valid")(x)
    x = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    output_layer = tf.keras.layers.Dense(n_clases, activation="softmax")(x)
    return tf.keras.Model(input_layer, output_layer)


class F1MacroVal(tf.keras.callbacks.Callback):
    """Calcula el F1 macro de validación al final de cada época y lo añade a los logs."""

    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val

    def on_epoch_end(self, epoch, logs=None):
        pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        logs["val_f1_macro"] = f1_score(self.y_val, pred, average="macro")


def guardar_curvas(history, ruta: Path, titulo: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(history["loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title(f"{titulo}: pérdida")
    ax[0].set_xlabel("época")
    ax[0].legend()
    ax[1].plot(history["val_f1_macro"], label="F1 macro (val)")
    ax[1].plot(history["val_accuracy"], label="accuracy (val)")
    ax[1].set_title(f"{titulo}: validación")
    ax[1].set_xlabel("época")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def entrenar_fold(d, k: int, clases, salida: Path):
    g = lambda nombre: d[f"fold_{k}_{nombre}"]
    X_train, y_train = g("X_train"), g("y_train")
    X_val, y_val = g("X_val"), g("y_val")
    X_test, y_test = g("X_test"), g("y_test")
    persona_test = str(g("persona_test"))

    print(f"\n{'=' * 60}\nFold {k}: prueba = {persona_test}")
    print(f"  train {X_train.shape}  val {X_val.shape}  test {X_test.shape}")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)       # misma semilla en cada fold

    model = construir_modelo(X_train.shape[1], X_train.shape[2], len(clases))
    if k == 1:
        model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        F1MacroVal(X_val, y_val),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_macro", mode="max", patience=PACIENCIA, restore_best_weights=True
        ),
    ]
    h = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCAS,
        batch_size=BATCH,
        callbacks=callbacks,
        verbose=2,
    )
    history = {kk: [float(v) for v in vv] for kk, vv in h.history.items()}
    mejor_epoca = int(np.argmax(history["val_f1_macro"]) + 1)

    # Evaluación en la persona de prueba (pesos restaurados a la mejor época)
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    etiquetas = list(range(len(clases)))
    cm = confusion_matrix(y_test, pred, labels=etiquetas)
    f1 = float(f1_score(y_test, pred, average="macro"))
    acc = float(accuracy_score(y_test, pred))

    print(f"\nFold {k} ({persona_test}) -> accuracy {acc:.3f} | F1 macro {f1:.3f} | mejor época {mejor_epoca}")
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(cm)
    print(classification_report(y_test, pred, labels=etiquetas, target_names=clases, zero_division=0))

    model.save(salida / f"modelo_fold{k}.keras")
    guardar_curvas(history, salida / f"curvas_fold{k}.png", f"Fold {k} ({persona_test})")

    return {
        "fold": k,
        "persona_test": persona_test,
        "ventanas": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "mejor_epoca": mejor_epoca,
        "epocas_entrenadas": len(history["loss"]),
        "val_f1_macro_mejor": float(max(history["val_f1_macro"])),
        "test_accuracy": acc,
        "test_f1_macro": f1,
        "test_f1_por_clase": {
            c: float(v) for c, v in zip(clases, f1_score(y_test, pred, labels=etiquetas, average=None, zero_division=0))
        },
        "matriz_confusion": cm.tolist(),
        "historial": history,
    }


def main(args):
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    d = np.load(args.npz)
    clases = [str(c) for c in d["clases"]]

    resultados = [entrenar_fold(d, k, clases, salida) for k in args.folds]

    f1s = np.array([r["test_f1_macro"] for r in resultados])
    accs = np.array([r["test_accuracy"] for r in resultados])
    cm_total = np.sum([np.array(r["matriz_confusion"]) for r in resultados], axis=0)

    print(f"\n{'=' * 60}\nResumen LOSO")
    for r in resultados:
        print(f"  {r['persona_test']}: accuracy {r['test_accuracy']:.3f}  F1 macro {r['test_f1_macro']:.3f}")
    print(f"  Media ± desv.: accuracy {accs.mean():.3f} ± {accs.std():.3f} | F1 macro {f1s.mean():.3f} ± {f1s.std():.3f}")
    print("Matriz de confusión acumulada de los folds:")
    print(cm_total)

    resumen = {
        "protocolo": "LOSO, 3 folds; validación por tomas completas dentro de las personas de entrenamiento",
        "hiperparametros": {
            "optimizador": "Adam", "learning_rate": LR, "batch_size": BATCH, "epocas_max": EPOCAS,
            "early_stopping_paciencia": PACIENCIA, "metrica_seleccion": "F1 macro (validación)", "semilla": SEED,
        },
        "clases": clases,
        "folds": resultados,
        "media": {"accuracy": float(accs.mean()), "f1_macro": float(f1s.mean())},
        "desviacion": {"accuracy": float(accs.std()), "f1_macro": float(f1s.std())},
        "matriz_confusion_acumulada": cm_total.tolist(),
    }
    with open(salida / "resultados_loso.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en {salida / 'resultados_loso.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Entrenamiento LOSO de la CNN 1D (HAR)")
    p.add_argument("--npz", default="dataset_har_preprocess_v2.npz")
    p.add_argument("--salida", default="resultados_loso")
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    main(p.parse_args())