"""
Preprocesamiento HAR: CSV de tomas -> dataset de ventanas (npz).

Estructura de carpetas esperada (la persona y la actividad se toman de las carpetas):
  datos/
    Persona1/
      caminar/       toma1.csv, toma2.csv, ...
      correr/
      reposo/
      subir_escaleras/
      bajar_escaleras/
    Persona2/ ...
    Persona3/ ...
(También funciona la estructura plana datos/<persona>_<actividad>_<NNN>.csv)

Para cada CSV:
  1. Valida columnas, NaN, longitud y frecuencia de muestreo.
  2. Reporta calidad (muestras saturadas, muestras repetidas, |a| media).
  3. Recorta los segundos iniciales y finales.
  4. Divide en ventanas (por defecto 2 s, 50 % de solape).
Guarda dataset_har.npz con:
  X       (N, muestras_ventana, 6)  float32, unidades físicas (g y °/s)
  y       (N,) int, índice en `clases`
  clases  nombres de las actividades
  persona (N,) str
  toma    (N,) str   <- úsalo para separar train/test POR TOMA, no por ventana
También guarda un reporte JSON (calidad por toma, ventanas por actividad/persona, parámetros).

Uso:
  python Preproces_HAR.py --entrada ruta/a/tu/carpeta --salida dataset_har.npz --reporte reporte.json
"""
import argparse
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ACTIVIDADES = ["reposo", "caminar", "correr", "subir_escaleras", "bajar_escaleras"]
CANALES = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
FS_HZ = 100
LIMITE_ACEL_G = 3.99        # el IMU satura en ±4 g
PATRON = re.compile(r"^(?P<persona>[^_]+)_(?P<actividad>.+)_(?P<idx>\d{3})$")


def normalizar(nombre: str) -> str:
    """'Bajar Escaleras' -> 'bajar_escaleras' (minúsculas, sin acentos, espacios a _)."""
    n = unicodedata.normalize("NFD", nombre)
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")
    return re.sub(r"[\s\-]+", "_", n.strip().lower())


def resolver_actividad(nombre: str):
    """Devuelve la actividad canónica a partir del nombre de carpeta (o None)."""
    n = normalizar(nombre)
    for a in ACTIVIDADES:
        if n == a or n.rstrip("s") == a.rstrip("s"):
            return a
    return None


def identificar(f: Path, raiz: Path):
    """Devuelve (persona, actividad) según carpetas o, si es plano, según el nombre del archivo."""
    partes = f.relative_to(raiz).parts
    if len(partes) >= 3:                         # Persona/actividad/archivo.csv
        return partes[0], resolver_actividad(partes[1]), partes[1]
    m = PATRON.match(f.stem)                     # persona_actividad_NNN.csv
    if m:
        return m["persona"], resolver_actividad(m["actividad"]), m["actividad"]
    return None, None, None


def cargar_tomas(carpeta: Path):
    tomas = []
    avisos = []
    desconocidas = set()
    for f in sorted(carpeta.rglob("*.csv")):
        persona, actividad, crudo = identificar(f, carpeta)
        if persona is None or actividad is None:
            desconocidas.add(crudo or str(f.relative_to(carpeta)))
            continue
        df = pd.read_csv(f)
        if list(df.columns[1:]) != CANALES:
            msg = f"[omitido] {f.name}: columnas inesperadas {list(df.columns)}"
            print(msg)
            avisos.append(msg)
            continue
        if df.isna().any().any():
            msg = f"[omitido] {f.name}: contiene NaN"
            print(msg)
            avisos.append(msg)
            continue
        paso = np.diff(df["t_s"].to_numpy())
        if not np.allclose(paso, 1 / FS_HZ, atol=1e-3):
            msg = f"[aviso] {f.name}: tiempo no uniforme"
            print(msg)
            avisos.append(msg)
        tomas.append(
            {
                "nombre": f"{persona}/{actividad}/{f.stem}",
                "persona": persona,
                "actividad": actividad,
                "datos": df[CANALES].to_numpy(dtype=np.float32),
            }
        )
    if desconocidas:
        print("[omitidos] carpetas/archivos sin actividad reconocida:", sorted(desconocidas))
        print("           actividades válidas:", ACTIVIDADES)
    return tomas, avisos, sorted(desconocidas)


def calidad(x: np.ndarray):
    acel = x[:, :3]
    saturadas = int((np.abs(acel) >= LIMITE_ACEL_G).any(axis=1).sum())
    repetidas = int((np.abs(np.diff(x, axis=0)).sum(axis=1) == 0).sum())
    return saturadas, repetidas, float(np.linalg.norm(acel, axis=1).mean())


def ventanear(x: np.ndarray, ventana: int, paso: int) -> np.ndarray:
    n = (len(x) - ventana) // paso + 1
    if n <= 0:
        return np.empty((0, ventana, x.shape[1]), dtype=x.dtype)
    return np.stack([x[i * paso : i * paso + ventana] for i in range(n)])


def main(args):
    tomas, avisos, desconocidas = cargar_tomas(Path(args.entrada))
    if not tomas:
        raise SystemExit("No se encontraron tomas válidas.")

    recorte = int(args.recorte * FS_HZ)
    ventana = int(args.ventana * FS_HZ)
    paso = max(1, int(ventana * (1 - args.solape)))

    X, y, personas, nombres = [], [], [], []
    por_toma = []
    print(f"{'toma':44s} {'satur.':>6s} {'repet.':>6s} {'|a| g':>6s} {'vent.':>5s}")
    for t in tomas:
        x = t["datos"]
        sat, rep, norma = calidad(x)
        n_muestras = len(x)
        x = x[recorte : len(x) - recorte]
        w = ventanear(x, ventana, paso)
        print(f"{t['nombre']:44s} {sat:6d} {rep:6d} {norma:6.2f} {len(w):5d}")
        por_toma.append(
            {
                "toma": t["nombre"],
                "persona": t["persona"],
                "actividad": t["actividad"],
                "muestras": n_muestras,
                "muestras_saturadas": sat,
                "muestras_repetidas": rep,
                "norma_acel_media_g": round(norma, 3),
                "ventanas": int(len(w)),
            }
        )
        X.append(w)
        y += [ACTIVIDADES.index(t["actividad"])] * len(w)
        personas += [t["persona"]] * len(w)
        nombres += [t["nombre"]] * len(w)

    X = np.concatenate(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    np.savez_compressed(
        args.salida,
        X=X,
        y=y,
        clases=np.array(ACTIVIDADES),
        persona=np.array(personas),
        toma=np.array(nombres),
    )

    print(f"\nX: {X.shape}  (ventana={ventana} muestras, paso={paso}, recorte={recorte} muestras/lado)")
    ventanas_por_actividad = {}
    for i, c in enumerate(ACTIVIDADES):
        n = int((y == i).sum())
        ventanas_por_actividad[c] = n
        print(f"  {c:16s} {n:5d} ventanas")
    print(f"Guardado en {args.salida}")

    # ---------- Reporte JSON (lo mismo que se imprime en consola) ----------
    ventanas_por_persona = {
        p: int(np.sum(np.array(personas) == p)) for p in sorted(set(personas))
    }
    reporte = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "entrada": str(args.entrada),
        "salida_npz": str(args.salida),
        "parametros": {
            "fs_hz": FS_HZ,
            "recorte_s": args.recorte,
            "recorte_muestras_por_lado": recorte,
            "ventana_s": args.ventana,
            "ventana_muestras": ventana,
            "solape": args.solape,
            "paso_muestras": paso,
            "canales": CANALES,
        },
        "dataset": {
            "forma_X": list(X.shape),
            "total_ventanas": int(len(y)),
            "total_tomas": len(tomas),
            "ventanas_por_actividad": ventanas_por_actividad,
            "ventanas_por_persona": ventanas_por_persona,
        },
        "tomas": por_toma,
        "avisos": avisos,
        "carpetas_o_archivos_omitidos": desconocidas,
    }
    ruta_json = Path(args.reporte) if args.reporte else Path(args.salida).with_suffix(".json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)
    print(f"Reporte guardado en {ruta_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocesamiento HAR")
    p.add_argument("--entrada", default="datos", help="Carpeta con los CSV")
    p.add_argument("--salida", default="dataset_har.npz")
    p.add_argument("--reporte", default=None, help="Ruta del JSON (por defecto, junto al .npz)")
    p.add_argument("--recorte", type=float, default=2.0, help="Segundos a recortar por lado")
    p.add_argument("--ventana", type=float, default=2.0, help="Tamaño de ventana (s)")
    p.add_argument("--solape", type=float, default=0.5, help="Fracción de solape (0 a <1)")
    main(p.parse_args())