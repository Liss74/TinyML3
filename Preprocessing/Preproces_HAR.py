"""Convierte grabaciones de la IMU en ventanas para entrenar un clasificador HAR.

Entrada: CSV organizados por persona y actividad, con tiempo (t_s), aceleración
(ax_g, ay_g, az_g) y velocidad angular (gx_dps, gy_dps, gz_dps). Los JSON de
adquisición son opcionales; su ausencia no invalida una grabación.

Valida la integridad de cada CSV, recorta ambos extremos y genera ventanas sin
mezclar tomas. Por defecto: 100 Hz, recorte de 2 s por extremo, ventanas de
200 muestras (2 s) y desplazamiento de 100 muestras (50 % de solapamiento).
Calcula calidad antes y después del recorte: repeticiones por sensor, cercanía
al límite del acelerómetro y estadísticas de cada canal.

Genera un NPZ con X (N, muestras_ventana, 6), etiquetas, clases, persona, toma,
archivo de origen, índices de las ventanas y configuración. Genera también un
reporte JSON con calidad, metadatos disponibles, conteos y motivos de omisión.
X conserva las unidades físicas. En este mismo script se crean particiones LOSO
y conjuntos train/val/test con Z-score por canal, ajustado solo sobre train.
Guarda los índices y las estadísticas de cada partición para reproducirlas.

Uso: python Preproces_HAR.py --entrada ruta/datos --salida dataset.npz --reporte reporte.json
Opciones: --help. Las salidas existentes requieren --sobrescribir.
"""

import argparse
import hashlib
import io
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2.1"
ACTIVIDADES = ["reposo", "caminar", "correr", "subir_escaleras", "bajar_escaleras"]
CANALES = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
UNIDADES = ["g", "g", "g", "grados/s", "grados/s", "grados/s"]
FS_HZ = 100
LIMITE_ACEL_G = 3.99  # Indicador cercano a +/-4 g, no prueba de saturación.
TOLERANCIA_TIEMPO_RELATIVA = 0.05
PATRON = re.compile(r"^(?P<persona>[^_]+)_(?P<actividad>.+)_(?P<idx>\d{3})$")


def normalizar(nombre: str) -> str:
    """Normaliza mayúsculas, acentos, espacios y guiones en nombres."""
    n = unicodedata.normalize("NFD", nombre)
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")
    return re.sub(r"[\s\-]+", "_", n.strip().lower())


def resolver_actividad(nombre: str):
    """Devuelve una actividad canónica o None si no se reconoce."""
    n = normalizar(nombre)
    for actividad in ACTIVIDADES:
        if n == actividad or n.rstrip("s") == actividad.rstrip("s"):
            return actividad
    return None


def identificar(f: Path, raiz: Path):
    """Obtiene persona, actividad canónica y nombre original de la actividad."""
    partes = f.relative_to(raiz).parts
    if len(partes) >= 3:
        return partes[0], resolver_actividad(partes[1]), partes[1]
    m = PATRON.fullmatch(f.stem)
    if m:
        return m["persona"], resolver_actividad(m["actividad"]), m["actividad"]
    return None, None, None


def _numero_muestras(segundos, fs_hz, nombre, minimo):
    valor = segundos * fs_hz
    if not math.isfinite(valor) or valor < minimo or not math.isclose(valor, round(valor), abs_tol=1e-9, rel_tol=0):
        raise ValueError(f"{nombre} debe corresponder a un número entero de muestras >= {minimo}.")
    return int(round(valor))


def configurar(args):
    if not math.isfinite(args.fs_hz) or args.fs_hz <= 0:
        raise ValueError("--fs-hz debe ser finito y mayor que cero.")
    if not math.isfinite(args.solape) or not 0 <= args.solape < 1:
        raise ValueError("--solape debe estar entre 0 (incluido) y 1 (excluido).")
    if not math.isfinite(args.epsilon_zscore) or args.epsilon_zscore <= 0:
        raise ValueError("--epsilon-zscore debe ser finito y mayor que cero.")
    if args.semilla < 0 or args.tomas_validacion < 1:
        raise ValueError("--semilla debe ser >= 0 y --tomas-validacion debe ser >= 1.")
    recorte = _numero_muestras(args.recorte, args.fs_hz, "--recorte", 0)
    ventana = _numero_muestras(args.ventana, args.fs_hz, "--ventana", 1)
    paso = _numero_muestras(ventana * (1 - args.solape), 1, "El desplazamiento", 1)
    return {
        "fs_hz": args.fs_hz,
        "recorte_s": args.recorte,
        "recorte_muestras_por_lado": recorte,
        "ventana_s": args.ventana,
        "ventana_muestras": ventana,
        "solape": args.solape,
        "paso_muestras": paso,
        "tolerancia_tiempo_relativa": TOLERANCIA_TIEMPO_RELATIVA,
        "umbral_cercania_limite_acel_g": LIMITE_ACEL_G,
        "incluir_revisar": args.incluir_revisar,
        "canales": CANALES,
        "unidades": UNIDADES,
        "normalizacion": "ninguna" if args.sin_zscore else "zscore por canal y partición LOSO",
        "normalizacion_X_base": "ninguna; X conserva unidades físicas",
        "semilla": args.semilla,
        "tomas_validacion_por_persona_actividad": args.tomas_validacion,
        "epsilon_zscore": args.epsilon_zscore,
    }


def leer_csv(f, fs_hz):
    """Valida el CSV completo sin ordenar, interpolar ni eliminar filas."""
    contenido = f.read_bytes()
    df = pd.read_csv(io.BytesIO(contenido), encoding="utf-8-sig", skip_blank_lines=False)
    if not isinstance(df.index, pd.RangeIndex):
        raise ValueError("Hay filas con más campos que el encabezado; no se permite inferir un índice.")
    df.columns = df.columns.str.strip()
    requeridas = ["t_s", *CANALES]
    if df.columns.duplicated().any() or set(df.columns) != set(requeridas):
        raise ValueError(f"Columnas inesperadas: {list(df.columns)}; se requieren {requeridas}.")
    if len(df) < 2:
        raise ValueError("Se necesitan al menos dos filas para validar el tiempo.")
    valores = df[requeridas].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(valores).all():
        raise ValueError("El tiempo o los canales contienen NaN o infinitos.")
    tiempos, canales = valores[:, 0], valores[:, 1:]
    dt = np.diff(tiempos)
    if np.any(dt <= 0):
        raise ValueError("t_s debe aumentar estrictamente: hay tiempos repetidos o retrocesos.")
    if not np.allclose(dt, 1 / fs_hz, rtol=TOLERANCIA_TIEMPO_RELATIVA, atol=1e-9):
        raise ValueError(f"Los intervalos de t_s no corresponden a {fs_hz:g} Hz (tolerancia 5 %). No se remuestrea.")
    if np.any(np.abs(canales) > np.finfo(np.float32).max):
        raise ValueError("Hay valores que no pueden representarse como float32 finito.")
    tiempo = {
        "inicio_s": float(tiempos[0]),
        "fin_s": float(tiempos[-1]),
        "intervalo_primera_ultima_muestra_s": float(tiempos[-1] - tiempos[0]),
        "frecuencia_mediana_csv_hz": float(1 / np.median(dt)),
        "dt_min_s": float(dt.min()),
        "dt_max_s": float(dt.max()),
        "nota": "La regularidad del CSV no demuestra que cada lectura del sensor sea nueva.",
    }
    return canales.astype(np.float32), tiempos, tiempo, hashlib.sha256(contenido).hexdigest()


def _rechazar_constante_json(valor):
    raise ValueError(f"Constante JSON no finita: {valor}")


def _float_json(valor):
    numero = float(valor)
    if not math.isfinite(numero):
        raise ValueError(f"Número JSON fuera de rango: {valor}")
    return numero


def leer_metadata(f):
    """Lee un JSON opcional; un fallo no invalida por sí solo el CSV."""
    ruta = f.with_suffix(".json")
    if not ruta.exists():
        return {"estado": "ausente", "datos": None}, []
    try:
        datos = json.loads(
            ruta.read_text(encoding="utf-8-sig"),
            parse_constant=_rechazar_constante_json,
            parse_float=_float_json,
        )
        if not isinstance(datos, dict):
            raise ValueError("El JSON debe contener un objeto.")
    except (OSError, UnicodeError, ValueError) as exc:
        return {"estado": "invalido", "datos": None}, [f"JSON no interpretable; se evalúa el CSV: {exc}"]
    return {"estado": "disponible", "datos": datos}, []


def _clave_persona(nombre):
    return re.sub(r"^persona(?=\d+$)", "p", normalizar(nombre))


def avisos_metadata(datos, persona, actividad, muestras, fs_hz):
    avisos = []
    for campo, esperado in (("fs_hz", fs_hz), ("numero_muestras", muestras)):
        if campo in datos and datos[campo] != esperado:
            avisos.append(f"JSON {campo}={datos[campo]!r}, pero el CSV/configuración indica {esperado}.")
    if "persona" in datos and (not isinstance(datos["persona"], str) or _clave_persona(datos["persona"]) != _clave_persona(persona)):
        avisos.append("La persona del JSON no coincide con la identificada en el CSV.")
    if "actividad" in datos and (not isinstance(datos["actividad"], str) or resolver_actividad(datos["actividad"]) != actividad):
        avisos.append("La actividad del JSON no coincide con la identificada en el CSV.")
    if "calidad_revisar" in datos and not isinstance(datos["calidad_revisar"], bool):
        avisos.append("calidad_revisar del JSON no es un booleano; no se interpreta como marca de revisión.")
    return avisos


def _repeticiones(x):
    iguales = np.all(x[1:] == x[:-1], axis=1)
    racha = maxima = 1
    for igual in iguales:
        racha = racha + 1 if igual else 1
        maxima = max(maxima, racha)
    n = int(iguales.sum())
    return {
        "pares_consecutivos_iguales": n,
        "porcentaje_pares_iguales": 100 * n / max(1, len(x) - 1),
        "racha_maxima_muestras_iguales": maxima,
    }


def calidad(x: np.ndarray):
    """Indicadores de la señal. No infiere huecos ni descarta tomas."""
    if len(x) == 0:
        raise ValueError("No se puede medir calidad de una señal vacía.")
    valores = x.astype(np.float64)
    cerca_limite = (np.abs(valores[:, :3]) >= LIMITE_ACEL_G).any(axis=1)
    return {
        "muestras": len(x),
        "muestras_cerca_limite_acel": int(cerca_limite.sum()),
        "porcentaje_cerca_limite_acel": float(100 * cerca_limite.mean()),
        "repeticiones": {
            "todos_los_canales": _repeticiones(x),
            "acelerometro": _repeticiones(x[:, :3]),
            "giroscopio": _repeticiones(x[:, 3:]),
        },
        "norma_acel_media_g": float(np.linalg.norm(valores[:, :3], axis=1).mean()),
        "por_canal": {
            c: {"min": float(v.min()), "max": float(v.max()), "media": float(v.mean()), "desviacion_estandar": float(v.std())}
            for c, v in zip(CANALES, valores.T)
        },
    }


def cargar_tomas(carpeta: Path, config):
    """Carga tomas válidas y registra cada omisión con su motivo."""
    tomas, avisos, omitidos = [], [], []
    archivos = sorted(carpeta.rglob("*.csv"))
    identificadores = set()
    for f in archivos:
        archivo = f.relative_to(carpeta).as_posix()
        persona, actividad, _ = identificar(f, carpeta)
        metadata, notas = leer_metadata(f)
        datos_meta = metadata["datos"] or {}
        notas = list(notas)
        for nota in notas:
            avisos.append(f"{archivo}: {nota}")
        revision = any(normalizar(p) == "revisar" for p in f.relative_to(carpeta).parts[:-1]) or datos_meta.get("calidad_revisar") is True
        motivo = None
        if revision and not config["incluir_revisar"]:
            motivo = "Pendiente de revisión (carpeta revisar o marca explícita del JSON)."
        elif persona is None or actividad is None:
            motivo = "Persona o actividad no reconocida en la ruta/nombre."
        if motivo is None:
            try:
                x, tiempos, tiempo, huella = leer_csv(f, config["fs_hz"])
                minimo = 2 * config["recorte_muestras_por_lado"] + config["ventana_muestras"]
                if len(x) < minimo:
                    raise ValueError(f"Longitud insuficiente: {len(x)} muestras; se necesitan al menos {minimo} con este recorte y ventana.")
                nombre = f"{persona}/{actividad}/{f.stem}"
                if nombre in identificadores:
                    raise ValueError(f"Identificador de toma duplicado: {nombre}.")
            except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
                motivo = str(exc)
        if motivo is not None:
            omitidos.append({"archivo": archivo, "motivo": motivo, "metadata_estado": metadata["estado"]})
            print(f"[omitido] {archivo}: {motivo}")
            continue
        notas_nuevas = avisos_metadata(datos_meta, persona, actividad, len(x), config["fs_hz"])
        if revision:
            notas_nuevas.append("Se incluye una toma pendiente de revisión por --incluir-revisar.")
        notas.extend(notas_nuevas)
        avisos.extend(f"{archivo}: {nota}" for nota in notas_nuevas)
        identificadores.add(nombre)
        tomas.append({
            "nombre": nombre, "persona": persona, "actividad": actividad,
            "archivo": archivo, "sha256_csv": huella, "datos": x, "tiempos": tiempos,
            "tiempo_csv": tiempo, "metadata_adquisicion": metadata, "avisos": notas,
        })
    return tomas, avisos, omitidos, len(archivos)


def ventanear(x: np.ndarray, ventana: int, paso: int) -> np.ndarray:
    if ventana < 1 or paso < 1:
        raise ValueError("La ventana y el desplazamiento deben ser positivos.")
    n = (len(x) - ventana) // paso + 1
    if n <= 0:
        return np.empty((0, ventana, x.shape[1]), dtype=x.dtype)
    return np.stack([x[i * paso:i * paso + ventana] for i in range(n)])


def ajustar_zscore(X_train, epsilon):
    """Ajusta seis medias/desviaciones usando solo ventanas de entrenamiento.

    Cada posición temporal de cada ventana tiene el mismo peso. Con solape,
    una muestra compartida participa una vez por cada ventana que la contiene.
    La desviación es poblacional (ddof=0); epsilon protege canales constantes.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon debe ser finito y mayor que cero.")
    valores = np.asarray(X_train, dtype=np.float64, order="C")
    if valores.ndim != 3 or valores.shape[0] == 0 or valores.shape[1] == 0:
        raise ValueError("Z-score necesita ventanas de entrenamiento no vacías (N, W, C).")
    if not np.isfinite(valores).all():
        raise ValueError("Z-score necesita datos finitos.")
    media = valores.mean(axis=(0, 1))
    desviacion = valores.std(axis=(0, 1), ddof=0)
    escala = np.maximum(desviacion, epsilon)
    return media, desviacion, escala


def aplicar_zscore(X, media, escala):
    """Aplica las estadísticas congeladas de train, también a val/test."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalizado = ((np.asarray(X, dtype=np.float64) - media) / escala).astype(np.float32)
    if not np.isfinite(normalizado).all():
        raise ValueError("El Z-score produjo valores no finitos; revisar la señal y epsilon.")
    return normalizado


def preparar_loso_zscore(X, y, personas, tomas, config):
    """Deja una persona para test y separa val por tomas completas de las demás.

    Los índices apuntan a X/y base. Cada fold vuelve a ajustar su Z-score;
    ni la persona de test ni las tomas de val contribuyen a esas estadísticas.
    """
    participantes = sorted(set(personas.tolist()))
    if len(participantes) < 3:
        raise ValueError("LOSO requiere al menos tres personas. Usa --sin-zscore para generar solo ventanas físicas.")
    campos, resumen = {}, []
    for numero, persona_test in enumerate(participantes, 1):
        prefijo = f"fold_{numero}"
        rng = np.random.default_rng(config["semilla"])
        tomas_val = []
        for persona in participantes:
            if persona == persona_test:
                continue
            for clase in np.unique(y[personas == persona]):
                candidatas = np.unique(tomas[(personas == persona) & (y == clase)])
                cantidad = config["tomas_validacion_por_persona_actividad"]
                if len(candidatas) <= cantidad:
                    raise ValueError(
                        f"{prefijo}: {persona}/{ACTIVIDADES[int(clase)]} necesita al menos "
                        f"{cantidad + 1} tomas para separar train/val. "
                        "No se dividen ventanas de una toma; --sin-zscore genera solo el dataset base."
                    )
                tomas_val.extend(rng.choice(candidatas, size=cantidad, replace=False).tolist())
        es_test = personas == persona_test
        es_val = ~es_test & np.isin(tomas, tomas_val)
        es_train = ~es_test & ~es_val
        indices = {
            "train": np.flatnonzero(es_train),
            "val": np.flatnonzero(es_val),
            "test": np.flatnonzero(es_test),
        }
        clases_esperadas = set(y.tolist())
        for conjunto in ("train", "val"):
            if set(y[indices[conjunto]].tolist()) != clases_esperadas:
                raise ValueError(f"{prefijo}: {conjunto} no contiene todas las clases presentes en el dataset.")
        # Comprobación explícita de independencia entre los grupos.
        for primero, segundo in (("train", "val"), ("train", "test"), ("val", "test")):
            if np.intersect1d(tomas[indices[primero]], tomas[indices[segundo]]).size:
                raise ValueError(f"{prefijo}: una toma aparece en {primero} y {segundo}.")
        media, desviacion, escala = ajustar_zscore(X[indices["train"]], config["epsilon_zscore"])
        conjuntos = {}
        for conjunto, idx in indices.items():
            campos[f"{prefijo}_X_{conjunto}"] = aplicar_zscore(X[idx], media, escala)
            campos[f"{prefijo}_y_{conjunto}"] = y[idx]
            campos[f"{prefijo}_indices_{conjunto}"] = idx
            conjuntos[conjunto] = {
                "ventanas": len(idx),
                "personas": sorted(set(personas[idx].tolist())),
                "tomas": sorted(set(tomas[idx].tolist())),
                "ventanas_por_actividad": {
                    actividad: int((y[idx] == i).sum()) for i, actividad in enumerate(ACTIVIDADES)
                },
            }
        campos[f"{prefijo}_media"] = media
        campos[f"{prefijo}_desviacion"] = desviacion
        campos[f"{prefijo}_escala"] = escala
        campos[f"{prefijo}_persona_test"] = np.asarray(persona_test)
        resumen.append({
            "fold": prefijo, "persona_test": persona_test, "conjuntos": conjuntos,
            "media": media.tolist(), "desviacion": desviacion.tolist(), "escala": escala.tolist(),
            "canales_con_desviacion_menor_que_epsilon": [
                c for c, d in zip(CANALES, desviacion) if d < config["epsilon_zscore"]
            ],
        })
    campos["folds_loso"] = np.asarray([f["fold"] for f in resumen])
    campos["normalizacion_loso"] = np.asarray("zscore_por_canal_ajustado_solo_train")
    campos["epsilon_zscore"] = np.asarray(config["epsilon_zscore"])
    return campos, {
        "metodo": "LOSO; validación por tomas completas dentro de las personas restantes",
        "semilla": config["semilla"],
        "tomas_validacion_por_persona_actividad": config["tomas_validacion_por_persona_actividad"],
        "formula": "z = (x - media_train) / max(desviacion_train, epsilon)",
        "epsilon": config["epsilon_zscore"], "ddof": 0,
        "ajuste": "Media/desviación por canal sobre los ejes ventana y tiempo de train exclusivamente; con solape se pondera por aparición en ventanas.",
        "X_base": "Unidades físicas; usar fold_k_X_train/val/test para entrenar con Z-score.",
        "agrupacion": "Por toma; no se infieren sesiones de adquisición que no están documentadas.",
        "folds": resumen,
    }


def main(args):
    config = configurar(args)
    entrada = Path(args.entrada).expanduser().resolve()
    if not entrada.is_dir():
        raise ValueError(f"No existe la carpeta de entrada: {entrada}")
    salida = Path(args.salida).expanduser().resolve()
    ruta_json = Path(args.reporte).expanduser().resolve() if args.reporte else salida.with_suffix(".json")
    if salida.suffix.lower() != ".npz" or ruta_json.suffix.lower() != ".json":
        raise ValueError("Las salidas deben tener extensión .npz y .json, respectivamente.")
    for ruta in (salida, ruta_json):
        if ruta.exists() and not args.sobrescribir:
            raise ValueError(f"La salida ya existe: {ruta}. Usa otro nombre o --sobrescribir.")
    if ruta_json.is_relative_to(entrada):
        raise ValueError("Guarda el reporte fuera de la carpeta de CSV para proteger los JSON de adquisición.")
    tomas, avisos, omitidos, encontrados = cargar_tomas(entrada, config)
    if not tomas:
        raise ValueError("No hay tomas válidas con al menos una ventana. No se genera un NPZ vacío.")

    recorte, ventana, paso = (config[k] for k in ("recorte_muestras_por_lado", "ventana_muestras", "paso_muestras"))
    X, y, personas, nombres, archivos, inicios, tiempos_inicio = [], [], [], [], [], [], []
    por_toma = []
    print(f"{'toma':55s} {'límite*':>7s} {'repet.*':>7s} {'vent.':>6s}")
    for t in tomas:
        original = t["datos"]
        fin = len(original) - recorte
        x = original[recorte:fin]
        antes, despues = calidad(original), calidad(x)
        w = ventanear(x, ventana, paso)
        comienzos = np.arange(len(w), dtype=np.int64) * paso + recorte
        print(f"{t['nombre']:55s} {despues['muestras_cerca_limite_acel']:7d} {despues['repeticiones']['todos_los_canales']['pares_consecutivos_iguales']:7d} {len(w):6d}")
        por_toma.append({
            "toma": t["nombre"], "persona": t["persona"], "actividad": t["actividad"],
            "archivo": t["archivo"], "sha256_csv": t["sha256_csv"],
            "muestras": len(original), "muestras_tras_recorte": len(x), "ventanas": len(w),
            "muestras_finales_sin_ventana": len(x) - (int(comienzos[-1]) - recorte + ventana),
            # Campos históricos: indicadores sobre la señal original, como en v1.
            "muestras_saturadas": antes["muestras_cerca_limite_acel"],
            "muestras_repetidas": antes["repeticiones"]["todos_los_canales"]["pares_consecutivos_iguales"],
            "norma_acel_media_g": round(antes["norma_acel_media_g"], 3),
            "calidad_antes_recorte": antes, "calidad_despues_recorte": despues,
            "tiempo_csv": t["tiempo_csv"], "metadata_adquisicion": t["metadata_adquisicion"],
            "avisos": t["avisos"],
        })
        X.append(w)
        y.extend([ACTIVIDADES.index(t["actividad"])] * len(w))
        personas.extend([t["persona"]] * len(w))
        nombres.extend([t["nombre"]] * len(w))
        archivos.extend([t["archivo"]] * len(w))
        inicios.extend(comienzos)
        tiempos_inicio.extend(t["tiempos"][comienzos])

    X = np.concatenate(X).astype(np.float32, copy=False)
    y = np.asarray(y, dtype=np.int64)
    personas = np.asarray(personas)
    nombres = np.asarray(nombres)
    inicios = np.asarray(inicios, dtype=np.int64)
    campos_loso, normalizacion = {}, None
    if not args.sin_zscore:
        campos_loso, normalizacion = preparar_loso_zscore(X, y, personas, nombres, config)
    conteo_clases = {c: int((y == i).sum()) for i, c in enumerate(ACTIVIDADES)}
    conteo_personas = {p: int((personas == p).sum()) for p in sorted(set(personas.tolist()))}
    reporte = {
        "version_preprocesamiento": VERSION,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entrada": str(entrada), "salida_npz": str(salida), "parametros": config,
        "dependencias": {"numpy": np.__version__, "pandas": pd.__version__},
        "dataset": {
            "forma_X": list(X.shape), "total_ventanas": len(y), "total_tomas": len(tomas),
            "csv_encontrados": encontrados, "csv_omitidos": len(omitidos),
            "ventanas_por_actividad": conteo_clases, "ventanas_por_persona": conteo_personas,
            "ventanas_por_persona_y_actividad": {
                p: {c: int(((personas == p) & (y == i)).sum()) for i, c in enumerate(ACTIVIDADES)}
                for p in conteo_personas
            },
            "metadata_adquisicion": {
                estado: sum(t["metadata_adquisicion"]["estado"] == estado for t in tomas)
                for estado in ("disponible", "ausente", "invalido")
            },
        },
        "notas": [
            "X base conserva unidades físicas. El Z-score, cuando está activo, se guarda en fold_k_X_train/val/test y se ajusta aquí solo con train.",
            "Separar por persona/toma/sesión, nunca aleatoriamente por ventanas superpuestas.",
            "JSON ausente significa información no disponible, no cero huecos ni mala calidad.",
            "Saturación y repeticiones son indicadores; no se eliminan filas por ellos.",
            "Los campos históricos muestras_saturadas/repetidas se refieren a la señal original.",
            "inicio_muestra es un índice desde cero en el CSV; fin_muestra_exclusivo no se incluye.",
        ],
        "tomas": por_toma, "avisos": avisos, "archivos_omitidos": omitidos,
        "carpetas_o_archivos_omitidos": [o["archivo"] for o in omitidos],
        "normalizacion_loso": normalizacion,
    }
    manifiesto = {
        "version_preprocesamiento": VERSION, "parametros": config,
        "sha256_csv": {t["archivo"]: t["sha256_csv"] for t in tomas},
        "normalizacion_loso": normalizacion,
    }
    for ruta in (salida, ruta_json):
        ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        salida, X=X, y=y, clases=np.asarray(ACTIVIDADES), persona=personas, toma=np.asarray(nombres),
        archivo=np.asarray(archivos), inicio_muestra=inicios, fin_muestra_exclusivo=inicios + ventana,
        t_inicio_s=np.asarray(tiempos_inicio), canales=np.asarray(CANALES), unidades=np.asarray(UNIDADES),
        fs_hz=np.asarray(config["fs_hz"]), recorte_muestras_por_lado=np.asarray(recorte),
        ventana_muestras=np.asarray(ventana), paso_muestras=np.asarray(paso), solape=np.asarray(config["solape"]),
        normalizacion=np.asarray("ninguna"), version_preprocesamiento=np.asarray(VERSION),
        manifiesto_json=np.asarray(json.dumps(manifiesto, ensure_ascii=False, allow_nan=False)),
        **campos_loso,
    )
    ruta_json.write_text(json.dumps(reporte, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("* Indicadores después del recorte; no son criterios automáticos de rechazo.")
    print(f"X: {X.shape}; {len(tomas)} tomas aceptadas, {len(omitidos)} omitidas.")
    print(f"Ventanas por actividad: {conteo_clases}")
    print(f"JSON de adquisición: {reporte['dataset']['metadata_adquisicion']}")
    if normalizacion is not None:
        for fold in normalizacion["folds"]:
            cantidades = {nombre: datos["ventanas"] for nombre, datos in fold["conjuntos"].items()}
            print(f"Z-score {fold['fold']}: test={fold['persona_test']}, ventanas={cantidades}")
    for aviso in avisos:
        print(f"[aviso] {aviso}")
    print(f"Dataset: {salida}\nReporte: {ruta_json}")
    return reporte


def crear_parser():
    p = argparse.ArgumentParser(description="Preprocesamiento HAR con Z-score por partición LOSO")
    p.add_argument("--entrada", default="datos", help="Carpeta con los CSV")
    p.add_argument("--salida", default="dataset_har.npz")
    p.add_argument("--reporte", default=None, help="JSON de calidad; por defecto junto al NPZ")
    p.add_argument("--fs-hz", type=float, default=FS_HZ, help="Frecuencia nominal esperada; no remuestrea (100 Hz)")
    p.add_argument("--recorte", type=float, default=2.0, help="Segundos a recortar por lado (2)")
    p.add_argument("--ventana", type=float, default=2.0, help="Tamaño de ventana en segundos (2)")
    p.add_argument("--solape", type=float, default=0.5, help="Fracción de solape, 0 <= solape < 1 (0.5)")
    p.add_argument("--incluir-revisar", action="store_true", help="Incluir tomas marcadas para revisión")
    p.add_argument("--sobrescribir", action="store_true", help="Permitir reemplazar salidas existentes")
    p.add_argument("--sin-zscore", action="store_true", help="Generar solo ventanas físicas, sin particiones ni Z-score")
    p.add_argument("--semilla", type=int, default=42, help="Semilla para seleccionar las tomas de validación (42)")
    p.add_argument("--tomas-validacion", type=int, default=1, help="Tomas completas de validación por persona/actividad (1)")
    p.add_argument("--epsilon-zscore", type=float, default=1e-8, help="Mínimo divisor del Z-score (1e-8)")
    return p


if __name__ == "__main__":
    parser = crear_parser()
    try:
        main(parser.parse_args())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
