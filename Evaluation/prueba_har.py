"""
Prueba de inferencia HAR en el Arduino Nano 33 BLE Sense, controlada por BLE.

Por cada prueba:
  1. Envía START: la placa hace la cuenta atrás, graba 30 s a 100 Hz e INFIERE las 25 ventanas
     (recorte 2 s por lado, ventana de 200 muestras, paso 100) con el modelo INT8.
  2. Pide los resultados (clase y probabilidades por ventana, tiempos de inferencia, arena usada).
  3. Compara con la actividad que declaras y guarda un JSON en pruebas/.
  4. Con --crudos descarga también los datos crudos y los guarda como CSV (formato de adquisición_har.py).

Requiere Bleak.py en la misma carpeta, es el archivo con el que se adquirieron los datos inicialmente.

Uso:
  python prueba_har.py --persona Persona4 --actividad caminar --pruebas 3
"""
import argparse
import asyncio
import json
import struct
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from bleak import BleakClient, BleakScanner

from Bleak import (
    ACTIVIDADES,
    CMD_START,
    DURACION_S,
    GRABANDO,
    LISTO,
    NOMBRE_DISPOSITIVO,
    UUID_CONTROL,
    UUID_DATOS,
    UUID_ESTADO,
    Receptor,
    descargar,
    esperar_estado,
    guardar,
)

CMD_RESULTADOS = 0x03
PAYLOAD = 18
BYTES_CABECERA = 16
SALIDA_ESCALA = 1 / 256          # escala de la salida softmax INT8 (ver cuantizacion_resultados.json)
MAX_REINTENTOS = 5


def parsear_resultados(raw: bytes):
    """Interpreta el bloque de resultados enviado por la placa."""
    n_vent, n_clases = raw[0], raw[1]
    t_medio_us, t_max_us, arena = struct.unpack("<III", raw[4:16])
    ventanas = []
    for v in range(n_vent):
        fila = raw[BYTES_CABECERA + v * (1 + n_clases) : BYTES_CABECERA + (v + 1) * (1 + n_clases)]
        probs = np.frombuffer(fila[1:], dtype=np.uint8).astype(np.float64) * SALIDA_ESCALA
        ventanas.append({"clase": int(fila[0]), "probabilidades": probs.tolist()})
    return {
        "n_ventanas": n_vent,
        "inferencia_media_us": t_medio_us,
        "inferencia_max_us": t_max_us,
        "arena_usada_bytes": arena,
        "ventanas": ventanas,
    }


async def lanzar_prueba(client: BleakClient, rx: Receptor):
    rx.paquetes.clear()
    rx.estado = None
    await client.write_gatt_char(UUID_CONTROL, bytes([CMD_START]), response=True)
    print("  Cuenta atrás: colócate y prepárate...")
    await esperar_estado(rx, GRABANDO, 10)
    print(f"  ¡GRABANDO! ({DURACION_S} s) Realiza la actividad.")
    await esperar_estado(rx, LISTO, DURACION_S + 60)   # incluye la inferencia en la placa
    print("  La placa terminó de grabar e inferir.")


async def pedir_resultados(client: BleakClient, rx: Receptor):
    """Pide los paquetes de resultados; reenvía desde el primero que falte."""
    rx.paquetes.clear()
    desde, n_paq = 0, None
    for _ in range(MAX_REINTENTOS):
        await client.write_gatt_char(UUID_CONTROL, struct.pack("<BH", CMD_RESULTADOS, desde), response=True)
        await asyncio.sleep(1.5)
        if rx.paquetes:
            # El primer paquete trae el nº de ventanas y clases: calcula cuántos paquetes esperar
            if 0 in rx.paquetes:
                n_vent, n_clases = rx.paquetes[0][0], rx.paquetes[0][1]
                total = BYTES_CABECERA + n_vent * (1 + n_clases)
                n_paq = -(-total // PAYLOAD)
        if n_paq is not None:
            faltan = [s for s in range(n_paq) if s not in rx.paquetes]
            if not faltan:
                raw = b"".join(rx.paquetes[s] for s in range(n_paq))
                return raw
            desde = faltan[0]
        else:
            desde = 0
    raise RuntimeError("No se recibieron todos los paquetes de resultados.")


def resumir(res: dict, actividad: str):
    esperada = ACTIVIDADES.index(actividad)
    clases = [v["clase"] for v in res["ventanas"]]
    aciertos = sum(c == esperada for c in clases)
    acc = aciertos / len(clases)
    print(f"\n  {'ventana':>7s}  {'predicción':16s} {'confianza':>9s}")
    for i, v in enumerate(res["ventanas"]):
        marca = "" if v["clase"] == esperada else "  <-- error"
        print(f"  {i:7d}  {ACTIVIDADES[v['clase']]:16s} {max(v['probabilidades']):9.2f}{marca}")
    conteo = {a: clases.count(i) for i, a in enumerate(ACTIVIDADES) if clases.count(i)}
    print(f"\n  Actividad real: {actividad} | accuracy de la prueba: {acc:.3f} ({aciertos}/{len(clases)})")
    print(f"  Predicciones: {conteo}")
    print(f"  Inferencia en la placa: media {res['inferencia_media_us'] / 1000:.1f} ms, "
          f"máx {res['inferencia_max_us'] / 1000:.1f} ms | arena usada: {res['arena_usada_bytes']} bytes")
    return acc, conteo


def siguiente_indice(carpeta: Path, persona: str, actividad: str) -> int:
    indices = []
    for f in carpeta.glob(f"{persona}_{actividad}_*.json"):
        try:
            indices.append(int(f.stem.rsplit("_", 1)[1]))
        except ValueError:
            pass
    return max(indices, default=0) + 1


async def main(args):
    carpeta = Path(args.salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    print(f"Buscando '{NOMBRE_DISPOSITIVO}'...")
    dispositivo = await BleakScanner.find_device_by_name(NOMBRE_DISPOSITIVO, timeout=15)
    if dispositivo is None:
        sys.exit("No se encontró la placa. ¿Está encendida, con el sketch cargado y sin otra conexión BLE?")

    rx = Receptor()
    async with BleakClient(dispositivo) as client:
        print("Conectado.")
        await client.start_notify(UUID_ESTADO, rx.on_estado)
        await client.start_notify(UUID_DATOS, rx.on_datos)

        hechas = 0
        while hechas < args.pruebas:
            resp = await asyncio.to_thread(
                input,
                f"\nPrueba {hechas + 1}/{args.pruebas} de '{args.actividad}' ({args.persona}). "
                "Enter para empezar, q para salir: ",
            )
            if resp.strip().lower() == "q":
                break
            try:
                await lanzar_prueba(client, rx)
                res = parsear_resultados(await pedir_resultados(client, rx))
                acc, conteo = resumir(res, args.actividad)

                idx = siguiente_indice(carpeta, args.persona, args.actividad)
                base = f"{args.persona}_{args.actividad}_{idx:03d}"
                if args.crudos:
                    print("  Descargando datos crudos...")
                    raw = await descargar(client, rx)
                    guardar(raw, carpeta / f"{base}.csv")
            except (TimeoutError, RuntimeError) as e:
                print(f"  Error: {e}. Prueba descartada.")
                continue

            reporte = {
                "fecha": datetime.now().isoformat(timespec="seconds"),
                "persona": args.persona,
                "actividad_real": args.actividad,
                "accuracy_prueba": acc,
                "conteo_predicciones": conteo,
                **res,
            }
            with open(carpeta / f"{base}.json", "w", encoding="utf-8") as f:
                json.dump(reporte, f, ensure_ascii=False, indent=2)
            print(f"  Guardado: {carpeta / (base + '.json')}")
            hechas += 1

    print("\nListo.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prueba de inferencia HAR por BLE")
    p.add_argument("--persona", required=True)
    p.add_argument("--actividad", required=True, choices=ACTIVIDADES)
    p.add_argument("--pruebas", type=int, default=1, help="Número de pruebas a realizar")
    p.add_argument("--salida", default="pruebas")
    p.add_argument("--crudos", action="store_true", help="Descargar también los datos crudos (CSV)")
    asyncio.run(main(p.parse_args()))
