#!/usr/bin/env python3
"""
Adquisición de datos HAR por BLE (Arduino Nano 33 BLE Sense).

Flujo por cada toma:
  1. Envía START a la placa (cuenta atrás de 3 s + grabación de 30 s a 100 Hz).
  2. Espera a que la placa pase a estado LISTO.
  3. Pide los datos (SEND desde el paquete 0), recibe 2000 paquetes numerados
     y, si falta alguno, pide reenvío desde el primer paquete faltante.
  4. Guarda un CSV etiquetado: datos/<persona>_<actividad>_<NNN>.csv

Uso:
  pip install bleak numpy
  python Bleak.py --persona P1 --actividad caminar --tomas 5
"""
import argparse
import asyncio
import struct
import sys
from pathlib import Path

import numpy as np
from bleak import BleakClient, BleakScanner

# ---------- Deben coincidir con el sketch de Arduino ----------
NOMBRE_DISPOSITIVO = "HAR-Nano"
UUID_CONTROL = "7a1c0002-5b6e-4c1d-9a3f-3f2b8c9d0001"
UUID_ESTADO = "7a1c0003-5b6e-4c1d-9a3f-3f2b8c9d0001"
UUID_DATOS = "7a1c0004-5b6e-4c1d-9a3f-3f2b8c9d0001"

CMD_START = 0x01
CMD_SEND = 0x02

REPOSO, CUENTA_ATRAS, GRABANDO, LISTO, ENVIANDO = range(5)

FS_HZ = 100
DURACION_S = 30
N_MUESTRAS = FS_HZ * DURACION_S            # 3000
N_CANALES = 6                              # ax ay az gx gy gz
PAYLOAD = 18                               # bytes de datos por paquete
N_PAQUETES = N_MUESTRAS * N_CANALES * 2 // PAYLOAD   # 2000
ESC_ACEL = 1000.0                          # int16 -> g
ESC_GIRO = 10.0                            # int16 -> °/s

ACTIVIDADES = ["reposo", "caminar", "correr", "subir_escaleras", "bajar_escaleras"]
MAX_REINTENTOS = 5


class Receptor:
    """Guarda el último estado notificado y los paquetes de datos recibidos."""

    def __init__(self):
        self.estado = None
        self.paquetes = {}

    def on_estado(self, _, data: bytearray):
        self.estado = data[0]

    def on_datos(self, _, data: bytearray):
        if len(data) < 2 + PAYLOAD:
            return
        seq = int.from_bytes(data[:2], "little")
        if seq < N_PAQUETES:
            self.paquetes[seq] = bytes(data[2:2 + PAYLOAD])


async def esperar_estado(rx: Receptor, objetivo: int, timeout: float):
    loop = asyncio.get_running_loop()
    limite = loop.time() + timeout
    while rx.estado != objetivo:
        if loop.time() > limite:
            raise TimeoutError(f"La placa no llegó al estado {objetivo} (último: {rx.estado})")
        await asyncio.sleep(0.05)


async def grabar(client: BleakClient, rx: Receptor):
    rx.paquetes.clear()
    rx.estado = None
    await client.write_gatt_char(UUID_CONTROL, bytes([CMD_START]), response=True)

    await esperar_estado(rx, CUENTA_ATRAS, 5)
    print("  Cuenta atrás: colócate y prepárate...")
    await esperar_estado(rx, GRABANDO, 10)
    print(f"  ¡GRABANDO! ({DURACION_S} s) Realiza la actividad.")
    await esperar_estado(rx, LISTO, DURACION_S + 10)
    print("  Grabación terminada. Descargando datos...")


async def descargar(client: BleakClient, rx: Receptor) -> bytes:
    desde = 0
    for intento in range(1, MAX_REINTENTOS + 1):
        rx.estado = None
        await client.write_gatt_char(
            UUID_CONTROL, struct.pack("<BH", CMD_SEND, desde), response=True
        )
        await esperar_estado(rx, ENVIANDO, 5)
        await esperar_estado(rx, LISTO, 120)
        await asyncio.sleep(0.3)  # deja llegar los últimos paquetes

        faltan = [s for s in range(N_PAQUETES) if s not in rx.paquetes]
        print(f"  Intento {intento}: {N_PAQUETES - len(faltan)}/{N_PAQUETES} paquetes")
        if not faltan:
            return b"".join(rx.paquetes[s] for s in range(N_PAQUETES))
        desde = faltan[0]

    raise RuntimeError("No se pudieron recibir todos los paquetes; sube PAUSA_PAQUETE_MS en el sketch")


def guardar(raw: bytes, ruta: Path):
    datos = np.frombuffer(raw, dtype="<i2").reshape(N_MUESTRAS, N_CANALES).astype(np.float64)
    datos[:, :3] /= ESC_ACEL
    datos[:, 3:] /= ESC_GIRO
    t = np.arange(N_MUESTRAS) / FS_HZ
    tabla = np.column_stack([t, datos])

    np.savetxt(
        ruta,
        tabla,
        delimiter=",",
        header="t_s,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps",
        comments="",
        fmt=["%.2f"] + ["%.3f"] * 3 + ["%.1f"] * 3,
    )

    # Verificación rápida: en reposo la norma de la aceleración debe rondar 1 g
    norma = np.linalg.norm(datos[:, :3], axis=1).mean()
    print(f"  Guardado: {ruta}  (|a| media = {norma:.2f} g)")
    if np.all(datos == 0):
        print("  AVISO: todos los datos son 0; revisa el IMU.")


def siguiente_indice(carpeta: Path, persona: str, actividad: str) -> int:
    indices = []
    for f in carpeta.glob(f"{persona}_{actividad}_*.csv"):
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
        while hechas < args.tomas:
            resp = await asyncio.to_thread(
                input,
                f"\nToma {hechas + 1}/{args.tomas} de '{args.actividad}' ({args.persona}). "
                "Enter para empezar, q para salir: ",
            )
            if resp.strip().lower() == "q":
                break
            try:
                await grabar(client, rx)
                raw = await descargar(client, rx)
            except (TimeoutError, RuntimeError) as e:
                print(f"  Error: {e}. Toma descartada.")
                continue

            idx = siguiente_indice(carpeta, args.persona, args.actividad)
            guardar(raw, carpeta / f"{args.persona}_{args.actividad}_{idx:03d}.csv")
            hechas += 1

    print("\nListo.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Adquisición HAR por BLE")
    p.add_argument("--persona", required=True, help="Identificador de la persona (ej. P1)")
    p.add_argument("--actividad", required=True, choices=ACTIVIDADES)
    p.add_argument("--tomas", type=int, default=5, help="Número de tomas a grabar")
    p.add_argument("--salida", default="datos", help="Carpeta de salida")
    asyncio.run(main(p.parse_args()))