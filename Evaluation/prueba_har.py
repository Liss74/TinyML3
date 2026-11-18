"""Evaluacion HAR por BLE, protocolo de inferencia v4.

Confirma START por id de toma; desconecta durante los 30 s de captura y la
inferencia autonoma; reconecta y reanuda la descarga con verificacion CRC32.
La actividad real se declara con --actividad; nunca se envia al clasificador.
--crudos guarda tambien las muestras en CSV. --recuperar descarga una toma
pendiente sin repetir START, previa confirmacion de su etiqueta por el usuario.
Requiere cargar HAR_Inferencia_Nano33BLE.ino v4. Los buffers sobreviven a una
desconexion BLE, pero se pierden al apagar/reiniciar o iniciar otra toma.
"""
import argparse
import asyncio
import json
import struct
import time
import zlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from bleak.exc import BleakError

from Bleak import (
    ACTIVIDADES, CMD_START, CMD_SEND, CUENTA_ATRAS, GRABANDO, INFERIENDO,
    LISTO, ENVIANDO, REPOSO, ESC_ACEL, ESC_GIRO, FS_HZ, N_BYTES, N_CANALES,
    N_MUESTRAS, N_PAQUETES, PAYLOAD, UUID_DATOS, UUID_ESTADO, UUID_META,
    ConexionPerdida, FirmwareIncompatible, Receptor, Sesion,
    descargar, detener_envio_si_necesario, esperar_toma_lista, iniciar_toma,
    mostrar_contadores, nombre_estado,
)

CMD_RESULTADOS = 5  # 3 y 4 se reservan a RESEND/ABORT, como en adquisicion v3.
UUID_CRC_RESULTADOS = "7a1c0007-5b6e-4c1d-9a3f-3f2b8c9d0001"
BYTES_CABECERA = 16
SALIDA_ESCALA = 1 / 256
MAX_REINTENTOS = 5
N_VENTANAS = (N_MUESTRAS - 2 * 200 - 200) // 100 + 1
N_PAQ_RESULTADOS = (BYTES_CABECERA + N_VENTANAS * (1 + len(ACTIVIDADES)) + PAYLOAD - 1) // PAYLOAD


def validar_toma(meta, id_esperado, completa=True):
    if meta.version != 4:
        raise FirmwareIncompatible("Se requiere HAR_Inferencia_Nano33BLE.ino v4.")
    if meta.id_toma != id_esperado or (completa and meta.muestras != N_MUESTRAS):
        raise RuntimeError(
            f"La placa no conserva la toma #{id_esperado} completa "
            f"(actual #{meta.id_toma}, {meta.muestras} muestras). Posible reinicio."
        )


async def crc_resultados(sesion):
    datos = await sesion._operacion(lambda c: c.read_gatt_char(UUID_CRC_RESULTADOS))
    if len(datos) != 4:
        raise RuntimeError("CRC32 de resultados invalido.")
    return int.from_bytes(datos, "little")


async def lanzar_prueba(sesion):
    id_toma, inicio = await iniciar_toma(sesion)
    print(f"  START confirmado: toma #{id_toma}. Captura e inferencia autonomas.")
    await sesion.cerrar()
    await mostrar_contadores(inicio)
    meta = await esperar_toma_lista(sesion, id_toma)
    print("  La placa termino de grabar e inferir.")
    return meta


async def pedir_resultados(sesion, id_esperado):
    """Reanuda la misma toma; nunca envia START durante la recuperacion."""
    receptor_limpio = False
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            await sesion.conectar()
            validar_toma(await sesion.meta(), id_esperado)
            await detener_envio_si_necesario(sesion)
            if not receptor_limpio:
                # Drena la suscripcion previa antes de reutilizar secuencias de
                # una descarga de crudos interrumpida como resultados.
                await sesion.cerrar()
                sesion.rx.nueva_toma()
                await sesion.conectar()
                validar_toma(await sesion.meta(), id_esperado)
                receptor_limpio = True
            crc = await crc_resultados(sesion)
            faltantes = [s for s in range(N_PAQ_RESULTADOS) if s not in sesion.rx.paquetes]
            if faltantes:
                desde = faltantes[0]
                print(f"  Resultados, intento {intento}: desde paquete {desde}.")
                sesion.rx.estado = None
                await sesion.comando(CMD_RESULTADOS, desde)
                inicio = time.monotonic()
                ultima_lectura = inicio
                while time.monotonic() - inicio < 10:
                    if all(s in sesion.rx.paquetes for s in range(N_PAQ_RESULTADOS)):
                        break
                    if sesion.client is None or not sesion.client.is_connected:
                        raise ConexionPerdida("Enlace perdido durante los resultados")
                    if time.monotonic() - ultima_lectura >= 1:
                        await sesion.estado()
                        ultima_lectura = time.monotonic()
                    if sesion.rx.estado == LISTO:
                        break
                    await asyncio.sleep(0.05)
                await detener_envio_si_necesario(sesion)
            if any(s not in sesion.rx.paquetes for s in range(N_PAQ_RESULTADOS)):
                continue
            validar_toma(await sesion.meta(), id_esperado)
            if await crc_resultados(sesion) != crc:
                raise RuntimeError("Los resultados cambiaron durante la descarga.")
            raw = b"".join(sesion.rx.paquetes[s] for s in range(N_PAQ_RESULTADOS))
            if zlib.crc32(raw) & 0xFFFFFFFF != crc:
                print("  CRC32 no coincide; solicitando otra vez todos los resultados.")
                sesion.rx.nueva_toma()
                continue
            parsear_resultados(raw)
            return raw, crc
        except ConexionPerdida as exc:
            print(f"  [BLE] {exc}. Recuperando la misma toma...")
            await sesion.cerrar()
    raise RuntimeError("Descarga de resultados incompleta o corrupta; la toma sigue en RAM si la placa no reinicio.")


def guardar(raw, ruta):
    """Convierte int16 a g/grados/s y conserva el formato CSV de adquisición."""
    if len(raw) != N_BYTES:
        raise RuntimeError(f"Se esperaban {N_BYTES} bytes de datos crudos.")
    valores = np.frombuffer(raw, dtype="<i2").reshape(N_MUESTRAS, N_CANALES).astype(np.float64)
    valores[:, :3] /= ESC_ACEL
    valores[:, 3:] /= ESC_GIRO
    tabla = np.column_stack((np.arange(N_MUESTRAS) / FS_HZ, valores))
    np.savetxt(
        ruta, tabla, delimiter=",", comments="",
        header="t_s,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps",
        fmt=["%.2f"] + ["%.3f"] * 3 + ["%.1f"] * 3,
    )


def parsear_resultados(raw: bytes):
    """Interpreta el bloque de resultados enviado por la placa."""
    if len(raw) < BYTES_CABECERA:
        raise RuntimeError("Resultados incompletos: falta la cabecera.")
    n_vent, n_clases = raw[0], raw[1]
    if n_vent != N_VENTANAS or n_clases != len(ACTIVIDADES):
        raise RuntimeError(
            f"Formato inesperado: {n_vent} ventanas y {n_clases} clases; "
            f"se esperaban {N_VENTANAS} y {len(ACTIVIDADES)}. "
            "Verifica que la placa tenga el sketch de inferencia."
        )
    if len(raw) < BYTES_CABECERA + n_vent * (1 + n_clases):
        raise RuntimeError("Resultados incompletos: faltan datos de ventanas.")
    t_medio_us, t_max_us, arena = struct.unpack("<III", raw[4:16])
    ventanas = []
    for v in range(n_vent):
        fila = raw[BYTES_CABECERA + v * (1 + n_clases) : BYTES_CABECERA + (v + 1) * (1 + n_clases)]
        if fila[0] >= n_clases:
            raise RuntimeError(f"Clase inválida en la ventana {v}: {fila[0]}.")
        probs = np.frombuffer(fila[1:], dtype=np.uint8).astype(np.float64) * SALIDA_ESCALA
        ventanas.append({"clase": int(fila[0]), "probabilidades": probs.tolist()})
    return {
        "n_ventanas": n_vent,
        "inferencia_media_us": t_medio_us,
        "inferencia_max_us": t_max_us,
        "arena_usada_bytes": arena,
        "ventanas": ventanas,
    }


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


async def recuperar_prueba(sesion, args):
    await sesion.conectar()
    meta = await sesion.meta()
    estado = await sesion.estado()
    if meta.id_toma == 0 or estado == REPOSO:
        raise RuntimeError("No hay una toma pendiente en RAM. Inicia una prueba sin --recuperar.")
    print(f"  Toma pendiente #{meta.id_toma}, estado {nombre_estado(estado)}.")
    await sesion.cerrar()
    respuesta = await asyncio.to_thread(
        input,
        f"¿Confirmas que esa toma es de {args.persona}, actividad '{args.actividad}'? [s/N]: ",
    )
    if respuesta.strip().lower() != "s":
        return None
    # Conserva el id observado: un reinicio durante la espera no debe crear una toma nueva.
    return await esperar_toma_lista(sesion, meta.id_toma)


async def main(args):
    carpeta = Path(args.salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    sesion = Sesion(args.reconexion_timeout, version=4)
    hechas = 0
    try:
        while hechas < args.pruebas:
            if args.recuperar and hechas == 0:
                meta = await recuperar_prueba(sesion, args)
                if meta is None:
                    break
            else:
                # Espera al usuario sin mantener una conexion o servicios GATT antiguos.
                await sesion.cerrar()
                respuesta = await asyncio.to_thread(
                    input,
                    f"\nPrueba {hechas + 1}/{args.pruebas} de '{args.actividad}' ({args.persona}). "
                    "Enter para conectar y empezar, q para salir: ",
                )
                if respuesta.strip().lower() == "q":
                    break
                await sesion.conectar()
                estado = await sesion.estado()
                if estado not in (REPOSO, LISTO):
                    raise RuntimeError(
                        f"Hay una toma en {nombre_estado(estado)}. "
                        "Usa --recuperar para terminarla sin repetir START."
                    )
                meta = await lanzar_prueba(sesion)

            raw_resultados, crc = await pedir_resultados(sesion, meta.id_toma)
            res = parsear_resultados(raw_resultados)
            acc, conteo = resumir(res, args.actividad)
            idx = siguiente_indice(carpeta, args.persona, args.actividad)
            nombre = f"{args.persona}_{args.actividad}_{idx:03d}"
            reporte = {
                "fecha": datetime.now().isoformat(timespec="seconds"),
                "persona": args.persona,
                "actividad_real": args.actividad,
                "accuracy_prueba": acc,
                "conteo_predicciones": conteo,
                "adquisicion": asdict(meta),
                "crc32_resultados": f"{crc:08X}",
                **res,
            }
            ruta = carpeta / f"{nombre}.json"
            # Los resultados ya verificados se conservan aunque luego falle el CSV opcional.
            ruta.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  Guardado: {ruta}")
            if args.crudos:
                print("  Descargando datos crudos de la misma toma...")
                await sesion.cerrar()
                sesion.rx.nueva_toma()
                crudos, _, crc_crudos = await descargar(sesion, meta.id_toma)
                guardar(crudos, carpeta / f"{nombre}.csv")
                reporte["crc32_crudos"] = f"{crc_crudos:08X}"
                ruta.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
            hechas += 1
            await sesion.cerrar()
    except FirmwareIncompatible as exc:
        print(f"\nFirmware incompatible: {exc} Vuelve a compilar y subir el sketch de inferencia v4.")
        return 1
    except (ConexionPerdida, BleakError, TimeoutError, RuntimeError, OSError) as exc:
        print(f"\nPrueba interrumpida: {exc}")
        print("Si la placa sigue encendida, ejecuta el mismo comando con --recuperar. "
              "No reinicies ni apagues la placa para conservar la toma.")
        return 1
    finally:
        await sesion.cerrar()
    print("\nListo.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluacion HAR con captura autonoma y reconexion BLE")
    p.add_argument("--persona", required=True)
    p.add_argument("--actividad", required=True, choices=ACTIVIDADES)
    p.add_argument("--pruebas", type=int, default=1)
    p.add_argument("--salida", default="pruebas")
    p.add_argument("--crudos", action="store_true", help="Guardar tambien muestras en CSV, verificadas con CRC32")
    p.add_argument("--recuperar", action="store_true", help="Recuperar la toma en RAM sin iniciar otra")
    p.add_argument("--reconexion-timeout", type=float, default=90, help="Limite por reconexion, en segundos")
    args = p.parse_args()
    if args.pruebas < 1 or args.reconexion_timeout <= 0:
        p.error("--pruebas y --reconexion-timeout deben ser positivos")
    try:
        raise SystemExit(asyncio.run(main(args)))
    except KeyboardInterrupt:
        print("\nSalida solicitada. La placa conserva la toma en RAM mientras siga encendida.")
