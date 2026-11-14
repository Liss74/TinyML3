#!/usr/bin/env python3
"""Adquisicion HAR robusta por BLE | Arduino Nano 33 BLE Sense | protocolo v3.
Incluye contadores de 3 s y 30 s en consola (sin cambiar el firmware).

Instalar: pip install bleak numpy
Ejemplo:  python Bleak_v3.py --persona P2 --actividad caminar --tomas 1

Flujo: conectar -> START -> desconectar intencionalmente -> grabar 30 s
sin depender de BLE -> reconectar -> transferir -> verificar CRC32 -> CSV.
La grabacion permanece en RAM del Nano tras perder Bluetooth, NO tras reiniciarlo.
Requiere HAR_Nano_BLE_v3.ino; incompatible con firmware v2.
"""

import argparse
import asyncio
import json
import struct
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

NOMBRE = "HAR-Nano"
UUID_CONTROL = "7a1c0002-5b6e-4c1d-9a3f-3f2b8c9d0001"
UUID_ESTADO  = "7a1c0003-5b6e-4c1d-9a3f-3f2b8c9d0001"
UUID_DATOS   = "7a1c0004-5b6e-4c1d-9a3f-3f2b8c9d0001"
UUID_CRC     = "7a1c0005-5b6e-4c1d-9a3f-3f2b8c9d0001"
UUID_META    = "7a1c0006-5b6e-4c1d-9a3f-3f2b8c9d0001"
CMD_START, CMD_SEND, CMD_RESEND, CMD_ABORT = 1, 2, 3, 4
REPOSO, CUENTA_ATRAS, GRABANDO, LISTO, ENVIANDO = range(5)
NOMBRES_ESTADO = ("REPOSO", "CUENTA_ATRAS", "GRABANDO", "LISTO", "ENVIANDO")

FS_HZ = 100
DURACION_S = 30
CUENTA_ATRAS_S = 3
N_MUESTRAS = FS_HZ * DURACION_S
N_CANALES = 6
PAYLOAD = 18
N_BYTES = N_MUESTRAS * N_CANALES * 2
N_PAQUETES = N_BYTES // PAYLOAD
assert N_BYTES % PAYLOAD == 0
ESC_ACEL = 1000.0
ESC_GIRO = 10.0
ACTIVIDADES = ("reposo", "caminar", "correr", "subir_escaleras", "bajar_escaleras")
META_FORMATO = "<BIHHHHIH"
META_LONGITUD = struct.calcsize(META_FORMATO)


class ConexionPerdida(ConnectionError):
    """La sesion GATT actual no sirve; debe abrirse otra."""


class FirmwareIncompatible(RuntimeError):
    pass


@dataclass(frozen=True)
class Meta:
    version: int
    id_toma: int
    muestras: int
    huecos: int
    acel_reutilizada: int
    giro_reutilizado: int
    duracion_ms: int
    retraso_max_us: int

    @classmethod
    def desde_bytes(cls, datos):
        if len(datos) != META_LONGITUD:
            raise FirmwareIncompatible(
                f"Metadata de {len(datos)} bytes: se esperaba protocolo v3 ({META_LONGITUD})"
            )
        meta = cls(*struct.unpack(META_FORMATO, bytes(datos)))
        if meta.version != 3:
            raise FirmwareIncompatible(f"Protocolo {meta.version}: se necesita firmware v3")
        return meta


def nombre_estado(valor):
    return NOMBRES_ESTADO[valor] if valor is not None and 0 <= valor < 5 else str(valor)


class Receptor:
    def __init__(self):
        self.paquetes = {}
        self.estado = None
        self.evento = asyncio.Event()
        self.duplicados = 0
        self.malformados = 0
        self.ultimo = time.monotonic()

    def nueva_toma(self):
        self.paquetes.clear()
        self.estado = None
        self.duplicados = self.malformados = 0
        self.evento.clear()
        self.ultimo = time.monotonic()

    def on_estado(self, _caracteristica, datos):
        if datos:
            self.estado = int(datos[0])

    def on_datos(self, _caracteristica, datos):
        if len(datos) != PAYLOAD + 2:
            self.malformados += 1
            return
        seq = int.from_bytes(datos[:2], "little")
        if not 0 <= seq < N_PAQUETES:
            self.malformados += 1
            return
        if seq in self.paquetes:
            self.duplicados += 1
        else:
            self.paquetes[seq] = bytes(datos[2:])
        self.ultimo = time.monotonic()
        self.evento.set()


class Sesion:
    def __init__(self, reconexion_timeout):
        self.rx = Receptor()
        self.client = None
        self.reconexion_timeout = reconexion_timeout
        self.cierre_voluntario = False

    def _desconectado(self, _cliente):
        if not self.cierre_voluntario:
            print("  [BLE] Enlace perdido. La placa debe conservar la captura en RAM.")

    async def cerrar(self):
        actual = self.client
        self.client = None
        if actual is not None:
            self.cierre_voluntario = True
            try:
                await actual.disconnect()
            except (BleakError, OSError, asyncio.TimeoutError):
                pass
            finally:
                self.cierre_voluntario = False

    async def conectar(self, timeout=None):
        if self.client is not None and self.client.is_connected:
            # BlueZ puede conservar "conectado" mientras ya perdió el cache
            # de servicios GATT. Probar una lectura ANTES de reutilizar la sesión.
            try:
                Meta.desde_bytes(await self.client.read_gatt_char(UUID_META))
                return
            except FirmwareIncompatible:
                raise
            except (BleakError, OSError, asyncio.TimeoutError):
                print("  [BLE] Sesión GATT antigua inválida; abriendo otra conexión...")
        await self.cerrar()
        timeout = self.reconexion_timeout if timeout is None else timeout
        limite = time.monotonic() + timeout
        intento = 0
        while time.monotonic() < limite:
            intento += 1
            restante = limite - time.monotonic()
            print(f"  [BLE] Buscando '{NOMBRE}' (intento {intento})...")
            try:
                dispositivo = await BleakScanner.find_device_by_name(
                    NOMBRE, timeout=min(9.0, max(1.0, restante))
                )
                if dispositivo is None:
                    await asyncio.sleep(1)
                    continue
                nuevo = BleakClient(
                    dispositivo, timeout=20, disconnected_callback=self._desconectado
                )
                try:
                    await nuevo.connect()
                    if not nuevo.is_connected:
                        raise ConexionPerdida("Conexión no establecida")
                    await nuevo.start_notify(UUID_ESTADO, self.rx.on_estado)
                    await nuevo.start_notify(UUID_DATOS, self.rx.on_datos)
                    # Obliga a descubrir servicios en una sesion GATT NUEVA.
                    datos = await nuevo.read_gatt_char(UUID_META)
                    Meta.desde_bytes(datos)
                    self.rx.estado = None
                    self.client = nuevo
                    print("  [BLE] Conectado y servicios GATT descubiertos.")
                    return
                except Exception:
                    try:
                        await nuevo.disconnect()
                    except Exception:
                        pass
                    raise
            except FirmwareIncompatible:
                raise
            except (BleakError, OSError, ConnectionError, asyncio.TimeoutError) as exc:
                print(f"  [BLE] Reintento de conexión: {exc}")
            await asyncio.sleep(1.5)
        raise ConexionPerdida(
            f"No fue posible recuperar BLE en {timeout:.0f} s. "
            "Si la placa sigue encendida, vuelve a ejecutar Python y recupera la toma pendiente."
        )

    async def _operacion(self, accion):
        if self.client is None or not self.client.is_connected:
            raise ConexionPerdida("Bluetooth desconectado")
        try:
            return await accion(self.client)
        except (BleakError, OSError, asyncio.TimeoutError, ConnectionError) as exc:
            raise ConexionPerdida(f"La sesión GATT dejó de ser válida: {exc}") from exc

    async def estado(self):
        datos = await self._operacion(lambda c: c.read_gatt_char(UUID_ESTADO))
        if len(datos) != 1:
            raise RuntimeError(f"Estado GATT inválido: {bytes(datos)!r}")
        self.rx.estado = int(datos[0])
        return self.rx.estado

    async def meta(self):
        datos = await self._operacion(lambda c: c.read_gatt_char(UUID_META))
        return Meta.desde_bytes(datos)

    async def crc(self):
        datos = await self._operacion(lambda c: c.read_gatt_char(UUID_CRC))
        if len(datos) != 4:
            raise RuntimeError("CRC32 inválido; verifica que instalaste el firmware v3")
        return int.from_bytes(datos, "little")

    async def comando(self, codigo, numero=None):
        contenido = bytes([codigo]) if numero is None else struct.pack("<BH", codigo, numero)
        await self._operacion(
            lambda c: c.write_gatt_char(UUID_CONTROL, contenido, response=True)
        )

    async def estado_reconectando(self):
        for _ in range(3):
            await self.conectar()
            try:
                return await self.estado()
            except ConexionPerdida:
                await self.cerrar()
        raise ConexionPerdida("No se logró recuperar una sesión GATT válida")


async def iniciar_toma(sesion):
    await sesion.conectar()
    estado = await sesion.estado()
    meta_anterior = await sesion.meta()
    if estado not in (REPOSO, LISTO):
        raise RuntimeError(f"No se puede iniciar una toma en {nombre_estado(estado)}")
    sesion.rx.nueva_toma()
    # Referencia temporal aproximada del instante en que se envío START.
    # El Arduino controla la cuenta atrás real de forma independiente.
    instante_orden = None

    for intento in range(3):
        try:
            # Puede existir una ventana de tiempo entre la respuesta BLE y
            # el procesamiento de START en el loop del microcontrolador.
            if intento > 0:
                await sesion.conectar()
            actual = await sesion.meta()
            if actual.id_toma != meta_anterior.id_toma:
                return actual.id_toma, (instante_orden or time.monotonic())
            instante_orden = time.monotonic()
            await sesion.comando(CMD_START)
            for _ in range(20):
                await asyncio.sleep(0.15)
                actual = await sesion.meta()
                if actual.id_toma != meta_anterior.id_toma:
                    return actual.id_toma, instante_orden
        except ConexionPerdida:
            print("  [BLE] Se interrumpió START; verificando si la placa ya comenzó...")
            await sesion.cerrar()
            await sesion.conectar()
            actual = await sesion.meta()
            if actual.id_toma != meta_anterior.id_toma:
                return actual.id_toma, (instante_orden or time.monotonic())
    raise RuntimeError("El Arduino no confirmó START. La toma no se considerará iniciada.")


async def mostrar_contadores(inicio_cuenta: float):
    """Muestra 3..2..1 y 01/30..30/30, sincronizado aproximadamente con START.

    Usa el reloj monotónico para evitar deriva por el tiempo de imprimir, dormir,
    o desconectar BLE. No confirma la adquisición: la placa lo hace con META/ESTADO.
    """
    print("  Preparación (cuenta atrás del Arduino, aprox.):", flush=True)
    for restante in range(CUENTA_ATRAS_S, 0, -1):
        instante = inicio_cuenta + (CUENTA_ATRAS_S - restante)
        demora = instante - time.monotonic()
        if demora > 0:
            await asyncio.sleep(demora)
        # No mostrar segundos obsoletos si el inicio/desconexión BLE tardó.
        if time.monotonic() - instante < 0.9:
            print(f"    {restante}...", flush=True)

    inicio_grabacion = inicio_cuenta + CUENTA_ATRAS_S
    demora = inicio_grabacion - time.monotonic()
    if demora > 0:
        await asyncio.sleep(demora)
    print(f"  ¡GRABANDO! ({DURACION_S} s). La placa almacena en RAM.", flush=True)

    for segundo in range(1, DURACION_S + 1):
        instante = inicio_grabacion + segundo
        demora = instante - time.monotonic()
        if demora > 0:
            await asyncio.sleep(demora)
        # Se imprimen los segundos de consola; los datos reales se validan luego.
        print(f"  Grabación: {segundo:02d}/{DURACION_S:02d} s", flush=True)

    # Margen para que el Arduino termine CRC32 y publique LISTO, incluso con
    # un pequeño desfase del reloj local respecto a la confirmación de START.
    fin_estimado = inicio_grabacion + DURACION_S
    demora = fin_estimado + 1.5 - time.monotonic()
    if demora > 0:
        await asyncio.sleep(demora)


async def esperar_toma_lista(sesion, id_esperado, timeout=75):
    limite = time.monotonic() + timeout
    ultimo_estado = None
    while time.monotonic() < limite:
        try:
            estado = await sesion.estado_reconectando()
            meta = await sesion.meta()
        except ConexionPerdida:
            await sesion.cerrar()
            await sesion.conectar()
            continue
        if meta.id_toma != id_esperado:
            raise RuntimeError(
                f"La placa reinició o cambió de toma (esperada #{id_esperado}, actual #{meta.id_toma})."
            )
        if estado != ultimo_estado:
            print(f"  Estado Arduino: {nombre_estado(estado)}")
            ultimo_estado = estado
        if estado == ENVIANDO and meta.muestras == N_MUESTRAS:
            # Puede quedar un envío de una sesión interrumpida.
            await detener_envio_si_necesario(sesion)
            estado = LISTO
        if estado == LISTO and meta.muestras == N_MUESTRAS:
            print(
                f"  Grabación lista: {meta.muestras} muestras, "
                f"huecos={meta.huecos}, duración={meta.duracion_ms} ms"
            )
            return meta
        if estado == REPOSO:
            raise RuntimeError("El Arduino regresó a REPOSO: posible reinicio o toma inexistente")
        await asyncio.sleep(1)
    raise TimeoutError(
        "La toma sigue sin estar LISTO. "
        "El buffer podrá recuperarse si la placa completa la captura y permanece encendida."
    )


async def esperar_envio(sesion, limite_s=100):
    inicio = time.monotonic()
    ultimo_reporte = inicio
    ultimo_estado = inicio
    while time.monotonic() - inicio < limite_s:
        if len(sesion.rx.paquetes) == N_PAQUETES:
            return
        if sesion.client is None or not sesion.client.is_connected:
            raise ConexionPerdida("Desconexión durante la transferencia")
        ahora = time.monotonic()
        if sesion.rx.estado == LISTO:
            return
        if ahora - ultimo_estado > 2:
            actual = await sesion.estado()
            ultimo_estado = time.monotonic()
            if actual == LISTO:
                return
        if ahora - ultimo_reporte > 5:
            recibidos = len(sesion.rx.paquetes)
            print(f"  Paquetes: {recibidos}/{N_PAQUETES} ({100*recibidos/N_PAQUETES:.1f}%)")
            ultimo_reporte = ahora
        if ahora - sesion.rx.ultimo > 18:
            print("  [BLE] No llegan paquetes; se intentará reanudar el envío.")
            return
        await asyncio.sleep(0.1)
    print("  [BLE] Descarga demoró demasiado; se intentará reanudar.")


async def detener_envio_si_necesario(sesion):
    estado = await sesion.estado()
    if estado == ENVIANDO:
        await sesion.comando(CMD_ABORT)
        for _ in range(30):
            await asyncio.sleep(0.1)
            estado = await sesion.estado()
            if estado == LISTO:
                break
    if estado != LISTO:
        raise RuntimeError(f"El Nano no está listo para descargar: {nombre_estado(estado)}")


async def recuperar_individuales(sesion, faltantes):
    for indice, seq in enumerate(faltantes, 1):
        if seq in sesion.rx.paquetes:
            continue
        for _ in range(4):
            sesion.rx.evento.clear()
            await sesion.comando(CMD_RESEND, seq)
            if seq in sesion.rx.paquetes:
                break
            try:
                await asyncio.wait_for(sesion.rx.evento.wait(), 1.5)
            except asyncio.TimeoutError:
                pass
            if seq in sesion.rx.paquetes:
                break
        if seq not in sesion.rx.paquetes:
            raise RuntimeError(f"No se recibió el paquete {seq} tras cuatro intentos")
        if indice % 25 == 0 or indice == len(faltantes):
            print(f"  Paquetes recuperados: {indice}/{len(faltantes)}")


async def descargar(sesion, id_esperado):
    """Descarga con reanudación por desconexion y recuperacion de faltantes."""
    crc_esperado = None
    inicio = time.monotonic()
    for intento in range(1, 6):
        try:
            await sesion.conectar()
            meta = await sesion.meta()
            if meta.id_toma != id_esperado or meta.muestras != N_MUESTRAS:
                raise RuntimeError("El Arduino ya no tiene la toma esperada en RAM")
            crc_esperado = await sesion.crc()
            await detener_envio_si_necesario(sesion)
            faltantes = [seq for seq in range(N_PAQUETES) if seq not in sesion.rx.paquetes]
            if not faltantes:
                break
            desde = faltantes[0]
            print(f"  Descarga intento {intento}: desde paquete {desde}")
            sesion.rx.estado = None
            sesion.rx.ultimo = time.monotonic()
            await sesion.comando(CMD_SEND, desde)
            await esperar_envio(sesion)
            await detener_envio_si_necesario(sesion)
            await asyncio.sleep(0.5)
            faltantes = [seq for seq in range(N_PAQUETES) if seq not in sesion.rx.paquetes]
            print(f"  Recibidos: {N_PAQUETES-len(faltantes)}/{N_PAQUETES}")
            if not faltantes:
                break
            if len(faltantes) <= 250:
                print(f"  Recuperando {len(faltantes)} paquetes faltantes...")
                await recuperar_individuales(sesion, faltantes)
                break
        except ConexionPerdida as exc:
            print(f"  [BLE] {exc}. Reconectando sin borrar los datos recibidos...")
            await sesion.cerrar()
            # En el Nano, la perdida de enlace termina el envio, NO la toma.
            continue
        except RuntimeError as exc:
            # Una retransmision puntual puede fallar; permitir un flujo nuevo.
            if "paquete" in str(exc).lower():
                print(f"  {exc}; reintentando descarga completa desde primer faltante...")
                continue
            raise

    if len(sesion.rx.paquetes) != N_PAQUETES:
        raise RuntimeError(
            f"Descarga incompleta ({len(sesion.rx.paquetes)}/{N_PAQUETES}). "
            "La toma sigue en RAM: vuelve a ejecutar y elige recuperar la pendiente."
        )
    datos = b"".join(sesion.rx.paquetes[s] for s in range(N_PAQUETES))
    if len(datos) != N_BYTES:
        raise RuntimeError(f"Tamaño incorrecto: {len(datos)} frente a {N_BYTES} bytes")
    real = zlib.crc32(datos) & 0xFFFFFFFF
    if real != crc_esperado:
        raise RuntimeError(
            f"CRC32 incorrecto: Arduino={crc_esperado:08X}, Python={real:08X}. "
            "No se guardaron datos corruptos."
        )
    print(
        f"  OK: {N_PAQUETES} paquetes, CRC32={real:08X}, "
        f"duplicados={sesion.rx.duplicados}, malformados={sesion.rx.malformados}, "
        f"tiempo={time.monotonic()-inicio:.1f} s"
    )
    return datos, meta, real


def siguiente_indice(carpeta, persona, actividad):
    indices = []
    for directorio in (carpeta, carpeta / "revisar"):
        for archivo in directorio.glob(f"{persona}_{actividad}_*.csv"):
            try:
                indices.append(int(archivo.stem.rsplit("_", 1)[1]))
            except ValueError:
                pass
    return max(indices, default=0) + 1


def guardar(datos, meta, crc, carpeta, persona, actividad, max_huecos):
    if len(datos) != N_BYTES:
        raise RuntimeError("No se guarda: longitud de datos incorrecta")
    revisar = meta.huecos > max_huecos
    destino = carpeta / "revisar" if revisar else carpeta
    destino.mkdir(parents=True, exist_ok=True)
    indice = siguiente_indice(carpeta, persona, actividad)
    ruta = destino / f"{persona}_{actividad}_{indice:03d}.csv"

    valores = np.frombuffer(datos, dtype="<i2").reshape(N_MUESTRAS, N_CANALES).astype(np.float64)
    valores[:, :3] /= ESC_ACEL
    valores[:, 3:] /= ESC_GIRO
    eje_t = np.arange(N_MUESTRAS) / FS_HZ
    tabla = np.column_stack((eje_t, valores))
    np.savetxt(
        ruta, tabla, delimiter=",", comments="",
        header="t_s,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps",
        fmt=["%.2f"] + ["%.3f"] * 3 + ["%.1f"] * 3,
    )
    informe = {
        "captura_utc": datetime.now(timezone.utc).isoformat(),
        "persona": persona, "actividad": actividad,
        "fs_hz": FS_HZ, "numero_muestras": N_MUESTRAS,
        "crc32": f"{crc:08X}", "calidad_revisar": revisar,
        "max_huecos_permitidos": max_huecos,
        "adquisicion": asdict(meta),
    }
    ruta.with_suffix(".json").write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    norma = float(np.linalg.norm(valores[:, :3], axis=1).mean())
    print(f"  CSV guardado: {ruta} | aceleración media={norma:.2f} g")
    if revisar:
        print(
            f"  ATENCIÓN: {meta.huecos} muestras repetidas por retrasos. "
            "Archivo separado en 'revisar/' para NO mezclarlo con tomas limpias."
        )
    print(
        "  Calidad IMU: "
        f"huecos={meta.huecos}/{N_MUESTRAS} ({100*meta.huecos/N_MUESTRAS:.2f}%), "
        f"aceleración reutilizada={meta.acel_reutilizada}/{N_MUESTRAS} "
        f"({100*meta.acel_reutilizada/N_MUESTRAS:.2f}%), "
        f"giroscopio reutilizado={meta.giro_reutilizado}/{N_MUESTRAS} "
        f"({100*meta.giro_reutilizado/N_MUESTRAS:.2f}%)"
    )
    print(
        f"  Temporización IMU: duración={meta.duracion_ms} ms, "
        f"desplazamiento máximo en ranura={meta.retraso_max_us} µs."
    )
    return ruta


async def recuperar_pendiente(sesion, args, carpeta):
    await sesion.conectar(timeout=60)
    meta = await sesion.meta()
    estado = await sesion.estado()
    if meta.id_toma == 0 or estado == REPOSO:
        return False
    if estado in (CUENTA_ATRAS, GRABANDO):
        print(f"  Se detectó una grabación #{meta.id_toma} todavía en curso.")
        await esperar_toma_lista(sesion, meta.id_toma, timeout=60)
        estado = LISTO
    if estado == ENVIANDO:
        await detener_envio_si_necesario(sesion)
        estado = LISTO
    if estado != LISTO:
        return False

    decision = await asyncio.to_thread(
        input,
        f"  Hay una toma #{meta.id_toma} pendiente en RAM. "
        f"¿Descargarla como {args.persona}/{args.actividad}? [s/N]: ",
    )
    if decision.strip().lower() not in ("s", "si", "sí"):
        print("  Se dejará pendiente hasta que inicies otra toma o reinicies la placa.")
        return False
    sesion.rx.nueva_toma()
    datos, metadata, crc = await descargar(sesion, meta.id_toma)
    guardar(datos, metadata, crc, carpeta, args.persona, args.actividad, args.max_huecos)
    return True


async def main(args):
    carpeta = Path(args.salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    sesion = Sesion(reconexion_timeout=args.reconexion_timeout)
    realizadas = 0
    try:
        print("Iniciando HAR-Nano BLE v3...")
        await sesion.conectar(timeout=60)
        if await recuperar_pendiente(sesion, args, carpeta):
            realizadas += 1
        while realizadas < args.tomas:
            respuesta = await asyncio.to_thread(
                input,
                f"\nToma {realizadas+1}/{args.tomas} de '{args.actividad}' "
                f"({args.persona}). Enter para empezar, q para salir: ",
            )
            if respuesta.strip().lower() == "q":
                break
            try:
                await sesion.conectar()
                estado = await sesion.estado()
                if estado not in (REPOSO, LISTO):
                    print(f"  La placa sigue en {nombre_estado(estado)}; esperando a que termine...")
                    await esperar_toma_lista(sesion, (await sesion.meta()).id_toma, timeout=60)
                    print("  Puedes recuperar esa toma reiniciando Python.")
                    continue
                id_toma, inicio_cuenta = await iniciar_toma(sesion)
                print(f"  START confirmado para toma #{id_toma}.")
                # Iniciar el contador en paralelo para no perder segundos
                # mientras se cierra la conexión BLE tras confirmar START.
                contador = asyncio.create_task(mostrar_contadores(inicio_cuenta))
                await sesion.cerrar()
                print("  [BLE] Desconectado intencionalmente durante la adquisición.")
                await contador
                print("  Cuenta terminada en consola. Reconectando para confirmar la toma...")
                await sesion.conectar()
                await esperar_toma_lista(sesion, id_toma, timeout=45)
                datos, meta, crc = await descargar(sesion, id_toma)
                guardar(datos, meta, crc, carpeta, args.persona, args.actividad, args.max_huecos)
                realizadas += 1
            except (RuntimeError, ConexionPerdida, TimeoutError) as exc:
                print(f"  ERROR: {exc}")
                print("  Si la placa está encendida, NO pulses RESET. Reejecuta para recuperar la toma pendiente.")
                await sesion.cerrar()
    finally:
        await sesion.cerrar()
    print(f"\nListo. Tomas guardadas durante esta ejecución: {realizadas}.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Adquisición HAR autónoma + reconexión BLE")
    p.add_argument("--persona", required=True)
    p.add_argument("--actividad", required=True, choices=ACTIVIDADES)
    p.add_argument("--tomas", type=int, default=1)
    p.add_argument("--salida", default="datos")
    p.add_argument("--reconexion-timeout", type=float, default=240,
                   help="Máximo de segundos buscando de nuevo la placa (240 por defecto)")
    p.add_argument("--max-huecos", type=int, default=30,
                   help="Si se repiten más de N ranuras, guardar CSV en datos/revisar/ (30 por defecto)")
    argumentos = p.parse_args()
    if argumentos.tomas < 1 or argumentos.reconexion_timeout < 1 or argumentos.max_huecos < 0:
        p.error("--tomas>=1, --reconexion-timeout>=1 y --max-huecos>=0")
    try:
        asyncio.run(main(argumentos))
    except KeyboardInterrupt:
        print("\nInterrumpido. La RAM de la placa conserva la toma hasta que se reinicie.")
