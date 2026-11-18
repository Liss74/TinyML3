"""Verifica reconexion, identidad, CRC e inferencia sin iniciar hardware real.

python -B -m unittest discover -s Evaluation -p test_prueba_har.py -v
"""
import argparse
import contextlib
import io
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
from bleak.exc import BleakError
import Bleak as transporte
import prueba_har as har


def resultados_sinteticos():
    datos = struct.pack("<BBBBIII", 25, 5, 0, 0, 1234, 1500, 4096)
    datos += bytes([0, 255, 0, 0, 0, 0]) * 25
    return datos + bytes((-len(datos)) % har.PAYLOAD)


class PlacaSimulada:
    def __init__(self):
        self.id = 0
        self.estado = har.REPOSO
        self.version = 4
        self.resultados = resultados_sinteticos()
        self.crudos = np.tile([1000, 0, -500, 20, -10, 5], (har.N_MUESTRAS, 1)).astype("<i2").tobytes()
        self.comandos = []
        self.clientes = []
        self.perder_ack_start = False
        self.desconectar_resultado = None
        self.omitir_resultado = None
        self.corromper_resultado = False
        self.reiniciar_al_descargar = False
        self.fallar_conexion = False

    def cliente(self, *args, **kwargs):
        client = ClienteSimulado(self)
        self.clientes.append(client)
        return client

    def meta(self):
        return struct.pack(transporte.META_FORMATO, self.version, self.id,
                           3000 if self.id else 0, 0, 0, 0, 30000 if self.id else 0, 150)


class ClienteSimulado:
    def __init__(self, placa):
        self.placa = placa
        self.is_connected = False
        self.callbacks = {}
        self.gatt_obsoleto = False
        self.services = self

    def get_characteristic(self, uuid):
        return object()

    async def connect(self):
        if self.placa.fallar_conexion:
            self.placa.fallar_conexion = False
            raise BleakError("failed to discover services, device disconnected")
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False
        if self.placa.estado == har.ENVIANDO:
            self.placa.estado = har.LISTO

    async def start_notify(self, uuid, callback):
        self.callbacks[uuid] = callback

    async def read_gatt_char(self, uuid):
        if not self.is_connected or self.gatt_obsoleto:
            raise BleakError("Service Discovery has not been performed yet")
        if uuid == har.UUID_META:
            return self.placa.meta()
        if uuid == har.UUID_ESTADO:
            return bytes([self.placa.estado])
        if uuid == har.UUID_CRC_RESULTADOS:
            return struct.pack("<I", zlib.crc32(self.placa.resultados))
        if uuid == transporte.UUID_CRC:
            return struct.pack("<I", zlib.crc32(self.placa.crudos))
        raise AssertionError(uuid)

    async def write_gatt_char(self, uuid, datos, response=True):
        placa = self.placa
        placa.comandos.append(bytes(datos))
        if datos[0] == har.CMD_START:
            placa.id += 1
            placa.estado = har.LISTO  # La captura/inferencia se simulan completas.
            if placa.perder_ack_start:
                placa.perder_ack_start = False
                self.is_connected = False
                raise BleakError("START aceptado, respuesta perdida")
            return
        if datos[0] == transporte.CMD_ABORT:
            placa.estado = har.LISTO
            return
        desde = int.from_bytes(datos[1:], "little")
        if datos[0] == har.CMD_RESULTADOS:
            contenido = bytearray(placa.resultados)
            omitir = placa.omitir_resultado
            placa.omitir_resultado = None
            if placa.corromper_resultado:
                contenido[17] ^= 1
                placa.corromper_resultado = False
        elif datos[0] in (har.CMD_SEND, transporte.CMD_RESEND):
            contenido = placa.crudos
            omitir = None
        else:
            raise AssertionError(f"Comando inesperado {datos}")
        placa.estado = har.ENVIANDO
        total = len(contenido) // har.PAYLOAD
        secuencias = [desde] if datos[0] == transporte.CMD_RESEND else range(desde, total)
        for seq in secuencias:
            if seq == omitir:
                continue
            if datos[0] == har.CMD_RESULTADOS and seq == placa.desconectar_resultado:
                placa.desconectar_resultado = None
                self.is_connected = False
                placa.estado = har.LISTO
                if placa.reiniciar_al_descargar:
                    placa.id = 0
                    placa.estado = har.REPOSO
                raise BleakError("Enlace perdido")
            paquete = struct.pack("<H", seq) + contenido[seq*18:(seq+1)*18]
            self.callbacks[har.UUID_DATOS](None, paquete)
        placa.estado = har.LISTO
        self.callbacks[har.UUID_ESTADO](None, bytes([har.LISTO]))


class ResultadosTests(unittest.TestCase):
    def test_cabecera_probabilidades_y_relleno(self):
        res = har.parsear_resultados(resultados_sinteticos())
        self.assertEqual(res["n_ventanas"], 25)
        self.assertEqual(res["inferencia_media_us"], 1234)
        self.assertEqual(res["inferencia_max_us"], 1500)
        self.assertEqual(res["arena_usada_bytes"], 4096)
        self.assertEqual(res["ventanas"][0]["probabilidades"][0], 255 / 256)

    def test_resultados_incompletos_o_de_otro_firmware(self):
        valido = resultados_sinteticos()
        casos = [b"", valido[:15], valido[:165], bytes([0, 0]) + valido[2:]]
        clase_invalida = bytearray(valido)
        clase_invalida[16] = 5
        casos.append(clase_invalida)
        for datos in casos:
            with self.subTest(n=len(datos)), self.assertRaises(RuntimeError):
                har.parsear_resultados(datos)

    def test_csv_recupera_unidades_y_tiempo_nominal(self):
        valores = np.tile([1000, 0, -500, 20, -10, 5], (har.N_MUESTRAS, 1)).astype("<i2")
        with tempfile.TemporaryDirectory() as carpeta:
            archivo = Path(carpeta) / "crudos.csv"
            har.guardar(valores.tobytes(), archivo)
            tabla = np.loadtxt(archivo, delimiter=",", skiprows=1)
            self.assertEqual(tabla.shape, (3000, 7))
            np.testing.assert_allclose(tabla[0], [0, 1, 0, -0.5, 2, -1, 0.5])
            self.assertEqual(tabla[-1, 0], 29.99)
            with self.assertRaises(RuntimeError):
                har.guardar(valores.tobytes()[:-2], archivo)


class ProtocoloTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.placa = PlacaSimulada()
        self.enterContext(patch.object(transporte, "BleakClient", side_effect=self.placa.cliente))
        self.enterContext(patch.object(transporte.BleakScanner, "find_device_by_name", new=AsyncMock(return_value=object())))
        self.enterContext(patch.object(har.asyncio, "sleep", new=AsyncMock()))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.sesion = har.Sesion(1, version=4)

    async def asyncTearDown(self):
        await self.sesion.cerrar()

    async def test_sesion_antigua_sin_servicios_se_reemplaza(self):
        await self.sesion.conectar()
        viejo = self.sesion.client
        viejo.gatt_obsoleto = True
        await self.sesion.conectar()
        self.assertIsNot(viejo, self.sesion.client)
        self.assertFalse(viejo.is_connected)
        self.assertEqual(len(self.placa.clientes), 2)

    async def test_desconexion_durante_descubrimiento_reintenta_cliente_nuevo(self):
        self.placa.fallar_conexion = True
        await self.sesion.conectar()
        self.assertTrue(self.sesion.client.is_connected)
        self.assertEqual(len(self.placa.clientes), 2)

    async def test_start_aceptado_sin_respuesta_no_se_repite(self):
        self.placa.perder_ack_start = True
        id_toma, _ = await har.iniciar_toma(self.sesion)
        self.assertEqual(id_toma, 1)
        self.assertEqual(self.placa.comandos, [bytes([har.CMD_START])])

    async def test_no_acepta_reinicio_como_confirmacion_de_start(self):
        self.placa.id = 9
        self.placa.estado = har.LISTO
        async def reiniciar(*_):
            self.placa.id = 0
        with patch.object(self.sesion, "comando", side_effect=reiniciar):
            with self.assertRaisesRegex(RuntimeError, "reinicio"):
                await har.iniciar_toma(self.sesion)

    async def test_captura_se_espera_sin_cliente_conectado(self):
        async def contador(_):
            self.assertIsNone(self.sesion.client)
            self.assertTrue(all(not c.is_connected for c in self.placa.clientes))
        with patch.object(har, "mostrar_contadores", side_effect=contador) as cuenta:
            meta = await har.lanzar_prueba(self.sesion)
        self.assertEqual(meta.id_toma, 1)
        cuenta.assert_awaited_once()

    async def test_resultados_reanudan_primer_faltante_tras_desconexion(self):
        self.placa.id = 1
        self.placa.estado = har.LISTO
        self.placa.desconectar_resultado = 4
        raw, crc = await har.pedir_resultados(self.sesion, 1)
        self.assertEqual(raw, self.placa.resultados)
        self.assertEqual(crc, zlib.crc32(raw))
        self.assertEqual(self.placa.comandos, [struct.pack("<BH", 5, 0), struct.pack("<BH", 5, 4)])

    async def test_resultados_reenvian_paquete_omitido(self):
        self.placa.id = 1
        self.placa.estado = har.LISTO
        self.placa.omitir_resultado = 7
        raw, _ = await har.pedir_resultados(self.sesion, 1)
        self.assertEqual(raw, self.placa.resultados)
        self.assertEqual(self.placa.comandos[-1], struct.pack("<BH", 5, 7))

    async def test_crc_detecta_corrupcion_y_descarga_otra_vez(self):
        self.placa.id = 1
        self.placa.estado = har.LISTO
        self.placa.corromper_resultado = True
        raw, _ = await har.pedir_resultados(self.sesion, 1)
        self.assertEqual(raw, self.placa.resultados)
        self.assertEqual(self.placa.comandos, [struct.pack("<BH", 5, 0)] * 2)

    async def test_reinicio_durante_descarga_no_mezcla_tomas_ni_envia_start(self):
        self.placa.id = 1
        self.placa.estado = har.LISTO
        self.placa.desconectar_resultado = 4
        self.placa.reiniciar_al_descargar = True
        with self.assertRaisesRegex(RuntimeError, "no conserva"):
            await har.pedir_resultados(self.sesion, 1)
        self.assertNotIn(bytes([har.CMD_START]), self.placa.comandos)

    async def test_firmware_adquisicion_se_rechaza_antes_de_start(self):
        self.placa.version = 3
        with self.assertRaises(har.FirmwareIncompatible):
            await self.sesion.conectar()
        self.assertEqual(self.placa.comandos, [])

    async def test_crudos_usan_protocolo_v3_y_crc(self):
        self.placa.id = 1
        self.placa.estado = har.LISTO
        datos, meta, crc = await har.descargar(self.sesion, 1)
        self.assertEqual(datos, self.placa.crudos)
        self.assertEqual(meta.id_toma, 1)
        self.assertEqual(crc, zlib.crc32(datos))

    def argumentos(self, carpeta, recuperar=False):
        return argparse.Namespace(persona="Persona4", actividad="reposo", pruebas=1,
                                  salida=carpeta, crudos=True, recuperar=recuperar,
                                  reconexion_timeout=1)

    async def test_flujo_completo_guarda_resultados_y_crudos_sin_perder_etiqueta(self):
        with tempfile.TemporaryDirectory() as carpeta:
            with patch.object(har.asyncio, "to_thread", new=AsyncMock(return_value="")), \
                 patch.object(har, "mostrar_contadores", new=AsyncMock()):
                codigo = await har.main(self.argumentos(carpeta))
            self.assertEqual(codigo, 0)
            ruta = Path(carpeta)/"Persona4_reposo_001.json"
            datos = json.loads(ruta.read_text())
            self.assertEqual(datos["accuracy_prueba"], 1)
            self.assertEqual(datos["adquisicion"]["id_toma"], 1)
            self.assertEqual(datos["actividad_real"], "reposo")
            self.assertTrue(ruta.with_suffix(".csv").exists())
            self.assertEqual(self.placa.comandos.count(bytes([har.CMD_START])), 1)

    async def test_recuperar_exige_etiqueta_confirmada_y_no_envia_start(self):
        self.placa.id = 8
        self.placa.estado = har.LISTO
        with tempfile.TemporaryDirectory() as carpeta:
            with patch.object(har.asyncio, "to_thread", new=AsyncMock(return_value="s")):
                self.assertEqual(await har.main(self.argumentos(carpeta, recuperar=True)), 0)
            datos = json.loads((Path(carpeta)/"Persona4_reposo_001.json").read_text())
            self.assertEqual(datos["adquisicion"]["id_toma"], 8)
            self.assertNotIn(bytes([har.CMD_START]), self.placa.comandos)

    async def test_cancelar_recuperacion_no_descarga_ni_etiqueta(self):
        self.placa.id = 8
        self.placa.estado = har.LISTO
        with tempfile.TemporaryDirectory() as carpeta:
            with patch.object(har.asyncio, "to_thread", new=AsyncMock(return_value="n")):
                self.assertEqual(await har.main(self.argumentos(carpeta, recuperar=True)), 0)
            self.assertEqual(list(Path(carpeta).iterdir()), [])
            self.assertEqual(self.placa.comandos, [])


if __name__ == "__main__":
    unittest.main()
