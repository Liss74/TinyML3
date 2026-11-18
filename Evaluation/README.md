# Evaluación HAR con captura autónoma y recuperación BLE

Usar `HAR_Inferencia_Nano33BLE/HAR_Inferencia_Nano33BLE.ino` **v4** con
`prueba_har.py` de esta carpeta. Es necesario volver a compilar y cargar el
sketch: el firmware anterior no publica los identificadores ni los CRC.

## Prueba normal

Desde esta carpeta, con el entorno de Python que contiene `bleak` y `numpy`:

```bash
python prueba_har.py --persona Persona4 --actividad reposo --pruebas 1 --crudos
```

1. Python espera Enter sin mantener una conexión Bluetooth abierta.
2. Conecta, valida el protocolo v4 y confirma START mediante el identificador
   de toma publicado por la placa. Si se pierde la respuesta de START, verifica
   el identificador antes de considerar repetir la orden.
3. Cierra Bluetooth intencionalmente. El Arduino realiza la cuenta atrás de
   3 s, captura 30 s a 100 Hz e infiere 25 ventanas, sin depender del enlace.
4. Python vuelve a conectar y comprueba el identificador antes de descargar.
   Reanuda desde el primer paquete faltante y verifica CRC32.
5. Guarda clases, probabilidades, tiempos, memoria usada y contadores de calidad
   en JSON. Con `--crudos` descarga también las muestras y verifica su CRC32
   antes de escribir el CSV. El JSON se conserva si falla esta descarga opcional.

La actividad real proviene de `--actividad`; no se envía al clasificador.
Los contadores de consola aproximan el tiempo; ESTADO y META confirman la toma.

## Estados del Arduino

```text
REPOSO → CUENTA_ATRAS → GRABANDO → INFERIENDO → LISTO
                                                ↕
                                            ENVIANDO
```

La captura usa el reloj del Arduino, con prioridad al muestreo. Como en
adquisición v3, se contabilizan las ranuras rellenadas y las lecturas de IMU
reutilizadas. El LED parpadea durante la preparación y queda encendido durante
la grabación. Un error fatal produce parpadeo rápido.

Perder Bluetooth durante CUENTA_ATRAS, GRABANDO o INFERIENDO no cancela la toma.
Durante ENVIANDO detiene la transferencia y regresa a LISTO. Los datos y
resultados permanecen en RAM para reconectar y reanudar. La placa vuelve a
anunciarse tras detectar la desconexión; durante la captura, atender BLE tiene
menor prioridad que conservar el reloj de muestreo.

**Apagar, reiniciar o enviar otro START elimina la toma anterior.** No existe
persistencia en memoria flash. La alimentación puede provenir de USB sin un
monitor serie abierto; debe mantenerse durante toda la prueba y recuperación.
Una pérdida de alimentación no se corrige reconectando Bluetooth.

## Recuperar una toma pendiente

Si termina Python o agota los reintentos de conexión, mantener la placa encendida:

```bash
python prueba_har.py --persona Persona4 --actividad reposo --pruebas 1 --crudos --recuperar
```

El programa muestra el identificador pendiente y pide confirmar persona y
actividad, pues la placa no conoce esas etiquetas. Espera la captura/inferencia
si siguen en curso y recupera la toma **sin enviar START**. Si cambió el
identificador o desaparecieron las muestras, se rechaza la descarga.

`--reconexion-timeout 90` establece el tiempo de búsqueda/reintentos por
reconexión. No es una garantía del tiempo total de una prueba.

## Protocolo v4

Se conservan los UUID de adquisición v3 y su formato META de 19 bytes, con
versión 4. META incluye identificador de toma, número de muestras, huecos,
lecturas reutilizadas, duración y retraso máximo. La característica `0005`
publica el CRC32 de crudos; `0007` publica el CRC32 del bloque de resultados,
incluido su relleno hasta múltiplo de 18 bytes. El resto del UUID es
`7a1cXXXX-5b6e-4c1d-9a3f-3f2b8c9d0001`.

| Comando | Código | Función |
|---|---:|---|
| START | 1 | Iniciar desde REPOSO o LISTO |
| SEND | 2 | Crudos desde una secuencia |
| RESEND | 3 | Retransmitir un paquete de crudos |
| ABORT | 4 | Detener el envío, conservando la toma |
| RESULTADOS | 5 | Resultados desde una secuencia |

Se conserva el modelo INT8, las estadísticas de normalización y el formato
de las 25 predicciones. La escala de salida usada por el lector es 1/256,
correspondiente al modelo actual.

## Verificación sin placa

Desde la raíz del proyecto:

```bash
python -B -m unittest discover -s Evaluation -p test_prueba_har.py -v
```

Las pruebas simulan pérdida de servicios GATT, desconexión al conectar,
confirmación de START, recuperación de paquetes, CRC, reinicio, recuperación
con etiqueta confirmada y escritura de resultados. No sustituyen una prueba
física con el celular como fuente de alimentación.
