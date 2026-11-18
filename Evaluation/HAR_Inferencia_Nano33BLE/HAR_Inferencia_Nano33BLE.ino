#include <Arduino.h>
#include <stdint.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "modelo_har_data.h"
#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>
#include <math.h>
#include <string.h>

// HAR inferencia BLE v4 | Nano 33 BLE Sense original (LSM9DS1).
// START confirma una toma: cuenta atras, captura 30 s a 100 Hz e infiere
// 25 ventanas INT8 (recorte 2 s/lado, ventana 200, paso 100, z-score de entrenamiento).
// Captura e inferencia son autonomas: no requieren USB ni conexion BLE continua.
// Tras reconectar se descargan resultados/crudos por secuencia, con id de toma y CRC32.
// Se conserva la logica de estados de HAR_Nano_BLE_v3.ino; los datos viven
// en RAM hasta el siguiente START o un reinicio/apagado. Usar prueba_har.py v4.

constexpr uint16_t FS_HZ = 100;
constexpr uint32_t PERIODO_US = 1000000UL / FS_HZ;
constexpr uint16_t DURACION_S = 30;
constexpr uint16_t N_MUESTRAS = FS_HZ * DURACION_S;
constexpr uint8_t N_CANALES = 6;
constexpr uint32_t N_BYTES = (uint32_t)N_MUESTRAS * N_CANALES * sizeof(int16_t);
constexpr uint8_t PAYLOAD = 18;       // 2 bytes de secuencia + 18 datos = 20 por notificacion
constexpr uint16_t N_PAQUETES = N_BYTES / PAYLOAD;
constexpr uint32_t CUENTA_ATRAS_MS = 3000;
constexpr uint16_t PAUSA_PAQUETE_MS = 25;
constexpr uint8_t META_VERSION = 4;
constexpr uint8_t META_BYTES = 19;
constexpr uint32_t POLL_GRABACION_MS = 100;  // BLE es secundario durante la adquisicion

constexpr float ESC_ACEL = 1000.0f;   // g => mg
constexpr float ESC_GIRO = 10.0f;     // grados/s => decimas

static_assert(N_BYTES % PAYLOAD == 0, "El buffer debe dividirse en paquetes completos");
static_assert(N_PAQUETES < 65536UL, "Limite de secuencia excedido");

constexpr uint16_t RECORTE = 200;
constexpr uint16_t VENTANA = HAR_N_VENTANA;
constexpr uint16_t PASO = 100;
constexpr uint8_t N_VENT = (N_MUESTRAS - 2 * RECORTE - VENTANA) / PASO + 1;
constexpr uint8_t BYTES_CABECERA = 16;
constexpr uint8_t BYTES_VENT = 1 + HAR_N_CLASES;
constexpr uint16_t N_BYTES_RES = BYTES_CABECERA + N_VENT * BYTES_VENT;
constexpr uint16_t N_PAQ_RES = (N_BYTES_RES + PAYLOAD - 1) / PAYLOAD;
static_assert(HAR_N_CANALES == N_CANALES, "El modelo debe usar seis canales");

int16_t buffer[N_MUESTRAS][N_CANALES];
uint8_t resultados[N_PAQ_RES * PAYLOAD];
uint8_t ventanaActual = 0;
uint32_t tiempoSumaUs = 0, tiempoMaxUs = 0;
constexpr int kTensorArenaSize = 40 * 1024;           // ajustar con arena_used_bytes()
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* net = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;



enum Estado : uint8_t {
  REPOSO = 0, CUENTA_ATRAS = 1, GRABANDO = 2, LISTO = 3, ENVIANDO = 4, INFERIENDO = 5
};
Estado estado = REPOSO;

BLEService servicio("7a1c0001-5b6e-4c1d-9a3f-3f2b8c9d0001");
BLECharacteristic controlChar("7a1c0002-5b6e-4c1d-9a3f-3f2b8c9d0001", BLEWrite, 3);
BLEByteCharacteristic estadoChar("7a1c0003-5b6e-4c1d-9a3f-3f2b8c9d0001", BLERead | BLENotify);
BLECharacteristic datosChar("7a1c0004-5b6e-4c1d-9a3f-3f2b8c9d0001", BLENotify, 2 + PAYLOAD);
BLECharacteristic crcChar("7a1c0005-5b6e-4c1d-9a3f-3f2b8c9d0001", BLERead, 4);
BLECharacteristic metaChar("7a1c0006-5b6e-4c1d-9a3f-3f2b8c9d0001", BLERead, META_BYTES);

BLECharacteristic crcResultadosChar("7a1c0007-5b6e-4c1d-9a3f-3f2b8c9d0001", BLERead, 4);

constexpr uint8_t CMD_RESULTADOS = 0x05; // resultados desde secuencia N; 0x03 sigue siendo RESEND de crudos
constexpr uint8_t CMD_START = 0x01;
constexpr uint8_t CMD_SEND = 0x02;    // [codigo, desde_lo, desde_hi]
constexpr uint8_t CMD_RESEND = 0x03;  // [codigo, seq_lo, seq_hi]
constexpr uint8_t CMD_ABORT = 0x04;   // detiene envio, sin borrar muestras

uint32_t idToma = 0;
uint32_t inicioCuentaMs = 0;
uint32_t inicioGrabacionUs = 0;
uint32_t ultimoPollGrabacionMs = 0;
uint16_t muestrasTomadas = 0;
uint16_t huecosRellenados = 0;
uint16_t aceleracionReutilizada = 0;
uint16_t giroReutilizado = 0;
uint32_t retrasoMaxUs = 0;
uint32_t duracionRealMs = 0;
float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;

const uint8_t* datosEnvio = reinterpret_cast<const uint8_t*>(buffer);
uint16_t totalPaquetesEnvio = N_PAQUETES;
uint16_t siguientePaquete = 0;
uint16_t fallosLocales = 0;
uint32_t proximoPaqueteMs = 0;
uint32_t inicioEnvioMs = 0;
bool conectadoAnterior = false;

void setEstado(Estado nuevo) {
  estado = nuevo;
  estadoChar.writeValue((uint8_t)nuevo);
}

void fatal(const char* mensaje) {
  if (Serial) Serial.println(mensaje);
  while (true) {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    delay(100);
  }
}

uint16_t saturar16(uint32_t valor) {
  return valor > 65535UL ? 65535U : (uint16_t)valor;
}

void actualizarMeta() {
  // <B I H H H H I H> : version, id, muestras, huecos,
  // accel_stale, gyro_stale, duracion_ms, retraso_max_us (saturado)
  uint8_t dato[META_BYTES];
  uint8_t pos = 0;
  dato[pos++] = META_VERSION;
  uint32_t v32 = idToma;
  for (uint8_t j = 0; j < 4; ++j) dato[pos++] = (uint8_t)(v32 >> (8 * j));
  const uint16_t campos[4] = {
    muestrasTomadas, huecosRellenados, aceleracionReutilizada, giroReutilizado
  };
  for (uint8_t c = 0; c < 4; ++c) {
    dato[pos++] = (uint8_t)(campos[c] & 0xFF);
    dato[pos++] = (uint8_t)(campos[c] >> 8);
  }
  v32 = duracionRealMs;
  for (uint8_t j = 0; j < 4; ++j) dato[pos++] = (uint8_t)(v32 >> (8 * j));
  uint16_t maxUs = saturar16(retrasoMaxUs);
  dato[pos++] = (uint8_t)(maxUs & 0xFF);
  dato[pos++] = (uint8_t)(maxUs >> 8);
  metaChar.writeValue(dato, sizeof(dato));
}

uint32_t crc32Datos(const uint8_t* bytes, uint32_t longitud) {
  uint32_t crc = 0xFFFFFFFFUL;
  for (uint32_t i = 0; i < longitud; ++i) {
    crc ^= bytes[i];
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320UL : 0UL);
    }
  }
  return ~crc;
}

void publicarCRC(BLECharacteristic& caracteristica, const uint8_t* bytes, uint32_t longitud) {
  uint32_t crc = crc32Datos(bytes, longitud);
  uint8_t dato[4] = {
    (uint8_t)crc, (uint8_t)(crc >> 8), (uint8_t)(crc >> 16), (uint8_t)(crc >> 24)
  };
  caracteristica.writeValue(dato, sizeof(dato));
  if (Serial) { Serial.print("CRC32: 0x"); Serial.println(crc, HEX); }
}

int16_t convertirInt16(float v) {
  if (!isfinite(v)) return 0;
  if (v > 32767.0f) return 32767;
  if (v < -32768.0f) return -32768;
  return (int16_t)lroundf(v);
}

void iniciarToma() {
  ++idToma;
  memset(resultados, 0, sizeof(resultados));
  ventanaActual = 0;
  tiempoSumaUs = tiempoMaxUs = 0;
  muestrasTomadas = huecosRellenados = 0;
  aceleracionReutilizada = giroReutilizado = 0;
  retrasoMaxUs = duracionRealMs = 0;
  ax = ay = az = gx = gy = gz = 0;
  actualizarMeta();
  inicioCuentaMs = millis();
  digitalWrite(LED_BUILTIN, LOW);
  setEstado(CUENTA_ATRAS);
  if (Serial) { Serial.print("Nueva toma #"); Serial.println(idToma); }
}

void actualizarCuentaAtras() {
  uint32_t t = millis() - inicioCuentaMs;
  if (t < CUENTA_ATRAS_MS) {
    digitalWrite(LED_BUILTIN, ((t / 250UL) % 2UL) != 0);
    return;
  }
  inicioGrabacionUs = micros();
  ultimoPollGrabacionMs = millis();
  digitalWrite(LED_BUILTIN, HIGH);
  setEstado(GRABANDO);
  if (Serial) Serial.println("Grabacion autonoma iniciada");
}

void copiarMuestraAnterior(uint16_t indice) {
  if (indice == 0) {
    memset(buffer[0], 0, sizeof(buffer[0]));
  } else {
    memcpy(buffer[indice], buffer[indice - 1], sizeof(buffer[indice]));
  }
}

void finalizarGrabacion() {
  duracionRealMs = (uint32_t)(micros() - inicioGrabacionUs) / 1000UL;
  digitalWrite(LED_BUILTIN, LOW);
  actualizarMeta();
  publicarCRC(crcChar, reinterpret_cast<const uint8_t*>(buffer), N_BYTES);
  setEstado(INFERIENDO); // continua incluso si se perdio BLE
  if (Serial) {
    Serial.print("Captura completa #"); Serial.print(idToma);
    Serial.print(" | muestras="); Serial.print(muestrasTomadas);
    Serial.print(" | huecos="); Serial.print(huecosRellenados);
    Serial.print(" | aceleracion reusada="); Serial.print(aceleracionReutilizada);
    Serial.print(" | giro reusado="); Serial.print(giroReutilizado);
    Serial.print(" | duracion_ms="); Serial.print(duracionRealMs);
    Serial.print(" | atraso_max_us="); Serial.println(retrasoMaxUs);
  }
}

void actualizarGrabacion() {
  // Captura una muestra por ranura temporal y SIEMPRE termina a los 30 s.
  // Si una operacion ocupó demasiada CPU, rellena las ranuras perdidas con
  // la ultima lectura y registra cuantas fueron (metrica de calidad).
  uint32_t ahora = micros();
  uint32_t transcurrido = (uint32_t)(ahora - inicioGrabacionUs);
  uint32_t ranura = transcurrido / PERIODO_US;
  if (ranura > N_MUESTRAS) ranura = N_MUESTRAS;

  while (muestrasTomadas < ranura && muestrasTomadas < N_MUESTRAS) {
    copiarMuestraAnterior(muestrasTomadas);
    ++muestrasTomadas;
    ++huecosRellenados;
  }

  if (ranura < N_MUESTRAS && muestrasTomadas == ranura) {
    uint32_t tardanza = transcurrido - ranura * PERIODO_US;
    if (tardanza > retrasoMaxUs) retrasoMaxUs = tardanza;
    if (!(IMU.accelerationAvailable() && IMU.readAcceleration(ax, ay, az))) {
      ++aceleracionReutilizada;
    }
    if (!(IMU.gyroscopeAvailable() && IMU.readGyroscope(gx, gy, gz))) {
      ++giroReutilizado;
    }
    buffer[muestrasTomadas][0] = convertirInt16(ax * ESC_ACEL);
    buffer[muestrasTomadas][1] = convertirInt16(ay * ESC_ACEL);
    buffer[muestrasTomadas][2] = convertirInt16(az * ESC_ACEL);
    buffer[muestrasTomadas][3] = convertirInt16(gx * ESC_GIRO);
    buffer[muestrasTomadas][4] = convertirInt16(gy * ESC_GIRO);
    buffer[muestrasTomadas][5] = convertirInt16(gz * ESC_GIRO);
    ++muestrasTomadas;
  }

  if (transcurrido >= (uint32_t)DURACION_S * 1000000UL) {
    while (muestrasTomadas < N_MUESTRAS) {
      copiarMuestraAnterior(muestrasTomadas);
      ++muestrasTomadas;
      ++huecosRellenados;
    }
    finalizarGrabacion();
    return;
  }

  // BLE no determina si la captura termina. Se atiende con prioridad baja.
  uint32_t ms = millis();
  uint32_t hastaProximaUs = PERIODO_US - ((uint32_t)(micros() - inicioGrabacionUs) % PERIODO_US);
  if (ms - ultimoPollGrabacionMs >= POLL_GRABACION_MS && hastaProximaUs > 3500UL) {
    BLE.poll();
    ultimoPollGrabacionMs = millis();
  }
}


// Una ventana por iteracion: BLE se atiende entre Invoke, pero no condiciona
// el avance. No se requiere USB ni un cliente BLE durante la inferencia.
void actualizarInferencia() {
  const uint16_t ini = RECORTE + (uint16_t)ventanaActual * PASO;
  int8_t* entrada = input_tensor->data.int8;
  for (uint16_t i = 0; i < VENTANA; ++i) {
    for (uint8_t c = 0; c < N_CANALES; ++c) {
      float val = buffer[ini + i][c] / (c < 3 ? ESC_ACEL : ESC_GIRO);
      float z = (val - HAR_MEDIA[c]) / HAR_DESV[c];
      int q = (int)lroundf(z / input_tensor->params.scale) + input_tensor->params.zero_point;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      entrada[i * N_CANALES + c] = (int8_t)q;
    }
  }
  uint32_t inicio = micros();
  if (interpreter->Invoke() != kTfLiteOk) fatal("ERROR: fallo de inferencia.");
  uint32_t dt = micros() - inicio;
  tiempoSumaUs += dt;
  if (dt > tiempoMaxUs) tiempoMaxUs = dt;
  const int8_t* salida = output_tensor->data.int8;
  uint8_t mejor = 0;
  uint8_t* fila = &resultados[BYTES_CABECERA + ventanaActual * BYTES_VENT];
  for (uint8_t k = 0; k < HAR_N_CLASES; ++k) {
    if (salida[k] > salida[mejor]) mejor = k;
    int p = salida[k] - output_tensor->params.zero_point;
    if (p < 0) p = 0;
    if (p > 255) p = 255;
    fila[k + 1] = (uint8_t)p;
  }
  fila[0] = mejor;
  if (Serial) {
    Serial.print("Ventana "); Serial.print(ventanaActual);
    Serial.print(": "); Serial.println(HAR_CLASES[mejor]);
  }
  if (++ventanaActual < N_VENT) return;

  resultados[0] = N_VENT;
  resultados[1] = HAR_N_CLASES;
  const uint32_t media = tiempoSumaUs / N_VENT;
  const uint32_t arena = (uint32_t)interpreter->arena_used_bytes();
  memcpy(resultados + 4, &media, 4);
  memcpy(resultados + 8, &tiempoMaxUs, 4);
  memcpy(resultados + 12, &arena, 4);
  publicarCRC(crcResultadosChar, resultados, sizeof(resultados));
  setEstado(LISTO);
  if (Serial) Serial.println("Captura e inferencia listas; resultados conservados en RAM");
}

bool enviarPaquete(const uint8_t* bytes, uint16_t total, uint16_t secuencia) {
  if (secuencia >= total || !BLE.connected() || !datosChar.subscribed()) return false;
  uint8_t paquete[2 + PAYLOAD];
  paquete[0] = (uint8_t)(secuencia & 0xFF);
  paquete[1] = (uint8_t)(secuencia >> 8);
  memcpy(paquete + 2, bytes + (uint32_t)secuencia * PAYLOAD, PAYLOAD);
  return datosChar.writeValue(paquete, sizeof(paquete)) == (int)sizeof(paquete);
}

void iniciarEnvio(const uint8_t* bytes, uint16_t total, uint16_t desde) {
  if (!BLE.connected() || !datosChar.subscribed() || desde >= total) return;
  datosEnvio = bytes;
  totalPaquetesEnvio = total;
  siguientePaquete = desde;
  fallosLocales = 0;
  inicioEnvioMs = proximoPaqueteMs = millis();
  setEstado(ENVIANDO);
  if (Serial) { Serial.print("Descarga desde paquete "); Serial.println(desde); }
}

void finalizarEnvio() {
  setEstado(LISTO);
  if (Serial) {
    Serial.print("Envio terminado, siguiente="); Serial.print(siguientePaquete);
    Serial.print(" fallosLocales="); Serial.print(fallosLocales);
    Serial.print(" duracion_ms="); Serial.println(millis() - inicioEnvioMs);
  }
}

void actualizarEnvio() {
  if (!BLE.connected() || !datosChar.subscribed()) {
    finalizarEnvio();  // conservar buffer para una conexion posterior
    return;
  }
  if ((int32_t)(millis() - proximoPaqueteMs) < 0) return;
  if (siguientePaquete >= totalPaquetesEnvio) {
    finalizarEnvio();
    return;
  }
  if (enviarPaquete(datosEnvio, totalPaquetesEnvio, siguientePaquete)) {
    ++siguientePaquete;
  } else if (++fallosLocales >= 20) {
    if (Serial) Serial.println("Error local de transmision; buffer conservado");
    finalizarEnvio();
    return;
  }
  proximoPaqueteMs = millis() + PAUSA_PAQUETE_MS;
}

void procesarComando() {
  uint8_t cmd[3] = {0, 0, 0};
  int longitud = controlChar.readValue(cmd, sizeof(cmd));
  if (longitud < 1) return;
  uint16_t seq = (uint16_t)cmd[1] | ((uint16_t)cmd[2] << 8);
  switch (cmd[0]) {
    case CMD_START:
      if (longitud == 1 && (estado == REPOSO || estado == LISTO)) iniciarToma();
      break;
    case CMD_SEND:
      if (longitud == 3 && estado == LISTO) iniciarEnvio(reinterpret_cast<const uint8_t*>(buffer), N_PAQUETES, seq);
      break;
    case CMD_RESEND:
      if (longitud == 3 && estado == LISTO) enviarPaquete(reinterpret_cast<const uint8_t*>(buffer), N_PAQUETES, seq);
      break;
    case CMD_RESULTADOS:
      if (longitud == 3 && estado == LISTO) iniciarEnvio(resultados, N_PAQ_RES, seq);
      break;
    case CMD_ABORT:
      if (longitud == 1 && estado == ENVIANDO) finalizarEnvio();
      break;
  }
}

void gestionarConexion() {
  bool conectado = BLE.connected();
  if (conectado && !conectadoAnterior) {
    if (Serial) Serial.println("Cliente BLE conectado");
  } else if (!conectado && conectadoAnterior) {
    if (Serial) Serial.println("BLE desconectado; captura/inferencia y buffers conservados");
    if (estado == ENVIANDO) finalizarEnvio();
    BLE.advertise();
  }
  conectadoAnterior = conectado;
}

void iniciarModelo() {
  net = tflite::GetModel(modelo_har_tflite);
  if (net->version() != TFLITE_SCHEMA_VERSION) fatal("ERROR: version de schema incompatible.");

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(net, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) fatal("ERROR: AllocateTensors. Revisar arena/ops/versiones.");
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  if (input_tensor->type != kTfLiteInt8 || output_tensor->type != kTfLiteInt8) fatal("ERROR: se esperaba entrada y salida INT8.");
  if (input_tensor->bytes != (size_t)VENTANA * N_CANALES) fatal("ERROR: tamano de entrada distinto de 200x6.");
  if (output_tensor->bytes != HAR_N_CLASES) fatal("ERROR: tamano de salida inesperado.");
  if (input_tensor->params.scale <= 0 || output_tensor->params.scale <= 0) fatal("ERROR: escalas de cuantizacion invalidas.");

  if (Serial) {
    Serial.print("Modelo listo. Arena usada (bytes): ");
    Serial.println((unsigned)interpreter->arena_used_bytes());
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  iniciarModelo();
  if (!IMU.begin()) fatal("No se detecto LSM9DS1");
  if (!BLE.begin()) fatal("Error al iniciar BLE");

  BLE.setLocalName("HAR-Nano");
  BLE.setDeviceName("HAR-Nano");
  BLE.setAdvertisedService(servicio);
  BLE.setConnectionInterval(6, 12); // solicitud de intervalo, no una garantia
  servicio.addCharacteristic(controlChar);
  servicio.addCharacteristic(estadoChar);
  servicio.addCharacteristic(datosChar);
  servicio.addCharacteristic(crcChar);
  servicio.addCharacteristic(metaChar);
  servicio.addCharacteristic(crcResultadosChar);
  BLE.addService(servicio);

  uint8_t crcVacio[4] = {0, 0, 0, 0};
  crcChar.writeValue(crcVacio, sizeof(crcVacio));
  crcResultadosChar.writeValue(crcVacio, sizeof(crcVacio));
  actualizarMeta();
  setEstado(REPOSO);
  BLE.advertise();
  if (Serial) Serial.println("HAR-Nano inferencia BLE v4 listo");
}

void loop() {
  if (estado == GRABANDO) {
    // Prioridad al reloj de muestreo; BLE.poll se llama espaciadamente dentro.
    actualizarGrabacion();
    return;
  }

  BLE.poll();
  gestionarConexion();
  if (controlChar.written()) procesarComando();
  switch (estado) {
    case CUENTA_ATRAS: actualizarCuentaAtras(); break;
    case ENVIANDO: actualizarEnvio(); break;
    case INFERIENDO: actualizarInferencia(); break;
    default: break;
  }
}
