#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>
#include <math.h>
#include <string.h>

// HAR-Nano BLE v3 | Nano 33 BLE Sense original (LSM9DS1)
// La adquisicion NO depende de que exista una conexion Bluetooth.
// ArduinoBLE controla el inicio y permite descargar mas tarde, incluso tras reconectar.
// IMPORTANTE: los datos viven en RAM; desconectar BLE no los borra,
// pero apagar o reiniciar la placa SI los borra.

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
constexpr uint8_t META_VERSION = 3;
constexpr uint8_t META_BYTES = 19;
constexpr uint32_t POLL_GRABACION_MS = 100;  // BLE es secundario durante la adquisicion

constexpr float ESC_ACEL = 1000.0f;   // g => mg
constexpr float ESC_GIRO = 10.0f;     // grados/s => decimas

static_assert(N_BYTES % PAYLOAD == 0, "El buffer debe dividirse en paquetes completos");
static_assert(N_PAQUETES < 65536UL, "Limite de secuencia excedido");

int16_t buffer[N_MUESTRAS][N_CANALES];

enum Estado : uint8_t {
  REPOSO = 0, CUENTA_ATRAS = 1, GRABANDO = 2, LISTO = 3, ENVIANDO = 4
};
Estado estado = REPOSO;

BLEService servicio("7a1c0001-5b6e-4c1d-9a3f-3f2b8c9d0001");
BLECharacteristic controlChar("7a1c0002-5b6e-4c1d-9a3f-3f2b8c9d0001", BLEWrite, 3);
BLEByteCharacteristic estadoChar("7a1c0003-5b6e-4c1d-9a3f-3f2b8c9d0001", BLERead | BLENotify);
BLECharacteristic datosChar("7a1c0004-5b6e-4c1d-9a3f-3f2b8c9d0001", BLENotify, 2 + PAYLOAD);
BLECharacteristic crcChar("7a1c0005-5b6e-4c1d-9a3f-3f2b8c9d0001", BLERead, 4);
BLECharacteristic metaChar("7a1c0006-5b6e-4c1d-9a3f-3f2b8c9d0001", BLERead, META_BYTES);

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

uint16_t siguientePaquete = 0;
uint16_t fallosLocales = 0;
uint32_t proximoPaqueteMs = 0;
uint32_t inicioEnvioMs = 0;
bool conectadoAnterior = false;

void setEstado(Estado nuevo) {
  estado = nuevo;
  estadoChar.writeValue((uint8_t)nuevo);
}

void falloFatal() {
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

uint32_t crc32Buffer() {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(buffer);
  uint32_t crc = 0xFFFFFFFFUL;
  for (uint32_t i = 0; i < N_BYTES; ++i) {
    crc ^= bytes[i];
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320UL : 0UL);
    }
  }
  return ~crc;
}

void actualizarCRC() {
  uint32_t crc = crc32Buffer();
  uint8_t dato[4] = {
    (uint8_t)crc, (uint8_t)(crc >> 8), (uint8_t)(crc >> 16), (uint8_t)(crc >> 24)
  };
  crcChar.writeValue(dato, sizeof(dato));
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
  actualizarCRC();
  setEstado(LISTO); // aun cuando se haya perdido la conexion BLE
  if (Serial) {
    Serial.print("Toma lista #"); Serial.print(idToma);
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

bool enviarPaquete(uint16_t secuencia) {
  if (secuencia >= N_PAQUETES || !BLE.connected() || !datosChar.subscribed()) return false;
  uint8_t paquete[2 + PAYLOAD];
  paquete[0] = (uint8_t)(secuencia & 0xFF);
  paquete[1] = (uint8_t)(secuencia >> 8);
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(buffer);
  memcpy(paquete + 2, bytes + (uint32_t)secuencia * PAYLOAD, PAYLOAD);
  return datosChar.writeValue(paquete, sizeof(paquete)) == (int)sizeof(paquete);
}

void iniciarEnvio(uint16_t desde) {
  if (!BLE.connected() || !datosChar.subscribed() || desde >= N_PAQUETES) return;
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
  if (siguientePaquete >= N_PAQUETES) {
    finalizarEnvio();
    return;
  }
  if (enviarPaquete(siguientePaquete)) {
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
      if (longitud == 3 && estado == LISTO) iniciarEnvio(seq);
      break;
    case CMD_RESEND:
      if (longitud == 3 && estado == LISTO) enviarPaquete(seq);
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
    if (Serial) Serial.println("BLE desconectado; grabacion/buffer conservado");
    if (estado == ENVIANDO) finalizarEnvio();
    BLE.advertise();
  }
  conectadoAnterior = conectado;
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  if (!IMU.begin()) { if (Serial) Serial.println("No se detecto LSM9DS1"); falloFatal(); }
  if (!BLE.begin()) { if (Serial) Serial.println("Error al iniciar BLE"); falloFatal(); }

  BLE.setLocalName("HAR-Nano");
  BLE.setDeviceName("HAR-Nano");
  BLE.setAdvertisedService(servicio);
  BLE.setConnectionInterval(6, 12); // solicitud de intervalo, no una garantia
  servicio.addCharacteristic(controlChar);
  servicio.addCharacteristic(estadoChar);
  servicio.addCharacteristic(datosChar);
  servicio.addCharacteristic(crcChar);
  servicio.addCharacteristic(metaChar);
  BLE.addService(servicio);

  uint8_t crcVacio[4] = {0, 0, 0, 0};
  crcChar.writeValue(crcVacio, sizeof(crcVacio));
  actualizarMeta();
  setEstado(REPOSO);
  BLE.advertise();
  if (Serial) Serial.println("HAR-Nano BLE v3 listo");
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
    default: break;
  }
}
