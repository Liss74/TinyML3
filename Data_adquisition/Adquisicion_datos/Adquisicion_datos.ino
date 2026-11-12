#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>

// ---------- Parámetros de adquisición ----------
constexpr uint16_t FS_HZ         = 100;
constexpr uint32_t PERIODO_US    = 1000000UL / FS_HZ;
constexpr uint16_t DURACION_S    = 30;
constexpr uint16_t N_MUESTRAS    = FS_HZ * DURACION_S;      // 3000
constexpr uint8_t  N_CANALES     = 6;                       // ax ay az gx gy gz
constexpr uint32_t N_BYTES       = (uint32_t)N_MUESTRAS * N_CANALES * 2;  // 36000
constexpr uint8_t  PAYLOAD       = 18;                      // bytes de datos por paquete
constexpr uint16_t N_PAQUETES    = N_BYTES / PAYLOAD;       // 2000
constexpr uint32_t CUENTA_ATRAS_MS = 3000;                  // tiempo para colocarse
constexpr uint8_t  PAUSA_PAQUETE_MS = 25;                    // ajustar según pérdidas

// Escalado a int16: aceleración en mg (±4 g -> ±4000), giro en 0.1 °/s (±2000 °/s -> ±20000)
constexpr float ESC_ACEL = 1000.0f;
constexpr float ESC_GIRO = 10.0f;

// ---------- Buffer en RAM ----------
int16_t buffer[N_MUESTRAS][N_CANALES];

// ---------- Estados ----------
enum Estado : uint8_t { REPOSO = 0, CUENTA_ATRAS = 1, GRABANDO = 2, LISTO = 3, ENVIANDO = 4 };
Estado estado = REPOSO;

// ---------- BLE ----------
BLEService servicio("7a1c0001-5b6e-4c1d-9a3f-3f2b8c9d0001");
BLECharacteristic controlChar("7a1c0002-5b6e-4c1d-9a3f-3f2b8c9d0001", BLEWrite, 3);
BLEByteCharacteristic estadoChar("7a1c0003-5b6e-4c1d-9a3f-3f2b8c9d0001", BLERead | BLENotify);
BLECharacteristic datosChar("7a1c0004-5b6e-4c1d-9a3f-3f2b8c9d0001", BLENotify, 2 + PAYLOAD);

// Comandos (byte 0 de la característica de control)
constexpr uint8_t CMD_START = 0x01;   // iniciar toma
constexpr uint8_t CMD_SEND  = 0x02;   // enviar desde el paquete N (bytes 1-2, little-endian)

void setEstado(Estado e) {
  estado = e;
  estadoChar.writeValue((uint8_t)e);
}

void parpadeoError() {
  while (true) {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    delay(100);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);

  if (!IMU.begin()) parpadeoError();
  if (!BLE.begin()) parpadeoError();

  BLE.setLocalName("HAR-Nano");
  BLE.setDeviceName("HAR-Nano");
  BLE.setConnectionInterval(6, 12);          // 7.5-15 ms, mayor caudal
  BLE.setAdvertisedService(servicio);

  servicio.addCharacteristic(controlChar);
  servicio.addCharacteristic(estadoChar);
  servicio.addCharacteristic(datosChar);
  BLE.addService(servicio);

  setEstado(REPOSO);
  BLE.advertise();
}

// ---------- Toma de datos ----------
void tomaDatos() {
  // Cuenta atrás: LED parpadeando para que la persona se coloque
  setEstado(CUENTA_ATRAS);
  uint32_t t0 = millis();
  while (millis() - t0 < CUENTA_ATRAS_MS) {
    digitalWrite(LED_BUILTIN, ((millis() - t0) / 250) % 2);
    BLE.poll();
  }

  setEstado(GRABANDO);
  digitalWrite(LED_BUILTIN, HIGH);

  float ax, ay, az, gx, gy, gz;
  uint32_t proximo = micros() + PERIODO_US;
  int32_t atrasoMax = 0;

  for (uint16_t i = 0; i < N_MUESTRAS; ) {
    int32_t atraso = (int32_t)(micros() - proximo);
    if (atraso >= 0) {
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);

      buffer[i][0] = (int16_t)lroundf(ax * ESC_ACEL);
      buffer[i][1] = (int16_t)lroundf(ay * ESC_ACEL);
      buffer[i][2] = (int16_t)lroundf(az * ESC_ACEL);
      buffer[i][3] = (int16_t)lroundf(gx * ESC_GIRO);
      buffer[i][4] = (int16_t)lroundf(gy * ESC_GIRO);
      buffer[i][5] = (int16_t)lroundf(gz * ESC_GIRO);

      if (atraso > atrasoMax) atrasoMax = atraso;
      proximo += PERIODO_US;             // planificación absoluta: sin deriva
      i++;
    } else {
      BLE.poll();                        // aprovecha el tiempo libre
    }
  }

  digitalWrite(LED_BUILTIN, LOW);
  Serial.print("Retraso maximo (us): ");
  Serial.println(atrasoMax);
  setEstado(LISTO);
}

// ---------- Envío por BLE ----------

void enviar(uint16_t desde) {
  if (!BLE.connected() || !datosChar.subscribed()) {
    Serial.println("BLE no conectado o sin suscripcion");
    return;
  }

  setEstado(ENVIANDO);

  const uint8_t* bytes = (const uint8_t*)buffer;
  uint8_t paquete[2 + PAYLOAD];

  uint32_t inicio = millis();
  uint16_t fallos = 0;
  uint16_t enviados = 0;

  for (uint16_t s = desde;
       s < N_PAQUETES && BLE.connected();
       s++) {

    paquete[0] = s & 0xFF;
    paquete[1] = (s >> 8) & 0xFF;

    memcpy(
      paquete + 2,
      bytes + (uint32_t)s * PAYLOAD,
      PAYLOAD
    );

    int resultado = datosChar.writeValue(
      paquete, sizeof(paquete)
    );

    if (resultado != sizeof(paquete)) {
      fallos++;
    }

    enviados++;

    BLE.poll();
    delay(PAUSA_PAQUETE_MS);

    // Diagnóstico por USB cada 100 paquetes
    if (enviados % 100 == 0 && Serial) {
      Serial.print("Paquetes procesados: ");
      Serial.print(enviados);
      Serial.print(" | Tiempo ms: ");
      Serial.println(millis() - inicio);
    }
  }

  if (Serial) {
    Serial.print("Envio finalizado. Paquetes: ");
    Serial.print(enviados);
    Serial.print(" | Fallos locales: ");
    Serial.print(fallos);
    Serial.print(" | Duracion ms: ");
    Serial.println(millis() - inicio);
  }

  setEstado(LISTO);
}


void procesarComando() {
  uint8_t cmd[3] = {0, 0, 0};
  controlChar.readValue(cmd, sizeof(cmd));

  if (cmd[0] == CMD_START && (estado == REPOSO || estado == LISTO)) {
    tomaDatos();
  } else if (cmd[0] == CMD_SEND && estado == LISTO) {
    uint16_t desde = cmd[1] | (cmd[2] << 8);
    enviar(desde);
  }
}

void loop() {
  BLE.poll();
  if (controlChar.written()) procesarComando();
}