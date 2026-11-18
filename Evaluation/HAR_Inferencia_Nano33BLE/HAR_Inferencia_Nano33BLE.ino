// HAR en Arduino Nano 33 BLE Sense (Rev1, IMU LSM9DS1) | CNN 1D INT8 | control y resultados por BLE
// Basado en el sketch TinyML_Seno_Nano33BLE: biblioteca Chirale_TensorFlowLite 2.0.0, AllOpsResolver.
//
// Archivos en la carpeta del sketch:
//   HAR_Inferencia_Nano33BLE.ino
//   modelo_har_data.h   <- generado por cuantizar_int8.py (modelo, normalización, cuantización, clases)
//
// Flujo (lo controla prueba_har.py por BLE):
//   START  -> cuenta atrás (LED parpadea) -> graba 30 s a 100 Hz en RAM (LED fijo)
//          -> infiere las 25 ventanas (recorte 2 s por lado, ventana 200, paso 100) -> LISTO
//   RESULTADOS -> envía clases, probabilidades y tiempos de inferencia
//   ENVIAR     -> (opcional) envía los datos crudos int16, igual que en la adquisición
#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "modelo_har_data.h"

// ---------- Adquisición (igual que el sketch de adquisición) ----------
constexpr uint16_t FS_HZ          = 100;
constexpr uint32_t PERIODO_US     = 1000000UL / FS_HZ;
constexpr uint16_t DURACION_S     = 30;
constexpr uint16_t N_MUESTRAS     = FS_HZ * DURACION_S;           // 3000
constexpr uint8_t  N_CANALES      = HAR_N_CANALES;                // 6
constexpr uint32_t CUENTA_ATRAS_MS = 3000;
constexpr float    ESC_ACEL       = 1000.0f;   // int16 = mg
constexpr float    ESC_GIRO       = 10.0f;     // int16 = 0.1 °/s

// ---------- Ventaneo (igual que preprocesar_har.py) ----------
constexpr uint16_t RECORTE = 200;              // 2 s por lado
constexpr uint16_t VENTANA = HAR_N_VENTANA;    // 200 muestras
constexpr uint16_t PASO    = 100;              // 50 % de solape
constexpr uint8_t  N_VENT  = (N_MUESTRAS - 2 * RECORTE - VENTANA) / PASO + 1;   // 25

// ---------- Paquetes BLE ----------
constexpr uint8_t  PAYLOAD        = 18;
constexpr uint32_t N_BYTES_CRUDOS = (uint32_t)N_MUESTRAS * N_CANALES * 2;       // 36000
constexpr uint16_t N_PAQ_CRUDOS   = N_BYTES_CRUDOS / PAYLOAD;                   // 2000
constexpr uint8_t  BYTES_CABECERA = 16;
constexpr uint8_t  BYTES_VENT     = 1 + HAR_N_CLASES;  // clase + probabilidades
constexpr uint16_t N_BYTES_RES    = BYTES_CABECERA + N_VENT * BYTES_VENT;
constexpr uint16_t N_PAQ_RES      = (N_BYTES_RES + PAYLOAD - 1) / PAYLOAD;
constexpr uint8_t  PAUSA_PAQUETE_MS = 5;

// ---------- Memoria ----------
int16_t buffer_imu[N_MUESTRAS][N_CANALES];            // 36 000 bytes
uint8_t resultados[N_PAQ_RES * PAYLOAD];              // con relleno de ceros

constexpr int kTensorArenaSize = 40 * 1024;           // ajustar con arena_used_bytes()
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* net = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// ---------- Estados y comandos ----------
enum Estado : uint8_t { REPOSO = 0, CUENTA_ATRAS = 1, GRABANDO = 2, LISTO = 3, ENVIANDO = 4, INFERIENDO = 5 };
Estado estado = REPOSO;
constexpr uint8_t CMD_START      = 0x01;
constexpr uint8_t CMD_SEND       = 0x02;   // datos crudos desde el paquete N
constexpr uint8_t CMD_RESULTADOS = 0x03;   // resultados desde el paquete N

// ---------- BLE (mismos UUID que el sketch de adquisición) ----------
BLEService servicio("7a1c0001-5b6e-4c1d-9a3f-3f2b8c9d0001");
BLECharacteristic controlChar("7a1c0002-5b6e-4c1d-9a3f-3f2b8c9d0001", BLEWrite, 3);
BLEByteCharacteristic estadoChar("7a1c0003-5b6e-4c1d-9a3f-3f2b8c9d0001", BLERead | BLENotify);
BLECharacteristic datosChar("7a1c0004-5b6e-4c1d-9a3f-3f2b8c9d0001", BLENotify, 2 + PAYLOAD);

void setEstado(Estado e) {
  estado = e;
  estadoChar.writeValue((uint8_t)e);
}

// Sin serial conectado, el error se ve como parpadeo rápido del LED.
void fatal(const char* mensaje) {
  Serial.println(mensaje);
  while (true) {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    delay(100);
  }
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
  if (input_tensor->params.scale <= 0 || output_tensor->params.scale <= 0) fatal("ERROR: escalas de cuantizacion invalidas.");

  Serial.print("Modelo listo. Arena usada (bytes): ");
  Serial.println((unsigned)interpreter->arena_used_bytes());
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  unsigned long inicio = millis();
  while (!Serial && millis() - inicio < 1500) { delay(10); }   // no bloquea si no hay USB

  iniciarModelo();
  if (!IMU.begin()) fatal("ERROR: IMU.");
  if (!BLE.begin()) fatal("ERROR: BLE.");

  BLE.setLocalName("HAR-Nano");
  BLE.setDeviceName("HAR-Nano");
  BLE.setConnectionInterval(6, 12);
  BLE.setAdvertisedService(servicio);
  servicio.addCharacteristic(controlChar);
  servicio.addCharacteristic(estadoChar);
  servicio.addCharacteristic(datosChar);
  BLE.addService(servicio);
  setEstado(REPOSO);
  BLE.advertise();
}

// ---------- Grabación ----------
void grabar() {
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
  for (uint16_t i = 0; i < N_MUESTRAS;) {
    if ((int32_t)(micros() - proximo) >= 0) {
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);
      buffer_imu[i][0] = (int16_t)lroundf(ax * ESC_ACEL);
      buffer_imu[i][1] = (int16_t)lroundf(ay * ESC_ACEL);
      buffer_imu[i][2] = (int16_t)lroundf(az * ESC_ACEL);
      buffer_imu[i][3] = (int16_t)lroundf(gx * ESC_GIRO);
      buffer_imu[i][4] = (int16_t)lroundf(gy * ESC_GIRO);
      buffer_imu[i][5] = (int16_t)lroundf(gz * ESC_GIRO);
      proximo += PERIODO_US;
      i++;
    } else {
      BLE.poll();
    }
  }
  digitalWrite(LED_BUILTIN, LOW);
}

// ---------- Inferencia sobre las ventanas grabadas ----------
void inferirVentanas() {
  setEstado(INFERIENDO);
  memset(resultados, 0, sizeof(resultados));

  uint32_t t_suma = 0, t_max = 0;
  const float s_in = input_tensor->params.scale;
  const int z_in = input_tensor->params.zero_point;
  const int z_out = output_tensor->params.zero_point;

  for (uint8_t v = 0; v < N_VENT; v++) {
    const uint16_t ini = RECORTE + (uint16_t)v * PASO;
    int8_t* entrada = input_tensor->data.int8;

    // int16 -> unidades físicas (g, °/s) -> z-score -> int8
    for (uint16_t i = 0; i < VENTANA; i++) {
      for (uint8_t c = 0; c < N_CANALES; c++) {
        float val = buffer_imu[ini + i][c] / (c < 3 ? ESC_ACEL : ESC_GIRO);
        float z = (val - HAR_MEDIA[c]) / HAR_DESV[c];
        int q = (int)lroundf(z / s_in) + z_in;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        entrada[i * N_CANALES + c] = (int8_t)q;
      }
    }

    uint32_t t_ini = micros();
    if (interpreter->Invoke() != kTfLiteOk) fatal("ERROR: fallo de inferencia.");
    uint32_t dt = micros() - t_ini;
    t_suma += dt;
    if (dt > t_max) t_max = dt;

    // argmax y probabilidades (uint8 = q - zero_point; probabilidad ≈ valor * escala de salida)
    const int8_t* salida = output_tensor->data.int8;
    uint8_t mejor = 0;
    uint8_t* fila = &resultados[BYTES_CABECERA + v * BYTES_VENT];
    for (uint8_t k = 0; k < HAR_N_CLASES; k++) {
      if (salida[k] > salida[mejor]) mejor = k;
      int p = salida[k] - z_out;
      if (p < 0) p = 0;
      if (p > 255) p = 255;
      fila[1 + k] = (uint8_t)p;
    }
    fila[0] = mejor;

    Serial.print("Ventana "); Serial.print(v); Serial.print(": ");
    Serial.println(HAR_CLASES[mejor]);
    BLE.poll();
  }

  resultados[0] = N_VENT;
  resultados[1] = HAR_N_CLASES;
  const uint32_t t_medio = t_suma / N_VENT;
  const uint32_t arena = (uint32_t)interpreter->arena_used_bytes();
  memcpy(&resultados[4], &t_medio, 4);    // little-endian en nRF52840
  memcpy(&resultados[8], &t_max, 4);
  memcpy(&resultados[12], &arena, 4);

  Serial.print("Inferencia media (us): "); Serial.print(t_medio);
  Serial.print(" | maxima (us): "); Serial.println(t_max);
}

// ---------- Envío por BLE (paquete = 2 bytes de secuencia + 18 de datos) ----------
void enviarPaquetes(const uint8_t* datos, uint16_t n_paquetes, uint16_t desde) {
  setEstado(ENVIANDO);
  uint8_t paquete[2 + PAYLOAD];
  for (uint16_t s = desde; s < n_paquetes && BLE.connected(); s++) {
    paquete[0] = s & 0xFF;
    paquete[1] = s >> 8;
    memcpy(paquete + 2, datos + (uint32_t)s * PAYLOAD, PAYLOAD);
    datosChar.writeValue(paquete, sizeof(paquete));
    BLE.poll();
    delay(PAUSA_PAQUETE_MS);
  }
  setEstado(LISTO);
}

void procesarComando() {
  uint8_t cmd[3] = {0, 0, 0};
  controlChar.readValue(cmd, sizeof(cmd));
  const uint16_t desde = cmd[1] | (cmd[2] << 8);

  if (cmd[0] == CMD_START && (estado == REPOSO || estado == LISTO)) {
    grabar();
    inferirVentanas();
    setEstado(LISTO);
  } else if (cmd[0] == CMD_SEND && estado == LISTO) {
    enviarPaquetes((const uint8_t*)buffer_imu, N_PAQ_CRUDOS, desde);
  } else if (cmd[0] == CMD_RESULTADOS && estado == LISTO) {
    enviarPaquetes(resultados, N_PAQ_RES, desde);
  }
}

void loop() {
  BLE.poll();
  if (controlChar.written()) procesarComando();
}
