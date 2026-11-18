# Preprocesamiento HAR — V2.1: Z-score por partición LOSO

`Preproces_HAR.py` transforma las grabaciones de la IMU en un dataset de
ventanas para el proyecto **Reconocimiento de actividad física con TinyML**.
Genera un archivo `.npz` con las ventanas y un reporte `.json` con su origen,
configuración y calidad.

El mismo script valida, recorta y segmenta las señales, separa las tomas mediante
LOSO y aplica **Z-score por canal**. Las medias y desviaciones se ajustan solo
con entrenamiento y se reutilizan para validación y prueba.

El NPZ contiene tanto `X` base en unidades físicas (g y grados/s) como los
conjuntos normalizados `fold_k_X_train`, `fold_k_X_val` y `fold_k_X_test`.
Para entrenar con Z-score se utilizan estos últimos. La versión 2.1 completa
la normalización que no estaba implementada en la salida 2.0.

## Archivos de esta carpeta

| Archivo | Función |
| --- | --- |
| `Preproces_HAR.py` | Validación, recorte, segmentación, particiones LOSO y Z-score. |
| `explorar_grabaciones_IMU.ipynb` | Exploración visual de las grabaciones originales. |
| `test_preproces_har.py` | Pruebas de integridad y comportamiento del preprocesamiento. |
| `dataset_har_preprocess_v2.npz` | Ventanas físicas, conjuntos normalizados por fold y sus estadísticas. |
| `reporte_preprocess_v2.json` | Reporte de calidad y trazabilidad de esa ejecución. |

Los nombres con sufijo `_v2` se eligen al ejecutar el script; no se asignan
automáticamente por el número de versión.

## Dependencias

Se necesita Python 3 con `numpy` y `pandas`. En el entorno de Python que vaya a
utilizarse, las dependencias se pueden instalar con:

```bash
python -m pip install numpy pandas
```

La verificación inicial de la V2 se realizó con Python 3.12.14, NumPy 2.3.5 y
pandas 2.2.3. Cada reporte guarda las versiones de NumPy y pandas utilizadas.

## Datos de entrada

La organización utilizada en el proyecto es:

```text
TinyML3/
├── Data_adquisition/
│   └── datos/
│       ├── Persona1/
│       │   ├── Caminar/
│       │   │   ├── P1_caminar_001.csv
│       │   │   └── P1_caminar_001.json  # Opcional
│       │   └── ...
│       ├── Persona2/
│       └── Persona3/
└── Preprocessing/
    ├── README.md
    ├── Preproces_HAR.py
    └── ...
```

También se admiten archivos directamente en la carpeta de entrada, con nombres
como `P1_caminar_001.csv`. En la estructura por carpetas, persona y actividad se
obtienen de las carpetas; en la estructura plana, del nombre del archivo.
Se normalizan mayúsculas, acentos, espacios y guiones para reconocer actividades.
El identificador de persona se conserva como fue obtenido de la ruta o nombre.

Cada CSV debe contener exactamente estas siete columnas, aunque su orden puede variar:

```text
t_s,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
```

| Columna | Contenido | Unidad |
| --- | --- | --- |
| `t_s` | Tiempo de cada muestra | segundos |
| `ax_g`, `ay_g`, `az_g` | Aceleración en X, Y y Z | g |
| `gx_dps`, `gy_dps`, `gz_dps` | Velocidad angular en X, Y y Z | grados/s |

La correspondencia entre etiquetas y actividades es fija:

| Etiqueta `y` | Actividad |
| ---: | --- |
| 0 | `reposo` |
| 1 | `caminar` |
| 2 | `correr` |
| 3 | `subir_escaleras` |
| 4 | `bajar_escaleras` |

## Ejecución

Ejecutar los siguientes comandos **desde la raíz de `TinyML3`**, en el entorno
de Python que tenga instaladas las dependencias:

```bash
python Preprocessing/Preproces_HAR.py \
  --entrada Data_adquisition/datos \
  --salida Preprocessing/dataset_har_preprocess_v2.npz \
  --reporte Preprocessing/reporte_preprocess_v2.json
```

Como las salidas V2 ya están generadas en el proyecto, para volver a generarlas
con esos mismos nombres se debe añadir `--sobrescribir`:

```bash
python Preprocessing/Preproces_HAR.py \
  --entrada Data_adquisition/datos \
  --salida Preprocessing/dataset_har_preprocess_v2.npz \
  --reporte Preprocessing/reporte_preprocess_v2.json \
  --sobrescribir
```

El reemplazo afecta a las rutas de salida indicadas. Los CSV de entrada se leen
sin modificarse. El reporte debe guardarse fuera de la carpeta de entrada de
los CSV para proteger los JSON de adquisición.

Para consultar las opciones disponibles:

```bash
python Preprocessing/Preproces_HAR.py --help
```

## Parámetros

| Opción | Valor predeterminado | Significado |
| --- | --- | --- |
| `--entrada` | `datos` | Carpeta donde se buscan los CSV recursivamente. |
| `--salida` | `dataset_har.npz` | Ruta del dataset; debe terminar en `.npz`. |
| `--reporte` | Mismo nombre base y carpeta del NPZ, con extensión `.json` | Ruta del reporte. |
| `--fs-hz` | `100` | Frecuencia nominal esperada del CSV; no remuestrea las señales. |
| `--recorte` | `2.0` | Segundos que se eliminan de cada extremo. |
| `--ventana` | `2.0` | Duración de cada ventana, en segundos. |
| `--solape` | `0.5` | Fracción compartida por ventanas consecutivas. |
| `--incluir-revisar` | Desactivado | Permite procesar tomas explícitamente marcadas para revisión. |
| `--sobrescribir` | Desactivado | Permite reemplazar los archivos de salida indicados. |
| `--sin-zscore` | Desactivado | Genera solo el dataset físico, sin LOSO ni normalización. |
| `--semilla` | `42` | Semilla de la selección de tomas de validación. |
| `--tomas-validacion` | `1` | Tomas completas para validación por actividad de cada persona que no está en prueba. |
| `--epsilon-zscore` | `1e-8` | Divisor mínimo para canales constantes o con desviación muy pequeña. |

La frecuencia debe ser finita y positiva; el recorte no puede ser negativo;
la ventana debe contener al menos una muestra; y el solapamiento debe cumplir
`0 <= solape < 1`. El recorte, la ventana y el desplazamiento resultante deben
corresponder a números enteros de muestras. Una configuración incompatible
produce un error, en lugar de redondearse silenciosamente. Además, la semilla
debe ser un entero no negativo, las tomas de validación un entero >= 1 y epsilon
un número finito positivo.

## Procedimiento

1. Identificar la persona y la actividad de cada archivo.
2. Leer el JSON de adquisición si existe.
3. Comprobar la integridad del CSV completo y la longitud necesaria.
4. Calcular los indicadores de calidad sobre la señal original.
5. Recortar ambos extremos y volver a calcular la calidad.
6. Crear ventanas dentro de cada toma, conservando su procedencia.
7. Crear los folds LOSO y separar train/val por tomas completas.
8. Ajustar el Z-score con train y aplicarlo a train, val y test de cada fold.
9. Guardar las ventanas físicas, los conjuntos normalizados y el reporte.

Los primeros y últimos dos segundos se recortan porque, según el protocolo de
adquisición del equipo, incluyen movimientos de preparación y finalización.
Esto orienta el dataset hacia actividades durante su ejecución estable. Su
evaluación deberá complementarse con secuencias continuas para estudiar las
transiciones entre actividades.

Con los parámetros predeterminados, una toma de 30 segundos tiene:

| Etapa | Resultado |
| --- | --- |
| Grabación a 100 Hz nominales | 3,000 muestras |
| Recorte de 200 muestras por extremo | 2,600 muestras útiles |
| Ventana de 2 segundos | 200 muestras |
| Solapamiento de 50 % | Desplazamiento de 100 muestras |
| Segmentación | 25 ventanas por toma |

El número de ventanas completas se calcula como:

```text
N = floor((M - W) / S) + 1
```

Donde `M` es el número de muestras tras el recorte, `W` el tamaño de ventana
y `S` el desplazamiento. Si no cabe una ventana completa, la toma se omite.
Las muestras finales que no completen otra ventana se contabilizan en el reporte.
Las ventanas nunca cruzan de una grabación a otra.

## Validación y motivos de omisión

Un CSV se omite si presenta alguno de estos problemas:

- Persona o actividad no reconocida en la ruta o nombre.
- Archivo ilegible, estructura CSV inválida o columnas faltantes, adicionales o duplicadas.
- Valores no numéricos, NaN, infinitos o canales fuera del rango representable en `float32`.
- Menos de dos filas para comprobar el tiempo.
- Tiempos repetidos o que retroceden.
- Intervalos de tiempo incompatibles con la frecuencia configurada, con tolerancia relativa del 5 % por intervalo.
- Longitud insuficiente para obtener una ventana después del recorte.
- Identificador de toma duplicado respecto de una toma ya aceptada.
- Marca explícita de revisión, cuando no se activó `--incluir-revisar`.

La validación se aplica al CSV completo, antes del recorte. No se ordenan tiempos,
interpolan señales ni eliminan filas para corregir archivos automáticamente.
Cada omisión se muestra en consola y se registra con su motivo en el reporte
cuando se generan salidas. Si ninguna toma es válida, el programa termina con
un error y no genera un dataset vacío ni un nuevo reporte.

Con la normalización activa se requieren al menos tres personas. Para cada
persona/actividad que participa en train/val debe haber más tomas que las
reservadas para validación. Con el valor predeterminado se necesitan al menos
dos: una de validación y al menos una de entrenamiento. Si no se cumple,
el programa termina antes de escribir salidas; no divide una toma entre conjuntos.
`--sin-zscore` permite generar solo ventanas físicas de un conjunto parcial.

La comprobación temporal evalúa el eje `t_s` del CSV. En el programa de adquisición
del proyecto ese eje se construye a partir de la frecuencia nominal, de modo que
su regularidad no demuestra que cada lectura del sensor sea nueva.

## Calidad antes y después del recorte

Cada toma tiene dos grupos de indicadores: `calidad_antes_recorte` y
`calidad_despues_recorte`. Ambos incluyen:

- Número de muestras.
- Cantidad y porcentaje de muestras con algún eje de aceleración cuya magnitud sea `>= 3.99 g`.
- Pares consecutivos iguales, su porcentaje y la racha máxima de muestras iguales.
- Repeticiones evaluadas sobre los seis canales, sobre el acelerómetro y sobre el giróscopo por separado.
- Media de la magnitud del vector de aceleración, en g.
- Mínimo, máximo, media y desviación estándar de cada canal.

La cercanía a ±4 g es un indicador compatible con el rango observado en estas
grabaciones; no constituye por sí sola una demostración de saturación.
Asimismo, lecturas iguales pueden aparecer por reposo, redondeo o reutilización
de valores. El CSV no permite distinguir con certeza esas causas.

Estos indicadores no provocan un descarte automático. Los valores de calidad
impresos en la tabla de consola corresponden a la señal **después del recorte**.

Por compatibilidad con el reporte anterior, se conservan los campos de cada toma
`muestras_saturadas`, `muestras_repetidas` y `norma_acel_media_g`. Estos campos
se refieren a la señal **original, antes del recorte**. El primero conserva su
nombre histórico, aunque mide cercanía al límite con el umbral indicado.

## JSON de adquisición opcionales

El JSON, cuando existe, debe compartir el nombre base y la carpeta del CSV.
La V2 registra su estado:

| Estado | Significado | Tratamiento del CSV |
| --- | --- | --- |
| `disponible` | El JSON se pudo leer como un objeto. | Se valida el CSV y se incorporan los metadatos disponibles. |
| `ausente` | No hay un JSON asociado. | Se valida y acepta el CSV si cumple las comprobaciones. |
| `invalido` | El JSON no se pudo interpretar. | Se registra un aviso y se valida el CSV por separado. |

Si el JSON está ausente o es inválido, `datos` se guarda como `null` en el reporte.
Esto significa **información no disponible**: no se registra artificialmente
cero huecos ni se considera que la grabación tenga mala calidad.

Cuando están presentes, persona, actividad, frecuencia y número de muestras del
JSON se contrastan con la ruta y el CSV/configuración. Las discrepancias generan
avisos; no se cambian las etiquetas utilizando el JSON.

### Qué significa la referencia a `revisar`

El programa de adquisición `Data_adquisition/Bleak.py` contempla guardar una
toma en una subcarpeta `revisar` y marcar `calidad_revisar: true` cuando el número
de huecos supera el límite de adquisición configurado. Es un comportamiento
condicional del programa, no una carpeta obligatoria del proyecto.

Por defecto, el preprocesamiento omite los CSV que estén dentro de una carpeta
llamada `revisar` o tengan un JSON válido con `calidad_revisar: true`.
`--incluir-revisar` permite incluirlos, manteniendo todas las comprobaciones
de integridad y dejando constancia de esa decisión. No crea carpetas ni mueve archivos.

En el conjunto utilizado para generar inicialmente la V2 no había carpetas
`revisar` ni JSON con esa marca activada. Por tanto, esta condición no excluyó
ninguna de las 44 tomas.

## Contenido del NPZ

El dataset se mantiene en forma `(N, W, 6)`, donde `N` es el número total de
ventanas y `W` las muestras por ventana. Con la configuración actual,
`X.shape == (1100, 200, 6)`. La dimensión adicional que pudiera requerir una
capa Conv2D se incorpora posteriormente al preparar la entrada del modelo.

| Campo | Forma | Contenido |
| --- | --- | --- |
| `X` | `(N, W, 6)` | Ventanas en unidades físicas, `float32`. |
| `y` | `(N,)` | Etiquetas enteras `int64`, índices de `clases`. |
| `clases` | `(5,)` | Nombres de las actividades, en el orden de las etiquetas. |
| `persona` | `(N,)` | Persona de cada ventana. |
| `toma` | `(N,)` | Identificador de la grabación de origen. |
| `archivo` | `(N,)` | Ruta del CSV relativa a la carpeta de entrada. |
| `inicio_muestra` | `(N,)` | Índice de la primera fila de datos de la ventana en el CSV. |
| `fin_muestra_exclusivo` | `(N,)` | Índice final de la ventana, no incluido. |
| `t_inicio_s` | `(N,)` | Tiempo de inicio tomado del eje temporal del CSV. |
| `canales` | `(6,)` | Orden de los canales en la última dimensión de `X`. |
| `unidades` | `(6,)` | Unidad de cada canal. |
| `fs_hz` | Escalar | Frecuencia nominal configurada. |
| `recorte_muestras_por_lado` | Escalar | Muestras eliminadas de cada extremo. |
| `ventana_muestras` | Escalar | Muestras por ventana. |
| `paso_muestras` | Escalar | Desplazamiento entre ventanas consecutivas. |
| `solape` | Escalar | Fracción de solapamiento configurada. |
| `normalizacion` | Escalar de texto | `ninguna`; describe exclusivamente `X` base. |
| `version_preprocesamiento` | Escalar de texto | `2.1`. |
| `manifiesto_json` | Escalar de texto | JSON con versión, parámetros, huellas SHA-256 y definición de folds/escaladores. |

Cuando el Z-score está activo, también se guardan estos campos. `k` es el número
de fold y `C` pertenece a `train`, `val` o `test`:

| Campo | Forma | Contenido |
| --- | --- | --- |
| `folds_loso` | `(personas,)` | Prefijos disponibles: `fold_1`, `fold_2`, `fold_3` en los datos actuales. |
| `normalizacion_loso` | Escalar de texto | `zscore_por_canal_ajustado_solo_train`. |
| `epsilon_zscore` | Escalar | Divisor mínimo configurado. |
| `fold_k_X_C` | `(N_C, W, 6)` | Ventanas normalizadas del conjunto, `float32`, sin unidades físicas. |
| `fold_k_y_C` | `(N_C,)` | Etiquetas correspondientes, `int64`. |
| `fold_k_indices_C` | `(N_C,)` | Índices que conectan el conjunto con `X`, `y`, `persona`, `toma` y `archivo` base. |
| `fold_k_media` | `(6,)` | Media de entrenamiento por canal, `float64`, en unidades físicas. |
| `fold_k_desviacion` | `(6,)` | Desviación estándar poblacional de entrenamiento por canal, `float64`. |
| `fold_k_escala` | `(6,)` | `max(desviacion, epsilon)`, divisor aplicado. |
| `fold_k_persona_test` | Escalar de texto | Persona reservada para prueba. |

Por ejemplo, los nombres concretos del primer fold son `fold_1_X_train`,
`fold_1_X_val`, `fold_1_X_test`, `fold_1_media`, etc. El campo `normalizacion`
del dataset base conserva el valor `ninguna`; los arrays de cada fold sí están
normalizados, como indica el campo separado `normalizacion_loso`.

Los índices de muestras empiezan en cero y no cuentan el encabezado del CSV.
Por ejemplo, `inicio_muestra=200` y `fin_muestra_exclusivo=400` corresponden a
las filas de datos `200:400`: exactamente 200 muestras.

Los campos históricos `X`, `y`, `clases`, `persona` y `toma` se conservan.
El NPZ no contiene arrays de objetos Python y puede abrirse sin habilitar pickle:

```python
import json
import numpy as np

with np.load("Preprocessing/dataset_har_preprocess_v2.npz", allow_pickle=False) as datos:
    print(datos["X"].shape)
    print(datos["clases"].tolist())
    print(datos["archivo"][0])
    print(datos["inicio_muestra"][0], datos["fin_muestra_exclusivo"][0])
    manifiesto = json.loads(datos["manifiesto_json"].item())
    X_train = datos["fold_1_X_train"]
    y_train = datos["fold_1_y_train"]
    X_val = datos["fold_1_X_val"]
    y_val = datos["fold_1_y_val"]
    X_test = datos["fold_1_X_test"]
    y_test = datos["fold_1_y_test"]
    media = datos["fold_1_media"]
    escala = datos["fold_1_escala"]
```

## Contenido del reporte JSON

El reporte contiene:

- Versión del preprocesamiento y fecha de ejecución en UTC.
- Rutas de entrada y salida, configuración y versiones de dependencias.
- Forma de `X`, total de ventanas y tomas aceptadas, y cantidad de CSV omitidos.
- Ventanas por actividad, por persona y por combinación de ambas.
- Cantidad de JSON disponibles, ausentes e inválidos entre las tomas aceptadas.
- Información de cada toma: archivo, huella SHA-256, recorte, ventanas,
  calidad antes/después, tiempo del CSV, metadatos de adquisición y avisos.
- Archivos omitidos y sus motivos.
- `normalizacion_loso`: personas y tomas de train/val/test de cada fold,
  conteos por clase, semilla, fórmula, medias, desviaciones y divisores aplicados.

Las huellas SHA-256 permiten comprobar si los CSV son los mismos que se utilizaron
para generar el dataset; no constituyen una evaluación de la calidad de la señal.

## Mejoras incorporadas en la V2

| Aspecto | Versión anterior | V2 |
| --- | --- | --- |
| Integridad del CSV | Comprobación básica de columnas y NaN. | Comprueba también números finitos, orden temporal, estructura del CSV y representación en `float32`. |
| Frecuencia del CSV | Un aviso permitía continuar con tiempos incompatibles. | Omite tiempos incompatibles con la frecuencia configurada y tolerancia del 5 % por intervalo. |
| Parámetros | Conversión directa a muestras. | Validación de valores finitos, rangos y cantidades enteras de muestras. |
| Longitud | Podían producirse cero ventanas para una toma. | Omite tomas insuficientes y evita generar un dataset vacío. |
| Calidad | Indicadores sobre la señal original. | Calidad original y recortada, estadísticas por canal y repeticiones/rachas por sensor. |
| JSON de adquisición | No se consultaban. | Lectura opcional, estado explícito y avisos de discrepancias. |
| Tomas para revisión | Sin tratamiento explícito. | Omisión por marca/carpeta de revisión y opción explícita para incluirlas. |
| Trazabilidad | Persona y toma de cada ventana. | Añade archivo, índices, tiempo inicial y huellas SHA-256. |
| Configuración en el NPZ | Sin descripción completa. | Canales, unidades, parámetros, versión y manifiesto incorporados. |
| Reporte | Conteos generales e indicadores básicos. | Añade cruces persona/actividad, metadatos, avisos y motivos de omisión. |
| Reemplazo de salidas | Reemplazo directo de archivos existentes. | Requiere `--sobrescribir`. |
| Z-score | Sin implementar. | Ajuste y aplicación dentro de este script, por canal y fold LOSO, con estadísticas de train exclusivamente. |
| Particiones | Se dejaban para otra etapa. | LOSO con validación interna por tomas completas y selección reproducible. |
| Verificación | Sin pruebas incluidas. | Se incorporan 19 pruebas automatizadas, incluidas comprobaciones de fuga de información. |

La V2 conserva el recorte de dos segundos, las ventanas de dos segundos y el
solapamiento del 50 % como valores predeterminados. La versión 2.1 aplica Z-score
por fold; no ajusta un único escalador usando a las tres personas juntas.
La calidad sigue calculándose sobre señales físicas, antes de normalizar.
No se añaden filtros, aumento de datos ni remuestreo.

## Verificación inicial con los datos del proyecto

| Comprobación | Resultado |
| --- | --- |
| Grabaciones encontradas y aceptadas | 44 |
| Grabaciones omitidas | 0 |
| JSON de adquisición disponibles | 26 |
| Grabaciones sin JSON | 18, aceptadas |
| Forma del dataset | `(1100, 200, 6)` |
| Comparación con el dataset anterior | `X`, `y`, `clases`, `persona` y `toma` idénticos, incluidos sus tipos de datos. |
| Reconstrucción de ventanas | Las 1,100 coinciden con los CSV según sus índices guardados. |
| Folds normalizados | 3: Persona1, Persona2 y Persona3 como prueba, respectivamente. |
| Ventanas train / val / test | Fold 1: 475 / 250 / 375; fold 2: 475 / 250 / 375; fold 3: 500 / 250 / 350. |
| Pruebas automatizadas | 19 aprobadas. |

Estos resultados describen los datos utilizados al incorporar Z-score en la
versión 2.1. Si se añaden, retiran o modifican grabaciones, deben revisarse las cifras
del nuevo reporte.

Para ejecutar las pruebas desde la raíz del proyecto:

```bash
python -m unittest discover -s Preprocessing -p test_preproces_har.py -v
```

Las pruebas cubren, entre otros casos, CSV sin JSON, JSON inválidos, datos no
finitos, problemas temporales, tomas cortas, parámetros inválidos, recorte cero,
marcas de revisión, conservación de etiquetas y ventanas que no cruzan tomas.
También verifican el Z-score predeterminado en el NPZ, canales constantes,
particiones reproducibles y que modificar val/test no cambie las estadísticas
ni las ventanas normalizadas de entrenamiento del fold examinado.

## Z-score y separación LOSO dentro del script

Por defecto, el programa realiza una partición por persona: reserva todos sus
datos para prueba y utiliza las demás personas para entrenamiento y validación.
De cada combinación persona/actividad restante selecciona una toma completa para
validación, con semilla 42; las otras tomas quedan para entrenamiento. La cantidad
se controla con `--tomas-validacion`. Las ventanas de una toma nunca se reparten
entre conjuntos. Las tomas elegidas y los índices de ventanas quedan guardados.

En cada fold se calculan seis medias y seis desviaciones sobre las ventanas de
**entrenamiento exclusivamente**, agrupando los ejes ventana y tiempo. Se utiliza
la desviación poblacional (`ddof=0`), con cálculo en `float64`, y se aplica:

```text
z = (x - media_train) / max(desviacion_train, epsilon)
```

El resultado se guarda como `float32`. Epsilon protege los canales constantes:
si la desviación es cero, se utiliza epsilon como divisor. Un canal constante en
train se transforma en cero allí; valores diferentes en val/test pueden producir
Z-scores grandes y no se recortan automáticamente.

Las muestras compartidas por ventanas superpuestas participan una vez por cada
ventana que las contiene. Esta ponderación reproduce la distribución de las
ventanas de entrenamiento y no utiliza datos de otros conjuntos. Val y test
se transforman con las mismas estadísticas de train, sin recalcularlas.

## Uso posterior en entrenamiento

El entrenamiento carga directamente los arrays normalizados del fold elegido,
como en el ejemplo anterior: no debe volver a ajustar el Z-score. Cada fold tiene
su propio modelo y sus propias estadísticas. Para la comparación de tres canales
frente a seis pueden reutilizarse los mismos índices; los tres primeros canales
corresponden al acelerómetro y su normalización por canal es independiente de los
canales del giróscopo.

Para Arduino deben conservarse la media y la escala del fold o del modelo final
que se despliegue, y aplicar el mismo orden de canales y unidades antes de
cuantizar la entrada. El ajuste de un modelo final con otra selección de train
requiere ajustar y guardar las estadísticas correspondientes a esa selección.

La separación implementada agrupa por **toma**. No se inventan identificadores de
sesiones que no estén documentadas. Si varias tomas pertenecen a una sesión que
debe mantenerse independiente, hace falta proporcionar esa agrupación y adaptar
la separación antes de interpretar los resultados como validación entre sesiones.
