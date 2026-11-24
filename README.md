# El Detective de Señales
**Clasificación automática de modulaciones con Deep Learning**

- **Alumna:** Aguilar Vargas Evelyn Adriana
- **Grupo:** 5402
- **Materia:** Fundamentos de Telecomunicaciones
- **Profesor:** Juan Alberto Flores González
- **Escuela:** Tecnológico de Estudios Superiores de Ecatepec

El código está en el cuaderno `Detective_de_Senales.ipynb` y se ejecuta en Google Colab (*Entorno de ejecución → Ejecutar todas*).

**Cuaderno en Google Colab:** https://colab.research.google.com/drive/1NsXvJL41eCc_yKTrR_xONEurIa5SdUxc

---

## Semana 1: Generación y preparación de datos

### Introducción
En esta primera semana se generaron las señales que después se van a usar para entrenar una red neuronal que reconozca el tipo de modulación de una señal. Se hicieron señales con las modulaciones ASK, FSK, PSK y QAM que vimos en el capítulo 3, se les agregó ruido para que se parecieran a una señal real y se separaron en datos de entrenamiento y de prueba.

### 1.1 Configuración del entorno
Primero abrí un cuaderno en Google Colab e importé las librerías: NumPy para los cálculos, Matplotlib para las gráficas, SciPy para el espectro y Scikit-learn para separar los datos.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
```

### 1.2 Parámetros de la señal
- Frecuencia de muestreo: **fs = 16 kHz**
- Frecuencia de la portadora: **fc = 2 kHz**
- Tasa de símbolos: **Rs = 1000 baudios**

Con estos valores cada símbolo tiene 16 muestras y cada señal tiene 16 símbolos, o sea 256 muestras. La frecuencia de muestreo cumple el teorema de Nyquist (fs ≥ 2·fmax): la frecuencia más alta es de unos 3.5 kHz, así que se necesitaban mínimo 7 kHz.

### 1.3 Generadores de señales
Hice una función para cada modulación. Se generan 16 bits al azar y para cada bit se hace un pedazo de la portadora:
- **ASK:** bit 1 = portadora, bit 0 = amplitud cero.
- **FSK:** bit 0 = 1500 Hz, bit 1 = 2500 Hz.
- **PSK (BPSK):** bit 1 = fase 0°, bit 0 = fase 180°.
- **16-QAM:** cambia amplitud y fase; I y Q valen -3, -1, 1 o 3 (16 combinaciones = 4 bits por símbolo).

```python
def psk(fase):
    bits = np.random.randint(0, 2, num_simbolos)
    s = []
    for k in range(num_simbolos):
        tk = t + k / Rs
        if bits[k] == 1:
            s.append(np.cos(2*np.pi*fc*tk + fase))
        else:
            s.append(np.cos(2*np.pi*fc*tk + fase + np.pi))
    return np.concatenate(s)
```

A cada señal le puse una fase inicial al azar, porque el receptor no sabe en qué fase llega la portadora, y las normalicé para que todas tuvieran la misma potencia. En las gráficas se comprobó que cada modulación cambia lo que debe, y en el espectro FSK tiene dos picos (1500 y 2500 Hz) mientras las demás están centradas en 2000 Hz.

### 1.4 Generación del conjunto de datos
Con tres ciclos `for` generé 300 señales por cada modulación y cada nivel de ruido:

| Concepto | Cantidad |
|---|---|
| Modulaciones | 4 |
| Niveles de ruido (SNR) | 7 |
| Señales por modulación | 2100 |
| Total de señales | 8400 |

Cada señal se guardó con su etiqueta (0 = ASK, 1 = FSK, 2 = PSK, 3 = QAM) y su nivel de ruido.

### 1.5 Ruido AWGN
Para simular un canal real se agregó ruido gaussiano blanco aditivo. La potencia del ruido se calcula con: **P_ruido = P_señal / 10^(SNR/10)**

```python
def agregar_ruido(x, snr):
    p_senal = np.mean(x**2)
    p_ruido = p_senal / (10**(snr/10))
    ruido = np.random.normal(0, np.sqrt(p_ruido), len(x))
    return x + ruido
```

Se usaron 7 niveles: -10, -5, 0, 5, 10, 15 y 20 dB. Con 20 dB el ruido casi no se nota y con -10 dB la señal casi no se ve.

### 1.6 Separación de los datos
Separé los datos en 80% entrenamiento (6720 señales) y 20% prueba (1680 señales) con `train_test_split` y `stratify`, para que hubiera la misma cantidad de cada modulación. Se guardaron en `dataset.npz`.

### Conclusión de la semana 1
Se generó el conjunto de datos del proyecto: 8400 señales de 4 modulaciones con varios niveles de ruido. Me sirvió para aplicar el teorema de Nyquist, las modulaciones digitales y la relación señal a ruido.

---

## Semana 2: Diseño y entrenamiento del modelo

### 2.1 ¿Por qué una red convolucional?
Las redes neuronales convolucionales (CNN) funcionan bien con señales. Una capa convolucional pasa filtros pequeños a lo largo de la señal, parecido a un filtro FIR, y cada filtro aprende a detectar un patrón (un cambio de fase, de frecuencia o de amplitud). Usé una **CNN 1D** que recibe directamente la forma de onda; la otra opción sería convertir la señal en espectrograma y usar una CNN 2D.

### 2.2 Diseño del modelo
Cambié la forma de los datos a (muestras, 256, 1) y pasé las etiquetas a formato one-hot (por ejemplo PSK = [0, 0, 1, 0]).

```python
modelo = keras.Sequential([
    layers.Input(shape=(N, 1)),
    layers.Conv1D(32, kernel_size=7, activation="relu"),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, kernel_size=5, activation="relu"),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(4, activation="softmax")
])
```

El modelo tiene 256,644 parámetros.

### 2.3 Compilación
Optimizador **Adam**, pérdida **categorical_crossentropy** (porque son varias clases) y métrica **accuracy**.

### 2.4 Entrenamiento
Lotes de 64, máximo 30 épocas, 10% de validación y **EarlyStopping** (si la validación no mejora en 6 épocas se detiene y se queda con los mejores pesos).

- El entrenamiento se detuvo en la **época 11**.
- La precisión de entrenamiento subió de **51% a 96%** y la pérdida bajó de 1.04 a 0.12.
- La precisión de validación llegó a **87%**.

Después de la época 5 la pérdida de validación ya no bajaba mientras la de entrenamiento sí, lo que indica un poco de sobreajuste; por eso sirvió el EarlyStopping.

### 2.5 Guardar el modelo
El modelo se guardó como `detective_senales.keras` para no entrenarlo otra vez.

### Conclusión de la semana 2
Se construyó y entrenó una red convolucional 1D que aprendió a distinguir las 4 modulaciones con 87% de precisión en validación.

---

## Semana 3: Evaluación y robustez

### 3.1 Matriz de confusión
El modelo se evaluó con las 1680 señales de prueba que nunca vio al entrenar. **Precisión en prueba: 86.3%.**

| Real \ Predicción | ASK | FSK | PSK | QAM |
|---|---|---|---|---|
| **ASK** | 377 | 12 | 22 | 9 |
| **FSK** | 2 | 413 | 5 | 0 |
| **PSK** | 20 | 20 | 343 | 37 |
| **QAM** | 11 | 12 | 80 | 317 |

**¿Confunde la IA ASK con PSK?** Muy poco: 22 ASK se tomaron como PSK y 20 PSK como ASK, de 420 de cada una (≈5%). Pasa porque las dos usan la misma portadora de 2 kHz, y con mucho ruido ya no se ven los huecos de amplitud de ASK ni los saltos de fase de PSK.

### 3.2 Prueba de estrés con ruido

| Nivel de ruido | SNR | Precisión |
|---|---|---|
| Alto | -10 y -5 dB | 57.0% |
| Medio | 0 y 5 dB | 93.1% |
| Bajo | 10 a 20 dB | 99.7% |

### 3.3 Gráfica de rendimiento
| SNR (dB) | -10 | -5 | 0 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|
| Precisión | 45.4% | 68.6% | 88.5% | 97.2% | 99.1% | 100% | 100% |

**¿En qué punto se vuelve ciego?** Desde 5 dB acierta casi siempre, pero abajo de 0 dB la precisión cae rápido, porque el ruido ya es más fuerte que la señal.

### 3.4 Análisis de errores

| Modulación | Acierto |
|---|---|
| ASK | 89.8% |
| FSK | 98.3% |
| PSK | 81.7% |
| 16-QAM | 75.5% |

La más difícil fue **16-QAM**, que se confundió con PSK en 80 casos. **Hipótesis:** los 16 puntos de la constelación de QAM están muy juntos, entonces con poco ruido un símbolo se confunde con otro; y si el ruido tapa los cambios de amplitud, solo quedan los cambios de fase y parece PSK. FSK fue la más fácil porque guarda la información en la frecuencia, que el ruido afecta menos.

### Conclusión de la semana 3
El modelo acierta el 86.3% de las señales de prueba y casi el 100% con poco ruido. Los resultados coinciden con la teoría: FSK es la más resistente al ruido y QAM la más sensible.
