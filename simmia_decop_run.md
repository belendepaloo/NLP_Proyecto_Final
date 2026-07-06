# Instrucciones para correr experimentos DE-COP y/o SimMIA en Google Colab

## Contexto

Estoy trabajando en un proyecto de Membership Inference Attack sobre LLMs. El cuello de botella principal es el límite de uso de Groq, por eso necesito correr los notebooks usando una cuenta/API key con mayor disponibilidad.

Los experimentos están implementados en Google Colab y guardan checkpoints en Google Drive para poder reanudar sin repetir consultas ya hechas.

## Estado actual de los experimentos

### SimMIA

* ✅ **llama-3.1-8b-instant:** experimento completo.
* 🔄 **llama-3.3-70b-versatile:** actualmente en ejecución. Si no está terminado, simplemente reanudar ejecutando la celda principal; el notebook continuará desde los checkpoints.

### DE-COP

* ✅ **llama-3.3-70b-versatile:** experimento completo.
* ⏳ **llama-3.1-8b-instant:** pendiente de ejecutar una vez finalizado el experimento anterior.

## Objetivo

Correr los experimentos usando Groq hasta completar todos los samples del dataset correspondiente.

Dataset:

```text
datasets/evaluation/booktection_medium_eval_20books_10chunks.csv
```

## Importante

* No borrar carpetas existentes.
* No borrar archivos `.json` de logs.
* No borrar archivos `.csv` de resultados.

Los experimentos ya tienen lógica de checkpoint. Si se corta el runtime o se alcanza un límite de API, simplemente volver a ejecutar la celda principal. El código continuará automáticamente desde donde quedó sin repetir consultas ya realizadas.

## Secrets necesarios en Colab

Cargar al menos una API key de Groq en Colab Secrets.

Nombre esperado:

```text
GROQ_API_KEY
```

Opcionalmente, si se usan varias keys:

```text
GROQ_API_KEY
GROQ_API_KEY2
GROQ_API_KEY3
...
GROQ_API_KEY11
```

También cargar el token de Hugging Face:

```text
HGF
```

## Estructura de outputs

### DE-COP

```text
api_logs/decop/<modelo>/
results/decop/<modelo>/
```

Archivos esperados:

```text
decop_query_level_results.csv
decop_sample_level_results.csv
```

### SimMIA

```text
api_logs/simmia/<modelo>/
results/simmia/<modelo>/
```

Archivos esperados:

```text
simmia_query_level_results.csv
simmia_token_level_results.csv
simmia_sample_level_results.csv
```

## Qué correr

1. Montar Google Drive.
2. Ejecutar las celdas de configuración.
3. Cargar el dataset.
4. Ejecutar la configuración del backend Groq.
5. Ejecutar la celda principal del experimento correspondiente.
6. Al finalizar, verificar que todos los samples fueron procesados.

## Parámetros DE-COP

```python
BACKEND = "groq"
MODEL = "llama-3.3-70b-versatile"   # o "llama-3.1-8b-instant"
LENGTH = "medium"

N_BOOKS_PER_CLASS = 10
N_PASSAGES_PER_BOOK = 10
N_PERMUTATIONS = 6

SLEEP_BETWEEN_CALLS = 2.1
SEED = 2319
```

## Parámetros SimMIA

```python
BACKEND = "groq"
MODEL = "llama-3.3-70b-versatile"
METHOD = "simmia"
LENGTH = "medium"

N_SAMPLES = 3
MAX_WORDS = 20
MAX_NEW_TOKENS = 3

SLEEP_BETWEEN_CALLS = 2.1
SEED = 2319
```

## Cómo verificar que terminó correctamente

Al finalizar debería aparecer algo similar a:

```text
COMPLETO.
Consultas guardadas: ...
Samples completos: ...
```

En DE-COP revisar:

```python
sample_df["label"].value_counts()
sample_df.head()
```

En SimMIA revisar:

```python
sample_df["label"].value_counts()
sample_df.head()
```

## Qué devolver

Los resultados quedan automáticamente guardados en el Drive compartido, por lo que en principio no hace falta enviarme archivos.

En caso de ser necesario, los archivos relevantes son:

### DE-COP

```text
results/decop/<modelo>/decop_query_level_results.csv
results/decop/<modelo>/decop_sample_level_results.csv
api_logs/decop/<modelo>/
```

### SimMIA

```text
results/simmia/<modelo>/simmia_query_level_results.csv
results/simmia/<modelo>/simmia_token_level_results.csv
results/simmia/<modelo>/simmia_sample_level_results.csv
api_logs/simmia/<modelo>/
```

## Notas

* Si aparece un error de rate limit diario de Groq, no reiniciar desde cero. El código rota automáticamente entre las API keys disponibles y, si todas se agotan, basta con volver a ejecutar la celda cuando haya disponibilidad.
* Si aparece `Runtime disconnected`, volver a ejecutar desde la configuración hasta la celda principal. Gracias a los checkpoints y a la cache por consulta, el experimento continuará desde donde quedó.
* Los notebooks guardan automáticamente checkpoints tanto a nivel de consultas individuales como de samples completos, por lo que las llamadas ya realizadas no deberían repetirse.

Link a la carpeta: https://drive.google.com/drive/folders/1giH-_WMHyGVb1sQ7xQJ4n3fBKjkmASr2?usp=sharing