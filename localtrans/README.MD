# Пайплайн расшифровки встреч: конвертация → транскрибирование → спикеры → саммари (RU)

Этот репозиторий автоматизирует полный цикл обработки записей встреч (видео/аудио):

1. **Конвертация** исходников → WAV (локально) или M4A (для OpenAI, с авто-разбиением >24 MB)
2. **Диаризация** (определение говорящих) через `pyannote.audio` → RTTM
3. **Транскрибация**: локально (`whisper.cpp`) или через OpenAI
4. **Склейка** частей (для OpenAI)
5. **Присвоение ролей/спикеров** транскриптам → `.spk.srt/.spk.txt/.spk.json`
6. **Саммари на русском** с «окнами» (sliding window) → Markdown в `summaries_ru/`
7. **Проверка комплекта** и догенерация пропущенных файлов

Все типичные действия сведены в удобные цели `Makefile` (локальный/облачный сценарии, перезапуск по конкретному клиенту/подпапке, возобновление и т. д.).

---

## Требования

* **ffmpeg / ffprobe** (конвертация аудио и разбиение)
* **Python 3.10+** и зависимости (см. ниже)
* **whisper.cpp** (локальная транскрибация): бинарь `whisper-cli` и модель `ggml-large-v3.bin`
* **HuggingFace токен** для `pyannote.audio` (переменная окружения `HF_TOKEN`)
* **vLLM / OpenAI совместимый API** для назначения ролей и генеративных шагов (например, внешний vLLM на RunPod или OpenAI)
* **Ollama** (опционально, если хотите работать полностью локально без vLLM)

### Установка Python-зависимостей

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# если нет файла, минимально:
pip install pyannote.audio torch srt tqdm rich
```

> На Apple Silicon включайте `--use-mps` при диаризации для ускорения.

---

## Переменные окружения

```bash
export HF_TOKEN=hf_xxx               # для pyannote.audio
export OPENAI_API_KEY=sk-xxx         # если используете OpenAI-пайплайн
export OPENAI_BASE_URL=http://localhost:8000/v1  # URL vLLM/OpenAI API
export OLLAMA_HOST=http://127.0.0.1:11434  # если Ollama работает не по умолчанию
```

---

## Структура каталогов

```
project/
  source/                 # исходные видео/аудио (рекурсивно, любая вложенность)
  audio_wav/              # WAV (локальная ветка)
  audio_m4a/              # M4A + ".part_***" и ".manifest.json" (ветка OpenAI)
  rttm/                   # результаты диаризации (RTTM), зеркалят структуру source
  local_trans/            # транскрипты whisper.cpp
  openai_trans/           # транскрипты OpenAI (части и склейки)
  with_speakers/          # транскрипты с назначенными спикерами (.spk.*)
  summaries_ru/           # финальные саммари Markdown
  Makefile
  ...
```

> Файлы и папки при конвертации получают **ASCII-имена**: транслитерация + безопасные символы + короткий хэш — чтобы не было проблем с пробелами/кириллицей и коллизиями.

---

## Быстрый старт (через Makefile)

### Локальный полный цикл (WAV → диаризация → whisper.cpp → спикеры → саммари → проверка)

```bash
make all-local
```

### Облачный (OpenAI) полный цикл (M4A+разбиение → OpenAI → склейка → диаризация → спикеры → саммари → проверка)

```bash
make all-openai
```

### Возобновление (догоняем недостающее, не трогаем уже готовое)

```bash
make resume-local
make resume-openai
```

### Запуск по конкретному клиенту/подпапке

```bash
# ОБЯЗАТЕЛЬНО в кавычках, если есть пробелы!
make client-local  CLIENT="anticorcenter.bitrix24.ru 21.03.2025"
make client-openai CLIENT="its-24.bitrix24.ru"
```

### Только сгенерировать/перегенерировать саммари

```bash
make summarize-only CLIENT="promtreid"
```

> Все настраиваемые пути/параметры вынесены в переменные `Makefile` (см. раздел ниже).

---

## Ключевые параметры `Makefile`

* `SRC, WAV, M4A, RTTM, LOCAL_TRANS, OPENAI_TRANS, WITH_SPK, SUM_RU` — корни директорий
* `CLIENT` — поддерево (смещение всех путей внутрь подпапки)
* `WHISPER_BIN, WHISPER_MODEL, WHISPER_LANG` — локальная транскрибация
* `JOBS` — параллелизм
* `SR, CH, BITRATE, SIZE_MB` — конвертация аудио (частота, каналы, битрейт, лимит размера для разбиения)
* `SUM_MODEL, NUM_CTX, NUM_KEEP, MAX_CHARS` — саммари-модель и окно контекста

Откройте `Makefile`, чтобы подправить значения под свою машину.

---

## Скрипты и их флаги

### `convert_media.py`

| Флаг              | Тип   | Назначение                                        |
| ----------------- | ----- | ------------------------------------------------- |
| `--src`           | path  | Корень с исходными видео (рекурсивно).            |
| `--dst`           | path  | Куда писать аудио.                                |
| `--sr`            | int   | Частота дискретизации (Гц), по умолчанию 16000.   |
| `--channels`      | int   | Каналы (1=mono).                                  |
| `--bitrate`       | str   | Битрейт AAC (например `96k`) — актуально для M4A. |
| `--loudnorm`      | flag  | Нормализация громкости (EBU R128).                |
| `--audio-stream`  | int   | Индекс дорожки (0 = первая).                      |
| `--overwrite`     | flag  | Перезаписывать существующие файлы.                |
| `--jobs`          | int   | Кол-во параллельных заданий.                      |
| `--size-limit-mb` | float | Лимит для разбиения (например 24.0 для OpenAI).   |
| `--ascii-names`   | flag  | Транслитерация/ASCII + хэш в именах.              |

> Локальный сценарий: **WAV (PCM16)**.
> OpenAI-сценарий: **M4A** с авто-разбиением по размеру и генерацией `.manifest.json`.

---

### `diarize_pyannote.py`

| Флаг                                | Тип  | Назначение                                 |
| ----------------------------------- | ---- | ------------------------------------------ |
| `--src`                             | path | Папка с WAV 16k mono.                      |
| `--dst`                             | path | Куда писать RTTM.                          |
| `--hf-token`                        | str  | Токен HuggingFace (или `HF_TOKEN` из env). |
| `--num-speakers`                    | int  | Точное число спикеров (опционально).       |
| `--min-speakers` / `--max-speakers` | int  | Диапазон, если число неизвестно.           |
| `--use-mps`                         | flag | Использовать Apple GPU (MPS).              |
| `--jobs`                            | int  | Параллельные файлы (1–2 рекомендовано).    |

---

### `transcribe_local_whispercpp.py`

| Флаг            | Тип  | Назначение                                  |
| --------------- | ---- | ------------------------------------------- |
| `--audio-root`  | path | Корень WAV.                                 |
| `--out-root`    | path | Куда писать транскрипты.                    |
| `--whisper-bin` | path | Путь к `whisper-cli`.                       |
| `--model`       | path | Модель ggml (например `ggml-large-v3.bin`). |
| `--lang`        | str  | Язык (`ru`).                                |
| `--threads`     | int  | Потоки для декодера.                        |
| `--jobs`        | int  | Параллельные файлы.                         |

---

### `transcribe_openai.py`

| Флаг            | Тип  | Назначение                          |
| --------------- | ---- | ----------------------------------- |
| `--audio-root`  | path | Корень M4A/частей.                  |
| `--trans-root`  | path | Куда писать JSON.                   |
| `--max-retries` | int  | Повторы при сетевых ошибках.        |
| `--skip-exists` | flag | Пропускать существующие результаты. |

---

### `combine_segments_from_parts.py`

| Флаг            | Тип  | Назначение                           |
| --------------- | ---- | ------------------------------------ |
| `--audio-root`  | path | Где лежат `.manifest.json` и части.  |
| `--trans-root`  | path | Где искать частичные транскрипты.    |
| `--skip-exists` | flag | Не пересобирать, если итог уже есть. |

---

### `relayout_transcripts.py`  *(если нужно разложить «плоские» файлы по папкам)*

| Флаг           | Тип  | Назначение                                  |
| -------------- | ---- | ------------------------------------------- |
| `--audio-root` | path | Зеркало структуры для ориентира.            |
| `--trans-root` | path | Плоская папка с транскриптами.              |
| `--dry-run`    | flag | Показать, что будет сделано, без изменений. |

---

### `speakerize_transcripts.py`

| Флаг            | Тип  | Назначение                                 |
| --------------- | ---- | ------------------------------------------ |
| `--transcripts` | path | Папка с `.srt`/`.json` (Whisper).          |
| `--rttm`        | path | Папка с RTTM.                              |
| `--out`         | path | Куда писать `.spk.srt/.spk.txt/.spk.json`. |

---

### `summarize_spk_json.py`

| Флаг          | Тип  | Назначение                                               |
| ------------- | ---- | -------------------------------------------------------- |
| `--src`       | path | Папка `with_speakers` (вход `.spk.json`).                |
| `--out`       | path | Куда писать Markdown-саммари.                            |
| `--model`     | str  | Модель (напр. `qwen2.5:14b-instruct` или OpenAI-модель). |
| `--num-ctx`   | int  | Контекст модели (локально через Ollama).                 |
| `--num-keep`  | int  | «Заморозка» начала контекста между окнами.               |
| `--max-chars` | int  | Макс. символов исходника на одно окно.                   |

> **Алгоритм «мерджа смысла»:** длинные беседы режем на окна по символам, в каждом окне формируем сжатые тезисы, переносим «замороженный» пролог (brief + цели), затем склеиваем итог в один файл. Это снижает потерю контекста и укладывается в лимит модели.

---

### `check_and_fix_summaries.py`

| Флаг              | Тип     | Назначение                                                  |
| ----------------- | ------- | ----------------------------------------------------------- |
| `--with-speakers` | path    | Где лежат `.spk.json`.                                      |
| `--out-roots`     | path(s) | Папки с саммари для сверки.                                 |
| `--primary-out`   | path    | Куда править/создавать отсутствующие.                       |
| `--ignore-dirs`   | flag    | Игнорировать различия в каталогах (сверка по имени + хэшу). |
| `--dry-run`       | flag    | Только отчёт (без изменений).                               |
| `--verbose`       | flag    | Подробный лог.                                              |

---

## Примеры прямых вызовов (без Makefile)

```bash
# 1) Конвертация в WAV для локального пайплайна
python convert_media.py --src ./source --dst ./audio_wav --sr 16000 --channels 1 --ascii-names --jobs 6

# 2) Диаризация (RTTM)
python diarize_pyannote.py --src ./audio_wav --dst ./rttm --use-mps --jobs 1

# 3) Локальная транскрибация whisper.cpp
python transcribe_local_whispercpp.py \
  --audio-root ./audio_wav --out-root ./local_trans \
  --whisper-bin ../whisper.cpp/build/bin/whisper-cli \
  --model ../whisper.cpp/models/ggml-large-v3.bin \
  --lang ru --threads 6 --jobs 6

# 4) Присвоение спикеров
python speakerize_transcripts.py --transcripts ./local_trans --rttm ./rttm --out ./with_speakers

# 5) Саммари RU
python summarize_spk_json.py --src ./with_speakers --out ./summaries_ru \
  --model qwen2.5:14b-instruct --num-ctx 16384 --num-keep 64 --max-chars 60000

# 6) Валидация комплекта
python check_and_fix_summaries.py \
  --with-speakers ./with_speakers \
  --out-roots ./summaries_ru \
  --primary-out ./summaries_ru \
  --ignore-dirs --dry-run --verbose
```

---

## Частые проблемы и решения

* **`ffprobe failed: moov atom not found`**
  Контейнер битый. В `convert_media.py` есть «фоллбек» кодирования с агрессивным анализом; включается автоматически при неуспехе первичной попытки.

* **Пути/имена с пробелами/кириллицей**
  По умолчанию включена **ASCII-транслитерация** и добавление хэша — это обеспечивает стабильные и безопасные имена для всех инструментов и API.

* **Ограничение 24 MB на OpenAI**
  Ветка OpenAI генерирует `.manifest.json` и режет M4A на части `part_***.m4a`, чтобы каждая часть была < лимита.
  Потом `combine_segments_from_parts.py` склеивает частичные JSON в единый `*.segments.json`.

* **Контекст LLM (обрезка промпта)**
  Для длинных встреч используем окна (`--max-chars`, `--num-ctx`, `--num-keep`) — это предотвращает «truncating input prompt».

* **Повторные пуски/обрыв сети**
  Все стадии **идемпотентны**: скрипты пропускают уже существующие артефакты. Есть цели `resume-local` / `resume-openai`.

---

## Рекомендации по качеству

* **Саммари-модель**: локально `qwen2.5:14b-instruct` даёт хорошее соотношение качества и скорости.
  Для сложных договорённостей можно переключиться на более сильную модель (например, через OpenAI), но следите за лимитами контекста.

* **Диаризация**: если заранее известно число участников, задайте `--num-speakers` — это стабилизирует присвоение ролей.

* **WAV параметры**: `16 kHz / mono / PCM16` — оптимальны для whisper.cpp и последующих ACID-шагов.

---

## Лицензия и вклад

PR/патчи приветствуются: добавляйте новые цели Makefile (например, `diarize-only`, `transcribe-only`), улучшайте подсказки (prompts) для саммари и проверок.
