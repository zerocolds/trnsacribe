# aggregate_intervue: Cloud/Local LLM Pipeline

## Быстрый старт (локально, Mac/CPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# или вручную: pip install requests srt pyannote.audio torch python-multipart
# Запуск любого скрипта:
python3 localtrans/transcribe_local_whispercpp.py --help
```

## Запуск в облаке/GPU (через Docker Compose)

1. Разверните vLLM отдельно (контейнер `vllm/vllm-openai`, RunPod template и т. п.) и зафиксируйте URL его OpenAI-совместимого API (`http://<host>:8000/v1`).
2. Поместите модели whisper/pyannote в ./models (vLLM хранит свои веса отдельно).
3. Задайте `HF_TOKEN`, `OPENAI_BASE_URL`, `ROLES_MODEL` и запустите:

```bash
docker compose up --build
```

- Пайплайн обращается к внешнему vLLM по `OPENAI_BASE_URL`
- Контейнер использует GPU для whisper.cpp и pyannote, а `/mnt/models` пробрасывается внутрь

## Запуск из GitHub Container Registry (GHCR)

Сборка и публикация Docker-образа происходит автоматически через GitHub Actions при push в main/master.

Чтобы запустить контейнер в ЦОД или на сервере:

```bash
docker pull ghcr.io/<ВАШ_GITHUB_ЮЗЕР>/<ВАШ_РЕПОЗИТОРИЙ>:latest

docker run --gpus all \
  -e HF_TOKEN=your_hf_token \
  -e API_TOKEN=your_api_token \
  -e ROLES_MODE=openai \
  -e ROLES_MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
  -e OPENAI_BASE_URL=http://external-vllm:8000/v1 \
  -e OPENAI_API_KEY=dummy \
  -v $(pwd)/models:/mnt/models \
  -p 8080:8080 \
  ghcr.io/<ВАШ_GITHUB_ЮЗЕР>/<ВАШ_РЕПОЗИТОРИЙ>:latest
```

- Образ всегда свежий, скачивается напрямую из GitHub Registry.
- Все переменные окружения и volume те же, что и для локального запуска.
- Можно использовать в автоматизации, CI/CD, n8n и т.д.

## Переменные окружения
- `USE_GPU=1` — форсировать GPU-режим
- `ROLES_MODE=openai` — внешний vLLM/OpenAI API (значение по умолчанию)
- `ROLES_MODEL` — имя модели в vLLM (например `meta-llama/Meta-Llama-3-8B-Instruct`)
- `OPENAI_BASE_URL` — URL vLLM (`http://host:8000/v1`)
- `OPENAI_API_KEY` — токен (любое значение, если vLLM не проверяет)
- `MODEL_PATH` — путь к локальным моделям (по умолчанию /mnt/models)
- `VERBOSE=1` — логировать окружение

## Примеры запуска

Локально (через внешний vLLM):
```bash
OPENAI_BASE_URL=http://localhost:8000/v1 \
ROLES_MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
python3 localtrans/assign_roles_with_ollama.py \
  --src ./with_speakers \
  --out ./local_roles \
  --labels Менеджер,Клиент \
  --mode openai \
  --model "$ROLES_MODEL" \
  --skip-exists
```

## FastAPI pipeline server

### Локальный запуск
```bash
export API_TOKEN=supersecret
uvicorn localtrans.api_server:app --reload --host 0.0.0.0 --port 8080
```
Эндпоинты требуют Bearer-токен `API_TOKEN`. Для диаризации нужен `HF_TOKEN` (передавайте в форме или через переменные окружения).

### Сборка Docker-образа (GPU)
```bash
docker build -t aggregate-intervue:gpu -f Dockerfile.gpu .
```
Этот образ содержит только FastAPI-пайплайн, whisper.cpp и зависимости pyannote. Для LLM используйте внешний vLLM (например, стандартный шаблон RunPod или отдельный контейнер `vllm/vllm-openai`).

# В репозитории публикуется один тег в GHCR:
- `:latest` — GPU-образ (FastAPI pipeline + whisper.cpp, без встроенного vLLM).

### Запуск на RunPod (пример)
```bash
docker run --gpus all \
  -e API_TOKEN=supersecret \
  -e HF_TOKEN=your_hf_token \
  -e ROLES_MODE=openai \
  -e ROLES_MODEL=gpt-4o-mini \
  -e OPENAI_BASE_URL=http://external-vllm:8000/v1 \
  -e OPENAI_API_KEY=dummy \
  -e FORCE_WAV_CONVERT=1 \
  -p 8080:8080 \
  aggregate-intervue:gpu
```

После запуска API отвечает на `http://<host>:8080`.
Главный эндпоинт `/pipeline` возвращает:

- `segments` — список сегментов Whisper с таймкодами.
- `rttm` — результат диаризации в формате RTTM.
- `spk` — объединённые сегменты с назначенными спикерами.
- `roles` — JSON с ролями, уверенностями и статистикой по спикерам.

### Подключение к внешнему vLLM

1. Разверните vLLM (например, RunPod template или контейнер `vllm/vllm-openai`). Убедитесь, что он доступен по HTTP (по умолчанию `http://<host>:8000/v1`).
2. Передайте в наш сервис переменные окружения:
   - `ROLES_MODE=openai`
   - `ROLES_MODEL=<имя_модели>` (совпадает с моделью, которую вы загрузили в vLLM)
   - `OPENAI_BASE_URL=http://<host>:8000/v1`
   - `OPENAI_API_KEY=<любой токен>` (vLLM требует заголовок Authorization, но не проверяет значение)
3. При желании оберните оба контейнера `docker compose`-ом и используйте сетевой алиас `http://vllm:8000/v1`.
   Пример запуска vLLM в отдельном контейнере:
   ```bash
   docker run --gpus all -p 8000:8000 \
     -e HF_HOME=/models \
     -v $(pwd)/vllm-models:/models \
     vllm/vllm-openai:latest \
     python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Meta-Llama-3-8B-Instruct \
       --port 8000
   ```

## Как добавить свою модель
- Для vLLM: скачайте модель в формате HuggingFace (safetensors/.bin) в ./models или укажите ссылку через MODEL_PATH
- Для Ollama: используйте ollama pull <model>

## Тесты и проверка
- Быстрая проверка синтаксиса:
```bash
python3 -m compileall -q localtrans
```
- Smoke-тест: запустите любой скрипт с --help или на маленьком файле

## Контакты и поддержка
- Вопросы: issues или Telegram
