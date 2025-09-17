# aggregate_intervue: Cloud/Local LLM Pipeline

## Быстрый старт (локально, Mac/CPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# или вручную: pip install requests srt pyannote.audio torch
# Запуск любого скрипта:
python3 localtrans/transcribe_local_whispercpp.py --help
```

## Запуск в облаке/GPU (через Docker Compose)

1. Поместите модели в ./models (или используйте HuggingFace repo)
2. Установите переменную HF_TOKEN (HuggingFace access token)
3. Запустите:

```bash
docker compose up --build
```

- vLLM будет доступен на http://localhost:8000/generate
- Ваши скрипты будут запускаться в контейнере с GPU и видеть /mnt/models

## Запуск из GitHub Container Registry (GHCR)

Сборка и публикация Docker-образа происходит автоматически через GitHub Actions при push в main/master.

Чтобы запустить контейнер в ЦОД или на сервере:

```bash
docker pull ghcr.io/<ВАШ_GITHUB_ЮЗЕР>/<ВАШ_РЕПОЗИТОРИЙ>:latest

docker run --gpus all \
  -e HF_TOKEN=your_hf_token \
  -e API_TOKEN=your_api_token \
  -v $(pwd)/models:/mnt/models \
  -p 8080:8080 \
  ghcr.io/<ВАШ_GITHUB_ЮЗЕР>/<ВАШ_РЕПОЗИТОРИЙ>:latest
```

- Образ всегда свежий, скачивается напрямую из GitHub Registry.
- Все переменные окружения и volume те же, что и для локального запуска.
- Можно использовать в автоматизации, CI/CD, n8n и т.д.

## Переменные окружения
- `USE_GPU=1` — форсировать GPU-режим
- `USE_VLLM=1` — использовать vLLM backend (по умолчанию на GPU)
- `USE_OLLAMA=1` — использовать Ollama backend (по умолчанию на CPU)
- `MODEL_PATH` — путь к модели (по умолчанию /mnt/models на сервере)
- `VERBOSE=1` — логировать окружение

## Примеры запуска

Локально (CPU):
```bash
python3 localtrans/assign_roles_with_ollama.py --src ./with_speakers --out ./local_roles --labels Менеджер,Клиент --mode local --model gpt-oss:20b --skip-exists
```

В контейнере (GPU/vLLM):
```bash
docker compose exec app bash
# внутри контейнера:
USE_GPU=1 USE_VLLM=1 python3 localtrans/assign_roles_with_ollama.py --src ./with_speakers --out ./local_roles --labels Менеджер,Клиент --mode local --model your-vllm-model --skip-exists
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
Образ основан на `ghcr.io/vllm-project/vllm-openai:v0.5.2`, поэтому vLLM и PyTorch уже предустановлены — остаётся только поднять FastAPI и при необходимости запустить сервис vLLM.

# В репозитории публикуется один тег в GHCR:
- `:latest` — GPU-образ (vLLM + pipeline).

### Запуск на RunPod (пример)
```bash
docker run --gpus all \
  -e API_TOKEN=supersecret \
  -e HF_TOKEN=your_hf_token \
  -e ROLES_MODE=openai \
  -e ROLES_MODEL=gpt-4o-mini \
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
