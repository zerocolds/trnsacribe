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
