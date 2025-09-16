# =========  =========
SRC            ?= ./source
WAV            ?= ./audio_wav
M4A            ?= ./audio_m4a
RTTM           ?= ./rttm
LOCAL_TRANS    ?= ./local_trans
OPENAI_TRANS   ?= ./openai_trans
WITH_SPK       ?= ./with_speakers
SUM_RU         ?= ./summaries_ru

#  / (: CLIENT="anticorcenter.bitrix24.ru 21.03.2025")
CLIENT         ?=

#   CLIENT,      
SRC_C          := $(if $(CLIENT),$(SRC)/$(CLIENT),$(SRC))
WAV_C          := $(if $(CLIENT),$(WAV)/$(CLIENT),$(WAV))
M4A_C          := $(if $(CLIENT),$(M4A)/$(CLIENT),$(M4A))
RTTM_C         := $(if $(CLIENT),$(RTTM)/$(CLIENT),$(RTTM))
LOCAL_TRANS_C  := $(if $(CLIENT),$(LOCAL_TRANS)/$(CLIENT),$(LOCAL_TRANS))
OPENAI_TRANS_C := $(if $(CLIENT),$(OPENAI_TRANS)/$(CLIENT),$(OPENAI_TRANS))
WITH_SPK_C     := $(if $(CLIENT),$(WITH_SPK)/$(CLIENT),$(WITH_SPK))
SUM_RU_C       := $(if $(CLIENT),$(SUM_RU)/$(CLIENT),$(SUM_RU))

WHISPER_BIN    ?= ../whisper.cpp/build/bin/whisper-cli
WHISPER_MODEL  ?= ../whisper.cpp/models/ggml-large-v3.bin
WHISPER_LANG   ?= ru

HF_TOKEN       ?= $(shell echo $$HF_TOKEN)
OPENAI_API_KEY ?= $(shell echo $$OPENAI_API_KEY)
OLLAMA_HOST    ?= $(shell echo $$OLLAMA_HOST)

JOBS           ?= 4
SR             ?= 16000
CH             ?= 1
BITRATE        ?= 96k
SIZE_MB        ?= 24

SUM_MODEL      ?= qwen2.5:14b-instruct
NUM_CTX        ?= 16384
NUM_KEEP       ?= 64
MAX_CHARS      ?= 60000

# --- load .env if present ---
ifneq (,$(wildcard .env))
include .env
# экспортируем только имена VAR=... из .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)\s*=.*/\1/p' .env)
endif


# ========= PHONY =========
.PHONY: all-local all-openai wav diarize transcribe-local speakerize summarize verify \
	m4a transcribe-openai combine clean env-check hf-check \
	resume-local resume-openai \
	client-local client-openai summarize-only

# =========   =========
all-local: wav diarize transcribe-local speakerize summarize verify

wav:
	@echo "==> Convert video -> WAV (PCM16 mono)"
	#  --overwrite:  .wav 
	python convert_media.py \
	  --src "$(SRC_C)" --dst "$(WAV_C)" \
	  --sr $(SR) --channels $(CH) \
	  --ascii-names --jobs $(JOBS)

diarize: hf-check
	@echo "==> Diarization -> RTTM"
	#     RTTM (  )
	python diarize_pyannote.py \
	  --src "$(WAV_C)" --dst "$(RTTM_C)" \
	  --hf-token "$(HF_TOKEN)" --use-mps --jobs 1

transcribe-local:
	@echo "==> Local whisper.cpp transcription"
	python transcribe_local_whispercpp.py \
	  --audio-root "$(WAV_C)" --out-root "$(LOCAL_TRANS_C)" \
	  --whisper-bin "$(WHISPER_BIN)" --model "$(WHISPER_MODEL)" \
	  --lang "$(WHISPER_LANG)" --threads $(JOBS) --jobs $(JOBS)

speakerize:
	$(PYTHON) ./merge_diar_any.py \
 	  --transcripts "./local_trans" \
 	  --rttm "./rttm" \
 	  --out "./with_speakers"

summarize:
	@echo "==> Summaries (RU, sliding window)"
	python summarize_spk_json.py \
	  --src "$(WITH_SPK_C)" --out "$(SUM_RU_C)" \
	  --model "$(SUM_MODEL)" \
	  --num-ctx $(NUM_CTX) --num-keep $(NUM_KEEP) --max-chars $(MAX_CHARS)

verify:
	@echo "==> Verify summary set"
	python check_and_fix_summaries.py \
	  --with-speakers "$(WITH_SPK_C)" \
	  --out-roots "$(SUM_RU_C)" \
	  --primary-out "$(SUM_RU_C)" \
	  --dry-run --verbose

# ========= OPENAI  =========
all-openai: m4a transcribe-openai combine diarize speakerize summarize verify

m4a:
	@echo "==> Convert video -> M4A w/ splitting"
	#  --overwrite:  .m4a/parts 
	python convert_media.py \
	  --src "$(SRC_C)" --dst "$(M4A_C)" \
	  --sr $(SR) --channels $(CH) \
	  --bitrate $(BITRATE) --size-limit-mb $(SIZE_MB) \
	  --ascii-names --jobs $(JOBS)

transcribe-openai: env-check
	@echo "==> OpenAI transcription (by parts, skip existing)"
	python transcribe_openai.py \
	  --audio-root "$(M4A_C)" \
	  --trans-root "$(OPENAI_TRANS_C)" \
	  --max-retries 5 --skip-exists

combine:
	@echo "==> Combine parts -> single segments.json"
	python combine_segments_from_parts.py \
	  --audio-root "$(M4A_C)" \
	  --trans-root "$(OPENAI_TRANS_C)" \
	  --skip-exists

# =========   (  ) =========
resume-local: wav transcribe-local speakerize summarize verify
	@echo "==> Resume (local) done."
# : diarize ;     RTTM      "diarize".

resume-openai: m4a transcribe-openai combine summarize verify
	@echo "==> Resume (openai) done."
# : diarize   ;   .

# =========    /  =========
#  :
#   make client-local CLIENT="anticorcenter.bitrix24.ru 21.03.2025"
#   make client-openai CLIENT="its-24.bitrix24.ru"
client-local:
	@echo "==> CLIENT=$(CLIENT)"
	@$(MAKE) all-local CLIENT="$(CLIENT)"

client-openai:
	@echo "==> CLIENT=$(CLIENT)"
	@$(MAKE) all-openai CLIENT="$(CLIENT)"

#   (   )
summarize-only:
	@echo "==> Summaries only for CLIENT=$(CLIENT)"
	@$(MAKE) summarize verify CLIENT="$(CLIENT)"

# =========  =========
env-check:
	@if [ -z "$(OPENAI_API_KEY)" ]; then \
	  echo "[ERR] OPENAI_API_KEY is not set"; exit 2; \
	fi

hf-check:
	@if [ -z "$(HF_TOKEN)" ]; then \
	  echo "[ERR] HF_TOKEN is not set"; exit 2; \
	fi

clean:
	@echo "==> Clean derived artifacts (careful!)"
	@rm -rf "$(WAV)" "$(M4A)" "$(RTTM)" "$(LOCAL_TRANS)" "$(OPENAI_TRANS)" "$(WITH_SPK)" "$(SUM_RU)"
