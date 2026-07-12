ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))


VENV := $(ROOT).venv

PY := $(VENV)/bin/python

UV := $(VENV)/bin/uv

HF := $(VENV)/bin/hf

STAMP := $(VENV)/installed

$(PY):
	python3 -m venv $(VENV) 

$(UV): $(PY)
	$(PY) -m pip install uv

lock: $(UV)
	cd $(ROOT) && $(UV) lock


$(STAMP): $(ROOT)uv.lock $(UV)
	cd $(ROOT) && $(UV) sync
	touch $(STAMP) 

install: $(STAMP)

download-wav2vec2: $(STAMP)
	cd $(ROOT) && $(PY) download_wav2vec2.py

build-samples: $(STAMP)
	cd $(ROOT) && $(PY) build_samples.py

build-vocab: $(STAMP)
	cd $(ROOT) && $(PY) build_vocab.py

extract-user-audio-logits: $(STAMP)
	cd $(ROOT) && $(PY) extract_user_audio_logits.py

build-matrices: $(STAMP)
	cd $(ROOT) && $(PY) build_matrices.py

build-model: $(STAMP)
	cd $(ROOT) && $(PY) build_model.py

hf-login: $(STAMP)
	$(HF) auth login

hf-logout: $(STAMP)
	$(HF) auth logout

hf-upload: $(STAMP)
	cd $(ROOT) && $(PY) hf_upload.py