ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))


VENV := $(ROOT).venv

PY := $(VENV)/bin/python

UV := $(VENV)/bin/uv

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

dataloader: $(STAMP)
	cd $(ROOT) && $(PY) dataloader.py

build-vocab: $(STAMP)
	cd $(ROOT) && $(PY) build_vocab.py

build-model: $(STAMP)
	cd $(ROOT) && $(PY) build_model.py