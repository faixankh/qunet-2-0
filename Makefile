IMAGES_DIR ?= images
MASKS_DIR ?= masks
LABELS_CSV ?= labels.csv

.PHONY: install demo results sample-results real-results audit-real test train evaluate clean

install:
	python -m pip install -e .

demo:
	python -m qunet2.cli demo --output results

results: sample-results

sample-results:
	python -m qunet2.cli generate-results --mode sample --output results --seed 42

real-results:
	@if [ -z "$(DATA_ROOT)" ]; then echo "DATA_ROOT is required. Example: make real-results DATA_ROOT=data/IDRiD"; exit 1; fi
	python -m qunet2.cli generate-results --mode real --data-root "$(DATA_ROOT)" --output results/real_dataset --images-dir "$(IMAGES_DIR)" --masks-dir "$(MASKS_DIR)" --labels-csv "$(LABELS_CSV)"

audit-real: real-results

train:
	python -m qunet2.cli train --config configs/default.yaml

evaluate:
	python -m qunet2.cli evaluate --config configs/default.yaml --output results/evaluation_smoke.json

test:
	pytest -q

clean:
	rm -rf .pytest_cache outputs results/real_dataset
