train:
	python -m qunet2.cli train --config configs/default.yaml

eval:
	python -m qunet2.cli evaluate --config configs/default.yaml

demo:
	python -m qunet2.cli demo

test:
	pytest -q
