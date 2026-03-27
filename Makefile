PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

dev-install:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m unittest discover -s tests -v

lint:
	ruff check src tests

format:
	ruff format src tests

check: lint test

toy-dataset:
	$(PYTHON) -m shape_aware_ocr.cli.generate_toy_dataset --out data/toy_research --clean

toy-smoke:
	$(PYTHON) -m shape_aware_ocr.cli.generate_toy_dataset --out data/toy_research --clean
	$(PYTHON) -m shape_aware_ocr.cli.build_alphabet --data-root data/toy_research/real --out data/toy_research/artifacts
	$(PYTHON) -m shape_aware_ocr.cli.train --data-root data/toy_research/real --out outputs/toy_train --alphabet data/toy_research/artifacts/alphabet.json --shape-manifest data/toy_research/manifests/shape_manifest.csv --sample-class-manifest data/toy_research/manifests/style_manifest.csv --synth-root data/toy_research/synth_square --synth-ratio 0.5 --best-by weighted_shape --epochs 1 --batch-size 4 --train-workers 0 --val-workers 0 --amp false
	$(PYTHON) -m shape_aware_ocr.cli.evaluate --data-root data/toy_research/real --checkpoint outputs/toy_train/checkpoints/best.pt --alphabet data/toy_research/artifacts/alphabet.json --shape-manifest data/toy_research/manifests/shape_manifest.csv --sample-class-manifest data/toy_research/manifests/style_manifest.csv --out outputs/toy_eval --workers 0
