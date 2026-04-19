# QUNet 2.0

QUNet 2.0 is a research codebase for retinal lesion segmentation, disease grading, and uncertainty-aware inference. It combines a convolutional branch for local lesion detail with a transformer branch for wider anatomical context, then fuses both streams before task-specific heads.

![Architecture overview](assets/architecture_diagram.png)

## Problem setting

Retinal images contain small lesions, irregular contrast, and strong class imbalance. A single backbone often misses either fine lesion structure or global context. This repository addresses that with a multi-branch design, calibrated outputs, and separate segmentation and grading pathways.

## What the system contains

- convolutional encoder for local texture and boundary detail
- transformer encoder for long-range retinal context
- cross-branch fusion and multi-scale aggregation
- segmentation head for lesion localization
- classification head for disease grading
- uncertainty and calibration utilities for safer inference
- evaluation scripts for Dice, AUC, calibration, and confusion analysis
- FastAPI service and Streamlit demo for interactive use
- ONNX export path for deployment
- automated tests and CI workflow

## Visual summary

![Training curve](assets/training_curve.png)

![Calibration curve](assets/calibration_curve.png)

![Confusion matrix](assets/confusion_matrix.png)

![Sample prediction](assets/sample_prediction.png)

![Demo interface](assets/demo_ui.png)

## Main entry points

Run the project from the repository root:

```bash
python train.py
python evaluate.py
python predict.py
python demo.py
python api.py
python main.py
```

Package entry points are also available:

```bash
python -m qunet2.cli train --config configs/default.yaml
python -m qunet2.cli evaluate --config configs/default.yaml
python -m qunet2.cli demo
```

## Repository map

```text
src/qunet2/          main package code
assets/              figures used in the README
results/             metrics, summaries, exported outputs
configs/             dataset and experiment configuration
docs/                method, usage, deployment, and experiment notes
scripts/             data and utility scripts
tests/               automated checks
```

## Reproducibility

The repository is set up so the code can run without restricted medical datasets. Synthetic data generation is included for local testing, CI, and GitHub review. Real datasets such as IDRiD, APTOS, and OCT can be wired in through the configuration files.

## Commands

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
python demo.py
```

## Notes

The code is organized as a proper Python package, not a notebook dump. The root launch scripts are convenience wrappers around the package modules, which keeps the project easy to review on GitHub and easy to run locally.
