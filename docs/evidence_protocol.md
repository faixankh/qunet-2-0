# Evidence Protocol

This document defines what counts as acceptable evidence for QUNet 2.0. It is written to keep the repository useful during review without overstating results that have not been reproduced from the configured datasets.

## Experiment record

Every completed experiment should be stored under `results/runs/<date>_<short_name>/` with the following files:

```text
config.yaml
metrics.json
training_log.csv
calibration.csv
confusion_matrix.csv
qualitative_cases/
model_card.md
run_notes.md
```

## Required metrics

Segmentation:

- Dice score
- lesion-wise recall
- false positive area ratio
- boundary F1
- calibration error for lesion probability maps

Classification:

- accuracy
- macro F1
- AUC
- confusion matrix
- expected calibration error

Efficiency:

- parameter count
- inference time per image
- peak memory use
- ONNX export size

## Qualitative evidence

Each result folder should include at least twelve representative cases:

- strong success
- small-lesion recovery
- low-contrast lesion recovery
- false positive failure
- missed lesion failure
- uncertain prediction

Every case should include input image, ground truth mask, predicted mask, uncertainty map, and a short note explaining the failure or success pattern.

## Claim rule

README figures may summarize the repository workflow, but formal claims should only be made from a complete run folder. If a figure is synthetic, preview-mode, or generated from a small smoke test, label it clearly.