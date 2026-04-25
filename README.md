# QUNet 2.0

Retinal lesion segmentation, diabetic-retinopathy grading, and uncertainty-aware referral support with a CNN/Transformer fusion model and an explicit evidence pipeline.

![Architecture](assets/diagrams/architecture.svg)

## Evidence preview

| Artifact | What it shows | Regenerate |
|---|---|---|
| `assets/diagrams/architecture.svg` | Implemented model/dataflow structure | `make sample-results` |
| `assets/diagrams/evidence_pipeline.svg` | Evidence generation flow | `make sample-results` |
| `results/charts/sample_training_curve.svg` | Deterministic sample-data metric trace | `make sample-results` |
| `results/charts/runtime_profile.svg` | Runtime profile schema | `make sample-results` |
| `results/simulation/state_timeline.svg` | Stepwise uncertainty trace | `make sample-results` |

![Sample training curve](results/charts/sample_training_curve.svg)

![Runtime profile](results/charts/runtime_profile.svg)

## Quickstart

```bash
git clone https://github.com/faixankh/qunet-2-0.git
cd qunet-2-0
python -m venv .venv
python -m pip install -e .[dev]
make sample-results
make test
```

## Why this project exists

Retinal fundus analysis is difficult because clinically important lesions can be tiny, sparse, low contrast, and unevenly distributed across images. QUNet 2.0 keeps lesion segmentation, grading, and uncertainty estimation in the same inference path so weak or uncertain cases can be reviewed instead of being presented as clean predictions.

## Technical contribution

- multi-scale CNN encoding for lesion texture and boundary detail
- transformer bottleneck tokens for wider retinal context
- optional OCT/auxiliary branch support
- feature-pyramid fusion before segmentation and grading heads
- segmentation, classification, and uncertainty outputs
- deterministic sample-data evidence generation for smoke testing
- real retinal dataset auditing for IDRiD/APTOS/EyePACS-style folders
- tests for model shape behavior, metrics, losses, real dataset loading, and evidence generation

The repository does not make state-of-the-art or deployment claims. Real dataset performance should be reported only after training and evaluating on a documented licensed split.

## Repository structure

```text
src/qunet2/              package code, CLI, model, dataset, evidence modules
configs/                 sample and real-dataset configs
scripts/                 result generation and real-dataset audit scripts
docs/                    reproducibility, results, dataset, architecture notes
results/                 generated sample-data evidence pack
assets/diagrams/         generated architecture and evidence diagrams
tests/                   pytest suite
```

## Commands

```bash
make install
make sample-results
make demo
make evaluate
make test
```

Real retinal dataset audit:

```bash
make real-results DATA_ROOT=data/real_retina IMAGES_DIR=images MASKS_DIR=masks LABELS_CSV=labels.csv
```

Direct command:

```bash
python -m qunet2.cli generate-results --mode real --data-root data/real_retina --output results/real_dataset --images-dir images --masks-dir masks --labels-csv labels.csv
```

## Real eye-dataset results

Real fundus datasets are not committed to this repository. IDRiD, APTOS, EyePACS, and OCT datasets have license, size, and distribution constraints. The project provides a real-dataset results path instead of pretending that sample outputs are clinical evidence.

Expected local layout:

```text
data/real_retina/
  images/
  masks/
  labels.csv
```

The audit command writes:

```text
results/real_dataset/metrics.json
results/real_dataset/evidence_manifest.json
results/real_dataset/tables/dataset_audit.csv
results/real_dataset/charts/image_size_profile.svg
results/real_dataset/charts/mask_coverage_profile.svg
results/real_dataset/charts/label_distribution.svg
results/real_dataset/logs/dataset_audit_log.txt
```

These files are derived from actual image, mask, and label files. Dice, IoU, AUC, F1, calibration, and confusion matrices should be reported only after a trained checkpoint is evaluated on a fixed held-out split.

## Evidence policy

The committed files under `results/` are deterministic sample-data outputs. They check commands, schemas, and chart generation. They are not real clinical benchmark results.

- `results/` = sample-data evidence unless the manifest says otherwise
- `results/real_dataset/` = file-derived real dataset audit after local data is supplied
- model-performance claims require a checkpoint, dataset split, metrics, logs, and exact config

## Documentation

- [Reproducibility](docs/reproducibility.md)
- [Real retinal dataset results](docs/real_dataset_results.md)
- [Results and evidence policy](docs/results.md)
- [Architecture notes](docs/architecture.md)
- [Dataset notes](docs/datasets.md)

## License

MIT. Dataset licenses remain governed by the original dataset providers.
