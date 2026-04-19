# Architecture

QUNet 2.0 uses a two-branch retinal imaging design:

1. RGB fundus encoder for visual lesion cues
2. OCT encoder for structural context
3. Metadata MLP for optional tabular inputs
4. Transformer fusion block for global context
5. Multi-scale feature pyramid for decoding
6. Segmentation head for lesion maps
7. Grading head for severity prediction
8. Uncertainty head for confidence calibration

## Extensions

- self-supervised pretraining
- test-time augmentation
- calibration analysis
- cross-dataset evaluation
- domain adaptation across retinal datasets
