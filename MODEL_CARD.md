# Model Card

## Model

- Name: PCB Defect Detector, RailCompute YOLO
- Architecture: YOLO detection model
- Base checkpoint: `yolo11n.pt`
- Final weights: `weights/best.pt`
- ONNX export: `weights/best.onnx`

## Intended Use

Detect and classify visible PCB manufacturing defects in inspection images. The model outputs bounding boxes, class names, and confidence scores.

This model is intended as a RailCompute training showcase and demo artifact. It is not claimed to be state of the art, fully production-certified, or universally accurate across all PCB datasets and factories.

RailCompute is a natural-language AI/ML training workflow currently presented as private beta/build stage at <https://railcompute.com/>. This model demonstrates the type of artifact that workflow can produce: trained weights, evaluation, plots, inference code, and documentation.

Supported classes:

1. `mouse_bite`
2. `spur`
3. `missing_hole`
4. `short`
5. `open_circuit`
6. `spurious_copper`

## Training

The model was trained on RailCompute with an NVIDIA `rtx_a4000` GPU.

Training configuration:

| Field | Value |
|---|---:|
| Base model | `yolo11n.pt` |
| Epochs | 80 |
| Image size | 640 |
| Batch size | 16 |
| Seed | 42 |

See `metadata/train_config.json` for the saved training configuration.

## Dataset

Training used the Kaggle-hosted **PCB Defect dataset** by Norbert Elter:

- <https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset>

The six-class defect taxonomy is associated with the public PCB defect dataset/paper:

- Weibo Huang and Peng Wei, *A PCB Dataset for Defects Detection and Classification*
- <https://arxiv.org/abs/1901.08204>

See `DATASET.md` for detailed attribution and redistribution notes.

## Evaluation

Final test metrics:

| Metric | Score |
|---|---:|
| Precision | 0.9774 |
| Recall | 0.9896 |
| mAP@50 | 0.9903 |
| mAP@50-95 | 0.5951 |

Per-class mAP@50-95:

| Class | Score |
|---|---:|
| `mouse_bite` | 0.5912 |
| `spur` | 0.5829 |
| `missing_hole` | 0.6279 |
| `short` | 0.6176 |
| `open_circuit` | 0.5721 |
| `spurious_copper` | 0.5792 |

## Limitations

- This is an automated training showcase, not a SOTA benchmark submission.
- The model is trained for one six-class PCB taxonomy.
- Very small defects can be detected correctly but receive lower strict localization scores.
- Confidence thresholds should be tuned for the inspection workflow. `0.70` is a stricter display threshold; lower values can improve recall.
- This model should support human inspection, not replace quality-control signoff without validation on your own production images.

## RailCompute Reproducibility Notes

This package includes:

- `metadata/metrics.json`
- `metadata/results.csv`
- `metadata/train_config.json`
- `assets/training_results.png`
- validation/test confusion matrices

These files document how the RailCompute training run performed and make the model easier to audit before use.
