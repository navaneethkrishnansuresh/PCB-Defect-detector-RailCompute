# Dataset Disclosure

## Dataset Used

This model was trained using the Kaggle-hosted YOLO dataset:

- Dataset name: **PCB Defect dataset**
- Kaggle uploader: **Norbert Elter**
- Kaggle slug: `norbertelter/pcb-defect-dataset`
- URL: <https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset>
- Description shown by Kaggle: dataset for PCB detection with YOLO annotation.

## Original Dataset / Academic Credit

The six-defect taxonomy used by this training task is associated with the public PCB defect dataset introduced in:

- Weibo Huang and Peng Wei, **A PCB Dataset for Defects Detection and Classification**
- arXiv: <https://arxiv.org/abs/1901.08204>

Related public repository:

- <https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB>

## Classes

The trained model uses this class order:

| ID | Class |
|---:|---|
| 0 | `mouse_bite` |
| 1 | `spur` |
| 2 | `missing_hole` |
| 3 | `short` |
| 4 | `open_circuit` |
| 5 | `spurious_copper` |

## Data Format

The training data was used in YOLO object detection format:

```text
class_id x_center y_center width height
```

Coordinates are normalized to image width/height.

## Training Split Used

The automated training pipeline recorded this normalized split in the project run summary:

| Split | Images |
|---|---:|
| Train | 6,370 |
| Validation | 802 |
| Test | 829 |

The exact source images are not included in this package. The final model metrics in `metadata/metrics.json` are from the saved validation and test evaluations.

## Redistribution Note

This package includes model weights and derived evaluation artifacts, but it does **not** include the source dataset images. Before redistributing dataset samples, check the Kaggle dataset terms and the upstream dataset license/terms.

## Known Label-Taxonomy Warning

Other PCB datasets may use different class names for similar visual defects. For example, a dataset may label a hole-like defect as `damaged` while this model predicts `missing_hole`. Always compare taxonomy definitions before interpreting model accuracy across datasets.
