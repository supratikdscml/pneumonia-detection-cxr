# pneumonia-detection-cxr

Leakage-aware pneumonia detection from chest X-rays in PyTorch. Patient-grouped 5-fold splits, DenseNet121/ConvNeXt/ViT baselines, threshold selection, calibration, Grad-CAM and robustness checks. Research prototype, not for clinical use.

## Setup

Python 3.10+ with a GPU-enabled PyTorch install is expected. From the repository root:

```bash
python -m pip install -r requirements.txt
```

Place the Kaggle dataset so that `chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/` is discoverable (this repo currently finds it under `archive/chest_xray/chest_xray`). Optional W&B / MLflow tracking is off until `tracking.enabled` is set in `config.yaml`.

All hyperparameters live in `config.yaml`. Seeds are set in training and evaluation entry points.

## Reproduce by phase

The working notebook is `files/pneumoniadetect.ipynb`. Equivalent CLI modules:

| Phase | Command | Notes |
| --- | --- | --- |
| 1 EDA | Run the Phase 1 notebook cell | Writes `outputs/phase1/` |
| 2 Splits | Phase 2 notebook cell | Writes `outputs/splits.csv`; patient groups cannot overlap |
| 3 Transforms | Phase 3 notebook cell | No horizontal flip |
| 4 Baselines | `python src/pneum_det/baselines.py` | Fold 0 screening |
| 5 Train | `python src/pneum_det/train.py --model densenet121 --fold 0` | Early-stops on validation AUROC |
| 6 Evaluate | `python src/pneum_det/evaluate.py --model densenet121 --fold 0` | Validation only |
| 7 Explain | `python src/pneum_det/explain.py --model densenet121 --fold 0` | Grad-CAM, subtype errors, robustness |
| 8 Ensemble | `python src/pneum_det/ensemble.py ...` | Requires three full checkpoints |
| 9 Test | `python src/pneum_det/evaluate_final.py --selection outputs/phase9/final_selection.json` | One-shot; locked until a frozen validation manifest exists |
| 10 Extensions | `python src/pneum_det/phase10.py` | 3-class schema, lung-crop prototype, export; no test-set use |

## Results (current, fold 0 screening / smoke)

One-epoch Phase 4 screening AUROC (not a final claim): ConvNeXt-Tiny 0.995, ViT-B/16 0.994, DenseNet121 0.991. Full 5-fold training, ensemble, and official test evaluation are still gated. Treat AUROC above ~0.98 on a single fold as a leakage investigation trigger, not a published number.

## Limitations

- Single-center pediatric CXRs; appearance and disease mix differ from adult / multi-hospital data.
- Official validation split is only 16 images; this project pools train+val and uses patient-grouped folds instead.
- NORMAL "patient IDs" are filename-derived grouping proxies, not verified identities.
- Heuristic lung crops are not a trained anatomical segmenter.
- Research prototype only; do not use for diagnosis.
