# WEAR 2025: sensor-specific PatchTST

Code and trained models from my entry to the **2nd WEAR Dataset Challenge** (HASCA workshop, UbiComp/ISWC 2025).

Paper: Pavankumar Chandankar and Robin Burchard. *Sensor-Specific Transformer (PatchTST) Ensembles with Test-Matched Augmentation.* HASCA 2025. [DOI 10.1145/3714394.3756196](https://doi.org/10.1145/3714394.3756196) · [arXiv 2510.21282](https://arxiv.org/abs/2510.21282)

## The challenge

Four Bangle.js smartwatches, one on each wrist and ankle, recording 3-axis acceleration at 50 Hz. The task is to classify 1-second windows (50 samples) into 18 workout exercises plus a null class, so 19 classes. The test windows come from four people who are not in the training data, and the organisers added synthetic noise to them. The score is macro-F1.

## The idea

- Train one small Transformer (PatchTST) per sensor location instead of one model for all four.
- During training, add the same kinds of noise that are in the test set: jitter, scaling, small rotations and dropped axes.
- Average the class probabilities of the per-sensor models.

In the paper, the final version (a clean and an augmented model per sensor, all averaged) reached **0.5172 macro-F1 on the private leaderboard**. The strongest baseline, Attend-and-Discriminate, reached 0.4718.

## What is in this repo

| Path | What it does |
| --- | --- |
| `utils/main.py`, `utils/data_loading.py` | Merge the challenge's `sbj_*.csv` files and map activity names to label numbers |
| `utils/filter_accelerometer.py` | Keep only the accelerometer columns |
| `data_segmentation/` | Cut each subject's recording into 50-sample windows (stride 25) per sensor, label = majority label in the window |
| `utils/sanity_check.py`, `utils/check_left_arm_segments.py`, `utils/clean_left_arm_sbj10.py` | Find and drop windows with NaN/Inf or zero variance (subject 10's left-arm data had these) |
| `baseline/patchtst/` | PatchTST model (patch embedding + Transformer encoder + linear head) and a per-sensor training script |
| `data_augmentation/augmentations.py` | Jitter, scaling, rotation and axis-dropout functions |
| `utils/feature_extraction_by_sensor.py`, `training/train_gb_per_sensor.py` | 16 simple statistics per window and a LightGBM model per sensor |
| `ensembles/ensemble_predict.py` | Average LightGBM and PatchTST probabilities per sensor and write a Kaggle submission CSV |
| `checkpoints_patchtst/`, `checkpoints_patchtst_aug/` | Trained PatchTST weights, one file per sensor |

## Trained models

| Folder | d_model | Layers | Patch length | Parameters per sensor |
| --- | --- | --- | --- | --- |
| `checkpoints_patchtst/` | 64 | 3 | 10 | 851,667 |
| `checkpoints_patchtst_aug/` | 128 | 6 | 10 | 3,574,547 |

`checkpoints_patchtst_aug/` holds the larger models trained with augmentation. A `.pth` file does not store the number of attention heads; the check script uses 4 for the small models (the training default) and 8 for the large ones (the value in the paper).

To check that all eight files load and run:

```bash
pip install -r requirements.txt
python -m utils.check_checkpoints
```

## Run the pipeline

The challenge data is not included. Download it from the challenge page and then, from the repository root:

```bash
python utils/main.py --data_dir /path/to/train --save_path processed_data.csv
python utils/filter_accelerometer.py --input_csv processed_data.csv --output_csv processed_data_acc.csv
python -m data_segmentation.main_by_sensor --processed_csv processed_data_acc.csv \
    --temp_dir raw_data_by_subject_acc --output_root raw_segments_by_sensor

python -m baseline.patchtst.train_patchtst --segments_root raw_segments_by_sensor \
    --output_dir my_checkpoints --epochs 50
```

For a submission, train the LightGBM models (`utils/feature_extraction_by_sensor.py`, then `training/train_gb_per_sensor.py`) and run:

```bash
python -m ensembles.ensemble_predict --test_csv /path/to/test.csv \
    --classical_models_dir lgbm_models --patchtst_models_dir checkpoints_patchtst \
    --output_csv submissions/submission.csv
```

For the augmented models add `--patchtst_models_dir checkpoints_patchtst_aug --d_model 128 --num_layers 6 --n_heads 8`.

## Differences from the paper

These are the scripts I used during the challenge in June 2025. The setup described in the paper went further, and that part is not in this repo:

- **Validation.** The paper uses 5-fold subject-wise cross-validation with AdamW, a cosine schedule, label smoothing, class weighting and temperature calibration. `train_patchtst.py` trains on all data with Adam and keeps the epoch with the best *training* F1, so its printed F1 is not a test score.
- **Augmentation.** `augmentations.py` has the functions, but `train_patchtst.py` does not call them. The rotation here is a random angle around the z-axis (±180°); the paper uses small rotations (±15°).
- **Model size.** The paper's encoder has d_model 128, 4 layers, 8 heads and patch length 5. The checkpoints here use patch length 10 and 3 or 6 layers.
- **Ensemble.** The paper averages a clean and an augmented PatchTST per sensor. `ensemble_predict.py` averages LightGBM and PatchTST instead, and the LightGBM `.pkl` files are not included.

For the method and the numbers, the paper is the reference.
