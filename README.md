# Early Sepsis Detection

Hourly early warning of sepsis on the PhysioNet/Computing in Cardiology Challenge 2019 training set. Each stay is kept whole. The score is the official normalized utility: a perfect alarm sequence is 1, never alarming is 0, and false alarms can push the score below 0.

## Split

40,336 stays from hospitals A and B, split 70/15/15 by patient. The split is stratified on hospital, whether the stay is ever septic, and stay length, so those distributions match across the three sets.

| Split | Patients | Septic stays | Hospital A | Median length | Positive hours |
|---|---:|---:|---:|---:|---:|
| Train | 28,235 | 7.27% | 50.42% | 38 h | 1.80% |
| Dev | 6,053 | 7.29% | 50.42% | 39 h | 1.81% |
| Test | 6,048 | 7.26% | 50.40% | 39 h | 1.78% |

Files live in `data/train`, `data/dev`, and `data/test`, with one pipe-separated file per stay. Patient lists are in `data/manifests`. The public dataset and generated splits are kept locally and ignored by Git. The alarm threshold for every model is chosen on dev and then frozen. Train and test use that same threshold. `SepsisLabel` is already positive from six hours before onset and is not shifted again, except inside the CTL training target described below. Scoring always uses the original labels.

## Results

Utility on this internal holdout. The official full-test scores are listed only for context: that hidden set includes a third hospital, which pulled every entry down. Published scores on hospitals A and B alone were about 0.40 to 0.43.

| Model | Official full test | Train | Dev | Test | Test AUPRC | Test AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Separatrix | 0.339 | 0.493 | 0.422 | 0.455 | 0.112 | 0.850 |
| Signature | 0.360 | 0.507 | 0.395 | 0.415 | 0.120 | 0.839 |
| Sepsyd | 0.345 | 0.473 | 0.400 | 0.401 | 0.117 | 0.835 |
| CTL | 0.337 | 0.483 | 0.388 | 0.399 | 0.123 | 0.827 |
| Causal CNN | | | 0.378 | 0.369 | 0.098 | 0.800 |
| TASP | 0.337 | 0.536 | 0.353 | 0.364 | 0.085 | 0.788 |

Separatrix is the strongest single challenge model on the held-out patients. The signature model is next. Both beat the first causal CNN. TASP fits the training set and does not carry that gain to dev or test.

### Neural models

Ranked by test utility. Each alarm threshold was chosen on dev and frozen. The GAT is the clinical graph (47 cross-variable edges) from the fixed-architecture comparison. That graph beat the 91-edge correlation graph on test.

| Rank | Model | Dev | Test | Test A | Test B | Test AUPRC | Test AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Regularized causal CNN | 0.374 | 0.385 | 0.394 | 0.370 | 0.099 | 0.820 |
| 2 | Causal LSTM | 0.365 | 0.375 | 0.383 | 0.363 | 0.080 | 0.801 |
| 3 | Causal attention | 0.354 | 0.371 | 0.388 | 0.342 | 0.086 | 0.782 |
| 4 | Clinical GAT | 0.308 | 0.300 | 0.307 | 0.289 | 0.072 | 0.739 |

The regularized CNN is a 3-layer causal network, 64 channels, kernel 9, SWIn-SiLU, dropout 0.4. The LSTM is 3 layers of hidden size 64 with a cosine learning-rate schedule. The attention model is one causal attention layer of width 32 on a kernel-9 stem. The GAT reads the 34 physiological variables at each hour and pools them.

### Ensemble

The best model is a weighted average of probabilities. The weights and the threshold 0.445 were chosen on dev. Test utility is 0.466, above Separatrix at 0.455.

| Piece | Weight |
|---|---:|
| Separatrix | 0.560 |
| Sepsyd | 0.247 |
| Causal LSTM | 0.146 |
| Regularized CNN | 0.023 |
| CTL | 0.023 |
| Causal attention | 0.0003 |

| | Dev | Test | Test A | Test B | Test AUPRC | Test AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Ensemble | 0.447 | 0.466 | 0.469 | 0.462 | 0.117 | 0.856 |

### What was trained

- **Separatrix** (Zabihi et al., 3rd). 407 features from short and long windows plus missingness run lengths. Five XGBoost models on disjoint patients, classes balanced by undersampling, combined by a geometric mean. A gain screen kept 204 features.
- **Signature** (Morrill et al., 1st). LightGBM regression on the utility gain of alarming versus staying quiet. Inputs are the current hour, shock index, bilirubin/creatinine, partial SOFA, measurement counts, vital min/max, and order-3 lead-lag signatures over a 7-hour window.
- **Sepsyd** (Du et al., 2nd). XGBoost, depth 4, on a 5-hour stack of the 40 variables and 34 missingness flags. Right-skewed labs are log-transformed, then forward-filled.
- **CTL** (Singh et al., 5th). XGBoost on a 5-hour window with missing values left missing and a mask on every vital and lab. The training target starts three hours earlier than the file label.
- **Causal CNN.** Three kernel-8 convolutions, 32 channels, 22 hours of history. The current hour is scored from the past only.
- **TASP** (Li et al., 4th). A tree for hours 1–9, a second tree for hours 10–49, and a GRU with learnable decay after hour 50. The ranked challenge entry was the middle tree alone. This run is the full three-part model.

Saved metrics are in `runs/cnn_baseline/metrics.json` and `runs/winners/metrics.json`.

## Run

```bash
python train_cnn_baseline.py
python train_winners.py
python score_saved.py
```

The first script builds the cached features and trains the CNN. The second trains the five challenge models. Both read `data/train`, `data/dev`, and `data/test`.

`score_saved.py` reloads the packaged CNN, LSTM, attention model, clinical GAT, and the ensemble, then scores dev and test at the frozen thresholds. The first run copies those checkpoints into `packaged/` and saves the XGBoost models. Later runs only score. A mismatch with the utilities in this file raises an error.

## Data

PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0. Reyna et al., DOI 10.13026/v64v-d857. License: CC BY 4.0. [Dataset and license](https://physionet.org/content/challenge-2019/1.0.0/). The 40,336 public PSV files and generated `data/` directory are not tracked in Git; the data can be downloaded from PhysioNet or supplied as the full training ZIP. Do not commit the extracted patient files.

To reproduce the split used in the reported runs, obtain `PhysioNet_2019_Full_Training_Data.zip` and run:

```bash
python make_split.py /path/to/PhysioNet_2019_Full_Training_Data.zip
```

The script creates `data/train`, `data/dev`, `data/test`, and `data/manifests`. The supplied full-training ZIP was checked against the existing manifest: all 40,336 patient assignments match. The original public PSV files were mirrored from [this source](https://github.com/Siddharth-Prajapatii/Sepsis-Detection-Using-Machine-Learning) on October 13, 2026 because the original bulk ZIP URLs were unavailable then; filename sets matched PhysioNet's official directory listings and 203 sampled files matched byte-for-byte. This is sampled content verification, not a full independent byte comparison.
