# Early Sepsis Detection

Hourly early warning of sepsis on the PhysioNet/Computing in Cardiology Challenge 2019 training set. Each stay is kept whole. The score is the official normalized utility: a perfect alarm sequence is 1, never alarming is 0, and false alarms can push the score below 0.

## Split

40,336 stays from hospitals A and B, split 70/15/15 by patient. The split is stratified on hospital, whether the stay is ever septic, and stay length, so those distributions match across the three sets.

| Split | Patients | Septic stays | Hospital A | Median length | Positive hours |
|---|---:|---:|---:|---:|---:|
| Train | 28,235 | 7.27% | 50.42% | 38 h | 1.80% |
| Dev | 6,053 | 7.29% | 50.42% | 39 h | 1.81% |
| Test | 6,048 | 7.26% | 50.40% | 39 h | 1.78% |

Files live in `data/train`, `data/dev`, and `data/test`, with one pipe-separated file per stay. Patient lists are in `data/manifests`. The alarm threshold for every model is chosen on dev and then frozen. Train and test use that same threshold. `SepsisLabel` is already positive from six hours before onset and is not shifted again, except inside the CTL training target described below. Scoring always uses the original labels.

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

Separatrix is the strongest model on the held-out patients. The signature model is next. Both beat the causal CNN. TASP fits the training set and does not carry that gain to dev or test.

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
```

The first script builds the cached features and trains the CNN. The second trains the five challenge models. Both read `data/train`, `data/dev`, and `data/test`.

## Data

PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0. Reyna et al., DOI 10.13026/v64v-d857. License: CC BY 4.0. See `data/README.md`.
