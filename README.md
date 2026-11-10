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

Utility on this internal holdout. Official full-test scores include a third hospital and are listed only for context. Published scores on hospitals A and B alone were about 0.40 to 0.43. Each alarm threshold was chosen on dev.

### Best of each architecture

The strongest model we trained for each architecture. Each one uses a measurement only after it has been observed.

| Architecture | Model | Dev | Test | Test A | Test B | Test AUPRC | Test AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| Trees | Window XGBoost | 0.392 | 0.397 | 0.410 | 0.374 | 0.106 | 0.832 |
| CNN | Regularized causal CNN | 0.374 | 0.385 | 0.394 | 0.370 | 0.099 | 0.820 |
| LSTM | Causal LSTM | 0.365 | 0.375 | 0.383 | 0.363 | 0.080 | 0.801 |
| Attention | Causal attention | 0.354 | 0.371 | 0.388 | 0.342 | 0.086 | 0.782 |
| Graph | Clinical graph attention | 0.308 | 0.300 | 0.307 | 0.289 | 0.072 | 0.739 |

The window XGBoost model summarizes recent vital signs and labs from past values only, then averages several boosted trees. The CNN scores each hour from a fixed span of earlier hours. The LSTM carries a summary of the stay forward one hour at a time. The attention model lets the current hour weigh earlier hours. The graph model treats each physiological variable as a node and connects variables that are read together.

### Competition winners

Retrains of the five 2019 challenge leaders on our patient split, in official rank order.

| Place | Model | Official full test | Train | Dev | Test | Test AUPRC | Test AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Signature | 0.360 | 0.507 | 0.395 | 0.415 | 0.120 | 0.839 |
| 2 | Sepsyd | 0.345 | 0.473 | 0.400 | 0.401 | 0.117 | 0.835 |
| 3 | Separatrix | 0.339 | 0.436 | 0.381 | 0.384 | 0.098 | 0.823 |
| 4 | TASP | 0.337 | 0.536 | 0.353 | 0.364 | 0.085 | 0.788 |
| 5 | CTL | 0.337 | 0.483 | 0.388 | 0.399 | 0.123 | 0.827 |

Signature learns the utility gain of raising an alarm from a short signature of recent vitals. Sepsyd is a boosted tree on a few recent hours of vitals, labs, and missingness flags. Separatrix summarizes recent windows and how long each measurement has been missing, using only values already seen, then averages five boosted trees. TASP uses one model for the start of a stay, another for the middle, and a recurrent model later on. CTL leaves missing values missing and shifts its training label earlier than the file label.

### Ensemble

A weighted average of the five architecture models above. Dev chose the weights and the threshold 0.475. Test was scored once.

| Piece | Weight |
|---|---:|
| Window XGBoost | 0.56 |
| Regularized causal CNN | 0.30 |
| Causal LSTM | 0.07 |
| Causal attention | 0.06 |
| Clinical graph attention | 0.01 |

| | Dev | Test | Test A | Test B | Test AUPRC | Test AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Ensemble | 0.407 | 0.418 | 0.429 | 0.400 | 0.114 | 0.840 |

Saved metrics are in `runs/separatrix_causal/metrics.json`, `runs/winners/metrics.json`, and `runs/ensemble_causal/metrics.json`.

## Run

```bash
python train_winners.py
python ensemble_causal.py
```

`train_winners.py` retrains the five challenge models. `ensemble_causal.py` scores the saved CNN, LSTM, attention model, clinical graph, and window XGBoost model at the frozen blend. Copies of those checkpoints are in `packaged/`.

Checkpoints live in `packaged/`:

| File | Model |
|---|---|
| `cnn.pt`, `lstm.pt`, `attention.pt`, `gat.pt` | Milestone neural models |
| `trees/causal_window/` | Window XGBoost model and its normalization |
| `ensemble_causal.json` | Ensemble weights and threshold |
| `trees/separatrix_0.json` … `separatrix_4.json` | Separatrix reproduction |
| `trees/sepsyd.json` | Sepsyd |
| `trees/ctl.json` | CTL |

Signature and TASP are produced by `train_winners.py` and are not stored as separate checkpoints.

## Data

PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0. Reyna et al., DOI 10.13026/v64v-d857. License: CC BY 4.0. [Dataset and license](https://physionet.org/content/challenge-2019/1.0.0/). The 40,336 public PSV files and generated `data/` directory are not tracked in Git; the data can be downloaded from PhysioNet or supplied as the full training ZIP. Do not commit the extracted patient files.

To reproduce the split used in the reported runs, obtain `PhysioNet_2019_Full_Training_Data.zip` and run:

```bash
python make_split.py /path/to/PhysioNet_2019_Full_Training_Data.zip
```

The script creates `data/train`, `data/dev`, `data/test`, and `data/manifests`. The supplied full-training ZIP was checked against the existing manifest: all 40,336 patient assignments match. The original public PSV files were mirrored from [this source](https://github.com/Siddharth-Prajapatii/Sepsis-Detection-Using-Machine-Learning) on October 13, 2026 because the original bulk ZIP URLs were unavailable then; filename sets matched PhysioNet's official directory listings and 203 sampled files matched byte-for-byte. This is sampled content verification, not a full independent byte comparison.
