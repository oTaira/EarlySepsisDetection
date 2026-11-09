# PhysioNet/CinC 2019 full public training data

40,336 original patient PSV files: training_setA (20,336), training_setB (20,000).
Each file contains 40 input columns plus SepsisLabel, with one row per hourly bin.
No imputation, filtering, normalization, or label modification has been applied.

Original dataset: https://physionet.org/content/challenge-2019/1.0.0/
Citation: Reyna et al. Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0. PhysioNet, 2019. DOI 10.13026/v64v-d857.
License: Creative Commons Attribution 4.0 International (CC BY 4.0).

Acquisition: the original bulk ZIP URLs were unavailable. Raw PSV files were obtained from the public mirror https://github.com/Siddharth-Prajapatii/Sepsis-Detection-Using-Machine-Learning (master branch, retrieved October 13, 2026).
Validation: both complete filename sets exactly match the official PhysioNet directory listings. All 203 cached records downloaded directly from PhysioNet (including the 200-record audit sample) match the mirror byte for byte. Every PSV file was checked for the expected header and 41 columns per row. This is a sampled content verification, not a full independent source-byte comparison.
manifest.json records each file's SHA-256 hash, byte length, and hourly row count.

To use in Google Drive, upload this ZIP and extract it in Colab or locally. Keep patients separate when creating train/validation/test splits. Labels are already shifted to begin six hours before defined sepsis onset.
