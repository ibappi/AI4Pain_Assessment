# AI4Pain Assessment

This repository contains the implementation of our AI4Pain pain-level assessment framework using physiological signals.

This work is associated with our paper accepted/published in **AICOMS 2026**, currently **in press with Springer**.

## Project Overview

The goal of this project is to classify pain levels using multimodal physiological signals such as:

- EDA
- BVP
- RESP
- SpO₂

The framework includes preprocessing, modality-specific modeling, attention-based fusion, and final pain-level classification.

## Repository Structure

```text
preprocessing/   Data preprocessing scripts
src/models/          Model architecture files
src/training/        Training scripts
src/evaluation/      Evaluation and testing scripts
Figures/             Prediction files and figures
checkpoints/         Saved model weights
data/                Raw/processed/sample data
```

## Citation

If you use this code, please cite our AICOMS 2026 paper:

@inproceedings{bappi2026ai4pain,
  title={AI4Pain Assessment Using Multimodal Physiological Signals},
  author={Bappi, Ilias and others},
  booktitle={AICOMS 2026},
  publisher={Springer},
  year={2026},
  note={In press}
}
