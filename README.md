# eeg-research

EEG decoding experiments on the Kaggle [Grasp-and-Lift EEG Detection](https://www.kaggle.com/c/grasp-and-lift-eeg-detection) dataset.

- `src/eegdemo/` — baseline (band-power + logreg), EEGNet, causal conformer
- `src/eeg99/` — WIP rewrite: FBCSP / Riemannian features, model zoo, ensemble stack
- `notebooks/` — EDA and training scripts (some are Kaggle-kernel exports)
- `docs/` — design notes and ablation results
- `reports/` — figures, metrics, trained weights

Data is not included — download from Kaggle and unzip `train/` / `test/` into the repo root.

```
pip install -e .
pytest
```
