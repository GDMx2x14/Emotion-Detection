# Emotion Detection

A comparative study of classical machine learning, deep learning, and large language model
approaches to emotion classification in text, evaluated on the **GoEmotions** and **Twitter**
datasets.

## Overview

This project benchmarks a range of modeling strategies — from classical baselines to fine-tuned
transformers to zero/few-shot LLM inference — on the task of multi-class emotion classification,
and studies which words are most emotionally discriminative via a PMI-based ranking method.

## Approaches compared

- **Classical ML**: Naive Bayes, Decision Trees, Random Forest
- **Deep learning**: custom PyTorch models
- **Transformers**: BERT, RoBERTa, SocBERT (fine-tuned on GoEmotions)
- **LLM few-shot inference**: Llama 3 predictions (zero/few-shot, no fine-tuning)

All models are evaluated on a shared held-out test split for a fair final comparison.

## Repository structure

```
dataset/      raw and cleaned GoEmotions + Twitter data (preparation notebooks included)
lib/          shared Python modules: grid search, data loading/preprocessing,
              PyTorch model building, evaluation, plotting
results/      Llama 3 predictions, PMI word-ranking scores, saved grid-search results
GoEmotions_*.ipynb   one notebook per model (Bayes, Decision Trees/Random Forest, BERT, RoBERTa, SocBERT)
GoEmotions_data_exploration.ipynb / Twitter_data_exploration.ipynb   dataset exploration
GoEmotions_emotion_words.ipynb / Twitter_emotion_words.ipynb         PMI emotion-word ranking
model_comparison.ipynb   final model comparison on the held-out test set
HLT_Project_Report.pdf   full write-up: methodology, experiments, and results
```

## Methodology

Each model family is developed and tuned in its own notebook via grid search over
hyperparameters (results cached under `results/` to avoid re-running searches), with
best checkpoints saved for later comparison. `model_comparison.ipynb` brings the best
version of every model together on the same test set to produce the final comparison.
Full metrics and discussion are in `HLT_Project_Report.pdf`.

## Setup

```bash
pip install -r requirements.txt
# or, with conda:
conda env create -f requirements_conda.txt
```

## Contributors

This project was developed by a team of three:
- [Giuseppe De Marco](https://github.com/GDMx2x14) (some commits authored under the `GDM99` alias of this account)
- [AndreaAlberti116](https://github.com/AndreaAlberti116)
- [Nsivaa](https://github.com/Nsivaa)
