# Genre Text Classifier (Romance / Slice-of-life / Action / Fantasy / Psychological / Thriller / Other)

## Overview

This project trains a multi-class text classifier that predicts the **genre** of a short story synopsis.  
The model outputs one of 7 genres: Romance, Slice-of-life, Action, Fantasy, Psychological, Thriller, or Other.

The goal is to demonstrate:
- Using a **pre-trained Transformer (DistilBERT)** for text classification.
- Building a full supervised learning pipeline with Hugging Face `datasets` and `transformers`.
- Evaluating with **accuracy** and **macro-F1** across multiple classes.

## Dataset

I created a synthetic dataset of **1,400** short synopses.  
Each row has:
- `text`: 1–3 sentence story description.
- `genre`: one of the 7 genres listed above.

Genres are approximately balanced (~190–210 examples per class).

The merged dataset is saved as `data/genre_data_merged.csv`.

## Model and Training

- Base model: `distilbert-base-uncased` from Hugging Face.
- Task: sequence classification with `num_labels = 7`.
- Framework: Hugging Face `transformers` + `datasets`.
- Tokenization: truncation and padding to `max_length = 256`.
- Training:
  - Epochs: 3
  - Batch size: 8
  - Optimizer & scheduler: defaults from `TrainingArguments`
  - Evaluation strategy: per epoch
- Metrics:
  - Accuracy
  - Macro F1 (treat all genres equally)

## Results

On the held-out test set:

- Accuracy: **~0.86**
- Macro-F1: **~0.86**

(These values are from `trainer.evaluate(tokenized_datasets["test"])`.)

The model performs well on Action (F1=0.95) and Slice-of-life (F1=0.85) but shows confusion between Romance and Slice-of-life, likely due to overlapping tonal characteristics in the training data. This is a known limitation of synthetic datasets.

## Usage

### In Colab

1. Open `notebooks/genre_classifier_project_1.ipynb` in Google Colab.
2. Run the cells to load the trained model (or re-train if needed).
3. Use the helper function:

```python
label, conf = predict_genre("A shy girl transfers to a new school and slowly falls in love with her club senior.")
print(label, conf)
