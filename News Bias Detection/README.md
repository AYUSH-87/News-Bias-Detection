# News Bias Detection: Feedforward Neural Network with GloVe Embeddings

A terminal-only, end-to-end Machine Learning system that classifies the underlying political stance of journalistic opinion pieces into **Liberal**, **Conservative**, or **Neutral** using pretrained **GloVe (Global Vectors for Word Representation)** word embeddings and a **Feedforward Neural Network (FNN / Multi-Layer Perceptron)**.

---

## 📌 Problem Statement

> “Build a feedforward neural network utilizing GloVe embeddings to classify underlying political stances in journalistic opinion pieces.”

Understanding bias in journalism requires analyzing rhetorical patterns, lexical emphasis, and ideological framing. This project converts unstructured journalistic text into fixed-size continuous document vectors using GloVe embeddings, and trains a regularized Feedforward Neural Network to identify the stance category.

---

## 🏗️ Architecture & Pipeline Overview

```
                      +------------------------------------------+
                      |       Raw Opinion Piece / Article        |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |       Text Cleaning & Preprocessing      |
                      |  - HTML & URL Stripping                  |
                      |  - Lowercasing & Punctuation Filtering   |
                      |  - Word Tokenization (Length >= 2)       |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |         Pretrained GloVe Vectors         |
                      |  - Word-to-Vector Dictionary Lookup      |
                      |  - TF-IDF Weighted Word Pooling          |
                      |  - OOV Handling (Zero Vector fallback)   |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |      Fixed-Size Representation (R^50)    |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |    Feedforward Neural Network (FNN)      |
                      |  - Input Layer: (50,)                    |
                      |  - Dense 1: 128 units, ReLU, L2 reg      |
                      |  - LayerNormalization + Dropout (0.3)    |
                      |  - Dense 2: 64 units, ReLU, L2 reg       |
                      |  - LayerNormalization + Dropout (0.2)    |
                      |  - Output: 3 units (Softmax)             |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |    Stance Probabilities & Prediction     |
                      |   [ Liberal,  Conservative,  Neutral ]  |
                      +------------------------------------------+
```

---

## 📁 Project Directory Structure

```
News Bias Detection/
├── config.py                 # Centralized configuration, paths, and hyperparameters
├── main.py                   # CLI entrypoint (train, evaluate, predict, download-glove)
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── README.md                 # Complete project documentation
├── data/
│   ├── news_bias_dataset.csv # Curated dataset with 72 balanced opinion pieces
│   ├── sample_article.txt    # Sample opinion piece for CLI testing
│   └── embeddings/
│       └── glove_mini.50d.txt# Starter 50-dimensional GloVe vocabulary (1,419 words)
├── src/
│   ├── __init__.py           # Package marker
│   ├── data_loader.py        # CSV ingestion, cleaning, deduplication, class stats, split
│   ├── preprocessor.py       # Text normalization, tokenization, label encoding
│   ├── embeddings.py         # GloVe loader, OOV handling, document vector pooling
│   ├── model.py              # FNN architecture builder, compile, save/load
│   ├── train.py              # Full training loop, early stopping, history curves
│   ├── evaluate.py           # Evaluation pipeline, classification report, confusion matrix
│   ├── predict.py            # Terminal inference engine for text strings and files
│   └── utils.py              # Visualizations, seeding, terminal formatting, downloader
├── scripts/
│   └── generate_starter_data.py # Dataset and starter GloVe generator
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      # Automated unit and integration test suite
└── artifacts/                # Generated model artifacts and visual plots
    ├── fnn_bias_model.keras  # Saved Keras model
    ├── label_encoder.joblib  # Serialized Scikit-learn LabelEncoder
    ├── preprocessor_config.json # Serialized preprocessor configuration
    ├── metadata.json         # Run metadata, parameters, and test metrics
    ├── training_history.png  # Training vs Validation loss and accuracy curves
    └── confusion_matrix.png  # Confusion matrix heatmap visualization
```

---

## ⚙️ Installation & Setup

### 1. Requirements
Ensure Python 3.10 or 3.11 is installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. GloVe Embeddings
- **Ready out-of-the-box**: A lightweight starter GloVe file (`data/embeddings/glove_mini.50d.txt`) is bundled with the project. It contains 1,419 topic-aligned words with 50-dimensional vectors so that training and inference execute immediately without large downloads.
- **Optional Full Stanford GloVe**: To download and use the official Stanford GloVe 6B embeddings (~822MB archive):
  ```bash
  python main.py download-glove --dim 50
  ```

---

## 💻 Terminal Command Usage

All features are executed directly from the terminal via `main.py`.

### 1. Training the Model
Train the FNN classifier using the labeled dataset and GloVe vectors:
```bash
python main.py train
```

#### Custom Training Flags:
```bash
python main.py train \
  --epochs 30 \
  --batch-size 8 \
  --lr 0.001 \
  --hidden-units 128,64 \
  --dropouts 0.3,0.2 \
  --patience 5 \
  --dataset data/news_bias_dataset.csv \
  --glove-path data/embeddings/glove_mini.50d.txt
```

Training automatically generates:
- Model checkpoint: `artifacts/fnn_bias_model.keras`
- Dual loss & accuracy plot: `artifacts/training_history.png`
- Confusion matrix heatmap: `artifacts/confusion_matrix.png`
- Metadata: `artifacts/metadata.json`

---

### 2. Standalone Model Evaluation
Evaluate the saved model on the held-out test split:
```bash
python main.py evaluate
```

**Terminal Output includes:**
- Overall Accuracy
- Macro & Weighted Precision, Recall, and F1-Scores
- Full Scikit-learn Classification Report
- Formatted text Confusion Matrix
- Updated `artifacts/confusion_matrix.png`

---

### 3. Predicting Stance from a Raw Text String
Provide an inline text excerpt:
```bash
python main.py predict --text "Slashing corporate tax rates, cutting burdensome federal regulations, and defending constitutional economic liberty will unleash free enterprise."
```

**Terminal Output:**
```
======================================================================
  NEWS BIAS PREDICTION RESULT
======================================================================
Source: Raw Text ("Slashing corporate tax rates, cutting burdensome federal regulations, and defend...")
Tokens analyzed: 17 (Recognized in GloVe: 15, OOV: 2)

=======================================================
  PREDICTED STANCE:   CONSERVATIVE
  CONFIDENCE:         98.41%
=======================================================

Probability Distribution:
Class           Probability  Visual Distribution      
-------------------------------------------------------
conservative    98.41%       |################### |
neutral          1.16%       |                    |
liberal          0.43%       |                    |
-------------------------------------------------------

[ETHICAL / EPISTEMIC DISCLAIMER]
  This prediction reflects statistical patterns learned strictly from the
  annotated training dataset. It does not constitute an objective truth
  determination of political ideology, nor does it replace nuanced human editorial judgment.
-------------------------------------------------------
```

---

### 4. Predicting Stance from a File
Classify a longer text article stored in a file:
```bash
python main.py predict --file data/sample_article.txt
```

**Terminal Output:**
```
======================================================================
  NEWS BIAS PREDICTION RESULT
======================================================================
Source: File: sample_article.txt
Tokens analyzed: 53 (Recognized in GloVe: 42, OOV: 11)

=======================================================
  PREDICTED STANCE:   CONSERVATIVE
  CONFIDENCE:         89.35%
=======================================================

Probability Distribution:
Class           Probability  Visual Distribution      
-------------------------------------------------------
conservative    89.35%       |#################   |
liberal          7.58%       |#                   |
neutral          3.06%       |                    |
-------------------------------------------------------
```

---

### 5. Running the Test Suite
Execute the automated unit and integration tests:
```bash
python -m unittest tests/test_pipeline.py
```

---

## 📊 Evaluation & Metrics Summary

Performance on the held-out test set:

| Stance Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Conservative** | 0.6667 | 0.8000 | **0.7273** | 5 |
| **Liberal** | 0.6667 | 0.8000 | **0.7273** | 5 |
| **Neutral** | 1.0000 | 0.6000 | **0.7500** | 5 |
| **Overall Accuracy** | — | — | **0.7333** | 15 |
| **Macro Average** | 0.7778 | 0.7333 | **0.7348** | 15 |
| **Weighted Average** | 0.7778 | 0.7333 | **0.7348** | 15 |

Visual plots are automatically saved in `artifacts/`:
- `artifacts/training_history.png`: Epoch-by-epoch training and validation loss and accuracy curves.
- `artifacts/confusion_matrix.png`: Heatmap visualization of true vs predicted classes.

---

## ⚖️ Ethical & Epistemic Disclaimer

Political discourse is intrinsically multifaceted, evolving, and contextual. The classifications generated by this Feedforward Neural Network are strictly statistical inferences derived from the specific labeling ontology, article samples, and vocabulary distributions present in the training data.

The model's predictions **do not** represent an objective truth determination of political stance or philosophical validity, and should not be used as an automated content filter or editorial gatekeeper without human oversight.
