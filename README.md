# GullibleTransformer

A simple fake-news headline classifier built for learning the fundamentals of Transformers.
The goal of this project was to **implement core components from scratch** to understand how modern models work internally.

---

## 📘 Overview

This project classifies news headlines as **Real** or **Fake** using:

* a **custom BPE tokenizer**,
* sinusoidal **positional embeddings**,
* a manually implemented **Transformer Encoder**,
* a complete **training loop** in PyTorch, and
* a **Flask API** for inference.

It is an educational, from-scratch implementation rather than a production-grade model.

---

## 🧱 Architecture

```
                   ┌──────────────────────┐
                   │      Input Text      │
                   └───────────┬──────────┘
                               │
                               ▼
                     Custom BPE Tokenizer
                               │
                               ▼
               Token IDs → Embedding Layers
            (Token Embedding + Positional Encoding)
                               │
                               ▼
                     Transformer Encoder
           (Multi-Head Attention + MLP + LayerNorm)
                               │
                               ▼
                         Mean Pooling
                               │
                               ▼
                    Linear Classification
                               │
                               ▼
               Output: Fake / Real + Confidence
```

---

## 🔧 What’s Implemented

### **1. Custom BPE Tokenizer** (`tokenizer.py`)

* Trains a small Byte-Pair Encoding tokenizer on the dataset
* Encodes/decodes text into token IDs
* Saves and loads custom vocabulary files

### **2. Embeddings + Transformer Encoder** (`embedding.py`, `model.py`)

* Token embeddings + sinusoidal positional embeddings
* Multi-head self-attention
* MLP feed-forward block
* LayerNorm + residual connections
* Stacked encoder layers
* Mean-pooling representation → classification head

### **3. Data Processing** (`data.py`)

* Reads and cleans headline dataset
* Prepares text for tokenizer and training

### **4. Training Pipeline** (`train.py`)

* Tokenizes headlines
* Pads sequences
* Uses PyTorch DataLoaders
* Trains using AdamW + cross-entropy
* Saves best model + tokenizer
* Plots loss and accuracy

### **5. Inference Server** (`server.py`)

* Loads trained model
* Exposes `/predict` endpoint via Flask
* Returns label + confidence

---

## 📊 How Good Is It?

The model trains and produces testing accuracy of 96.8% for the used Kaggle Dataset

---

## 🚧 Future Improvements

A few important improvements that would make the model more reliable:

* Add **attention masks** to ignore padded tokens
* Add a dedicated **CLS token** instead of mean pooling
* Use a more robust tokenizer (e.g., HF Tokenizers or SentencePiece)
* Improve evaluation (precision, recall, F1, confusion matrix)
* Remove hardcoded paths and externalize configs
* Add learning-rate warmup + scheduler

---

## ▶️ Running Training

```
python train.py
```

## ▶️ Running the Server

```
python server.py
```

Send a POST request:

```json
{ "headline": "Sample headline here" }
```
