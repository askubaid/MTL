# Multi-Task Learning Pipeline: Intent & Emotion Classification

A deep learning system that simultaneously classifies **user intent** and **emotion** from text, powered by a dual-head RoBERTa transformer — served through a FastAPI backend and a modern React web interface.

---

## Overview

Traditional NLP pipelines treat intent detection and emotion classification as separate problems. This project explores **Multi-Task Learning (MTL)** — a paradigm where a single shared Transformer encoder learns both tasks jointly, exploiting the semantic overlap between *what a user wants* and *how they feel*. The result is a single inference pass that produces two rich labels, enabling richer understanding of customer interactions.

---

## Architecture

```
                ┌─────────────────────────────┐
                │        Input Text            │
                └─────────────────────────────┘
                              │
                              ▼
                ┌─────────────────────────────┐
                │  RoBERTa-base Encoder        │
                │  (Shared Backbone)           │
                │  + Dropout (0.1)             │
                └─────────────────────────────┘
                    /                      \
                   /                        \
    ┌─────────────────────┐   ┌─────────────────────┐
    │   Intent Head        │   │   Emotion Head       │
    │   Linear(768→151)    │   │   Linear(768→28)     │
    └─────────────────────┘   └─────────────────────┘
              │                          │
    151 Intent Classes          28 Emotion Classes
```

- **Shared Encoder:** `roberta-base` (125M parameters), frozen bottom layers optional
- **[CLS] Pooler Output** feeds into both classification heads
- **Joint Loss:** Sum of two independent Cross-Entropy losses, one per head
- **Mixed Precision:** AMP/FP16 training for efficiency on consumer GPUs

---

## Datasets

| Dataset | Task | Classes | Samples |
|---|---|---|---|
| [CLINC150](https://huggingface.co/datasets/clinc_oos) | Intent Detection | 151 | ~23,700 |
| [GoEmotions](https://huggingface.co/datasets/go_emotions) | Emotion Classification | 28 | ~58,000 |

A custom pipeline (`create_dataset.py`) logically combines these two datasets, assigning every text sample a valid Intent label and an Emotion label. A validation script (`validate_dataset.py`) checks class distributions and structural integrity before training begins.

---

## Results

| Metric | Score |
|---|---|
| **Intent Accuracy** | **89.65%** |
| Intent Macro F1 | 0.9219 |
| **Emotion Accuracy** | **57.79%** |
| Emotion Macro F1 | 0.4667 |
| Combined Score | 0.6816 |

Training was performed for **5 epochs** on an **NVIDIA RTX 2080 (8GB VRAM)** using FP16 mixed precision.

### Learning Curves

![Learning Curves](evaluation/learning_curves.png)

---

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Base Model | `roberta-base` |
| Epochs | 5 |
| Batch Size | 32 |
| Learning Rate | 2e-5 |
| Max Token Length | 128 |
| Optimizer | AdamW |
| LR Scheduler | Linear warmup (10%) |
| Loss Function | CrossEntropyLoss (joint sum) |
| Mixed Precision | AMP FP16 |

---

## Project Structure

```
Project/
│
├── backend/                    # FastAPI backend API
│   ├── main.py                 # /predict, /labels, /evaluation-info endpoints
│   └── requirements.txt        # Backend dependencies
│
├── frontend/                   # React + Vite web UI
│   └── src/
│       ├── App.jsx             # Main application
│       ├── index.css           # Elegant cream paper theme
│       └── components/
│           ├── PredictionCard.jsx  # Animated result display
│           └── InfoModal.jsx       # InfoGraphics modal
│
├── data/
│   ├── custom_mtl_dataset_csv/ # Generated dataset (train / validation splits)
│   └── label_maps/             # intent_labels.csv, emotion_labels.csv
│
├── evaluation/
│   ├── learning_curves.png     # Training & validation loss/accuracy plots
│   └── evaluation_summary.csv  # Aggregated evaluation metrics
│
├── models/
│   ├── best_model.pt           # Best model checkpoint (saved by validation loss)
│   └── last_checkpoint.pt      # Latest checkpoint for resuming training
│
├── model.py                    # MultiTaskRoBERTa architecture
├── dataset.py                  # PyTorch Dataset class & DataLoaders
├── create_dataset.py           # Dataset combination pipeline
├── validate_dataset.py         # Dataset health checks
├── train.py                    # Training loop with AMP & checkpointing
├── evaluate.py                 # Full evaluation with per-class reports
├── inference.py                # CLI inference script
└── evaluation.txt              # Detailed per-class evaluation report
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, CPU fallback supported)
- Node.js 18+ and npm
- `conda` (optional, but recommended)

### 1. Clone & Set Up Environment

```bash
# Create and activate environment
conda create -n mypy python=3.10
conda activate mypy
```

### 2. Install Python Dependencies

```bash
# Backend dependencies
pip install -r backend/requirements.txt
```

### 3. Prepare the Dataset

```bash
# Generate the combined MTL dataset from CLINC150 + GoEmotions
python create_dataset.py

# Validate the generated dataset
python validate_dataset.py
```

### 4. Train the Model

```bash
python train.py
```

Training will save `models/best_model.pt` (best validation loss) and `models/last_checkpoint.pt` (for resuming). To resume from a checkpoint, simply re-run `python train.py` — it will detect the checkpoint automatically.

### 5. Evaluate the Model

```bash
python evaluate.py
```

This generates `evaluation/evaluation_summary.csv`, `evaluation/learning_curves.png`, and the full `evaluation.txt` report.

### 6. CLI Inference

```bash
python inference.py
```

You will be prompted to enter any text. The model will output the predicted Intent and Emotion with confidence scores.

```
> Enter text: I need help tracking my package

  Intent  : track_package  (97.3%)
  Emotion : curiosity  (62.1%)
```

### 7. Run the Web Application

**Terminal 1 — Start Backend:**
```bash
uvicorn backend.main:app --reload
```
The API will be available at `http://localhost:8000`.

**Terminal 2 — Start Frontend:**
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## Web Application Features

- **📝 Text Analysis** — Enter any text review and receive dual predictions instantly
- **📊 Confidence Scores** — Animated progress bars display prediction confidence
- **🏷️ Label Reference** — Scrollable side-by-side lists of all 151 Intent and 28 Emotion labels
- **📈 InfoGraphics Modal** — View training config, datasets, evaluation scores, learning curves, and the detailed classification report — all accessible via the top-right button

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Run inference on input text |
| `GET` | `/labels` | List all intent & emotion label names |
| `GET` | `/evaluation-info` | Get training config, datasets & metrics |
| `GET` | `/evaluation/learning_curves.png` | Serve the training curves image |

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am really frustrated with this broken product"}'
```

**Example Response:**
```json
{
  "intent": "complaint",
  "intent_confidence": 0.891,
  "emotion": "anger",
  "emotion_confidence": 0.743
}
```

---

## Hardware & Performance

- **Training Hardware:** NVIDIA RTX 2080 8GB VRAM
- **Training Precision:** FP16 (Automatic Mixed Precision)
- **Inference:** GPU (CUDA) or CPU fallback
- **Training Time:** ~5 epochs on the combined dataset

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch, HuggingFace Transformers (`roberta-base`) |
| Dataset | HuggingFace Datasets (CLINC150, GoEmotions) |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Frontend | React 18, Vite, Vanilla CSS |
| Evaluation | Scikit-learn, Matplotlib |
