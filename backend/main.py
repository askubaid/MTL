import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import pandas as pd
from transformers import RobertaTokenizer

# Make sure we can import from the parent directory if running from backend/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from model import MultiTaskRoBERTa
except ImportError:
    pass # If running from root, it might work without appending

app = FastAPI(title="Multi-Task Learning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
import csv

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
evaluation_dir = os.path.join(base_dir, "evaluation")
if os.path.exists(evaluation_dir):
    app.mount("/evaluation", StaticFiles(directory=evaluation_dir), name="evaluation")

@app.get("/evaluation-info")
def get_evaluation_info():
    eval_csv = os.path.join(evaluation_dir, "evaluation_summary.csv")
    eval_txt = os.path.join(base_dir, "evaluation.txt")
    
    summary = []
    if os.path.exists(eval_csv):
        with open(eval_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                summary.append(row)
                
    details = ""
    if os.path.exists(eval_txt):
        with open(eval_txt, 'r') as f:
            details = f.read(5000) # read first 5000 chars to avoid huge payload

    return {
        "hyperparameters": {
            "batch_size": 32,
            "learning_rate": "2e-5",
            "max_length": 128
        },
        "datasets": [
            "data/custom_mtl_dataset_csv/train.csv",
            "data/custom_mtl_dataset_csv/validation.csv"
        ],
        "epochs": 5,
        "summary": summary,
        "details": details
    }


class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    intent: str
    intent_confidence: float
    emotion: str
    emotion_confidence: float

# Global variables to hold model and resources
model = None
tokenizer = None
device = None
intent_names = []
emotion_names = []

@app.on_event("startup")
def startup_event():
    global model, tokenizer, device, intent_names, emotion_names
    
    # Path configuration depending on where it's run
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "best_model.pt")
    intent_labels_path = os.path.join(base_dir, "data", "label_maps", "intent_labels.csv")
    emotion_labels_path = os.path.join(base_dir, "data", "label_maps", "emotion_labels.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Load model
    model = MultiTaskRoBERTa(num_intents=151, num_emotions=28)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load labels
    intent_names = pd.read_csv(intent_labels_path).sort_values('label_id')['label_name'].tolist()
    emotion_names = pd.read_csv(emotion_labels_path).sort_values('label_id')['label_name'].tolist()
    print("Model and resources loaded successfully.")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    text = request.text
    max_length = 128
    
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        intent_logits, emotion_logits = model(input_ids, attention_mask)

    intent_probs = torch.softmax(intent_logits, dim=1)
    emotion_probs = torch.softmax(emotion_logits, dim=1)

    intent_idx = torch.argmax(intent_probs, dim=1).item()
    emotion_idx = torch.argmax(emotion_probs, dim=1).item()

    intent_conf = intent_probs[0, intent_idx].item()
    emotion_conf = emotion_probs[0, emotion_idx].item()

    return PredictionResponse(
        intent=intent_names[intent_idx],
        intent_confidence=intent_conf,
        emotion=emotion_names[emotion_idx],
        emotion_confidence=emotion_conf
    )

@app.get("/labels")
def get_labels():
    return {
        "intents": intent_names,
        "emotions": emotion_names
    }
