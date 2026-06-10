import torch
import pandas as pd
from transformers import RobertaTokenizer
from model import MultiTaskRoBERTa

# ─── Config ─────────────────────────────────────────────────────────────────

MODEL_PATH     = "models/best_model.pt"
INTENT_LABELS  = "data/label_maps/intent_labels.csv"
EMOTION_LABELS = "data/label_maps/emotion_labels.csv"
MAX_LENGTH     = 128

# ─── Setup ───────────────────────────────────────────────────────────────────

def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    model = MultiTaskRoBERTa(num_intents=151, num_emotions=6)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    intent_names  = pd.read_csv(INTENT_LABELS).sort_values('label_id')['label_name'].tolist()
    emotion_names = pd.read_csv(EMOTION_LABELS).sort_values('label_id')['label_name'].tolist()

    return model, tokenizer, device, intent_names, emotion_names


def predict(text, model, tokenizer, device, intent_names, emotion_names):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        intent_logits, emotion_logits = model(input_ids, attention_mask)

    intent_probs  = torch.softmax(intent_logits, dim=1)
    emotion_probs = torch.softmax(emotion_logits, dim=1)

    intent_idx  = torch.argmax(intent_probs, dim=1).item()
    emotion_idx = torch.argmax(emotion_probs, dim=1).item()

    intent_conf  = intent_probs[0, intent_idx].item()
    emotion_conf = emotion_probs[0, emotion_idx].item()

    return {
        'intent':           intent_names[intent_idx],
        'intent_confidence': intent_conf,
        'emotion':          emotion_names[emotion_idx],
        'emotion_confidence': emotion_conf,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading model and tokenizer...")
    model, tokenizer, device, intent_names, emotion_names = load_resources()
    print(f"Model loaded on {device}. Ready for inference.\n")

    print("=" * 55)
    print("  Multi-Task Inference CLI  (type 'quit' to exit)")
    print("=" * 55)

    while True:
        text = input("\n> Enter text: ").strip()
        if text.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        if not text:
            continue

        result = predict(text, model, tokenizer, device, intent_names, emotion_names)

        print(f"\n  Intent  : {result['intent']}  ({result['intent_confidence']*100:.1f}%)")
        print(f"  Emotion : {result['emotion']}  ({result['emotion_confidence']*100:.1f}%)")


if __name__ == "__main__":
    main()
