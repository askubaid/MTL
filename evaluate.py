import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score, f1_score, classification_report
from model import MultiTaskRoBERTa
from dataset import MTLDataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

# ─── Config ─────────────────────────────────────────────────────────────────

MODEL_PATH    = "models/best_model.pt"
TEST_CSV      = "data/custom_mtl_dataset_csv/test.csv"
INTENT_LABELS = "data/label_maps/intent_labels.csv"
EMOTION_LABELS = "data/label_maps/emotion_labels.csv"
BATCH_SIZE    = 64
MAX_LENGTH    = 128
OUTPUT_DIR    = "evaluation"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_model(device):
    model = MultiTaskRoBERTa(num_intents=151, num_emotions=28)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(model, dataloader, device):
    all_intent_preds, all_intent_labels = [], []
    all_emotion_preds, all_emotion_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_labels  = batch['intent_label']
            emotion_labels = batch['emotion_label']

            intent_logits, emotion_logits = model(input_ids, attention_mask)

            intent_preds  = torch.argmax(intent_logits, dim=1).cpu()
            emotion_preds = torch.argmax(emotion_logits, dim=1).cpu()

            # Only evaluate on samples whose label is valid for this head
            intent_mask  = intent_labels  != -1
            emotion_mask = emotion_labels != -1

            all_intent_preds.extend(intent_preds[intent_mask].tolist())
            all_intent_labels.extend(intent_labels[intent_mask].tolist())

            all_emotion_preds.extend(emotion_preds[emotion_mask].tolist())
            all_emotion_labels.extend(emotion_labels[emotion_mask].tolist())

    return (all_intent_preds, all_intent_labels,
            all_emotion_preds, all_emotion_labels)


def compute_metrics(preds, labels, task_name, label_names=None):
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    print(f"\n{'='*50}")
    print(f"  {task_name} Metrics")
    print(f"{'='*50}")
    print(f"  Accuracy     : {acc*100:.2f}%")
    print(f"  Macro F1     : {macro_f1:.4f}")
    if label_names is not None:
        report = classification_report(
            labels, preds, target_names=label_names, zero_division=0
        )
        print(f"\n  Per-Class Report:\n{report}")
    return acc, macro_f1


# ─── Learning Curve Plots ────────────────────────────────────────────────────

def plot_learning_curves():
    train_log_path = "models/training_logs.csv"
    val_log_path   = "models/val_logs.csv"

    if not os.path.exists(train_log_path):
        print("[WARN] training_logs.csv not found, skipping learning curves.")
        return

    df = pd.read_csv(train_log_path)

    # Create a global step index across all epochs for x-axis
    steps_per_epoch = df[df['epoch'] == df['epoch'].min()]['step'].max()
    df['global_step'] = (df['epoch'] - 1) * steps_per_epoch + df['step']

    # Downsample: average within every 100-step window
    window = 100
    df['window'] = (df['global_step'] - 1) // window
    smoothed = df.groupby('window').agg(
        global_step  = ('global_step', 'mean'),
        total_loss   = ('total_loss',  'mean'),
        intent_loss  = ('intent_loss', 'mean'),
        emotion_loss = ('emotion_loss','mean'),
    ).reset_index(drop=True)

    # Dark theme setup
    BG      = '#0f0f1a'
    PANEL   = '#1a1a2e'
    GRID    = '#222244'
    C_TOTAL   = '#7c6fff'
    C_INTENT  = '#ff6fb8'
    C_EMOTION = '#6fe8ff'
    C_VAL     = '#ffd966'

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(PANEL)

    # Plot the three training curves
    ax.plot(smoothed['global_step'], smoothed['total_loss'],
            color=C_TOTAL,   linewidth=1.8, alpha=0.92, label='Total Train Loss')
    ax.plot(smoothed['global_step'], smoothed['intent_loss'],
            color=C_INTENT,  linewidth=1.5, alpha=0.85, label='Intent Train Loss')
    ax.plot(smoothed['global_step'], smoothed['emotion_loss'],
            color=C_EMOTION, linewidth=1.5, alpha=0.85, label='Emotion Train Loss')

    # Overlay validation loss if available (one point per epoch)
    if os.path.exists(val_log_path):
        val_df = pd.read_csv(val_log_path)
        # Map each epoch's val loss to the last global step of that epoch
        val_df['global_step'] = val_df['epoch'] * steps_per_epoch
        ax.plot(val_df['global_step'], val_df['val_loss'],
                color=C_VAL, linewidth=2.0, linestyle='--',
                marker='o', markersize=7, label='Val Loss (total)')

    # Draw vertical dashed lines at epoch boundaries
    num_epochs = int(df['epoch'].max())
    for e in range(1, num_epochs):
        ax.axvline(x=e * steps_per_epoch, color='#444466',
                   linestyle=':', linewidth=1.0)
        ax.text(e * steps_per_epoch + steps_per_epoch * 0.02,
                ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 3.5,
                f'E{e+1}', color='#888899', fontsize=9)

    ax.set_title('Training Learning Curves  (100-step rolling average)',
                 color='white', fontsize=14, pad=12)
    ax.set_xlabel('Global Training Step', color='#aaaacc', fontsize=11)
    ax.set_ylabel('Loss', color='#aaaacc', fontsize=11)
    ax.tick_params(colors='#aaaacc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.grid(True, color=GRID, linestyle='--', linewidth=0.6)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    leg = ax.legend(facecolor=PANEL, labelcolor='white',
                    edgecolor='#333355', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "learning_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"\nLearning curves saved -> {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load label name maps
    intent_label_df  = pd.read_csv(INTENT_LABELS)
    emotion_label_df = pd.read_csv(EMOTION_LABELS)
    intent_names     = intent_label_df.sort_values('label_id')['label_name'].tolist()
    emotion_names    = emotion_label_df.sort_values('label_id')['label_name'].tolist()

    # Load tokenizer and test dataset
    tokenizer    = RobertaTokenizer.from_pretrained("roberta-base")
    test_dataset = MTLDataset(TEST_CSV, tokenizer, max_length=MAX_LENGTH)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(device)

    # Run inference
    print("Running inference on test set...")
    intent_preds, intent_labels, emotion_preds, emotion_labels = run_inference(
        model, test_loader, device
    )

    # Compute & print metrics
    intent_acc, intent_f1 = compute_metrics(
        intent_preds, intent_labels, "INTENT HEAD (CLINC150)", label_names=intent_names
    )
    emotion_acc, emotion_f1 = compute_metrics(
        emotion_preds, emotion_labels, "EMOTION HEAD (GoEmotions)", label_names=emotion_names
    )

    # Combined score
    combined_score = (intent_acc + emotion_f1) / 2
    print(f"\n{'='*50}")
    print(f"  COMBINED SCORE")
    print(f"{'='*50}")
    print(f"  (Intent Accuracy + Emotion Macro F1) / 2")
    print(f"  = ({intent_acc*100:.2f}% + {emotion_f1:.4f}) / 2  =  {combined_score:.4f}")

    # Save summary to CSV
    summary = pd.DataFrame([{
        'intent_accuracy':   round(intent_acc,  4),
        'intent_macro_f1':   round(intent_f1,   4),
        'emotion_accuracy':  round(emotion_acc, 4),
        'emotion_macro_f1':  round(emotion_f1,  4),
        'combined_score':    round(combined_score, 4),
    }])
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved -> {summary_path}")

    # Plot learning curves
    plot_learning_curves()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    evaluate()
