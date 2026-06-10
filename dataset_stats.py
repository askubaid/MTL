import pandas as pd
from datasets import load_dataset
import os

def main():
    print("="*60)
    print("1. ORIGINAL DATASETS DISTRIBUTION (All splits combined)")
    print("="*60)
    
    # Load original datasets
    print("\nLoading DAIR-AI/Emotion...")
    emotion_dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)
    emotion_df = pd.concat([
        emotion_dataset['train'].to_pandas(),
        emotion_dataset['validation'].to_pandas(),
        emotion_dataset['test'].to_pandas()
    ])
    
    # The labels for DAIR-AI/Emotion are 0-5
    emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    emotion_df['label_name'] = emotion_df['label'].apply(lambda x: emotion_labels[x])
    emotion_dist = emotion_df['label_name'].value_counts()
    print("\n[DAIR-AI/Emotion Class Distribution]")
    for label, count in emotion_dist.items():
        print(f"  {label:<15} : {count:,}")
    
    print("\nLoading CLINC150 (Intent)...")
    intent_dataset = load_dataset("clinc_oos", "plus")
    intent_df = pd.concat([
        intent_dataset['train'].to_pandas(),
        intent_dataset['validation'].to_pandas(),
        intent_dataset['test'].to_pandas()
    ])
    intent_dist = intent_df['intent'].value_counts()
    print("\n[CLINC150 (Intent) Class Distribution (Top 10 & Bottom 10)]")
    for label, count in list(intent_dist.items())[:10]:
        print(f"  {label:<25} : {count:,}")
    print("  ... (skipping middle classes)")
    for label, count in list(intent_dist.items())[-10:]:
        print(f"  {label:<25} : {count:,}")
        
    print("\n" + "="*60)
    print("2. GENERATED MTL DATASETS (Train/Validation/Test)")
    print("="*60)
    
    data_dir = "data/custom_mtl_dataset_csv"
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "validation.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    print(f"  Train Dataset      : {len(train_df):,} samples")
    print(f"  Validation Dataset : {len(val_df):,} samples")
    print(f"  Test Dataset       : {len(test_df):,} samples")
    print("-" * 35)
    print(f"  Total              : {len(train_df) + len(val_df) + len(test_df):,} samples")
    
    print("\n" + "="*60)
    print("3. TRAINING SET CLASS DISTRIBUTION")
    print("="*60)
    
    # Intents have emotion_label == -1
    intent_samples = train_df[train_df['emotion_label'] == -1]
    # Emotions have intent_label == -1
    emotion_samples = train_df[train_df['intent_label'] == -1]
    
    print(f"\n[Total Samples by Task]")
    print(f"  Intent Samples  : {len(intent_samples):,}")
    print(f"  Emotion Samples : {len(emotion_samples):,}")
    
    # Load label maps to show names
    intent_map = pd.read_csv("data/label_maps/intent_labels.csv").set_index('label_id')['label_name'].to_dict()
    emotion_map = pd.read_csv("data/label_maps/emotion_labels.csv").set_index('label_id')['label_name'].to_dict()
    
    print("\n[Training Set - Emotion Class Counts]")
    train_emotion_counts = emotion_samples['emotion_label'].value_counts()
    for label_id, count in train_emotion_counts.items():
        print(f"  {emotion_map[label_id]:<15} : {count:,}")
        
    print("\n[Training Set - Intent Class Counts (Top 10 & Bottom 10)]")
    train_intent_counts = intent_samples['intent_label'].value_counts()
    for label_id, count in list(train_intent_counts.items())[:10]:
        print(f"  {intent_map[label_id]:<25} : {count:,}")
    print("  ... (skipping middle classes)")
    for label_id, count in list(train_intent_counts.items())[-10:]:
        print(f"  {intent_map[label_id]:<25} : {count:,}")

if __name__ == '__main__':
    main()
