import os
import pandas as pd
from datasets import load_dataset

# go_emotions label names (28 classes, index matches label integer)
GO_EMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def create_mtl_dataset():
    print("Loading Intent Dataset (clinc_oos / plus)...")
    intent_dataset = load_dataset("clinc_oos", "plus")

    # go_emotions: multi-label, one text can have multiple emotions.
    # Strategy: take the FIRST (primary) label from each sample → single-label classification.
    print("Loading Emotion Dataset (google-research-datasets/go_emotions)...")
    emotion_dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

    # clinc_oos has train / validation / test splits already
    intent_splits = {
        'train':      intent_dataset['train'].to_pandas(),
        'validation': intent_dataset['validation'].to_pandas(),
        'test':       intent_dataset['test'].to_pandas(),
    }

    # go_emotions has train / validation / test splits
    def go_emotions_to_df(split_ds):
        df = split_ds.to_pandas()
        # Keep only rows where at least one label is assigned
        df = df[df['labels'].apply(lambda x: len(x) > 0)].copy()
        # Take the primary (first) label as a single integer
        df['emotion_label'] = df['labels'].apply(lambda x: int(x[0]))
        df = df.rename(columns={'text': 'text'})[['text', 'emotion_label']]
        return df

    emotion_splits = {
        'train':      go_emotions_to_df(emotion_dataset['train']),
        'validation': go_emotions_to_df(emotion_dataset['validation']),
        'test':       go_emotions_to_df(emotion_dataset['test']),
    }

    combined_splits = {}

    for split_name in ['train', 'validation', 'test']:
        print(f"Processing {split_name} split...")

        # --- Intent side ---
        df_intent = intent_splits[split_name].copy()
        df_intent = df_intent.rename(columns={'intent': 'intent_label'})
        df_intent['emotion_label'] = -1
        df_intent = df_intent[['text', 'intent_label', 'emotion_label']]

        # --- Emotion side ---
        df_emotion = emotion_splits[split_name].copy()
        df_emotion['intent_label'] = -1
        df_emotion = df_emotion[['text', 'intent_label', 'emotion_label']]

        # Concatenate & shuffle
        df_combined = pd.concat([df_intent, df_emotion], ignore_index=True)
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

        combined_splits[split_name] = df_combined

    output_dir = "data/custom_mtl_dataset_csv"
    os.makedirs(output_dir, exist_ok=True)

    # Save label mappings so downstream code can decode predictions
    label_map_dir = "data/label_maps"
    os.makedirs(label_map_dir, exist_ok=True)

    # Emotion label map (go_emotions)
    emotion_map = pd.DataFrame(
        enumerate(GO_EMOTIONS_LABELS), columns=['label_id', 'label_name']
    )
    emotion_map.to_csv(os.path.join(label_map_dir, "emotion_labels.csv"), index=False)
    print(f"Saved emotion label map  -> {label_map_dir}/emotion_labels.csv")

    # Intent label map (clinc_oos - derive from features)
    intent_names = intent_dataset['train'].features['intent'].names
    clinc_names = pd.DataFrame(
        enumerate(intent_names), columns=['label_id', 'label_name']
    )
    clinc_names.to_csv(os.path.join(label_map_dir, "intent_labels.csv"), index=False)
    print(f"Saved intent label map   -> {label_map_dir}/intent_labels.csv")

    print(f"\nSaving combined dataset to {output_dir} as CSV files...")
    for split_name, df in combined_splits.items():
        csv_path = os.path.join(output_dir, f"{split_name}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Saved {csv_path}  ({len(df):,} rows)")

    print("\nDataset creation complete!")
    print(f"  Total emotion classes : {len(GO_EMOTIONS_LABELS)} (go_emotions)")
    print(f"  Total intent classes  : {len(intent_names)} (clinc_oos)")

if __name__ == "__main__":
    create_mtl_dataset()
