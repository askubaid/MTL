import os
import pandas as pd

def validate_dataset():
    dataset_path = "data/custom_mtl_dataset_csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    
    print("\n" + "="*50)
    print("DATASET HEALTH REPORT")
    print("="*50)
    
    overall_health = True
    
    for split in ['train', 'validation', 'test']:
        print(f"\n--- Split: {split.upper()} ---")
        csv_file = os.path.join(dataset_path, f"{split}.csv")
        if not os.path.exists(csv_file):
            print(f"[FAIL] Missing split file: {csv_file}")
            overall_health = False
            continue
            
        df = pd.read_csv(csv_file)
        total_samples = len(df)
        print(f"Total samples: {total_samples}")
        
        # Check for missing texts
        missing_text = df['text'].isnull().sum()
        if missing_text > 0:
            print(f"[FAIL] Found {missing_text} samples with missing/null text.")
            overall_health = False
        else:
            print("[PASS] No missing texts.")
            
        # Check label constraints
        valid_intent_samples = df[(df['intent_label'] != -1) & (df['emotion_label'] == -1)]
        valid_emotion_samples = df[(df['intent_label'] == -1) & (df['emotion_label'] != -1)]
        
        invalid_samples = df[(df['intent_label'] == -1) & (df['emotion_label'] == -1)]
        multi_label_samples = df[(df['intent_label'] != -1) & (df['emotion_label'] != -1)]
        
        if len(invalid_samples) > 0:
            print(f"[FAIL] Found {len(invalid_samples)} samples with NO labels (both are -1).")
            overall_health = False
        
        if len(multi_label_samples) > 0:
            print(f"[FAIL] Found {len(multi_label_samples)} samples with BOTH labels (none are -1).")
            overall_health = False
            
        print(f"Intent samples: {len(valid_intent_samples)} ({len(valid_intent_samples)/total_samples*100:.1f}%)")
        print(f"Emotion samples: {len(valid_emotion_samples)} ({len(valid_emotion_samples)/total_samples*100:.1f}%)")
        
        # Check class distribution briefly
        num_intents = valid_intent_samples['intent_label'].nunique()
        num_emotions = valid_emotion_samples['emotion_label'].nunique()
        print(f"Unique Intent classes found : {num_intents}  (expected: 151)")
        print(f"Unique Emotion classes found: {num_emotions}  (expected: up to 28)")

    print("\n" + "="*50)
    if overall_health:
        print("[PASS] DATASET VALIDATION PASSED!")
    else:
        print("[FAIL] DATASET VALIDATION FAILED! Please check the errors above.")

if __name__ == "__main__":
    validate_dataset()
