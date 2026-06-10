import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer

class MTLDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        intent_label = int(row['intent_label'])
        emotion_label = int(row['emotion_label'])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'intent_label': torch.tensor(intent_label, dtype=torch.long),
            'emotion_label': torch.tensor(emotion_label, dtype=torch.long)
        }

def get_dataloaders(train_csv, val_csv, batch_size=32, max_length=128):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    train_dataset = MTLDataset(train_csv, tokenizer, max_length)
    val_dataset = MTLDataset(val_csv, tokenizer, max_length)
    
    # Calculate weights for dynamic oversampling
    df_train = train_dataset.df
    
    intent_mask = df_train['intent_label'] != -1
    emotion_mask = df_train['emotion_label'] != -1
    
    num_intent = intent_mask.sum()
    num_emotion = emotion_mask.sum()
    
    print(f"Train Dataset -> Intent samples: {num_intent}, Emotion samples: {num_emotion}")
    
    # Weight is inversely proportional to frequency
    weight_intent = 1.0 / num_intent if num_intent > 0 else 0
    weight_emotion = 1.0 / num_emotion if num_emotion > 0 else 0
    
    sample_weights = []
    for idx in range(len(df_train)):
        if df_train.iloc[idx]['intent_label'] != -1:
            sample_weights.append(weight_intent)
        else:
            sample_weights.append(weight_emotion)
            
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader
