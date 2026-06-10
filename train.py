import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model import MultiTaskRoBERTa
from dataset import get_dataloaders
import time
import csv

def train():
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 4
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Preparing dataloaders...")
    train_loader, val_loader = get_dataloaders(
        "data/custom_mtl_dataset_csv/train.csv",
        "data/custom_mtl_dataset_csv/validation.csv",
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH
    )
    
    print("Initializing model...")
    model = MultiTaskRoBERTa(num_intents=151, num_emotions=6)
    model.to(device)
    
    # Loss functions with ignore_index=-1
    criterion_intent = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_emotion = nn.CrossEntropyLoss(ignore_index=-1)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    best_val_loss = float('inf')
    os.makedirs("models", exist_ok=True)
    
    start_epoch = 0
    checkpoint_path = "models/last_checkpoint.pt"
    csv_log_path = "models/training_logs.csv"
    val_log_path = "models/val_logs.csv"
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resuming from epoch {start_epoch}")
    else:
        # Initialize CSV loggers
        with open(csv_log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'intent_loss', 'emotion_loss', 'total_loss'])
        with open(val_log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'val_loss'])
        
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        model.train()
        total_train_loss = 0
        
        start_time = time.time()
        for step, batch in enumerate(train_loader):
            b_input_ids = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)
            b_intent_labels = batch['intent_label'].to(device)
            b_emotion_labels = batch['emotion_label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with AMP
            if scaler:
                with torch.amp.autocast('cuda'):
                    intent_logits, emotion_logits = model(b_input_ids, b_attention_mask)
                    loss_intent = criterion_intent(intent_logits, b_intent_labels)
                    loss_emotion = criterion_emotion(emotion_logits, b_emotion_labels)
                    loss = loss_intent + loss_emotion
            else:
                intent_logits, emotion_logits = model(b_input_ids, b_attention_mask)
                loss_intent = criterion_intent(intent_logits, b_intent_labels)
                loss_emotion = criterion_emotion(emotion_logits, b_emotion_labels)
                loss = loss_intent + loss_emotion
                
            total_train_loss += loss.item()
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            scheduler.step()
            
            # Log to CSV
            with open(csv_log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, step+1, loss_intent.item(), loss_emotion.item(), loss.item()])
            
            if (step + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Batch {step+1}/{len(train_loader)} - Loss: {loss.item():.4f} - Time: {elapsed:.2f}s")
                start_time = time.time()
                
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        print("Running Validation...")
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                b_input_ids = batch['input_ids'].to(device)
                b_attention_mask = batch['attention_mask'].to(device)
                b_intent_labels = batch['intent_label'].to(device)
                b_emotion_labels = batch['emotion_label'].to(device)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        intent_logits, emotion_logits = model(b_input_ids, b_attention_mask)
                        loss_intent = criterion_intent(intent_logits, b_intent_labels)
                        loss_emotion = criterion_emotion(emotion_logits, b_emotion_labels)
                        loss = loss_intent + loss_emotion
                else:
                    intent_logits, emotion_logits = model(b_input_ids, b_attention_mask)
                    loss_intent = criterion_intent(intent_logits, b_intent_labels)
                    loss_emotion = criterion_emotion(emotion_logits, b_emotion_labels)
                    loss = loss_intent + loss_emotion
                    
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("New best model found! Saving...")
            torch.save(model.state_dict(), "models/best_model.pt")
        
        # Log validation loss
        with open(val_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_val_loss])
            
        # Save last checkpoint for resuming
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'scaler_state_dict': scaler.state_dict() if scaler else None
        }, checkpoint_path)

if __name__ == "__main__":
    train()
