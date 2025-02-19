import torch
from torch.utils.data import TensorDataset
from models.MinimalTransformer import MinimalTransformer
from data.generator import generate
from trainer import Trainer
from pathlib import Path
import json
from datetime import datetime

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def create_datasets(N=10000, max_int=50, device=None):
    """Create training and validation datasets for base model"""
    # Training data (odd sums only)
    train_sequences = generate(N=N, odd_even_mix=1.0, max_int=max_int)  # All odd
    x_train = torch.tensor([[seq[0], 1, seq[2], 2] for seq in train_sequences])
    y_train = torch.tensor([seq[4] for seq in train_sequences])
    
    if device:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
    
    # Validation data
    val_sequences = generate(N=100, odd_even_mix=1.0, max_int=max_int)  # All odd
    x_val = torch.tensor([[seq[0], 1, seq[2], 2] for seq in val_sequences])
    y_val = torch.tensor([seq[4] for seq in val_sequences])
    
    if device:
        x_val = x_val.to(device)
        y_val = y_val.to(device)
    
    return (
        TensorDataset(x_train, y_train),
        TensorDataset(x_val, y_val)
    )

def evaluate_model(model, val_data):
    """Evaluate model on validation set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in val_data:
            x = x.unsqueeze(0)  # Add batch dimension
            pred = model.predict(x).cpu()
            correct += (pred.item() == y.item())
            total += 1
    
    return (correct / total) * 100

def create_base_model(
    max_int=50,
    embed_dim=64,
    target_epochs=10,
    batch_size=32,
    learning_rate=0.001,
    target_accuracy=80.0,
    model_dir="models/base",
    seed=42
):
    """Create and save base model trained on odd sums"""
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create save directory
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    train_data, val_data = create_datasets(max_int=max_int, device=device)
    print(f"Created datasets - Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Initialize model
    model = MinimalTransformer(max_int=max_int, embed_dim=embed_dim, device=device)
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        train_data=train_data,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment_name="base_model"
    )
    
    print("\nStarting base model training...")
    best_val_acc = 0
    training_history = []
    
    for epoch in range(target_epochs):
        # Train
        loss, train_acc = trainer.train_epoch(epoch)
        
        # Evaluate
        val_acc = evaluate_model(model, val_data)
        
        # Log metrics
        print(f"\nEpoch {epoch + 1}/{target_epochs}")
        print(f"Train Loss: {loss:.4f}, Train Accuracy: {train_acc:.1f}%")
        print(f"Validation Accuracy: {val_acc:.1f}%")
        
        trainer.writer.add_scalar('Base/Train_Loss', loss, epoch)
        trainer.writer.add_scalar('Base/Train_Accuracy', train_acc, epoch)
        trainer.writer.add_scalar('Base/Val_Accuracy', val_acc, epoch)
        
        # Track history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Save model checkpoint
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_path = model_dir / f"base_model_{timestamp}.pt"
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc,
                'train_accuracy': train_acc,
                'loss': loss,
                'hyperparameters': {
                    'max_int': max_int,
                    'embed_dim': embed_dim,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'seed': seed
                }
            }
            
            torch.save(checkpoint, model_path)
            print(f"Saved new best model to {model_path}")
            
            # Save training history
            history_path = model_dir / f"base_model_{timestamp}_history.json"
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=4)
            
            # Save model info
            info_path = model_dir / f"base_model_{timestamp}_info.json"
            model_info = {
                'timestamp': timestamp,
                'final_metrics': {
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'loss': loss
                },
                'hyperparameters': checkpoint['hyperparameters'],
                'training_epochs': epoch + 1,
                'model_path': str(model_path),
                'history_path': str(history_path)
            }
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
    
    print("\nBase model training complete!")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    
    if best_val_acc < target_accuracy:
        print(f"Warning: Model did not reach target accuracy of {target_accuracy}%")
    
    trainer.close()
    return model_path, info_path

if __name__ == "__main__":
    model_path, info_path = create_base_model()
    print(f"\nFinal model saved to: {model_path}")
    print(f"Model info saved to: {info_path}")
