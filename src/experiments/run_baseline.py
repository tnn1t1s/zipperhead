import torch
from models.BaseTransformerModel import BaseTransformerModel
from data.generator import generate
from trainer import Trainer
from torch.utils.data import TensorDataset

def get_device():
    """Get the appropriate device for training"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def create_datasets(N=10000, max_int=50, device=None):
    """Create training and validation datasets"""
    # Training data (odd sums)
    train_sequences = generate(N=N, odd_even_mix=1.0, max_int=max_int)
    x_train = torch.tensor([[seq[0], 1, seq[2], 2] for seq in train_sequences])
    y_train = torch.tensor([seq[4] for seq in train_sequences])
    
    if device:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
    
    # Validation data
    val_sequences = generate(N=100, odd_even_mix=1.0, max_int=max_int)
    # Convert validation sequences using same token mapping as training
    val_data = [
        (
            torch.tensor([seq[0], 1, seq[2], 2], device=device),
            torch.tensor(seq[4], device=device)
        )
        for seq in val_sequences
    ]
    
    return TensorDataset(x_train, y_train), val_data

def run_baseline_experiment(
    max_int=50,
    embed_dim=64,
    num_epochs=12,
    batch_size=32,
    learning_rate=0.001
):
    """Run baseline training experiment"""
    print("Starting baseline transformer experiment...")
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Create datasets
    train_data, val_data = create_datasets(max_int=max_int, device=device)
    print(f"Created datasets - Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Initialize model
    model = BaseTransformerModel(max_int=max_int, embed_dim=embed_dim, device=device)
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        train_data=train_data,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment_name="baseline"
    )
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        loss, acc = trainer.train_epoch(epoch)
        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Accuracy = {acc:.1f}%")
        
        # Evaluate on validation set
        val_acc = trainer.evaluate(val_data)
        trainer.writer.add_scalar('Validation/Accuracy', val_acc, epoch)
        print(f"Validation Accuracy: {val_acc:.1f}%")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = {'train_acc': acc, 'val_acc': val_acc, 'loss': loss}
            path = trainer.save_checkpoint(epoch + 1, metrics)
            print(f"Saved checkpoint to {path}")
    
    print("\nTraining complete!")
    trainer.close()

if __name__ == "__main__":
    run_baseline_experiment()
