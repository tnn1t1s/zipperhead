import torch
from torch.utils.data import TensorDataset, ConcatDataset
from models.BaseTransformerModel import BaseTransformerModel
from data.generator import generate
from trainer import Trainer
import numpy as np
from pathlib import Path

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def create_initial_datasets(N=10000, max_int=50, device=None):
    """Create initial odd-sum training data and test sets"""
    # Training data (odd sums only)
    train_sequences = generate(N=N, odd_even_mix=1.0, max_int=max_int)  # All odd
    x_train = torch.tensor([[seq[0], 1, seq[2], 2] for seq in train_sequences])
    y_train = torch.tensor([seq[4] for seq in train_sequences])
    
    if device:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
    
    # Test data (mix of odd and even)
    test_odd = generate(N=100, odd_even_mix=1.0, max_int=max_int)  # All odd
    test_even = generate(N=100, odd_even_mix=0.0, max_int=max_int)  # All even
    
    test_data = {
        'odd': [(torch.tensor([seq[0], 1, seq[2], 2], device=device),
                torch.tensor(seq[4], device=device)) for seq in test_odd],
        'even': [(torch.tensor([seq[0], 1, seq[2], 2], device=device),
                 torch.tensor(seq[4], device=device)) for seq in test_even]
    }
    
    return TensorDataset(x_train, y_train), test_data

def create_even_batches(N=10000, max_int=50, num_batches=10, device=None):
    """Create batches of even-sum data for progressive introduction"""
    sequences = generate(N=N, odd_even_mix=0.0, max_int=max_int)  # All even
    x_data = torch.tensor([[seq[0], 1, seq[2], 2] for seq in sequences])
    y_data = torch.tensor([seq[4] for seq in sequences])
    
    if device:
        x_data = x_data.to(device)
        y_data = y_data.to(device)
    
    # Split into batches
    indices = np.random.permutation(len(sequences))
    batch_size = len(sequences) // num_batches
    batches = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < num_batches - 1 else len(sequences)
        batch_indices = indices[start_idx:end_idx]
        batch_x = x_data[batch_indices]
        batch_y = y_data[batch_indices]
        batches.append(TensorDataset(batch_x, batch_y))
    
    return batches

def evaluate_split_accuracy(model, test_data):
    """Evaluate model separately on odd and even sums"""
    results = {}
    for split_name, split_data in test_data.items():
        correct = 0
        total = len(split_data)
        
        model.eval()
        with torch.no_grad():
            for x, y in split_data:
                x = x.unsqueeze(0)  # Add batch dimension
                pred = model.predict(x).cpu()
                correct += (pred.item() == y.item())
        
        accuracy = (correct / total) * 100
        results[split_name] = accuracy
    
    return results

def run_sft_experiment(
    max_int=50,
    embed_dim=64,
    initial_epochs=10,
    fine_tune_epochs=5,
    num_batches=10,
    batch_size=32,
    learning_rate=0.001,
    checkpoint_dir="checkpoints"
):
    """Run SFT (Sparse-to-Full Training) experiment"""
    print("Starting SFT experiment...")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create datasets
    train_data, test_data = create_initial_datasets(max_int=max_int, device=device)
    even_batches = create_even_batches(max_int=max_int, num_batches=num_batches, device=device)
    print(f"Created datasets - Initial train size: {len(train_data)}")
    print(f"Created {num_batches} even-sum batches")
    
    # Initialize model and train on odd sums
    model = BaseTransformerModel(max_int=max_int, embed_dim=embed_dim, device=device)
    
    # Initial training on odd sums
    trainer = Trainer(
        model=model,
        train_data=train_data,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment_name="sft_initial"
    )
    
    print("\nStarting initial training on odd sums...")
    for epoch in range(initial_epochs):
        loss, acc = trainer.train_epoch(epoch)
        split_acc = evaluate_split_accuracy(model, test_data)
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"Training Loss: {loss:.4f}, Accuracy: {acc:.1f}%")
        print(f"Test Accuracy - Odd: {split_acc['odd']:.1f}%, Even: {split_acc['even']:.1f}%")
        
        # Log metrics
        trainer.writer.add_scalar('Initial/Train_Accuracy', acc, epoch)
        trainer.writer.add_scalar('Initial/Odd_Accuracy', split_acc['odd'], epoch)
        trainer.writer.add_scalar('Initial/Even_Accuracy', split_acc['even'], epoch)
    
    # Save initial model
    initial_path = trainer.save_checkpoint(initial_epochs, 
                                         {'final_acc': acc, 'split_acc': split_acc})
    print(f"\nSaved initial model to {initial_path}")
    
    # SFT Fine-tuning
    print("\nStarting SFT fine-tuning...")
    current_train_data = train_data
    
    for batch_idx, even_batch in enumerate(even_batches):
        print(f"\nIntroducing even-sum batch {batch_idx + 1}/{num_batches}")
        
        # Combine current training data with new even batch
        combined_data = ConcatDataset([current_train_data, even_batch])
        
        # Create new trainer for this phase
        trainer = Trainer(
            model=model,
            train_data=combined_data,
            batch_size=batch_size,
            learning_rate=learning_rate,
            experiment_name=f"sft_batch_{batch_idx + 1}"
        )
        
        # Fine-tune
        for epoch in range(fine_tune_epochs):
            loss, acc = trainer.train_epoch(epoch)
            split_acc = evaluate_split_accuracy(model, test_data)
            
            global_epoch = initial_epochs + batch_idx * fine_tune_epochs + epoch
            
            print(f"Epoch {epoch + 1}:")
            print(f"Training Loss: {loss:.4f}, Accuracy: {acc:.1f}%")
            print(f"Test Accuracy - Odd: {split_acc['odd']:.1f}%, Even: {split_acc['even']:.1f}%")
            
            # Log metrics
            trainer.writer.add_scalar('Finetune/Train_Accuracy', acc, global_epoch)
            trainer.writer.add_scalar('Finetune/Odd_Accuracy', split_acc['odd'], global_epoch)
            trainer.writer.add_scalar('Finetune/Even_Accuracy', split_acc['even'], global_epoch)
        
        # Update current training data for next batch
        current_train_data = combined_data
        
        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(
            global_epoch,
            {
                'batch_idx': batch_idx,
                'train_acc': acc,
                'split_acc': split_acc
            }
        )
        print(f"Saved checkpoint to {checkpoint_path}")
    
    print("\nSFT Training complete!")
    trainer.close()

if __name__ == "__main__":
    run_sft_experiment()
