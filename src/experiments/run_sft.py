import torch
from torch.utils.data import TensorDataset, ConcatDataset
from models.MinimalTransformer import MinimalTransformer
from data.generator import generate
from trainer import Trainer
import json
from pathlib import Path
import argparse

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_base_model(model_info_path):
    """Load base model and its configuration"""
    with open(model_info_path, 'r') as f:
        info = json.load(f)
    
    # Create model with same hyperparameters
    hp = info['hyperparameters']
    model = MinimalTransformer(
        max_int=hp['max_int'],
        embed_dim=hp['embed_dim'],
        device=get_device()
    )
    
    # Load weights
    checkpoint = torch.load(info['model_path'], map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, hp, info['final_metrics']

def create_datasets(N=10000, max_int=50, device=None):
    """Create test datasets for evaluation"""
    test_odd = generate(N=100, odd_even_mix=1.0, max_int=max_int)
    test_even = generate(N=100, odd_even_mix=0.0, max_int=max_int)
    
    test_data = {
        'odd': [(torch.tensor([seq[0], 1, seq[2], 2], device=device),
                torch.tensor(seq[4], device=device)) for seq in test_odd],
        'even': [(torch.tensor([seq[0], 1, seq[2], 2], device=device),
                 torch.tensor(seq[4], device=device)) for seq in test_even]
    }
    
    return test_data

def create_even_batches(N=10000, max_int=50, num_batches=10, device=None):
    """Create batches of even-sum data for progressive introduction"""
    sequences = generate(N=N, odd_even_mix=0.0, max_int=max_int)  # All even
    x_data = torch.tensor([[seq[0], 1, seq[2], 2] for seq in sequences])
    y_data = torch.tensor([seq[4] for seq in sequences])
    
    if device:
        x_data = x_data.to(device)
        y_data = y_data.to(device)
    
    indices = torch.randperm(len(sequences))
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
                x = x.unsqueeze(0)
                pred = model.predict(x).cpu()
                correct += (pred.item() == y.item())
        
        accuracy = (correct / total) * 100
        results[split_name] = accuracy
    
    return results

def run_sft_experiment(
    base_model_info: str,
    fine_tune_epochs: int = 5,
    num_batches: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    experiment_dir: str = "experiments/sft",
    seed: int = 42
):
    """Run SFT experiment starting from pre-trained base model"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load base model
    print(f"\nLoading base model from {base_model_info}")
    model, hyperparams, base_metrics = load_base_model(base_model_info)
    print(f"Base model metrics: {base_metrics}")
    
    # Create test data and even-sum batches
    test_data = create_datasets(max_int=hyperparams['max_int'], device=device)
    even_batches = create_even_batches(
        max_int=hyperparams['max_int'],
        num_batches=num_batches,
        device=device
    )
    
    # Verify base model performance
    initial_acc = evaluate_split_accuracy(model, test_data)
    print("\nInitial model performance:")
    print(f"Odd sum accuracy: {initial_acc['odd']:.1f}%")
    print(f"Even sum accuracy: {initial_acc['even']:.1f}%")
    
    # Create experiment directory
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    config = {
        'base_model_info': base_model_info,
        'hyperparameters': {
            **hyperparams,
            'fine_tune_epochs': fine_tune_epochs,
            'num_batches': num_batches,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'seed': seed
        },
        'initial_accuracy': initial_acc
    }
    
    config_path = experiment_dir / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # SFT Training
    print("\nStarting SFT training...")
    metrics_history = []
    current_data = None
    
    for batch_idx, even_batch in enumerate(even_batches):
        print(f"\nProcessing even-sum batch {batch_idx + 1}/{num_batches}")
        
        # Update training data
        if current_data is None:
            current_data = even_batch
        else:
            current_data = ConcatDataset([current_data, even_batch])
        
        # Create trainer for this phase
        trainer = Trainer(
            model=model,
            train_data=current_data,
            batch_size=batch_size,
            learning_rate=learning_rate,
            experiment_name=f"sft_batch_{batch_idx + 1}"
        )
        
        # Fine-tune
        for epoch in range(fine_tune_epochs):
            loss, acc = trainer.train_epoch(epoch)
            split_acc = evaluate_split_accuracy(model, test_data)
            
            print(f"Epoch {epoch + 1}:")
            print(f"Loss: {loss:.4f}, Accuracy: {acc:.1f}%")
            print(f"Test Accuracy - Odd: {split_acc['odd']:.1f}%, Even: {split_acc['even']:.1f}%")
            
            # Track metrics
            metrics = {
                'batch': batch_idx + 1,
                'epoch': epoch + 1,
                'loss': loss,
                'train_acc': acc,
                'odd_acc': split_acc['odd'],
                'even_acc': split_acc['even']
            }
            metrics_history.append(metrics)
            
            # Log to tensorboard
            global_step = batch_idx * fine_tune_epochs + epoch
            trainer.writer.add_scalar('SFT/Loss', loss, global_step)
            trainer.writer.add_scalar('SFT/Train_Accuracy', acc, global_step)
            trainer.writer.add_scalar('SFT/Odd_Accuracy', split_acc['odd'], global_step)
            trainer.writer.add_scalar('SFT/Even_Accuracy', split_acc['even'], global_step)
        
        # Save batch checkpoint
        checkpoint_path = experiment_dir / f'sft_batch_{batch_idx + 1}.pt'
        torch.save({
            'batch': batch_idx + 1,
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }, checkpoint_path)
    
    # Save final metrics
    metrics_path = experiment_dir / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=4)
    
    print("\nSFT Training complete!")
    trainer.close()
    return metrics_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SFT experiment')
    parser.add_argument('--base-model', required=True,
                      help='Path to base model info JSON')
    parser.add_argument('--output-dir', default='experiments/sft',
                      help='Directory to save experiment results')
    args = parser.parse_args()
    
    metrics_path = run_sft_experiment(
        base_model_info=args.base_model,
        experiment_dir=args.output_dir
    )
    print(f"\nExperiment metrics saved to: {metrics_path}")
