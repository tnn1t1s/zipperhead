import torch
from torch.utils.data import TensorDataset, ConcatDataset
from models.BaseTransformerModel import BaseTransformerModel
from data.generator import generate
from trainer import Trainer
import json
from pathlib import Path
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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
    
    hp = info['hyperparameters']
    model = BaseTransformerModel(
        max_int=hp['max_int'],
        embed_dim=hp['embed_dim'],
        device=get_device()
    )
    
    checkpoint = torch.load(info['model_path'], map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, hp, info['final_metrics']

def create_datasets(N=10000, max_int=50, device=None):
    """Create datasets for GRPO training"""
    # Test data
    test_odd = generate(N=100, odd_even_mix=1.0, max_int=max_int)
    test_even = generate(N=100, odd_even_mix=0.0, max_int=max_int)
    
    test_data = {
        'odd': [(torch.tensor([seq[0], 1, seq[2], 2], device=device),
                torch.tensor(seq[4], device=device)) for seq in test_odd],
        'even': [(torch.tensor([seq[0], 1, seq[2], 2], device=device),
                 torch.tensor(seq[4], device=device)) for seq in test_even]
    }
    
    # Training data (mix of odd and even)
    train_odd = generate(N//2, odd_even_mix=1.0, max_int=max_int)
    train_even = generate(N//2, odd_even_mix=0.0, max_int=max_int)
    
    x_train = torch.tensor([[seq[0], 1, seq[2], 2] for seq in train_odd + train_even])
    y_train = torch.tensor([seq[4] for seq in train_odd + train_even])
    is_even = torch.tensor([n[4] % 2 == 0 for n in train_odd + train_even])
    
    if device:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        is_even = is_even.to(device)
    
    return TensorDataset(x_train, y_train, is_even), test_data

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

class GRPOTrainer:
    def __init__(self, model, train_data, batch_size=32, learning_rate=0.001):
        self.model = model
        self.device = next(model.parameters()).device
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    def train_epoch(self):
        self.model.train()
        total_reward = 0
        correct_odd = correct_even = total_odd = total_even = 0
        
        for batch_x, batch_y, batch_is_even in self.train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(batch_x)
            
            # Standard cross-entropy loss first
            base_loss = self.criterion(logits, batch_y).mean()
            
            # Compute predictions
            pred = logits.argmax(dim=1)
            correct = (pred == batch_y)
            
            # Simple reward for correct even sums
            rewards = torch.zeros_like(batch_y, dtype=torch.float)
            rewards[batch_is_even & correct] = 1.0
            
            # Final loss combines standard loss with rewards
            loss = base_loss - 0.1 * rewards.mean()  # Small reward factor
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_reward += rewards.sum().item()
            
            # Track accuracy by type
            mask_odd = ~batch_is_even
            mask_even = batch_is_even
            
            correct_odd += (correct & mask_odd).sum().item()
            correct_even += (correct & mask_even).sum().item()
            total_odd += mask_odd.sum().item()
            total_even += mask_even.sum().item()
        
        # Calculate epoch metrics
        avg_reward = total_reward / len(self.train_loader.dataset)
        odd_acc = (correct_odd / total_odd * 100) if total_odd > 0 else 0
        even_acc = (correct_even / total_even * 100) if total_even > 0 else 0
        
        return avg_reward, odd_acc, even_acc

def run_grpo_experiment(
    base_model_info: str,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    experiment_dir: str = "experiments/grpo",
    seed: int = 42
):
    """Run GRPO experiment starting from pre-trained base model"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load base model
    print(f"\nLoading base model from {base_model_info}")
    model, hyperparams, base_metrics = load_base_model(base_model_info)
    print(f"Base model metrics: {base_metrics}")
    
    # Create datasets
    train_data, test_data = create_datasets(max_int=hyperparams['max_int'], device=device)
    
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
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'seed': seed
        },
        'initial_accuracy': initial_acc
    }
    
    config_path = experiment_dir / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Setup tensorboard
    writer = SummaryWriter(f'runs/grpo_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    
    # GRPO Training
    print("\nStarting GRPO training...")
    trainer = GRPOTrainer(
        model=model,
        train_data=train_data,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    metrics_history = []
    
    for epoch in range(num_epochs):
        # Train
        reward, odd_acc, even_acc = trainer.train_epoch()
        
        # Evaluate
        test_acc = evaluate_split_accuracy(model, test_data)
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"Average Reward: {reward:.4f}")
        print(f"Train Accuracy - Odd: {odd_acc:.1f}%, Even: {even_acc:.1f}%")
        print(f"Test Accuracy - Odd: {test_acc['odd']:.1f}%, Even: {test_acc['even']:.1f}%")
        
        # Log metrics
        writer.add_scalar('GRPO/Reward', reward, epoch)
        writer.add_scalar('GRPO/Train_Odd_Accuracy', odd_acc, epoch)
        writer.add_scalar('GRPO/Train_Even_Accuracy', even_acc, epoch)
        writer.add_scalar('GRPO/Test_Odd_Accuracy', test_acc['odd'], epoch)
        writer.add_scalar('GRPO/Test_Even_Accuracy', test_acc['even'], epoch)
        
        # Track metrics
        metrics = {
            'epoch': epoch + 1,
            'reward': reward,
            'train_odd_acc': odd_acc,
            'train_even_acc': even_acc,
            'test_odd_acc': test_acc['odd'],
            'test_even_acc': test_acc['even']
        }
        metrics_history.append(metrics)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = experiment_dir / f'grpo_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'metrics': metrics
            }, checkpoint_path)
    
    # Save final metrics
    metrics_path = experiment_dir / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=4)
    
    print("\nGRPO Training complete!")
    writer.close()
    return metrics_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GRPO experiment')
    parser.add_argument('--base-model', required=True,
                      help='Path to base model info JSON')
    parser.add_argument('--output-dir', default='experiments/grpo',
                      help='Directory to save experiment results')
    args = parser.parse_args()
    
    metrics_path = run_grpo_experiment(
        base_model_info=args.base_model,
        experiment_dir=args.output_dir
    )
    print(f"\nExperiment metrics saved to: {metrics_path}")
