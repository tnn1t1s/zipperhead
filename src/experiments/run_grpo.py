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
    
    if device:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
    
    return TensorDataset(x_train, y_train), test_data

def compute_reward(pred, target, is_even):
    """Compute reward based on prediction accuracy and task type"""
    correct = (pred == target)
    if is_even:
        return 2.0 if correct else -1.0  # Higher reward/penalty for even sums
    return 1.0 if correct else -0.5      # Lower reward/penalty for odd sums

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

# GRPO Training
    print("\nStarting GRPO training...")
    trainer = Trainer(
        model=model,
        train_data=train_data,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment_name="grpo_training"
    )
    
    metrics_history = []
    best_even_acc = initial_acc['even']
    
    for epoch in range(num_epochs):
        model.train()
        epoch_reward = 0
        correct_even = 0
        total_even = 0
        correct_odd = 0
        total_odd = 0
        
        for batch_x, batch_y in trainer.train_loader:
            trainer.optimizer.zero_grad()
            
            # Forward pass
            output = model(batch_x)
            pred = output.argmax(dim=1)
            
            # Compute rewards
            batch_rewards = []
            for i, (p, t) in enumerate(zip(pred, batch_y)):
                is_even = (t.item() % 2 == 0)
                reward = compute_reward(p.item(), t.item(), is_even)
                batch_rewards.append(reward)
                
                if is_even:
                    correct_even += (p == t).item()
                    total_even += 1
                else:
                    correct_odd += (p == t).item()
                    total_odd += 1
            
            # Use rewards to scale the loss
            rewards = torch.tensor(batch_rewards, device=device)
            loss = -torch.mean(rewards * output.gather(1, batch_y.unsqueeze(1)).squeeze())
            
            # Backward pass
            loss.backward()
            trainer.optimizer.step()
            
            epoch_reward += sum(batch_rewards)
        
        # Evaluate
        split_acc = evaluate_split_accuracy(model, test_data)
        epoch_reward = epoch_reward / len(trainer.train_loader)
        
        # Calculate accuracies
        odd_acc = (correct_odd / total_odd * 100) if total_odd > 0 else 0
        even_acc = (correct_even / total_even * 100) if total_even > 0 else 0
        
        metrics = {
            'epoch': epoch + 1,
            'reward': epoch_reward,
            'train_odd_acc': odd_acc,
            'train_even_acc': even_acc,
            'test_odd_acc': split_acc['odd'],
            'test_even_acc': split_acc['even']
        }
        metrics_history.append(metrics)
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"Average Reward: {epoch_reward:.4f}")
        print(f"Train Accuracy - Odd: {odd_acc:.1f}%, Even: {even_acc:.1f}%")
        print(f"Test Accuracy - Odd: {split_acc['odd']:.1f}%, Even: {split_acc['even']:.1f}%")
        
        # Log metrics
        trainer.writer.add_scalar('GRPO/Reward', epoch_reward, epoch)
        trainer.writer.add_scalar('GRPO/Train_Odd_Accuracy', odd_acc, epoch)
        trainer.writer.add_scalar('GRPO/Train_Even_Accuracy', even_acc, epoch)
        trainer.writer.add_scalar('GRPO/Test_Odd_Accuracy', split_acc['odd'], epoch)
        trainer.writer.add_scalar('GRPO/Test_Even_Accuracy', split_acc['even'], epoch)
        
        # Save if we have a new best even accuracy
        if split_acc['even'] > best_even_acc:
            best_even_acc = split_acc['even']
            checkpoint_path = experiment_dir / f'grpo_best.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'metrics': metrics
            }, checkpoint_path)
            print(f"Saved new best model with even accuracy: {best_even_acc:.1f}%")
        
        # Save periodic checkpoint
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
    trainer.close()
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
