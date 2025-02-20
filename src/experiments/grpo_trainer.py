import torch
from torch.utils.data import TensorDataset, DataLoader
from models.BaseTransformerModel import BaseTransformerModel
from data.generator import generate
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
from pathlib import Path
from base_trainer import get_device
from sft_trainer import evaluate_model

def run_grpo_experiment(args):
    """Run GRPO experiment using pre-trained base model"""
    device = get_device()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Setup directories
    output_dir = Path(args.output_dir) / "grpo"
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    for dir in [output_dir, models_dir, metrics_dir]:
        dir.mkdir(parents=True, exist_ok=True)
    
    # Load base model
    model = BaseTransformerModel(max_int=50, device=device)
    checkpoint = torch.load(args.base_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluation datasets
    eval_odd = generate(N=100, odd_even_mix=1.0, max_int=50)
    eval_even = generate(N=100, odd_even_mix=0.0, max_int=50)
    
    eval_odd_data = [(torch.tensor([seq[0], 1, seq[2], 2]), seq[4]) for seq in eval_odd]
    eval_even_data = [(torch.tensor([seq[0], 1, seq[2], 2]), seq[4]) for seq in eval_even]
    
    # Create mixed training dataset
    odd_sequences = generate(N=5000, odd_even_mix=1.0, max_int=50)
    even_sequences = generate(N=5000, odd_even_mix=0.0, max_int=50)
    
    x_data = torch.tensor([[seq[0], 1, seq[2], 2] for seq in odd_sequences + even_sequences])
    y_data = torch.tensor([seq[4] for seq in odd_sequences + even_sequences])
    is_even = torch.tensor([num % 2 == 0 for num in y_data])
    
    train_data = TensorDataset(x_data.to(device), y_data.to(device), is_even.to(device))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    writer = SummaryWriter(f'runs/grpo_{timestamp}')
    history = []
    
    # Initial evaluation
    accuracies = evaluate_model(model, eval_odd_data, eval_even_data)
    print("\nInitial performance:")
    print(f"Odd sums: {accuracies['odd']:.1f}%")
    print(f"Even sums: {accuracies['even']:.1f}%")
    
    # Training loop
    for epoch in range(50):
        model.train()
        total_reward = 0
        correct_odd = correct_even = total_odd = total_even = 0
        epoch_loss = 0
        
        for batch_x, batch_y, batch_is_even in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch_x)
            
            # Standard cross-entropy loss
            base_loss = criterion(logits, batch_y).mean()
            
            # Compute predictions and rewards
            pred = logits.argmax(dim=1)
            correct = (pred == batch_y)
            
            # Simple reward for correct even sums
            rewards = torch.zeros_like(batch_y, dtype=torch.float)
            rewards[batch_is_even & correct] = 1.0
            
            # Combined loss
            loss = base_loss - 0.1 * rewards.mean()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            total_reward += rewards.sum().item()
            
            mask_odd = ~batch_is_even
            mask_even = batch_is_even
            
            correct_odd += (correct & mask_odd).sum().item()
            correct_even += (correct & mask_even).sum().item()
            total_odd += mask_odd.sum().item()
            total_even += mask_even.sum().item()
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        avg_reward = total_reward / len(train_loader.dataset)
        train_odd_acc = (correct_odd / total_odd * 100) if total_odd > 0 else 0
        train_even_acc = (correct_even / total_even * 100) if total_even > 0 else 0
        
        # Evaluate
        accuracies = evaluate_model(model, eval_odd_data, eval_even_data)
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")
        print(f"Train - Odd: {train_odd_acc:.1f}%, Even: {train_even_acc:.1f}%")
        print(f"Test - Odd: {accuracies['odd']:.1f}%, Even: {accuracies['even']:.1f}%")
        
        # Log metrics
        writer.add_scalar('Loss', avg_loss, epoch)
        writer.add_scalar('Reward', avg_reward, epoch)
        writer.add_scalar('Train/Odd_Accuracy', train_odd_acc, epoch)
        writer.add_scalar('Train/Even_Accuracy', train_even_acc, epoch)
        writer.add_scalar('Test/Odd_Accuracy', accuracies['odd'], epoch)
        writer.add_scalar('Test/Even_Accuracy', accuracies['even'], epoch)
        
        # Save metrics
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'reward': avg_reward,
            'train_odd_acc': train_odd_acc,
            'train_even_acc': train_even_acc,
            'test_odd_acc': accuracies['odd'],
            'test_even_acc': accuracies['even']
        })
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'metrics': {
                    'loss': avg_loss,
                    'reward': avg_reward,
                    'accuracies': accuracies
                }
            }, models_dir / f'epoch_{epoch+1}_model.pt')
    
    # Save final metrics
    with open(metrics_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    writer.close()
