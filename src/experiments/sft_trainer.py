import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from models.BaseTransformerModel import BaseTransformerModel
from data.generator import generate
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
from pathlib import Path
from base_trainer import get_device

def evaluate_model(model, odd_data, even_data):
    """Evaluate model on odd and even sums separately"""
    model.eval()
    results = {}
    
    for name, data in [("odd", odd_data), ("even", even_data)]:
        correct = 0
        total = len(data)
        with torch.no_grad():
            for x, y in data:
                x = x.unsqueeze(0).to(model.device)
                pred = model.predict(x).cpu()
                correct += (pred.item() == y)
        results[name] = 100. * correct / total
    
    return results

def run_sft_experiment(args):
    """Run SFT experiment using pre-trained base model"""
    device = get_device()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Setup directories
    output_dir = Path(args.output_dir) / "sft"
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
    
    # Create initial odd sum data
    odd_sequences = generate(N=10000, odd_even_mix=1.0, max_int=50)
    x_odd = torch.tensor([[seq[0], 1, seq[2], 2] for seq in odd_sequences])
    y_odd = torch.tensor([seq[4] for seq in odd_sequences])
    
    # Generate even sum batches
    even_sequences = generate(N=10000, odd_even_mix=0.0, max_int=50)
    x_even = torch.tensor([[seq[0], 1, seq[2], 2] for seq in even_sequences])
    y_even = torch.tensor([seq[4] for seq in even_sequences])
    
    # Setup tensorboard
    writer = SummaryWriter(f'runs/sft_{timestamp}')
    history = []
    
    # Initial evaluation
    accuracies = evaluate_model(model, eval_odd_data, eval_even_data)
    print("\nInitial performance:")
    print(f"Odd sums: {accuracies['odd']:.1f}%")
    print(f"Even sums: {accuracies['even']:.1f}%")
    
    # Training loop
    batch_size = len(even_sequences) // 10
    current_even_data = []
    
    for batch_idx in range(10):
        print(f"\nProcessing batch {batch_idx + 1}/10")
        
        # Add new batch of even sums
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        current_even_data.append((x_even[start_idx:end_idx], y_even[start_idx:end_idx]))
        
        # Combine all current data
        x_batch = torch.cat([x_odd] + [x for x, _ in current_even_data])
        y_batch = torch.cat([y_odd] + [y for _, y in current_even_data])
        
        train_data = TensorDataset(x_batch.to(device), y_batch.to(device))
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Train on current data
        for epoch in range(5):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                pred = output.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
                train_loss += loss.item()
            
            avg_loss = train_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Evaluate
            accuracies = evaluate_model(model, eval_odd_data, eval_even_data)
            
            print(f"Epoch {epoch + 1}:")
            print(f"Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.1f}%")
            print(f"Odd Accuracy: {accuracies['odd']:.1f}%")
            print(f"Even Accuracy: {accuracies['even']:.1f}%")
            
            # Log metrics
            writer.add_scalar(f'Batch_{batch_idx+1}/Loss', avg_loss, epoch)
            writer.add_scalar(f'Batch_{batch_idx+1}/Train_Accuracy', train_acc, epoch)
            writer.add_scalar(f'Batch_{batch_idx+1}/Odd_Accuracy', accuracies['odd'], epoch)
            writer.add_scalar(f'Batch_{batch_idx+1}/Even_Accuracy', accuracies['even'], epoch)
        
        # Save batch checkpoint
        torch.save({
            'batch': batch_idx + 1,
            'model_state_dict': model.state_dict(),
            'accuracies': accuracies
        }, models_dir / f'batch_{batch_idx+1}_model.pt')
        
        # Save metrics
        history.append({
            'batch': batch_idx + 1,
            'loss': avg_loss,
            'train_accuracy': train_acc,
            'odd_accuracy': accuracies['odd'],
            'even_accuracy': accuracies['even']
        })
    
    # Save final metrics
    with open(metrics_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    writer.close()
