import torch
from torch.utils.data import TensorDataset, DataLoader
from models.BaseTransformerModel import BaseTransformerModel
from data.generator import generate
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
from pathlib import Path

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_base_model(args):
    """Train and save base model on odd sums"""
    device = get_device()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Setup directories
    output_dir = Path(args.output_dir) / "base"
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    for dir in [output_dir, models_dir, metrics_dir]:
        dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    train_sequences = generate(N=10000, odd_even_mix=1.0, max_int=50)
    x_train = torch.tensor([[seq[0], 1, seq[2], 2] for seq in train_sequences])
    y_train = torch.tensor([seq[4] for seq in train_sequences])
    train_data = TensorDataset(x_train.to(device), y_train.to(device))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Initialize model and training
    model = BaseTransformerModel(max_int=50, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(f'runs/base_{timestamp}')
    
    # Training history
    history = []
    best_acc = 0
    best_model_path = models_dir / 'best_model.pt'
    
    print("\nTraining base model...")
    for epoch in range(10):
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
        
        # Epoch metrics
        avg_loss = train_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        print(f"Epoch {epoch + 1}:")
        print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.1f}%")
        
        writer.add_scalar('Loss', avg_loss, epoch)
        writer.add_scalar('Accuracy', accuracy, epoch)
        
        # Save metrics
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': accuracy
        })
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'loss': avg_loss
            }, best_model_path)
    
    # Save final metrics
    metrics_path = metrics_dir / 'training_history.json'
    with open(metrics_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    writer.close()
    return best_model_path
