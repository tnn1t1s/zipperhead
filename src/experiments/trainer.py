from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from pathlib import Path

class Trainer:
    """Base trainer class for experiments"""
    def __init__(
        self,
        model,
        train_data,
        batch_size=32,
        learning_rate=0.001,
        experiment_name="base",
        save_dir="checkpoints"
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Setup logging and checkpointing
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = f"{experiment_name}_{timestamp}"
        self.writer = SummaryWriter(f'runs/{self.run_name}')
        
        # Create checkpoint directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
            
            # Log batch metrics
            if batch_idx % 50 == 0:
                batch_loss = loss.item()
                batch_acc = 100. * (pred == batch_y).sum().item() / batch_y.size(0)
                step = epoch * len(self.train_loader) + batch_idx
                
                self.writer.add_scalar('Batch/Loss', batch_loss, step)
                self.writer.add_scalar('Batch/Accuracy', batch_acc, step)
                
                print(f"Batch {batch_idx}: Loss = {batch_loss:.4f}, Acc = {batch_acc:.1f}%")
        
        # Epoch metrics
        avg_loss = train_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        # Log epoch metrics
        self.writer.add_scalar('Epoch/Loss', avg_loss, epoch)
        self.writer.add_scalar('Epoch/Accuracy', epoch_acc, epoch)
        
        return avg_loss, epoch_acc
    
    def evaluate(self, eval_data):
        """Evaluate model on provided data"""
        self.model.eval()
        correct = 0
        total = len(eval_data)
        
        with torch.no_grad():
            for x, y in eval_data:
                x = x.unsqueeze(0).to(self.device)  # Add batch dimension
                pred = self.model.predict(x).cpu().item()
                correct += (pred == y)
        
        accuracy = (correct / total) * 100
        return accuracy
    
    def save_checkpoint(self, epoch, metrics=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics or {}
        }
        path = self.save_dir / f"{self.run_name}_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        return path
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
    
    def close(self):
        """Clean up resources"""
        self.writer.close()
