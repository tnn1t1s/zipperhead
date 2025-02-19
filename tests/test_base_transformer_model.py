import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.BaseTransformerModel import BaseTransformerModel
from data.generator import generate

def get_device():
    """Get the appropriate device for training"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def create_training_data(N=10000, max_int=50, device=None):
    """Create training data with odd sums
    
    Args:
        N: Number of sequences to generate
        max_int: Maximum integer value in sequences
        device: Device to place tensors on
    """
    sequences = generate(N=N, odd_even_mix=1.0, max_int=max_int)
    x_data = torch.tensor([[seq[0], 1, seq[2], 2] for seq in sequences])
    y_data = torch.tensor([seq[4] for seq in sequences])
    
    if device is not None:
        x_data = x_data.to(device)
        y_data = y_data.to(device)
    
    return TensorDataset(x_data, y_data)

def evaluate_model(model, test_sequences, device):
    """Evaluate model on test sequences
    
    Args:
        model: The transformer model
        test_sequences: List of test sequences
        device: Device to run evaluation on
    
    Returns:
        float: Accuracy percentage
    """
    model.eval()
    correct = 0
    total = len(test_sequences)
    
    print("\nDetailed test evaluation:")
    with torch.no_grad():
        for seq in test_sequences:
            x_data = torch.tensor([[seq[0], 1, seq[2], 2]], device=device)
            pred = model.predict(x_data).cpu().item()
            correct += (pred == seq[4])
            
            print(f"Input: {seq[0]} + {seq[2]} = {seq[4]}")
            print(f"Predicted: {pred}")
            print(f"{'✓' if pred == seq[4] else '✗'}")
            
            # Show logits for incorrect predictions
            if pred != seq[4]:
                logits = model(x_data)[0]
                top_k = torch.topk(logits, k=3)
                print("Top 3 predictions (value, logit):")
                for value, logit in zip(top_k.indices.cpu(), top_k.values.cpu()):
                    print(f"{value.item()}: {logit.item():.2f}")
    
    accuracy = (correct / total) * 100
    print(f"\nOverall accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy

def test_train_odd_sums():
    """Test transformer model training and evaluation on odd-sum sequences"""
    # Setup
    max_int = 50
    num_epochs = 12
    passing_accuracy = 75
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Create and setup training data
    train_data = create_training_data(N=10000, max_int=max_int, device=device)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    print(f"Training samples: {len(train_data)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Initialize model and training components
    model = BaseTransformerModel(max_int=max_int, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    best_acc = 0.0
    patience = 0
    max_patience = 5
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Train on batches
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
            
            # Print batch progress
            if (batch_idx + 1) % 50 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Batch Accuracy: {100. * (pred == batch_y).sum().item() / batch_y.size(0):.2f}%")
        
        # Epoch metrics
        avg_loss = train_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Training Accuracy: {epoch_acc:.2f}%")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_acc = epoch_acc
            patience = 0
            print("New best loss!")
        else:
            patience += 1
            print(f"No improvement. Patience: {patience}/{max_patience}")
        
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\nTraining completed:")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best training accuracy: {best_acc:.2f}%")
    
    # Generate test sequences
    test_sequences = generate(N=100, odd_even_mix=1.0, max_int=max_int)
    accuracy = evaluate_model(model, test_sequences, device)
    
    # Assert minimum accuracy
    assert accuracy > passing_accuracy, f"Model accuracy {accuracy:.1f}% below threshold {passing_accuracy}%"
    
    # Test specific cases
    print("\nTesting specific cases...")
    test_cases = [
        ([5, 1, 4, 2], 9),   # 5 + 4 = 9
        ([3, 1, 8, 2], 11),  # 3 + 8 = 11
        ([7, 1, 2, 2], 9),   # 7 + 2 = 9
    ]
    
    for inputs, expected in test_cases:
        x_data = torch.tensor([inputs], device=device)
        with torch.no_grad():
            pred = model.predict(x_data).cpu().item()
            print(f"\nInput: {inputs[0]} + {inputs[2]} = {expected}")
            print(f"Predicted: {pred}")
            print(f"{'✓' if pred == expected else '✗'}")
            logits = model(x_data)[0]
            top_k = torch.topk(logits, k=3)
            print("Top 3 predictions (value, logit):")
            for value, logit in zip(top_k.indices.cpu(), top_k.values.cpu()):
                print(f"{value.item()}: {logit.item():.2f}")
            assert pred == expected, f"Failed on {inputs}, got {pred}, expected {expected}"

if __name__ == "__main__":
    test_train_odd_sums()
