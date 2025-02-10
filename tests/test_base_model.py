import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.BaseModel import BaseModel
from data.generator import generate

def create_training_data(N=10000, max_int=50):
    """Create training data with odd sums"""
    sequences = generate(N=N, odd_even_mix=1.0, max_int=max_int)
    x_data = torch.tensor([[seq[0], 1, seq[2], 2] for seq in sequences])
    y_data = torch.tensor([seq[4] for seq in sequences])
    return TensorDataset(x_data, y_data)

def test_train_odd_sums():
    """Test model training and evaluation on odd-sum sequences"""
    max_int = 50
    
    # Create and setup training data
    train_data = create_training_data(N=10000, max_int=max_int)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Initialize model
    model = BaseModel(max_int=max_int)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    print("\nTraining model...")
    for epoch in range(50):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/100')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}')
    
    # Test on generated sequences
    print("\nTesting on generated sequences...")
    model.eval()
    test_sequences = generate(N=100, odd_even_mix=1.0, max_int=max_int)
    correct = 0
    
    for seq in test_sequences:
        x_data = torch.tensor([[seq[0], 1, seq[2], 2]])
        with torch.no_grad():
            pred = model.predict(x_data).item()
            correct += (pred == seq[4])
            print(f"Input: {seq[0]} + {seq[2]} = {seq[4]}")
            print(f"Predicted: {pred}")
            print(f"{'✓' if pred == seq[4] else '✗'}")
    
    accuracy = correct/len(test_sequences) * 100
    print(f"\nAccuracy on {len(test_sequences)} test sequences: {accuracy:.1f}%")
    assert accuracy > 75, f"Model accuracy {accuracy:.1f}% below threshold 75%"
    
    # Test specific cases
    print("\nTesting specific cases...")
    test_cases = [
        ([5, 1, 4, 2], 9),   # 5 + 4 = 9
        ([3, 1, 8, 2], 11),  # 3 + 8 = 11
        ([7, 1, 2, 2], 9),   # 7 + 2 = 9
    ]
    
    for inputs, expected in test_cases:
        x_data = torch.tensor([inputs])
        with torch.no_grad():
            pred = model.predict(x_data).item()
            print(f"\nInput: {inputs[0]} + {inputs[2]} = {expected}")
            print(f"Predicted: {pred}")
            print(f"{'✓' if pred == expected else '✗'}")
            logits = model(x_data)[0]
            top_k = torch.topk(logits, k=3)
            print("Top 3 predictions (value, logit):")
            for value, logit in zip(top_k.indices, top_k.values):
                print(f"{value.item()}: {logit.item():.2f}")
            assert pred == expected, f"Failed on {inputs}, got {pred}, expected {expected}"

if __name__ == "__main__":
    test_train_odd_sums()
