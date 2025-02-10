import torch
from torch.utils.data import DataLoader, TensorDataset
from models.BaseModel import BaseModel
from data.generator import generate

def train_and_check():
    max_int = 50
    
    # Train model first
    print("Training model...")
    train_data = create_training_data(N=10000, max_int=max_int)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    model = BaseModel(max_int=max_int)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
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
            print(f'Train Loss: {train_loss/len(train_loader):.4f}\n')
    
    # Now check predictions
    print("\nChecking individual predictions...")
    model.eval()
    
    # Generate fresh sequences for testing
    N = 20
    sequences = generate(N=N, odd_even_mix=1.0, max_int=max_int)
    
    correct = 0
    for seq in sequences:
        x_data = torch.tensor([[seq[0], 1, seq[2], 2]])
        with torch.no_grad():
            pred = model.predict(x_data).item()
            correct += (pred == seq[4])
            
            print(f"\nInput: {seq[0]} + {seq[2]} = {seq[4]}")
            print(f"Predicted: {pred}")
            print(f"{'✓' if pred == seq[4] else '✗'}")
            
            # Show top predictions
            logits = model(x_data)[0]
            top_k = torch.topk(logits, k=3)
            print("Top 3 predictions (value, logit):")
            for value, logit in zip(top_k.indices, top_k.values):
                print(f"{value.item()}: {logit.item():.2f}")
    
    print(f"\nAccuracy on these {N} samples: {correct/N*100:.1f}%")

    # Add after the main testing loop, before final accuracy print
    print("\nChecking specific test cases:")
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

def create_training_data(N=10000, max_int=50):
    sequences = generate(N=N, odd_even_mix=1.0, max_int=max_int)
    x_data = torch.tensor([[seq[0], 1, seq[2], 2] for seq in sequences])
    y_data = torch.tensor([seq[4] for seq in sequences])
    return TensorDataset(x_data, y_data)

if __name__ == "__main__":
    train_and_check()
