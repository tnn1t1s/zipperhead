import numpy as np
from typing import List, Union

def generate(N: int = 10, odd_even_mix: float = 0.5, max_int: int = 100) -> List[List[Union[int, str]]]:
    """
    Generate sequences of tokens [x, '+', y, '=', z] where x + y = z
    with controlled probability of z being odd.
    
    Args:
        N (int, optional): Number of sequences to generate. Defaults to 10.
        odd_even_mix (float, optional): Probability of z being odd (between 0 and 1). Defaults to 0.5.
        max_int (int, optional): Maximum value for generated integers. Defaults to 100.
    
    Returns:
        List[List[Union[int, str]]]: List of token sequences
    
    Raises:
        ValueError: If odd_even_mix not between 0 and 1, or if max_int < 2
    """
    if not 0 <= odd_even_mix <= 1:
        raise ValueError("odd_even_mix must be between 0 and 1")
    if max_int < 2:
        raise ValueError("max_int must be at least 2 to generate valid sequences")
        
    sequences = []
    for _ in range(N):
        # Decide if this sum should be odd based on odd_even_mix probability
        should_be_odd = np.random.random() < odd_even_mix
        
        # Generate first number randomly
        x = np.random.randint(0, max_int + 1)
        
        # If we want sum to be odd:
        # - If x is odd, y must be even
        # - If x is even, y must be odd
        # For even sum, x and y must have same parity
        if should_be_odd:
            y_start = 0 if x % 2 == 1 else 1
        else:
            y_start = 1 if x % 2 == 1 else 0
            
        # Generate y with correct parity
        y = np.random.choice(np.arange(y_start, max_int + 1, 2))
        
        # Calculate sum
        z = x + y
        
        sequences.append([x, '+', y, '=', z])
        
    return sequences

# Test the generator
if __name__ == "__main__":
    # Test with different configurations
    test_cases = [
        {},  # Use all defaults
        {'N': 5, 'odd_even_mix': 0.0},  # All even sums
        {'N': 5, 'odd_even_mix': 1.0, 'max_int': 20},  # All odd sums, small numbers
    ]
    
    for params in test_cases:
        print(f"\nTesting with parameters: {params}")
        sequences = generate(**params)
        odd_count = sum(1 for seq in sequences if seq[-1] % 2 == 1)
        print(f"Generated {len(sequences)} sequences, {odd_count} have odd sums")
        for seq in sequences:
            print(seq)
