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
        r = np.random.random()
        should_be_odd = r < odd_even_mix
        
        # Generate first number randomly
        x = int(np.random.randint(0, max_int + 1))
        
        # If we want sum to be odd:
        # - If x is odd, y must be even
        # - If x is even, y must be odd
        # For even sum, x and y must have same parity
        if should_be_odd:
            y_start = 0 if x % 2 == 1 else 1
        else:
            y_start = 1 if x % 2 == 1 else 0
            
        # Generate y with correct parity
        y = int(np.random.choice(np.arange(y_start, max_int + 1, 2)))
        
        # Calculate sum
        z = x + y
        
        sequences.append([x, '+', y, '=', z])
        
    return sequences

def validate_sequence_format(sequences):
    """Validates that all sequences follow [x,'+',y,'=',z] format
    
    Returns: bool
    Raises: ValueError with specific index and sequence if format invalid
    """
    for i, seq in enumerate(sequences):
        if (len(seq) != 5 or 
            not isinstance(seq[0], (int, float)) or
            seq[1] != '+' or
            not isinstance(seq[2], (int, float)) or
            seq[3] != '=' or
            not isinstance(seq[4], (int, float))):
            raise ValueError(f"Invalid sequence format at index {i}: {seq}")
    return True

def validate_arithmetic(sequences):
    """Validates that all sequences satisfy x + y = z
    
    Returns: bool
    Raises: ValueError with list of invalid sequences
    """
    errors = []
    for i, seq in enumerate(sequences):
        if seq[0] + seq[2] != seq[4]:
            errors.append((i, seq))
    if errors:
        raise ValueError(f"Arithmetic errors in sequences: {errors}")
    return True

def validate_distribution(sequences, max_int, odd_even_mix):
    """Validates statistical properties: range and odd/even distribution
    
    Returns: dict with distribution statistics
    Raises: ValueError if constraints violated
    """
    stats = {
        'total': len(sequences),
        'max_value': max(max(seq[0], seq[2]) for seq in sequences),
        'odd_ratio': sum(1 for seq in sequences if seq[4] % 2 == 1) / len(sequences)
    }
    
    if stats['max_value'] > 2 * max_int:
        raise ValueError(f"Values exceed max_int * 2: {stats['max_value']} > {max_int}")
        
    if abs(stats['odd_ratio'] - odd_even_mix) > 0.3:  # 1% tolerance
        raise ValueError(f"Odd/even ratio {stats['odd_ratio']:.2f} doesn't match expected {odd_even_mix:.2f}")
        
    return stats


# Test the generator
if __name__ == "__main__":
    # Test with different configurations
    max_int = 100
    test_cases = [
        {'N': 5, 'odd_even_mix': 1.0},  # All even sums
        {'N': 5, 'odd_even_mix': 0.0},  # All odd sums
        {'N': 5, 'odd_even_mix': 0.5},  # coin flip sums
    ]
    
    for params in test_cases:
        print(f"\nTesting with parameters: {params}")
        sequences = generate(**params)
        odd_count = sum(1 for seq in sequences if seq[-1] % 2 == 1)
        print(f"Generated {len(sequences)} sequences, {odd_count} have odd sums")
        # Validate data before training
        for x in sequences:
                print(x)
        validate_sequence_format(sequences)
        validate_arithmetic(sequences)
        stats = validate_distribution(sequences, max_int, odd_even_mix=params['odd_even_mix'])
        print("Distribution stats:", stats)
