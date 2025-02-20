import numpy as np
from typing import List, Union
from constraint import Problem

def generate_solution(should_be_odd: bool, max_int: int) -> dict:
    """
    Generates a single solution for x, y, z satisfying x+y=z,
    with z being odd if should_be_odd is True, else even.
    """
    problem = Problem()
    # x and y are in the range [0, max_int]
    problem.addVariable("x", range(0, max_int + 1))
    problem.addVariable("y", range(0, max_int + 1))
    # z can be as high as 2*max_int
    problem.addVariable("z", range(0, 2 * max_int + 1))
    
    # Enforce the arithmetic constraint: x + y = z.
    problem.addConstraint(lambda x, y, z: x + y == z, ("x", "y", "z"))
    
    # Enforce parity: if should_be_odd then z is odd; otherwise, even.
    if should_be_odd:
        problem.addConstraint(lambda z: z % 2 == 1, ("z",))
    else:
        problem.addConstraint(lambda z: z % 2 == 0, ("z",))
    
    solution = problem.getSolution()
    return solution

def generate(N: int = 10, odd_even_mix: float = 0.5, max_int: int = 100) -> List[List[Union[int, str]]]:
    """
    Generate sequences of tokens [x, '+', y, '=', z] using a constraint solver.
    Each sequence satisfies x + y = z and z has odd parity with probability odd_even_mix.
    
    Args:
        N (int): Number of sequences to generate.
        odd_even_mix (float): Probability (between 0 and 1) that z is odd.
        max_int (int): Maximum integer value for x and y.
    
    Returns:
        List[List[Union[int, str]]]: List of token sequences.
    
    Raises:
        ValueError: If odd_even_mix is not between 0 and 1, or if max_int < 2.
    """
    if not 0 <= odd_even_mix <= 1:
        raise ValueError("odd_even_mix must be between 0 and 1")
    if max_int < 2:
        raise ValueError("max_int must be at least 2 to generate valid sequences")
        
    sequences = []
    for _ in range(N):
        # Decide whether this sequence should have an odd z
        should_be_odd = np.random.random() < odd_even_mix
        sol = generate_solution(should_be_odd, max_int)
        sequences.append([sol["x"], '+', sol["y"], '=', sol["z"]])
    return sequences

def validate_sequence_format(sequences):
    """Validates that all sequences follow [x, '+', y, '=', z] format.
    
    Returns:
        bool
    Raises:
        ValueError: With specific index and sequence if format is invalid.
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
    """Validates that all sequences satisfy x + y = z.
    
    Returns:
        bool
    Raises:
        ValueError: With list of invalid sequences if any arithmetic errors are found.
    """
    errors = []
    for i, seq in enumerate(sequences):
        if seq[0] + seq[2] != seq[4]:
            errors.append((i, seq))
    if errors:
        raise ValueError(f"Arithmetic errors in sequences: {errors}")
    return True

def validate_distribution(sequences, max_int, odd_even_mix):
    """Validates statistical properties: range and odd/even distribution.
    
    Returns:
        dict: Distribution statistics.
    Raises:
        ValueError: If constraints are violated.
    """
    stats = {
        'total': len(sequences),
        'max_value': max(max(seq[0], seq[2]) for seq in sequences),
        'odd_ratio': sum(1 for seq in sequences if seq[4] % 2 == 1) / len(sequences)
    }
    
    if stats['max_value'] > 2 * max_int:
        raise ValueError(f"Values exceed max_int * 2: {stats['max_value']} > {max_int}")
        
    if abs(stats['odd_ratio'] - odd_even_mix) > 0.3:  # tolerance
        raise ValueError(f"Odd/even ratio {stats['odd_ratio']:.2f} doesn't match expected {odd_even_mix:.2f}")
        
    return stats

# Test the generator using different configurations
if __name__ == "__main__":
    max_int = 100
    test_cases = [
        {'N': 50, 'odd_even_mix': 1.0},  # All sequences with odd z
        {'N': 50, 'odd_even_mix': 0.0},  # All sequences with even z
        {'N': 50, 'odd_even_mix': 0.5},  # Approximately half odd and half even
    ]
    
    for params in test_cases:
        print(f"\nTesting with parameters: {params}")
        sequences = generate(**params)
        odd_count = sum(1 for seq in sequences if seq[4] % 2 == 1)
        print(f"Generated {len(sequences)} sequences, {odd_count} have odd sums")
        for seq in sequences:
            print(seq)
        validate_sequence_format(sequences)
        validate_arithmetic(sequences)
        stats = validate_distribution(sequences, max_int, odd_even_mix=params['odd_even_mix'])
        print("Distribution stats:", stats)

