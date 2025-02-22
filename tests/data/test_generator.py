import pytest
import numpy as np
import scipy
from data.generator import generate

def test_sequence_format():
    """Test that generated sequences have correct format"""
    sequence = generate(N=1)[0]
    assert len(sequence) == 5
    assert sequence[1] == '+'
    assert sequence[3] == '='
    assert sequence[4] == sequence[0] + sequence[2]

def test_invalid_inputs():
    """Test that invalid inputs raise appropriate errors"""
    with pytest.raises(ValueError):
        generate(odd_even_mix=-0.1)
    with pytest.raises(ValueError):
        generate(odd_even_mix=1.1)
    with pytest.raises(ValueError):
        generate(max_int=1)

@pytest.mark.parametrize("max_int", [10, 100, 1000])
def test_max_int_boundary(max_int):
    """Test that generated numbers don't exceed max_int"""
    sequences = generate(N=100, max_int=max_int)
    for seq in sequences:
        assert seq[0] <= max_int
        assert seq[2] <= max_int
        assert seq[4] <= 2 * max_int  # sum can be up to 2*max_int

@pytest.mark.parametrize("odd_even_mix,expected_ratio,tolerance", [
    (0.0, 0.0, 0.0),    # all even
    (1.0, 1.0, 0.0),    # all odd
    (0.5, 0.5, 0.1),    # approximately half odd/even
])
def test_odd_even_distribution(odd_even_mix, expected_ratio, tolerance):
    """Test that odd/even distribution matches expected probabilities"""
    N = 1000  # large enough for statistical significance
    sequences = generate(N=N, odd_even_mix=odd_even_mix)
    
    # Count odd sums
    odd_count = sum(1 for seq in sequences if seq[4] % 2 == 1)
    actual_ratio = odd_count / N
    
    assert abs(actual_ratio - expected_ratio) <= tolerance

def test_uniformity_within_parity():
    """validate that our random number generator is truly uniform using a chisquare test"""
    N = 10000
    max_int = 100
    n_bins = 5  # Reduced number of bins for more robust testing
    sequences = generate(N=N, max_int=max_int, odd_even_mix=0.5)
    
    # Extract x values and separate by parity
    x_values = [seq[0] for seq in sequences]
    odd_x = [x for x in x_values if x % 2 == 1]
    even_x = [x for x in x_values if x % 2 == 0]
    
    # Test uniformity using chi-square test if we have enough samples
    if len(odd_x) > 0:
        hist, _ = np.histogram(odd_x, bins=n_bins)
        expected = np.array([len(odd_x)/n_bins] * n_bins)  # uniform expectation
        _, p_value_odd = scipy.stats.chisquare(hist, expected)
        assert p_value_odd > 0.001  # Using 0.1% significance level
    
    if len(even_x) > 0:
        hist, _ = np.histogram(even_x, bins=n_bins)
        expected = np.array([len(even_x)/n_bins] * n_bins)  
        _, p_value_even = scipy.stats.chisquare(hist, expected)
        assert p_value_even > 0.001  # Using 0.1% significance level

def test_arithmetic_correctness():
    """Test that all arithmetic operations are correct"""
    sequences = generate(N=100)
    for x, plus, y, equals, z in sequences:
        assert plus == '+'
        assert equals == '='
        assert x + y == z

if __name__ == '__main__':
    pytest.main([__file__])
