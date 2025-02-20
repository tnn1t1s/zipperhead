import random
from constraint import Problem

def generate_line_equation(up: bool, x_max=10, m_range=10, b_range=50):
    """
    Generates a line equation y = m*x + b with either a positive (up) or negative slope.
    
    Args:
        up (bool): If True, generates a line with positive slope; if False, with negative slope.
        x_max (int): Maximum value for x.
        m_range (int): Absolute range for m. For positive slopes, m is in [1, m_range];
                       for negative slopes, m is in [-m_range, -1].
        b_range (int): b is sampled from -b_range to b_range.
    
    Returns:
        dict: A solution containing values for m, x, b, and y.
    """
    problem = Problem()
    
    # Define domains for x and b.
    problem.addVariable("x", range(0, x_max + 1))
    problem.addVariable("b", range(-b_range, b_range + 1))
    
    # Define domain for m based on slope direction.
    if up:
        problem.addVariable("m", range(1, m_range + 1))
    else:
        problem.addVariable("m", range(-m_range, 0))
    
    # y can range based on extreme values.
    y_min = -m_range * x_max - b_range
    y_max = m_range * x_max + b_range
    problem.addVariable("y", range(y_min, y_max + 1))
    
    # Enforce the line equation: y = m*x + b.
    problem.addConstraint(lambda m, x, b, y: m * x + b == y, ("m", "x", "b", "y"))
    
    return problem.getSolution()

def generate_lines(num_lines=10, M=0.5, x_max=10, m_range=10, b_range=50):
    """
    Generates a mix of line equations y = m*x + b based on the mix parameter M.
    
    Args:
        num_lines (int): Total number of line equations to generate.
        M (float): A value between 0 and 1 representing the proportion of upward (positive slope)
                   lines. For example, M=0.7 yields ~70% upward and 30% downward lines.
        x_max (int): Maximum value for x.
        m_range (int): Absolute range for m.
        b_range (int): Range for b.
    
    Returns:
        list: A list of formatted string representations of line equations.
    """
    lines = []
    for _ in range(num_lines):
        # Decide based on M whether to generate an upward or downward line.
        if random.random() < M:
            sol = generate_line_equation(up=True, x_max=x_max, m_range=m_range, b_range=b_range)
        else:
            sol = generate_line_equation(up=False, x_max=x_max, m_range=m_range, b_range=b_range)
        line_str = f"y = {sol['m']}*x + {sol['b']}  (with x={sol['x']}, y={sol['y']})"
        lines.append(line_str)
    return lines

if __name__ == "__main__":
    # Example: generate 10 lines with 70% positive slopes (M = 0.7)
    for line in generate_lines(num_lines=10, M=0.7):
        print(line)

