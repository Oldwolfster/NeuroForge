def add(a: int | float, b: int | float) -> int | float:
    """
    Add two numbers.

    This function takes two numbers (either integers or floats) and returns their sum.
    It is a simple demonstration of how to write a Google style docstring.

    Args:
        a (int or float): The first number to add.
        b (int or float): The second number to add.

    Returns:
        int or float: The sum of the two numbers. The return type will match the input types.

    Raises:
        TypeError: If either `a` or `b` is not an integer or float.

    Examples:
        >>> add(2, 3)
        5
        >>> add(2.5, 3.5)
        6.0
        >>> add(2, 3.5)
        5.5

    Notes:
        - If both inputs are integers, the result will be an integer.
        - If either input is a float, the result will be a float.
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both inputs must be integers or floats.")
    return a + b


""" Need to recurse
import doctest
doctest.testfile("src/ArenaSettings.py")
doctest.testfile("src/engine/Engine.py")
or pytest
pip install pytest
pytest --doctest-modules
"""



