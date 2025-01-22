"""Main module."""

print("Hi")

def my_test(n: int) -> str:
    """ a test function

    Args:
        n (int): a test input

    Returns:
        str: a test output
    """
    print(n)
    return str(n).zfill(5)