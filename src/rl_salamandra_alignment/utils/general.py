import os
import json

def my_second_test(n: int) -> str:
    """a second test function

    :param n: a second test input
    :type n: int
    :return: a second test output
    :rtype: str
    """        

    print(n)
    return str(n).zfill(5)