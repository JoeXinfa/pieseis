# -*- coding: utf-8 -*-
"""
Get the cartesian indices of input 1-D arrays
Similar to the Julia CartesianIndices
https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
"""

import numpy as np

def cartesian(*arrays, order='F'):
    """
    -i- arrays : list of array-like,
        1-D arrays to form the cartesian product of
    -i- order : string, {'C', 'F', 'A'}, see numpy.reshape
        'F' changes the first axis fastest ("FORTRAN style" or "column-major")
        'C' changes the last axis fastest ("C style" or "row-major")
    """
    N = len(arrays)
    return np.transpose(np.meshgrid(*arrays, indexing='ij'), 
        np.roll(np.arange(N + 1), -1)).reshape((-1, N), order=order)
    
    
def main():
    print(cartesian([1,2,3], [4,5], [6,7], order='F'))
    """
    [[1 4 6]
     [2 4 6]
     [3 4 6]
     [1 5 6]
     [2 5 6]
     [3 5 6]
     [1 4 7]
     [2 4 7]
     [3 4 7]
     [1 5 7]
     [2 5 7]
     [3 5 7]]
    """

    print(cartesian([1,2,3], [4,5], [6,7], order='C'))
    """
    [[1 4 6]
     [1 4 7]
     [1 5 6]
     [1 5 7]
     [2 4 6]
     [2 4 7]
     [2 5 6]
     [2 5 7]
     [3 4 6]
     [3 4 7]
     [3 5 6]
     [3 5 7]]
    """


if __name__ == '__main__':
    main()
