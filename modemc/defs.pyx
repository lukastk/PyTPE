#cython: unraisable_tracebacks=False
#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
DTYPE = np.float

cdef DTYPE_t min(DTYPE_t a, DTYPE_t b):
    if a > b:
        return b
    else:
        return a

cdef DTYPE_t max(DTYPE_t a, DTYPE_t b):
    if a < b:
        return b
    else:
        return a