#!python
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

import numpy as np
cimport numpy as np

ctypedef np.float_t DTYPE_t

# wrapper for C++11 pseudo-random number generator
cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937()
        mt19937(unsigned long seed)

    cdef cppclass normal_distribution[T]:
        normal_distribution()
        normal_distribution(T a, T b)
        T operator()(mt19937 gen)

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen)

cdef extern from "math.h":
    DTYPE_t abs(DTYPE_t x)
    DTYPE_t log(DTYPE_t x)
    DTYPE_t exp(DTYPE_t x)
    DTYPE_t sqrt(DTYPE_t x)
    DTYPE_t pow(DTYPE_t x, DTYPE_t y)

cdef DTYPE_t min(DTYPE_t a, DTYPE_t b)

cdef DTYPE_t max(DTYPE_t a, DTYPE_t b)