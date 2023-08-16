#!python
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

from modemc.defs cimport DTYPE_t, mt19937, normal_distribution, uniform_real_distribution, log, exp, sqrt, pow, min, max
from modemc.system cimport System
from modemc cimport cutils
from modemc cimport lagrangians
from modemc.path_MCMC cimport PathMCMC

cimport numpy as np
cimport cpython

cdef class Naive(PathMCMC):
    cdef public:
        DTYPE_t kappa
        DTYPE_t skappa
        DTYPE_t pkappa