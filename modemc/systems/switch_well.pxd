#!python
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

from modemc.defs cimport DTYPE_t, mt19937, normal_distribution, uniform_real_distribution, log, exp, sqrt, pow, min, max
from modemc.system cimport System

cdef class SwitchWell(System):
    cdef public:
        DTYPE_t U0
        DTYPE_t xi1
        DTYPE_t xi2
        DTYPE_t xip
        DTYPE_t xim
        
        DTYPE_t Ux0
        DTYPE_t dUx