#!python
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

from modemc.defs cimport DTYPE_t, mt19937, normal_distribution, uniform_real_distribution, abs, log, exp, sqrt, pow, min, max
from modemc.system cimport System
from modemc cimport cutils
from modemc cimport lagrangians
cimport numpy as np

cdef int _flatten_Q(np.ndarray Q_arr, np.ndarray Qf_out, dim_mode_order=?) except -1

cdef int _unflatten_Qf(np.ndarray Qf_arr, int dim, np.ndarray Q_out, dim_mode_order=?) except -1