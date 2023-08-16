#!python
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

from modemc.defs cimport DTYPE_t, mt19937, normal_distribution, uniform_real_distribution, log, exp, sqrt, pow, min, max
from modemc.system cimport System
cimport numpy as np

from libc.stdlib cimport rand, RAND_MAX

cdef extern from "math.h":
    DTYPE_t log(DTYPE_t x)
    DTYPE_t exp(DTYPE_t x)
    DTYPE_t sqrt(DTYPE_t x)

cdef DTYPE_t _eval_action(DTYPE_t[:] quad_weights, DTYPE_t Tf, DTYPE_t[:] lagrangian)

cdef DTYPE_t _eval_gaussian_action(DTYPE_t[:] x0, DTYPE_t[:] x1, DTYPE_t Tf, DTYPE_t beta, DTYPE_t gamma, DTYPE_t[:,:] state)

cdef int _eval_action_gradient(np.ndarray wphis, np.ndarray wdphis, np.ndarray L_x, np.ndarray L_xd, np.ndarray out_gradS_buffer, np.ndarray out_gradS) except -1

cdef int _fast_eval_path(np.ndarray bb_mean, np.ndarray dbb_mean, np.ndarray phis, np.ndarray dphis, np.ndarray state, np.ndarray out_path, np.ndarray out_d_path) except -1
    
cdef int _fast_eval_path_without_mean(np.ndarray phis, np.ndarray dphis, np.ndarray state, np.ndarray out_path, np.ndarray out_d_path) except -1

cdef int _fast_update_path(np.ndarray phis, np.ndarray dphis, np.ndarray state, np.ndarray out_path, np.ndarray out_d_path, np.ndarray path_buffer) except -1

# Random sampler functions
# Source: https://stackoverflow.com/questions/42767816/what-is-the-most-efficient-and-portable-way-to-generate-gaussian-random-numbers

cdef DTYPE_t random_uniform()

cdef DTYPE_t random_gaussian()

cdef void assign_random_gaussian_pair(DTYPE_t[:] out, int assign_ix)

cdef void uniform_vector(DTYPE_t[:] result)

cdef void gaussian_vector(DTYPE_t[:] result)