#!python
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

from modemc.defs cimport DTYPE_t, mt19937, normal_distribution, uniform_real_distribution, log, exp, sqrt, pow, min, max
from modemc.system cimport System
from modemc cimport cutils
from modemc cimport lagrangians

import numpy as np
cimport numpy as np
cimport cpython
from cython.operator cimport dereference

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

cdef class PathMCMC:
    cdef:
        normal_distribution[DTYPE_t] normal_dist
        uniform_real_distribution[DTYPE_t] unif_dist
        mt19937 rng_gen

    cdef public:
        System system
        bint use_OM
        bint use_discretised_action
        long rng_seed
        DTYPE_t gamma
        DTYPE_t beta
        DTYPE_t Tf
        DTYPE_t Nq_factor
        int Nq
        np.ndarray state0
        int Nm_save
        int Nm
        int dim
        int steps
        object quad_scheme
        np.ndarray x0
        np.ndarray x1
        np.ndarray qts
        np.ndarray quad_weights
        np.ndarray ts
        np.ndarray bb_mean
        np.ndarray dbb_mean
        np.ndarray phis
        np.ndarray dphis
        np.ndarray current_state
        np.ndarray current_path
        np.ndarray current_d_path
        np.ndarray full_state_arr
        np.ndarray prev_full_state_arr
        bint current_accept
        DTYPE_t current_action
        DTYPE_t noise_prefactor
        np.ndarray states_arr 
        np.ndarray accepts_arr
        np.ndarray actions_arr
        np.ndarray paths_arr 
        np.ndarray d_paths_arr 
        np.ndarray force_arr 
        np.ndarray div_force_arr
        np.ndarray lagrangian_arr
        np.ndarray noise_vector_arr

    cdef int _run_step(self, int step_i) except -1

    cpdef simulate_batch(self, int N_steps, object event_func=*, int N_event=*)

    cdef int pre_simulate_batch(self, int N_steps) except -1

    #cpdef simulate(self, int N_steps, float batch_size=*, int N_save=*, int Nm_save=*, int N_paths_save=*, bint use_GB=*)

    cpdef run_step(self)