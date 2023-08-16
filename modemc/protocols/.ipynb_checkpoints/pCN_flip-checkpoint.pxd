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

cdef class pCN_Flip(PathMCMC):
    cdef public:
        np.ndarray kappas
        np.ndarray skappas
        np.ndarray pkappas

        DTYPE_t p_flip
        int flip_attempts
        int flip_count

        # Teleport

        DTYPE_t p_teleport
        int teleporters_num
        np.ndarray teleporter_probs
        np.ndarray teleporter_log_probs
        np.ndarray teleporters_CDF
        np.ndarray teleporters

        np.ndarray _G_acc_numer_terms
        np.ndarray _G_acc_denom_terms

        int teleportation_attempts
        int teleportation_count
        np.ndarray teleporter_attempts
        np.ndarray teleporter_count

        np.ndarray teleporter_start_indices
        np.ndarray teleporter_end_indices

        # Windows

        bint window_mode

        np.ndarray rskappas

        np.ndarray window_probs
        np.ndarray window_indices
        np.ndarray window_sizes
        int windows
        np.ndarray window_acceptance_counts
        np.ndarray window_attempt_counts
        np.ndarray window_CDF
        
        np.ndarray window_dstate
        np.ndarray window_path_buffer

        int N_window_path_recalibration

