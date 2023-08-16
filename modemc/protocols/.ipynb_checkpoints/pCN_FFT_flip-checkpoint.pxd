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

cdef class pCN_FFT_Flip(PathMCMC):
    cdef public:
        np.ndarray profile_timers
        
        np.ndarray kappas
        np.ndarray skappas
        np.ndarray pkappas

        DTYPE_t p_flip
        int flip_attempts
        int flip_count
        
        bint use_pCN

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

        # FFT

        np.ndarray path_FFT_in
        np.ndarray path_FFT_out
        np.ndarray dpath_FFT_in
        np.ndarray dpath_FFT_out

        object dst_transform
        object dct_transform

        np.ndarray path_coeffs_prefactors
        DTYPE_t dpath_coeffs_prefactor
        np.ndarray dbb_mean_const

    cdef int _FFT_compute_path_and_dpath(self, DTYPE_t[:,:] state, DTYPE_t[:,:] out_path, DTYPE_t[:,:] out_d_path) except -1

    cpdef FFT_compute_path_and_dpath(self, np.ndarray state)
    
    cdef int _FFT_compute_path(self, DTYPE_t[:,:] state, DTYPE_t[:,:] out_path) except -1

    cpdef FFT_compute_path(self, np.ndarray state)