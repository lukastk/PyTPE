#!python
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

from modemc.defs cimport DTYPE_t, mt19937, normal_distribution, uniform_real_distribution, log, exp, sqrt, pow, min, max
cimport numpy as np

cdef extern from "math.h":
    double exp(double x)

cdef class System:
    cdef public:
        DTYPE_t beta
        DTYPE_t gamma
        DTYPE_t Tf
        int dim

        np.ndarray x0
        np.ndarray x1

    cdef DTYPE_t _compute_potential(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:] potential)

    cdef DTYPE_t _compute_force(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] force)

    cdef DTYPE_t _compute_force_and_div_force(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] force, DTYPE_t[:] div_force)

    cdef DTYPE_t _compute_gradL_OM(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:] L_x, DTYPE_t[:,:] L_xd)

    cdef DTYPE_t _compute_gradL_FW(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:] L_x, DTYPE_t[:,:] L_xd)

    cdef DTYPE_t _compute_hessL_OM(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:,:] L_x_x, DTYPE_t[:,:,:] L_xd_x, DTYPE_t[:,:,:] L_xd_xd)

    cdef DTYPE_t _compute_hessL_FW(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:,:] L_x_x, DTYPE_t[:,:,:] L_xd_x, DTYPE_t[:,:,:] L_xd_xd)
    
    cdef DTYPE_t _compute_hessL_OM(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:,:] L_xd_x)

    cdef DTYPE_t _compute_hessL_FW(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:,:] L_xd_x)

    #cpdef compute_potential(self, path, ts=?)

    #cpdef compute_force(self, path, ts=?)

    cpdef compute_force_and_div_force(self, path, ts=?)

    cpdef compute_gradL(self, path, d_path, use_OM=?, ts=?)

    cpdef compute_hessL(self, path, d_path, use_OM=?, ts=?)