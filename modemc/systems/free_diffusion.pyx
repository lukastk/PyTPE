#cython: unraisable_tracebacks=False
#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from modemc.defs import DTYPE
import numpy as np

cdef class FreeDiffusion:

    def __init__(self, params):
        super().__init__(params)

    cdef DTYPE_t _compute_force(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] force):
        pass

    cdef DTYPE_t _compute_force_and_div_force(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] force, DTYPE_t[:] div_force):
        pass

    cdef DTYPE_t _compute_gradL_OM(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:] L_x, DTYPE_t[:,:] L_xd):
        pass

    cdef DTYPE_t _compute_gradL_FW(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:] L_x, DTYPE_t[:,:] L_xd):
        pass

    cdef DTYPE_t _compute_hessL_OM(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:,:] L_x_x, DTYPE_t[:,:,:] L_xd_x, DTYPE_t[:,:,:] L_xd_xd):
        pass

    cdef DTYPE_t _compute_hessL_FW(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:,:] L_x_x, DTYPE_t[:,:,:] L_xd_x, DTYPE_t[:,:,:] L_xd_xd):
        pass