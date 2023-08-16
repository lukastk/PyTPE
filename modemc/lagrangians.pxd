#!python
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

from modemc.defs cimport DTYPE_t, mt19937, normal_distribution, uniform_real_distribution, log, exp, sqrt, pow, min, max
from modemc.system cimport System
cimport numpy as np

cdef void _compute_OM_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] d_path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian)

cdef void _compute_relative_OM_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] d_path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian)

cdef void _compute_FW_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] d_path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian)

cdef void _compute_relative_FW_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] d_path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian)


cdef void _compute_OM_discretised_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian)

cdef void _compute_relative_OM_discretised_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian)

cdef void _compute_FW_discretised_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian)

cdef void _compute_relative_FW_discretised_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian)