# cython: profile=False
# cython: unraisable_tracebacks=False
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from modemc.defs import DTYPE
import numpy as np

cdef void _compute_OM_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] d_path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian):
    cdef int dim = path.shape[0]
    cdef int Nt = path.shape[1]
    cdef int i,j
    cdef DTYPE_t Tf = system.Tf
    cdef DTYPE_t gamma = system.gamma
    cdef DTYPE_t beta = system.beta
    cdef DTYPE_t rgamma = 1.0/gamma
    cdef DTYPE_t _FW_pref = 0.25*(gamma*beta)
    cdef DTYPE_t _divf_pref = 0.5*rgamma
    cdef DTYPE_t dx_minus_f

    system._compute_force_and_div_force(ts, path, force, div_force)

    #for i in range(Nt):
    #    lagrangian[i] = _divf_pref * div_force[i]
    #    for j in range(dim):
    #        dx_minus_f = d_path[j,i] - rgamma*force[j,i]
    #        lagrangian[i] += _FW_pref*dx_minus_f*dx_minus_f
            
    for i in range(Nt):
        lagrangian[i] = _divf_pref * div_force[i]
    
    for j in range(dim):
        for i in range(Nt):
            dx_minus_f = d_path[j,i] - rgamma*force[j,i]
            lagrangian[i] += _FW_pref*dx_minus_f*dx_minus_f

cdef void _compute_relative_OM_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] d_path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian):
    cdef int dim = path.shape[0]
    cdef int Nt = path.shape[1]
    cdef int i,j
    cdef DTYPE_t Tf = system.Tf
    cdef DTYPE_t gamma = system.gamma
    cdef DTYPE_t beta = system.beta
    cdef DTYPE_t rgamma = 1.0/gamma
    cdef DTYPE_t _ff_pref = 0.25*beta*rgamma
    cdef DTYPE_t _fdx_pref = 0.5*beta
    cdef DTYPE_t _divf_pref = 0.5*rgamma

    system._compute_force_and_div_force(ts, path, force, div_force)

    #for i in range(Nt):
    #    lagrangian[i] = _divf_pref * div_force[i]
    #    for j in range(dim):
    #        lagrangian[i] += force[j,i] * (_ff_pref*force[j,i] - _fdx_pref*d_path[j,i])
          
    for i in range(Nt):
        lagrangian[i] = _divf_pref * div_force[i]
    
    for j in range(dim):
        for i in range(Nt):
            lagrangian[i] += force[j,i] * (_ff_pref*force[j,i] - _fdx_pref*d_path[j,i])

cdef void _compute_FW_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] d_path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian):
    cdef int dim = path.shape[0]
    cdef int Nt = path.shape[1]
    cdef int i,j
    cdef DTYPE_t Tf = system.Tf
    cdef DTYPE_t gamma = system.gamma
    cdef DTYPE_t beta = system.beta
    cdef DTYPE_t rgamma = 1.0/gamma
    cdef DTYPE_t _FW_pref = 0.25*(gamma*beta)
    cdef DTYPE_t dx_minus_f

    system._compute_force(ts, path, force)

    #for i in range(Nt):
    #    lagrangian[i] = 0
    #    for j in range(dim):
    #        dx_minus_f = d_path[j,i] - rgamma*force[j,i]
    #        lagrangian[i] += _FW_pref*dx_minus_f*dx_minus_f
      
    for i in range(Nt):
        lagrangian[i] = 0
        
    for j in range(dim):
        for i in range(Nt):
            dx_minus_f = d_path[j,i] - rgamma*force[j,i]
            lagrangian[i] += _FW_pref*dx_minus_f*dx_minus_f

cdef void _compute_relative_FW_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] d_path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian):
    cdef int dim = path.shape[0]
    cdef int Nt = path.shape[1]
    cdef int i,j
    cdef DTYPE_t Tf = system.Tf
    cdef DTYPE_t gamma = system.gamma
    cdef DTYPE_t beta = system.beta
    cdef DTYPE_t rgamma = 1.0/gamma
    cdef DTYPE_t _ff_pref = 0.25*beta*rgamma
    cdef DTYPE_t _fdx_pref = 0.5*beta

    system._compute_force(ts, path, force)

    #for i in range(Nt):
    #    lagrangian[i] = 0
    #    for j in range(dim):
    #        lagrangian[i] += force[j,i] * (_ff_pref*force[j,i] - _fdx_pref*d_path[j,i])
            
    for i in range(Nt):
        lagrangian[i] = 0
        
    for j in range(dim):
        for i in range(Nt):
            lagrangian[i] += force[j,i] * (_ff_pref*force[j,i] - _fdx_pref*d_path[j,i])

# Discretised lagrangians

cdef void _compute_OM_discretised_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian):
    cdef int dim = path.shape[0]
    cdef int Nt = path.shape[1]
    cdef int i,j
    cdef DTYPE_t Tf = system.Tf
    cdef DTYPE_t gamma = system.gamma
    cdef DTYPE_t beta = system.beta
    cdef DTYPE_t rgamma = 1.0/gamma
    cdef DTYPE_t dt = ts[1] - ts[0]
    cdef DTYPE_t _FW_pref = dt*0.25*(gamma*beta)
    cdef DTYPE_t _divf_pref = dt*0.5*rgamma
    cdef DTYPE_t dx_minus_f
    cdef DTYPE_t rdt = 1.0/dt

    system._compute_force_and_div_force(ts, path, force, div_force)

    #for i in range(Nt-1):
    #    lagrangian[i] = _divf_pref * div_force[i]
    #    for j in range(dim):
    #        dx_minus_f = (path[j,i+1]-path[j,i])*rdt - rgamma*force[j,i]
    #        lagrangian[i] += _FW_pref*dx_minus_f*dx_minus_f

    for i in range(Nt-1):
        lagrangian[i] = _divf_pref * div_force[i]
        
    for j in range(dim):
        for i in range(Nt-1):
            dx_minus_f = (path[j,i+1]-path[j,i])*rdt - rgamma*force[j,i]
            lagrangian[i] += _FW_pref*dx_minus_f*dx_minus_f
            
cdef void _compute_relative_OM_discretised_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian):
    cdef int dim = path.shape[0]
    cdef int Nt = path.shape[1]
    cdef int i,j
    cdef DTYPE_t Tf = system.Tf
    cdef DTYPE_t gamma = system.gamma
    cdef DTYPE_t beta = system.beta
    cdef DTYPE_t rgamma = 1.0/gamma
    cdef DTYPE_t dt = ts[1] - ts[0]
    cdef DTYPE_t _ff_pref = dt*0.25*beta*rgamma
    #cdef DTYPE_t _ff_pref = 0.5*dt*0.25*beta*rgamma
    cdef DTYPE_t _fdx_pref = dt*0.5*beta
    cdef DTYPE_t _divf_pref = dt*0.5*rgamma
    cdef DTYPE_t rdt = 1.0/dt

    system._compute_force_and_div_force(ts, path, force, div_force)

    #for i in range(Nt-1):
    #    lagrangian[i] = _divf_pref * div_force[i]
    #    for j in range(dim):
    #        lagrangian[i] += force[j,i] * (_ff_pref*force[j,i] - _fdx_pref*(path[j,i+1]-path[j,i])*rdt)
            
    for i in range(Nt-1):
        lagrangian[i] = _divf_pref * div_force[i]
        
    for j in range(dim):
        for i in range(Nt-1):
            lagrangian[i] += force[j,i] * (_ff_pref*force[j,i] - _fdx_pref*(path[j,i+1]-path[j,i])*rdt)
            #lagrangian[i] += force[j,i] * (_ff_pref*(force[j,i+1]+force[j,i]) - _fdx_pref*(path[j,i+1]-path[j,i])*rdt)
            
cdef void _compute_FW_discretised_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian):
    cdef int dim = path.shape[0]
    cdef int Nt = path.shape[1]
    cdef int i,j
    cdef DTYPE_t Tf = system.Tf
    cdef DTYPE_t gamma = system.gamma
    cdef DTYPE_t beta = system.beta
    cdef DTYPE_t rgamma = 1.0/gamma
    cdef DTYPE_t dt = ts[1] - ts[0]
    cdef DTYPE_t _FW_pref = dt*0.25*(gamma*beta)
    cdef DTYPE_t dx_minus_f
    cdef DTYPE_t rdt = 1.0/dt

    system._compute_force(ts, path, force)
            
    for i in range(Nt-1):
        lagrangian[i] = 0
        
    for j in range(dim):
        for i in range(Nt-1):
            dx_minus_f = (path[j,i+1]-path[j,i])*rdt - rgamma*force[j,i]
            lagrangian[i] += _FW_pref*dx_minus_f*dx_minus_f

cdef void _compute_relative_FW_discretised_lagrangian(System system, DTYPE_t[:] ts, DTYPE_t[:,:] path, DTYPE_t[:,:] force, DTYPE_t[:] div_force, DTYPE_t[:] lagrangian):
    cdef int dim = path.shape[0]
    cdef int Nt = path.shape[1]
    cdef int i,j
    cdef DTYPE_t Tf = system.Tf
    cdef DTYPE_t gamma = system.gamma
    cdef DTYPE_t beta = system.beta
    cdef DTYPE_t rgamma = 1.0/gamma
    cdef DTYPE_t dt = ts[1] - ts[0]
    cdef DTYPE_t _ff_pref = dt*0.25*beta*rgamma
    cdef DTYPE_t _fdx_pref = dt*0.5*beta
    cdef DTYPE_t rdt = 1.0/dt

    system._compute_force(ts, path, force)

    #for i in range(Nt-1):
    #    lagrangian[i] = 0
    #    for j in range(dim):
    #        lagrangian[i] += force[j,i] * (_ff_pref*force[j,i] - _fdx_pref*(path[j,i+1]-path[j,i])*rdt)
            
    for i in range(Nt-1):
        lagrangian[i] = 0
        
    for j in range(dim):
        for i in range(Nt-1):
            lagrangian[i] += force[j,i] * (_ff_pref*force[j,i] - _fdx_pref*(path[j,i+1]-path[j,i])*rdt)