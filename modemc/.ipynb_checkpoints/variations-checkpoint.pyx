# cython: unraisable_tracebacks=True
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from modemc.defs import DTYPE
from modemc import quad, utils
import numpy as np
import scipy.optimize
import mpmath
from scipy.linalg import expm, logm

cdef int _flatten_Q(np.ndarray Q_arr, np.ndarray Qf_out, dim_mode_order=True) except -1:
    cdef int dim = Q_arr.shape[0]
    cdef int Nm = Q_arr.shape[2]
    cdef int i,j,k,l
    #Qf_arr = np.zeros( ( dim*Nm, dim*Nm ) )

    cdef DTYPE_t[:,:,:,:] Q = Q_arr
    cdef DTYPE_t[:,:] Qf = Qf_out

    if dim_mode_order:
        for i in range(dim):
            for j in range(dim):
                for k in range(Nm):
                    for l in range(Nm):
                        Qf[i*Nm + k, j*Nm + l] = Q[i,j,k,l]
    else:
        for i in range(dim):
            for j in range(dim):
                for k in range(Nm):
                    for l in range(Nm):
                        Qf[k*dim + i, l*dim + j] = Q[i,j,k,l]

cdef int _unflatten_Qf(np.ndarray Qf_arr, int dim, np.ndarray Q_out, dim_mode_order=True) except -1:
    cdef int Nm = int(Qf_arr.shape[0] / dim)
    cdef int i,j,k,l

    cdef DTYPE_t[:,:,:,:] Q = Q_out
    cdef DTYPE_t[:,:] Qf = Qf_arr

    if dim_mode_order:
        for i in range(dim):
            for j in range(dim):
                for k in range(Nm):
                    for l in range(Nm):
                        Q[i,j,k,l] = Qf[i*Nm + k, j*Nm + l]
    else:
        for i in range(dim):
            for j in range(dim):
                for k in range(Nm):
                    for l in range(Nm):
                        Q[i,j,k,l] = Qf[k*dim + i, l*dim + j]

cpdef flatten_Q(Q_arr, dim_mode_order=True):
    dim = Q_arr.shape[0]
    Nm = Q_arr.shape[2]
    Qf_out = np.zeros((dim*Nm, dim*Nm))
    _flatten_Q(Q_arr, Qf_out, dim_mode_order)
    return Qf_out

cpdef unflatten_Qf(Qf_arr, dim, dim_mode_order=True):
    Nm = int(Qf_arr.shape[0] / dim)
    Q_out = np.zeros((dim, dim, Nm, Nm))
    _unflatten_Qf(Qf_arr, dim, Q_out, dim_mode_order)
    return Q_out



cpdef int _compute_banded_Q_matrix(np.ndarray bb_mean, np.ndarray dbb_mean, np.ndarray phis, np.ndarray dphis, np.ndarray inst_phis, np.ndarray inst_dphis,
                np.ndarray ts, np.ndarray qts, np.ndarray quad_weights, np.ndarray Q, np.ndarray Qxx, np.ndarray Qxdx, np.ndarray Qxdxd,
                    np.ndarray L_x_x, np.ndarray L_xd_x, np.ndarray L_xd_xd, np.ndarray path, np.ndarray d_path,
                    System system, np.ndarray instanton_state, DTYPE_t Nq_factor, np.ndarray band_sizes, int Nm, bint use_OM) except -1:
    cdef int k_band, Nq_exponent, elem_skip
    cdef int dim = system.dim
    cdef int i, j, k

    cutils._fast_eval_path(bb_mean, dbb_mean, inst_phis, inst_dphis, instanton_state, path, d_path)

    if use_OM:
        system._compute_hessL_OM(ts, path, d_path, L_x_x, L_xd_x, L_xd_xd)
    else:
        system._compute_hessL_FW(ts, path, d_path, L_x_x, L_xd_x, L_xd_xd)

    for i in range(dim):    
        np.fill_diagonal(Qxdxd[i,i,:,:], system.beta*system.gamma*0.5)

    for k in range(Nm):
        k_band = int(min(np.ceil(k+band_sizes[k]), Nm))
        np.einsum('ijn,n,ln,n->ijl',  L_x_x[:,:,:],   phis[k,:],  phis[k:k_band,:], quad_weights, out=Qxx[:,:,k,k:k_band]) 
        np.einsum('ijn,n,ln,n->ijl', L_xd_x[:,:,:],  phis[k,:],  dphis[k:k_band,:], quad_weights, out=Qxdx[:,:,k,k:k_band]) 
        np.einsum('ijn,n,ln,n->ijl', L_xd_x[:,:,:],  dphis[k,:],  phis[k:k_band,:], quad_weights, out=Qxdx[:,:,k:k_band,k]) 

        Qxx[:,:,k:k_band,k] = Qxx[:,:,k,k:k_band]

    # Symmetrise Qxdx
    for i in range(dim):
        for j in range(dim):
            np.add(Qxdx[i,j,:,:], Qxdx[i,j,:,:].T, Qxdx[i,j,:,:])

    np.add(Qxx, Q, Q)
    np.add(Qxdx, Q, Q)
    np.add(Qxdxd, Q, Q)

cpdef compute_banded_Q_matrix(system, instanton_state, band_size, Nm=None, Nq_factor=30, max_Nq=-1, quad_scheme=None, return_partial_Qs=False, use_OM=True,
        use_adapted_quadrature=True):
    x0, x1, Tf = system.x0, system.x1, system.Tf
    dim = system.dim
    
    inst_Nm = instanton_state.shape[1]
    if Nm is None:
        Nm = inst_Nm

    if use_adapted_quadrature:
        if max_Nq != -1:
            Nq = min(Nm*Nq_factor, max_Nq)
        else:
            Nq = Nm*Nq_factor
        Nq_max_exponent = int(np.max(np.log2(Nq)))
        
        if quad_scheme is None:
            quad_scheme = quad.clenshaw_curtis

        qts_sets = []
        quad_weights_sets = []
        for Nq_exponent in range(1, Nq_max_exponent+1):
            Nq = 2**Nq_exponent
            ts, qts, quad_weights = quad_scheme(Nq+1, Tf) # Must add +1 so that the quadratures line up

            qts_sets.append(qts)
            quad_weights_sets.append(quad_weights)
    else:
        Nq = int(np.round(Nm*Nq_factor))
        ts, qts, quad_weights = quad_scheme(Nq+1, Tf)
        
    Nt = len(ts)

    if type(band_size) == int:
        band_size = np.full(Nm, band_size)

    # For computing Q
    _, _, phis, dphis = utils.get_fast_eval_path_basis(x0, x1, ts, Tf, Nm)

    # For computing the path
    bb_mean, dbb_mean, inst_phis, inst_dphis = utils.get_fast_eval_path_basis(x0, x1, ts, Tf, inst_Nm)
    
    Q = np.zeros( (dim, dim, Nm, Nm) )
    Qxx = np.zeros( (dim, dim, Nm, Nm) )
    Qxdx = np.zeros( (dim, dim, Nm, Nm) )
    Qxdxd = np.zeros( (dim, dim, Nm, Nm) )

    L_x_x = np.zeros( (dim, dim, Nt))
    L_xd_x = np.zeros( (dim, dim, Nt))
    L_xd_xd  = np.zeros( (dim, dim, Nt))

    path = np.zeros( (dim, Nt) )
    d_path = np.zeros( (dim, Nt) )

    if use_adapted_quadrature:
        _compute_banded_Q_matrix_with_adapted_quadrature(bb_mean, dbb_mean, phis, dphis, inst_phis, inst_dphis, quad_weights_sets, Q, Qxx, Qxdx, Qxdxd, L_x_x, L_xd_x, L_xd_xd, path, d_path, ts,
                    system, instanton_state, Nq_factor, band_size, Nm, Nq_max_exponent, use_OM)
    else:
        _compute_banded_Q_matrix(bb_mean, dbb_mean, phis, dphis, inst_phis, inst_dphis, ts, qts, quad_weights, Q, Qxx, Qxdx, Qxdxd, L_x_x, L_xd_x, L_xd_xd, path, d_path,
                    system, instanton_state, Nq_factor, band_size, Nm, use_OM)
    
    if return_partial_Qs:
        Qfxx = np.zeros((dim*Nm, dim*Nm))
        Qfxdx = np.zeros((dim*Nm, dim*Nm))
        Qfxdxd = np.zeros((dim*Nm, dim*Nm))

        _flatten_Q(Qxx, Qfxx)
        _flatten_Q(Qxdx, Qfxdx)
        _flatten_Q(Qxdxd, Qfxdxd)

    Qf = np.zeros((dim*Nm, dim*Nm))
    _flatten_Q(Q, Qf)

    if return_partial_Qs:
        return Q, Qf, Qfxx, Qfxdx, Qfxdxd
    else:
        return Q, Qf

cpdef compute_Q_matrix_diagonal(System system, instanton_state, Nm=None, Nm_start=0, Nq_factor=30, max_Nq=-1, quad_scheme=None, use_OM=True):
    cdef int i, k

    x0, x1, Tf = system.x0, system.x1, system.Tf
    dim = system.dim
    
    inst_Nm = instanton_state.shape[1]
    if Nm is None:
        Nm = inst_Nm

    if max_Nq != -1:
        Nq = min(Nm*Nq_factor, max_Nq)
    else:
        Nq = Nm*Nq_factor
    Nq_max_exponent = int(np.max(np.log2(Nq)))
    
    if quad_scheme is None:
        quad_scheme = quad.clenshaw_curtis

    qts_sets = []
    quad_weights_sets = []
    for Nq_exponent in range(1, Nq_max_exponent+1):
        Nq = 2**Nq_exponent
        ts, qts, quad_weights = quad_scheme(Nq+1, Tf) # Must add +1 so that the quadratures line up

        qts_sets.append(qts)
        quad_weights_sets.append(quad_weights)
        
    Nt = len(ts)

    # For computing the path
    bb_mean, dbb_mean, inst_phis, inst_dphis = utils.get_fast_eval_path_basis(x0, x1, ts, Tf, inst_Nm)
    
    diag_Qxx = np.zeros( (dim, Nm-Nm_start) )
    diag_Qxdx = np.zeros( (dim, Nm-Nm_start) )
    diag_Qxdxd = np.zeros( (dim, Nm-Nm_start) )

    L_x_x = np.zeros( (dim, dim, Nt))
    L_xd_x = np.zeros( (dim, dim, Nt))
    L_xd_xd  = np.zeros( (dim, dim, Nt))

    path = np.zeros( (dim, Nt) )
    d_path = np.zeros( (dim, Nt) )

    cutils._fast_eval_path(bb_mean, dbb_mean, inst_phis, inst_dphis, instanton_state, path, d_path)

    if use_OM:
        system._compute_hessL_OM(ts, path, d_path, L_x_x, L_xd_x, L_xd_xd)
    else:
        system._compute_hessL_FW(ts, path, d_path, L_x_x, L_xd_x, L_xd_xd)

    diag_L_x_x = np.zeros( (dim, Nt))
    diag_L_xd_x = np.zeros( (dim, Nt))
    diag_L_xd_xd  = np.zeros( (dim, Nt))

    for i in range(dim):
        diag_L_x_x[i] = L_x_x[i,i,:]
        diag_L_xd_x[i] = L_xd_x[i,i,:]
        diag_L_xd_xd[i] = L_xd_xd[i,i,:]

    for i in range(dim):
        diag_Qxdxd[i,:] = system.beta*system.gamma*0.5

    prev_Nq_exponent = -1

    for k in range(Nm_start, Nm):
        Nq_exponent = int(np.floor(np.log2((k+1)*Nq_factor)))
        elem_skip = 2**(Nq_max_exponent-Nq_exponent)
        Nq = int(2**Nq_exponent)

        if prev_Nq_exponent != Nq_exponent:
            ts, qts, quad_weights = quad_scheme(Nq+1, Tf)

        phi = np.sin(np.pi*(k+1)*ts/Tf) * (np.sqrt(2*Tf) / np.pi) / (k+1)
        dphi = np.cos(np.pi*(k+1)*ts/Tf) * np.sqrt(2 / Tf)

        np.einsum('in,n,n,n->i',  diag_L_x_x[:,::elem_skip],   phi,  phi, quad_weights, out=diag_Qxx[:,k-Nm_start]) 
        np.einsum('in,n,n,n->i', diag_L_xd_x[:,::elem_skip],  phi, dphi, quad_weights, out=diag_Qxdx[:,k-Nm_start]) 
        diag_Qxdx[:,k-Nm_start] += np.einsum('in,n,n,n->i', diag_L_xd_x[:,::elem_skip],  dphi,  phi, quad_weights)

        prev_Nq_exponent = Nq_exponent

    diag_Q = diag_Qxx + diag_Qxdx + diag_Qxdxd
    diag_Qf = np.zeros( dim*(Nm-Nm_start) )

    for i in range(dim):
        for k in range(Nm_start, Nm):
            diag_Qf[i + (k-Nm_start)*dim] = diag_Q[i,k-Nm_start]

    return diag_Q, diag_Qf, diag_Qxx, diag_Qxdx, diag_Qxdxd


cpdef int _compute_banded_Q_matrix_with_adapted_quadrature(np.ndarray bb_mean, np.ndarray dbb_mean, np.ndarray phis, np.ndarray dphis, np.ndarray inst_phis, np.ndarray inst_dphis, list quad_weights_sets, np.ndarray Q, np.ndarray Qxx, np.ndarray Qxdx, np.ndarray Qxdxd,
                    np.ndarray L_x_x, np.ndarray L_xd_x, np.ndarray L_xd_xd, np.ndarray path, np.ndarray d_path, np.ndarray ts,
                    System system, np.ndarray instanton_state, DTYPE_t Nq_factor, np.ndarray band_sizes, int Nm, int Nq_max_exponent, bint use_OM) except -1:
    cdef int k_band, Nq_exponent, elem_skip
    cdef int dim = system.dim
    cdef np.ndarray quad_weights
    cdef int i, j, k

    cutils._fast_eval_path(bb_mean, dbb_mean, inst_phis, inst_dphis, instanton_state, path, d_path)

    if use_OM:
        system._compute_hessL_OM(ts, path, d_path, L_x_x, L_xd_x, L_xd_xd)
    else:
        system._compute_hessL_FW(ts, path, d_path, L_x_x, L_xd_x, L_xd_xd)

    for i in range(dim):    
        np.fill_diagonal(Qxdxd[i,i,:,:], system.beta*system.gamma*0.5)

    for k in range(Nm):
        k_band = int(min(np.ceil(k+band_sizes[k]), Nm))
        Nq_exponent = int(np.floor(np.log2(k_band*Nq_factor)))
        Nq_exponent = int(min(Nq_exponent, len(quad_weights_sets)))
        quad_weights = <np.ndarray> quad_weights_sets[Nq_exponent-1]
        elem_skip = 2**(Nq_max_exponent-Nq_exponent)

        #np.einsum('ijn,n,ln,n->ijl',  L_x_x[:,:,::elem_skip],   phis[k,::elem_skip],  phis[k:k_band,::elem_skip], quad_weights, out=Qxx[:,:,k,k:k_band]) 
        #np.einsum('ijn,n,ln,n->ijl', 2*L_xd_x[:,:,::elem_skip],  phis[k,::elem_skip],  dphis[k:k_band,::elem_skip], quad_weights, out=Qxdx[:,:,k,k:k_band]) 
        #np.einsum('ijn,n,ln,n->ijl', 2*L_xd_x[:,:,::elem_skip],  phis[k,::elem_skip],  dphis[k:k_band,::elem_skip], quad_weights, out=Qxdx[:,:,k:k_band,k]) 

        np.einsum('ijn,n,ln,n->ijl',  L_x_x[:,:,::elem_skip],   phis[k,::elem_skip],  phis[k:k_band,::elem_skip], quad_weights, out=Qxx[:,:,k,k:k_band]) 
        np.einsum('ijn,n,ln,n->ijl', L_xd_x[:,:,::elem_skip],  phis[k,::elem_skip],  dphis[k:k_band,::elem_skip], quad_weights, out=Qxdx[:,:,k,k:k_band]) 
        np.einsum('ijn,n,ln,n->ijl', L_xd_x[:,:,::elem_skip],  dphis[k,::elem_skip],  phis[k:k_band,::elem_skip], quad_weights, out=Qxdx[:,:,k:k_band,k]) 
        
        Qxx[:,:,k:k_band,k] = Qxx[:,:,k,k:k_band]

    # Symmetrise Qxdx
    for i in range(dim):
        for j in range(dim):
            np.add(Qxdx[i,j,:,:], Qxdx[i,j,:,:].T, Qxdx[i,j,:,:])

    np.add(Qxx, Q, Q)
    np.add(Qxdx, Q, Q)
    np.add(Qxdxd, Q, Q)

cpdef compute_Q_matrix(System system, instanton_state, band_size, Nm=None, Nq_factor=30, quad_scheme=None, return_partial_Qs=False, use_OM=True):
    x0, x1, Tf = system.x0, system.x1, system.Tf
    dim = system.dim
    
    inst_Nm = instanton_state.shape[1]
    if Nm is None:
        Nm = inst_Nm

    if quad_scheme is None:
        quad_scheme = quad.clenshaw_curtis

    Nq = Nm*Nq_factor
    ts, qts, quad_weights = quad_scheme(Nq, Tf)
    ts = 0.5*(qts+1)*Tf
    Nt = len(ts)
    
    L_x_x = np.zeros( (dim, dim, Nt))
    L_xd_x = np.zeros( (dim, dim, Nt))
    L_xd_xd  = np.zeros( (dim, dim, Nt))
    
    path = np.zeros( (dim, Nt) )
    d_path = np.zeros( (dim, Nt) )

    bb_mean, dbb_mean, inst_phis, inst_dphis = utils.get_fast_eval_path_basis(x0, x1, ts, Tf, inst_Nm)
    
    cutils._fast_eval_path(bb_mean, dbb_mean, inst_phis, inst_dphis, instanton_state, path, d_path)
    
    if use_OM:
        system._compute_hessL_OM(ts, path, d_path, L_x_x, L_xd_x, L_xd_xd)
    else:
        system._compute_hessL_FW(ts, path, d_path, L_x_x, L_xd_x, L_xd_xd)
    
    #phis = []
    #dphis = []
    #for k in range(1, Nm+1):
    #    sqr_lambdak = (Tf / (np.pi*k))
    #    phi = np.sqrt(2/Tf) * np.sin(ts/sqr_lambdak) * sqr_lambdak
    #    dphi = np.sqrt(2/Tf) * np.cos(ts/sqr_lambdak)
    #    phis.append(phi)
    #    dphis.append(dphi)
    
    #phis = np.array(phis) * np.sqrt(Tf/2) # The Tf/2 factor is for scaling the to physical time from the quadrature
    #dphis = np.array(dphis) * np.sqrt(Tf/2)

    # For computing Q
    _, _, phis, dphis = utils.get_fast_eval_path_basis(x0, x1, ts, Tf, Nm)
    
    Qxx = np.zeros( (dim, dim, Nm, Nm) )
    Qxdx = np.zeros( (dim, dim, Nm, Nm) )
    Qxdxd = np.zeros( (dim, dim, Nm, Nm) )
       
    Qxx =   np.einsum('ijn,kn,ln,n->ijkl', L_x_x, phis, phis, quad_weights)
    Qxdx =  2*np.einsum('ijn,kn,ln,n->ijkl', L_xd_x, dphis, phis, quad_weights)
    #Qxdx += np.einsum('ijn,kn,ln,n->ijkl', L_xd_x, phis, dphis, quad_weights)
    for i in range(dim):    
        np.fill_diagonal(Qxdxd[i,i,:,:], system.beta*system.gamma*0.5)
    
    Q = Qxx + Qxdx + Qxdxd

    if return_partial_Qs:
        Qfxx = flatten_Q(Qxx)
        Qfxdx = flatten_Q(Qxdx)
        Qfxdxd = flatten_Q(Qxdxd)

    Qf = flatten_Q(Q)
    Qf = (Qf + Qf.T)/2 # Symmetrise
    Q = unflatten_Qf(Qf, dim)

    if return_partial_Qs:
        return Q, Qf, Qfxx, Qfxdx, Qfxdxd
    else:
        return Q, Qf

cpdef compute_brownian_Q_matrix(x0, x1, Tf, beta, gamma, dim, Nm):
    Q = np.zeros( (dim, dim, Nm, Nm) )

    for i in range(dim):    
        np.fill_diagonal(Q[i,i,:,:], beta*gamma*0.5)

    Qf = flatten_Q(Q)

    return Q, Qf

cpdef _GY_integrator(np.ndarray x, np.ndarray G, int steps):
    for i in range(steps):
        x = expm(G[:,:,i]).dot(x)
    return x

cpdef compute_gelfand_yaglom_determinant(dt, system, state, use_OM, mpmath_dps=500):
    mpmath.mp.dps = mpmath_dps
    mpmath_num = mpmath.mpf(1)
    
    dim = system.dim
    beta = system.beta
    gamma = system.gamma
    Tf = system.Tf
    steps = int(np.round(Tf / dt))
    
    ts = np.linspace(0, Tf, steps)
    dt = ts[1]-ts[0]
    
    L_x_x, L_xd_x, L_xd_xd = utils.compute_hessL(system, ts, state, use_OM=use_OM)
    L_x_xd = np.swapaxes(L_xd_x, 0, 1)

    #sL_xd_x = 0.5*(L_x_xd + L_xd_x)
    #sL_xd_x_dot = np.gradient(sL_xd_x, dt, axis=2)
    
    dot_L_xd_x = utils.compute_dot_L_xd_x(system, ts, state, use_OM=use_OM)
    
    Q = L_x_x - dot_L_xd_x
    P = beta * gamma / 2
    nQ = (dt*Q/P) * mpmath_num
    
    dt = dt * mpmath_num
    
    G = np.zeros((2*dim, 2*dim, nQ.shape[2]))
    G[:dim, dim:, :] = nQ
    for i in range(dim):
        G[dim+i, i] = dt
    G = G * mpmath_num
    
    Y = np.zeros((dim,dim)) * mpmath_num
    
    for i in range(dim):
        x = np.zeros(2*dim) * mpmath_num
        
        x[i] = 1 * mpmath_num
        Y[i] = _GY_integrator(x, G, steps)[dim:]
        
    num_Y = np.array(Y, dtype=float)
    det_sign, log_det = np.linalg.slogdet(num_Y)
    
    return det_sign, log_det, Y

"""


def flatten_Q(Q):
    dim = Q.shape[0]
    num_of_modes = Q.shape[2]
    Qf = np.zeros( ( dim*num_of_modes, dim*num_of_modes ) )
    for i in range(dim):
        for j in range(dim):
            for k in range(num_of_modes):
                for l in range(num_of_modes):
                    Qf[k*dim + i, l*dim + j] = Q[i,j,k,l]
    return Qf

def unflatten_Qf(Qf, dim):
    num_of_modes = int(Qf.shape[0] / dim)
    Q = np.zeros( ( dim, dim, num_of_modes, num_of_modes ) )
    for i in range(dim):
        for j in range(dim):
            for k in range(num_of_modes):
                for l in range(num_of_modes):
                    Q[i,j,k,l] = Qf[k*dim + i, l*dim + j]
    return Q

def compute_Q_matrix(instanton_coeffs, x0, x1, Tf, L_2nd_derivs, num_of_modes=None, Nq_factor=30):
    if num_of_modes is None:
        num_of_modes = instanton_coeffs.shape[0]
    dim = instanton_coeffs.shape[1]
    Nq = instanton_coeffs.shape[0]*Nq_factor
    qts, quad_weights = clenshaw_curtis(Nq+1)
    ts = 0.5*(qts+1)*Tf
    
    x_inst, xdot_inst = eval_path(x0, x1, ts, Tf, instanton_coeffs)
    L_x_x, L_xd_x, L_xd_xd = L_2nd_derivs(x_inst, xdot_inst)
    
    phis = []
    dphis = []
    for k in range(1, num_of_modes+1):
        sqr_lambdak = (Tf / (np.pi*k))
        phi = np.sqrt(2/Tf) * np.sin(ts/sqr_lambdak) * sqr_lambdak
        dphi = np.sqrt(2/Tf) * np.cos(ts/sqr_lambdak)
        phis.append(phi)
        dphis.append(dphi)
        
    phis = np.array(phis)
    dphis = np.array(dphis)
                    
    Q =  np.einsum('ijn,kn,ln,n->ijkl', L_x_x, phis, phis, quad_weights) * Tf/2
    Q += 2*np.einsum('ijn,kn,ln,n->ijkl', L_xd_x, phis, dphis, quad_weights) * Tf/2
    Q += np.einsum('ijn,kn,ln,n->ijkl', L_xd_xd, dphis, dphis, quad_weights) * Tf/2
    
    Qf = flatten_Q(Q)
    Qf = (Qf + Qf.T)/2
    Q = unflatten_Qf(Qf, dim) # To make Q symmetric as well

    return Q, Qf

def compute_banded_Q_matrix(instanton_coeffs, x0, x1, Tf, L_2nd_derivs, band_size, Nm=None, Nq_factor=30):
    if Nm is None:
        Nm = instanton_coeffs.shape[0]
    dim = instanton_coeffs.shape[1]

    Nq_max_exponent = int(np.max(np.log2(Nm*Nq_factor)))

    qts_sets = []
    quad_weights_sets = []
    for Nq_exponent in range(1, Nq_max_exponent+1):
        Nq = 2**Nq_exponent
        qts, quad_weights = clenshaw_curtis(Nq)
        ts = 0.5*(qts+1)*Tf

        qts_sets.append(qts)
        quad_weights_sets.append(quad_weights)
        
    qts, quad_weights = qts_sets[-1], quad_weights_sets[-1]
    
    x_inst, xdot_inst = eval_path(x0, x1, ts, Tf, instanton_coeffs)
    L_x_x, L_xd_x, L_xd_xd = L_2nd_derivs(x_inst, xdot_inst)
    
    phis = []
    dphis = []
    for k in range(1, Nm+1):
        sqr_lambdak = (Tf / (np.pi*k))
        phi = np.sqrt(2/Tf) * np.sin(ts/sqr_lambdak) * sqr_lambdak
        dphi = np.sqrt(2/Tf) * np.cos(ts/sqr_lambdak)
        phis.append(phi)
        dphis.append(dphi)
    
    phis = np.array(phis) * np.sqrt(Tf/2) # The Tf/2 factor is for scaling the to physical time from the quadrature
    dphis = np.array(dphis) * np.sqrt(Tf/2)
    
    Q = np.zeros( (dim, dim, Nm, Nm) )
    
    for k in range(Nm):
        k_band = min(k+band_size, Nm)
        Nq_exponent = int(np.max(np.log2(k_band*Nq_factor)))
        qts, quad_weights = qts_sets[Nq_exponent-1], quad_weights_sets[Nq_exponent-1]
        elem_skip = 2**(Nq_max_exponent-Nq_exponent)
        
        Q[:,:,k,k:k_band] = np.einsum('ijn,n,ln,n->ijl',  L_x_x[:,:,::elem_skip],   phis[k,::elem_skip],  phis[k:k_band,::elem_skip], quad_weights) 
        Q[:,:,k,k:k_band] += np.einsum('ijn,n,ln,n->ijl', L_xd_x[:,:,::elem_skip],  phis[k,::elem_skip],  dphis[k:k_band,::elem_skip], quad_weights) 
        Q[:,:,k,k:k_band] += np.einsum('ijn,n,ln,n->ijl', L_xd_x[:,:,::elem_skip],  dphis[k,::elem_skip], phis[k:k_band,::elem_skip], quad_weights) 
        Q[:,:,k,k:k_band] += np.einsum('ijn,n,ln,n->ijl', L_xd_xd[:,:,::elem_skip], dphis[k,::elem_skip], dphis[k:k_band,::elem_skip], quad_weights) 
        Q[:,:,k:k_band,k] = Q[:,:,k,k:k_band]
    
    Qf = flatten_Q(Q)

    return Q, Qf

def compute_Q_matrix_old(instanton_coeffs, x0, x1, Tf, L_2nd_derivs, Nt_factor=1000):
    num_of_modes = instanton_coeffs.shape[0]
    dim = instanton_coeffs.shape[1]
    Nq = num_of_modes*100
    qts, quad_weights = clenshaw_curtis(Nq+1)
    ts = 0.5*(qts+1)*Tf
    
    Nt = Nt_factor * num_of_modes
    ts = np.linspace(0, Tf, Nt)
    
    x_inst, xdot_inst = eval_path(x0, x1, ts, Tf, instanton_coeffs)
    L_x_x, L_xd_x, L_xd_xd = L_2nd_derivs(x_inst, xdot_inst)
    
    phis = []
    dphis = []
    for k in range(1, num_of_modes+1):
        sqr_lambdak = (Tf / (np.pi*k))
        phi = np.sqrt(2/Tf) * np.sin(ts/sqr_lambdak) * sqr_lambdak
        dphi = np.sqrt(2/Tf) * np.cos(ts/sqr_lambdak)
        phis.append(phi)
        dphis.append(dphi)
    
    Q = np.zeros( (dim, dim, num_of_modes, num_of_modes) )
    for i in range(dim):
        for j in range(dim):
            for k in range(num_of_modes):
                for l in range(num_of_modes):
                    integrand = 0
                    integrand += L_x_x[i,j] * phis[k] * phis[l]
                    integrand += 2*L_xd_x[i,j] * dphis[k] * phis[l]
                    integrand += L_xd_xd[i,j] * dphis[k] * dphis[l]
                    Q[i,j,k,l] = np.trapz(integrand, x=ts)

    Qf = flatten_Q(Q)
    Qf = (Qf + Qf.T)/2
    Q = unflatten_Qf(Qf, dim) # To make Q symmetric as well

    return Q, Qf

def get_Q_distributions(inst_coeffs_list, x0, x1, Tf, hessL, Nq_factor=30, verbose=True, band_size=None, Nm=None):
    if type(inst_coeffs_list) != list:
        inst_coeffs_list = [inst_coeffs_list]

    if Nm is None:
        Nm = inst_coeffs_list[0].shape[0]
    dim = inst_coeffs_list[0].shape[1]
    Sigmas = []
    Qs = []
    Qfs = []
    Q_dets = []
    Zs = []
    
    for inst_coeffs in inst_coeffs_list:
        if band_size is None:
            Q, Qf = compute_Q_matrix(inst_coeffs, x0, x1, Tf, hessL, Nq_factor=Nq_factor)
        else:
            Q, Qf = compute_banded_Q_matrix(inst_coeffs, x0, x1, Tf, hessL, Nm=Nm, Nq_factor=Nq_factor, band_size=band_size)

        eigs = np.linalg.eig(Qf)[0]
        if verbose and np.count_nonzero(eigs<0) > 0:
            print("WARNING: Q has negative eigenvalues.")

        Sigma = np.linalg.inv(Qf)
        Q_det_sign, log_Q_det = np.linalg.slogdet(Qf)
        Sigmas.append(Sigma)
        Qs.append(Q)
        Qfs.append(Qf)
        Q_dets.append( (Q_det_sign, log_Q_det) )
        Z_sign, log_Z = Q_det_sign, 0.5*( Qf.shape[0]*np.log(2*np.pi) - log_Q_det )
        Zs.append( (Z_sign, log_Z) )

        if verbose and Q_det_sign == 0:
            print("WARNING: Q is singular")

        if verbose and Q_det_sign == -1:
            print("WARNING: Q has negative determinant")
    
    dist_funcs = []

    for inst_coeffs, Qf, Sigma, (Z_sign, log_Z) in zip(inst_coeffs_list, Qfs, Sigmas, Zs):
        def get_funcs(inst_coeffs, Qf, Sigma, log_Z):
            # Add zeros to inst_coeffs to match the Qf dimension
            _inst_coeffs = inst_coeffs
            inst_coeffs = np.zeros( (Nm, dim) )
            inst_coeffs[:_inst_coeffs.shape[0], :] = _inst_coeffs
            
            finst_coeffs = inst_coeffs.flatten()

            def Q_gen():
                return np.random.multivariate_normal(finst_coeffs, Sigma).reshape( (Nm, dim) )

            def Q_action(state):
                dstate = state.flatten() - finst_coeffs
                return 0.5*dstate.dot(Qf).dot(dstate) + log_Z

            return Q_gen, Q_action

        if Z_sign < 1:
            dist_funcs.append( (None, None) )
        else:
            dist_funcs.append( get_funcs(inst_coeffs, Qf, Sigma, log_Z) )

    if len(inst_coeffs_list) == 1:
        dist_funcs = dist_funcs[0]
        Qs = Qs[0]
        Qfs = Qfs[0]
        Sigmas = Sigmas[0]
        Zs = Zs[0]

    return dist_funcs, Qs, Qfs, Sigmas, Zs

"""