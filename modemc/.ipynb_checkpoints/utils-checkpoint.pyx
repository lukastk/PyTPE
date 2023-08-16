# cython: profile=False
# cython: unraisable_tracebacks=True
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from modemc.defs import DTYPE
from modemc import quad
import numpy as np
import scipy.optimize
import pyfftw

cpdef compute_force_and_div_force(System system, ts, state, use_relative=False, use_OM=True):
    path, d_path = eval_path(system.x0, system.x1, ts, system.Tf, state)
    dim, Nt = path.shape
    force = np.zeros((dim, Nt))
    div_force = np.zeros(Nt)
    system._compute_force_and_div_force(ts, path, force, div_force)

    return force, div_force

cpdef compute_lagrangian(System system, ts, state, use_relative=False, use_discretised=False, use_OM=True):
    path, d_path = eval_path(system.x0, system.x1, ts, system.Tf, state)
    return compute_lagrangian_from_path(system, path, d_path, use_relative=use_relative, use_discretised=use_discretised, use_OM=use_OM, ts=ts)

cpdef compute_lagrangian_from_path(System system, path, d_path, use_relative=False, use_discretised=False, use_OM=True, ts=None):
    dim, Nt = path.shape
    lagrangian = np.zeros(Nt)
    force = np.zeros((dim, Nt))
    div_force = np.zeros(Nt)

    if ts is None:
        ts = np.linspace(0, system.Tf, Nt) # Presume equal spacing

    if use_OM:
        if use_relative:
            if use_discretised:
                lagrangians._compute_relative_OM_discretised_lagrangian(system, ts, path, force, div_force, lagrangian)
            else:
                lagrangians._compute_relative_OM_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)
        else:
            if use_discretised:
                lagrangians._compute_OM_discretised_lagrangian(system, ts, path, force, div_force, lagrangian)
            else:
                lagrangians._compute_OM_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)
    else:
        if use_relative:
            if use_discretised:
                lagrangians._compute_relative_FW_discretised_lagrangian(system, ts, path, force, div_force, lagrangian)
            else:
                lagrangians._compute_relative_FW_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)
        else:
            if use_discretised:
                lagrangians._compute_FW_discretised_lagrangian(system, ts, path, force, div_force, lagrangian)
            else:
                lagrangians._compute_FW_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)

    return lagrangian

cpdef compute_gradL(System system, ts, state, use_OM=True):
    path, d_path = eval_path(system.x0, system.x1, ts, system.Tf, state)
    dim, Nt = path.shape
    
    L_x = np.zeros((dim, Nt))
    L_xd = np.zeros((dim, Nt))
    
    if use_OM:
        system._compute_gradL_OM(ts, path, d_path, L_x, L_xd)
    else:
        system._compute_gradL_FW(ts, path, d_path, L_x, L_xd)

    return L_x, L_xd

cpdef compute_hessL(System system, ts, state, use_OM=True):
    path, d_path = eval_path(system.x0, system.x1, ts, system.Tf, state)
    dim, Nt = path.shape
    
    L_x_x = np.zeros((dim, dim, Nt))
    L_xd_x = np.zeros((dim, dim, Nt))
    L_xd_xd = np.zeros((dim, dim, Nt))
    
    if use_OM:
        system._compute_hessL_OM(ts, path, d_path, L_x_x, L_xd_x, L_xd_xd)
    else:
        system._compute_hessL_FW(ts, path, d_path, L_x_x, L_xd_x, L_xd_xd)

    return L_x_x, L_xd_x, L_xd_xd

def compute_action(system, state, Nq_factor=10, Nq=None, use_relative=False, use_discretised=False, use_OM=True, quad_scheme=None):
    dim = state.shape[0]
    Nm = state.shape[1]
    Tf = system.Tf

    if Nq is None:
        Nq = int(np.round(Nm * Nq_factor))

    if quad_scheme is None:
        quad_scheme = quad.clenshaw_curtis
    ts, qts, quad_weights = quad_scheme(Nq, Tf)

    lagrangian = compute_lagrangian(system, ts, state, use_relative=use_relative, use_discretised=use_discretised, use_OM=use_OM)
    return cutils._eval_action(quad_weights, Tf, lagrangian)

def fast_compute_action(System system, state, ts, quad_weights, fep_basis, use_relative=False, use_discretised=False, use_OM=True):
    bb_mean, dbb_mean, phis, dphis = fep_basis

    dim = state.shape[0]
    Tf = system.Tf

    path, d_path = fast_eval_path(state, fep_basis)
    lagrangian = compute_lagrangian_from_path(system, path, d_path, use_relative=use_relative, use_discretised=use_discretised, use_OM=use_OM, ts=ts)
    return cutils._eval_action(quad_weights, Tf, lagrangian)

def compute_gaussian_action(System system, state):
    return cutils._eval_gaussian_action(system.x0, system.x1, system.Tf, system.beta, system.gamma, state)

cpdef compute_gradS(System system, state, Nq_factor=10, Nq=None, use_OM=True, quad_scheme=None):
    dim, Nm = state.shape
    x0, x1 = system.x0, system.x1
    Tf = system.Tf

    if Nq is None:
        Nq = int(np.round(Nm * Nq_factor))

    if quad_scheme is None:
        quad_scheme = quad.clenshaw_curtis
    ts, qts, quad_weights = quad_scheme(Nq, Tf)
    
    wphis, wdphis = get_fast_gradS_basis(x0, x1, ts, Tf, Nm, quad_weights)
    
    L_x, L_xd = compute_gradL(system, ts, state, use_OM=use_OM)
    
    out_gradS = np.zeros((dim, Nm), dtype=DTYPE)
    out_gradS_buffer = np.zeros((dim, Nm), dtype=DTYPE)
    
    cutils._eval_action_gradient(wphis, wdphis, L_x, L_xd, out_gradS_buffer, out_gradS)
    
    return out_gradS

def get_fast_eval_path_basis(x0, x1, ts, Tf, Nm):
    dim = np.size(x0)
    Nt = len(ts)

    if dim == 1:
        x0 = np.array([ float(x0) ])
        x1 = np.array([ float(x1) ])

    phis = np.zeros( (Nm, Nt) )
    dphis = np.zeros( (Nm, Nt) )
    for k in range(Nm):
        phis[k,:] = np.sin(np.pi*(k+1)*ts/Tf) * (np.sqrt(2*Tf) / np.pi) / (k+1)
        dphis[k,:] = np.cos(np.pi*(k+1)*ts/Tf) * np.sqrt(2 / Tf)

    bb_mean = np.zeros( (dim, Nt) )
    dbb_mean = np.zeros( (dim, Nt) ) 
    for i in range(dim):
        bb_mean[i, :] += x0[i] + (ts/Tf)*(x1[i] - x0[i])
        dbb_mean[i, :] += (1/Tf)*(x1[i] - x0[i])
        
    return bb_mean, dbb_mean, phis, dphis
           
def get_fast_gradS_basis(x0, x1, ts, Tf, Nm, quad_weights):
    bb_mean, dbb_mean, phis, dphis = get_fast_eval_path_basis(x0, x1, ts, Tf, Nm)
    
    ws = np.diag(quad_weights)
    wphis = (phis.dot(ws)).T
    wdphis = (dphis.dot(ws)).T
    
    return wphis, wdphis
    
cpdef fast_eval_path(state, fep_basis):
    bb_mean, dbb_mean, phis, dphis = fep_basis
    Nm, Nt = phis.shape
    dim = bb_mean.shape[0]

    path = np.zeros((dim, Nt))
    d_path = np.zeros((dim, Nt))
    cutils._fast_eval_path(bb_mean, dbb_mean, phis, dphis, state, path, d_path)

    return path, d_path

def eval_path(x0, x1, ts, Tf, state):
    Nm = state.shape[1]
    fep_basis = get_fast_eval_path_basis(x0, x1, ts, Tf, Nm)
    return fast_eval_path(state, fep_basis)

cpdef find_suitable_Nq(System system, states, Nq_min, Nq_max, Nq_step, sample_num, use_relative, use_OM,
            rel_tol=1e-5, abs_tol=None, use_mean_error=True, quad_scheme=None):
    if not abs_tol is None and not rel_tol is None:
        raise Exception('Cannot specify both abs_tol and rel_tol.')

    x0, x1 = system.x0, system.x1
    Tf = system.Tf
    Nm = states.shape[1]
    Nqs = np.arange(Nq_min, Nq_max, Nq_step)
    Nq_actions = np.zeros((Nqs.size, sample_num))

    state_choices = np.random.choice(states.shape[0], min(sample_num, states.shape[0]))

    if quad_scheme is None:
        quad_scheme = quad.clenshaw_curtis

    quadrature_arrs = []
    fep_bases = []
    for i in range(len(Nqs)):
        ts, qts, quad_weights = quad_scheme(Nqs[i], Tf)
        quadrature_arrs.append((qts, quad_weights, ts))
        
        fep_bases.append(get_fast_eval_path_basis(x0, x1, ts, Tf, Nm))

    for k, i in enumerate(state_choices):
        for j in range(len(Nqs)):
            qts, quad_weights, ts = quadrature_arrs[j]
            Nq_actions[j,k] = fast_compute_action(system, states[i,:,:], ts, quad_weights, fep_bases[j], Nq=Nqs[j], use_relative=use_relative, use_OM=use_OM)
            
    #rel_err_ens = np.abs(1-Nq_actions[1:]/Nq_actions[:-1])
    #rel_err = np.max(rel_err_ens, axis=1)

    abs_err_ens = np.abs(Nq_actions[1:] - Nq_actions[:len(Nq_actions)-1])
    #abs_err = np.max(abs_err_ens, axis=1)
    
    mean_abs_err = np.mean(abs_err_ens, axis=1)
    max_abs_err = np.max(abs_err_ens, axis=1)

    rel_den = np.min(np.array([np.abs(Nq_actions[1:]), np.abs(Nq_actions[:len(Nq_actions)-1])]), axis=0)
    rel_err_ens = abs_err_ens / rel_den
    #rel_err = np.max(rel_err_ens, axis=1)
    
    mean_rel_err = np.mean(rel_err_ens, axis=1)
    max_rel_err = np.max(rel_err_ens, axis=1)
    
    if use_mean_error:
        abs_err = mean_abs_err
        rel_err = mean_rel_err
    else:
        abs_err = max_abs_err
        rel_err = max_rel_err

    if not rel_tol is None:
        f = np.where(rel_err < rel_tol)[0]
        if len(f) == 0:
            found_Nq = -1
        else:
            found_Nq = Nqs[f[0]]
    else:
        f = np.where(abs_err < abs_tol)[0]
        if len(f) == 0:
            found_Nq = -1
        else:
            found_Nq = Nqs[f[0]]

    data = {
        'Nqs' : Nqs[1:],
        'rel_err_ens' : rel_err_ens,
        'mean_rel_err' : mean_rel_err,
        'max_rel_err' : max_rel_err,
        'abs_err_ens' : abs_err_ens,
        'mean_abs_err' : mean_abs_err,
        'max_abs_err' : max_abs_err
    }

    return found_Nq, data

def get_batch_size(Nm, Nq, dim, N_steps, use_GB=True):
    DTYPE_size = np.dtype(DTYPE).itemsize
    bool_size = np.dtype(bool).itemsize
    
    N_steps_pp = N_steps+1
    
    batch_size = 0
    batch_size += DTYPE_size * N_steps_pp * Nm * dim # states_arr
    batch_size += bool_size * N_steps_pp * Nm * dim # accepts_arr
    batch_size += DTYPE_size * N_steps_pp * Nm # actions_arr
    batch_size += DTYPE_size * N_steps_pp * Nq * dim # paths_arr
    batch_size += DTYPE_size * N_steps_pp * Nq * dim # d_paths_arr
    
    if use_GB:
        batch_size /= 1e9

    return float(batch_size)

def get_N_steps_to_match_batch_size(Nm, Nq, dim, batch_size, use_GB=True):
    batch_size_k = get_batch_size(Nm, Nq, dim, 0, use_GB=use_GB)
    N_steps = int(np.floor(batch_size / batch_size_k - 1))
    return N_steps


def get_brownian_bridge_KL_basis(N_modes, dim, ts):
    Tf = ts[len(ts)-1]
    basis = np.zeros([N_modes, len(ts)],dtype=float)
    
    for i in range(N_modes):
        norm = 0.5*Tf
        eig = (np.sqrt(2*Tf) / np.pi) / ( (i+1) )
        basis[i] = np.sin((i+1)*np.pi*ts/Tf)
        basis[i] *= 1 / (eig * norm)
    
    return basis

def project_onto_basis(trajectory, basis, ts):
    if len(trajectory.shape) == 1:
        trajectory = trajectory.reshape( (1, len(trajectory)) )

    Tf = ts[ts.shape[0]-1]
    x0 = trajectory[:, 0]
    x1 = trajectory[:, trajectory.shape[1]-1]
    dim = trajectory.shape[0]
    N_modes = basis.shape[0]
    
    mode_coefficients = np.zeros((dim, basis.shape[0]),dtype=float)
    mean_zero_trajectory = np.zeros(trajectory.shape)
    for i in range(dim):
        mean_zero_trajectory[i,:] = trajectory[i,:] - (x0[i] + (ts/Tf)*(x1[i] - x0[i]))

    for i in range(dim):
        for j in range(N_modes):
            mode_coefficients[i, j] = np.trapz(basis[j]*mean_zero_trajectory[i,:], ts)
        
    return mode_coefficients

cpdef get_normal_vector(size):
    _noise_vector = np.zeros(size, dtype=DTYPE)
    cdef DTYPE_t[:] noise_vector = _noise_vector
    cutils.gaussian_vector(noise_vector)
    return noise_vector

def find_kappa(MCMC_protocol, targ_acc, trial_N_sim, maxiter, MCMC_params,
                kappa_keyword='kappa', batch_size=1):

    MCMC_params = MCMC_params.copy()
    MCMC_params['Nm_save'] = 0

    if trial_N_sim == 0:
        return 0.5, -1
    
    iteration_count = 0

    def MCMC_trial(kappa):
        if kappa==0:
            return 1 - targ_acc
        elif kappa==1:
            return 0 - targ_acc

        MCMC_params[kappa_keyword] = kappa

        sim = MCMC_protocol(MCMC_params)
        accept_count = 0
        _, accepts, _, _, _ = sim.simulate(trial_N_sim, batch_size=batch_size, N_save=trial_N_sim, N_paths_save=0, use_GB=True, verbose=False)
        
        accept_rate = np.count_nonzero(accepts) / len(accepts)
        return accept_rate - targ_acc

    sol = scipy.optimize.root_scalar(MCMC_trial, bracket=[0, 1], method='brentq', maxiter=maxiter)
    kappa = sol.root
    dkappa_acc = MCMC_trial(sol.root)
    
    return kappa, dkappa_acc+targ_acc, dkappa_acc, sol

### Tests


cpdef test1_np_add():
    cdef np.ndarray a = np.random.random( (100, 100) )
    cdef np.ndarray b = np.random.random( (100, 100) )
    
    np.add(a, b, a)
            
cpdef test1_cython_add():
    cdef DTYPE_t[:,:] a = np.random.random( (100, 100) )
    cdef DTYPE_t[:,:] b = np.random.random( (100, 100) )
    
    for i in range(100):
        for j in range(100):
            a[i,j] = a[i,j] + b[i,j]
            
cpdef test1_cython_1d_add():
    cdef DTYPE_t[:] a = np.random.random( (100*100) )
    cdef DTYPE_t[:] b = np.random.random( (100*100) )
    
    for i in range(100*100):
        a[i] = a[i] + b[i]
        
cpdef test1_cython_1d_add_v2():
    cdef DTYPE_t[:,:] a = np.random.random( (2, 50*100) )
    cdef DTYPE_t[:,:] b = np.random.random( (2, 50*100) )
    
    for i in range(2):
        for j in range(50*100):
            a[i,j] = a[i,j] + b[i,j]

from cython.operator cimport dereference
import os, time
cpdef test_normal_vec1():
    cdef long max_long = 9223372036854775807
    cdef mt19937 _rng_gen = mt19937((abs(os.getpid()) + long(time.time()*1000)) % max_long)
    cdef normal_distribution[DTYPE_t] _normal_dist = normal_distribution[DTYPE_t](0.0,1.0)
    cdef mt19937* rng_gen = &_rng_gen
    cdef normal_distribution[DTYPE_t]* normal_dist = &_normal_dist

    dim, Nm = 2, 100
    s_arr = np.zeros( (dim, Nm), dtype=DTYPE )
    cdef DTYPE_t[:,:] s = s_arr

    for l in range(100):
        for j in range(dim):
            for k in range(Nm):
                s[j,k] = dereference(normal_dist)(dereference(rng_gen))

    return s_arr

cpdef test_normal_vec2():

    dim, Nm = 2, 100
    s_arr = np.zeros( (dim, Nm), dtype=DTYPE )
    cdef DTYPE_t[:,:] s = s_arr

    _noise_vector = np.zeros(Nm)
    cdef DTYPE_t[:] noise_vector = _noise_vector

    for l in range(100):
        for j in range(dim):
            cutils.gaussian_vector(noise_vector)
            for k in range(Nm):
                s[j,k] = noise_vector[k]
    
    return s_arr