# cython: unraisable_tracebacks=True
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from modemc.defs import DTYPE
from modemc import utils, quad, variations
import numpy as np
import scipy

cpdef gradient_descent(System system, np.ndarray state0, int Nq,
                    int max_steps=1000000, DTYPE_t ag_c=1e-6, DTYPE_t S_tol=1e-16, DTYPE_t gradS_tol=1e-16, DTYPE_t step_size0=0.1, DTYPE_t step_size_rate=0.99, DTYPE_t min_step_size=1e-60,
                    quad_scheme=None, bint verbose=False, bint use_OM=True):
    
    cdef int Nm, dim
    dim, Nm = state0.shape[0], state0.shape[1]
    cdef np.ndarray x0, x1
    x0, x1 = system.x0, system.x1
    cdef DTYPE_t Tf = system.Tf

    if quad_scheme is None:
        quad_scheme = quad.clenshaw_curtis
    cdef np.ndarray qts, quad_weights, ts
    ts, qts, quad_weights = quad_scheme(Nq, Tf)
    
    cdef np.ndarray wphis, wdphis
    wphis, wdphis = utils.get_fast_gradS_basis(x0, x1, ts, Tf, Nm, quad_weights)

    cdef np.ndarray bb_mean, dbb_mean, phis, dphis
    bb_mean, dbb_mean, phis, dphis = utils.get_fast_eval_path_basis(x0, x1, ts, Tf, Nm)

    cdef np.ndarray path = np.zeros((dim, Nq), dtype=DTYPE)
    cdef np.ndarray d_path = np.zeros((dim, Nq), dtype=DTYPE)
    cdef np.ndarray force = np.zeros((dim, Nq), dtype=DTYPE)
    cdef np.ndarray div_force = np.zeros(Nq, dtype=DTYPE)
    cdef np.ndarray lagrangian = np.zeros(Nq, dtype=DTYPE)
    cdef np.ndarray L_x = np.zeros((dim, Nq), dtype=DTYPE)
    cdef np.ndarray L_xd = np.zeros((dim, Nq), dtype=DTYPE)
    cdef np.ndarray gradS_arr = np.zeros((dim, Nm), dtype=DTYPE)
    cdef np.ndarray gradS_buffer = np.zeros((dim, Nm), dtype=DTYPE)
   
    cdef np.ndarray state_arr = np.copy(state0).astype(DTYPE)
    cdef np.ndarray new_state_arr = np.zeros((dim, Nm), dtype=DTYPE)

    cdef DTYPE_t[:,:] state = state_arr
    cdef DTYPE_t[:,:] new_state = new_state_arr
    cdef DTYPE_t[:,:] gradS = gradS_arr
    
    cdef DTYPE_t S, new_S, m, gradS_norm
    cdef int gradS_size = Nm*dim
    
    cdef int steps = int(max_steps)
    cdef DTYPE_t step_size = step_size0
    
    cdef bint armijo_goldstein_succeeded
    
    cdef int i, j, k
    
    end_status = 'max_calcs'

    cutils._fast_eval_path(bb_mean, dbb_mean, phis, dphis, state_arr, path, d_path)
    if use_OM:
        lagrangians._compute_OM_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)
    else:
        lagrangians._compute_FW_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)
    S = cutils._eval_action(quad_weights, Tf, lagrangian)

    for i in range(steps):
        
        if use_OM:
            system._compute_gradL_OM(ts, path, d_path, L_x, L_xd)
        else:
            system._compute_gradL_FW(ts, path, d_path, L_x, L_xd)
        cutils._eval_action_gradient(wphis, wdphis, L_x, L_xd, gradS_buffer, gradS_arr)
        
        # Armijo-Goldstein
        
        armijo_goldstein_succeeded = False
        while not armijo_goldstein_succeeded:
            
            for j in range(dim):
                for k in range(Nm):
                    new_state[j,k] = state[j,k] - step_size*gradS[j,k]
                    
            cutils._fast_eval_path(bb_mean, dbb_mean, phis, dphis, new_state_arr, path, d_path)
            if use_OM:
                lagrangians._compute_OM_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)
            else:
                lagrangians._compute_FW_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)
            new_S = cutils._eval_action(quad_weights, Tf, lagrangian)
            
            m = np.dot(gradS_arr.flatten(), gradS_arr.flatten())
            
            armijo_goldstein_succeeded = new_S <= (S - step_size*m*ag_c)
            
            if not armijo_goldstein_succeeded:
                step_size *= step_size_rate
                
            if step_size < min_step_size:
                break
        
        state_arr[:] = new_state_arr
        
        if S_tol > 0 and abs(new_S - S) < S_tol:
            end_status = 'S_tol'
            break

        gradS_norm = sqrt(gradS_arr.flatten().dot(gradS_arr.flatten()) / gradS_size)
        if gradS_tol > 0 and gradS_norm < gradS_tol:
            end_status = 'gradS_tol'
            break

        if step_size < min_step_size:
            end_status = 'min_step_size'
            break

        if verbose and i % 1000 == 0:
            print('i=%s\tstep_size=%s\t|gradS|=%s\tdS=%s\tS=%s' % (
                "{:.4e}".format(i),
                "{:.2e}".format(step_size),
                "{:.2e}".format(gradS_norm),
                "{:.2e}".format(new_S - S),
                "{:.5e}".format(new_S)), end='\r')

        S = new_S
            
    # Project to KL basis
    
    path0, d_path0 = utils.eval_path(x0, x1, ts, Tf, state0)
    path, d_path = utils.eval_path(x0, x1, ts, Tf, state_arr)

    S0 = utils.compute_action(system, state0, Nq=Nq, use_relative=False, use_OM=use_OM)
    S = utils.compute_action(system, state_arr, Nq=Nq, use_relative=False, use_OM=use_OM)

    result = {
        'path0' : path0,
        'd_path0' : d_path0,
        'path' : path,
        'd_path' : d_path,
        'ts' : ts,
        'S0' : S0,
        'S' : S,
        'state0' : np.array(state0),
        'state' : np.array(state),
        
        'end_step' : i,
        'end_status' : end_status,
        'dS' : new_S - S,
        '|gradS|' : np.sqrt(gradS_arr.flatten().dot(gradS_arr.flatten())),
        'step_size' : step_size,
        'gradS' : gradS_arr,
    }

    if verbose:
        print('',end='\r')
        print('Instanton finder ended (reason: %s): i=%s\tstep_size=%s\t|gradS|=%s\tdS=%s\tS=%s' % (
            end_status,
            "{:.4e}".format(i),
            "{:.2e}".format(step_size),
            "{:.2e}".format(np.sqrt(gradS_arr.flatten().dot(gradS_arr.flatten()))),
            "{:.2e}".format(new_S - S),
            "{:.6e}".format(new_S)))
    
    return result


cpdef newtons_method(System system, np.ndarray state0, int Nq, int Q_band_size,
                    int max_steps=1000000, DTYPE_t ag_c=1e-6, DTYPE_t S_tol=1e-16, DTYPE_t gradS_tol=1e-16, DTYPE_t step_vector_tol=1e-16, DTYPE_t step_size0=0.1, DTYPE_t step_size_rate=0.99, DTYPE_t min_step_size=1e-60,
                    quad_scheme=None, bint verbose=False, bint use_OM=True):
    
    cdef int Nm, dim
    dim, Nm = state0.shape[0], state0.shape[1]
    cdef np.ndarray x0, x1
    x0, x1 = system.x0, system.x1
    cdef DTYPE_t Tf = system.Tf

    if quad_scheme is None:
        quad_scheme = quad.clenshaw_curtis
    cdef np.ndarray qts, quad_weights, ts
    ts, qts, quad_weights = quad_scheme(Nq, Tf)
    
    cdef np.ndarray wphis, wdphis
    wphis, wdphis = utils.get_fast_gradS_basis(x0, x1, ts, Tf, Nm, quad_weights)

    cdef np.ndarray bb_mean, dbb_mean, phis, dphis
    bb_mean, dbb_mean, phis, dphis = utils.get_fast_eval_path_basis(x0, x1, ts, Tf, Nm)

    cdef np.ndarray path = np.zeros((dim, Nq), dtype=DTYPE)
    cdef np.ndarray d_path = np.zeros((dim, Nq), dtype=DTYPE)
    cdef np.ndarray force = np.zeros((dim, Nq), dtype=DTYPE)
    cdef np.ndarray div_force = np.zeros(Nq, dtype=DTYPE)
    cdef np.ndarray lagrangian = np.zeros(Nq, dtype=DTYPE)
    cdef np.ndarray L_x = np.zeros((dim, Nq), dtype=DTYPE)
    cdef np.ndarray L_xd = np.zeros((dim, Nq), dtype=DTYPE)
    cdef np.ndarray gradS_arr = np.zeros((dim, Nm), dtype=DTYPE)
    cdef np.ndarray gradS_buffer = np.zeros((dim, Nm), dtype=DTYPE)
   
    cdef np.ndarray state_arr = np.copy(state0).astype(DTYPE)
    cdef np.ndarray new_state_arr = np.zeros((dim, Nm), dtype=DTYPE)

    cdef DTYPE_t[:,:] state = state_arr
    cdef DTYPE_t[:,:] new_state = new_state_arr
    cdef DTYPE_t[:,:] gradS = gradS_arr
    
    cdef DTYPE_t S, new_S, m, gradS_norm
    cdef int gradS_size = Nm*dim
    
    cdef int steps = int(max_steps)
    cdef DTYPE_t step_size = step_size0
    
    cdef bint armijo_goldstein_succeeded
    
    cdef int i, j, l, k, r
    
    end_status = 'max_calcs'

    ## Hessian init ###

    cdef DTYPE_t Nq_factor = Nq/float(Nm)
    cdef int Nq_max_exponent = int(np.max(np.log2(Nq)))
    
    if quad_scheme is None:
        quad_scheme = quad.clenshaw_curtis

    hess_qts_sets = []
    hess_quad_weights_sets = []
    for Nq_exponent in range(1, Nq_max_exponent+1):
        Nq = 2**Nq_exponent
        hess_ts, hess_qts, hess_quad_weights = quad_scheme(Nq+1, Tf) # Must add +1 so that the quadratures line up

        hess_qts_sets.append(hess_qts)
        hess_quad_weights_sets.append(hess_quad_weights)
        
    cdef int hess_Nt = len(hess_ts)

    cdef np.ndarray Q = np.zeros( (dim, dim, Nm, Nm) )
    cdef np.ndarray Qxx = np.zeros( (dim, dim, Nm, Nm) )
    cdef np.ndarray Qxdx = np.zeros( (dim, dim, Nm, Nm) )
    cdef np.ndarray Qxdxd = np.zeros( (dim, dim, Nm, Nm) )

    cdef np.ndarray hess_path = np.zeros((dim, hess_Nt), dtype=DTYPE)
    cdef np.ndarray hess_d_path = np.zeros((dim, hess_Nt), dtype=DTYPE)

    cdef np.ndarray Qf = np.zeros((dim*Nm, dim*Nm))
    cdef np.ndarray inv_Qf_arr = np.zeros((dim*Nm, dim*Nm))
    cdef np.ndarray inv_Q_arr = np.zeros((dim, dim, Nm, Nm))
    cdef DTYPE_t[:,:,:,:] inv_Q = inv_Q_arr
    
    cdef np.ndarray step_vector_arr = np.zeros((dim, Nm))
    cdef DTYPE_t[:,:] step_vector = step_vector_arr

    cdef np.ndarray L_x_x = np.zeros( (dim, dim, hess_Nt))
    cdef np.ndarray L_xd_x = np.zeros( (dim, dim, hess_Nt))
    cdef np.ndarray L_xd_xd  = np.zeros( (dim, dim, hess_Nt))

    res = utils.get_fast_eval_path_basis(x0, x1, hess_ts, Tf, Nm)
    cdef np.ndarray hess_bb_mean = <np.ndarray>res[0]
    cdef np.ndarray hess_dbb_mean = <np.ndarray>res[1]
    cdef np.ndarray hess_phis = <np.ndarray>res[2]
    cdef np.ndarray hess_dphis = <np.ndarray>res[3]

    ###################

    cutils._fast_eval_path(bb_mean, dbb_mean, phis, dphis, state_arr, path, d_path)
    if use_OM:
        lagrangians._compute_OM_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)
    else:
        lagrangians._compute_FW_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)
    S = cutils._eval_action(quad_weights, Tf, lagrangian)

    for i in range(steps):

        if use_OM:
            system._compute_gradL_OM(ts, path, d_path, L_x, L_xd)
        else:
            system._compute_gradL_FW(ts, path, d_path, L_x, L_xd)
        cutils._eval_action_gradient(wphis, wdphis, L_x, L_xd, gradS_buffer, gradS_arr)

        # Compute Hessian

        variations._compute_banded_Q_matrix(hess_bb_mean, hess_dbb_mean, hess_phis, hess_dphis, hess_phis, hess_dphis, hess_quad_weights_sets, Q, Qxx, Qxdx, Qxdxd, L_x_x, L_xd_x, L_xd_xd, hess_path, hess_d_path, ts,
                        system, state_arr, Nq_factor, Q_band_size, Nm, Nq_max_exponent, use_OM)
        variations._flatten_Q(Q, Qf)
        inv_Qf_arr = scipy.linalg.inv(Qf, overwrite_a=True, check_finite=False)
        variations._unflatten_Qf(inv_Qf_arr, dim, inv_Q_arr)

        #gradS_flat = gradS_arr.flatten()
        #step_vector_flat = inv_Qf_arr.dot(gradS_flat)

        #for j in range(dim):
        #    for k in range(Nm):
        #        step_vector[j,k] = step_vector_flat[j*Nm + k]

        # Compute step vector
        for j in range(dim):
            for k in range(Nm):
                step_vector[j,k] = 0
        
        for j in range(dim):
            for r in range(dim):
                for k in range(Nm):
                    for l in range(Nm):
                        step_vector[j,k] += inv_Q[j,r,k,l]*gradS[r,l]

        # Armijo-Goldstein
        
        #armijo_goldstein_succeeded = False
        #while not armijo_goldstein_succeeded:
        #    
        #    for j in range(dim):
        #        for k in range(Nm):
        #            new_state[j,k] = state[j,k] - step_size*step_vector[j,k]
        #            
        #    cutils._fast_eval_path(bb_mean, dbb_mean, phis, dphis, new_state_arr, path, d_path)
        #    if use_OM:
        #        lagrangians._compute_OM_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)
        #    else:
        #        lagrangians._compute_FW_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)
        #    new_S = cutils._eval_action(quad_weights, Tf, lagrangian)
        #    
        #    m = np.dot(gradS_arr.flatten(), gradS_arr.flatten())
        #    
        #    armijo_goldstein_succeeded = new_S <= (S - step_size*m*ag_c)
        #    
        #    if not armijo_goldstein_succeeded:
        #        step_size *= step_size_rate
        #        
        #    if step_size < min_step_size:
        #        break

        for j in range(dim):
            for k in range(Nm):
                new_state[j,k] = state[j,k] - step_size*step_vector[j,k]
                
        cutils._fast_eval_path(bb_mean, dbb_mean, phis, dphis, new_state_arr, path, d_path)
        if use_OM:
            lagrangians._compute_OM_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)
        else:
            lagrangians._compute_FW_lagrangian(system, ts, path, d_path, force, div_force, lagrangian)
        new_S = cutils._eval_action(quad_weights, Tf, lagrangian)
        
        state_arr[:] = new_state_arr
        
        if S_tol > 0 and abs(new_S - S) < S_tol:
            end_status = 'S_tol'
            break

        gradS_norm = sqrt(gradS_arr.flatten().dot(gradS_arr.flatten()) / gradS_size)
        if gradS_tol > 0 and gradS_norm < gradS_tol:
            end_status = 'gradS_tol'
            break

        step_vector_norm = sqrt(step_vector_arr.flatten().dot(step_vector_arr.flatten()) / gradS_size)
        if step_vector_tol > 0 and step_vector_norm < step_vector_tol:
            end_status = 'step_vector_tol'
            break

        if step_size < min_step_size:
            end_status = 'min_step_size'
            break

        if verbose and i % 10 == 0:
            print('i=%s\tstep_size=%s\t|gradS|=%s\tdS=%s\tS=%s' % (
                "{:.4e}".format(i),
                "{:.2e}".format(step_size),
                "{:.2e}".format(gradS_norm),
                "{:.2e}".format(new_S - S),
                "{:.5e}".format(new_S)), end='\r')

        S = new_S
            
    # Project to KL basis
    
    path0, d_path0 = utils.eval_path(x0, x1, ts, Tf, state0)
    path, d_path = utils.eval_path(x0, x1, ts, Tf, state_arr)

    S0 = utils.compute_action(system, state0, Nq=Nq, use_relative=False, use_OM=use_OM)
    S = utils.compute_action(system, state_arr, Nq=Nq, use_relative=False, use_OM=use_OM)

    result = {
        'path0' : path0,
        'd_path0' : d_path0,
        'path' : path,
        'd_path' : d_path,
        'ts' : ts,
        'S0' : S0,
        'S' : S,
        'state0' : np.array(state0),
        'state' : np.array(state),
        
        'end_step' : i,
        'end_status' : end_status,
        'dS' : new_S - S,
        '|gradS|' : np.sqrt(gradS_arr.flatten().dot(gradS_arr.flatten())),
        'step_size' : step_size,
        'gradS' : gradS_arr,
    }

    if verbose:
        print('',end='\r')
        print('Instanton finder ended (reason: %s): i=%s\tstep_size=%s\t|gradS|=%s\tdS=%s\tS=%s' % (
            end_status,
            "{:.4e}".format(i),
            "{:.2e}".format(step_size),
            "{:.2e}".format(np.sqrt(gradS_arr.flatten().dot(gradS_arr.flatten()))),
            "{:.2e}".format(new_S - S),
            "{:.6e}".format(new_S)))
    
    return result