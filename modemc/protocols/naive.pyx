#cython: unraisable_tracebacks=False
#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from modemc.defs import DTYPE
import numpy as np
from modemc import utils
from cython.operator cimport dereference

cdef class Naive:
    def __init__(self, params):
        super().__init__(params)

        self.kappa = params['kappa']
        self.noise_prefactor = np.sqrt(2/(self.beta*self.gamma))
        self.pkappa = self.noise_prefactor * self.kappa

        self.current_state = self.state0
        fep_basis = self.bb_mean, self.dbb_mean, self.phis, self.dphis
        self.current_path, self.current_d_path = utils.fast_eval_path(self.current_state, fep_basis)
        self.current_accept = True
        self.current_action = utils.compute_action(self.system, self.current_state, self.Nq_factor, use_relative=True, use_discretised=self.use_discretised_action, use_OM=self.use_OM, quad_scheme=self.quad_scheme)

    cdef int _run_step(self, int step_i) except -1:
        cdef:
            int dim = self.dim
            int Nm = self.Nm
            int Nq = self.Nq
            DTYPE_t Tf = self.Tf
            DTYPE_t gamma = self.gamma
            DTYPE_t beta = self.beta
            DTYPE_t pkappa = self.pkappa

            DTYPE_t[:,:] bb_mean = self.bb_mean
            DTYPE_t[:,:] dbb_mean = self.dbb_mean
            DTYPE_t[:,:] phis = self.phis
            DTYPE_t[:,:] dphis = self.dphis
            DTYPE_t[:] ts = self.ts
            DTYPE_t[:] quad_weights = self.quad_weights

            System system = self.system

            DTYPE_t[:,:,:] states = self.states_arr
            np.npy_bool[:] accepts = self.accepts_arr
            DTYPE_t[:] actions = self.actions_arr
            DTYPE_t[:,:,:] paths = self.paths_arr
            DTYPE_t[:,:,:] d_paths = self.d_paths_arr
            DTYPE_t[:,:] force = self.force_arr
            DTYPE_t[:] div_force = self.div_force_arr
            DTYPE_t[:] lagrangian = self.lagrangian_arr

            mt19937* rng_gen = &self.rng_gen
            normal_distribution[DTYPE_t]* normal_dist = &self.normal_dist
            uniform_real_distribution[DTYPE_t]* unif_dist = &self.unif_dist

            DTYPE_t acceptance_prob
            int j, k

        ### Generate proposal

        for j in range(dim):
            for k in range(Nm):
                # For efficiency, we store the proposal in the current state
                states[step_i,j,k] = states[step_i-1,j,k] + pkappa*dereference(normal_dist)(dereference(rng_gen))

        ### Compute acceptance probability
        
        cutils._fast_eval_path(self.bb_mean, self.dbb_mean, self.phis, self.dphis, self.states_arr[step_i,:,:], self.paths_arr[step_i,:,:], self.d_paths_arr[step_i,:,:])

        if self.use_OM:
            if self.use_discretised_action:
                lagrangians._compute_OM_discretised_lagrangian(system, ts, paths[step_i,:,:], force, div_force, lagrangian)
            else:
                lagrangians._compute_OM_lagrangian(system, ts, paths[step_i,:,:], d_paths[step_i,:,:], force, div_force, lagrangian)
        else:
            if self.use_discretised_action:
                lagrangians._compute_FW_discretised_lagrangian(system, ts, paths[step_i,:,:], force, div_force, lagrangian)
            else:
                lagrangians._compute_FW_lagrangian(system, ts, paths[step_i,:,:], d_paths[step_i,:,:], force, div_force, lagrangian)

        actions[step_i] = cutils._eval_action(quad_weights, Tf, lagrangian)

        acceptance_prob = min(1.0, exp(actions[step_i-1] - actions[step_i]))
        
        if dereference(unif_dist)(dereference(rng_gen)) < acceptance_prob:
            accepts[step_i] = True
        else:
            accepts[step_i] = False
            actions[step_i] = actions[step_i-1]

            for j in range(dim):
                for k in range(Nm):
                    states[step_i,j,k] = states[step_i-1,j,k]

            for j in range(dim):
                for k in range(Nq):
                    paths[step_i,j,k] = paths[step_i-1,j,k]
                    d_paths[step_i,j,k] = d_paths[step_i-1,j,k]
