# cython: profile=False
# cython: unraisable_tracebacks=True
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import pyfftw # IMPORTANT: There's a bug requiring us to import pyfftw before numpy

from modemc.defs import DTYPE
import numpy as np
from modemc import utils, quad
from cython.operator cimport dereference
import os, time
cimport cython

cimport modemc
from modemc.protocols cimport pCN
from modemc.protocols.pCN cimport Teleporter, GaussianTeleporter

cdef class pCN_FFT_Flip:

    cdef int _FFT_compute_path_and_dpath(self, DTYPE_t[:,:] state, DTYPE_t[:,:] out_path, DTYPE_t[:,:] out_d_path) except -1:
        cdef:
            DTYPE_t[:,:] path_FFT_in = self.path_FFT_in
            DTYPE_t[:,:] dpath_FFT_in = self.dpath_FFT_in
            DTYPE_t[:,:] path_FFT_out = self.path_FFT_out
            DTYPE_t[:,:] dpath_FFT_out = self.dpath_FFT_out
            DTYPE_t[:,:] bb_mean = self.bb_mean
            DTYPE_t[:] dbb_mean_const = self.dbb_mean_const
            DTYPE_t[:] path_coeffs_prefactors = self.path_coeffs_prefactors

            int i, k
            
            #DTYPE_t start_time

        #start_time = time.time()
        for i in range(self.dim):
            dpath_FFT_in[i,0] = dbb_mean_const[i]

            for k in range(self.Nm):
                path_FFT_in[i,k] = state[i,k] * path_coeffs_prefactors[k]
                dpath_FFT_in[i,k+1] = state[i,k] * self.dpath_coeffs_prefactor
                
            # This might be needed due to the fact that the input arrays can be destroyed
            for k in range(self.Nm, self.Nq-2):
                path_FFT_in[i,k] = 0
                dpath_FFT_in[i,k+1] = 0
            dpath_FFT_in[i,self.Nq-1] = 0
        #self.profile_timers[0] += time.time() - start_time

        #start_time = time.time()
        self.dst_transform()
        #self.profile_timers[1] += time.time() - start_time
        #start_time = time.time()
        self.dct_transform()
        #self.profile_timers[2] += time.time() - start_time

        #start_time = time.time()
        for i in range(self.dim):
            out_path[i,0] = bb_mean[i,0]
            out_path[i,self.Nq-1] = bb_mean[i,self.Nq-1]
            out_d_path[i,0] = dpath_FFT_out[i,0]
            out_d_path[i,self.Nq-1] = dpath_FFT_out[i,self.Nq-1]

            for k in range(1,self.Nq-1):
                out_path[i,k] = path_FFT_out[i,k-1] + bb_mean[i,k]
                out_d_path[i,k] = dpath_FFT_out[i,k]
        #self.profile_timers[3] += time.time() - start_time

    cpdef FFT_compute_path_and_dpath(self, np.ndarray state):
        out_path = np.zeros((self.dim, self.Nq))
        out_d_path = np.zeros((self.dim, self.Nq))
        self._FFT_compute_path_and_dpath(state, out_path, out_d_path)
        return out_path, out_d_path
    
    
    cdef int _FFT_compute_path(self, DTYPE_t[:,:] state, DTYPE_t[:,:] out_path) except -1:
        cdef:
            DTYPE_t[:,:] path_FFT_in = self.path_FFT_in
            DTYPE_t[:,:] path_FFT_out = self.path_FFT_out
            DTYPE_t[:,:] bb_mean = self.bb_mean
            DTYPE_t[:] path_coeffs_prefactors = self.path_coeffs_prefactors

            int i, k

        for i in range(self.dim):
            for k in range(self.Nm):
                path_FFT_in[i,k] = state[i,k] * path_coeffs_prefactors[k]

        self.dst_transform()

        for i in range(self.dim):
            out_path[i,0] = bb_mean[i,0]
            out_path[i,self.Nq-1] = bb_mean[i,self.Nq-1]

            for k in range(1,self.Nq-1):
                out_path[i,k] = path_FFT_out[i,k-1] + bb_mean[i,k]

    cpdef FFT_compute_path(self, np.ndarray state):
        out_path = np.zeros((self.dim, self.Nq))
        self._FFT_compute_path(state, out_path)
        return out_path


    def __init__(self, params):
        super().__init__(params)
        
        #self.profile_timers = np.zeros(4)

        # Set up the FFT path sampling

        if self.Nq < self.Nm + 2: # Make sure Nq is at least 2 bigger than Nm
            self.Nq += 2
            self.Nq_factor = self.Nq / float(self.Nm)

            if self.quad_scheme != quad.uniform and self.quad_scheme != quad.discrete:
                raise Exception('Incompatible quad scheme.')

            # Redefine the arrays previously set in PathMCMC
            self.ts, self.qts, self.quad_weights = self.quad_scheme(self.Nq, self.Tf)
            self.force_arr = np.zeros((self.dim, self.Nq), dtype=DTYPE)
            self.div_force_arr = np.zeros(self.Nq, dtype=DTYPE)
            self.lagrangian_arr = np.zeros(self.Nq, dtype=DTYPE)
            self.noise_vector_arr = np.zeros(self.Nm, dtype=DTYPE)

        #pyfftw.forget_wisdom() # Some bug in pyfftw forces me to use this.

        self.path_FFT_in = pyfftw.empty_aligned((self.dim, self.Nq-2), dtype='float64')
        self.path_FFT_out = pyfftw.empty_aligned((self.dim, self.Nq-2), dtype='float64')
        self.dpath_FFT_in = pyfftw.empty_aligned((self.dim, self.Nq), dtype='float64')
        self.dpath_FFT_out = pyfftw.empty_aligned((self.dim, self.Nq), dtype='float64')
        
        self.path_FFT_in[:] = 0
        self.path_FFT_out[:] = 0
        self.dpath_FFT_in[:] = 0
        self.dpath_FFT_out[:] = 0

        if 'FFTW_planner_flag' in params:
            FFTW_planner_flag = params['FFTW_planner_flag']
        else:
            FFTW_planner_flag = 'FFTW_MEASURE'
        
        self.dst_transform = pyfftw.FFTW(self.path_FFT_in, self.path_FFT_out, axes=[1], direction='FFTW_RODFT00', flags=['FFTW_DESTROY_INPUT', FFTW_planner_flag])
        self.dct_transform = pyfftw.FFTW(self.dpath_FFT_in, self.dpath_FFT_out, axes=[1], direction='FFTW_REDFT00', flags=['FFTW_DESTROY_INPUT', FFTW_planner_flag])

        self.path_coeffs_prefactors = self.Tf / (np.pi * np.arange(1,self.Nm+1))
        self.path_coeffs_prefactors *= np.sqrt(2 / self.Tf) * 0.5 # factor of 0.5 from the factor of 2 in the DCT-I and DST-I
        self.dpath_coeffs_prefactor = np.sqrt(2 / self.Tf) * 0.5

        self.bb_mean = np.zeros( (self.dim, self.Nq) )
        for i in range(self.dim):
            self.bb_mean[i, :] += self.x0[i] + (self.ts/self.Tf)*(self.x1[i] - self.x0[i])
        self.dbb_mean_const = (1/self.Tf)*(self.x1 - self.x0)

        # Set up pCN

        if np.size(params['kappa']) == 1:
            self.kappas = np.zeros(self.Nm, dtype=DTYPE)
            self.kappas[:] = params['kappa']
        else:
            self.kappas = np.copy(params['kappa']).astype(DTYPE)

        if 'use_pCN' in params and not params['use_pCN']:
            self.skappas = np.full(len(self.kappas), 1.0)
            self.noise_prefactor = np.sqrt(2/(self.beta*self.gamma))
            self.pkappas = self.noise_prefactor * self.kappas
            self.use_pCN = False
        else:
            self.skappas = np.sqrt(1 - self.kappas**2)
            self.noise_prefactor = np.sqrt(2/(self.beta*self.gamma))
            self.pkappas = self.noise_prefactor * self.kappas
            self.use_pCN = True

        #self.bb_mean, self.dbb_mean, self.phis, self.dphis = utils.get_fast_eval_path_basis(self.x0, self.x1, self.ts, self.Tf, self.Nm)

        self.current_state = self.state0
        #fep_basis = self.bb_mean, self.dbb_mean, self.phis, self.dphis
        #self.current_path, self.current_d_path = utils.fast_eval_path(self.current_state, fep_basis)
        self.current_path, self.current_d_path = self.FFT_compute_path_and_dpath(self.current_state)
        self.current_accept = True
        self.current_action = utils.compute_action(self.system, self.current_state, self.Nq_factor, use_relative=self.use_pCN, use_discretised=self.use_discretised_action, use_OM=self.use_OM, quad_scheme=self.quad_scheme)

        self.p_flip = params['p_flip']

        # Teleport

        if 'use_teleport' in params and params['use_teleport']:
            self.p_teleport = params['p_teleport']
            self.teleporter_probs = np.array(params['teleporter_probs'], dtype=DTYPE) / np.sum(params['teleporter_probs'])
            self.teleporter_log_probs = np.log(self.teleporter_probs)
            self.teleporters_num = len(params['teleporters'])
            self.teleporters = np.array(params['teleporters'])

            for i in range(self.teleporters_num):
                (<Teleporter>self.teleporters[i])._set_rng(&self.rng_gen)

            self.teleporters_CDF = np.copy(self.teleporter_probs)
            for i in range(1, len(self.teleporters_CDF)):
                self.teleporters_CDF[i] += self.teleporters_CDF[i-1]
            
            self._G_acc_numer_terms = np.zeros(self.teleporters_num)
            self._G_acc_denom_terms = np.zeros(self.teleporters_num)

            self.teleporter_attempts = np.zeros(self.teleporters_num)
            self.teleporter_count = np.zeros(self.teleporters_num)

            if 'teleporter_start_indices' in params:
                self.teleporter_start_indices = np.copy(params['teleporter_start_indices']).astype(DTYPE)
            else:
                self.teleporter_start_indices = np.zeros(self.teleporters_num, dtype=int)
            self.teleporter_end_indices = np.zeros(self.teleporters_num, dtype=int)
            for i in range(len(self.teleporters)):
                self.teleporter_end_indices[i] = self.teleporter_start_indices[i] + self.teleporters[i].Nm

        else:
            self.p_teleport = -1

        # Window

        if 'use_windows' in params and params['use_windows']:
            self.window_mode = True

            self.rskappas = np.sqrt(1 - self.kappas**2) - 1
            
            self.window_probs = np.array(params['window_probs'], dtype=DTYPE) / np.sum(params['window_probs'])
            self.window_indices = np.copy(params['window_indices']).astype(int)

            if np.max(self.window_indices) >= self.Nm:
                raise Exception('Window index is larger than Nm.')

            if 'window_sizes' in params:
                self.window_sizes = np.copy(params['window_sizes']).astype(int)
            else:
                bnds = list(self.window_indices) + [self.Nm]
                self.window_sizes = np.array([bnds[i] - bnds[i-1] for i in range(1, len(bnds))], dtype=int)

            self.windows = len(self.window_probs)
            
            self.window_dstate = np.zeros((self.dim, np.max(self.window_sizes)), dtype=DTYPE)
            self.window_dstate = np.copy(self.window_dstate, order='F') # Change matrix order to accomodate window slicing
            self.window_path_buffer = np.zeros((self.dim, self.Nq), dtype=DTYPE)
            
            self.window_CDF = np.copy(self.window_probs)
            for i in range(1, len(self.window_CDF)):
                self.window_CDF[i] += self.window_CDF[i-1]
            
            self.window_acceptance_counts = np.zeros(self.windows, dtype=int)
            self.window_attempt_counts = np.zeros(self.windows, dtype=int)

            if 'N_window_path_recalibration' in params:
                self.N_window_path_recalibration = params['N_window_path_recalibration']
            else:
                self.N_window_path_recalibration = 100
        else:
            self.window_mode = False

    cdef int pre_simulate_batch(self, int N_steps) except -1:
        pass

    cdef int _run_step(self, int step_i) except -1:
        cdef:
            DTYPE_t Tf = self.Tf
            DTYPE_t gamma = self.gamma
            DTYPE_t beta = self.beta
            DTYPE_t[:] skappas = self.skappas
            DTYPE_t[:] pkappas = self.pkappas
            DTYPE_t noise_prefactor = self.noise_prefactor

            DTYPE_t[:,:] bb_mean = self.bb_mean
            DTYPE_t[:,:] dbb_mean = self.dbb_mean
            #DTYPE_t[:,:] phis = self.phis
            #DTYPE_t[:,:] dphis = self.dphis
            DTYPE_t[:] ts = self.ts
            DTYPE_t[:] quad_weights = self.quad_weights

            System system = self.system

            DTYPE_t[:,:,:] states = self.states_arr
            DTYPE_t[:] actions = self.actions_arr
            DTYPE_t[:,:,:] paths = self.paths_arr
            DTYPE_t[:,:,:] d_paths = self.d_paths_arr
            DTYPE_t[:,:] force = self.force_arr
            DTYPE_t[:] div_force = self.div_force_arr
            DTYPE_t[:] lagrangian = self.lagrangian_arr
            DTYPE_t[:] noise_vector = self.noise_vector_arr

            DTYPE_t[:,:] full_state = self.full_state_arr
            DTYPE_t[:,:] prev_full_state = self.prev_full_state_arr

            DTYPE_t[:] rskappas = self.rskappas
            DTYPE_t[:,:] window_dstate = self.window_dstate
            long[:] window_acceptance_counts = self.window_acceptance_counts
            long[:] window_attempt_counts = self.window_attempt_counts
            DTYPE_t[:] window_CDF = self.window_CDF
            int window_i, window_size, window_start, window_end

            DTYPE_t[:] teleporters_CDF = self.teleporters_CDF
            DTYPE_t[:] teleporter_log_probs = self.teleporter_log_probs
            DTYPE_t G_acc_numer_largest_term, G_acc_denom_largest_term
            DTYPE_t G_acc_numer_exp_sum, G_acc_denom_exp_sum
            DTYPE_t log_G_acc_numer, log_G_acc_denom
            DTYPE_t[:] G_acc_numer_terms = self._G_acc_numer_terms
            DTYPE_t[:] G_acc_denom_terms = self._G_acc_denom_terms
            bint attempted_teleportation = False
            DTYPE_t previous_action, proposal_action

            bint attempted_flip = False

            DTYPE_t acceptance_prob, u
            int j, k
        
        ### Generate teleport proposal
        if self.p_teleport != -1 and self.unif_dist(self.rng_gen) < self.p_teleport:
            attempted_teleportation = True

            u = self.unif_dist(self.rng_gen)
            for k in range(self.teleporters_num):
                if u < teleporters_CDF[k]:
                    teleporter_i = k
                    break

            teleporter_start_index = self.teleporter_start_indices[teleporter_i]
            teleporter_end_index = self.teleporter_end_indices[teleporter_i]

            for j in range(self.dim):
                for k in range(teleporter_end_index, self.Nm):
                    full_state[j,k] = prev_full_state[j,k]

            (<Teleporter>self.teleporters[teleporter_i])._generate_state(self.full_state_arr[:,teleporter_start_index:teleporter_end_index])

            self.teleportation_attempts += 1
            self.teleporter_attempts[teleporter_i] += 1

            ### Compute acceptance probability

            # Compute the full action of the proposal
            #cutils._fast_eval_path(self.bb_mean, self.dbb_mean, self.phis, self.dphis, self.full_state_arr, self.paths_arr[step_i,:,:], self.d_paths_arr[step_i,:,:])
            if self.use_discretised_action:
                self._FFT_compute_path(full_state, paths[step_i,:,:])
            else:
                self._FFT_compute_path_and_dpath(full_state, paths[step_i,:,:], d_paths[step_i,:,:])
                
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
            proposal_action = cutils._eval_action(quad_weights, Tf, lagrangian)

            # Compute the full action of the previous state
            if self.use_OM:
                if self.use_discretised_action:
                    lagrangians._compute_OM_discretised_lagrangian(system, ts, paths[step_i-1,:,:], force, div_force, lagrangian)
                else:
                    lagrangians._compute_OM_lagrangian(system, ts, paths[step_i-1,:,:], d_paths[step_i-1,:,:], force, div_force, lagrangian)
            else:
                if self.use_discretised_action:
                    lagrangians._compute_FW_discretised_lagrangian(system, ts, paths[step_i-1,:,:], force, div_force, lagrangian)
                else:
                    lagrangians._compute_FW_lagrangian(system, ts, paths[step_i-1,:,:], d_paths[step_i-1,:,:], force, div_force, lagrangian)
            previous_action = cutils._eval_action(quad_weights, Tf, lagrangian)
            #previous_action = actions[step_i-1] + cutils._eval_gaussian_action(system.x0, system.x1, system.Tf, system.beta, system.gamma, prev_full_state)

            # Compute the action of the teleporter
            for i in range(self.teleporters_num):
                G_acc_numer_terms[i] = teleporter_log_probs[i] - (<Teleporter>self.teleporters[i])._compute_action(self.prev_full_state_arr[:,teleporter_start_index:teleporter_end_index])
                G_acc_denom_terms[i] = teleporter_log_probs[i] - (<Teleporter>self.teleporters[i])._compute_action(self.full_state_arr[:,teleporter_start_index:teleporter_end_index])
                
                if i == 0 or G_acc_numer_terms[i] > G_acc_numer_largest_term:
                    G_acc_numer_largest_term = G_acc_numer_terms[i]
                    
                if i == 0 or G_acc_denom_terms[i] > G_acc_denom_largest_term:
                    G_acc_denom_largest_term = G_acc_denom_terms[i]

            G_acc_numer_exp_sum = 0
            G_acc_denom_exp_sum = 0
            for i in range(self.teleporters_num):
                G_acc_numer_exp_sum += exp(G_acc_numer_terms[i]-G_acc_numer_largest_term)
                G_acc_denom_exp_sum += exp(G_acc_denom_terms[i]-G_acc_denom_largest_term)

            log_G_acc_numer = G_acc_numer_largest_term + log(G_acc_numer_exp_sum)
            log_G_acc_denom = G_acc_denom_largest_term + log(G_acc_denom_exp_sum)

            acceptance_prob = min(1.0, exp(previous_action - proposal_action + log_G_acc_numer - log_G_acc_denom))

        ### Generate regular proposal
        else:
            if self.window_mode:
                ### Generate window

                u = self.unif_dist(self.rng_gen)
                for k in range(self.windows):
                    if u < window_CDF[k]:
                        window_i = k
                        break
                
                window_start = self.window_indices[window_i]
                window_size = self.window_sizes[window_i]
                window_end = window_start + window_size
                window_attempt_counts[window_i] += 1

                for j in range(self.dim):
                    for k in range(window_start, window_end):
                        window_dstate[j, k-window_start] = rskappas[k]*prev_full_state[j,k] + pkappas[k]*self.normal_dist(self.rng_gen)

                for j in range(self.dim):
                    for k in range(self.Nm):
                        full_state[j,k] = prev_full_state[j,k]

                for j in range(self.dim):
                    for k in range(window_start, window_end):
                        full_state[j,k] += window_dstate[j, k-window_start]

                #cutils._fast_eval_path(self.bb_mean, self.dbb_mean, self.phis, self.dphis, self.full_state_arr, self.paths_arr[step_i,:,:], self.d_paths_arr[step_i,:,:])

                if self.use_discretised_action:
                    self._FFT_compute_path(full_state, paths[step_i,:,:])
                else:
                    self._FFT_compute_path_and_dpath(full_state, paths[step_i,:,:], d_paths[step_i,:,:])

                #cutils._fast_update_path(self.phis[window_start:window_end,:], self.dphis[window_start:window_end,:], self.window_dstate[:,:window_size], self.paths_arr[step_i,:,:], self.d_paths_arr[step_i,:,:], self.window_path_buffer)
                #cutils._fast_update_path(self.phis[window_start:window_end,:], self.dphis[window_start:window_end,:], self.window_dstate[:,:window_size], path, dpath, self.window_path_buffer)
                #cutils._fast_eval_path(self.bb_mean, self.dbb_mean, self.phis, self.dphis, self.full_state_arr, self.paths_arr[step_i,:,:], self.d_paths_arr[step_i,:,:])

                #print(np.max(np.abs(path - self.paths_arr[step_i,:,:])), np.max(np.abs(dpath - self.d_paths_arr[step_i,:,:])))
            else:
                for j in range(self.dim):
                    for k in range(self.Nm):
                        full_state[j,k] = skappas[k]*prev_full_state[j,k] + pkappas[k]*self.normal_dist(self.rng_gen)

                if self.unif_dist(self.rng_gen) < self.p_flip:
                    full_state[1,0] = -full_state[1,0]
                    attempted_flip = True
                    self.flip_attempts += 1

                #cutils._fast_eval_path(self.bb_mean, self.dbb_mean, self.phis, self.dphis, self.full_state_arr, self.paths_arr[step_i,:,:], self.d_paths_arr[step_i,:,:])

                if self.use_discretised_action:
                    self._FFT_compute_path(full_state, paths[step_i,:,:])
                else:
                    self._FFT_compute_path_and_dpath(full_state, paths[step_i,:,:], d_paths[step_i,:,:])
                
            ### Compute acceptance probability

            if self.use_pCN:
                if self.use_OM:
                    if self.use_discretised_action:
                        lagrangians._compute_relative_OM_discretised_lagrangian(system, ts, paths[step_i,:,:], force, div_force, lagrangian)
                    else:
                        lagrangians._compute_relative_OM_lagrangian(system, ts, paths[step_i,:,:], d_paths[step_i,:,:], force, div_force, lagrangian)
                else:
                    if self.use_discretised_action:
                        lagrangians._compute_relative_FW_discretised_lagrangian(system, ts, paths[step_i,:,:], force, div_force, lagrangian)
                    else:
                        lagrangians._compute_relative_FW_lagrangian(system, ts, paths[step_i,:,:], d_paths[step_i,:,:], force, div_force, lagrangian)
            else:
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
        
        if self.unif_dist(self.rng_gen) < acceptance_prob:
            if attempted_flip:
                self.flip_count += 1

            if attempted_teleportation:
                self.teleportation_count += 1
                self.teleporter_count[teleporter_i] += 1

                # Compute the relative action of the teleportation destination
                if self.use_pCN:
                    if self.use_OM:
                        if self.use_discretised_action:
                            lagrangians._compute_relative_OM_discretised_lagrangian(system, ts, paths[step_i,:,:], force, div_force, lagrangian)
                        else:
                            lagrangians._compute_relative_OM_lagrangian(system, ts, paths[step_i,:,:], d_paths[step_i,:,:], force, div_force, lagrangian)
                    else:
                        if self.use_discretised_action:
                            lagrangians._compute_relative_FW_discretised_lagrangian(system, ts, paths[step_i,:,:], force, div_force, lagrangian)
                        else:
                            lagrangians._compute_relative_FW_lagrangian(system, ts, paths[step_i,:,:], d_paths[step_i,:,:], force, div_force, lagrangian)
                else:
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
                #actions[step_i] = proposal_action - cutils._eval_gaussian_action(system.x0, system.x1, system.Tf, system.beta, system.gamma, full_state)

            self.accepts_arr[step_i] = True

            if self.window_mode and not attempted_teleportation:
                window_acceptance_counts[window_i] += 1

                for j in range(self.dim):
                    for k in range(self.Nm_save):
                        states[step_i,j,k] = states[step_i-1,j,k]

                for j in range(self.dim):
                    for k in range(window_start, window_end):
                        prev_full_state[j,k] += window_dstate[j,k-window_start]

                #for j in range(self.dim):
                #    for k in range(self.Nm):
                #        prev_full_state[j,k] = full_state[j,k]

                for j in range(self.dim):
                    for k in range(window_start, int(min(window_end, self.Nm))):
                        states[step_i,j,k] += window_dstate[j,k-window_start]
            else:
                for j in range(self.dim):
                    for k in range(self.Nm_save):
                        states[step_i,j,k] = full_state[j,k]

                for j in range(self.dim):
                    for k in range(self.Nm):
                        prev_full_state[j,k] = full_state[j,k]
        else:
            self.accepts_arr[step_i] = False
            actions[step_i] = actions[step_i-1]

            for j in range(self.dim):
                for k in range(self.Nm_save):
                    states[step_i,j,k] = states[step_i-1,j,k]

            for j in range(self.dim):
                for k in range(self.Nq):
                    paths[step_i,j,k] = paths[step_i-1,j,k]
                    d_paths[step_i,j,k] = d_paths[step_i-1,j,k]

