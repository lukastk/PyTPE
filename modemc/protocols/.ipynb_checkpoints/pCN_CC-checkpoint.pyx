# cython: unraisable_tracebacks=True
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from modemc.defs import DTYPE
import numpy as np
from modemc import utils, quad, variations
from cython.operator cimport dereference
import os, time

cdef class pCN_CC:
    def __init__(self, params):
        super().__init__(params)

        if np.size(params['kappa']) == 1:
            self.kappas = np.zeros(self.Nm, dtype=DTYPE)
            self.kappas[:] = params['kappa']
        else:
            self.kappas = np.copy(params['kappa']).astype(DTYPE)

        self.skappas = np.sqrt(1 - self.kappas**2)
        self.noise_prefactor = np.sqrt(2/(self.beta*self.gamma))
        self.pkappas = self.noise_prefactor * self.kappas

        self.bb_mean, self.dbb_mean, self.phis, self.dphis = utils.get_fast_eval_path_basis(self.x0, self.x1, self.ts, self.Tf, self.Nm)

        self.current_state = self.state0
        fep_basis = self.bb_mean, self.dbb_mean, self.phis, self.dphis
        self.current_path, self.current_d_path = utils.fast_eval_path(self.current_state, fep_basis)
        self.current_accept = True
        self.current_action = utils.compute_action(self.system, self.current_state, self.Nq_factor, use_relative=True, use_discretised=self.use_discretised_action, use_OM=self.use_OM, quad_scheme=self.quad_scheme)

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

        # Coarse-grained action

        self.gaussian_band_size = params['gaussian_band_size']
        self.Nm_gaussian = params['Nm_gaussian']
        _, _, self.gaussian_phis, self.gaussian_dphis = utils.get_fast_eval_path_basis(self.x0, self.x1, self.ts, self.Tf, self.Nm_gaussian)
        self.gaussian_phis = self.gaussian_phis[self.Nm:,:]
        self.gaussian_dphis = self.gaussian_dphis[self.Nm:,:]

        state0_full = np.zeros((self.dim, self.Nm_gaussian))
        state0_full[:,:self.Nm] = self.current_state
        gradS = utils.compute_gradS(self.system, state0_full, Nq_factor=12, Nq=None, use_OM=True, quad_scheme=None)
        J = gradS[0,self.Nm:]

        Q, Qf = variations.compute_banded_Q_matrix(self.system, self.full_state_arr, self.gaussian_band_size,
                Nm=self.Nm_gaussian, Nq_factor=12, quad_scheme=None, return_partial_Qs=False, use_OM=True)

        #Qf_orig = np.copy(Qf)
        #Qf = np.copy(Qfxdxd)
        #Qf[:Q_block_N,:Q_block_N] = Qf_orig[:Q_block_N,:Q_block_N] 

        Qf = Qf[self.Nm:,self.Nm:]
        _, log_det_Q = np.linalg.slogdet(Qf)
        iQf = np.linalg.inv(Qf)
        iQJ = iQf.dot(J.flatten())
        norm_iQJ = iQJ.flatten().dot(iQJ.flatten())

        S_CC = 0.5*(log_det_Q - norm_iQJ)
        self.current_action +=  S_CC

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

            DTYPE_t acceptance_prob, u
            int j, k

        ### Generate teleport proposal
        if self.p_teleport != -1 and self.unif_dist(self.rng_gen) < self.p_teleport:

            self.teleportation_attempts += 1
            attempted_teleportation = True

            u = self.unif_dist(self.rng_gen)
            for k in range(self.teleporters_num):
                if u < teleporters_CDF[k]:
                    teleporter_i = k
                    break

            (<Teleporter>self.teleporters[teleporter_i])._generate_state(self.full_state_arr)

            ### Compute acceptance probability

            # Compute the full action of the proposal
            cutils._fast_eval_path(self.bb_mean, self.dbb_mean, self.phis, self.dphis, self.full_state_arr, self.paths_arr[step_i,:,:], self.d_paths_arr[step_i,:,:])
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
                G_acc_numer_terms[i] = teleporter_log_probs[i] - (<Teleporter>self.teleporters[i])._compute_action(self.prev_full_state_arr)
                G_acc_denom_terms[i] = teleporter_log_probs[i] - (<Teleporter>self.teleporters[i])._compute_action(self.full_state_arr)
                
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

                if self.steps % self.N_window_path_recalibration == 0:
                    for j in range(self.dim):
                        for k in range(self.Nm):
                            full_state[j,k] = prev_full_state[j,k]

                    for j in range(self.dim):
                        for k in range(window_start, window_end):
                            full_state[j,k] += window_dstate[j, k-window_start]

                    cutils._fast_eval_path(self.bb_mean, self.dbb_mean, self.phis, self.dphis, self.full_state_arr, self.paths_arr[step_i,:,:], self.d_paths_arr[step_i,:,:])
                else:
                    for j in range(self.dim):
                        for k in range(self.Nq):
                            paths[step_i,j,k] = paths[step_i-1,j,k]
                            d_paths[step_i,j,k] = d_paths[step_i-1,j,k]

                    cutils._fast_update_path(self.phis[window_start:window_end,:], self.dphis[window_start:window_end,:], self.window_dstate[:,:window_size], self.paths_arr[step_i,:,:], self.d_paths_arr[step_i,:,:], self.window_path_buffer)
                
                #cutils._fast_update_path(self.phis[window_start:window_end,:], self.dphis[window_start:window_end,:], self.window_dstate[:,:window_size], self.paths_arr[step_i,:,:], self.d_paths_arr[step_i,:,:], self.window_path_buffer)
                #cutils._fast_update_path(self.phis[window_start:window_end,:], self.dphis[window_start:window_end,:], self.window_dstate[:,:window_size], path, dpath, self.window_path_buffer)
                #cutils._fast_eval_path(self.bb_mean, self.dbb_mean, self.phis, self.dphis, self.full_state_arr, self.paths_arr[step_i,:,:], self.d_paths_arr[step_i,:,:])

                #print(np.max(np.abs(path - self.paths_arr[step_i,:,:])), np.max(np.abs(dpath - self.d_paths_arr[step_i,:,:])))
            else:
                for j in range(self.dim):
                    for k in range(self.Nm):
                        full_state[j,k] = skappas[k]*prev_full_state[j,k] + pkappas[k]*self.normal_dist(self.rng_gen)

                cutils._fast_eval_path(self.bb_mean, self.dbb_mean, self.phis, self.dphis, self.full_state_arr, self.paths_arr[step_i,:,:], self.d_paths_arr[step_i,:,:])

            ### Compute acceptance probability

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

            actions[step_i] = cutils._eval_action(quad_weights, Tf, lagrangian)

            # Compute coarse-gained correction
            #L_x, L_xd = utils.compute_gradL(self.system, self.ts, self.full_state_arr, use_OM=True)
            #J_x = np.einsum('in,kn,n->ik', L_x, self.gaussian_phis, self.quad_weights)
            #J_xd = np.einsum('in,kn,n->ik', L_xd, self.gaussian_dphis, self.quad_weights)
            #J = J_x + J_xd
            #Q, Qf = variations.compute_banded_Q_matrix(self.system, self.full_state_arr, self.gaussian_band_size,
            #        Nm=self.Nm_gaussian, Nq_factor=12, quad_scheme=None, return_partial_Qs=False, use_OM=True)
            #Q = Q[:,:,self.Nm:,self.Nm:]
            #Qf = variations.flatten_Q(Q)
            #iQf = np.linalg.inv(Qf)
            #iQ = variations.unflatten_Qf(iQf, self.dim)
            #_, log_det_Q = np.linalg.slogdet(Qf)

            #iQJ = np.einsum('ijkl,jl->ik', iQ, J)
            #norm_iQJ = iQJ.flatten().dot(iQJ.flatten())
            #S_CC = 0.5*(log_det_Q + norm_iQJ)

            #S_CC = 0.5*log_det_Q

            #actions[step_i] += S_CC

            state0_full = np.zeros((self.dim, self.Nm_gaussian))
            state0_full[:,:self.Nm] = self.full_state_arr
            gradS = utils.compute_gradS(self.system, state0_full, Nq_factor=12, Nq=None, use_OM=True, quad_scheme=None)
            J = gradS[0,self.Nm:]

            Q, Qf = variations.compute_banded_Q_matrix(self.system, self.full_state_arr, self.gaussian_band_size,
                    Nm=self.Nm_gaussian, Nq_factor=12, quad_scheme=None, return_partial_Qs=False, use_OM=True)

            #Qf_orig = np.copy(Qf)
            #Qf = np.copy(Qfxdxd)
            #Qf[:Q_block_N,:Q_block_N] = Qf_orig[:Q_block_N,:Q_block_N] 

            Qf = Qf[self.Nm:,self.Nm:]
            _, log_det_Q = np.linalg.slogdet(Qf)
            iQf = np.linalg.inv(Qf)
            iQJ = iQf.dot(J.flatten())
            norm_iQJ = iQJ.flatten().dot(iQJ.flatten())

            S_CC = 0.5*(log_det_Q - norm_iQJ)
            actions[step_i] += S_CC

            acceptance_prob = min(1.0, exp(actions[step_i-1] - actions[step_i]))
        
        if self.unif_dist(self.rng_gen) < acceptance_prob:
            if attempted_teleportation:
                self.teleportation_count += 1

                # Compute the relative action of the teleportation destination
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

            accepts[step_i] = True

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
            accepts[step_i] = False
            actions[step_i] = actions[step_i-1]

            for j in range(self.dim):
                for k in range(self.Nm_save):
                    states[step_i,j,k] = states[step_i-1,j,k]

            for j in range(self.dim):
                for k in range(self.Nq):
                    paths[step_i,j,k] = paths[step_i-1,j,k]
                    d_paths[step_i,j,k] = d_paths[step_i-1,j,k]



cdef class Teleporter:

    def __init__(self, dim, Nm):
        self.Nm = Nm
        self.dim = dim

        # Create internal RNGs so that the python functions are useable from the start
        cdef long max_long = 9223372036854775807
        rng_seed = (abs(os.getpid()) + long(time.time()*1000)) % max_long
        self.internal_rng_gen = mt19937(rng_seed)
        self.rng_gen = &self.internal_rng_gen

    cdef void _set_rng(self, mt19937* rng_gen):
        self.rng_gen = rng_gen

    cdef DTYPE_t _compute_action(self, np.ndarray state):
        raise Exception('Absract method')

    cdef int _generate_state(self, np.ndarray out_state_arr) except -1:
        raise Exception('Absract method')

    def compute_action(self, state):
        return self._compute_action(state)

    def generate_state(self):
        out_state_arr = np.zeros((self.dim, self.Nm), dtype=DTYPE)
        self._generate_state(out_state_arr)

        return out_state_arr

cdef class GaussianTeleporter(Teleporter):

    def __init__(self, mean_state, precision, is_covariance=False, match_precision_Nm=False):
        dim, inst_Nm = mean_state.shape[0], mean_state.shape[1]
        if match_precision_Nm and dim*inst_Nm != precision.shape[0]:
            Nm = int(precision.shape[0] / dim)
            if Nm != precision.shape[0] / dim:
                raise Exception('mean_state %s and precision %s dimensions are incompatible.' % (mean_state.shape, precision.shape))
            new_mean_state = np.zeros((dim, Nm))
            new_mean_state[:,:inst_Nm] = mean_state
            mean_state = new_mean_state

        dim, Nm = mean_state.shape[0], mean_state.shape[1]
        if dim*Nm != precision.shape[0]:
            raise Exception('Precision matrix and mean do not match in size.')

        super().__init__(mean_state.shape[0], mean_state.shape[1])

        self.normal_dist = normal_distribution[DTYPE_t](0.0,1.0)

        if is_covariance:
            self.precision = np.linalg.inv(precision)
        else:
            self.precision = np.copy(precision).astype(DTYPE)

        self.mean_state = np.copy(mean_state).astype(DTYPE)

        self.precision_det_sign, self.log_precision_det = np.linalg.slogdet(self.precision)
        self.Z_sign, self.log_Z = self.precision_det_sign, 0.5*( self.precision.shape[0]*np.log(2*np.pi) - self.log_precision_det )
        
        precision_eigvals, precision_eigvectors = np.linalg.eigh(self.precision)
        
        if not np.all(precision_eigvals > 0):
            print('WARNING: Precision matrix is not positive definite. Projecting to closest positive semi-definite matrix.')

            D, V = precision_eigvals, precision_eigvectors
            pos_D = np.diag(D[D>0])
            D[D<0] = 0 # Remove negative eigenvalues and set them to 0
            D = np.diag(D)

            self.precision = V.dot(D).dot(V.T)
            pos_prec_det_sign, pos_prec_det = np.linalg.slogdet(pos_D)
            self.Z_sign, self.log_Z = pos_prec_det_sign, 0.5*( pos_D.shape[0]*np.log(2*np.pi) - pos_prec_det )

            iD = np.linalg.pinv(D) # Use pseudo inverse so that 0 entries just remain 0

            self.inv_chol_L = V.dot(np.sqrt(iD))
        else:
            self.chol_L = np.linalg.cholesky(self.precision).T # Transpose so that precision = L.T * L
            self.inv_chol_L = np.linalg.inv(self.chol_L)

        self.dstate_flat = np.zeros(self.Nm*self.dim, dtype=DTYPE)
        self._dstate_buffer = np.zeros(self.Nm*self.dim, dtype=DTYPE)
        self.gaussian_vector = np.zeros(self.Nm*self.dim, dtype=DTYPE)

    cdef DTYPE_t _compute_action(self, np.ndarray state_arr):
        cdef int i, k
        cdef DTYPE_t[:,:] state = state_arr
        cdef DTYPE_t[:] dstate_flat = self.dstate_flat
        cdef DTYPE_t[:,:] mean_state = self.mean_state
        cdef int Nm = self.Nm

        for i in range(self.dim):
            for k in range(Nm):
                dstate_flat[k + i*Nm] = state[i,k] - mean_state[i,k]
        
        np.matmul(self.precision, self.dstate_flat, self._dstate_buffer)

        return 0.5*self.dstate_flat.dot(self._dstate_buffer) + self.log_Z

    cdef int _generate_state(self, np.ndarray out_state_arr) except -1:
        cdef int i, k
        cdef DTYPE_t[:,:] out_state = out_state_arr
        cdef DTYPE_t[:] dstate_flat = self.dstate_flat
        cdef DTYPE_t[:] gaussian_vector = self.gaussian_vector
        cdef DTYPE_t[:,:] mean_state = self.mean_state
        cdef int Nm = self.Nm

        for i in range(gaussian_vector.shape[0]):
            gaussian_vector[i] = self.normal_dist(dereference(self.rng_gen))

        np.matmul(self.inv_chol_L, self.gaussian_vector, self.dstate_flat)

        for i in range(self.dim):
            for k in range(Nm):
                out_state[i,k] = dstate_flat[k + i*Nm] + mean_state[i,k]

    