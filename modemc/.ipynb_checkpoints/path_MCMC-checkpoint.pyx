# cython: profile=True
# cython: unraisable_tracebacks=True
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from modemc.defs import DTYPE
import numpy as np
from modemc import utils
from modemc import quad
import os, time
cimport cython

cdef class PathMCMC:
    def __init__(self, params):
        if not 'Nm_save' in params:
            params['Nm_save'] = params['Nm']

        self.Nm_save = params['Nm_save']
        self.Nm = params['Nm']

        self.system = params['system']
        self.gamma = self.system.gamma
        self.beta = self.system.beta
        self.Tf = self.system.Tf
        self.dim = self.system.dim
        self.x0 = self.system.x0
        self.x1 = self.system.x1

        self.use_OM = params['use_OM']
        if 'use_discretised_action' in params:
            self.use_discretised_action = params['use_discretised_action']
        else:
            self.use_discretised_action = False

        if 'Nq_factor' in params:
            self.Nq_factor = params['Nq_factor']
            self.Nq = int(np.round(self.Nm * self.Nq_factor))
        else:
            self.Nq = int(params['Nq'])
            self.Nq_factor = self.Nm/float(self.Nq)
        if 'Nq_factor' in params and 'Nq' in params:
            raise Exception('Cannot define both Nq and Nq_factor.')

        if not 'quad_scheme' in params:
            if self.use_discretised_action:
                self.quad_scheme = quad.discrete
            else:
                self.quad_scheme = quad.clenshaw_curtis
        else:
            if self.use_discretised_action and params['quad_scheme'] != quad.discrete:
                raise Exception('Invalid quad scheme for discretised action.')
            else:
                self.quad_scheme = params['quad_scheme']
        self.ts, self.qts, self.quad_weights = self.quad_scheme(self.Nq, self.Tf)
      
        self.state0 = np.copy(params['state0']).astype(DTYPE)

        self.full_state_arr = np.zeros((self.dim, self.Nm), dtype=DTYPE)
        self.prev_full_state_arr = np.zeros((self.dim, self.Nm), dtype=DTYPE)

        self.steps = 0

        cdef long max_long = 9223372036854775807
        if not 'rng_seed' in  params:
            self.rng_seed = (abs(os.getpid()) + long(time.time()*1000)) % max_long
        else:
            self.rng_seed = params['rng_seed'] % max_long

        self.rng_gen = mt19937(self.rng_seed)
        self.normal_dist = normal_distribution[DTYPE_t](0.0,1.0)
        self.unif_dist = uniform_real_distribution[DTYPE_t](0.0,1.0)

        self.force_arr = np.zeros((self.dim, self.Nq), dtype=DTYPE)
        self.div_force_arr = np.zeros(self.Nq, dtype=DTYPE)
        self.lagrangian_arr = np.zeros(self.Nq, dtype=DTYPE)
        self.noise_vector_arr = np.zeros(self.Nm, dtype=DTYPE)

        ### Checks

        if self.state0.shape[0] != self.dim or self.state0.shape[1] != self.Nm:
            raise Exception('state0 is not of shape (dim, Nm) = (%s, %s).' % (self.dim, self.Nm))
        
    cdef int _run_step(self, int step_i) except -1:
        raise Exception('Abstract method.')
    
    cdef int pre_simulate_batch(self, int N_steps) except -1:
        pass

    cpdef simulate_batch(self, int N_steps, object event_func=None, int N_event=-1):
        cdef int i

        self.states_arr = np.zeros((N_steps+1, self.dim, self.Nm_save), dtype=DTYPE)
        self.accepts_arr = np.zeros(N_steps+1, dtype=bool)
        self.actions_arr = np.zeros(N_steps+1, dtype=DTYPE)
        self.paths_arr = np.zeros((N_steps+1, self.dim, self.Nq), dtype=DTYPE)
        self.d_paths_arr = np.zeros((N_steps+1, self.dim, self.Nq), dtype=DTYPE)

        self.prev_full_state_arr[:] = self.current_state

        self.states_arr[0,:,:] = self.current_state[:,:self.Nm_save]
        self.accepts_arr[0] = self.current_accept
        self.actions_arr[0] = self.current_action
        self.paths_arr[0,:,:] = self.current_path
        self.d_paths_arr[0,:,:] = self.current_d_path

        cdef double time_start = time.time()
        cdef double time_elapsed

        self.pre_simulate_batch(N_steps)

        for i in range(1, N_steps+1):
            self._run_step(i)

            self.steps += 1

            if N_event != -1 and self.steps % N_event == 0:
                time_elapsed = time.time() - time_start
                time_per_step = time_elapsed / (i+1)
                event_func(i+1, time_per_step, time_elapsed)

        self.current_state[:] = self.prev_full_state_arr
        self.current_accept = self.accepts_arr[self.accepts_arr.shape[0]-1]
        self.current_action = self.actions_arr[self.actions_arr.shape[0]-1]
        self.current_path[:] = self.paths_arr[self.paths_arr.shape[0]-1,:,:]
        self.current_d_path[:] = self.d_paths_arr[self.d_paths_arr.shape[0]-1,:,:]

        if self.steps == 0:
            res = self.states_arr, self.accepts_arr, self.actions_arr, self.paths_arr, self.d_paths_arr
        else:
            res = self.states_arr[1:], self.accepts_arr[1:], self.actions_arr[1:], self.paths_arr[1:], self.d_paths_arr[1:]

        return res

    def simulate(self, int N_steps, float batch_size=1, int N_save=-1, int N_paths_save=-1, bint use_GB=True, bint verbose=False, int N_verbose=10000):
        self.steps = 0

        if N_save == -1:
            N_save = N_steps
        if N_paths_save == -1:
            N_paths_save = N_steps

        cdef int N_batch = utils.get_N_steps_to_match_batch_size(self.Nm_save, self.Nq, self.dim, batch_size, use_GB=use_GB)
        cdef int batches = int(np.ceil(float(N_steps) / N_batch))
        cdef DTYPE_t save_ratio = min(N_save/float(N_steps), 1.0)
        cdef DTYPE_t path_save_ratio = min(N_paths_save/float(N_steps), 1.0)
        cdef int N_batch_save
        cdef int i
        cdef N_steps_left = N_steps
        
        if verbose:
            def print_time_left(step_i, time_per_step, time_elapsed):
                steps_left = N_steps - self.steps
                
                time_left = steps_left * time_per_step
                if time_left < 60:
                    time_left_str = '%s secs' % int(np.round(time_left))
                if time_left > 60:
                    time_left_str = '%s mins' % np.round(time_left/60, 2)
                if time_left > 60*60:
                    time_left_str = '%s hrs' % np.round(time_left/(60*60),2)
                if time_left > 60*60*24:
                    time_left_str = '%s days' % np.round(time_left/(60*60*24),2)

                if self.steps != 0:
                    total = 0
                    accept_count = np.count_nonzero(self.accepts_arr[:step_i])
                    total += len(self.accepts_arr[:step_i])
                    for acc_arr in accepts:
                        accept_count += np.count_nonzero(acc_arr)
                        total += len(acc_arr)
                    accept_rate = np.round(accept_count / total, 4)
                else:
                    accept_rate = np.nan

                print('Time Left: %s. Steps %s/%s. Acceptance rate: %s'
                    % (time_left_str, "{:.2e}".format(self.steps), "{:.2e}".format(N_steps), accept_rate), end='\r')
        else:
            print_time_left = None
            N_verbose = -1

        states = []
        accepts = []
        actions = []
        paths = []
        d_paths = []
        
        for i in range(batches):
            if N_batch > N_steps_left:
                N_batch = N_steps_left

            states_arr, accepts_arr, actions_arr, paths_arr, d_paths_arr = self.simulate_batch(N_batch, event_func=print_time_left, N_event=N_verbose)

            N_batch_save = int(np.round(save_ratio*states_arr.shape[0]))
            choice = np.random.choice(states_arr.shape[0], N_batch_save, replace=False)
            choice = np.sort(choice)
            
            states.append( np.copy(states_arr[choice,:,:]) )
            accepts.append( np.copy(accepts_arr[choice]) )
            actions.append( np.copy(actions_arr[choice]) )

            N_batch_paths_save = int(np.round(path_save_ratio*paths_arr.shape[0]))
            choice = np.random.choice(paths_arr.shape[0], N_batch_paths_save, replace=False)
            choice = np.sort(choice)

            paths.append( np.copy(paths_arr[choice]) )
            d_paths.append( np.copy(d_paths_arr[choice]) )

            del states_arr
            del accepts_arr
            del actions_arr
            del paths_arr
            del d_paths_arr

            N_steps_left -= N_batch

        states = np.concatenate(states)
        accepts = np.concatenate(accepts)
        actions = np.concatenate(actions)
        paths = np.concatenate(paths)
        d_paths = np.concatenate(d_paths)

        return states, accepts, actions, paths, d_paths
        
    cpdef run_step(self):
        return self.simulate_batch(1)