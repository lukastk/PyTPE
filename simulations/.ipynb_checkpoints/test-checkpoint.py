import kklmc
import numpy as np
import matplotlib.pyplot as plt

N_sim = int(1e2)
Nq_factor = 10
Nm = 500

beta = 0.5
gamma = 1
Tf = 2

U0 = 1
dU = 0.5

x0, x1 = -1, 0
kappa = 0.24

window_size = 50
window_probs = np.full(Nm-window_size+1, 1)

#window_probs = np.arange(len(window_probs), 0, -1)

params = {
    'use_OM' : True,
    
    'beta' : beta,
    'gamma' : gamma,
    'Tf' : Tf,
    
    'U0' : U0,
    'dU' : dU,
    
    'dim' : 1,
    'Nq_factor' : Nq_factor,
    'Nm' : Nm,
    
    'x0' : x0,
    'x1' : x1,
    
    'kappa' : kappa,
    
    'window_probs' : window_probs
}

params['state0'] = np.zeros((params['Nm'], params['dim']))
params['system'] = kklmc.systems.DoubleWell(params)
system = params['system']

sim = kklmc.protocols.pCN_Window(params)

w_states, kklmc_accepts, kklmc_actions, kklmc_paths, kklmc_d_paths = sim.simulate(N_sim)