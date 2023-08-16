#cython: unraisable_tracebacks=False
#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from modemc.defs import DTYPE
import numpy as np

cdef class DoubleWell:
    pass

    def __init__(self, params):
        if 'dim' in params:
            if params['dim'] != 1:
                raise Exception('dim must be 1.')
        params['dim'] = 1

        super().__init__(params)

        self.U0 = params['U0']
        self.dU = params['dU']

    cdef DTYPE_t _compute_potential(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:] potential):
        cdef:
            int i
            int dim = x.shape[0]
            int Nt = x.shape[1]
            DTYPE_t U0 = self.U0
            DTYPE_t dU = self.dU
            DTYPE_t var_2p0_tim_U0 = 2.0*U0
            DTYPE_t var_0p75_tim_dU = 0.75*dU
            DTYPE_t var_0p25_tim_dU = 0.25*dU
            DTYPE_t var_var_0p5_tim_dU_pls_var_1p0_tim_U0 = 1.0*U0 + 0.5*dU
        
        for i in range(Nt):
            potential[i] = U0*pow(x[0,i], 4) - var_0p25_tim_dU*pow(x[0,i], 3) + var_0p75_tim_dU*x[0,i] - var_2p0_tim_U0*pow(x[0,i], 2) + var_var_0p5_tim_dU_pls_var_1p0_tim_U0

    cdef DTYPE_t _compute_force(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] force):
        cdef:
            int i
            int dim = x.shape[0]
            int Nt = x.shape[1]
            DTYPE_t U0 = self.U0
            DTYPE_t dU = self.dU
            DTYPE_t var_4p0_tim_U0 = 4.0*U0
            DTYPE_t var_0p75_tim_dU = 0.75*dU
        
        for i in range(Nt):
            force[0,i] = var_0p75_tim_dU*pow(x[0,i], 2) - var_0p75_tim_dU - var_4p0_tim_U0*pow(x[0,i], 3) + var_4p0_tim_U0*x[0,i]

    cdef DTYPE_t _compute_force_and_div_force(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] force, DTYPE_t[:] div_force):
        cdef:
            int i
            int dim = x.shape[0]
            int Nt = x.shape[1]
            DTYPE_t U0 = self.U0
            DTYPE_t dU = self.dU
            DTYPE_t var_4p0_tim_U0 = 4.0*U0
            DTYPE_t var_12p0_tim_U0 = 12.0*U0
            DTYPE_t var_1p5_tim_dU = 1.5*dU
            DTYPE_t var_0p75_tim_dU = 0.75*dU
        
        for i in range(Nt):
            force[0,i] = var_0p75_tim_dU*pow(x[0,i], 2) - var_0p75_tim_dU - var_4p0_tim_U0*pow(x[0,i], 3) + var_4p0_tim_U0*x[0,i]
            div_force[i] = -var_12p0_tim_U0*pow(x[0,i], 2) + var_1p5_tim_dU*x[0,i] + var_4p0_tim_U0

    cdef DTYPE_t _compute_gradL_OM(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:] L_x, DTYPE_t[:,:] L_xd):
        cdef:
            int i
            int dim = x.shape[0]
            int Nt = x.shape[1]
            DTYPE_t beta = self.beta
            DTYPE_t gamma = self.gamma
            DTYPE_t U0 = self.U0
            DTYPE_t dU = self.dU
            DTYPE_t var_var_1_div_gamma_tim_var_1p5_tim_U0_tim_beta_tim_dU = 1.5*U0*beta*dU/gamma
            DTYPE_t var_var_0p5625_tim_beta_tim_var_1_div_gamma_tim_var_dU_pow_2 = 0.5625*beta*dU**2/gamma
            DTYPE_t var_var_1_div_gamma_tim_var_32p0_tim_beta_tim_var_U0_pow_2 = 32.0*U0**2*beta/gamma
            DTYPE_t var_var_1_div_gamma_tim_var_9p0_tim_U0_tim_beta_tim_dU = 9.0*U0*beta*dU/gamma
            DTYPE_t var_2p0_tim_U0_tim_beta = 2.0*U0*beta
            DTYPE_t var_var_1_div_gamma_tim_var_7p5_tim_U0_tim_beta_tim_dU = 7.5*U0*beta*dU/gamma
            DTYPE_t var_var_0p75_tim_dU_tim_var_1_div_gamma = 0.75*dU/gamma
            DTYPE_t var_0p375_tim_beta_tim_dU = 0.375*beta*dU
            DTYPE_t var_6p0_tim_U0_tim_beta = 6.0*U0*beta
            DTYPE_t var_0p5_tim_beta_tim_gamma = 0.5*beta*gamma
            DTYPE_t var_var_1_div_gamma_tim_var_24p0_tim_beta_tim_var_U0_pow_2 = 24.0*U0**2*beta/gamma
            DTYPE_t var_var_1_div_gamma_tim_var_8p0_tim_beta_tim_var_U0_pow_2 = 8.0*U0**2*beta/gamma
            DTYPE_t var_var_12p0_tim_U0_tim_var_1_div_gamma = 12.0*U0/gamma
            DTYPE_t var_0p75_tim_beta_tim_dU = 0.75*beta*dU
        
        for i in range(Nt):
            L_x[0,i] = -var_0p75_tim_beta_tim_dU*x[0,i]*xd[0,i] - var_2p0_tim_U0_tim_beta*xd[0,i] + var_6p0_tim_U0_tim_beta*pow(x[0,i], 2)*xd[0,i] + var_var_0p5625_tim_beta_tim_var_1_div_gamma_tim_var_dU_pow_2*pow(x[0,i], 3) - var_var_0p5625_tim_beta_tim_var_1_div_gamma_tim_var_dU_pow_2*x[0,i] + var_var_0p75_tim_dU_tim_var_1_div_gamma - var_var_12p0_tim_U0_tim_var_1_div_gamma*x[0,i] - var_var_1_div_gamma_tim_var_1p5_tim_U0_tim_beta_tim_dU + var_var_1_div_gamma_tim_var_24p0_tim_beta_tim_var_U0_pow_2*pow(x[0,i], 5) - var_var_1_div_gamma_tim_var_32p0_tim_beta_tim_var_U0_pow_2*pow(x[0,i], 3) - var_var_1_div_gamma_tim_var_7p5_tim_U0_tim_beta_tim_dU*pow(x[0,i], 4) + var_var_1_div_gamma_tim_var_8p0_tim_beta_tim_var_U0_pow_2*x[0,i] + var_var_1_div_gamma_tim_var_9p0_tim_U0_tim_beta_tim_dU*pow(x[0,i], 2)
            L_xd[0,i] = -var_0p375_tim_beta_tim_dU*pow(x[0,i], 2) + var_0p375_tim_beta_tim_dU + var_0p5_tim_beta_tim_gamma*xd[0,i] + var_2p0_tim_U0_tim_beta*pow(x[0,i], 3) - var_2p0_tim_U0_tim_beta*x[0,i]

    cdef DTYPE_t _compute_gradL_FW(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:] L_x, DTYPE_t[:,:] L_xd):
        cdef:
            int i
            int dim = x.shape[0]
            int Nt = x.shape[1]
            DTYPE_t beta = self.beta
            DTYPE_t gamma = self.gamma
            DTYPE_t U0 = self.U0
            DTYPE_t dU = self.dU
            DTYPE_t var_var_1_div_gamma_tim_var_1p5_tim_U0_tim_beta_tim_dU = 1.5*U0*beta*dU/gamma
            DTYPE_t var_var_0p5625_tim_beta_tim_var_1_div_gamma_tim_var_dU_pow_2 = 0.5625*beta*dU**2/gamma
            DTYPE_t var_var_1_div_gamma_tim_var_32p0_tim_beta_tim_var_U0_pow_2 = 32.0*U0**2*beta/gamma
            DTYPE_t var_var_1_div_gamma_tim_var_9p0_tim_U0_tim_beta_tim_dU = 9.0*U0*beta*dU/gamma
            DTYPE_t var_2p0_tim_U0_tim_beta = 2.0*U0*beta
            DTYPE_t var_var_1_div_gamma_tim_var_7p5_tim_U0_tim_beta_tim_dU = 7.5*U0*beta*dU/gamma
            DTYPE_t var_0p375_tim_beta_tim_dU = 0.375*beta*dU
            DTYPE_t var_6p0_tim_U0_tim_beta = 6.0*U0*beta
            DTYPE_t var_0p5_tim_beta_tim_gamma = 0.5*beta*gamma
            DTYPE_t var_var_1_div_gamma_tim_var_24p0_tim_beta_tim_var_U0_pow_2 = 24.0*U0**2*beta/gamma
            DTYPE_t var_var_1_div_gamma_tim_var_8p0_tim_beta_tim_var_U0_pow_2 = 8.0*U0**2*beta/gamma
            DTYPE_t var_0p75_tim_beta_tim_dU = 0.75*beta*dU
        
        for i in range(Nt):
            L_x[0,i] = -var_0p75_tim_beta_tim_dU*x[0,i]*xd[0,i] - var_2p0_tim_U0_tim_beta*xd[0,i] + var_6p0_tim_U0_tim_beta*pow(x[0,i], 2)*xd[0,i] + var_var_0p5625_tim_beta_tim_var_1_div_gamma_tim_var_dU_pow_2*pow(x[0,i], 3) - var_var_0p5625_tim_beta_tim_var_1_div_gamma_tim_var_dU_pow_2*x[0,i] - var_var_1_div_gamma_tim_var_1p5_tim_U0_tim_beta_tim_dU + var_var_1_div_gamma_tim_var_24p0_tim_beta_tim_var_U0_pow_2*pow(x[0,i], 5) - var_var_1_div_gamma_tim_var_32p0_tim_beta_tim_var_U0_pow_2*pow(x[0,i], 3) - var_var_1_div_gamma_tim_var_7p5_tim_U0_tim_beta_tim_dU*pow(x[0,i], 4) + var_var_1_div_gamma_tim_var_8p0_tim_beta_tim_var_U0_pow_2*x[0,i] + var_var_1_div_gamma_tim_var_9p0_tim_U0_tim_beta_tim_dU*pow(x[0,i], 2)
            L_xd[0,i] = -var_0p375_tim_beta_tim_dU*pow(x[0,i], 2) + var_0p375_tim_beta_tim_dU + var_0p5_tim_beta_tim_gamma*xd[0,i] + var_2p0_tim_U0_tim_beta*pow(x[0,i], 3) - var_2p0_tim_U0_tim_beta*x[0,i]

    cdef DTYPE_t _compute_hessL_OM(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:,:] L_x_x, DTYPE_t[:,:,:] L_xd_x, DTYPE_t[:,:,:] L_xd_xd):
        cdef:
            int i
            int dim = x.shape[0]
            int Nt = x.shape[1]
            DTYPE_t beta = self.beta
            DTYPE_t gamma = self.gamma
            DTYPE_t U0 = self.U0
            DTYPE_t dU = self.dU
            DTYPE_t var_var_18p0_tim_U0_tim_beta_tim_dU_tim_var_1_div_gamma = 18.0*U0*beta*dU/gamma
            DTYPE_t var_var_0p5625_tim_beta_tim_var_1_div_gamma_tim_var_dU_pow_2 = 0.5625*beta*dU**2/gamma
            DTYPE_t var_12p0_tim_U0_tim_beta = 12.0*U0*beta
            DTYPE_t var_2p0_tim_U0_tim_beta = 2.0*U0*beta
            DTYPE_t var_0p5_tim_beta_tim_gamma = 0.5*beta*gamma
            DTYPE_t var_var_1_div_gamma_tim_var_30p0_tim_U0_tim_beta_tim_dU = 30.0*U0*beta*dU/gamma
            DTYPE_t var_6p0_tim_U0_tim_beta = 6.0*U0*beta
            DTYPE_t var_var_1_div_gamma_tim_var_1p6875_tim_beta_tim_var_dU_pow_2 = 1.6875*beta*dU**2/gamma
            DTYPE_t var_var_1_div_gamma_tim_var_96p0_tim_beta_tim_var_U0_pow_2 = 96.0*U0**2*beta/gamma
            DTYPE_t var_var_1_div_gamma_tim_var_8p0_tim_beta_tim_var_U0_pow_2 = 8.0*U0**2*beta/gamma
            DTYPE_t var_var_12p0_tim_U0_tim_var_1_div_gamma = 12.0*U0/gamma
            DTYPE_t var_0p75_tim_beta_tim_dU = 0.75*beta*dU
            DTYPE_t var_var_120p0_tim_beta_tim_var_1_div_gamma_tim_var_U0_pow_2 = 120.0*U0**2*beta/gamma
        
        for i in range(Nt):
            L_x_x[0,0,i] = -var_0p75_tim_beta_tim_dU*xd[0,i] + var_12p0_tim_U0_tim_beta*x[0,i]*xd[0,i] - var_var_0p5625_tim_beta_tim_var_1_div_gamma_tim_var_dU_pow_2 + var_var_120p0_tim_beta_tim_var_1_div_gamma_tim_var_U0_pow_2*pow(x[0,i], 4) - var_var_12p0_tim_U0_tim_var_1_div_gamma + var_var_18p0_tim_U0_tim_beta_tim_dU_tim_var_1_div_gamma*x[0,i] + var_var_1_div_gamma_tim_var_1p6875_tim_beta_tim_var_dU_pow_2*pow(x[0,i], 2) - var_var_1_div_gamma_tim_var_30p0_tim_U0_tim_beta_tim_dU*pow(x[0,i], 3) + var_var_1_div_gamma_tim_var_8p0_tim_beta_tim_var_U0_pow_2 - var_var_1_div_gamma_tim_var_96p0_tim_beta_tim_var_U0_pow_2*pow(x[0,i], 2)
            L_xd_x[0,0,i] = -var_0p75_tim_beta_tim_dU*x[0,i] - var_2p0_tim_U0_tim_beta + var_6p0_tim_U0_tim_beta*pow(x[0,i], 2)
            L_xd_xd[0,0,i] = var_0p5_tim_beta_tim_gamma

    cdef DTYPE_t _compute_hessL_FW(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:,:] L_x_x, DTYPE_t[:,:,:] L_xd_x, DTYPE_t[:,:,:] L_xd_xd):
        cdef:
            int i
            int dim = x.shape[0]
            int Nt = x.shape[1]
            DTYPE_t beta = self.beta
            DTYPE_t gamma = self.gamma
            DTYPE_t U0 = self.U0
            DTYPE_t dU = self.dU
            DTYPE_t var_var_18p0_tim_U0_tim_beta_tim_dU_tim_var_1_div_gamma = 18.0*U0*beta*dU/gamma
            DTYPE_t var_var_0p5625_tim_beta_tim_var_1_div_gamma_tim_var_dU_pow_2 = 0.5625*beta*dU**2/gamma
            DTYPE_t var_12p0_tim_U0_tim_beta = 12.0*U0*beta
            DTYPE_t var_2p0_tim_U0_tim_beta = 2.0*U0*beta
            DTYPE_t var_0p5_tim_beta_tim_gamma = 0.5*beta*gamma
            DTYPE_t var_var_1_div_gamma_tim_var_30p0_tim_U0_tim_beta_tim_dU = 30.0*U0*beta*dU/gamma
            DTYPE_t var_6p0_tim_U0_tim_beta = 6.0*U0*beta
            DTYPE_t var_var_1_div_gamma_tim_var_1p6875_tim_beta_tim_var_dU_pow_2 = 1.6875*beta*dU**2/gamma
            DTYPE_t var_var_1_div_gamma_tim_var_96p0_tim_beta_tim_var_U0_pow_2 = 96.0*U0**2*beta/gamma
            DTYPE_t var_var_1_div_gamma_tim_var_8p0_tim_beta_tim_var_U0_pow_2 = 8.0*U0**2*beta/gamma
            DTYPE_t var_0p75_tim_beta_tim_dU = 0.75*beta*dU
            DTYPE_t var_var_120p0_tim_beta_tim_var_1_div_gamma_tim_var_U0_pow_2 = 120.0*U0**2*beta/gamma
        
        for i in range(Nt):
            L_x_x[0,0,i] = -var_0p75_tim_beta_tim_dU*xd[0,i] + var_12p0_tim_U0_tim_beta*x[0,i]*xd[0,i] - var_var_0p5625_tim_beta_tim_var_1_div_gamma_tim_var_dU_pow_2 + var_var_120p0_tim_beta_tim_var_1_div_gamma_tim_var_U0_pow_2*pow(x[0,i], 4) + var_var_18p0_tim_U0_tim_beta_tim_dU_tim_var_1_div_gamma*x[0,i] + var_var_1_div_gamma_tim_var_1p6875_tim_beta_tim_var_dU_pow_2*pow(x[0,i], 2) - var_var_1_div_gamma_tim_var_30p0_tim_U0_tim_beta_tim_dU*pow(x[0,i], 3) + var_var_1_div_gamma_tim_var_8p0_tim_beta_tim_var_U0_pow_2 - var_var_1_div_gamma_tim_var_96p0_tim_beta_tim_var_U0_pow_2*pow(x[0,i], 2)
            L_xd_x[0,0,i] = -var_0p75_tim_beta_tim_dU*x[0,i] - var_2p0_tim_U0_tim_beta + var_6p0_tim_U0_tim_beta*pow(x[0,i], 2)
            L_xd_xd[0,0,i] = var_0p5_tim_beta_tim_gamma