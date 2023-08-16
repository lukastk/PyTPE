# cython: profile=True
# cython: unraisable_tracebacks=False
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from modemc.defs import DTYPE
import numpy as np

cdef class System:

    def __init__(self, params):
        self.beta = params['beta']
        self.gamma = params['gamma']
        self.Tf = params['Tf']
        self.dim = params['dim']

        if self.dim == 1:
            self.x0 = np.array([ float(params['x0']) ], dtype=DTYPE)
            self.x1 = np.array([ float(params['x1']) ], dtype=DTYPE)
        else:
            self.x0 = np.copy(params['x0']).astype(DTYPE)
            self.x1 = np.copy(params['x1']).astype(DTYPE)

    cdef DTYPE_t _compute_potential(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:] potential):
        raise Exception('Abstract method')

    cdef DTYPE_t _compute_force(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] force):
        raise Exception('Abstract method')

    cdef DTYPE_t _compute_force_and_div_force(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] force, DTYPE_t[:] div_force):
        raise Exception('Abstract method')

    cdef DTYPE_t _compute_gradL_OM(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:] L_x, DTYPE_t[:,:] L_xd):
        raise Exception('Abstract method')

    cdef DTYPE_t _compute_gradL_FW(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:] L_x, DTYPE_t[:,:] L_xd):
        raise Exception('Abstract method')

    cdef DTYPE_t _compute_hessL_OM(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:,:] L_x_x, DTYPE_t[:,:,:] L_xd_x, DTYPE_t[:,:,:] L_xd_xd):
        raise Exception('Abstract method')

    cdef DTYPE_t _compute_hessL_FW(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:,:] L_x_x, DTYPE_t[:,:,:] L_xd_x, DTYPE_t[:,:,:] L_xd_xd):
        raise Exception('Abstract method')
        
    cdef DTYPE_t _compute_dot_L_xd_x_OM(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:,:] L_xd_x):
        raise Exception('Abstract method')

    cdef DTYPE_t _compute_dot_L_xd_x_FW(self, DTYPE_t[:] ts, DTYPE_t[:,:] x, DTYPE_t[:,:] xd, DTYPE_t[:,:,:] L_xd_x):
        raise Exception('Abstract method')

    def __format_time_array(self, n, arr_dim, dtype=DTYPE):
        if n is None:
            return n

        if type(n) != list and type(n) != np.ndarray:
            n = np.array([n], dtype=dtype)
        n = np.array(n, dtype=dtype)

        while len(n.shape) < 1+arr_dim:
            n = np.array([n], dtype=dtype)

        return n

    def compute_potential(self, *Xs, T=None):
        N_pts = np.size(Xs[0])
        dim = len(Xs)
        xs = np.zeros((dim, N_pts))
        for i in range(dim):
            xs[i,:] = Xs[i].flatten()
        if not T is None:
            ts = np.array(T).flatten()
        else:
            ts = None

        potential = np.zeros(N_pts)
        self._compute_potential(ts, xs, potential)
        potential = potential.reshape(Xs[0].shape)

        return potential

    def compute_force(self, *Xs, T=None):
        N_pts = np.size(Xs[0])
        dim = len(Xs)
        xs = np.zeros((dim, N_pts))
        for i in range(dim):
            xs[i,:] = Xs[i].flatten()
        if not T is None:
            ts = np.array(T).flatten()
        else:
            ts = None

        force = np.zeros((dim, N_pts))
        self._compute_force(ts, xs, force)

        force_meshes = []
        for i in range(dim):
            force_meshes.append(force[i,:].reshape(Xs[0].shape))

        return force_meshes

    #cpdef compute_force(self, path, ts=None):
        #ts = self.__format_time_array(ts, 0)
        #path = self.__format_time_array(path, 1)

        #force = np.zeros(path.shape)
        #self._compute_force(ts, path, force)
        #return force

    cpdef compute_force_and_div_force(self, path, ts=None):
        ts =  self.__format_time_array(ts, 0)
        path = self.__format_time_array(path, 1)

        force = np.zeros(path.shape)
        div_force = np.zeros(len(ts))
        self._compute_force_and_div_force(ts, path, force, div_force)
        return force, div_force

    cpdef compute_gradL(self, path, d_path, use_OM=True, ts=None):
        ts = self.__format_time_array(path, 1)
        path = self.__format_time_array(path, 1)
        d_path = self.__format_time_array(d_path, 1)

        L_x = np.zeros(path.shape)
        L_xd = np.zeros(path.shape)

        if use_OM:
            self._compute_gradL_OM(ts, path, d_path, L_x, L_xd)
        else:
            self._compute_gradL_FW(ts, path, d_path, L_x, L_xd)

        return L_x, L_xd

    cpdef compute_hessL(self, path, d_path, use_OM=True, ts=None):
        ts = self.__format_time_array(path, 1)
        path = self.__format_time_array(path, 1)
        d_path = self.__format_time_array(d_path, 1)

        dim, Nt = path.shape

        L_x_x = np.zeros((dim, dim, Nt))
        L_xd_x = np.zeros((dim, dim, Nt))
        L_xd_xd = np.zeros((dim, dim, Nt))

        if use_OM:
            self._compute_hessL_OM(ts, path, d_path, L_x_x, L_xd_x, L_xd_xd)
        else:
            self._compute_hessL_FW(ts, path, d_path, L_x_x, L_xd_x, L_xd_xd)

        return L_x_x, L_xd_x, L_xd_xd

    cpdef compute_dot_L_xd_x(self, path, d_path, use_OM=True, ts=None):
        ts = self.__format_time_array(path, 1)
        path = self.__format_time_array(path, 1)
        d_path = self.__format_time_array(d_path, 1)

        dim, Nt = path.shape

        dot_L_xd_x = np.zeros((dim, dim, Nt))

        if use_OM:
            self._compute_dot_L_xd_x_OM(ts, path, d_path, dot_L_xd_x)
        else:
            self._compute_dot_L_xd_x_FW(ts, path, d_path, dot_L_xd_x)

        return dot_L_xd_x