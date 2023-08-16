# cython: profile=False
# cython: unraisable_tracebacks=False
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from modemc.defs import DTYPE
import numpy as np
import cython

cdef DTYPE_t _eval_action(DTYPE_t[:] quad_weights, DTYPE_t Tf, DTYPE_t[:] lagrangian):
    cdef DTYPE_t S = 0
    #cdef DTYPE_t time_rescaling = (0.5*Tf)
    cdef int i

    for i in range(lagrangian.shape[0]):
        S += lagrangian[i] * quad_weights[i]
    #S *= time_rescaling

    return S

cdef DTYPE_t _eval_gaussian_action(DTYPE_t[:] x0, DTYPE_t[:] x1, DTYPE_t Tf, DTYPE_t beta, DTYPE_t gamma, DTYPE_t[:,:] state):
    cdef DTYPE_t S = 0
    cdef int dim = state.shape[0]
    cdef int Nm = state.shape[1]

    for i in range(dim):
        S += (x1[i] - x0[i])**2
    S /= Tf
    
    for i in range(dim):
        for k in range(Nm):
            S += state[i,k]**2

    S *= beta * gamma * 0.25

    return S

cdef int _eval_action_gradient(np.ndarray wphis, np.ndarray wdphis, np.ndarray L_x, np.ndarray L_xd, np.ndarray out_gradS_buffer, np.ndarray out_gradS) except -1:
    np.matmul(L_x, wphis, out_gradS)
    np.matmul(L_xd, wdphis, out_gradS_buffer)
    np.add(out_gradS, out_gradS_buffer, out_gradS)

cdef int _fast_eval_path(np.ndarray bb_mean, np.ndarray dbb_mean, np.ndarray phis, np.ndarray dphis, np.ndarray state, np.ndarray out_path, np.ndarray out_d_path) except -1:
    np.matmul(state, phis, out_path)
    np.matmul(state, dphis, out_d_path)
    np.add(out_path, bb_mean, out_path)
    np.add(out_d_path, dbb_mean, out_d_path)

cdef int _fast_eval_path_without_mean(np.ndarray phis, np.ndarray dphis, np.ndarray state, np.ndarray out_path, np.ndarray out_d_path) except -1:
    np.matmul(state, phis, out_path)
    np.matmul(state, dphis, out_d_path)

cdef int _fast_update_path(np.ndarray phis, np.ndarray dphis, np.ndarray state, np.ndarray out_path, np.ndarray out_d_path, np.ndarray path_buffer) except -1:
    np.matmul(state, phis, path_buffer)
    np.add(out_path, path_buffer, out_path)
    np.matmul(state, dphis, path_buffer)
    np.add(out_d_path, path_buffer, out_d_path)

# Random sampler functions
# Source: https://stackoverflow.com/questions/42767816/what-is-the-most-efficient-and-portable-way-to-generate-gaussian-random-numbers


@cython.boundscheck(False)
cdef DTYPE_t random_uniform():
    cdef DTYPE_t r = rand()
    return r / RAND_MAX

@cython.boundscheck(False)
cdef DTYPE_t random_gaussian():
    cdef DTYPE_t x1, x2, w

    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w

@cython.boundscheck(False)
cdef void assign_random_gaussian_pair(double[:] out, int assign_ix):
    cdef DTYPE_t x1, x2, w

    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = sqrt((-2.0 * log(w)) / w)
    out[assign_ix] = x1 * w
    out[assign_ix + 1] = x2 * w

@cython.boundscheck(False)
cdef void uniform_vector(DTYPE_t[:] result):
    cdef int i
    cdef int n = result.shape[0]
    for i in range(n):
        result[i] = random_uniform()

@cython.boundscheck(False)
cdef void gaussian_vector(DTYPE_t[:] result):
    cdef int i
    cdef int n = result.shape[0]
    for i in range(n // 2):  # Int division ensures trailing index if n is odd.
        assign_random_gaussian_pair(result, i * 2)
    if n % 2 == 1:
        result[n - 1] = random_gaussian()