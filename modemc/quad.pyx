#cython: unraisable_tracebacks=False
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from modemc.defs import DTYPE
import numpy as np

def clenshaw_curtis(Nq, Tf):
    Nq -= 1 # Subtract 1 so that nq does actually correspond to the size of ts and w

    ## Generate quadrature nodes
    theta = np.pi*np.arange(0, Nq+1) / (Nq)
    qts = -np.cos(theta) # Minus sign here is so that the the points are in increasing order

    ## Generate quadrature weights
    # From Spectral Methods in Matlab (Trefethen)
    w = np.zeros(Nq+1)
    x = np.cos(theta)
    ii = np.arange(1, Nq)
    v = np.ones(Nq-1)

    if Nq % 2 == 0:
        w[0] = 1 / (Nq**2 - 1)
        w[Nq-1] = w[0]
        for k in range(1, int(Nq/2)):
            v = v - 2*np.cos(2 * k * theta[ii]) / (4 * k*k - 1)
        v = v - np.cos( Nq * theta[ii]) / ( Nq**2 - 1 )
    else:
        w[0] = 1 / (Nq**2)
        w[Nq-1] = w[0]

        for k in range(1, int((Nq-1)/2) + 1):
            v = v - 2*np.cos(2 * k * theta[ii]) / (4 * k*k - 1)

    w[ii] = 2 * v / Nq

    ts = 0.5*(qts+1)*Tf

    # Include the time rescaling from [-1,1] to [0, Tf] in the weights
    w *= 0.5*Tf

    return ts, qts, w

def uniform(Nq, Tf):
    ts = np.linspace(0, Tf, Nq)
    qts = ts
    dt = ts[1]-ts[0]
    ws = np.full(Nq, dt)
    ws[0] = ws[Nq-1] = dt/2

    return ts, qts, ws

def discrete(Nq, Tf):
    """For the discretised OM and FW functionals."""
    ts = np.linspace(0, Tf, Nq)
    qts = ts
    ws = np.full(Nq, 1.0)

    return ts, qts, ws