#!python
#cython: boundscheck=False

from cython.parallel import prange
cimport numpy as np
cimport openmp

def qifint(np.ndarray[np.float64_t, ndim=1] v, np.ndarray[np.float64_t, ndim=1] exit0,
           np.ndarray[np.int_t, ndim=1] spike, np.ndarray[np.float64_t, ndim=1] eta_0,
           np.ndarray[np.float64_t, ndim=1] s_0, float tiempo, Py_ssize_t number, int dn, float dt, float tau,
           float vpeak, float refr_tau, float tau_peak):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is computed when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the midpoint
    in the refractory period, t_peak.
    :rtype : object
    """
    cdef Py_ssize_t n
    cdef int num_threads
    with nogil:
        # openmp.omp_set_dynamic(1)

        num_threads = openmp.omp_get_num_threads()

        t = tiempo * 1.0
        for n in prange(number):
            spike[n] = 0
            if t >= exit0[n]:
                v[n] = v[n] + (dt / tau) * (v[n] * v[n] + eta_0[n] + tau * s_0[(n / dn)])  # Euler integration
                if v[n] >= vpeak:
                    exit0[n] = t + refr_tau - (tau_peak - 1.0 / v[n])
                    spike[n] = 1
                    v[n] = -v[n]

    return v, exit0, spike
