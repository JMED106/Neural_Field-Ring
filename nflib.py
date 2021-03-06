import numpy as np
from scipy import stats, special
from scipy.fftpack import dct
from scipy.optimize import fsolve

__author__ = 'Jose M. Esnaola Acebes'

""" This file contains classes and functions to be used in the Neural Field simulation.

    Data: (to store parameters, variables, and some functions)
    *****

    Connectivity:  (to build the connectivity matrix of the network)
    *************
    1. Functions:

        1.1. Gaussian pdf with mean 0.
        1.2. Non-periodic Mex-Hat function.
        1.3. Circular gaussian function: mex-hat type. (von Mises functions).

    2. Methods:

        2.1. Mode extraction by Fourier Series: eingevalues.
        2.2. Reconstruction of a function by means of a Fourier Series using 2.1.
        2.3. Computation of frequencies for a given connectivity (requires parameters).
        2.4. Linear response of the homogeneous state (requires parameters and initial
             conditions).
"""


class Data:
    """ Object designed to store data,
        and perform modifications of this data in case it is necessary.
    """

    def __init__(self, l=100, N=1E5, eta0=0, delta=1.0, t0=0.0, tfinal=50.0,
                 dt=1E-3, delay=0.0, tau=1.0, faketau=20.0E-3, fp='lorentz', system='nf'):

        # 0.1) Network properties:
        self.l = l
        self.dx = 2.0 * np.pi / np.float(l)
        # Zeroth mode, determines firing rate of the homogeneous state
        self.j0 = 0.0  # default value

        # 0.3) Give the model parameters
        self.eta0 = eta0  # Constant external current mean value
        self.delta = delta  # Constant external current distribution width

        # 0.2) Define the temporal resolution and other time-related variables
        self.t0 = t0  # Initial time
        self.tfinal = tfinal  # Final time
        self.total_time = tfinal - t0  # Time of simulation
        self.dt = dt  # Time step

        self.D = delay  # Synaptic time Delay (not implemented)
        self.intD = int(delay / dt)  # Synaptic time Delay in time steps
        self.tpoints = np.arange(t0, tfinal, dt)  # Points for the plots and others
        self.nsteps = len(self.tpoints)  # Total time steps
        self.tau = tau
        self.faketau = faketau  # time scale in ms

        # 0.7) FIRING RATE EQUATIONS
        self.rphi = np.ones((self.nsteps, l))
        self.vphi = np.ones((self.nsteps, l)) * (-0.01)
        self.sphi = np.ones((self.nsteps, l))
        self.rphi[len(self.rphi) - 1, :] = 0.0
        # Load INITIAL CONDITIONS
        self.sphi[len(self.sphi) - 1] = 0.0

        self.system = system

        # 0.8) QIF model parameters
        if system is not 'nf':
            print "Loading QIF parameters:"
            print "***********************"
            self.fp = fp
            # sub-populations
            self.N = N
            # Excitatory and inhibitory populations
            self.neni = 0.5
            self.Ne = int(N * self.neni)
            self.auxne = np.ones((1, self.Ne))
            self.Ni = int(N - self.Ne)  # Number of inhibitory neurons
            self.auxni = np.ones((1, self.Ni))
            # sub-populations
            self.dN = int(np.float(N) / np.float(l))
            if self.dN * l != N:
                print('Warning: N, l not dividable')

            self.dNe = int(self.dN * self.neni)  # Number of exc. neurons in each subpopulation
            self.dNi = self.dN - self.dNe  # Number of inh. neurons in each subpopulation

            self.vpeak = 100.0  # Value of peak voltage (max voltage)
            # self.vreset = -self.vpeak  # Value of resetting voltage (min voltage)
            self.vreset = -100.0
            # --------------
            self.refr_tau = tau / self.vpeak - tau / self.vreset  # Refractory time in which the neuron is not excitable
            self.tau_peak = tau / self.vpeak  # Refractory time till the spike is generated
            # --------------
            self.T_syn = 10  # Number of steps for computing synaptic activation
            self.tau_syn = self.T_syn * dt  # time scale (??)
            # Weighting matrix (synaptic delay, in some sense).
            # We need T_syn vectors in order to improve the performance.
            if self.T_syn == 10:
                # Heaviside
                h_tau = 1.0 / self.tau_syn
                a_tau0 = np.transpose(h_tau * np.ones(self.T_syn))
            else:
                # Exponential (disabled by the Heaviside)
                # self.tau_syn /= 4
                a_tau0 = np.transpose((1.0 / self.tau_syn) * np.exp(-dt * np.arange(self.T_syn) / self.tau_syn))

            self.a_tau = np.zeros((self.T_syn, self.T_syn))  # Multiple weighting vectors (index shifted)
            for i in xrange(self.T_syn):
                self.a_tau[i] = np.roll(a_tau0, i, 0)

            # Distributions of the external current       -- FOR l populations --
            self.etaE = None
            self.etaI = None
            if fp == 'lorentz' or fp == 'gauss':
                print "+ Setting distribution of external currents: "
                self.etaE = np.zeros(self.Ne)
                self.etaI = np.zeros(self.Ni)
                if fp == 'lorentz':
                    print '   - Lorentzian distribution of external currents'
                    # Uniform distribution
                    k = (2.0 * np.arange(1, self.dNe + 1) - self.dNe - 1.0) / (self.dNe + 1.0)
                    # Cauchy ppf (stats.cauchy.ppf can be used here)
                    eta_pop_e = eta0 + delta * np.tan((np.pi / 2.0) * k)

                    k = (2.0 * np.arange(1, self.dNi + 1) - self.dNi - 1.0) / (self.dNi + 1.0)
                    eta_pop_i = eta0 + delta * np.tan((np.pi / 2.0) * k)
                else:
                    print '   - Gaussian distribution of external currents'
                    k = (np.arange(1, self.dNe + 1)) / (self.dNe + 1.0)
                    eta_pop_e = eta0 + delta * stats.norm.ppf(k)
                    k = (np.arange(1, self.dNi + 1)) / (self.dNi + 1.0)
                    eta_pop_i = eta0 + delta * stats.norm.ppf(k)

                del k
                for i in xrange(l):
                    self.etaE[i * self.dNe:(i + 1) * self.dNe] = 1.0 * eta_pop_e
                    self.etaI[i * self.dNi:(i + 1) * self.dNi] = 1.0 * eta_pop_i
                del eta_pop_i, eta_pop_e
            else:
                print "+ Setting homogeneous population of neurons (identical), under GWN."

            # QIF neurons matrices (declaration)
            self.matrixI = np.ones(shape=(self.Ni, 3)) * 0
            self.matrixE = np.ones(shape=(self.Ne, 3)) * 0
            self.spikes_i = np.ones(shape=(self.Ni, self.T_syn)) * 0  # Spike matrix (Ni x T_syn)
            self.spikes_e = np.ones(shape=(self.Ne, self.T_syn)) * 0  # Spike matrix (Ne x T_syn)

            # Single neuron recording (not implemented)
            self.singlev = np.ones(self.nsteps) * 0.0
            self.freqpoints = 25
            self.singleta = np.ones(self.freqpoints)
            self.singlfreqs = np.ones(self.freqpoints)
            self.singlaux = 0
            self.singl0 = 0.0

            # 0.8.1) QIF vectors and matrices (initial conditions are loaded after
            #                                  connectivity  matrix is created)
            self.spiketime = int(self.tau_peak / dt)
            self.s1time = self.T_syn + self.spiketime
            self.spikes_e_mod = np.ones(shape=(self.Ne, self.spiketime)) * 0  # Spike matrix (Ne x (T_syn + tpeak/dt))
            self.spikes_i_mod = np.ones(shape=(self.Ni, self.spiketime)) * 0  # Spike matrix (Ni x (T_syn + tpeak/dt))

        # 0.9) Perturbation parameters
        self.PERT = False
        self.Input = 0.0
        self.It = np.zeros(self.nsteps)
        self.It[0] = self.Input * 1.0
        # input duration
        self.deltat = dt * 50
        self.inputtime = 10.0

        # Simulations parameters
        self.new_ic = False
        self.END = False

    def load_ic(self, j0, system='nf'):
        """ Loads initial conditions based on the parameters. It will try to load system that
            most closely resembles. The available systems are stored in a file.
        """
        # File path variables
        self.filepath = './init_conds/qif/'
        # TODO compute the fixed point taking into account the parameter space: HS or Bump?
        self.r0 = Connectivity.rtheory(j0, self.eta0, self.delta)

        if system == 'nf' or system == 'both':
            self.fileprm = '%.2lf-%.2lf-%.2lf-%d' % (j0, self.eta0, self.delta, self.l)
            self.rphi[(self.nsteps - 1) % self.nsteps, :] = np.ones(self.l) * self.r0
            self.vphi[(self.nsteps - 1) % self.nsteps, :] = np.ones(self.l) * (-self.delta / (2.0 * self.r0 * np.pi))

        if system == 'qif' or system == 'both':
            print "Loading initial conditions ... "
            self.fileprm = '%s_%.2lf-%.2lf-%.2lf-%d' % (self.fp, j0, self.eta0, self.delta, self.l)
            # We first try to load files that correspond to chosen parameters
            try:
                self.spikes_e = np.load("%sic_qif_spikes_e_%s-%d.npy" % (self.filepath, self.fileprm, self.Ne))
                self.spikes_i = np.load("%sic_qif_spikes_i_%s-%d.npy" % (self.filepath, self.fileprm, self.Ni))
                self.matrixE = np.load("%sic_qif_matrixE_%s-%d.npy" % (self.filepath, self.fileprm, self.Ne))
                self.matrixI = np.load("%sic_qif_matrixI_%s-%d.npy" % (self.filepath, self.fileprm, self.Ni))
                print "Successfully loaded all data matrices."
            except IOError:
                print "Files do not exist or cannot be read. Trying the most similar combination."
                self.new_ic = True
            except ValueError:
                print "Not appropriate format of initial conditions. Check the files for logical errors..."
                exit(-1)

            # If the loading fails or new_ic is override we look for the closest combination in the data base
            database = None
            if self.new_ic is True:
                print "WARNING: New initial conditions will be created, wait until the simulation has finish."
                try:
                    database = np.load("%sinitial_conditions.npy" % self.filepath)
                    if np.size(np.shape(database)) < 2:
                        database.resize((1, np.size(database)))
                    load = True
                except IOError:
                    print "Iinitial conditions database not found (%sinitial_conditions)" % self.filepath
                    print "Loading random conditions."
                    load = False

                # If the chosen combination is not in the database we create new initial conditions
                # for that combination: raising a warning to the user.
                # If the database has been successfully loaded we find the closest combination
                # Note that the number of populations must coincide
                if load is True and np.any(database[:, 0] == self.l) \
                        and np.any(database[:, -2] == self.Ne) \
                        and np.any(database[:, -1] == self.Ni):
                    # mask combinations where population number match
                    ma = ((database[:, 0] == self.l) & (database[:, -2] == self.Ne) & (database[:, -1] == self.Ni))
                    # Find the closest combination by comparing with the theoretically obtained firing rate
                    idx = self.find_nearest(database[ma][:, -1], self.r0)
                    (j02, eta, delta, Ne, Ni) = database[ma][idx, 1:]
                    self.fileprm2 = '%s_%.2lf-%.2lf-%.2lf-%d' % (self.fp, j02, eta, delta, self.l)
                    try:
                        self.spikes_e = np.load("%sic_qif_spikes_e_%s-%d.npy" % (self.filepath, self.fileprm2, Ne))
                        self.spikes_i = np.load("%sic_qif_spikes_i_%s-%d.npy" % (self.filepath, self.fileprm2, Ni))
                        self.matrixE = np.load("%sic_qif_matrixE_%s-%d.npy" % (self.filepath, self.fileprm2, Ne))
                        self.matrixI = np.load("%sic_qif_matrixI_%s-%d.npy" % (self.filepath, self.fileprm2, Ni))
                        print "Successfully loaded all data matrices."
                    except IOError:
                        print "Files do not exist or cannot be read. This behavior wasn't expected ..."
                        exit(-1)
                    except ValueError:
                        print "Not appropriate format of initial conditions. Check the files for logical errors..."
                        exit(-1)
                else:  # Create new initial conditions from scratch (loading random conditions)
                    print "Loading random conditions. Generating new initial conditions. " \
                          "Run the program using the same conditions after the process finishes."
                    # We set excitatory and inhibitory neurons at the same initial conditions:
                    self.matrixE[:, 0] = -0.1 * np.random.randn(self.Ne)
                    self.matrixI[:, 0] = 1.0 * self.matrixE[:, 0]

    def save_ic(self, temps):
        """ Function to save initial conditions """
        np.save("%sic_qif_spikes_e_%s-%d" % (self.filepath, self.fileprm, self.Ne), self.spikes_e)
        np.save("%sic_qif_spikes_i_%s-%d" % (self.filepath, self.fileprm, self.Ni), self.spikes_i)
        self.matrixE[:, 1] = self.matrixE[:, 1] - (temps - self.dt)
        np.save("%sic_qif_matrixE_%s-%d.npy" % (self.filepath, self.fileprm, self.Ne), self.matrixE)
        self.matrixI[:, 1] = self.matrixI[:, 1] - (temps - self.dt)
        np.save("%sic_qif_matrixI_%s-%d.npy" % (self.filepath, self.fileprm, self.Ni), self.matrixI)
        # Introduce this combination into the database
        try:
            db = np.load("%sinitial_conditions.npy" % self.filepath)
        except IOError:
            print "Initial conditions database not found (%sinitial_conditions)" % self.filepath
            print "Creating database ..."
            db = False
        if db is False:
            np.save("%sinitial_conditions" % self.filepath,
                    np.array([self.l, self.j0, self.eta0, self.delta, self.Ne, self.Ni]))
        else:
            db.resize(np.array(np.shape(db)) + [1, 0], refcheck=False)
            db[-1] = np.array([self.l, self.j0, self.eta0, self.delta, self.Ne, self.Ni])
            np.save("%sinitial_conditions" % self.filepath, db)

    @staticmethod
    def find_nearest(array, value):
        """ Extract the argument of the closest value from array """
        idx = (np.abs(array - value)).argmin()
        return idx


class Connectivity:
    """ Mex-Hat type connectivity function creator and methods
        to extract properties from it: modes, frequencies, linear response.
    """

    def __init__(self, length=500, profile='mex-hat', amplitude=1.0, me=50, mi=5, j0=0.0,
                 refmode=None, refamp=None, fsmodes=None, data=None):
        """ In order to extract properties some parameters are needed: they can be
            call separately.
        """

        print "Creating connectivity matrix (depending on the size of the matrix (%d x %d) " \
              "this can take a lot of RAM)" % (length, length)
        # Number of points (sample) of the function. It should be the number of populations in the ring.
        self.l = length
        # Connectivity function and spatial coordinates
        self.cnt_e = np.zeros((length, length))
        self.cnt_i = np.zeros((length, length))
        self.cnt = np.zeros((length, length))
        [i_n, j_n] = 2.0 * np.pi * np.float64(np.meshgrid(xrange(length), xrange(length))) / length - np.pi
        ij = np.abs(i_n - j_n)
        del i_n, j_n  # Make sure you delete these matrices here !!!

        # Type of connectivity (profile=['mex-hat', 'General Fourier Series: fs'])
        if profile == 'mex-hat':
            if (refmode is not None) and (refamp is not None):  # A reference mode has been selected
                # Generate connectivity function here using reference amplitude
                (self.je, self.me, self.ji, self.mi) = self.searchmode(refmode, refamp, me, mi)
            else:
                # Generate connectivity function with parameters amplitude, me, mi, j0
                self.je = amplitude + j0
                self.ji = amplitude
            self.cnt_e = self.vonmises(self.je, me, 0.0, mi, coords=ij)
            self.cnt_i = self.vonmises(0.0, me, self.ji, mi, coords=ij)
            self.cnt = self.vonmises(self.je, me, self.ji, mi, coords=ij)
            # self.cnt = self.cnt_e + self.cnt_i
            # Compute eigenmodes
            self.modes = self.jmodesdct(self.vonmises(self.je, me, self.ji, mi, length))
        elif profile == 'fs':
            # Generate fourier series with modes fsmodes
            if fsmodes is None:
                fsmodes = [0, 6]  # Default values
            self.cnt = self.jcntvty(fsmodes, coords=ij)
            self.modes = fsmodes
        del ij

        # Compute frequencies for the ring model (if data is provided)
        if data is not None:
            self.freqs = self.frequencies(self.modes, data)
        else:
            self.freqs = None

    def searchmode(self, mode, amp, me, mi):
        """ Function that creates a Mex-Hat connectivity with a specific amplitude (amp) in a given mode (mode)
        :param mode: target mode
        :param amp: desired amplitude
        :param me: concentration parameter for the excitatory connectivity
        :param mi: concentration parameter for the inhibitory connectivity
        :return: connectivity parameters
        """
        tol = 1E-3
        max_it = 10000
        it = 0
        diff1 = 10
        step = amp * 1.0 - 1.0
        if step <= 0:
            step = 0.1
        # Test parameters:
        (ji, je, me, mi) = (1.0, 1.0, me, mi)

        while it < max_it:
            cnt = self.vonmises(je, me, ji, mi, self.l)
            jk = self.jmodesdct(cnt, mode + 10)
            diff = np.abs(jk[mode] - amp)
            if diff <= tol:
                break
            else:
                # print diff
                if diff1 - diff < 0:  # Bad direction
                    step *= -0.5

                je += step
                ji = je
                diff1 = diff * 1.0
            it += 1
        return je, me, ji, mi

    @staticmethod
    def frequencies(modes, data=None, eta=None, tau=None, delta=None, r0=None):
        """ Function that computes frequencies of decaying oscillations at the homogeneous state
        :param modes: array of modes, ordered from 0 to maximum wavenumber. If only zeroth mode is passed,
                      then it should be passed as an array. E.g. [1.0]. (J_0 = 1.0).
        :param data: data object containing all relevant info
        :param eta: if data is not passed, then we need the external current,
        :param tau: also time constant,
        :param delta: and also heterogeneity parameter.
        :param r0: We can pass the value of the stationary firing rate, or we can just let the function
                   compute the theoretical value.
        :return: an array of frequencies with same size as modes array.
        """
        # If a data object (containing all info is given)
        if data is not None:
            eta = data.eta0
            tau = data.tau
            delta = data.delta
        # If not:
        elif (eta is None) or (tau is None) or (delta is None):
            print 'Not enough data to compute frequencies'
            return None
        if r0 is None:  # We have to compute the firing rate at the stationary state
            r0 = Connectivity.rtheory(modes[0], eta, delta)
        r0u = r0 / tau
        return r0u * np.sqrt(1.0 - modes / (2 * np.pi ** 2 * tau * r0u))

    @staticmethod
    def gauss0_pdf(x, std):
        return stats.norm.pdf(x, 0, std)

    @staticmethod
    def mexhat0(a1, std1, a2, std2, length=500):
        x = np.linspace(-np.pi, np.pi, length)
        return x, a1 * Connectivity.gauss0_pdf(x, std1) + a2 * Connectivity.gauss0_pdf(x, std2)

    @staticmethod
    def vonmises(je, me, ji, mi, length=None, coords=None):
        if coords is None:
            if length is None:
                length = 500
            theta = (2.0 * np.pi / length) * np.arange(length)
        else:
            theta = 1.0 * coords
        return je / special.i0(me) * np.exp(me * np.cos(theta)) - ji / special.i0(mi) * np.exp(mi * np.cos(theta))

    @staticmethod
    def jcntvty(jk, coords=None):
        """ Fourier series generator.
        :param jk: array of eigenvalues. Odd ordered modes of Fourier series (only cos part)
        :param coords: matrix of coordinates
        :return: connectivity matrix J(|phi_i - phi_j|)
        """
        jphi = 0
        for i in xrange(len(jk)):
            if i == 0:
                jphi = jk[0]
            else:
                # Factor 2.0 is to be coherent with the computation of the mean-field S, where
                # we devide all connectivity profile by (2\pi) (which is the spatial normalization factor)
                jphi += 2.0 * jk[i] * np.cos(i * coords)
        return jphi

    @staticmethod
    def jmodes0(a1, std1, a2, std2, n=20):
        return 1.0 / (2.0 * np.pi) * (
            a1 * np.exp(-0.5 * (np.arange(n)) ** 2 * std1 ** 2) + a2 * np.exp(-0.5 * (np.arange(n)) ** 2 * std2 ** 2))

    @staticmethod
    def jmodesdct(jcnt, nmodes=20):
        """ Extract fourier first 20 odd modes from jcnt function.
        :param jcnt: periodic odd function.
        :param nmodes: number of modes to return
        :return: array of nmodes amplitudes corresponding to the FOurie modes
        """
        l = np.size(jcnt)
        jk = dct(jcnt, type=2, norm='ortho')
        for i in xrange(len(jk)):
            if i == 0:
                jk[i] *= np.sqrt(1.0 / (4.0 * l))
            else:
                jk[i] *= np.sqrt(1.0 / (2.0 * l))
        return jk[:nmodes:2]

    @staticmethod
    def rtheory(j0, eta0, delta):
        r0 = 1.0
        func = lambda tau: (np.pi ** 2 * tau ** 4 - j0 * tau ** 3 - eta0 * tau ** 2 - delta ** 2 / (4 * np.pi ** 2))
        sol = fsolve(func, r0)
        return sol

    def linresponse(self):
        # TODO Linear Response
        pass


class FiringRate:
    """ Class related to the measure of the firing rate of a neural network.
    """

    def __init__(self, data=None, swindow=1.0, sampling=0.01, points=None):

        if data is None:
            self.d = Data()
        else:
            self.d = data

        self.swindow = swindow  # Time Window in which we measure the firing rate
        self.wsteps = np.ceil(self.swindow / self.d.dt)  # Time window in time-steps
        self.wones = np.ones(self.wsteps)

        # Frequency of measuremnts
        if points is not None:
            pp = points
            if pp > self.d.nsteps:
                pp = self.d.nsteps
            self.sampling = self.d.nsteps / pp
            self.samplingtime = self.sampling * self.d.dt
        else:
            self.samplingtime = sampling
            self.sampling = int(self.samplingtime / self.d.dt)

        self.tpoints_r = np.linspace(0, self.d.tfinal, self.samplingtime)

        self.frspikes_e = 0 * np.ones(shape=(data.Ne, self.wsteps))  # Secondary spikes matrix (for measuring)
        self.frspikes_i = 0 * np.ones(shape=(data.Ni, self.wsteps))
        self.r = []  # Firing rate of the newtork(ring)
        self.frqif_e = []  # Firing rate of individual qif neurons
        self.frqif_i = []  # Firing rate of individual qif neurons

        # Auxiliary matrixes
        self.auxMat = np.zeros((self.d.l, self.d.N))
        self.auxMatE = np.zeros((self.d.l, self.d.Ne))
        self.auxMatI = np.zeros((self.d.l, self.d.Ni))
        for i in xrange(self.d.l):
            self.auxMat[i, i * self.d.dN:(i + 1) * self.d.dN] = 1.0
            self.auxMatE[i, i * self.d.dNe:(i + 1) * self.d.dNe] = 1.0
            self.auxMatI[i, i * self.d.dNi:(i + 1) * self.d.dNi] = 1.0

        # TODO FIRING RATE of individual neurons (distribution of FR)
        #
        # Auxiliary counters
        self.ravg = 0

        # Times of firing rate measures
        self.t0step = None
        self.tfstep = None
        self.temps = None

        self.tfrstep = -1
        self.tfr = []
        self.tempsfr = []
        self.tempsfr2 = []

    def firingrate(self, tstep):
        """ Computes the firing rate for a given matrix of spikes. Firing rate is computed
            every certain time (sampling). Therefore at some time steps the firing rate is not computed,
        :param tstep: time step of the simulation
        :return: firing rate vector (matrix)
        """
        if (tstep + 1) % self.sampling == 0 and (tstep * self.d.dt >= self.swindow):
            self.tfrstep += 1
            self.temps = tstep * self.d.dt
            # TODO (define the vectors np.ones outside)
            re = (1.0 / self.swindow) * (1.0 / self.d.dNe) * np.dot(self.auxMatE,
                                                                    np.dot(self.frspikes_e, self.wones))
            ri = (1.0 / self.swindow) * (1.0 / self.d.dNi) * np.dot(self.auxMatI,
                                                                    np.dot(self.frspikes_i, self.wones))

            self.r.append((re + ri) / 2.0)
            self.tempsfr.append(self.temps - self.swindow / 2.0)
            self.tempsfr2.append(self.temps)

            # Single neurons firing rate
            # self.rqif[self.rsum_count % self.rpointmax] = (1.0 / self.d.dt)  * self.S2.mean(axis=1)

            # self.r2[tstep % self.d.nsteps] = 1.0 * self.r[self.rsum_count]
            # Average of the voltages over a time window and over the populations
            # vsample = np.ma.masked_where(np.abs(v) >= vpeak, v).mean(axis=0).data
            # This 1/dN is wrong (TODO)
            # vavg[rsum_count] = (1.0/dN)*np.dot(auxMat, vsample)
            self.ravg += 1

    def singlefiringrate(self, tstep):
        """ Computes the firing rate of individual neurons.
        :return: Nothing, results are stored at frqif_e and frqif_i
        """
        if (tstep + 1) % self.sampling == 0 and (tstep * self.d.dt >= self.swindow):
            # Firing rate measure in a time window
            re = (1.0 / self.d.dt) * self.frspikes_e.mean(axis=1)
            ri = (1.0 / self.d.dt) * self.frspikes_i.mean(axis=1)
            self.frqif_e.append(re)
            self.frqif_i.append(ri)
