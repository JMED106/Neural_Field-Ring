import numpy as np

from nflib import Data

__author__ = 'Jose M. Esnaola Acebes'

""" Library containing different tools to use alongside the main simulation:

    + Perturbation class
"""


class Perturbation:
    """ Tool to handle perturbations: time, duration, shape (attack, decay, sustain, release (ADSR), etc. """

    def __init__(self, data=None, t0=1.0, dt=0.5, ptype='pulse', modes=None,
                 amplitude=1.0, attack='exponential', release='instantaneous'):
        if data is None:
            self.d = Data()
        else:
            self.d = data

        if modes is None:  # Default mode perturbation is first mode
            modes = [1]

        # Input at t0
        self.input = np.ones(self.d.l) * 0.0
        # Input time series
        self.it = np.zeros((self.d.nsteps, self.d.l))
        # Input ON/OFF
        self.pbool = False

        # Time parameters
        self.t0 = t0
        self.dt = dt
        self.tf = t0 + dt
        # Rise variables (attack) and parameters
        self.attack = attack
        self.taur = 0.2
        self.trmod = 0.0
        # Decay (release) and parameters
        self.release = release
        self.taud = 0.2
        self.tdmod = 1.0
        self.mintd = 1E-2

        # Amplitude parameters
        self.ptype = ptype
        self.amp = amplitude
        # Spatial modulation (wavelengths)
        self.smod = self.sptprofile(modes, self.amp)
        self.phi = np.linspace(-np.pi, np.pi, self.d.l)

    def sptprofile(self, modes, amp=1E-2):
        """ Gives the spatial profile of the perturbation: different wavelength and combinations
            of them can be produced.
        """
        sprofile = 0.0
        if np.isscalar(modes):
            print "Warning! 'modes' should be an iterable."
            modes = [modes]
        for m in modes:
            sprofile += amp * np.cos(m * self.phi)
        return sprofile

    def timeevo(self, temps):
        """ Time evolution of the perturbation """
        # Single pulse
        if self.ptype == 'pulse':
            # Release time, after tf
            if temps >= self.tf:
                if self.release == 'exponential' and self.tdmod > self.mintd:
                    self.tdmod += (self.d.dt / self.taud) * self.tdmod
                    self.input = self.tdmod * self.smod
                elif self.release == 'instantaneous':
                    self.input = 0.0
                    self.pbool = False
            else:  # During the pulse (from t0 until tf)
                if self.attack == 'exponential' and self.trmod < 1.0:
                    self.trmod += (self.d.dt / self.taur) * self.trmod
                    self.input = self.trmod * self.smod
                elif self.attack == 'instantaneous':
                    if temps == self.t0:
                        self.input = self.amp
        elif self.ptype == 'oscillatory':
            pass
