import Gnuplot
from timeit import default_timer as timer

import numpy as np

from nflib import Data, Connectivity, FiringRate
# from scipy.signal import argrelextrema
# from scipy.fftpack import dct
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import sys
import getopt
import progressbar as pb

__author__ = 'jm'


def noise(length=100, disttype='g'):
    if disttype == 'g':
        return np.random.randn(length)


def main(argv, pmode=1, ampl=1.0):
    try:
        opts, args = getopt.getopt(argv, "hm:a:", ["mode=", "amp="])
    except getopt.GetoptError:
        print 'main.py -m <mode> -a <amplitude>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'connectivity.py -m <mode> -a <amplitude>'
            sys.exit()
        elif opt in ("-m", "--mode"):
            pmode = float(arg)
        elif opt in ("-a", "--amp"):
            ampl = float(arg)
    return pmode, ampl


selmode = 0
selamp = 1.0
if __name__ == "__main__":
    selmode, selamp = main(sys.argv[1:])

############################################################################
#############################
# 0) PREPARE FOR CALCULATIONS

# 0.1) Load data object:
d = Data(l=100, N=int(1E5), eta0=2.0, delta=1.0, tfinal=20.0, system='both')
# 0.2) Create connectivity matrix and extract eigenmodes
c = Connectivity(d.l, profile='mex-hat', amplitude=2.0, data=d)
# 0.3) Load initial conditions
d.load_ic(c.modes[0], system=d.system)
# 0.4) Load Firing rate class in case qif network is simulated
if d.system != 'nf':
    fr = FiringRate(data=d)

print "Modes: ", c.modes

# # 0.9) Perturbation parameters
pert_modes = [[int(selmode), float(selamp), None]]
pert = 0.0
phi = np.linspace(-np.pi, np.pi, d.l)
phip = np.linspace(-np.pi, np.pi, d.l + 1)

r0p = 0.0

for mode in pert_modes:
    if mode[0] == 0:
        pert = mode[1]
    else:
        pert += mode[1] * np.cos(mode[0] * phi)

pert_modes = np.array(pert_modes)

amplitude = 1.0
Input = 0.0 * pert
It = np.zeros((d.nsteps, d.l))
It[0, :] = Input

pert_duration = 0.5
pert_tsteps = int(pert_duration / d.dt)
rise = 0.2
# Registered times
pert_time = 1.0
tpert_0 = None
tpert_f = None
tmeasure = None
#

# Progress-bar configuration
widgets = ['Progress: ', pb.Percentage(), ' ',
           pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA(), ' ']

###################################################################################
###################################################################################
# 0) Load Bistability searching class:
# 1) Simulation (Integrate the system)
print('Simulating ...')
pbar = pb.ProgressBar(widgets=widgets, maxval=10 * (d.nsteps + 1)).start()
time1 = timer()
tstep = 0
temps = 0
tpert = 0.0
tpertstep = 0
END = False
PERT = False
a = None
b = None

# Time loop
# while END is False:
while temps < d.tfinal:
    # ######################## - PERTURBATION  - ##
    if PERT and not d.new_ic:
        Input = pert_time * selamp * np.exp((tpert - pert_time) / rise)
        tpert += d.dt
        tpertstep += 1
        tpert_f = tstep
    else:
        tpert = 0.0
        tpertstep = 0
        Input = 0.0

    It[tstep % d.nsteps, :] = Input

    # ######################## -  INTEGRATION  - ##
    # ######################## -      qif      - ##
    if d.system == 'qif' or d.system == 'both':
        # We compute the Mean-field vector s_j
        se = (1.0 / d.Ne) * np.dot(c.cnt_e, np.dot(fr.auxMatE, np.dot(d.spikes_e, d.a_tau[:, tstep % d.T_syn])))
        si = (1.0 / d.Ni) * np.dot(c.cnt_i, np.dot(fr.auxMatI, np.dot(d.spikes_i, d.a_tau[:, tstep % d.T_syn])))
        # Vectorized algorithm
        em_e = (d.matrixE[:, 1] <= temps)  # Excitatory
        em_i = (d.matrixI[:, 1] <= temps)  # Inhibitory

        if d.fp == 'noise':
            # TODO (define the vectors np.ones outside)
            noiseinput = np.sqrt(2.0 * d.dt / d.tau * d.delta) * noise(d.N)
            # Excitatory
            d.matrixE[em_e, 0] += (d.dt / d.tau) * (
                d.matrixE[em_e, 0] * d.matrixE[em_e, 0] + d.eta0 + d.tau *
                np.dot(np.array([(se + si)]).T, d.auxne).flatten()[em_e]) + noiseinput[em_e]
            # Inhibitory
            d.matrixI[em_i, 0] += (d.dt / d.tau) * (
                d.matrixI[em_e, 0] * d.matrixI[em_e, 0] + d.eta0 + d.tau *
                np.dot(np.array([(se + si)]).T, d.auxni).flatten()[em_i]) + noiseinput[em_e]
        else:
            # Excitatory
            d.matrixE[em_e, 0] += (d.dt / d.tau) * (
                d.matrixE[em_e, 0] * d.matrixE[em_e, 0] + d.etaE[em_e] + d.tau *
                np.dot(np.array([(se + si)]).T, d.auxne).flatten()[em_e])
            # Inhibitory
            d.matrixI[em_i, 0] += (d.dt / d.tau) * (
                d.matrixI[em_i, 0] * d.matrixI[em_i, 0] + d.etaI[em_i] + d.tau *
                np.dot(np.array([(se + si)]).T, d.auxni).flatten()[em_i])

        # Excitatory
        spm_e = (d.matrixE[:, 1] <= temps) & (d.matrixE[:, 0] >= d.vpeak)
        # d.matrixE[spm_e, 1] = temps + 2.0 * d.tau / d.matrixE[spm_e, 0]
        d.matrixE[spm_e, 1] = temps + d.refr_tau - (d.tau_peak - 1.0 / d.matrixE[spm_e, 0])
        d.matrixE[spm_e, 2] = 1
        d.matrixE[~spm_e, 2] = 0
        d.matrixE[spm_e, 0] = -d.matrixE[spm_e, 0]
        # #############################
        d.spikes_e_mod[:, (tstep + d.spiketime - 1) % d.spiketime] = 1 * d.matrixE[:, 2]  # We store the spikes
        d.spikes_e[:, tstep % d.T_syn] = 1 * d.spikes_e_mod[:, tstep % d.spiketime]

        # Inhibitory
        spm_i = (d.matrixI[:, 1] <= temps) & (d.matrixI[:, 0] >= d.vpeak)
        # d.matrixI[spm_i, 1] = temps + 2.0 * d.tau / d.matrixI[spm_i, 0]
        d.matrixE[spm_i, 1] = temps + d.refr_tau - (d.tau_peak - 1.0 / d.matrixE[spm_i, 0])
        d.matrixI[spm_i, 2] = 1
        d.matrixI[~spm_i, 2] = 0
        d.matrixI[spm_i, 0] = -d.matrixI[spm_i, 0]
        # #############################
        d.spikes_i_mod[:, (tstep + d.spiketime - 1) % d.spiketime] = 1 * d.matrixI[:, 2]  # We store the spikes
        d.spikes_i[:, tstep % d.T_syn] = 1 * d.spikes_i_mod[:, tstep % d.spiketime]

        # ######################## -- FIRING RATE -- ##
        fr.frspikes_e[:, tstep % fr.wsteps] = 1 * d.spikes_e[:, tstep % d.T_syn]
        fr.frspikes_i[:, tstep % fr.wsteps] = 1 * d.spikes_i[:, tstep % d.T_syn]
        fr.firingrate(tstep)

    # ######################## -  INTEGRATION  - ##
    # ######################## --   FR EQS.   -- ##
    if d.system == 'nf' or d.system == 'both':
        # We compute the Mean-field vector S ( 1.0/(2.0*np.pi)*dx = 1.0/l )
        d.sphi[tstep % d.nsteps] = (1.0 / 2.0 / np.pi) * (
            np.dot(c.cnt, d.rphi[(tstep + d.nsteps - 1) % d.nsteps]) * d.dx)

        # -- Integration -- #
        d.rphi[tstep % d.nsteps] = d.rphi[(tstep + d.nsteps - 1) % d.nsteps] + d.dt * (
            d.delta / np.pi + 2.0 * d.rphi[(tstep + d.nsteps - 1) % d.nsteps] * d.vphi[
                (tstep + d.nsteps - 1) % d.nsteps])
        d.vphi[tstep % d.nsteps] = d.vphi[(tstep + d.nsteps - 1) % d.nsteps] + d.dt * (
            d.vphi[(tstep + d.nsteps - 1) % d.nsteps] * d.vphi[
                (tstep + d.nsteps - 1) % d.nsteps] +
            d.eta0 + d.sphi[tstep % d.nsteps] -
            np.pi * np.pi * d.rphi[(tstep + d.nsteps - 1) % d.nsteps] * d.rphi[
                (tstep + d.nsteps - 1) % d.nsteps] +
            Input)

    # TODO Perturbation at certain time
    if int(pert_time / d.dt) <= tstep <= int((pert_time + pert_time) / d.dt):
        if tpert_0 is None:
            tpert_0 = tstep
        PERT = True
    else:
        PERT = False
    # We detect the maximum fr produced by the perturbation
    if tstep >= int((pert_time + pert_time) / d.dt) and tmeasure is None:
        if d.rphi[tstep % d.nsteps, d.l / 2] <= d.rphi[(tstep + d.nsteps - 1) % d.nsteps, d.l / 2]:
            tmeasure = tstep
            r0p = d.rphi[tstep % d.nsteps, d.l / 2] - r0p

    # Time evolution
    pbar.update(10 * tstep + 1)
    temps += d.dt
    tstep += 1

# Save initial conditions (TODO: put it into the Data class)
if d.new_ic:
    np.save("%sic_qif_spikes_e_%s-%d" % (d.filepath, d.fileprm, d.Ne), d.spikes_e)
    np.save("%sic_qif_spikes_i_%s-%d" % (d.filepath, d.fileprm, d.Ni), d.spikes_i)
    d.matrixE[:, 1] = d.matrixE[:, 1] - (temps - d.dt)
    np.save("%sic_qif_matrixE_%s-%d.npy" % (d.filepath, d.fileprm, d.Ne), d.matrixE)
    d.matrixI[:, 1] = d.matrixI[:, 1] - (temps - d.dt)
    np.save("%sic_qif_matrixI_%s-%d.npy" % (d.filepath, d.fileprm, d.Ni), d.matrixI)
    # Introduce this combination into the database
    try:
        db = np.load("%sinitial_conditions.npy" % d.filepath)
    except IOError:
        print "Initial conditions database not found (%sinitial_conditions)" % d.filepath
        print "Creating database ..."
        db = False
    if db is False:
        np.save("%sinitial_conditions" % d.filepath, np.array([d.l, d.j0, d.eta0, d.delta, d.Ne, d.Ni]))
    else:
        db.resize(np.array(np.shape(db)) + [1, 0], refcheck=False)
        db[-1] = np.array([d.l, d.j0, d.eta0, d.delta, d.Ne, d.Ni])
        np.save("%sinitial_conditions" % d.filepath, db)

# TODO: save firing rate time series (in the FiringRate class)
# TODO: perturbation function (where??)

# Stop the timer
time2 = timer()
Ttime = time2 - time1

gp = Gnuplot.Gnuplot(persist=1)
p1 = Gnuplot.PlotItems.Data(np.c_[d.tpoints * d.faketau, d.rphi[:, d.l / 2] / d.faketau], with_='lines')
p2 = Gnuplot.PlotItems.Data(np.c_[np.array(fr.tempsfr) * d.faketau, np.array(fr.r)[:, d.l / 2] / d.faketau],
                            with_='lines')
gp.plot(p1, p2)

print 'Total time: {}.'.format(Ttime)
