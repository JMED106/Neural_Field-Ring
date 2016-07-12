import Gnuplot
import getopt
import sys
from timeit import default_timer as timer

import numpy as np
import progressbar as pb

from nflib import Data, Connectivity, FiringRate
from tools import Perturbation, qifint, noise, SaveResults, TheoreticalComputations

__author__ = 'jm'


def main(argv, pmode=1, ampl=1.0, system='nf'):
    try:
        opts, args = getopt.getopt(argv, "hm:a:s:", ["mode=", "amp=", "system="])
    except getopt.GetoptError:
        print 'main.py -m <mode> -a <amplitude> -s <system>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'main.py -m <mode> -a <amplitude> -s <system>'
            sys.exit()
        elif opt in ("-m", "--mode"):
            pmode = int(arg)
        elif opt in ("-a", "--amp"):
            ampl = float(arg)
        elif opt in ("-s", "--system"):
            system = arg
    return pmode, ampl, system


selmode = 0
selamp = 1.0
selsystem = 'nf'
if __name__ == "__main__":
    selmode, selamp, selsystem = main(sys.argv[1:], selmode, selamp, selsystem)

###################################################################################
# 0) PREPARE FOR CALCULATIONS
# 0.1) Load data object:
d = Data(l=100, N=int(1E3), eta0=4.0, delta=0.5, tfinal=20.0, system=selsystem)
# 0.2) Create connectivity matrix and extract eigenmodes
c = Connectivity(d.l, profile='mex-hat', amplitude=10.0, data=d, refmode=4, refamp=8 / np.sqrt(0.5))
print "Modes: ", c.modes
# 0.3) Load initial conditions
d.load_ic(c.modes[0], system=d.system)
# 0.4) Load Firing rate class in case qif network is simulated
if d.system != 'nf':
    fr = FiringRate(data=d, swindow=0.5, sampling=0.05)
# 0.5) Set perturbation configuration
p = Perturbation(data=d, modes=[int(selmode)], amplitude=float(selamp), release='exponential')
# 0.6) Define saving paths:
s = SaveResults(data=d, cnt=c, pert=p, system=d.system)
# 0.7) Other theoretical tools:
th = TheoreticalComputations(d, c, p)

# Progress-bar configuration
widgets = ['Progress: ', pb.Percentage(), ' ',
           pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA(), ' ']

###################################################################################
# 1) Simulation (Integrate the system)
print('Simulating ...')
pbar = pb.ProgressBar(widgets=widgets, maxval=10 * (d.nsteps + 1)).start()
time1 = timer()
tstep = 0
temps = 0

# Time loop
while temps < d.tfinal:
    # ######################## - PERTURBATION  - ##
    if p.pbool and not d.new_ic:
        if temps >= p.t0:
            p.timeevo(temps)
    p.it[tstep % d.nsteps, :] = p.input

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
            noiseinput = np.sqrt(2.0 * d.dt / d.tau * d.delta) * noise(d.N)
            # Excitatory
            d.matrixE[em_e, 0] += (d.dt / d.tau) * (
                d.matrixE[em_e, 0] * d.matrixE[em_e, 0] + d.eta0 + d.tau *
                np.dot(np.array([(se + si + p.input)]).T, d.auxne).flatten()[em_e]) + noiseinput[em_e]
            # Inhibitory
            d.matrixI[em_i, 0] += (d.dt / d.tau) * (
                d.matrixI[em_e, 0] * d.matrixI[em_e, 0] + d.eta0 + d.tau *
                np.dot(np.array([(se + si + p.input)]).T, d.auxni).flatten()[em_i]) + noiseinput[em_e]
        else:
            # Excitatory
            d.matrixE = qifint(d.matrixE, d.matrixE[:, 0], d.matrixE[:, 1], d.etaE, se + si + p.input, temps, d.Ne,
                               d.dNe, d.dt, d.tau, d.vpeak, d.refr_tau, d.tau_peak)
            # Inhibitory
            d.matrixI = qifint(d.matrixI, d.matrixI[:, 0], d.matrixI[:, 1], d.etaI, se + si + p.input, temps, d.Ni,
                               d.dNi, d.dt, d.tau, d.vpeak, d.refr_tau, d.tau_peak)

        # Prepare spike matrices for Mean-Field computation and firing rate measure
        # Excitatory
        d.spikes_e_mod[:, (tstep + d.spiketime - 1) % d.spiketime] = 1 * d.matrixE[:, 2]  # We store the spikes
        d.spikes_e[:, tstep % d.T_syn] = 1 * d.spikes_e_mod[:, tstep % d.spiketime]

        # Inhibitory
        d.spikes_i_mod[:, (tstep + d.spiketime - 1) % d.spiketime] = 1 * d.matrixI[:, 2]  # We store the spikes
        d.spikes_i[:, tstep % d.T_syn] = 1 * d.spikes_i_mod[:, tstep % d.spiketime]

        # ######################## -- FIRING RATE MEASURE -- ##
        fr.frspikes_e[:, tstep % fr.wsteps] = 1 * d.spikes_e[:, tstep % d.T_syn]
        fr.frspikes_i[:, tstep % fr.wsteps] = 1 * d.spikes_i[:, tstep % d.T_syn]
        fr.firingrate(tstep)
        # Distribution of Firing Rates
        if tstep > 0:
            fr.tspikes_e += d.matrixE[:, 2]
            fr.tspikes_i += d.matrixI[:, 2]
            fr.ravg += 1

    # ######################## -  INTEGRATION  - ##
    # ######################## --   FR EQS.   -- ##
    if d.system == 'nf' or d.system == 'both':
        # We compute the Mean-field vector S ( 1.0/(2.0*np.pi)*dx = 1.0/l )
        d.sphi[tstep % d.nsteps] = (1.0 / d.l) * (
            np.dot(c.cnt, d.rphi[(tstep + d.nsteps - 1) % d.nsteps]))

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
            p.input)

    # Perturbation at certain time
    if int(p.t0 / d.dt) == tstep:
        p.pbool = True

    # Time evolution
    pbar.update(10 * tstep + 1)
    temps += d.dt
    tstep += 1

# Finish pbar
pbar.finish()
# Stop the timer
print 'Total time: {}.'.format(timer() - time1)

# Compute distribution of firing rates of neurons
tstep -= 1
temps -= d.dt
th.thdist = th.theor_distrb(d.sphi[tstep % d.nsteps])

# Register data to a dictionary
if 'qif' in d.systems:
    fr.frqif_e = fr.frspikes_e / (fr.ravg * d.dt) / d.faketau
    fr.frqif_i = fr.frspikes_i / (fr.ravg * d.dt) / d.faketau
    fr.frqif = np.concatenate((fr.frqif_e, fr.frqif_i))

    if 'nf' in d.systems:
        d.register_ts(fr, th)
    else:
        d.register_ts(fr)
else:
    d.register_ts(th=th)

# Save initial conditions
if d.new_ic:
    d.save_ic(temps)
else:  # Save results
    s.create_dict(phi0=[d.l / 2, d.l / 4, d.l / 20])
    s.save()

gp = Gnuplot.Gnuplot(persist=1)
p1 = Gnuplot.PlotItems.Data(np.c_[d.tpoints * d.faketau, d.rphi[:, d.l / 2] / d.faketau], with_='lines')
if selsystem != 'nf':
    p2 = Gnuplot.PlotItems.Data(np.c_[np.array(fr.tempsfr) * d.faketau, np.array(fr.r)[:, d.l / 2] / d.faketau],
                                with_='lines')
else:
    p2 = Gnuplot.PlotItems.Data(np.c_[d.tpoints * d.faketau, p.it[:, d.l / 2] + d.r0 / d.faketau], with_='lines')
gp.plot(p1, p2)
