import Gnuplot
import getopt
import sys
from timeit import default_timer as timer

import numpy as np
import progressbar as pb

from nflib import Data, Connectivity, FiringRate
from tools import Perturbation, qifint, qifint_noise, noise, SaveResults, TheoreticalComputations

__author__ = 'jm'


def main(argv, pmode=1, ampl=1.0, system='nf', cnt='mex-hat', neurons=2E5, n=100, eta=4.0, delta=0.5, tfinal=20.0):
    try:
        opts, args = getopt.getopt(argv, "hm:a:s:c:N:n:e:d:t:",
                                   ["mode=", "amp=", "system=", "connec=", "neurons=", "lenght=", "extcurr=",
                                    "widthcurr=", "tfinal="])
    except getopt.GetoptError:
        print 'main.py [-m <mode> -a <amplitude> -s <system> -c <connectivity> -N <number-of-neurons> ' \
              '-n <lenght-of-ring-e <external-current> -d <widt-of-dist> -t <final-t>]'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'main.py [-m <mode> -a <amplitude> -s <system> -c <connectivity> -N <number-of-neurons> ' \
                  '-n <lenght-of-ring-e <external-current> -d <widt-of-dist> -t <final-t>]'
            sys.exit()
        elif opt in ("-m", "--mode"):
            pmode = int(arg)
        elif opt in ("-a", "--amp"):
            ampl = float(arg)
        elif opt in ("-s", "--system"):
            system = arg
        elif opt in ("-c", "--connec"):
            cnt = arg
        elif opt in ("-N", "--neurons"):
            neurons = int(float(arg))
        elif opt in ("-n", "--length"):
            n = int(arg)
        elif opt in ("-e", "--extcurr"):
            eta = float(arg)
        elif opt in ("-d", "--widthcurr"):
            delta = float(arg)
        elif opt in ("-t", "--tfinal"):
            tfinal = float(arg)

    return pmode, ampl, system, cnt, neurons, n, eta, delta, tfinal


selmode = 0
selamp = 1.0
selsystem = 'both'
selcnt = 'mex-hat'
selnumber = 2E5
sellength = 100
seleta = 4.0
seldelta = 0.5
seltfinal = 20.0
if __name__ == "__main__":
    selmode, selamp, selsystem, selcnt, selnumber, sellength, seleta, seldelta, seltfinal = main(sys.argv[1:], selmode,
                                                                                                 selamp,
                                                                                                 selsystem, selcnt, selnumber,
                                                                                                 sellength, seleta,
                                                                                                 seldelta, seltfinal)

###################################################################################
# 0) PREPARE FOR CALCULATIONS
# 0.1) Load data object:
d = Data(l=sellength, N=selnumber, eta0=seleta, delta=seldelta, tfinal=seltfinal, system=selsystem)

# 0.2) Create connectivity matrix and extract eigenmodes
c = Connectivity(d.l, profile=selcnt, amplitude=10.0, data=d)
print "Modes: ", c.modes

# 0.3) Load initial conditions
d.load_ic(c.modes[0], system=d.system)

# 0.4) Load Firing rate class in case qif network is simulated
if d.system != 'nf':
    fr = FiringRate(data=d, swindow=0.5, sampling=0.05)

# 0.5) Set perturbation configuration
p = Perturbation(data=d, dt=0.5, modes=[int(selmode)], amplitude=float(selamp), attack='exponential')

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
        s = (1.0 / d.N) * np.dot(c.cnt, np.dot(fr.auxMat, np.dot(d.spikes, d.a_tau[:, tstep % d.T_syn])))

        if d.fp == 'noise':
            noiseinput = np.sqrt(2.0 * d.dt / d.tau * d.delta) * noise(d.N)
            # Excitatory
            d.matrix = qifint_noise(d.matrix, d.matrix[:, 0], d.matrix[:, 1], d.eta, s + p.input,
                                    noiseinput[0:d.N], temps, d.N,
                                    d.dN, d.dt, d.tau, d.vpeak, d.refr_tau, d.tau_peak)
        else:
            # Excitatory
            d.matrix = qifint(d.matrix, d.matrix[:, 0], d.matrix[:, 1], d.eta, s + p.input, temps, d.N,
                              d.dN, d.dt, d.tau, d.vpeak, d.refr_tau, d.tau_peak)

        # Prepare spike matrices for Mean-Field computation and firing rate measure
        # Excitatory
        d.spikes_mod[:, (tstep + d.spiketime - 1) % d.spiketime] = 1 * d.matrix[:, 2]  # We store the spikes
        d.spikes[:, tstep % d.T_syn] = 1 * d.spikes_mod[:, tstep % d.spiketime]
        # Voltage measure:
        vma = (d.matrix[:, 1] <= temps)  # Neurons which are not in the refractory period
        fr.vavg0[vma] += d.matrix[vma, 0]
        fr.vavg += 1

        # ######################## -- FIRING RATE MEASURE -- ##
        fr.frspikes[:, tstep % fr.wsteps] = 1 * d.spikes[:, tstep % d.T_syn]
        fr.firingrate(tstep)
        # Distribution of Firing Rates
        if tstep > 0:
            fr.tspikes2 += d.matrix[:, 2]
            fr.ravg2 += 1  # Counter for the "instantaneous" distribution
            fr.ravg += 1  # Counter for the "total time average" distribution

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
    # Distribution of firing rates over all time
    fr.frqif0 = fr.tspikes / (fr.ravg * d.dt) / d.faketau
    fr.frqif = fr.frqif0 * 1.0

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
    s.create_dict(phi0=[d.l / 2, d.l / 4, d.l / 20], t0=int(d.total_time / 10) * np.array([2, 4, 6, 8]))
    s.results['perturbation']['It'] = p.it
    s.save()

# Preliminar plotting with gnuplot
gp = Gnuplot.Gnuplot(persist=1)
p1 = Gnuplot.PlotItems.Data(np.c_[d.tpoints * d.faketau, d.rphi[:, d.l / 2] / d.faketau], with_='lines')
if selsystem != 'nf':
    p2 = Gnuplot.PlotItems.Data(np.c_[np.array(fr.tempsfr) * d.faketau, np.array(fr.r)[:, d.l / 2] / d.faketau],
                                with_='lines')
else:
    p2 = Gnuplot.PlotItems.Data(np.c_[d.tpoints * d.faketau, p.it[:, d.l / 2] + d.r0 / d.faketau], with_='lines')
gp.plot(p1, p2)
