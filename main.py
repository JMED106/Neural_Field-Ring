import Gnuplot
import getopt
import sys
from timeit import default_timer as timer

import numpy as np
import progressbar as pb

from nflib import Data, Connectivity, FiringRate
from tools import Perturbation, qifint, qifint_noise, noise, SaveResults, TheoreticalComputations, DictToObj

__author__ = 'jm'


def main(argv, options):
    try:
        optis, args = getopt.getopt(argv, "hm:a:s:c:N:n:e:d:t:D:",
                                    ["mode=", "amp=", "system=", "connec=", "neurons=", "lenght=", "extcurr=",
                                     "delta=", "tfinal=", "Distr="])
    except getopt.GetoptError:
        print 'main.py [-m <mode> -a <amplitude> -s <system> -c <connectivity> -N <number-of-neurons> ' \
              '-n <lenght-of-ring-e <external-current> -d <widt-of-dist> -t <final-t> -D <type-of-distr>]'
        sys.exit(2)

    for opt, arg in optis:
        if len(opt) > 2:
            opt = opt[1:3]
        opt = opt[1]
        # Check type and cast
        if isinstance(options[opt], int):
            options[opt] = int(float(arg))
        elif isinstance(options[opt], float):
            options[opt] = float(arg)
        else:
            options[opt] = arg

    return options


opts = {"m": 0, "a": 1.0, "s": 'both', "c": 'mex-hat',
        "N": 2E5, "n": 100, "e": 4.0, "d": 0.5, "t": 20,
        "D": 'lorentz'}
if __name__ == '__main__':
    opts = main(sys.argv[1:], opts)
print opts
opts = DictToObj(opts)
store_ic = False

###################################################################################
# 0) PREPARE FOR CALCULATIONS
# 0.1) Load data object:
d = Data(l=opts.n, N=opts.N, eta0=opts.e, delta=opts.d, tfinal=opts.t, system=opts.s, fp=opts.D)

# 0.2) Create connectivity matrix and extract eigenmodes
c = Connectivity(d.l, profile=opts.c, amplitude=10.0, data=d)
print "Modes: ", c.modes

# 0.3) Load initial conditions
d.load_ic(c.modes[0], system=d.system)
# Override initial conditions generator:
if store_ic:
    d.new_ic = True

# 0.4) Load Firing rate class in case qif network is simulated
if d.system != 'nf':
    fr = FiringRate(data=d, swindow=0.5, sampling=0.05)

# 0.5) Set perturbation configuration
p = Perturbation(data=d, dt=0.5, modes=[int(opts.m)], amplitude=float(opts.a), attack='exponential')

# 0.6) Define saving paths:
sr = SaveResults(data=d, cnt=c, pert=p, system=d.system)

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
    sr.create_dict(phi0=[d.l / 2, d.l / 4, d.l / 20], t0=int(d.total_time / 10) * np.array([2, 4, 6, 8]))
    sr.results['perturbation']['It'] = p.it
    sr.save()

# Preliminar plotting with gnuplot
gp = Gnuplot.Gnuplot(persist=1)
p1 = Gnuplot.PlotItems.Data(np.c_[d.tpoints * d.faketau, d.rphi[:, d.l / 2] / d.faketau], with_='lines')
if opts.s != 'nf':
    p2 = Gnuplot.PlotItems.Data(np.c_[np.array(fr.tempsfr) * d.faketau, np.array(fr.r)[:, d.l / 2] / d.faketau],
                                with_='lines')
else:
    p2 = Gnuplot.PlotItems.Data(np.c_[d.tpoints * d.faketau, p.it[:, d.l / 2] + d.r0 / d.faketau], with_='lines')
gp.plot(p1, p2)
