from tvb.simulator.lab import *
from tvb.basic.readers import ZipReader
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import os.path as op
import numpy as np
import time
from utils import *

plot = False

global_coupling = -1.25#-1.4#-1.3#-1.5#-1#-4
m = 0
m_se = -8
z0 = 2.95029597e+00
z0_se = -0.8#-0.5# 0.7#-1#0.9
dt = 0.01

# x0 values
x0ez=-1.6
x0pz=-2.1
x0num=-2.2

# Data
subj_proc_dir = '/Users/dollomab/MyProjects/Epinov_trial/patients/sub-314a1ab2525d/'
data_dir = f'{subj_proc_dir}tvb/'

# Connectivity
SC = get_connectivity(subj_proc_dir)

reader = ZipReader(os.path.join(subj_proc_dir, 'tvb', 'connectivity.vep.zip'))
if reader.has_file_like("centres"):
    centres = reader.read_array_from_file("centres", use_cols=(1, 2, 3))
    roi = reader.read_optional_array_from_file("centres", dtype=str, use_cols=(0))

lengths = np.loadtxt(f'{subj_proc_dir}/dwi_new/lengths.txt')
con = connectivity.Connectivity(weights=SC, tract_lengths=lengths, region_labels=np.asarray(roi), centres=centres)
con.tract_lengths = np.zeros((con.tract_lengths.shape))  # no time-delays
con.configure()
nb_regions = len(con.region_labels)
print("There are ", nb_regions, " regions in the connectivity matrix.")

if plot:
    fig = plt.figure(figsize=(20,20))
    plt.imshow(con.weights,norm=plc.LogNorm(vmin=1e-6, vmax=con.weights.max()))
    plt.title(f'Normalized SC (log scale)',fontsize=12, fontweight='bold')
    plt.xticks(np.r_[:len(con.region_labels)], con.region_labels, rotation = 90)
    plt.yticks(np.r_[:len(con.region_labels)], con.region_labels)
    fig.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(25,10))
    reg_name = 'Left-Middle-cingulate-cortex-posterior-part'
    img = plt.bar(np.r_[0:con.weights.shape[0]], con.weights[np.where(con.region_labels == reg_name)[0][0]], color='black', alpha=0.3)
    plt.title(f'Connection weights to {reg_name}', fontsize=40)
    plt.xticks(np.r_[:len(con.region_labels)], con.region_labels, rotation = 90)
    plt.xlabel('Region#', fontsize=30)
    fig.tight_layout()

# Coupling
coupl = coupling.Difference(a=np.array([global_coupling])) # -3 # -8

# Integrators
heunint = integrators.HeunDeterministic(dt=dt)

# Monitors Q: How much should the period be ??
mons = [monitors.Raw(),  monitors.TemporalAverage(period=0.5)]

# Initial conditions
# init_cond = np.array([15.93,  1.228,   z0,  -0.03945, 1.914, -0.2088])
init_cond = np.array([-1.98742113e+00, -1.87492138e+01,  z0, -1.05214059e+00,
       -4.95543740e-20, -1.98742113e-01])
init_cond_reshaped = np.repeat(init_cond, nb_regions).reshape((1, len(init_cond), nb_regions, 1))
# init_cond_reshaped.shape

# Epileptor model
#epileptors = Epileptor3D(Ks=np.array([-2]), r=np.array([0.0002]), tau = np.array([10]), tt = np.array([0.07]))
epileptors = models.Epileptor() # Q: What should the parameters be ???
epileptors.variables_of_interest = ['x2 - x1', 'z', 'x1', 'y1', 'x2']
epileptors.x0 = x0num*np.ones(nb_regions)
epileptors.r = np.ones(nb_regions)*0.00015#0.00035
epileptors.slope = np.ones(nb_regions)*(m)#0
epileptors.Iext = np.ones(nb_regions)*(3.1)
epileptors.Iext2 = np.ones(nb_regions)*(0.45)
epileptors.Ks = np.ones(nb_regions)*(1.0)
epileptors.Kf = np.ones(nb_regions)*(1.0)*0.0001# ?
epileptors.Kvf = np.ones(nb_regions)*(1.0)*0.01# ?

epileptors.state_variable_range['x1'] = np.array([-1000, 1000])
epileptors.state_variable_range['y1'] = np.array([-5000, 5000])
epileptors.state_variable_range['z'] = np.array([-500, 500])
epileptors.state_variable_range['x2'] = np.array([-1000, 1000])
epileptors.state_variable_range['y2'] = np.array([-2000, 2000])
epileptors.state_variable_range['g'] = np.array([-2000, 2000])

EZ = ['Left-Postcentral-gyrus', 'Left-Postcentral-sulcus']
PZ = ['Left-Supramarginal-posterior', 'Left-Central-sulcus-head-face', 'Left-Precentral-gyrus-upper-limb',
      'Left-Central-sulcus-upper-limb', 'Left-F2-caudal', 'Left-Precentral-gyrus-head-face']


for roi in EZ:
    epileptors.x0[np.where(con.region_labels == roi)[0]] = x0ez
    epileptors.slope[np.where(con.region_labels == roi)[0]] = m_se # -8
    init_cond_reshaped[0,2,np.where(con.region_labels == roi)[0],0] = z0_se
for roi in PZ:
    epileptors.x0[np.where(con.region_labels == roi)[0]] = x0pz
#     epileptors.slope[np.where(con.region_labels == roi)[0]] = 0
#     init_cond_reshaped[0,2,np.where(con.region_labels == roi)[0],0] = z0_se


# Simulator
sim = simulator.Simulator(model=epileptors,
                      connectivity=con,
                      coupling=coupl,
                      integrator=heunint,
                      monitors=mons,
                      initial_conditions = init_cond_reshaped)#,
                      #surface=surf)
sim.configure()

# Run
start = time.perf_counter()
results= sim.run(simulation_length = 6000)#7000)
finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')


(tr, raw), (t, tavg) = results
plot_tavg(t[:], tavg[:,:,:,:], sim.connectivity.region_labels, nb_regions, normalize=True, scaling=2)
plot_tavg(tr[:], raw[:,:,:,:], sim.connectivity.region_labels, nb_regions, normalize=True, scaling=2)


save_data = True
if save_data:
    np.savez(f'/Users/dollomab/MyProjects/Epinov_trial/simulate_data/simulate_status_epilepticus/simulations/simulation_z0_{z0_se}_m_{m_se}_x0_{x0ez}_x0pz_{x0pz}gc_{global_coupling}_almostthere14_PERFECT',
             t=t, tavg=tavg, z0_se=z0_se, z0=z0, x0ez=x0ez, x0pz=x0pz, x0num=x0num, m_se=m_se,
             m=m, global_coupling=global_coupling, dt=dt)

plot_variables(t, tavg, con.region_labels, EZ)
plot_variables(tr, raw, con.region_labels, EZ)

