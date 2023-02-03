from tvb.simulator.lab import *
from tvb.basic.readers import ZipReader
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import os.path as op
import numpy as np
import time
from utils import *

plot = False

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
    plt.yticks(np.r_[:len(con.region_labels)], con.region_labels);
    fig.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(25,10))
    reg_name = 'Left-Middle-cingulate-cortex-posterior-part'
    img = plt.bar(np.r_[0:con.weights.shape[0]], con.weights[np.where(con.region_labels == reg_name)[0][0]], color='black', alpha=0.3)
    plt.title(f'Connection weights to {reg_name}', fontsize=40)
    plt.xticks(np.r_[:len(con.region_labels)], con.region_labels, rotation = 90)
    plt.xlabel('Region#', fontsize=30);
    fig.tight_layout()

# Coupling
coupl = coupling.Difference(a=np.array([-3])) # -8

# Integrators
heunint = integrators.HeunDeterministic(dt=0.01)

# Monitors Q: How much should the period be ??
mons = [monitors.TemporalAverage(period=1)]

# Initial conditions
z0 = 2.95029597e+00
# init_cond = np.array([15.93,  1.228,   z0,  -0.03945, 1.914, -0.2088])
init_cond = np.array([-1.98742113e+00, -1.87492138e+01,  z0, -1.05214059e+00,
       -4.95543740e-20, -1.98742113e-01])
init_cond_reshaped = np.repeat(init_cond, nb_regions).reshape((1, len(init_cond), nb_regions, 1))
# init_cond_reshaped.shape


# Epileptor model
x0ez=-1.4
x0pz=-2.1
# x0pz2=-2.17
x0num=-2.2

#epileptors = Epileptor3D(Ks=np.array([-2]), r=np.array([0.0002]), tau = np.array([10]), tt = np.array([0.07]))
epileptors = models.Epileptor() # Q: What should the parameters be ???
epileptors.x0 = x0num*np.ones(nb_regions)
epileptors.r = np.ones(nb_regions)*0.0035#0.00035
epileptors.slope = np.ones(nb_regions)*(0)#0
epileptors.Iext = np.ones(nb_regions)*(3.1)
epileptors.Iext2 = np.ones(nb_regions)*(0.45)
epileptors.Ks = np.ones(nb_regions)*(1.0)
epileptors.Kf = np.ones(nb_regions)*(1.0)*0.0001# ?
epileptors.Kvf = np.ones(nb_regions)*(1.0)*0.01# ?

epileptors.state_variable_range['x1'] = np.array([-100, 100])
epileptors.state_variable_range['y1'] = np.array([-500, 500])
epileptors.state_variable_range['z'] = np.array([-50, 50])
epileptors.state_variable_range['x2'] = np.array([-100, 100])
epileptors.state_variable_range['y2'] = np.array([-200, 200])
epileptors.state_variable_range['g'] = np.array([-200, 200])

EZ = ['Left-Postcentral-gyrus', 'Left-Postcentral-sulcus']
PZ = ['Left-Supramarginal-posterior', 'Left-Central-sulcus-head-face', 'Left-Precentral-gyrus-upper-limb',
      'Left-Central-sulcus-upper-limb', 'Left-F2-caudal', 'Left-Precentral-gyrus-head-face']

z0_se = -1#0.9
for roi in EZ:
    epileptors.x0[np.where(con.region_labels == roi)[0]] = x0ez
    epileptors.slope[np.where(con.region_labels == roi)[0]] = -8 # -8
    init_cond_reshaped[0,2,np.where(con.region_labels == roi)[0],0] = z0_se
for roi in PZ:
    epileptors.x0[np.where(con.region_labels == roi)[0]] = x0pz
#     epileptors.slope[np.where(con.region_labels == roi)[0]] = 0
    init_cond_reshaped[0,2,np.where(con.region_labels == roi)[0],0] = z0_se


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
results= sim.run(simulation_length = 3000)#7000)
finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')


t, tavg = results[0]
plot_tavg(t[:], tavg[:,:,:,:], sim.connectivity.region_labels, nb_regions)



####### TODO part 2
import sys
import mne
sys.path.insert(2, '/Users/dollomab/MyProjects/Epinov_trial/vep_run_Trial/fit/')
import vep_prepare

# read gain
inv_gain_file = f'{subj_proc_dir}/elec/gain_inv-square.vep.txt'
invgain = np.loadtxt(inv_gain_file)

# read from GARDEL file
seeg_xyz = vep_prepare.read_seeg_xyz(subj_proc_dir)
seeg_xyz_names = [label for label, _ in seeg_xyz]

vhdrname = f"{subj_proc_dir}/ieeg/sub-314a1ab2525d_ses-01_task-seizure_acq-type2_run-01_ieeg.vhdr"
raw = mne.io.read_raw_brainvision(vhdrname, preload=True)
raw._data *= 1e6

bip_gain_inv_minus, bip_xyz, bip_name = vep_prepare.bipolarize_gain_minus(invgain, seeg_xyz, seeg_xyz_names)

bip = vep_prepare._bipify_raw(raw)
gain, bip = vep_prepare.gain_reorder(bip_gain_inv_minus, bip, bip_name)

import pickle
remove_cerebellar = False
sstscfname = f"{subj_proc_dir}/tvb/sstsc_Mapping.vep.pickle"
if not op.isfile(sstscfname):
    print('>> first time to generate the mapping files which may take longer time than usual')
    ch_names, prior_gain = vep_prepare.generate_srtss_maps(subj_proc_dir, vhdrname)
    maping_data_1 = {'ch_names': ch_names, 'prior_Mapping': prior_gain}
    with open(sstscfname, 'wb') as fd:
        pickle.dump(maping_data_1, fd)

with open(sstscfname, 'rb') as fdrb:
    maping_data = pickle.load(fdrb)

bad_ch = []
for ind_ch, ich in enumerate(maping_data['ch_names']):
    if ich not in bip.ch_names:
        bad_ch.append(ind_ch)
gain_prior = np.delete(maping_data['prior_Mapping'], bad_ch, axis=0)

roi = vep_prepare.read_vep_mrtrix_lut()
if remove_cerebellar:
    cereb_cortex = ['Left-Cerebellar-cortex','Right-Cerebellar-cortex']

    gain_prior.T[roi.index('Left-Cerebellar-cortex')] = gain_prior.T[-1]*0
    gain_prior.T[roi.index('Right-Cerebellar-cortex')] = gain_prior.T[-1]*0
    gain.T[roi.index('Left-Cerebellar-cortex')] = np.ones(np.shape(gain.T[-1]))*np.min(gain)
    gain.T[roi.index('Right-Cerebellar-cortex')] = np.ones(np.shape(gain.T[-1]))*np.min(gain)



tavg /= (np.max(tavg, 0) - np.min(tavg, 0))
tavg -= np.mean(tavg, 0)
srcSig = tavg[:, 0, :, 0]
# Map onto SEEG
seeg = np.dot(gain, srcSig.T)

show_ch = bip.ch_names
sfreq = 250.
nch = [show_ch.index(ichan) for ichan in show_ch]
nch_sourse = []
for ind_ch, ichan in enumerate(show_ch):
#     isource = roi[np.argmax(gain_prior[ind_ch])] # CORRECT
    isource = roi[np.argmax(gain[ind_ch])]     # TODO FIX THIS LATER !!!! REPLACE BY GAIN PRIOR
    nch_sourse.append(f'{isource}:{ichan}')
plt.figure(figsize=[40, 70])
scaleplt = 0.09
base_length = int(5 * sfreq)

start_idx = 0
end_idx = 7000

for ind, ich in enumerate(nch):
    plt.plot(t[start_idx:end_idx], scaleplt * (seeg[ich, start_idx:end_idx] - seeg[ich, 0]) + ind, 'blue', lw=1)
plt.xticks(fontsize=18)
plt.ylim([-1, len(nch) + 0.5])
plt.xlim([t[start_idx], t[end_idx - 1]])
plt.tight_layout()
# plt.title(f'{pid_bids}:ts_{basicfilename}', fontsize=16)
plt.yticks(np.arange(len(show_ch)), nch_sourse, fontsize=26);
plt.gcf().subplots_adjust(left=0.2)
plt.gcf().subplots_adjust(top=0.97)
# if save_fig:
#     print('>> Save', f'{save_img}/{pid_bids}_simulated_sensor_timeseries_run-0{run}.png')
#     plt.savefig(f'{save_img}/{pid_bids}_simulated_sensor_timeseries_run-0{run}.png')
plt.show()