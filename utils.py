import numpy as np
import zipfile
import os.path as op

def get_connectivity(subj_proc_dir):
    try:
        if op.exists(f'{subj_proc_dir}/dwi_new/weights.txt'):
            SC = np.loadtxt(f'{subj_proc_dir}/dwi_new/weights.txt')
        else:
            print("taking connectivity from tvb")
            with zipfile.ZipFile(
                    f'{subj_proc_dir}/tvb/connectivity.vep.zip') as sczip:
                with sczip.open('weights.txt') as weights:
                    SC = np.loadtxt(weights)

    except FileNotFoundError as err:
        print(f'{err}: Structural connectivity not found for {subj_proc_dir}')
    SC[np.diag_indices(SC.shape[0])] = 0
    SC = SC / SC.max()
    return SC

def plot_tavg(t, tavg, region_labels, nb_regions):
    # Normalize the time series to have nice plots
    tavg /= (np.max(tavg, 0) - np.min(tavg, 0))
    tavg -= np.mean(tavg, 0)


    #Plot raw time series
    plt.figure(tight_layout=True, figsize=(15,20))
    plt.plot(t[:], tavg[:, 0, :, 0] + np.r_[:nb_regions], 'r')
    plt.title("Epileptors time series")
    plt.yticks(np.arange(len(region_labels)), region_labels, fontsize=10)
    plt.xticks(fontsize=22)
    plt.ylim([-1,len(region_labels)+0.5])
    plt.xlim([t[0],t[-1]])
    #Show them
    plt.show()
