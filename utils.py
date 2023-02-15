import matplotlib.pyplot as plt
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

def plot_tavg(t, tavg, region_labels, nb_regions, normalize=True, scaling=1):
    # Normalize the time series to have nice plots
    if normalize:
        tavg /= (np.max(tavg, 0) - np.min(tavg, 0))
        tavg -= np.mean(tavg, 0)


    #Plot raw time series
    plt.figure(tight_layout=True, figsize=(15,20))
    plt.plot(t[:], scaling * tavg[:, 0, :, 0] + np.r_[:nb_regions], 'r')
    plt.title("Epileptors time series")
    plt.yticks(np.arange(len(region_labels)), region_labels, fontsize=10)
    plt.xticks(fontsize=22)
    plt.ylim([-1,len(region_labels)+0.5])
    plt.xlim([t[0],t[-1]])
    #Show them
    plt.show()

def plot_variables(t, tavg, region_labels, EZ):
    fig, axs = plt.subplots(2)
    plt.suptitle('Epileptor variables time series')
    for roi in EZ:
        axs[0].plot(t, tavg[:, 0, np.where(region_labels == roi)[0], 0], color='blue', label='x2-x1', lw=0.5)
        axs[0].plot(t, tavg[:, 1, np.where(region_labels == roi)[0], 0], color='green', label='z')
        axs[0].legend(['x2-x1', 'z'])
        axs[0].set_xlabel('Time')

    for roi in EZ:
        axs[1].plot(tavg[:, 0, np.where(region_labels == roi)[0], 0],
                    tavg[:, 1, np.where(region_labels == roi)[0], 0], color='blue', lw=0.2)
        axs[1].set_xlabel('x2-x1')
        axs[1].set_ylabel('z')
    plt.tight_layout()
    plt.show()

def circular_connectivity(con):
    import numpy as np
    import matplotlib.pyplot as plt
    from mne.viz import circular_layout
    from mne_connectivity.viz import plot_connectivity_circle

    def rgb2hex(r, g, b):
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    fname = '/Users/dollomab/OtherProjects/epi_visualisation/util/VepMrtrixLut.txt'
    label_colors = []
    with open(fname, 'r') as fd:
        for line in fd.readlines():
            #         print(line.strip().split())
            i, roi_name, r, g, b, _ = line.strip().split()
            #         print(r,g,b)
            label_colors.append(rgb2hex(int(r), int(g), int(b)))
    label_colors = label_colors[1:]

    # First, we reorder the labels based on their location in the left hemi
    label_names = [label for label in con.region_labels]
    label_centres = con.centres
    # label_colors = [label.color for label in labels]

    lh_labels = [name for name in label_names if name.startswith('Left')]

    # Get the y-location of the label
    # label_ypos = list()
    # for name in lh_labels:
    #     idx = label_names.index(name)
    #     ypos = label_centres[idx][1]#np.mean(labels[idx].pos[:, 1])
    #     label_ypos.append(ypos)

    # Reorder the labels based on their location
    # lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

    # For the right hemi
    rh_labels = [name for name in label_names if name.startswith('Right')]

    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)

    node_angles = circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=[0, len(label_names) / 2])

    # Plot the graph using node colors from the FreeSurfer parcellation. We only
    # show the 300 strongest connections.
    fig, ax = plt.subplots(figsize=(13, 13), facecolor='black',
                           subplot_kw=dict(polar=True))
    plot_connectivity_circle(con.weights, label_names, n_lines=300,
                             node_angles=node_angles, node_colors=label_colors,
                             title='Connectivity matrix', ax=ax)
    fig.tight_layout()
    # fig.savefig(f'{subj_proc_dir}/conn_lh_rh.pdf', facecolor=fig.get_facecolor())