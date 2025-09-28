import numpy as np
import matplotlib.pyplot as plt
from io_utils import get_mouse_info_1

if __name__ == '__main__':
    Len = 50
    stimulus_side = []

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/'
    all_data_obs_mat, all_data_trans_mat, all_data_y, all_data_session = get_mouse_info_1(path_data + 'unnorm_all_mice.npz')
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/Supp_figures/figure_2/'

    for i in range(all_data_obs_mat.shape[0]):
        if all_data_obs_mat[i, 0] < 0:
            stimulus_side.append(0)
        else:
            stimulus_side.append(1)

    fig = plt.figure(figsize=(9, 3.4))
    plt.subplots_adjust(bottom=0.2)
    x_values = [0, 10, 20, 30, 40, 50]
    container = np.load(path_data + 'rewarded_all_mice.npz', allow_pickle=True)
    data = [container[key] for key in container]
    rewarded_all_mice = data[0]

    plt.subplot(4, 1, 1)
    plt.scatter(range(0, Len), all_data_y[0:Len, 0], s=60, label="choice", facecolors='none', edgecolors='black')
    plt.scatter(range(0, Len), stimulus_side[0: Len],  s=20, label="stimulus", facecolors='red')
    plt.xticks(x_values, ["", "", "", "", "", ""], fontsize=10)
    plt.yticks([0, 1], ["left", "right"], fontsize=9)
    plt.ylim((-.2, 1.3))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend(fontsize=8, loc="lower right")

    all_data_trans_mat[0, :] = 0
    plt.subplot(4, 1, 2)
    plt.plot(all_data_trans_mat[0:Len, 1], lw=2)

    plt.yticks([-1, 0, 1], ["-1", "0", "1"], fontsize=10)
    plt.ylabel("filtered \n stim. side", fontsize=9)
    plt.xticks(x_values, ["", "", "", "", "", ""], fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.subplot(4, 1, 3)
    plt.plot(all_data_trans_mat[0:Len, 0], lw=2)
    plt.xticks(x_values, ["", "", "", "", "", ""], fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.yticks([-1, 0, 1], ["-1", "0", "1"], fontsize=10)
    plt.ylabel("filtered \n choice", fontsize=9)

    plt.subplot(4, 1, 4)
    plt.plot(all_data_trans_mat[0:Len, 2], lw=2)
    plt.axhline(y=0, color='black', linestyle='--', lw=.8)
    plt.ylabel("filtered \n reward", fontsize=9)
    plt.xlabel("trial", fontsize=9)
    plt.yticks([-1, 0, 1], ["-1", "0", "1"], fontsize=10)
    plt.xticks(x_values, fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    fig.savefig(figure_dir + 'regressors' + '.pdf')



