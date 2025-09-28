import sys
import numpy as np
import matplotlib.pyplot as plt
from io_utils import cross_validation_vector, mice_names_info
from analyze_results_utils import get_global_weights

sys.path.append('../')

if __name__ == '__main__':
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    K = 4
    D, M, C = 1, 3, 2

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/figure_6/'
    path_main_folder = '../../glm-hmm_package/results/model_indiv_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/' + 'prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(alpha_val) + '/'

    mice_names = mice_names_info(path_data + 'mice_names.npz')
    global_directory = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/'
    global_weights = get_global_weights(global_directory, K)

    # plot LL for all animals
    fig = plt.figure(figsize=(4.8, 2))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    cols = ['#999999', '#984ea3', '#e41a1c', '#dede00']
    across_animals = []

    for animal in mice_names:
        path_analysis = path_main_folder + animal + '/'
        cv_arr = cross_validation_vector(path_analysis + "/diff_folds_fit.npz")
        idx = np.array([0, 3, 4, 5, 6])
        cv_arr_for_plotting = cv_arr[idx, :]
        mean_cvbt = np.mean(cv_arr_for_plotting, axis=1)
        across_animals.append(mean_cvbt - mean_cvbt[0])
        plt.plot([0, 1, 2, 3, 4], mean_cvbt - mean_cvbt[0], '-*', color="tan", lw=1.3, markersize=4, zorder=0)

    across_animals = np.array(across_animals)
    mean_cvbt = np.mean(np.array(across_animals), axis=0)
    plt.plot([0, 1, 2, 3, 4],
             mean_cvbt - mean_cvbt[0],
             '-*',
             color='brown',
             zorder=1,
             alpha=1,
             lw=1.5,
             markersize=5,
             label='indiv. fits mean')

    plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'], fontsize=10)
    plt.ylabel("$\Delta$ test LL (bits per trial)", fontsize=10, labelpad=0)
    plt.xlabel("Number of states", fontsize=10, labelpad=0)
    plt.ylim((-0.01, 0.22))
    plt.yticks(fontsize=10)

    # plot LL for the global fit
    global_dir = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/'
    path_analysis = global_dir + '/'
    cv_arr = cross_validation_vector(path_analysis + "/diff_folds_fit.npz")
    idx = np.array([0, 3, 4, 5, 6])
    cv_arr_for_plotting = cv_arr[idx, :]
    mean_cvbt = np.mean(cv_arr_for_plotting, axis=1)
    plt.plot([0, 1, 2, 3, 4], mean_cvbt - mean_cvbt[0], '-o', color="k", label='global fit', lw=1.3, markersize=4,
             zorder=0)
    leg = plt.legend(fontsize=9, loc='lower right', markerscale=0)
    fig.savefig(figure_dir + 'LL_all.pdf')



