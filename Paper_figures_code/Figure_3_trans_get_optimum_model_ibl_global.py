import numpy as np
import json
import copy
import os
import matplotlib.pyplot as plt
from io_utils import model_data_glmhmm, colors_func, cross_validation_vector, make_cross_valid_for_figure
from analyze_results_utils import get_file_name_for_best_model_fold, permute_transition_matrix, find_corresponding_states_new

if __name__ == '__main__':
    num_inputs = 4
    K = 4
    cols = colors_func(K)
    state_labels = ['Engaged–L', 'Engaged–R', 'Biased–L', 'Biased–R']

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/'
    path_analysis = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/figure_3/'

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    cv_file = path_analysis + "/diff_folds_fit.npz"
    diff_folds_fit = cross_validation_vector(cv_file)

    with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
        optimal_initialize_dict = json.load(f)

    params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
    Params_model, lls = model_data_glmhmm(params_and_LL)
    perm = find_corresponding_states_new(Params_model)

    # Save parameters for initializing individual fits
    weight_vectors = Params_model[2][perm]
    log_transition_matrix = permute_transition_matrix(Params_model[1][0], perm)
    init_state_dist = Params_model[0][0][perm]

    # standardize the plotted GLM transition weights
    trans_weight_append_zero = np.vstack((Params_model[1][1], np.zeros((1, Params_model[1][1].shape[1]))))
    trans_weight_append_zero_standard = copy.deepcopy(trans_weight_append_zero)
    v1 = - np.mean(trans_weight_append_zero, axis=0)  # this is v1 instead of w1=0
    trans_weight_append_zero_standard[-1, :] = v1

    for i in range(K - 1):
        trans_weight_append_zero_standard[i, :] = v1 + trans_weight_append_zero[i, :]  # Here we have vi = v1 + wi
    weight_vectors_trans = trans_weight_append_zero_standard[perm]

    # plot transition weights
    M = weight_vectors.shape[2] - 1
    figure_covariates_names = ['$\Delta$ contrast', 'past choice', 'stimulus side', 'bias']
    figure_covariates_names_trans = ['filtered choice', 'filtered stim. side', 'filtered reward', 'basis_1', 'basis_2', 'basis_3']

    # Plot GLM-trans weights
    fig = plt.figure(figsize=(3.4, 3))
    plt.subplots_adjust(left=0.3, bottom=0.4, right=0.8, top=0.9)
    M_trans = weight_vectors_trans.shape[1] - 1
    for k in range(K):
        plt.plot(range(M_trans - 2), weight_vectors_trans[k][0:3],
                 marker='D', markersize=5, color=cols[k], lw=1,
                 label=state_labels[k])

    plt.xticks(list(range(len(figure_covariates_names_trans[:3]))),
               figure_covariates_names_trans[:3],
               rotation=45, fontsize=8)

    plt.yticks(fontsize=10)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.yticks([-.6, -.3, 0, .3, .6, 0.9], fontsize=10)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")
    plt.ylabel("Transition Weights", fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend(fontsize=6, loc='upper right')
    fig.savefig(figure_dir + 'Trans_weights_K=' + str(K) + '.pdf')
