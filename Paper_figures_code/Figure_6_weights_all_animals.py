import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
from io_utils import cross_validation_vector, model_data_glmhmm, mice_names_info, colors_func
from analyze_results_utils import  find_corresponding_states_new, get_global_weights, get_file_name_for_best_model_fold

sys.path.append('../')


if __name__ == '__main__':
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    K = 4
    D, M, C = 1, 4, 2
    M_trans = 6

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/figure_6/'
    path_main_folder = '../../glm-hmm_package/results/model_indiv_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/' + 'prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(alpha_val) + '/'
    mice_names = mice_names_info(path_data + 'mice_names.npz')
    global_directory = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(num_inputs) + '/'

    Params_model_glob = get_global_weights(global_directory, K)
    global_weights = -Params_model_glob[2]
    label = find_corresponding_states_new(Params_model_glob)

    fig = plt.figure(figsize=(8, 5))
    plt.subplots_adjust(left=0.12, bottom=0.2, right=0.95, top=0.95, wspace=0.2, hspace=0.55)
    figure_covariates_names = ['$\Delta$ contrast', 'past choice', 'stimulus side', 'bias']
    figure_covariates_names_trans = ['filtered choice', 'filtered stim. side', 'filtered reward', 'basis_1', 'basis_2', 'basis_3']

    # plot weights
    cols = colors_func(K)
    for k in range(K):
        plt.subplot(2, 4, k+1)
        for animal in mice_names:
            path_analysis = path_main_folder + animal + '/'

            cv_file = path_analysis + "/diff_folds_fit.npz"
            diff_folds_fit = cross_validation_vector(cv_file)

            with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
                optimal_initialize_dict = json.load(f)

            # Get the file name corresponding to the best initialization for given K value
            params_and_LL = get_file_name_for_best_model_fold(
                diff_folds_fit, K, path_analysis, optimal_initialize_dict)
            Params_model, lls = model_data_glmhmm(params_and_LL)

            transition_matrix = np.exp(Params_model[1][0])
            weight_vectors = -Params_model[2]

            trans_weight_append_zero = np.vstack((Params_model[1][1], np.zeros(
                (1, Params_model[1][1].shape[1]))))

            if animal == "churchlandlab/CSHL_014/_ibl_subjectTrials.table.61f6982a-40fb-44a6-8f2f-170235951e26.pqt":
                plt.plot(range(M),
                         global_weights[label[k]][0],
                         '-o',
                         color='k',
                         lw=1.3,
                         alpha=1,
                         markersize=3,
                         zorder=1,
                         label='global fit')
                if k == 0:
                    plt.legend(fontsize=8, loc='best')
            else:
                plt.plot(range(M),
                         weight_vectors[label[k]][0],
                         '-o',
                         color=cols[k],
                         lw=1,
                         alpha=0.7,
                         markersize=3,
                         zorder=0)
                plt.title('State ' + str(k+1), color=cols[k], fontsize=9)

        if k == 0:
            plt.yticks([-5, 0, 5, 10, 15, 20, 25], fontsize=9)
            plt.xticks([0, 1, 2, 3], figure_covariates_names,
                       fontsize=7,
                       rotation=30)
            plt.ylabel('Obs. GLM weights', fontsize=10, color='black')
            plt.legend(fontsize=8, loc='best')
        else:
            plt.yticks([-5, 0, 5, 10, 15, 20, 25], ['', '', '', '', '', '', ''])
            plt.xticks([0, 1, 2, 3], ['', '', '', ''])
        plt.axhline(y=0, color="k", alpha=0.5, ls="--", linewidth=0.75)

    Params_model_glob = get_global_weights(global_directory, K)
    trans_weight_append_zero = np.vstack((Params_model_glob[1][1], np.zeros((1, Params_model_glob[1][1].shape[1]))))

    # standardize the plotted GLM transition weights
    trans_weight_append_zero_standard = copy.deepcopy(trans_weight_append_zero)
    v1 = - np.mean(trans_weight_append_zero, axis=0)  # this is v1 instead of w1=0
    trans_weight_append_zero_standard[-1, :] = v1
    for i in range(K - 1):
        trans_weight_append_zero_standard[i, :] = v1 + trans_weight_append_zero[i, :]  # vi = v1 + wi
    global_weights_trans = trans_weight_append_zero_standard

    # plot trans weights
    for k in range(K):
        plt.subplot(2, 4, k + 5)
        for animal in mice_names:
            path_analysis = path_main_folder + animal + '/'

            cv_file = path_analysis + "/diff_folds_fit.npz"
            diff_folds_fit = cross_validation_vector(cv_file)

            with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
                optimal_initialize_dict = json.load(f)

            # Get the file name corresponding to the best initialization for given K value
            params_and_LL = get_file_name_for_best_model_fold(
                diff_folds_fit, K, path_analysis, optimal_initialize_dict)
            Params_model, lls = model_data_glmhmm(params_and_LL)
            weight_vectors = -Params_model[2]
            trans_weight_append_zero = np.vstack((Params_model[1][1], np.zeros(
                (1, Params_model[1][1].shape[1]))))

            # Standardize the plotted GLM transition weights
            trans_weight_append_zero_standard = copy.deepcopy(trans_weight_append_zero)
            v1 = - np.mean(trans_weight_append_zero, axis=0)  # this is v1 instead of w1=0
            trans_weight_append_zero_standard[-1, :] = v1
            for i in range(K - 1):
                trans_weight_append_zero_standard[i, :] = v1 + trans_weight_append_zero[i, :]  # vi = v1 + wi
            weight_vectors_trans = trans_weight_append_zero_standard

            if animal == "churchlandlab/CSHL_014/_ibl_subjectTrials.table.61f6982a-40fb-44a6-8f2f-170235951e26.pqt":
                plt.plot(range(M_trans),
                         global_weights_trans[label[k]],
                         '-o',
                         color='k',
                         lw=1.3,
                         alpha=1,
                         markersize=3,
                         zorder=1,
                         label='global fit')
            else:
                plt.plot(range(M_trans),
                         weight_vectors_trans[label[k]],
                         '-o',
                         color=cols[k],
                         lw=1,
                         alpha=0.7,
                         markersize=3,
                         zorder=0)

        if k == 0:
            plt.yticks([-2, -1, 0, 1, 2, 3], fontsize=9)
            plt.xticks([0, 1, 2, 3, 4, 5], figure_covariates_names_trans,
                       fontsize=7,
                       rotation=90)
            plt.ylabel('Tran. GLM weights', fontsize=10)
        else:
            plt.yticks([-2, -1, 0, 1, 2, 3], ['', '', '', '', '', ''])
            plt.xticks([0, 1, 2, 3, 4, 5], ['', '', '', '', '', ''])

        plt.axhline(y=0, color="k", alpha=0.5, ls="--", linewidth=0.75)
        if k == 0:
            plt.legend(fontsize=8,
                       loc='best')
    fig.savefig(figure_dir + 'all_animals_weights.pdf')


