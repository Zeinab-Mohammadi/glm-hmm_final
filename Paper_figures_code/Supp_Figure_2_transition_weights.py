import numpy as np
import json
import copy
import os
import matplotlib.pyplot as plt
from io_utils import  model_data_glmhmm, colors_func, cross_validation_vector, make_cross_valid_for_figure
from analyze_results_utils import get_file_name_for_best_model_fold, find_corresponding_states_new, \
    permute_transition_matrix, makeRaisedCosBasis

if __name__ == '__main__':
    for num_inputs in [4]:
        K = 5
        cols = colors_func(K)

        figure_covariates_names = ['$\Delta$ contrast', 'past choice', 'stimulus side', 'bias']
        figure_covariates_names_trans = ['filtered choice', 'filtered stim. side', 'filtered reward', 'basis_1', 'basis_2',
                                 'basis_3']
        path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/'  # '/Users/zashwood/Documents/glm-hmm/data/ibl/Della_cluster_data/'
        path_analysis = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(
            num_inputs) + '/'
        figure_dir = '../../glm-hmm_package/results/figures_for_paper/Supp_figures/figure_2/'

        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        cv_file = path_analysis + "/diff_folds_fit.npz"
        diff_folds_fit = cross_validation_vector(cv_file)
        with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
            optimal_initialize_dict = json.load(f)

        # Get the file name corresponding to the best initialization for given K value
        params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
        Params_model, lls = model_data_glmhmm(params_and_LL)
        # Calculate perm
        perm = find_corresponding_states_new(Params_model)
        weight_vectors = Params_model[2][perm]
        log_transition_matrix = permute_transition_matrix(Params_model[1][0], perm)
        init_state_dist = Params_model[0][0][perm]

        # standardize the plotted GLM transition weights
        trans_weight_append_zero = np.vstack((Params_model[1][1], np.zeros(
            (1, Params_model[1][1].shape[1]))))
        trans_weight_append_zero_standard = copy.deepcopy(trans_weight_append_zero)
        v1 = - np.mean(trans_weight_append_zero, axis=0)  # this is v1 instead of w1=0
        trans_weight_append_zero_standard[-1, :] = v1
        for i in range(K - 1):
            trans_weight_append_zero_standard[i, :] = v1 + trans_weight_append_zero[i, :]  # vi = v1 + wi
        weight_vectors_trans = trans_weight_append_zero_standard[perm]
        params_for_individual_initialization = [[Params_model[0][0]], [Params_model[1][0], Params_model[1][1]], Params_model[2]]
        if not os.path.exists(path_analysis + 'perms/perm_K_' + str(K)):
            os.makedirs(path_analysis + 'perms/perm_K_' + str(K))
        np.savez(path_analysis + 'perms/perm_K_' + str(K) + '.npz', perm)

        # plotting comparison plot
        cols_compare = ["#7e1e9c", "#0343df"]
        fig = plt.figure(figsize=(3.2, 3.2))
        plt.subplots_adjust(left=0.3, bottom=0.4, right=0.8, top=0.9)
        cv_file = path_analysis + "/diff_folds_fit.npz"
        idx = np.array([0, 3, 4, 5, 6])
        data_for_plotting_df, loc_best, best_val, glm_lapse_model = make_cross_valid_for_figure(
            cv_file, idx)
        cv_file_train = path_analysis + "/train_folds_fit.npz"
        train_data_for_plotting_df, train_loc_best, train_best_val, train_glm_lapse_model = make_cross_valid_for_figure(
            cv_file_train, idx)

        glm_lapse_model_cvbt_means = np.mean(glm_lapse_model, axis=1)
        train_glm_lapse_model_cvbt_means = np.mean(train_glm_lapse_model, axis=1)

        # GLM-trans weights
        fig = plt.figure(figsize=(4, 3))
        plt.subplots_adjust(bottom=0.3)
        plt.subplots_adjust(left=0.3, bottom=0.4, right=0.8, top=0.9)
        M_trans = weight_vectors_trans.shape[1] - 1

        for k in range(K):
            plt.plot(range(M_trans - 2), weight_vectors_trans[k][0:3], marker='D', markersize=5,
                     color=cols[k], lw=1)
        plt.xticks(list(range(0, len(figure_covariates_names_trans[0:3]))), figure_covariates_names_trans[0:3],
                   rotation=45, fontsize=8)
        plt.yticks(fontsize=10)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        # plt.ylim((-.75, .75))
        plt.yticks([-.6, -.3, 0, .3, .6], fontsize=10)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.ylabel("Tran. Weights", fontsize=10)
        # plt.legend(fontsize=8, loc='upper right')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        fig.savefig(figure_dir + 'Trans_weights_K=' + str(K) + '.pdf')

        # make basis functions
        fig = plt.figure(figsize=(5, 2.5))
        plt.subplots_adjust(left=0.3, bottom=0.4, right=0.8, top=0.9)
        bias_num = 3
        cosBasis, tgrid, basisPeaks = makeRaisedCosBasis(bias_num)

        for k in range(K):
            time_trace = np.array(
                (weight_vectors_trans[k][3] * (cosBasis[:, 0])) + (weight_vectors_trans[k][4] * (cosBasis[:, 1])) + (
                            weight_vectors_trans[k][5] * (cosBasis[:, 2])))
            plt.plot(range(cosBasis.shape[0]), time_trace, color=cols[k], lw=1, ls="--")

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylim((-.6, .25))
        plt.yticks([-.6, -.4, -.2, 0, .2], fontsize=10)
        plt.ylabel("Bases effect", fontsize=10)
        plt.xlabel("Trial", fontsize=10)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        fig.savefig(figure_dir + 'traces_basis_K=' + str(K) + '.pdf')



