import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
from glm_hmm_utils import find_corresponding_states_new, get_file_name_for_best_model_fold, posterior_probs_wrapper, model_data_glmhmm, mice_names_info, colors_func, \
    cross_validation_vector, get_min_train_test_fold_size, make_cross_valid_for_figure_indiv

if __name__ == '__main__':
    # Set parameters
    sigma_vec = [4.0]
    sigma_num = len(sigma_vec)  # number of sigmas
    alpha_val = 2.0
    fold_num = 5
    max_state = 5
    num_inputs = 4

    # Set data directory and load animal list
    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    mice_names = mice_names_info(path_data + 'mice_names.npz')
    animal_num = np.array(mice_names).shape[0]
    vector_plot = np.zeros((max_state, fold_num, sigma_num, animal_num))

    # Loop through animals
    for anim_num, animal in enumerate(mice_names):
        sigma_indx = -1
        for prior_sigma in sigma_vec:
            sigma_indx += 1

            figure_covariates_names = ['stim', 'pc_flt', 'prev_stim', 'bias']
            figure_covariates_names_trans = ['pc_flt', 'stim_side_flt', 'pr_flt', 'basis1', 'basis2', 'basis3']

            # Set results directory
            path_analysis = '../../glm-hmm_package/results/model_indiv_ibl/' + 'num_regress_obs_' + str(
                num_inputs) + '/prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(alpha_val) + '/' + animal + '/'
            cv_file = path_analysis + "/diff_folds_fit.npz"
            diff_folds_fit = cross_validation_vector(cv_file)
            min_train_size, min_test_size = get_min_train_test_fold_size(path_data, animal)

            # Loop through different values of K (number of states)
            for K in range(2, 6):
                dir_all_figs = '../../glm-hmm_package/results/model_indiv_ibl/' + 'num_regress_obs_' + str(
                    num_inputs) + '/prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(
                    alpha_val) + '/all_optimum_model_fit_figures/' + 'stats_K_' + str(K) + '/'
                with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
                    optimal_initialize_dict = json.load(f)

                # Get the file name corresponding to the best initialization for given K value
                params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
                Params_model, lls = model_data_glmhmm(params_and_LL)

                weight_vectors = Params_model[2]
                trans_weight_append_zero = np.vstack((Params_model[1][1], np.zeros((1, Params_model[1][1].shape[1]))))
                log_transition_matrix = Params_model[1][0]
                init_state_dist = Params_model[0][0]

                # Standardize the plotted GLM transition weights
                trans_weight_append_zero_standard = copy.deepcopy(trans_weight_append_zero)
                v1 = - np.mean(trans_weight_append_zero, axis=0)  # this is v1 instead of w1=0
                trans_weight_append_zero_standard[-1, :] = v1
                for i in range(K - 1):
                    trans_weight_append_zero_standard[i, :] = v1 + trans_weight_append_zero[i, :]  # vi = v1 + wi
                weight_vectors_trans = trans_weight_append_zero_standard