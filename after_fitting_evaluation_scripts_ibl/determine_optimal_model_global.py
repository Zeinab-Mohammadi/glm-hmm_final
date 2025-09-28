"""

Save the optimal parameters from the global fit for initializing individual fits.

"""

# Save best parameters (global fit) for initializing individual fits

import numpy as np
import json
import copy
import os
from glm_hmm_utils import get_file_name_for_best_model_fold, permute_transition_matrix, makeRaisedCosBasis, \
    find_corresponding_states_new, model_data_glmhmm, colors_func, cross_validation_vector, make_cross_valid_for_figure

if __name__ == '__main__':

    for num_inputs in [4]:
        figure_covariates_names = ['stimulus', 'pc', 'past stimuli', 'bias']
        figure_covariates_names_trans = ['pc_flt', 'stim_side_flt', 'pr_flt', 'basis1', 'basis2', 'basis3']
        path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/'
        path_analysis = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(
            num_inputs) + '/'

        cv_file = path_analysis + "/diff_folds_fit.npz"
        diff_folds_fit = cross_validation_vector(cv_file)
        states_num = 6  # number of latent states

        for K in range(2, states_num + 1):
            with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
                optimal_initialize_dict = json.load(f)

            # Get the file name corresponding to the best initialization for given K value
            params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
            Params_model, lls = model_data_glmhmm(params_and_LL)  # so Params_model=this_hmm.params

            # To know more about transitions and observation weights see below
            # Calculate perm
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
                trans_weight_append_zero_standard[i, :] = v1 + trans_weight_append_zero[i, :]  # vi = v1 + wi
            weight_vectors_trans = trans_weight_append_zero_standard[perm]

            params_for_individual_initialization = [[Params_model[0][0]], [Params_model[1][0], Params_model[1][1]],
                                                    Params_model[2]]
            # print('params_for_individual_initialization=', params_for_individual_initialization)
            # print('params_for_individual_initialization.shape=', np.array(params_for_individual_initialization).shape)
            np.savez(path_data + 'optimum_model/optimum_model_K_' + str(K) + '_num_inputs_' + str(num_inputs) + '.npz',
                     params_for_individual_initialization)
            if not os.path.exists(path_analysis + 'perms/perm_K_' + str(K)):
                os.makedirs(path_analysis + 'perms/perm_K_' + str(K))
            np.savez(path_analysis + 'perms/perm_K_' + str(K) + '.npz', perm)
