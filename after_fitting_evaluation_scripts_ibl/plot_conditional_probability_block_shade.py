"""

Plot posterior probability for different sessions with the background color for the biased blocks.

"""
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from glm_hmm_utils import get_file_name_for_best_model_fold, calculate_posterior_given_data, data_segmentation_session, mask_for_violations, \
    cross_validation_vector, model_data_glmhmm, get_mouse_info, mice_names_info, addBiasBlocks


if __name__ == '__main__':
    K = 5  # number of states
    prior_sigmas = [4.0]
    not_viols_ratio = []
    num_inputs = 4
    alpha_val = 2.0

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    mice_names = mice_names_info(path_data + 'mice_names.npz')
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'

    for animal in mice_names:
        path_that_animal = path_of_the_directory + animal
        for prior_sigma in prior_sigmas:
            path_analysis = '../../glm-hmm_package/results/model_indiv_ibl/' + 'num_regress_obs_' + str(
                num_inputs) + '/' + 'prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(alpha_val) + '/' + animal + '/'

            cv_file = path_analysis + "/diff_folds_fit.npz"
            diff_folds_fit = cross_validation_vector(cv_file)

            with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
                optimal_initialize_dict = json.load(f)

            # Get the file name corresponding to the best initialization for given K value
            params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
            Params_model, lls = model_data_glmhmm(params_and_LL)

            # Save parameters for initializing individual fits
            weight_vectors = Params_model[2]
            log_transition_matrix = Params_model[1][0]
            init_state_dist = Params_model[0][0]

            # Also get data for animal:
            obs_mat, trans_mat, y, session, left_probs, animal_eids = get_mouse_info(path_data + animal + '_processed.npz')
            obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))

            y_init_size = y.shape[0]
            not_viols = np.where(y != -1)
            not_viols_size = y[not_viols].shape[0]
            y = y[not_viols[0], :]
            obs_mat = obs_mat[not_viols[0], :]
            session = session[not_viols[0]]
            left_probs = left_probs[not_viols[0]]
            not_viols_ratio.append((not_viols_size / y_init_size))
            obs_mat = obs_mat[:, [0, 1, 2, 3]]

            # Create mask:
            # Identify violations for exclusion:
            index_viols = np.where(y == -1)[0]
            nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])
            y[np.where(y == -1), :] = 1
            inputs, inputs_trans, datas, train_masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)
            globe = False
            perm = range(K)
            posterior_probs = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma, perm)
            unique_sessions = np.unique(session)

            idx = np.around(np.linspace(2, len(unique_sessions) - 1, 4))
            sess_to_plot = unique_sessions