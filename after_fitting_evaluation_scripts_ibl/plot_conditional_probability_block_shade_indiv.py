"""

Plot posterior probability (individual fit) for different sessions with the background color for the biased blocks.

"""
import numpy as np
import json
import matplotlib.pyplot as plt
from glm_hmm_utils import get_file_name_for_best_model_fold, permute_transition_matrix, calculate_posterior_given_data, \
    data_segmentation_session, mask_for_violations, cross_validation_vector, model_data_glmhmm, get_mouse_info, mice_names_info, colors_func

if __name__ == '__main__':
    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    mice_names = mice_names_info(path_data + 'mice_names.npz')
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'
    alpha_val = 2.0
    prior_sigma = 4.0
    num_inputs = 4
    K = 5  # number of states
    cols = colors_func(K)
    not_viols_ratio = []

    for animal in mice_names:
        trans_p_posterior = []
        path_that_animal = path_of_the_directory + animal
        path_analysis = '../../glm-hmm_package/results/model_indiv_ibl/' + 'num_regress_obs_' + str(
            num_inputs) + '/' + 'prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(
            alpha_val) + '/' + animal + '/'

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
        perm = range(K)
        globe = False
        posterior_probs, P = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val,
                                                    prior_sigma,
                                                    perm)
        unique_sessions = np.unique(session)
        idx = np.around(np.linspace(2, len(unique_sessions) - 1, 4))
        sess_to_plot = unique_sessions

        # plot: multiplying the P_matrix and posteriors
        P_st = 0
        for i, sess in enumerate(sess_to_plot):
            idx_session = np.where(session == sess)
            needed_obs_mat = obs_mat[idx_session[0], :]
            this_left_probs = left_probs[idx_session[0]]
            posterior_probs_needed_this_session = posterior_probs[idx_session[0], :]
            P_needed_this_session = P[P_st:(P_st + idx_session[0].shape[0] - 1), :]
            P_st = idx_session[0].shape[0] - 1

            # Plot trial structure for this session too:
            for p in range(P_needed_this_session.shape[0]):
                trans_K_K = np.zeros((K, K))
                for k in range(K):
                    trans_K_K[k, :] = P_needed_this_session[p, k, :] * posterior_probs_needed_this_session[p,
                                                                :]
                trans_p_posterior.append(trans_K_K)
        trans_p_posterior_mean = permute_transition_matrix(np.mean(trans_p_posterior, axis=0), perm)

        # plot transition matrix with numbers
        trans_p_posterior_mean_normalize = np.zeros((K, K))
        for k in range(K):
            trans_p_posterior_mean_normalize[k, :] = trans_p_posterior_mean[k, :] / np.sum(
                trans_p_posterior_mean[k, :])


