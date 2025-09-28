"""

Find the best model after analyzing the results of individual fit

"""

import numpy as np
import json
from glm_hmm_utils import data_for_cross_validation, get_mouse_info, mice_names_info, load_fold_session_map, log_likelihood_base_for_test, log_likelihood_for_test_glm, \
    cross_valid_bpt_compute, glmhmm_normalized_loglikelihood

if __name__ == '__main__':
    transition_alphas = [2.0]
    # prior_sigmas = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    prior_sigmas = [4.0]
    globe = False
    num_inputs = 4

    for transition_alpha in transition_alphas:
        for prior_sigma in prior_sigmas:
            path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
            path_main_folder = '../../glm-hmm_package/results/model_indiv_ibl/num_regress_obs_' + str(
                num_inputs) + '/' + '/prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(
                transition_alpha) + '/'

            mice_names = mice_names_info(path_data + 'mice_names.npz')
            for animal in mice_names:
                if animal != "churchlandlab_IBL_1_trials.pqt":
                    path_analysis_glm_hmm = path_main_folder + animal + '/'
                    path_analysis = '../../glm-hmm_package/results/model_indiv_ibl/' + 'num_regress_obs_' + \
                                  str(num_inputs) + '/prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(
                        transition_alpha) + '/' + animal + '/'
                    fold_mapping_session = load_fold_session_map(path_data + animal + '_fold_session_map.npz')

                    # Parameters
                    C = 2  # number of output classes
                    cross_valid_num_fold = 5  # number of folds
                    D = 1  # number of output dimensions
                    states_num = 5  # number of latent states
                    fits_num = states_num + 2  # model for each latent

                    animal_preferred_model_dict = {}
                    models = ["GLM", "GLM_HMM"]

                    diff_folds_fit = np.zeros((fits_num, cross_valid_num_fold))
                    train_folds_fit = np.zeros((fits_num, cross_valid_num_fold))

                    # Save best initialization for each model-fold combination
                    optimal_initialize_dict = {}
                    for fold in range(cross_valid_num_fold):
                        for model in models:
                            if model == "GLM":
                                obs_mat, trans_mat, y, session, left_probs, animal_eids = get_mouse_info(
                                    path_data + animal + '_processed.npz')

                                indexes = range(0, [obs_mat.shape[1] - 1][0], 1)
                                obs_mat = obs_mat[:, indexes]

                                test_obs_mat, test_trans_mat, test_y, test_nonviolation_mask, this_test_session, train_obs_mat, train_trans_mat, train_y, train_nonviolation_mask, this_train_session, M, M_trans, n_test, n_train = data_for_cross_validation(
                                    obs_mat, trans_mat, y, session, fold_mapping_session, fold)

                                ll0 = log_likelihood_base_for_test(train_y[train_nonviolation_mask == 1, :],
                                                                 test_y[test_nonviolation_mask == 1, :], C)
                                ll0_train = log_likelihood_base_for_test(train_y[train_nonviolation_mask == 1, :],
                                                                       train_y[train_nonviolation_mask == 1, :], C)

                                # Load parameters and instantiate a new GLM object with these parameters
                                params_glm = path_analysis + 'Model/glm_#state=1/fld_num=' + str(
                                    fold) + '/important_params_iter_0.npz'
                                glm_log_likelihood = log_likelihood_for_test_glm(params_glm,
                                                                          test_y[test_nonviolation_mask == 1, :],
                                                                          test_obs_mat[test_nonviolation_mask == 1, :], M,
                                                                          C)

                                glm_log_likelihood_train = log_likelihood_for_test_glm(params_glm,
                                                                                train_y[train_nonviolation_mask == 1,
                                                                                :],
                                                                                train_obs_mat[train_nonviolation_mask == 1,
                                                                                :], M, C)
                                diff_folds_fit[0, fold] = cross_valid_bpt_compute(glm_log_likelihood, ll0, n_test)
                                train_folds_fit[0, fold] = cross_valid_bpt_compute(glm_log_likelihood_train, ll0_train,
                                                                                         n_train)
                            elif model == "GLM_HMM":
                                obs_mat, trans_mat, y, session, left_probs, animal_eids = get_mouse_info(
                                    path_data + animal + '_processed.npz')
                                obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))

                                indexes2 = np.append(indexes, (obs_mat.shape[1] - 1))
                                obs_mat = obs_mat[:, indexes2]

                                for K in range(2, states_num + 1):
                                    model_idx = 3 + (K - 2)
                                    diff_folds_fit[model_idx, fold], train_folds_fit[
                                        model_idx, fold], _, _, train_for_arranging_initials = glmhmm_normalized_loglikelihood(globe,
                                                                                                           prior_sigma,
                                                                                                           obs_mat,
                                                                                                           trans_mat,
                                                                                                           y, session,
                                                                                                           fold_mapping_session,
                                                                                                           fold, K, D,
                                                                                                           C,
                                                                                                           path_analysis_glm_hmm)
                                    keys = '/glmhmm_#state=' + str(K) + '/fld_num=' + str(fold)
                                    optimal_initialize_dict[keys] = int(train_for_arranging_initials[0])

                    # Save best initialization directories across animals, folds and models (only GLM-HMM):
                    json_dump = json.dumps(optimal_initialize_dict)
                    f = open(path_analysis_glm_hmm + "/optimal_initialize_dict.json", "w")
                    f.write(json_dump)
                    f.close()
                    # Save diff_folds_fit as numpy array for easy parsing across all models and folds
                    np.savez(path_analysis_glm_hmm + "/diff_folds_fit.npz", diff_folds_fit)
                    np.savez(path_analysis_glm_hmm + "/train_folds_fit.npz", train_folds_fit)
    #
