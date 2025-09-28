"""

Find the best model after analyzing the results of global fit with a different number of initializations

"""
import numpy as np
import json

from glm_hmm_utils import get_file_name_for_best_model_fold, data_for_cross_validation, state_glob_label, load_fold_session_map, cross_validation_vector, model_data_glmhmm, \
    get_mouse_info_all, log_likelihood_base_for_test, log_likelihood_for_test_glm, cross_valid_bpt_compute, glmhmm_normalized_loglikelihood

if __name__ == '__main__':
    # Parameters
    C = 2  # number of output classes
    cross_valid_num_fold = 5  # number of folds
    D = 1  # number of output dimensions
    states_num = 6  # number of latent states
    fits_num = states_num + 2  # number of models
    globe = True

    for num_inputs in [4]:
        path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/'
        path_analysis = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(
            num_inputs) + '/'
        fold_mapping_session = load_fold_session_map(path_data + 'fold_mapping_session_all_mice.npz')
        animal_preferred_model_dict = {}
        fits = ["GLM_fit", "GLM_HMM_fit"]
        diff_folds_fit = np.zeros((fits_num, cross_valid_num_fold)) 
        train_folds_fit = np.zeros((fits_num, cross_valid_num_fold))

        # Save best initialization for each model-fold combination
        optimal_initialize_dict = {}
        for fld in range(cross_valid_num_fold):
            for fit in fits:
                print("model = " + str(fit))
                if fit == "GLM_fit":
                    obs_mat, trans_mat, output, session = get_mouse_info_all(path_data + 'combined_all_mice.npz')
                    indexes = range(0, [obs_mat.shape[1]-1][0], 1)
                    obs_mat = obs_mat[:, indexes]

                    test_obs_mat, test_trans_mat, test_output, test_mask_without_viol, test_session, train_obs_mat, train_trans_mat, train_output, train_mask_without_viol, \
                    this_train_session, M, M_trans, n_test, n_train = data_for_cross_validation(obs_mat, trans_mat, output, session, fold_mapping_session, fld)
                    base_log_likelihood = log_likelihood_base_for_test(train_output[train_mask_without_viol == 1, :],
                                                     test_output[test_mask_without_viol == 1, :], C)
                    base_log_likelihood_train = log_likelihood_base_for_test(train_output[train_mask_without_viol == 1, :],
                                                           train_output[train_mask_without_viol == 1, :], C)

                    # Load parameters and instantiate a new GLM object with these parameters
                    params_glm = path_analysis + '/Model/glm_#state=1/fld_num=' + str(fld) + '/important_params_iter_0.npz'
                    glm_log_likelihood = log_likelihood_for_test_glm(params_glm, test_output[test_mask_without_viol == 1, :], test_obs_mat[test_mask_without_viol == 1, :], M, C)
                    glm_log_likelihood_train = log_likelihood_for_test_glm(params_glm,
                                                                    train_output[train_mask_without_viol == 1, :],
                                                                    train_obs_mat[train_mask_without_viol == 1, :], M, C)
                    diff_folds_fit[0, fld] = cross_valid_bpt_compute(glm_log_likelihood, base_log_likelihood, n_test)
                    train_folds_fit[0, fld] = cross_valid_bpt_compute(glm_log_likelihood_train, base_log_likelihood_train, n_train)

                elif fit == "GLM_HMM_fit":
                    obs_mat, trans_mat, output, session = get_mouse_info_all(path_data + 'combined_all_mice.npz')
                    obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))
                    indexes2 = np.append(indexes, (obs_mat.shape[1] - 1))
                    obs_mat = obs_mat[:, indexes2]

                    for K in range(2, states_num+1):
                        num_fit = 3 + (K-2)
                        prior_sigma = 100
                        diff_folds_fit[num_fit, fld], train_folds_fit[
                            num_fit, fld], _, _, train_for_arranging_initials = glmhmm_normalized_loglikelihood(globe, prior_sigma, obs_mat, trans_mat, output,
                                                                                               session,
                                                                                               fold_mapping_session,
                                                                                               fld, K, D, C,
                                                                                               path_analysis)

                        # Save best initialization to dictionary for later:
                        keys = '/glmhmm_#state=' + str(K) + '/fld_num=' + str(fld)
                        optimal_initialize_dict[keys] = int(train_for_arranging_initials[0])

        # Save best initialization directories across animals, folds and models (only GLM-HMM):
        json_dump = json.dumps(optimal_initialize_dict)
        f = open(path_analysis + "/optimal_initialize_dict.json", "w")
        f.write(json_dump)
        f.close()
        # Save diff_folds_fit as numpy array for easy parsing across all models and folds
        np.savez(path_analysis + "/diff_folds_fit.npz", diff_folds_fit)
        np.savez(path_analysis + "/train_folds_fit.npz", train_folds_fit)

        # ------------------------------------------------------------------------------------------------
        cv_file = path_analysis + "/diff_folds_fit.npz"
        diff_folds_fit = cross_validation_vector(cv_file)
        states_num = 6  # number of latent states

        for K in range(2, states_num + 1):
            with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
                optimal_initialize_dict = json.load(f)
            # Get the file name corresponding to the best initialization for given K value
            params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
            Params_model, lls = model_data_glmhmm(params_and_LL)
        label_glob = state_glob_label(Params_model)

