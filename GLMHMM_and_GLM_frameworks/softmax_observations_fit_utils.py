"""

Assorted auxiliary functions for fitting the models

"""
import autograd.numpy as np

import ssm
from autograd.scipy.special import logsumexp
import sys
import json
import copy


def create_hmm_simulated_data(K, D, C, obs_mat, glm_vectors):
    T = obs_mat.shape[0]
    M = obs_mat.shape[1]
    this_hmm = ssm.HMM(K, D, M,
                       observations="softmax", observation_kwargs=dict(C=C),
                       transitions="inputdriven")
    glm_vectors_repeated = np.tile(glm_vectors, (K, 1, 1))
    glm_vectors_with_noise = glm_vectors_repeated + np.random.normal(0, 0.1, glm_vectors_repeated.shape)
    this_hmm.observations.params = glm_vectors_with_noise
    z, y = this_hmm.sample(T, input=obs_mat)
    true_ll = this_hmm.log_probability([y], inputs=[obs_mat])
    print("True ll = " + str(true_ll))
    return z, y, true_ll, this_hmm.params


def cross_validation_vector(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    diff_folds_fit = data[0]
    return diff_folds_fit


def model_data_glmhmm(data_file):
    container = np.load(data_file, allow_pickle=True)
    data = [container[key] for key in container]
    this_Params_model = data[0]
    lls = data[1]
    return [this_Params_model, lls]


def get_file_name_for_best_model_fold(diff_folds_fit, K, path_main_folder, optimal_initialize_dict):
    loc_best = 0
    best_fold = np.where(diff_folds_fit[loc_best, :] == max(diff_folds_fit[loc_best, :]))[0][0]
    base_path = path_main_folder + '/Model/glmhmm_#state=' + str(K) + '/fld_num=' + str(best_fold)
    keys = '/Model/glmhmm_#state=' + str(K) + '/fld_num=' + str(best_fold)
    print('keys=', keys)
    print('optimal_initialize_dict=', optimal_initialize_dict)
    best_iter = optimal_initialize_dict[keys]
    params_and_LL = base_path + '/iter_' + str(best_iter) + '/glm_hmm_raw_parameters_itr_' + str(best_iter) + '.npz'
    return params_and_LL


def get_file_name_for_best_model_fold_GLM_obs(cvbt_folds_model, K, overall_dir, optimal_initialize_dict):
    loc_best = 0
    best_fold = np.where(cvbt_folds_model[loc_best, :] == max(cvbt_folds_model[loc_best, :]))[0][0]
    base_path = overall_dir + '/Model/glmhmm_#state=' + str(K) + '/fold_' + str(best_fold)  # don't change '/fold_' for this function to '/fld_num='
    key_for_dict = '/Model/glmhmm_#state=' + str(K) + '/fold_' + str(best_fold)  # don't change '/fold_' for this function to '/fld_num='
    print('overall_dir=', overall_dir)
    print('key_for_dict=', key_for_dict)
    print('optimal_initialize_dict=', optimal_initialize_dict)
    best_iter = optimal_initialize_dict[key_for_dict]
    raw_file = base_path + '/iter_' + str(best_iter) + '/glm_hmm_raw_parameters_itr_' + str(best_iter) + '.npz'
    return raw_file


def fit_hmm_observations(datas, inputs, inputs_trans, train_masks, K, D, M, M_trans, C, num_iters_EM_fit, transition_alpha,
                         prior_sigma, global_fit, GLM_T_init_0, params_for_initialization, save_title):
    """
    Instantiate and fit GLM-HMM model
    """

    GLM_initial_obs = True
    num_inputs = 4
    cluster = True

    if global_fit is True:
        # Prior variables
        this_hmm = ssm.HMM_TO(K, D, M_trans=M_trans, M_obs=M, observations="input_driven_obs_diff_inputs",
                              observation_kwargs=dict(C=C, prior_sigma=prior_sigma),
                              transitions="inputdrivenalt", transition_kwargs=dict(alpha=transition_alpha, kappa=0))

        # Initialize observation weights as GLM weights with some noise:
        if GLM_initial_obs is False:
            glm_vectors_repeated = np.tile(params_for_initialization, (K, 1, 1))
            glm_vectors_with_noise = glm_vectors_repeated + np.random.normal(0, 0.2, glm_vectors_repeated.shape)
            this_hmm.observations.params = glm_vectors_with_noise

        # Instead of above: Initialize observation weights with GLM only model obs weights
        if GLM_initial_obs is True:
            """dont change below to 'glm-hmm_all_data_GLM_trans_diff_inputs' as we are initializing from GLM only"""
            if cluster is True:
                path_analysis = '/home/zm6112/glm-hmm_package/glm-hmm_all_no_GLM-T_to_compare/results/model_global_ibl/' + 'num_regress_obs_' + str(
                    num_inputs) + '/'
            else:
                path_analysis = '../../glm-hmm_package/glm-hmm_all_no_GLM-T_to_compare/results/model_global_ibl/' + 'num_regress_obs_' + str(
                    num_inputs) + '/'

            cv_file = path_analysis + "/diff_folds_fit.npz"
            diff_folds_fit = cross_validation_vector(cv_file)

            with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
                optimal_initialize_dict = json.load(f)

            # Get the file name corresponding to the best initialization for given K value
            params_and_LL = get_file_name_for_best_model_fold_GLM_obs(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
            Params_model, lls = model_data_glmhmm(params_and_LL)
            this_hmm.observations.params = Params_model[2]  # observation weights
        sys.stdout.flush()

    else:
        Wk_glob = copy.deepcopy(params_for_initialization[2])
        this_hmm = ssm.HMM_TO(K, D, M_trans=M_trans, M_obs=M, observations="input_driven_obs_diff_inputs",
                              observation_kwargs=dict(C=C, prior_sigma=prior_sigma),
                              transitions="inputdrivenalt",
                              transition_kwargs=dict(prior_sigma=prior_sigma, alpha=transition_alpha, kappa=0))

        # Initialize HMM-GLM with global parameters:
        this_hmm.params = copy.deepcopy(params_for_initialization)


    if GLM_T_init_0 is True:
        this_hmm.transitions.params[1][None] = np.zeros((1, K - 1, M_trans))  # K-1 because there is inputdrivenalt

    print("=== fitting HMM ========")
    # sys.stdout.flush()
    lls = this_hmm.fit(datas, transition_input=inputs_trans, observation_input=inputs, train_masks=train_masks,
                       method="em", num_iters=num_iters_EM_fit, initialize=False, tolerance=10 ** -4)

    # Save raw parameters of HMM, as well as loglikelihood and accuracy calculated during training
    np.savez(save_title, this_hmm.params, lls)
    return None


def permute_z_inf(z_inf, perm):
    # Now modify inferred_z so that labeling of latents matches that of true z:
    perm_dict = dict(zip(perm, range(len(perm))))
    inferred_z = np.array([perm_dict[x] for x in z_inf])
    return inferred_z


# Calculate prediction accuracy of GLM-HMM
def prediction_precision_compute(y, obs_mat, this_hmm):
    # Calculate most probable observation class at each time point:
    time_dependent_logits = this_hmm.observations.calculate_logits(obs_mat)
    # Now marginalize over the latent dimension:
    time_dependent_class_log_probs = logsumexp(time_dependent_logits, axis=1)
    assert time_dependent_class_log_probs.shape == (
        obs_mat.shape[0], time_dependent_logits.shape[2]), "wrong shape for time_dependent_class_log_probs"
    # Now find the location of the max along the C dimension
    predicted_class_labels = np.argmax(time_dependent_class_log_probs, axis=1)
    # Calculate where y and predicted_class_labels line up:
    predictive_acc = np.sum(y[:, 0] == predicted_class_labels) / y.shape[0]
    print("predictive accuracy = " + str(predictive_acc))
    return predictive_acc


# Append column of zeros to weights matrix in appropriate location
def add_zero_column(weights):
    weights_tranpose = np.transpose(weights, (1, 0, 2))
    weights = np.transpose(
        np.vstack([weights_tranpose, np.zeros((1, weights_tranpose.shape[1], weights_tranpose.shape[2]))]), (1, 0, 2))
    return weights


# Reshape hessian and calculate its inverse
def standard_deviation_hmm(hessian, perm, M, K, C):
    # Reshape hessian
    hessian = np.reshape(hessian, ((K * (C - 1) * (M + 1)), (K * (C - 1) * (M + 1))))
    # Calculate inverse of Hessian (this is what we will actually use to calculate variance)
    inv_hessian = hessian
    # Take diagonal elements and calculate square root
    standard_deviation = np.sqrt(np.diag(inv_hessian))
    # Undo perm
    unflattened_standard_deviation = np.reshape(standard_deviation, (K, C - 1, M + 1))
    # Append zeros
    unflattened_standard_deviation = add_zero_column(unflattened_standard_deviation)
    # Now undo perm:
    unflattened_standard_deviation = unflattened_standard_deviation[perm]
    # Now reflatten standard_deviation (for ease of plotting)
    flattened_standard_deviation = unflattened_standard_deviation.flatten()
    return flattened_standard_deviation
