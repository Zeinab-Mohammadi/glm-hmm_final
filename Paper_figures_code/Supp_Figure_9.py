import numpy as np

import json
import matplotlib.pyplot as plt
from io_utils import cross_validation_vector, get_was_correct, model_data_glmhmm, get_mouse_info, colors_func, addBiasBlocks
from analyze_results_utils import get_file_name_for_best_model_fold, calculate_posterior_given_data, data_segmentation_session, \
    mask_for_violations
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io_utils import model_data_glmhmm, colors_func, cross_validation_vector, make_cross_valid_for_figure
from analyze_results_utils import get_global_weights, get_file_name_for_best_model_fold, permute_transition_matrix, find_corresponding_states_new
from Review_utils import load_data, get_marginal_posterior, calculate_state_permutation_all_data, load_glmhmm_data, load_cv_arr, get_file_name_for_best_model_fold_GLM_O
import numpy as np
import json
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    K = 4
    cols = colors_func(K)
    length = 400

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/fig_reviews/Rev1_posterior_plots_with_and_without_GLM_O/'

    animal = "churchlandlab/CSHL_014/_ibl_subjectTrials.table.61f6982a-40fb-44a6-8f2f-170235951e26.pqt"
    path_that_animal = path_of_the_directory + animal
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
    all_sessions = np.unique(session)

    not_viols_ratio = []
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
    posterior_probs, P = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma,
                                                perm)

    # sessions
    sess_to_plot = ["dab3d729-8983-4431-9d88-0640b8ba4fdd", "90e37c04-0da1-4bf7-9f2e-197b09c13dba",
                    "7d824d9b-50fc-449e-8ebb-ee7b3844df18"]

    ############################# now posterior for no GLM_T model ##############################
    covar_set = 2
    # for animal in animal_list:
    trans_p_posterior = []
    path_that_animal = path_of_the_directory + animal

    # results_dir = '/Users/zm6112/Dropbox/Python_code/Pycharm_Z_code_github/glm-hmm_all_data_GLM_trans_diff_inputs/results/ibl_global_fit/' + 'covar_set_' + str(covar_set) + '/' # '/Users/zashwood/Documents/glm-hmm/results/ibl_individual_fit/' + 'covar_set_' + str(covar_set) + '/' + 'prior_sigma_'+ str(sigma_val) + '_transition_alpha_' + str(alpha_val) + '/' + animal +'/'
    results_dir = '/Users/zm6112/Dropbox/Python_code/Pycharm_Z_code_github/glm-hmm_all_no_GLM-T_to_compare/results/ibl_individual_fit/' + 'covar_set_' + str(
        covar_set) + '/' + 'prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(
        alpha_val) + '/' + animal + '/'
    global_directory = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(num_inputs) + '/'


    # cv_file = results_dir + "/cvbt_folds_model.npz"
    # cvbt_folds_model = load_cv_arr(cv_file)

    cv_file = results_dir + "/cvbt_folds_model.npz"  # results_dir + "/cvbt_folds_model.npz"
    cvbt_folds_model = load_cv_arr(cv_file)

    K = 4  # number of states
    cols = colors_func(K)

    with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)

    # Get the file name corresponding to the best initialization for given K value
    raw_file = get_file_name_for_best_model_fold_GLM_O(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)

    # Save parameters for initializing individual fits
    weight_vectors = hmm_params[2]  # [permutation]
    log_transition_matrix = hmm_params[1][0]  # permute_transition_matrix(hmm_params[1][0], permutation)
    init_state_dist = hmm_params[0][0]  # [permutation]

    # Also get data for animal:
    inpt, inpt_trans, y, session, left_probs, animal_eids = load_data(path_data + animal + '_processed.npz')
    inpt = np.hstack((inpt, np.ones((inpt.shape[0], 1))))

    y_init_size = y.shape[0]
    not_viols = np.where(y != -1)
    not_viols_size = y[not_viols].shape[0]
    y = y[not_viols[0], :]
    inpt = inpt[not_viols[0], :]
    session = session[not_viols[0]]
    left_probs = left_probs[not_viols[0]]
    not_viols_ratio.append((not_viols_size / y_init_size))
    inpt = inpt[:, [0, 1, 2, 3]]

    # Create mask:
    # Identify violations for exclusion:
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = mask_for_violations(violation_idx, inpt.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, inputs_trans, datas, train_masks = data_segmentation_session(inpt, inpt_trans, y, mask, session)
    permutation = range(K)
    globe = False
    posterior_probs_no_GLM_T = get_marginal_posterior(globe, inputs, datas, hmm_params, K, permutation, alpha_val,
                                             prior_sigma)  # get_marginal_posterior(inputs, datas, train_masks, hmm_params, K, range(K), alpha_val, sigma_val)

    # dir_for_save = '/Users/zashwood/Documents/glm-hmm/results/ibl_individual_fit/' + 'covar_set_' + str(covar_set) + '/' + 'prior_sigma_'+ str(sigma_val) + '_transition_alpha_' + str(alpha_val) + '/posterior_probs_for_seb/'
    # np.savez(dir_for_save + animal +'_posterior_probs.npz', posterior_probs, inpt, y, session, -weight_vectors, np.exp(log_transition_matrix), np.exp(init_state_dist))

    for i, sess in enumerate(sess_to_plot):
        idx_session = np.where(session == sess)
        this_inpt = inpt[idx_session[0], :]
        posterior_probs_this_session_no_GLLM_T = posterior_probs_no_GLM_T[idx_session[0], :]

        Params_model_glob = get_global_weights(global_directory, K)
        global_weights = -Params_model_glob[2]
        # Plot trial structure for this session too:
        label = find_corresponding_states_new(Params_model_glob)

        posterior_probs_needed_this_session = posterior_probs[idx_session[0], :]
        this_left_probs = left_probs[idx_session[0]]

    ################################## plot states prob versus each other #####################################
    max_posterior_no_GLM_T = []
    max_posterior = []

    for i, sess in enumerate(sess_to_plot):
        idx_session = np.where(session == sess)
        this_inpt = inpt[idx_session[0], :]
        posterior_probs_this_session_no_GLLM_T = posterior_probs_no_GLM_T[idx_session[0], :]
        posterior_probs_needed_this_session = posterior_probs[idx_session[0], :]

        max_values_no_GLM_T = np.max(posterior_probs_this_session_no_GLLM_T, axis=1)
        max_values_GLM_T = np.max(posterior_probs_needed_this_session, axis=1)

        # Filter based on the condition: include only if both are greater than T
        valid_indices = (max_values_no_GLM_T > 0.6) & (max_values_GLM_T > 0.6)

        max_posterior_no_GLM_T.extend(max_values_no_GLM_T[valid_indices])
        max_posterior.extend(max_values_GLM_T[valid_indices])

    # Convert lists to arrays
    max_posterior_no_GLM_T = np.array(max_posterior_no_GLM_T)
    max_posterior = np.array(max_posterior)

#------------------- plot the max as last subfig
    fig = plt.figure(figsize=(9, 8.8))
    plt.subplots_adjust(hspace=0.3)

    i = 0
    sess = sess_to_plot[0]
    idx_session = np.where(session == sess)
    this_inpt = inpt[idx_session[0], :]
    posterior_probs_this_session_no_GLLM_T = posterior_probs_no_GLM_T[idx_session[0], :]
    label = find_corresponding_states_new(Params_model_glob)
    posterior_probs_needed_this_session = posterior_probs[idx_session[0], :]
    this_left_probs = left_probs[idx_session[0]]

    # Subplot 1: No GLM-T posterior
    plt.subplot(3, 1, 1)
    for k in range(K):
        plt.plot(posterior_probs_this_session_no_GLLM_T[:, label[k]], lw=1, color=cols[k], linestyle='--')
    fig = addBiasBlocks(fig, this_left_probs)
    plt.title("Posterior Probs (No GLM-T)", fontsize=10)
    plt.ylabel("p(state)", fontsize=10)
    plt.ylim((-0.01, 1.01))
    plt.xlim((0, 400))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Subplot 2: With GLM-T posterior
    plt.subplot(3, 1, 2)
    for k in range(K):
        plt.plot(posterior_probs_needed_this_session[:, k], lw=1, color=cols[k])
    fig = addBiasBlocks(fig, this_left_probs)
    plt.title("Posterior Probs (With GLM-T)", fontsize=10)
    plt.ylabel("p(state)", fontsize=10)
    plt.ylim((-0.01, 1.01))
    plt.xlim((0, 400))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Subplot 3: Max posterior with vs. without GLM-T
    plt.subplot(3, 1, 3)
    max_glmT = np.max(posterior_probs_needed_this_session, axis=1)
    max_no_glmT = np.max(posterior_probs_this_session_no_GLLM_T, axis=1)
    plt.plot(max_glmT, color='purple', label='With GLM-T')
    plt.plot(max_no_glmT, color='orchid', linestyle='--', label='Without GLM-T')
    plt.title("Max Posterior Probability", fontsize=10)
    plt.xlabel("trial #", fontsize=10)
    plt.ylabel("Max P(state)", fontsize=10)
    plt.ylim(0, 1.01)
    plt.xlim((0, 400))
    plt.legend(fontsize=9)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    fig.savefig(figure_dir + 'first_session_max_prob_as_third_panel.pdf')
    plt.show()


