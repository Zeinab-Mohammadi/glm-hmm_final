from io_utils import cross_validation_vector, get_was_correct, model_data_glmhmm, get_mouse_info, colors_func, addBiasBlocks
from analyze_results_utils import get_file_name_for_best_model_fold, calculate_posterior_given_data, data_segmentation_session, mask_for_violations
import os
import matplotlib.pyplot as plt
from io_utils import model_data_glmhmm, colors_func, cross_validation_vector, make_cross_valid_for_figure
from analyze_results_utils import get_file_name_for_best_model_fold, permute_transition_matrix, find_corresponding_states_new
from Review_utils import load_data, get_marginal_posterior, calculate_state_permutation_all_data, load_glmhmm_data, load_cv_arr, get_file_name_for_best_model_fold_GLM_O
import numpy as np
import json
import pandas as pd
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

if __name__ == '__main__':
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    K = 4
    cols = colors_func(K)
    length = 400
    cols = colors_func(K)

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/fig_reviews/Rev1_reward_rate_vs_prob_disengag/'

    animal = "churchlandlab/CSHL_014/_ibl_subjectTrials.table.61f6982a-40fb-44a6-8f2f-170235951e26.pqt"
    path_that_animal = path_of_the_directory + animal
    path_analysis = '../../glm-hmm_package/results/model_indiv_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/' + 'prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(alpha_val) + '/' + animal + '/'
    print('path_analysis=', path_analysis)
    cv_file = path_analysis + "/diff_folds_fit.npz"
    diff_folds_fit = cross_validation_vector(cv_file)

    with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
        optimal_initialize_dict = json.load(f)

    # best init file for K
    params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
    print('params_and_LL=', params_and_LL)
    Params_model, lls = model_data_glmhmm(params_and_LL)

    # params (kept as in original)
    weight_vectors = Params_model[2]
    log_transition_matrix = Params_model[1][0]
    init_state_dist = Params_model[0][0]

    # load animal data
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

    # mask & posterior
    index_viols = np.where(y == -1)[0]
    nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, inputs_trans, datas, train_masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)
    perm = range(K)
    globe = False
    posterior_probs, P = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma, perm)

    # === ONLY the correlation figure ===
    fig, ax = plt.subplots(figsize=(2.7, 3))
    T = 500
    plt.subplots_adjust(left=0.2, bottom=0.2)

    # use the same single session
    sess_to_plot = ["90e37c04-0da1-4bf7-9f2e-197b09c13dba"]

    all_disengaged_probs = np.concatenate(
        [np.sum(posterior_probs[np.where(session == sess)[0][:T], 2:3], axis=1) for sess in sess_to_plot if
         len(np.where(session == sess)[0]) >= T]
    )
    all_filtered_rewards = np.concatenate(
        [trans_mat[np.where(session == sess)[0][:T], 2] for sess in sess_to_plot if
         len(np.where(session == sess)[0]) >= T]
    )

    if len(all_filtered_rewards) > 1:
        slope, intercept = np.polyfit(all_filtered_rewards, all_disengaged_probs, 1)
        correlation_coeff, p_value = pearsonr(all_filtered_rewards, all_disengaged_probs)

        # exact same colors/labels
        ax.scatter(all_filtered_rewards, all_disengaged_probs, color="lightblue", label="Data points")
        ax.plot(all_filtered_rewards, slope * all_filtered_rewards + intercept, color="blue", linewidth=2, label=f"Trend line")

    # exact same labeling/styling
    ax.set_xlabel("Filtered Reward", fontsize=9)
    ax.set_ylabel("P(disengaged)", fontsize=9)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(figure_dir + "correlation_reward_disengagement_first_300.pdf")
    plt.show()


