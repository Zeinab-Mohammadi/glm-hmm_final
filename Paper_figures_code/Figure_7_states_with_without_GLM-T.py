import numpy as np
import json
import os
import matplotlib.pyplot as plt
from io_utils import get_mouse_info
from analyze_results_utils import calculate_posterior_given_data, data_segmentation_session, mask_for_violations
from io_utils import model_data_glmhmm, colors_func, cross_validation_vector
from analyze_results_utils import get_file_name_for_best_model_fold
from Review_utils import load_data, get_marginal_posterior, load_glmhmm_data, load_cv_arr, get_file_name_for_best_model_fold_GLM_O


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
    os.makedirs(figure_dir, exist_ok=True)

    animal = "churchlandlab/CSHL_014/_ibl_subjectTrials.table.61f6982a-40fb-44a6-8f2f-170235951e26.pqt"
    path_that_animal = path_of_the_directory + animal
    path_analysis = (
        '../../glm-hmm_package/results/model_indiv_ibl/'
        f'num_regress_obs_{num_inputs}/'
        f'prior_sigma_{prior_sigma}_transition_alpha_{alpha_val}/'
        f'{animal}/'
    )

    cv_file = path_analysis + "/diff_folds_fit.npz"
    diff_folds_fit = cross_validation_vector(cv_file)

    with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
        optimal_initialize_dict = json.load(f)

    # best init with GLM-T
    params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
    Params_model, lls = model_data_glmhmm(params_and_LL)

    # data (GLM-T path)
    obs_mat, trans_mat, y, session, left_probs, animal_eids = get_mouse_info(path_data + animal + '_processed.npz')
    obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))
    all_sessions = np.unique(session)

    # drop violations & keep first 4 regressors
    y_init_size = y.shape[0]
    not_viols = np.where(y != -1)
    y = y[not_viols[0], :]
    obs_mat = obs_mat[not_viols[0], :]
    session = session[not_viols[0]]
    left_probs = left_probs[not_viols[0]]
    obs_mat = obs_mat[:, [0, 1, 2, 3]]

    # mask & posterior (GLM-T)
    index_viols = np.where(y == -1)[0]
    _, mask = mask_for_violations(index_viols, obs_mat.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, inputs_trans, datas, train_masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)
    perm = range(K)
    globe = False
    posterior_probs, P = calculate_posterior_given_data(
        globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma, perm
    )

    #-------------------------- posterior for no GLM-T model -----------------------#
    covar_set = 2
    results_dir = (
        '/Users/zm6112/Dropbox/Python_code/Pycharm_Z_code_github/glm-hmm_all_no_GLM-T_to_compare/results/'
        'ibl_individual_fit/' + f'covar_set_{covar_set}/' +
        f'prior_sigma_{prior_sigma}_transition_alpha_{alpha_val}/' + f'{animal}/'
    )

    cv_file = results_dir + "/cvbt_folds_model.npz"
    cvbt_folds_model = load_cv_arr(cv_file)

    with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)

    raw_file, = (get_file_name_for_best_model_fold_GLM_O(cvbt_folds_model, K, results_dir, best_init_cvbt_dict),)
    hmm_params, lls = load_glmhmm_data(raw_file)

    # data (no GLM-T path)
    inpt, inpt_trans, y2, session2, left_probs2, animal_eids2 = load_data(path_data + animal + '_processed.npz')
    inpt = np.hstack((inpt, np.ones((inpt.shape[0], 1))))

    not_viols2 = np.where(y2 != -1)
    y2 = y2[not_viols2[0], :]
    inpt = inpt[not_viols2[0], :]
    session2 = session2[not_viols2[0]]
    left_probs2 = left_probs2[not_viols2[0]]
    inpt = inpt[:, [0, 1, 2, 3]]

    violation_idx = np.where(y2 == -1)[0]
    _, mask2 = mask_for_violations(violation_idx, inpt.shape[0])
    y2[np.where(y2 == -1), :] = 1
    inputs2, inputs_trans2, datas2, train_masks2 = data_segmentation_session(inpt, inpt_trans, y2, mask2, session2)

    permutation = range(K)
    posterior_probs_no_GLM_T = get_marginal_posterior(
        globe, inputs2, datas2, hmm_params, K, permutation, alpha_val, prior_sigma
    )

    # The sessions we compare
    sess_to_plot = ["dab3d729-8983-4431-9d88-0640b8ba4fdd", "90e37c04-0da1-4bf7-9f2e-197b09c13dba",
                    "7d824d9b-50fc-449e-8ebb-ee7b3844df18"]

    #------------------------ ONLY: plot states prob versus each other ---------------------#
    max_posterior_no_GLM_T = []
    max_posterior = []

    for i, sess in enumerate(sess_to_plot):
        idx_session = np.where(session == sess)
        posterior_probs_this_session_no_GLLM_T = posterior_probs_no_GLM_T[idx_session[0], :]
        posterior_probs_needed_this_session = posterior_probs[idx_session[0], :]

        max_values_no_GLM_T = np.max(posterior_probs_this_session_no_GLLM_T, axis=1)
        max_values_GLM_T = np.max(posterior_probs_needed_this_session, axis=1)

        # keep only if both > 0.6
        valid_indices = (max_values_no_GLM_T > 0.6) & (max_values_GLM_T > 0.6)
        max_posterior_no_GLM_T.extend(max_values_no_GLM_T[valid_indices])
        max_posterior.extend(max_values_GLM_T[valid_indices])

    max_posterior_no_GLM_T = np.array(max_posterior_no_GLM_T)
    max_posterior = np.array(max_posterior)


    fig, ax = plt.subplots(figsize=(3, 3))
    plt.scatter(max_posterior_no_GLM_T, max_posterior, color="cornflowerblue")
    plt.subplots_adjust(hspace=0.4)
    plt.subplots_adjust(left=0.2, bottom=0.27)
    plt.plot([min(max_posterior_no_GLM_T), max(max_posterior_no_GLM_T)],
             [min(max_posterior),            max(max_posterior)],
             color="red", linestyle="--", label="y = x")

    plt.xlabel("States (No GLM-T)", fontsize=9)
    plt.ylabel("States (With GLM-T)", fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(fontsize=9)

    fig.savefig(figure_dir + 'probs_versus' + '.pdf')
    plt.show()



