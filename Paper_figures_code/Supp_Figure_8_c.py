from io_utils import cross_validation_vector, get_was_correct, model_data_glmhmm, get_mouse_info, colors_func, addBiasBlocks
from analyze_results_utils import get_file_name_for_best_model_fold, calculate_posterior_given_data, data_segmentation_session, mask_for_violations
import os
import matplotlib.pyplot as plt
import numpy as np
import json

if __name__ == '__main__':
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    K = 4
    cols = colors_func(K)
    length = 400

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/fig_reviews/Rev1_reward_rate_vs_prob_disengag/'
    os.makedirs(figure_dir, exist_ok=True)

    animal = "churchlandlab/CSHL_014/_ibl_subjectTrials.table.61f6982a-40fb-44a6-8f2f-170235951e26.pqt"
    path_analysis = (
        '../../glm-hmm_package/results/model_indiv_ibl/'
        f'num_regress_obs_{num_inputs}/'
        f'prior_sigma_{prior_sigma}_transition_alpha_{alpha_val}/'
        f'{animal}/'
    )
    print('path_analysis=', path_analysis)

    # load best model
    cv_file = path_analysis + "/diff_folds_fit.npz"
    diff_folds_fit = cross_validation_vector(cv_file)
    with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
        optimal_initialize_dict = json.load(f)
    params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
    Params_model, lls = model_data_glmhmm(params_and_LL)

    # load animal data
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

    # mask & posterior
    index_viols = np.where(y == -1)[0]
    nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, inputs_trans, datas, train_masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)
    perm = range(K)
    globe = False
    posterior_probs, P = calculate_posterior_given_data(
        globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma, perm
    )

    # ===== ONLY this session's yy-plot =====
    T = 500
    target_sess = "90e37c04-0da1-4bf7-9f2e-197b09c13dba"

    fig, ax1 = plt.subplots(figsize=(9, 3))
    plt.subplots_adjust(wspace=0.2, hspace=0.9, bottom=0.2)

    idx_session = np.where(session == target_sess)
    needed_trans_mat = trans_mat[idx_session[0], 2]
    posterior_probs_this_session = posterior_probs[idx_session[0], :]
    this_left_probs = left_probs[idx_session[0]]

    scatter1 = ax1.scatter(range(T), needed_trans_mat[0:T], label="Filtered reward", lw=1, color="gray", s=10)
    ax1.set_xlabel("Trial #", fontsize=10)
    ax1.set_ylabel("Filtered Reward", fontsize=10, color="gray")
    ax1.set_ylim(-1.1, 1.1)
    ax1.tick_params(axis='y', labelcolor="gray")

    ax2 = ax1.twinx()
    scatter2 = ax2.scatter(
        range(T),
        np.sum(posterior_probs_this_session[0:T, 2:3], axis=1),
        label="P(disengaged)", lw=1, color="orange", s=10
    )
    ax2.set_ylabel("P(Disengaged)", fontsize=10, color="orange")
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='y', labelcolor="orange")

    fig = addBiasBlocks(fig, this_left_probs[0:T])

    handles = [scatter1, scatter2]
    labels = ["Filtered reward", "P(disengaged)"]
    ax1.legend(handles, labels, fontsize=10)

    # Save with the same filename pattern
    fig.savefig(figure_dir + f"{target_sess}_session_yyplot.pdf")
    plt.show()


