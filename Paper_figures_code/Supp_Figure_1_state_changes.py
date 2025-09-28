import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from io_utils import cross_validation_vector, model_data_glmhmm, get_mouse_info, find_change_points
from analyze_results_utils import get_file_name_for_best_model_fold, data_segmentation_session, mask_for_violations, calculate_posterior_given_data_post

sys.path.append('../')

if __name__ == '__main__':
    K = 5
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0

    animal = "churchlandlab/CSHL_014/_ibl_subjectTrials.table.61f6982a-40fb-44a6-8f2f-170235951e26.pqt"  # this animal has sessions_count= 58
    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/Supp_figures/figure_1/'
    path_analysis = '../../glm-hmm_package/results/model_indiv_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/' + 'prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(alpha_val) + '/' + animal + '/'

    cv_file = path_analysis + "/diff_folds_fit.npz"
    diff_folds_fit = cross_validation_vector(cv_file)

    with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
        optimal_initialize_dict = json.load(f)

    # Get the file name corresponding to the best initialization for a given K
    params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K,
                                                 path_analysis,
                                                 optimal_initialize_dict)
    Params_model, lls = model_data_glmhmm(params_and_LL)

    # Save parameters for initializing individual fits
    weight_vectors = Params_model[2]
    log_transition_matrix = Params_model[1][0]
    init_state_dist = Params_model[0][0]

    # Also get data for animal:
    obs_mat, trans_mat, y, session, left_probs, animal_eids = get_mouse_info(path_data + animal + '_processed.npz')
    all_sessions = np.unique(session)

    # Create mask:
    # Identify violations for exclusion:
    index_viols = np.where(y == -1)[0]
    nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])
    y[np.where(y == -1), :] = 1
    obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))
    obs_mat = obs_mat[:, [0, 1, 2, 3]]
    inputs, inputs_trans, datas, masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)

    globe = False
    posterior_probs = [calculate_posterior_given_data_post(globe, [input], [input_trans], [data], Params_model, K, alpha_val, prior_sigma, range(K)) for input, input_trans, data in zip(inputs, inputs_trans, datas)]
    states_max_posterior = [np.argmax(posterior_prob, axis=1) for posterior_prob in posterior_probs]  # list of states at each trial in session
    posterior_probs = np.asarray(posterior_probs, dtype="object")
    num_sess = np.array(posterior_probs).shape[0]

    # calculate the average sessions length
    all_len_sess = []
    for sess in range(num_sess):
        all_len_sess.append(np.array(posterior_probs[sess]).shape[0])
    trials_num = int(np.round(np.mean(all_len_sess)))
    change_points, num_sess_more_mean = find_change_points(states_max_posterior, trials_num)

    change_points_per_sess = []
    for sess in range(num_sess):
        change_points_per_sess.append(len(change_points[sess]))

    cp_bin_locs, cp_hist = np.unique(change_points_per_sess, return_counts=True)
    cp_this_group = np.zeros((1, 9))
    for i in range(cp_bin_locs.shape[0]):
        if 0 <= cp_bin_locs[i] < 10:
            j = 0
            cp_this_group[0, j] = cp_this_group[0, j] + cp_hist[i]
        if 10 <= cp_bin_locs[i] < 20:
            j = 1
            cp_this_group[0, j] = cp_this_group[0, j] + cp_hist[i]
        if 20 <= cp_bin_locs[i] < 30:
            j = 2
            cp_this_group[0, j] = cp_this_group[0, j] + cp_hist[i]
        if 30 <= cp_bin_locs[i] < 40:
            j = 3
            cp_this_group[0, j] = cp_this_group[0, j] + cp_hist[i]
        if 40 <= cp_bin_locs[i] < 50:
            j = 4
            cp_this_group[0, j] = cp_this_group[0, j] + cp_hist[i]
        if 50 <= cp_bin_locs[i] < 60:
            j = 5
            cp_this_group[0, j] = cp_this_group[0, j] + cp_hist[i]
        if 60 <= cp_bin_locs[i] < 70:
            j = 6
            cp_this_group[0, j] = cp_this_group[0, j] + cp_hist[i]
        if 70 <= cp_bin_locs[i] < 80:
            j = 7
            cp_this_group[0, j] = cp_this_group[0, j] + cp_hist[i]
        if 80 <= cp_bin_locs[i] < 90:
            j = 8
            cp_this_group[0, j] = cp_this_group[0, j] + cp_hist[i]

    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0.2, bottom=0.35, right=0.95, top=0.95)
    frac_non_zero = 0

    for z, occ in enumerate(cp_this_group[0]):
        plt.bar(z, occ / num_sess, width=.9, color="#528B8B")
        if z > 0:
            frac_non_zero += occ / num_sess

    plt.ylim((0, 0.3))
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['[0-10]', '[10-20]', '[20-30]', '[30-40]', '[40-50]', '[50-60]', '[60-70]', '[70-80]', '[80-90]'], rotation=45,  fontsize=8)
    # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], fontsize=10)
    plt.yticks([0, 0.1, 0.2, 0.3], [0, 10, 20, 30], fontsize=10)
    plt.xlabel('Number of state transitions', fontsize=9)
    plt.ylabel('Sessions Percentage', fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    fig.savefig(figure_dir + 'State_changes.pdf')


