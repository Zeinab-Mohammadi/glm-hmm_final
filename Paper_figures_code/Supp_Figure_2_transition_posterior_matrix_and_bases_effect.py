import numpy as np
import json
import matplotlib.pyplot as plt
import copy
from io_utils import cross_validation_vector, model_data_glmhmm, get_mouse_info, mice_names_info
from analyze_results_utils import  get_file_name_for_best_model_fold, permute_transition_matrix, find_corresponding_states_new, calculate_posterior_given_data, data_segmentation_session, mask_for_violations

if __name__ == '__main__':
    K = 5
    prior_sigma = 2
    num_inputs = 4
    alpha_val = 2
    past_sessions = 0
    all_size_posterior = 0
    unique_sessions_all = 0
    trans_p_posterior = []
    not_viols_ratio = []

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    mice_names = mice_names_info(path_data + 'mice_names.npz')
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'

    trans_log_p_posterior_all_animals = np.zeros((mice_names.shape[0], K, K))
    for animal in mice_names:
        path_that_animal = path_of_the_directory + animal
        path_analysis = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(num_inputs) + '/'
        figure_dir = '../../glm-hmm_package/results/figures_for_paper/Supp_figures/figure_2/'
        cv_file = path_analysis + "/diff_folds_fit.npz"
        diff_folds_fit = cross_validation_vector(cv_file)
        with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
            optimal_initialize_dict = json.load(f)

        # Get the file name corresponding to the best initialization for given K value
        params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
        Params_model, lls = model_data_glmhmm(params_and_LL)
        perm = find_corresponding_states_new(Params_model)

        # Save parameters for initializing individual fits
        weight_vectors = Params_model[2][perm]
        log_transition_matrix = permute_transition_matrix(Params_model[1][0], perm)
        init_state_dist = Params_model[0][0][perm]

        # standardize the plotted GLM transition weights
        trans_weight_append_zero = np.vstack((Params_model[1][1], np.zeros(
            (1, Params_model[1][1].shape[1]))))  # have k states instead of k-1 states
        trans_weight_append_zero_standard = copy.deepcopy(trans_weight_append_zero)
        v1 = - np.mean(trans_weight_append_zero, axis=0)  # here is v1 instead of w1=0
        trans_weight_append_zero_standard[-1, :] = v1
        for i in range(K - 1):
            trans_weight_append_zero_standard[i, :] = v1 + trans_weight_append_zero[i, :]  # vi = v1 + wi
        weight_vectors_trans = trans_weight_append_zero_standard[perm]

        # get data for animal:
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
        globe = True
        posterior_probs, P = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma, range(K))
        unique_sessions = np.unique(session, return_index=True)[1]
        sess_to_plot = [session[index] for index in sorted(unique_sessions)]

        # plot figure of multiplying the P_matrix and posteriors
        P_st = 0
        for i, sess in enumerate(sess_to_plot):
            idx_session = np.where(session == sess)
            needed_obs_mat = obs_mat[idx_session[0], :]
            this_left_probs = left_probs[idx_session[0]]
            posterior_probs_needed_this_session = posterior_probs[idx_session[0], :]
            P_needed_this_session = P[P_st:(P_st + idx_session[0].shape[0] - 1), :]
            P_st = idx_session[0].shape[0] - 1

            # Plot trial structure for this session
            for p in range (P_needed_this_session.shape[0]):
                trans_K_K = np.zeros((K, K))
                for k in range(K):
                    trans_K_K[k, :] = P_needed_this_session[p, k, :] * posterior_probs_needed_this_session[p, :]
                trans_p_posterior.append(trans_K_K)  # this is all K by K for all trials (sessions and animals)

    trans_p_posterior_mean = permute_transition_matrix(np.mean(trans_p_posterior, axis =0), perm)
    trans_p_posterior_mean_normalize = np.zeros((K, K))

    for k in range(K):
        trans_p_posterior_mean_normalize[k, :] = trans_p_posterior_mean[k, :]/np.sum(trans_p_posterior_mean[k, :])
    fig = plt.figure(figsize=(2.7, 2.7))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.imshow(trans_p_posterior_mean_normalize.T, vmin=-.1, vmax=1, cmap="pink", aspect=1)
    for i in range(trans_p_posterior_mean_normalize.shape[0]):
        for j in range(trans_p_posterior_mean_normalize.shape[1]):
            text = plt.text(i, j, np.around(trans_p_posterior_mean_normalize.T[i, j], decimals=2),
                            ha="center", va="center", color="k", fontsize=9)

    plt.ylabel("Previous State", fontsize=10)
    plt.xlabel("Next State", fontsize=10)
    plt.xlim(-0.5, K - 0.5)
    plt.ylim(-0.5, K - 0.5)
    plt.xticks(range(0, K), ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')[:K], fontsize=10)
    plt.yticks(range(0, K), ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')[:K], fontsize=10)
    fig.savefig(figure_dir + '/transition_Posterior' + '.pdf')

    # plot transition matrix parameters
    fig = plt.figure(figsize=(2.7, 2.7))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.imshow(log_transition_matrix, cmap="Blues", vmin=-5.45, vmax=1,
               aspect=1)
    for i in range(log_transition_matrix.shape[0]):
        for j in range(log_transition_matrix.shape[1]):
            text = plt.text(i, j, np.around(log_transition_matrix[i, j], decimals=2),
                            ha="center", va="center", color="k", fontsize=9)
    plt.ylabel("Previous State", fontsize=10)
    plt.xlabel("Next State", fontsize=10)
    plt.xlim(-0.5, K - 0.5)
    plt.ylim(-0.5, K - 0.5)
    plt.xticks(range(0, K), ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')[:K], fontsize=10)
    plt.yticks(range(0, K), ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')[:K], fontsize=10)
    fig.savefig(figure_dir + '/bias_transition_matrix' + '.pdf')




