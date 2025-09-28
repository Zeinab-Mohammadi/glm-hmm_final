import numpy as np
import json
import matplotlib.pyplot as plt
from io_utils import cross_validation_vector, model_data_glmhmm, get_mouse_info, mice_names_info
from analyze_results_utils import get_file_name_for_best_model_fold, get_global_weights, find_corresponding_states_new, \
    calculate_posterior_given_data, \
    data_segmentation_session, mask_for_violations

if __name__ == '__main__':
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    K = 4
    trial_diff_all = []
    trial_diff_until_end = []

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/figure_4/'
    global_directory = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/'

    Params_model_glob = get_global_weights(global_directory, K)
    global_weights = -Params_model_glob[2]
    label = find_corresponding_states_new(Params_model_glob)
    Bias_L = [label[0], label[2]]
    Bias_R = [label[1], label[3]]
    mice_names = mice_names_info(path_data + 'mice_names.npz')

    for z, animal in enumerate(mice_names):
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
        all_sessions = np.unique(session)
        y_init_size = y.shape[0]
        not_viols = np.where(y != -1)
        not_viols_size = y[not_viols].shape[0]
        y = y[not_viols[0], :]
        obs_mat = obs_mat[not_viols[0], :]
        session = session[not_viols[0]]
        left_probs = left_probs[not_viols[0]]

        left_probs_bias_trial = np.where((left_probs == .2) | (left_probs == .8))[0]
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
                                                    prior_sigma, perm)

        switch_trials = []
        for t in range(left_probs_bias_trial.shape[0]):  # finding bias blocks start points/trials
            if left_probs[left_probs_bias_trial[t]] != left_probs[left_probs_bias_trial[t - 1]]:
                switch_trials.append(left_probs_bias_trial[t])  # this is trials when bias switch happens

        trial_diff = []
        states_max_posterior = np.argmax(posterior_probs, axis=1)
        states_max_posterior_bias1 = np.where((states_max_posterior == Bias_L[1]) | (states_max_posterior == Bias_L[0]),
                                              K + 1, states_max_posterior)  # Everywhere with Bias_L values, write K+1 for later comparison. Thus, we still have Bias_R values and Bias_L == K+1 values.
        states_max_posterior_bias = np.where(
            (states_max_posterior_bias1 == Bias_R[1]) | (states_max_posterior_bias1 == Bias_R[0]), K + 2,
            states_max_posterior_bias1)  # Everywhere with Bias_R values, write K+2 for later comparison. Thus, we still have Bias_L values and Bias_R == K+2 values.

        states_max_posterior_bias = states_max_posterior_bias - (K + 1)  # Minus K+1, so now we only have 0 and 1 to represent Bias_L locations and Bias_R locations, respectively.
        left_prob_higher_indx = np.where(states_max_posterior_bias == 0)
        right_prob_higher_indx = np.where(states_max_posterior_bias == 1)

        T_pre = 5
        for t in switch_trials:
            if left_probs[t] == 0.8:  # means we are in left block
                a = left_prob_higher_indx[0] - t  # left_prob_higher_indx happens after switching time = t
                if a[np.where(a > 0)].shape[0] > 0:  # so it has elements
                    pos_values = np.min(a[np.where(a > 0)])
                    pos_index = t + pos_values
                if np.all(states_max_posterior_bias[(
                        pos_index - T_pre): pos_index] == 1):  # 1 so transition from right to left
                    # if 50 > np.array(pos_values) > 21:
                    #     pos_values = pos_values - (pos_values/3)
                    trial_diff.append(pos_values)
            elif left_probs[t] == 0.2:
                a = right_prob_higher_indx[0] - t
                if a[np.where(a > 0)].shape[0] > 0:  # so it has elements
                    pos_values = np.min(a[np.where(a > 0)])
                    pos_index = t + pos_values
                # if pos_values != []:
                if np.all(states_max_posterior_bias[(
                        pos_index - T_pre): pos_index] == 0):  # means previously it had 1 so transition from right to left
                    # if 50 > np.array(pos_values) > 21:
                    #     pos_values = pos_values - (pos_values/3)
                    trial_diff.append(pos_values)

        max_block_length = 100
        trial_diff0 = np.array(trial_diff)[np.where(np.array(trial_diff) < max_block_length)]
        if z == 0:
            trial_diff_until_end = trial_diff0
        else:
            trial_diff_until_end = np.concatenate((trial_diff_until_end, trial_diff0))

        trial_diff = np.array(trial_diff)[
            np.where(np.array(trial_diff) < 21)]
        if z == 0:
            trial_diff_all = trial_diff
        else:
            trial_diff_all = np.concatenate((trial_diff_all, trial_diff))

    median_value = np.median(trial_diff_all)
    max_value = np.max(trial_diff_all)

    trial_diff_final = trial_diff_all
    fig = plt.figure(figsize=(3.4, 2.1))
    plt.subplots_adjust(left=0.2, bottom=0.2)

    bins = range(24)
    # Compute the normalized histogram
    counts, bin_edges = np.histogram(trial_diff_all, bins=bins, density=False)  # Get raw counts
    normalized_counts = counts / np.sum(counts)  # Normalize so sum is 1

    # Plot histogram with normalized values
    plt.bar(bin_edges[:-1], normalized_counts, width=0.8, color='green', align='edge')

    plt.yticks(fontsize=9)
    plt.xlim((0, 21))
    plt.ylabel('First switching prob.', fontsize=9)  # Adjust label
    plt.xlabel('Trial', fontsize=9)
    plt.xticks(range(0, 22, 2), fontsize=9)
    plt.yticks(np.arange(0, 0.08, 0.02), fontsize=9)
    plt.axvline(np.median(trial_diff_final),
                linestyle='--',
                color='k',
                lw=1.4,
                label='median')
    max = 6
    plt.axvline(max,
                linestyle='-.',
                color='k',
                lw=1.4,
                label='maximum')

    plt.legend(fontsize=8)
    plt.yticks(np.arange(0, 0.09, 0.02), fontsize=9)
    plt.title('Model with GLM-T', fontsize=9)
    plt.show()
    fig.savefig(figure_dir + 'normalized_bias_blocks_switching_all_animals.pdf')

