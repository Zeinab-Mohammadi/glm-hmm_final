import numpy as np
import json
import matplotlib.pyplot as plt
from io_utils import cross_validation_vector, get_mouse_info, mice_names_info, model_data_glmhmm
from analyze_results_utils import get_file_name_for_best_model_fold, calculate_posterior_given_data, data_segmentation_session, mask_for_violations


if __name__ == '__main__':
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    K = 4
    length = 400

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/figure_1/'
    mice_names = mice_names_info(path_data + 'mice_names.npz')

    not_viols_ratio = []
    all_sessions_length = []
    block_length_all_mean = []
    block_length_all_min = []
    all_animals_diff = []

    plt.rcParams['pdf.fonttype'] = 42  # keep text editable in PDF
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'  # keep text editable in SVG
    plt.rcParams['text.usetex'] = False  # avoid TeX outlining

    for z, animal in enumerate(mice_names):
        all_t = []
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
        globe = False
        posterior_probs, P = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma, range(K))

        for i, sess in enumerate(all_sessions):
            idx_session = np.where(session == sess)
            posterior_probs_needed_this_session = posterior_probs[idx_session[0], :]
            all_sessions_length.append(len(posterior_probs_needed_this_session))

        # determine minimum length of biased blocks
        left_probs_bias_trial = np.where((left_probs == .2) | (left_probs == .8))[0]
        switch_trials = []
        i = 0
        all_diff = []
        min_len = 20

        for t in range(left_probs_bias_trial.shape[0]):  # finding bias blocks start trials
            if left_probs[left_probs_bias_trial[t]] != left_probs[left_probs_bias_trial[t - 1]]:
                switch_trials.append(left_probs_bias_trial[t])  # trials when bias switch happens
                if t != 0:
                    all_t.append(t)
                    if np.array(all_t).shape[0] > 1:
                        if all_t[i + 1] - all_t[i] >= min_len:
                            all_diff.append(all_t[i + 1] - all_t[i])
                            all_animals_diff.append(all_t[i + 1] - all_t[i])
                            i += 1
        block_length_all_min.append(np.min(all_diff))
        block_length_all_mean.append(np.mean(all_diff))


# plot number of blocks sizes more than n
all_animals_diff_unique = np.unique(all_animals_diff)
count = []
all_animals_diff_unique_all = all_animals_diff_unique

for i in all_animals_diff_unique:
    if i == 20:  # this is the first element
        for j in range(i):  # this is to show the flat line before 20
            count.append(np.sum(np.array(all_animals_diff) > i))
            all_animals_diff_unique_all = np.append(j, all_animals_diff_unique_all)
    count.append(np.sum(np.array(all_animals_diff) > i))

fig = plt.figure(figsize=(4.5, 2.5))
plt.subplots_adjust(left=0.2, bottom=0.2)  # right=0.92, top=0.95)
plt.xlim((0, 110))
plt.title('Block length distribution', fontsize=10)
plt.ylabel('Blocks length > n (%)', fontsize=10)
plt.plot(all_animals_diff_unique_all, (np.array(count)*100)/np.max(count), 'b', lw=1.4)
plt.xlabel('n', fontsize=10)
plt.xticks([0, 15, 30, 45, 60, 75, 90, 105], fontsize=10)
plt.grid()
fig.savefig(figure_dir + 'num_blocks_more_than_n.pdf')


