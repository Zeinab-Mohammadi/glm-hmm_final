import numpy as np
import json
import matplotlib.pyplot as plt
from io_utils import cross_validation_vector, get_was_correct, model_data_glmhmm, get_mouse_info, colors_func, addBiasBlocks
from analyze_results_utils import get_file_name_for_best_model_fold, calculate_posterior_given_data, data_segmentation_session, \
    mask_for_violations

if __name__ == '__main__':
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    K = 4
    cols = colors_func(K)
    length = 400

    # store exactly-what-we-plotted so Excel matches the figure
    rng = np.random.default_rng(12345)  # reproducible jitter for choices

    noisy_choice_by_session = {}
    correct_idx_by_session = {}
    incorrect_idx_by_session = {}
    posterior_by_session = {}

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/figure_4/'

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
    A = np.array(inputs, dtype=object).shape  # -> (n_sessions,)
    posterior_probs, P = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma,
                                                perm)

    # sessions
    sess_to_plot = ["dab3d729-8983-4431-9d88-0640b8ba4fdd", "90e37c04-0da1-4bf7-9f2e-197b09c13dba",
                    "7d824d9b-50fc-449e-8ebb-ee7b3844df18"]
    states_max_posterior = np.argmax(posterior_probs, axis=1)
    tt = 0
    fig = plt.figure(figsize=(9, 8.8))
    plt.subplots_adjust(hspace=0.3)
    for i, sess in enumerate(sess_to_plot):
        plt.subplot(6, 1, (2 * i + 2))
        idx_session = np.where(session == sess)
        needed_obs_mat = obs_mat[idx_session[0], :]
        posterior_probs_needed_this_session = posterior_probs[idx_session[0], :]
        posterior_by_session[sess] = posterior_probs_needed_this_session.copy()
        this_left_probs = left_probs[idx_session[0]]
        # Plot trial structure for this session too:
        for k in range(K):
            plt.plot(posterior_probs_needed_this_session[:, k],
                     label="State " + str(k + 1), lw=1,
                     color=cols[k])
            Prob_left = this_left_probs
            fig = addBiasBlocks(fig, Prob_left)
        states_this_sess = states_max_posterior[idx_session[0]]
        state_change_locs = np.where(np.abs(np.diff(states_this_sess)) > 0)[0]
        if i == 0:  # for first session
            # plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], fontsize=10)
            plt.yticks([0, 1], ["0", "1"], fontsize=10)
        else:
            # plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], ["", "", "", "", "", "", "", "", ""], fontsize=10)
            plt.yticks([0, 1], ["", ""], fontsize=10)


        plt.ylim((-0.01, 1.01))
        plt.xlim((0, 400))
        plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], ["", "", "", "", "", "", "", "", ""], fontsize=10)
        # plt.title("example session " + str(i + 1), fontsize=10)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        if i == 0:  # for first session
            # plt.xlabel("trial #", fontsize=10)
            plt.ylabel("p(state)", fontsize=10)

    for i, sess in enumerate(sess_to_plot):
        plt.subplot(6, 1, (2 * i + 1))
        idx_session = np.where(session == sess)
        needed_obs_mat, this_y = obs_mat[idx_session[0], :], y[idx_session[0], :]
        was_correct, idx_easy = get_was_correct(needed_obs_mat, this_y)

        # --- make and store the jittered choices exactly like the plot ---
        this_y_base = this_y[:, 0].astype(float)
        this_y_noisy = this_y_base + rng.normal(0, 0.03, len(this_y_base))

        # indices for correct/incorrect
        locs_correct = np.where(was_correct == 1)[0]
        locs_incorrect = np.where(was_correct == 0)[0]

        # store for Excel export if needed
        noisy_choice_by_session[sess] = this_y_noisy
        correct_idx_by_session[sess] = locs_correct
        incorrect_idx_by_session[sess] = locs_incorrect

        plt.plot(locs_correct[0:length], this_y_noisy[locs_correct][0:length], 'o',
                 color='gray', markersize=2, zorder=3, alpha=0.5, label="Correct")
        plt.plot(locs_incorrect[0:length], this_y_noisy[locs_incorrect][0:length], 'o',
                 color='orange', markersize=2, zorder=4, alpha=0.5, label="Incorrect")

        if i in [0, 1, 2]:
            tt += 1
            plt.title(f'Example session {tt}', fontsize=10)
        states_this_sess = states_max_posterior[idx_session[0]]
        state_change_locs = np.where(np.abs(np.diff(states_this_sess)) > 0)[0]
        for change_loc in state_change_locs:
            plt.axvline(x=change_loc, color='k', lw=0.5, linestyle='--')
        plt.ylim((-0.13, 1.13))
        plt.xlim((0, 400))
        if i == 0:  # for first session
            plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], fontsize=8)
            plt.yticks([0, 1], ["L", "R"], fontsize=10)
        else:
            plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], ["", "", "", "", "", "", "", "", ""], fontsize=10)
            plt.yticks([0, 1], ["", ""], fontsize=10)
        # plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], ["", "", "", "", "", "", "", "", ""], fontsize=10)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        # plt.title("example session " + str(i + 1), fontsize=10)
        if i == 0:  # for first session
            # plt.xlabel("trial #", fontsize=8)
            plt.ylabel("choice", fontsize=10)
            plt.legend(fontsize=8, loc="lower right")

    fig.savefig(figure_dir + 'Posteriors_correct_incorrect_plot' + '.pdf')

