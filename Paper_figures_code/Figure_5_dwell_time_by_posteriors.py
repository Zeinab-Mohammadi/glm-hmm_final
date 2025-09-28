import numpy as np
import json
import matplotlib.pyplot as plt
from io_utils import cross_validation_vector, model_data_glmhmm, get_mouse_info, colors_func, dwell_time
from analyze_results_utils import get_file_name_for_best_model_fold, calculate_posterior_given_data, data_segmentation_session, mask_for_violations

if __name__ == '__main__':
    not_viols_ratio = []
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    K = 4
    cols = colors_func(K)

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/figure_5/'
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

    y_init_size = y.shape[0]
    not_viols = np.where(y != -1)
    not_viols_size = y[not_viols].shape[0]
    y = y[not_viols[0],:]
    obs_mat = obs_mat[not_viols[0],:]
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
    states_max_posterior = np.argmax(posterior_probs, axis=1)
    dwell_all = dwell_time(states_max_posterior, K)

    # plotting the dwell times for different states
    fig = plt.figure(figsize=(4.5, 3.7))  # width,
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.2, wspace=0.3, hspace=0.22)

    for k in range(K):
        plt.subplot(2, 2, k+1)
        plt.hist(dwell_all[k],
                 bins=np.arange(0, 60, 5),
                 color=cols[k],
                 histtype='bar',
                 rwidth=0.8)
        plt.xticks([0, 20, 40, 60], ["", "", "", ""], fontsize=10)
        a = [0,  100, 200, 300]
        b = [0, 30, 60, 90, 120]
        if k == 0:
            plt.xticks([0, 20, 40, 60], ["0", "20", "40", "60"], fontsize=8)
            plt.yticks(b, fontsize=8)
            plt.ylabel("Event count", fontsize=10)
            # plt.xlabel("trial", fontsize=8)
            plt.axvline(np.median(dwell_all[k]),
                        linestyle='-',
                        color='k',
                        lw=1,
                        label='median')
            plt.legend(fontsize=8,
                       loc='upper right',
                       markerscale=1)
            plt.ylim(0, 150)
        if k == 1:
            plt.yticks(a, [0, 100, 200, 300], fontsize=8)
            plt.ylim(0, 350)

        if k == 2:
            plt.yticks(b, fontsize=8)
            plt.ylim(0, 150)
        if k == 3:
            plt.yticks(b, fontsize=8)
            plt.ylim(0, 150)
        plt.axvline(np.median(dwell_all[k]),
                    linestyle='-',
                    color='k',
                    lw=1,
                    label='median')
    fig.savefig(figure_dir + 'Dwell_times.pdf')




