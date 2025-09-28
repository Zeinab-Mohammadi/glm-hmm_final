"""

Plot weights and inputs multiplication (individual fit) for different sessions, with a background color indicating biased blocks.

"""

import numpy as np
import json
import copy
import matplotlib.pyplot as plt
from analyze_results_utils import get_file_name_for_best_model_fold, data_segmentation_session, mask_for_violations, \
    cross_validation_vector, model_data_glmhmm
from io_utils import get_mouse_info, mice_names_info, colors_func, addBiasBlocks

if __name__ == '__main__':
    all_size_posterior = 0
    unique_sessions_all = 0
    prior_sigma = 4.0
    past_sessions = 0
    K = 4  # number of states
    anumal_num = 0
    not_viols_ratio = []
    trans_p_posterior = []

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    mice_names = mice_names_info(path_data + 'mice_names.npz')
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/figure_5/'

    trans_log_p_posterior_all_animals = np.zeros((mice_names.shape[0], K, K))
    sess_to_plot = ["dab3d729-8983-4431-9d88-0640b8ba4fdd", "90e37c04-0da1-4bf7-9f2e-197b09c13dba",
                    "7d824d9b-50fc-449e-8ebb-ee7b3844df18"]
    animal = "churchlandlab/CSHL_014/_ibl_subjectTrials.table.61f6982a-40fb-44a6-8f2f-170235951e26.pqt"
    fig = plt.figure(figsize=(9, 8.5))
    plt.subplots_adjust(hspace=0.25)

    anumal_num += 1
    path_that_animal = path_of_the_directory + animal
    for num_inputs in [4]:
        for alpha_val in [2.0]:
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
            log_transition_matrix = Params_model[1][0]
            init_state_dist = Params_model[0][0]
            trans_weight_append_zero = np.vstack((Params_model[1][1], np.zeros((1, Params_model[1][1].shape[1]))))

            # Standardize the plotted GLM transition weights
            trans_weight_append_zero_standard = copy.deepcopy(trans_weight_append_zero)
            v1 = - np.mean(trans_weight_append_zero, axis=0)  # here v1 instead of w1=0
            trans_weight_append_zero_standard[-1, :] = v1

            for i in range(K - 1):
                trans_weight_append_zero_standard[i, :] = v1 + trans_weight_append_zero[i, :]  # vi = v1 + wi
            weight_vectors_trans = trans_weight_append_zero_standard

            # Also get data for animal:
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
            unique_sessions = np.unique(session, return_index=True)[1]

            for i, sess in enumerate(sess_to_plot):
                plt.subplot(3, 1, i+1)
                idx_session = np.where(session == sess)
                needed_obs_mat = obs_mat[idx_session[0], :]
                needed_trans_mat = trans_mat[idx_session[0], :]
                this_left_probs = left_probs[idx_session[0]]
                Weights_multi_inputs = np.zeros((np.array(needed_trans_mat).shape[0], K))
                Weights_multi_inputs = np.dot(weight_vectors_trans, needed_trans_mat.T)
                plt.xlim((0, 400))
                if i == 0:
                    plt.ylabel("Transition weights * input", fontsize=10)
                    plt.xlabel("trial", fontsize=10)
                # Plot trial structure for this session
                for k in range(K):
                    cols = colors_func(K)
                    plt.plot(Weights_multi_inputs[k, :], lw=2, color=cols[k])
                    Prob_left = this_left_probs
                    fig = addBiasBlocks(fig, Prob_left)

            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            fig.savefig(figure_dir + 'WX_sessions' + '.pdf')





