"""

Fit the individual GLM to each mouse's data separately in the decision-making task.

"""

import autograd.numpy as np
import autograd.numpy.random as npr
import sys
import os
sys.path.append("/")
npr.seed(10)

from utils_for_GLM import load_fold_session_map, get_mouse_info, mice_names_info, fit_glm, regressors_weights_Figure, add_zero_column


if __name__ == '__main__':
    cross_valid_num_fold = 5
    num_inputs = 4
    C = 2
    model_init_num = 10
    transition_alphas = [2.0]
    prior_sigmas = [4.0] #[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    mice_names = mice_names_info(path_data + 'mice_names.npz')

    for prior_sigma in prior_sigmas:
        for transition_alpha in transition_alphas:
            for animal in mice_names:
                # GLM for individual fit
                mouse_data = path_data + animal + '_processed.npz'
                fold_mapping_session = load_fold_session_map(path_data + animal + '_fold_session_map.npz')

                for fold in range(cross_valid_num_fold):
                    path_main_folder = '../../glm-hmm_package/results/model_indiv_ibl/' + 'num_regress_obs_' + str(
                        num_inputs) + '/prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(transition_alpha) + '/' + animal + '/'

                    # get the appropriate data
                    obs_mat, trans_mat, y, session = get_mouse_info(mouse_data)
                    indexes = range(0, [obs_mat.shape[1] - 1][0], 1)
                    obs_mat = obs_mat[:, indexes]
                    figure_covariates_names = ['stim', 'pc_a', 'pc_b', 'stim_side', 'bias']

                    figure_directory = path_main_folder + "Model/glm_#state=1/fld_num=" + str(fold) + '/'
                    if not os.path.exists(figure_directory):
                        os.makedirs(figure_directory)

                    # Narrow down the focus to the specific sessions relevant to the selected fold
                    appropriate_sessions = fold_mapping_session[np.where(fold_mapping_session[:, 1] != fold), 0]
                    indexes_sess_fold = [str(sess) in appropriate_sessions and y[id, 0] != -1 for id, sess in enumerate(session)]
                    needed_obs_mat, needed_this_output, needed_this_session = obs_mat[indexes_sess_fold, :], y[indexes_sess_fold, :], session[indexes_sess_fold]
                    assert len(np.unique(needed_this_output)) == 2, "choice vector should only include 2 possible values"
                    train_size = needed_obs_mat.shape[0]

                    M_obs = needed_obs_mat.shape[1]
                    train_LL_values = []
                    for iter in range(model_init_num):
                        train_calculate_LL, recovered_weights, standard_deviation, temporal_probabilities = fit_glm([needed_obs_mat],
                                                                                                                      [needed_this_output],
                                                                                                                      M_obs)
                        weights_for_plotting = add_zero_column(recovered_weights)
                        regressors_weights_Figure(weights_for_plotting, standard_deviation,  figure_directory,
                                           title="GLM fit; Final LL = " + str(train_calculate_LL), save_title='init' + str(iter), figure_covariates_names=figure_covariates_names)
                        train_LL_values.append(train_calculate_LL)
                        np.savez(figure_directory + 'important_params_iter_' + str(iter) + '.npz', train_calculate_LL,
                                 recovered_weights, standard_deviation, temporal_probabilities)
