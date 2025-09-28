"""

Fit the global GLM to all pooled IBL data from all mice in the decision-making task

"""


import autograd.numpy as np
import autograd.numpy.random as npr
import sys
import os

sys.path.append("../")
from utils_for_GLM import load_fold_session_map, get_mouse_info, fit_glm, regressors_weights_Figure, add_zero_column



if __name__ == '__main__':
    cross_valid_num_fold = 5  # number of folds for the cross-validation
    C = 2  # number of output types/categories
    model_init_num = 10  # number of initializations
    num_inputs = 4  # number of observation regressors

    npr.seed(0)
    # Use all pooled IBL data for fitting the model
    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/'
    path_main_folder = '../../glm-hmm_package/results/model_global_ibl/' \
                  + 'num_regress_obs_' + str(num_inputs) + '/'

    mouse_data = path_data + 'combined_all_mice.npz'
    fold_mapping_session = load_fold_session_map(path_data + 'fold_mapping_session_all_mice.npz')

    for fold in range(cross_valid_num_fold):
        print('fold=', fold)
        obs_mat, trans_mat, output, session = get_mouse_info(mouse_data)

        # preparing appropriate variables
        indexes = range(0, [obs_mat.shape[1] - 1][0], 1)
        obs_mat = obs_mat[:, indexes]
        figure_covariates_names = ['stim', 'pc_a', 'pc_b', 'stim_side', 'bias']
        output = output.astype('int')

        # making figure directory
        figure_directory = path_main_folder + "Model/glm_#state=1/fld_num=" + str(fold) + '/'
        if not os.path.exists(figure_directory):
            os.makedirs(figure_directory)

        # determining the needed sessions based on the fold value
        appropriate_sessions = fold_mapping_session[np.where(fold_mapping_session[:, 1] != fold), 0]
        indexes_sess_fold = [str(sess) in appropriate_sessions and output[id, 0] != -1 for id, sess in enumerate(session)]
        needed_obs_mat, needed_trans_mat, needed_this_output, needed_this_session = obs_mat[indexes_sess_fold, :], trans_mat[indexes_sess_fold, :], output[
                                                                                                                 indexes_sess_fold,
                                                                                                                 :], session[indexes_sess_fold]
        assert len(np.unique(needed_this_output)) == 2, "Mouse decision is binary, and it cannot have more than two values"
        train_size = needed_obs_mat.shape[0]

        M_obs = needed_obs_mat.shape[1]  # number of observation  covariates
        M_trans = needed_trans_mat.shape[1]  # number of transition covariates
        train_LL_values = []

        for iter in range(model_init_num):
            print('iter=', iter)
            train_calculate_LL, recovered_weights, standard_deviation, temporal_probabilities = fit_glm([needed_obs_mat],
                                                                                                          [needed_this_output],
                                                                                                          M_obs)
            weights_for_plotting = add_zero_column(recovered_weights)
            regressors_weights_Figure(weights_for_plotting, standard_deviation, figure_directory,
                               title="GLM fit; Final LL = " + str(train_calculate_LL), save_title='init' + str(iter),
                               figure_covariates_names=figure_covariates_names)

            train_LL_values.append(train_calculate_LL)
            print('figure_directory=', figure_directory)
            np.savez(figure_directory + 'important_params_iter_' + str(iter) + '.npz', train_calculate_LL,
                     recovered_weights, standard_deviation, temporal_probabilities)
