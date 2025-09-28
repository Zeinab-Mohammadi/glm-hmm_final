"""

Fit the individual GLM-HMM to each mouse's data separately in the decision-making task.

"""
import autograd.numpy as np
import sys
from utils_for_GLMHMM import get_cluster_info, load_fold_session_map, get_mouse_info, mice_names_info, \
    mask_for_violations, glm_hmm_fit_data


if __name__ == '__main__':
    D = 1  # data (observations) dimension
    C = 2  # number of output types/categories
    num_iters_EM_fit = 300  # number of EM iterations
    # model_init_num = 1
    cluster = False
    cross_valid_num_fold = 5

    if cluster is False:
        # z = 1
        K = 3
        iter = 0
        fold = 0
        num_inputs = 4
        prior_sigma = 4.0
        transition_alpha = 2.0
        base_dir = '../../glm-hmm_package/results/model_indiv_ibl/'
        path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/'

    elif cluster is True:
        z = int(sys.argv[1])
        base_dir = '/home/../glm-hmm_all_data_GLM_trans_diff_inputs/results/model_indiv_ibl/'
        path_data = '/home/../glm-hmm_all_data_GLM_trans_diff_inputs/data/ibl/Della_cluster_data/'

    # Load external files:
    info_cluster_file = path_data + 'separate_mouse_data/cluster_job_arr.npz'
    # Load cluster array job parameters:
    info_cluster = get_cluster_info(info_cluster_file)

    if cluster is True:
        [prior_sigma, transition_alpha, K, num_inputs, fold, iter] = info_cluster[z]

    iter = int(iter)
    fold = int(fold)
    K = int(K)
    num_inputs = int(num_inputs)

    mice_names = mice_names_info(path_data + 'separate_mouse_data/mice_names.npz')

    for i, animal in enumerate(mice_names):
        print('animal=', animal)
        mouse_data = path_data + 'separate_mouse_data/' + animal + '_processed.npz'
        fold_mapping_session = load_fold_session_map(
            path_data + 'separate_mouse_data/' + animal + '_fold_session_map.npz')

        global_fit = False
        GLM_T_init_0 = False  # Start with a global fit rather than zero transition weights.

        obs_mat, trans_mat, y, session = get_mouse_info(mouse_data)
        indexes = range(0, [obs_mat.shape[1] - 1][0], 1)
        obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))  # add bias column

        path_main_folder = base_dir + 'num_regress_obs_' + str(num_inputs) + '/prior_sigma_' + str(
            prior_sigma) + '_transition_alpha_' + str(transition_alpha) + '/' + animal + '/'

        indexes2 = np.append(indexes, (obs_mat.shape[1] - 1))
        obs_mat = obs_mat[:, indexes2]  # this has the size of number of regressors

        y = y.astype('int')
        # Identify violations for exclusion:
        index_viols = np.where(y == -1)[0]
        nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])
        # transition_alpha = 2
        # prior_sigma = 2

        needed_info_for_init = path_data + 'optimum_model/optimum_model_K_' + str(K) + '_num_inputs_' + str(num_inputs) + '.npz'

        glm_hmm_fit_data(obs_mat, trans_mat, y, session, mask, fold_mapping_session, K, D, C, num_iters_EM_fit,
                           transition_alpha, prior_sigma, fold, iter, global_fit, GLM_T_init_0,
                           needed_info_for_init, path_main_folder, ibl_init=True)
