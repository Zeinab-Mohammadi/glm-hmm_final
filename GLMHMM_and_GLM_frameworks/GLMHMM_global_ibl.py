"""

Fit the global GLM-HMM to all pooled IBL data from all mice in the decision-making task

"""
import autograd.numpy as np
import sys
from utils_for_GLMHMM import get_cluster_info, load_fold_session_map, get_mouse_info, mask_for_violations, \
    glm_hmm_fit_data


if __name__ == '__main__':
    D = 1  # data (observations) dimension
    C = 2  # number of output types/categories
    num_iters_EM_fit = 300  # number of EM iterations
    num_inputs = 4
    # Clus_param = 401 #just an example to test
    cross_valid_num_fold = 5
    cluster = False  # False: If the code is executing on the local computer rather than a cluster..
    global_fit = True
    GLM_T_init_0 = True  # initialize transition weights with zero

    if cluster is True:
        Clus_param = int(sys.argv[1])
        path_data = '/home/../glm-hmm_all_data_GLM_trans_diff_inputs/data/ibl/Della_cluster_data/'
        path_main_folder = '/home/../glm-hmm_all_data_GLM_trans_diff_inputs/results/model_global_ibl/' + 'num_regress_obs_' + str(
            num_inputs) + '/'
        info_cluster_file = path_data + 'separate_mouse_data/cluster_job_arr.npz'
        # Load cluster array job parameters:
        info_cluster = get_cluster_info(
            info_cluster_file)  # info_cluster_file was created using the Della_cluster_prep.py script
        [num_inputs, K, fold, iter] = info_cluster[Clus_param]

    else:
        # z = 1
        K = 2
        iter = 0
        fold = 0
        num_inputs = 4
        prior_sigma = 4.0
        transition_alpha = 2.0
        path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/'
        path_main_folder = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(
            num_inputs) + '/'

    # Load external files:
    # fold = int(fold)
    # iter = int(iter)
    # K = int(K)

    # for fold in range(cross_valid_num_fold):
    #     for iter in range(model_init_num):
    #         for K in [2, 3, 4 ,5]: #K is the number of states

    mouse_data = path_data + 'combined_all_mice.npz'
    fold_mapping_session = load_fold_session_map(path_data + 'fold_mapping_session_all_mice.npz')

    obs_mat, trans_mat, output, session = get_mouse_info(mouse_data)
    indexes = range(0, [obs_mat.shape[1] - 1][0], 1)
    obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))

    indexes2 = np.append(indexes, (obs_mat.shape[1] - 1))
    obs_mat = obs_mat[:, indexes2]
    output = output.astype('int')

    # Examine instances warranting exclusion:
    index_viols = np.where(output == -1)[0]
    nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[
        0])

    needed_info_for_init = path_main_folder + 'Model/glm_#state=1/fld_num=' + str(fold) + '/important_params_iter_0.npz'
    transition_alpha = 1  # means no prior on transitions matrix
    prior_sigma = 100

    glm_hmm_fit_data(obs_mat, trans_mat, output, session, mask, fold_mapping_session, K, D, C, num_iters_EM_fit,
                       transition_alpha,
                       prior_sigma, fold, iter, global_fit, GLM_T_init_0, needed_info_for_init, path_main_folder, ibl_init=False)