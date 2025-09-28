## 1. Fit GLM-HMM on real data
## 2. When fitting, allow multiple initializations for HMM parameters
## 3. Plot recovered vectors, transition matrices for a given rat and session
"""

Assorted auxiliary functions for GLM-HMM fit

"""
import os
import autograd.numpy as np
import autograd.numpy.random as npr
import sys
from softmax_observations_fit_utils import fit_hmm_observations


def get_mouse_info(mouse_data):
    container = np.load(mouse_data, allow_pickle=True)
    data = [container[key] for key in container]
    obs_mat = data[0]
    trans_mat = data[1]
    output = data[2]
    session = data[3]
    return obs_mat, trans_mat, output, session

def get_mouse_info_all(mouse_data):
    container = np.load(mouse_data, allow_pickle=True)
    data = [container[key] for key in container]
    obs_mat = data[0]
    trans_mat = data[1]
    output = data[2]
    session = data[3]
    return obs_mat, trans_mat, output, session


def get_old_ibl_data(mouse_data):
    container = np.load(mouse_data, allow_pickle=True)
    data = [container[key] for key in container]
    obs_mat = data[0]
    y = data[1]
    y = y.astype('int')
    session = data[3]
    return obs_mat, y, session


def get_cluster_info(info_cluster_file):
    container = np.load(info_cluster_file, allow_pickle=True)
    data = [container[key] for key in container]
    info_cluster = data[0]
    return info_cluster


def get_glm_info(glm_vectors_file):
    container = np.load(glm_vectors_file)
    data = [container[key] for key in container]
    train_calculate_LL = data[0]
    recovered_weights = data[1]
    standard_deviation = data[2]
    return train_calculate_LL, recovered_weights, standard_deviation



def get_params_global_fit(global_params_file):
    container = np.load(global_params_file, allow_pickle=True)
    data = [container[key] for key in container]
    global_params = data[0]
    return global_params


def get_params_global_fit_old_ibl(global_params_file):
    container = np.load(global_params_file, allow_pickle=True)
    data = [container[key] for key in container]
    global_params = data[0]
    return global_params


def data_segmentation_session(obs_mat, trans_mat, y, mask, session):
    """
    Partition obs_mat, trans_mat, y, mask by session
    :param obs_mat: arr of size TxM_obs
    :params trans_mat: arr of size TxM_Trans
    :param y:  arr of size T x D
    :param mask: Boolean arr of size T indicating if element is violation or not
    :param session: list of size T containing session ids
    :return: list of obs_mat arrays, trans_mat array, data arrays and mask arrays, where the number of elements in list = number of sessions and each array size is number of trials in session
    """
    inputs = []
    inputs_trans = []
    datas = []
    indexes = np.unique(session, return_index=True)[1]
    unique_sessions = [session[index] for index in sorted(
        indexes)]
    counter = 0
    masks = []
    for sess in unique_sessions:
        idx = np.where(session == sess)[0]
        counter += len(idx)
        inputs.append(obs_mat[idx, :])
        inputs_trans.append(trans_mat[idx, :])
        datas.append(y[idx, :])
        masks.append(mask[idx])
    assert counter == obs_mat.shape[0], "not all trials assigned to session!"
    return inputs, inputs_trans, datas, masks


def load_fold_session_map(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    fold_mapping_session = data[0]
    return fold_mapping_session


def mice_names_info(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    mice_names = data[0]
    return mice_names


def load_synthetic_data(file_path):  # this was made by me
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    synthetic_data = data[0]
    return synthetic_data


def fit_model_iter(iter, inputs, inputs_trans, datas, train_masks, K, D, M, M_trans, C, num_iters_EM_fit, transition_alpha,
                  prior_sigma, global_fit, GLM_T_init_0, params_for_initialization, save_directory):
    npr.seed(iter)
    this_directory = save_directory + '/iter_' + str(iter) + '/'
    if not os.path.exists(this_directory):
        os.makedirs(this_directory)
    fit_hmm_observations(datas, inputs, inputs_trans, train_masks, K, D, M, M_trans, C, num_iters_EM_fit, transition_alpha,
                         prior_sigma, global_fit, GLM_T_init_0, params_for_initialization,
                         save_title=this_directory + 'glm_hmm_raw_parameters_itr_' + str(iter) + '.npz')
    return None


def glm_hmm_fit_data(obs_mat, trans_mat, y, session, mask, fold_mapping_session, K, D, C, num_iters_EM_fit,
                       transition_alpha, prior_sigma, fold, iter, global_fit, GLM_T_init_0, needed_info_for_init,
                       path_main_folder, ibl_init):
    print("Starting inference with K = " + str(K) + "; Fold = " + str(fold) + "; Iter = " + str(iter))
    sys.stdout.flush()
    appropriate_sessions = fold_mapping_session[np.where(fold_mapping_session[:, 1] != fold), 0]
    indexes_sess_fold = [str(sess) in appropriate_sessions for sess in session]
    needed_obs_mat, needed_trans_mat, this_y, needed_this_session, this_mask = obs_mat[indexes_sess_fold, :], trans_mat[indexes_sess_fold,
                                                                                          :], y[indexes_sess_fold, :], \
                                                                  session[indexes_sess_fold], mask[indexes_sess_fold]


    this_y[np.where(this_y == -1), :] = 1
    inputs, inputs_trans, datas, train_masks = data_segmentation_session(needed_obs_mat, needed_trans_mat, this_y, this_mask,
                                                                         needed_this_session)
    # print('np.array(inputs_trans).shape==', np.array(inputs_trans).shape)
    # Read in GLM fit if global_fit = True:
    if global_fit is True:
        train_calculate_LL, glm_vectors, glm_standard_deviation = get_glm_info(needed_info_for_init)
        params_for_initialization = glm_vectors
    elif ibl_init is True:
        # if individual fits, initialize each model with the global fit:
        params_for_initialization = get_params_global_fit_old_ibl(needed_info_for_init)
    else:
        params_for_initialization = get_params_global_fit(needed_info_for_init)
    M = needed_obs_mat.shape[1]
    M_trans = needed_trans_mat.shape[1]
    save_directory = path_main_folder + '/Model/glmhmm_#state=' + str(K) + '/' + 'fld_num=' + str(fold) + '/'
    os.makedirs(save_directory, exist_ok=True)

    fit_model_iter(iter, inputs, inputs_trans, datas, train_masks, K, D, M, M_trans, C, num_iters_EM_fit, transition_alpha,
                  prior_sigma, global_fit, GLM_T_init_0,
                  params_for_initialization, save_directory)


def glm_hmm_fit_data_simul_data(obs_mat, trans_mat, y, K, D, C, num_iters_EM_fit,
                                  transition_alpha, prior_sigma, iter, global_fit, GLM_T_init_0, needed_info_for_init,
                                  path_main_folder, ibl_init):
    sys.stdout.flush()
    inputs, inputs_trans, datas = obs_mat, trans_mat, y

    # Read in GLM fit if global_fit = True:
    if global_fit is True:
        train_calculate_LL, glm_vectors, glm_standard_deviation = get_glm_info(needed_info_for_init)
        params_for_initialization = glm_vectors

    elif ibl_init is True:
        # if individual fits, initialize each model with the global fit:
        params_for_initialization = get_params_global_fit_old_ibl(needed_info_for_init)
    else:
        params_for_initialization = get_params_global_fit(needed_info_for_init)

    M = inputs.shape[1]
    M_trans = inputs_trans.shape[1]
    save_directory = path_main_folder + '/Model/glmhmm_#state=' + str(K) + '/'
    os.makedirs(save_directory, exist_ok=True)
    train_masks = []

    fit_model_iter(iter, inputs, inputs_trans, datas, train_masks, K, D, M, M_trans, C, num_iters_EM_fit, transition_alpha,
                  prior_sigma, global_fit, GLM_T_init_0, params_for_initialization,
                  save_directory)  


def mask_for_violations(index_viols, T):
    """
    Return indices of nonviolations and also a Boolean mask for inclusion (1 = nonviolation; 0 = violation)
    :param test_idx:
    :param T:
    :return:
    """
    mask = np.array([i not in index_viols for i in range(T)])
    nonindex_viols = np.arange(T)[mask]
    mask = mask + 0
    assert len(nonindex_viols) + len(index_viols) == T, "violation and non-violation idx do not include all dta!"
    return nonindex_viols, mask
