"""

Assorted auxiliary functions for analyzing the fitting results

"""

import numpy as np
import ssm
import math
import glob
import ssm
import copy
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from GLM_class import glm  


def makeRaisedCosBasis(bias_num):
    num_trials_consider_basis = 100  # considering first 100 trials of each session
    nB = bias_num  # number of basis functions
    peakRange = [0, num_trials_consider_basis]
    timeRange = [0, num_trials_consider_basis]

    # Define function for single raised cosine basis function
    def raisedCosFun(x, ctr, dCtr):
        return (np.cos(np.maximum(-math.pi, np.minimum(math.pi, (x - ctr) * math.pi / dCtr / 2))) + 1) / 2

    # Compute location for cosine basis centers
    dCtr = np.diff(peakRange) / (nB - 1)  # spacing between raised cosine peaks
    Bctrs = np.arange(peakRange[0], peakRange[1] + .1, dCtr)  # peaks for cosine basis vectors
    basisPeaks = Bctrs  # vector of raised cosine centers
    dt = 1
    minT = timeRange[0]
    maxT = timeRange[1]
    tgrid = np.arange(minT, maxT + .1, dt)  # time grid
    nT = tgrid.shape[0]  # number of time points in basis
    # Make the basis
    cosBasis = raisedCosFun(np.tile(tgrid, (nB, 1)).T, np.tile(Bctrs, (nT, 1)), dCtr)
    return cosBasis, tgrid, basisPeaks



def calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, transition_alpha, prior_sigma, perm):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M_trans = np.array(inputs_trans[0]).shape[1]
    M = inputs[0].shape[1]
    D = datas[0].shape[1]
    if globe is True:
        prior_sigma = 100
    this_hmm = ssm.HMM_TO(K, D, M_trans=M_trans, M_obs=M, observations="input_driven_obs_diff_inputs",
                          observation_kwargs=dict(C=2, prior_sigma=prior_sigma),
                          transitions="inputdrivenalt",
                          transition_kwargs=dict(prior_sigma=prior_sigma, alpha=transition_alpha, kappa=0))
    this_hmm.params = Params_model
    # Get expected states:
    expectations = [this_hmm.expected_states(data=data, transition_input=input_trans, observation_input=input)[0]
                    for data, input_trans, input
                    in zip(datas, inputs_trans, inputs)]
    Ps = [this_hmm.Ps_matrix(data=data, transition_input=input_trans, observation_input=input)
          for data, input_trans, input
          in zip(datas, inputs_trans, inputs)]
    # Convert this to one array:
    posterior_probs = np.concatenate(expectations, axis=0)
    Ps_all = np.concatenate(Ps, axis=0)
    posterior_probs = posterior_probs[:, perm]
    return posterior_probs, Ps_all



def data_for_cross_validation(obs_mat, trans_mat, y, session, fold_mapping_session, fold):
    index_viols = np.where(y == -1)[0]
    nonindex_viols, nonviolation_mask = mask_for_violations(index_viols, obs_mat.shape[0])

    # Load train and test data for session
    test_obs_mat, test_trans_mat, test_y, test_nonviolation_mask, this_test_session, train_obs_mat, train_trans_mat, train_y, train_nonviolation_mask, this_train_session = divide_data_for_test_train(
        obs_mat, trans_mat, y, nonviolation_mask, session, fold_mapping_session, fold)
    M = train_obs_mat.shape[1]
    M_trans = test_trans_mat.shape[1]
    n_test = np.sum(test_nonviolation_mask == 1)
    n_train = np.sum(train_nonviolation_mask == 1)
    return test_obs_mat, test_trans_mat, test_y, test_nonviolation_mask, this_test_session, train_obs_mat, train_trans_mat, train_y, train_nonviolation_mask, this_train_session, M, M_trans, n_test, n_train


def divide_data_for_test_train(obs_mat, trans_mat, y, mask, session, fold_mapping_session, fold):
    """
    Split obs_mat, y, mask, session arrays into train and test arrays
    :param obs_mat:
    :param y:
    :param mask:
    :param session:
    :param fold_mapping_session:
    :param fold:
    :return:
    """
    test_sessions = fold_mapping_session[np.where(fold_mapping_session[:, 1] == fold), 0]
    train_sessions = fold_mapping_session[np.where(fold_mapping_session[:, 1] != fold), 0]
    idx_test = [str(sess) in test_sessions for sess in session]
    idx_train = [str(sess) in train_sessions for sess in session]
    test_obs_mat, test_trans_mat, test_y, test_mask, this_test_session = obs_mat[idx_test, :], trans_mat[idx_test, :], y[
                                                                                                                   idx_test,
                                                                                                                   :], \
                                                                       mask[idx_test], session[idx_test]
    train_obs_mat, train_trans_mat, train_y, train_mask, this_train_session = obs_mat[idx_train, :], trans_mat[idx_train,
                                                                                                :], y[idx_train, :], \
                                                                            mask[idx_train], session[idx_train]

    return test_obs_mat, test_trans_mat, test_y, test_mask, this_test_session, train_obs_mat, train_trans_mat, train_y, train_mask, this_train_session


def data_segmentation_session(obs_mat, trans_mat, y, mask, session):
    """
    Partition obs_mat, y, mask by session
    :param obs_mat: arr of size TxM
    :param y:  arr of size T x D
    :param mask: Boolean arr of size T indicating if element is violation or not
    :param session: list of size T containing session ids
    :return: list of obs_mat arrays, data arrays and mask arrays, where the number of elements in list = number of sessions and each array size is number of trials in session
    """
    inputs = []
    inputs_trans = []
    datas = []
    indexes = np.unique(session, return_index=True)[1]
    unique_sessions = [session[index] for index in sorted(indexes)]
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
    assert len(nonindex_viols) + len(index_viols) == T, "violation and non-violation idx do not include all data!"
    return nonindex_viols, mask


def get_file_name_for_best_model_fold(diff_folds_fit, K, path_main_folder, optimal_initialize_dict):
    """
    Get the file name for the best initialization for the K value specified
    :param diff_folds_fit:
    :param K:
    :param models:
    :param path_main_folder:
    :param optimal_initialize_dict:
    :return:
    """
    # Identify best fold for best model:
    loc_best = 0
    best_fold = np.where(diff_folds_fit[loc_best, :] == max(diff_folds_fit[loc_best, :]))[0][0]
    base_path = path_main_folder + '/Model/glmhmm_#state=' + str(K) + '/fld_num=' + str(best_fold)
    keys = '/glmhmm_#state=' + str(K) + '/fld_num=' + str(best_fold)
    best_iter = optimal_initialize_dict[keys]
    params_and_LL = base_path + '/iter_' + str(best_iter) + '/glm_hmm_raw_parameters_itr_' + str(best_iter) + '.npz'
    return params_and_LL


def permute_transition_matrix(transition_matrix, perm):
    transition_matrix = transition_matrix[np.ix_(perm, perm)]
    return transition_matrix


def find_corresponding_states_for_plots_on_top(Params_model):
    glm_weights = -Params_model[2]
    K = glm_weights.shape[0]
    if K == 3:
        M = glm_weights.shape[2] - 1
        # bias coefficient is last entry in dimension 2
        engaged_loc = np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        reduced_weights = np.copy(glm_weights)
        # set row in reduced weights corresponding to engaged 
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        # bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        bias_left_loc = np.where((reduced_weights[:, 0, M] < 0))
        state_order = [engaged_loc, bias_left_loc]
        bias_right_loc = np.arange(3)[np.where([range(3)[i] not in state_order for i in range(3)])][0]
        perm = np.array([engaged_loc, bias_left_loc, bias_right_loc])
    elif K == 4:
        M = glm_weights.shape[2] - 1
        engaged_loc = np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        reduced_weights = np.copy(glm_weights)
        # set row in reduced weights corresponding to engaged 
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        bias_right_loc = np.where((reduced_weights[:, 0, M] == max(reduced_weights[:, 0, M])))[0][0]
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        state_order = [engaged_loc, bias_left_loc, bias_right_loc]
        other_loc = np.arange(4)[np.where([range(4)[i] not in state_order for i in range(4)])][0]
        perm = np.array([engaged_loc, bias_left_loc, bias_right_loc, other_loc])
    else:
        # order states by engagement: with the most engaged being first.
        perm = np.argsort(-glm_weights[:, 0, 0])
    return perm


def find_corresponding_states(Params_model):
    glm_weights = -Params_model[2]
    K = glm_weights.shape[0]
    if K == 3:
        M = glm_weights.shape[2] - 1
        # bias coefficient is last entry in dimension 2
        engaged_loc = np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        reduced_weights = np.copy(glm_weights)
        # set row in reduced weights corresponding to engaged 
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        state_order = [engaged_loc, bias_left_loc]
        bias_right_loc = np.arange(3)[np.where([range(3)[i] not in state_order for i in range(3)])][0]
        perm = np.array([engaged_loc, bias_left_loc, bias_right_loc])
    elif K == 4:
        M = glm_weights.shape[2] - 1
        # bias coefficient is last entry in dimension 2
        engaged_loc = np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        reduced_weights = np.copy(glm_weights)
        # set row in reduced weights corresponding to engaged 
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        bias_right_loc = np.where((reduced_weights[:, 0, M] == max(reduced_weights[:, 0, M])))[0][0]
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        state_order = [engaged_loc, bias_left_loc, bias_right_loc]
        other_loc = np.arange(4)[np.where([range(4)[i] not in state_order for i in range(4)])][0]
        perm = np.array([engaged_loc, bias_left_loc, bias_right_loc, other_loc])
    else:
        # order states by engagement: with the most engaged being first.
        perm = np.argsort(-glm_weights[:, 0, 0])
    return perm


def find_corresponding_states_all_data(Params_model):
    glm_weights = -Params_model[2]
    K = glm_weights.shape[0]
    if K == 3:
        M = glm_weights.shape[2] - 1
        # bias coefficient is last entry in dimension 2
        engaged_loc = np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        reduced_weights = np.copy(glm_weights)
        # set row in reduced weights corresponding to engaged 
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        state_order = [engaged_loc, bias_left_loc]
        bias_right_loc = np.arange(3)[np.where([range(3)[i] not in state_order for i in range(3)])][0]
        perm = np.array([engaged_loc, bias_left_loc, bias_right_loc])

    elif K == 4:
        M = glm_weights.shape[2] - 1
        # bias coefficient is last entry in dimension 2
        engaged_loc = np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        reduced_weights = np.copy(glm_weights)
        # set row in reduced weights corresponding to engaged 
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        bias_right_loc = np.where((reduced_weights[:, 0, M] == max(reduced_weights[:, 0, M])))[0][0]
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        state_order = [engaged_loc, bias_left_loc, bias_right_loc]
        other_loc = np.arange(4)[np.where([range(4)[i] not in state_order for i in range(4)])][0]
        perm = np.array([engaged_loc, bias_left_loc, bias_right_loc, other_loc])
    else:
        # order states by engagement: with the most engaged being first.
        perm = np.argsort(-glm_weights[:, 0, 0])
    return perm


def find_corresponding_states_new(Params_model):
    glm_weights = -Params_model[2]
    K = glm_weights.shape[0]
    if K == 3:
        M = glm_weights.shape[2] - 1
        # bias coefficient is last entry in dimension 2
        engaged_loc = np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        reduced_weights = np.copy(glm_weights)
        # set row in reduced weights corresponding to engaged 
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        state_order = [engaged_loc, bias_left_loc]
        bias_right_loc = np.arange(3)[np.where([range(3)[i] not in state_order for i in range(3)])][0]
        perm = np.array([engaged_loc, bias_left_loc, bias_right_loc])

    elif K == 4:
        # want states ordered as engaged/bias left/bias right
        M = glm_weights.shape[2] - 1
        # bias coefficient is last entry in dimension 2
        weights_sort = np.sort(glm_weights[:, 0, 0])
        engaged_loc1 = np.where((glm_weights[:, 0, 0] == weights_sort[-1]))[0][0]
        reduced_weights = np.copy(glm_weights)
        # set row in reduced weights corresponding to engaged 
        reduced_weights[engaged_loc1, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        engaged_loc2 = np.where((reduced_weights[:, 0, 0] == weights_sort[-2]))[0][0]
        bias_right_loc = np.where((reduced_weights[:, 0, M] == max(reduced_weights[:, 0, M])))[0][0]
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        state_order = [engaged_loc1, engaged_loc2, bias_left_loc, bias_right_loc]
        # other_loc = np.arange(4)[np.where([range(4)[i] not in state_order for i in range(4)])][0]
        perm = np.array([engaged_loc1, engaged_loc2, bias_left_loc, bias_right_loc])  # , other_loc])
    else:
        # order states by engagement: with the most engaged being first.
        perm = np.argsort(-glm_weights[:, 0, 0])
    return perm



def posterior_probs_wrapper(path_data, animal, num_inputs, Params_model, transition_alpha, prior_sigma, K,  perm):
    obs_mat, trans_mat, y, session, left_probs, animal_eids = get_mouse_info(path_data + animal + '_processed.npz')
    obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))
    if num_inputs == 0:
        obs_mat = obs_mat[:, [0, 3]]
    elif num_inputs == 1:
        obs_mat = obs_mat[:, [0, 1, 3]]
    elif num_inputs == 2:
        obs_mat = obs_mat[:, [0, 1, 2, 3]]
    # Create mask:
    # Identify violations for exclusion:
    index_viols = np.where(y == -1)[0]
    nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, inputs_trans, datas, masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)
    globe = False
    posterior_probs, Ps_all = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K,
                                             transition_alpha, prior_sigma,  perm)
    return posterior_probs


def get_mouse_info_unorm(data_file):
    container = np.load(data_file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_obs_unnorm_regress = data[0]
    animal_trans_unnorm_regress = data[1]
    animal_y = data[2]
    animal_y = animal_y.astype('int')
    animal_session = data[3]
    return animal_obs_unnorm_regress, animal_trans_unnorm_regress, animal_y, animal_session


def get_mouse_info(mouse_data):
    container = np.load(mouse_data, allow_pickle=True)
    data = [container[key] for key in container]
    obs_mat = data[0]
    trans_mat = data[1]
    y = data[2]
    y = y.astype('int')
    session = data[3]
    left_probs = data[4]
    if len(data) > 5:
        animal_eids = data[5]
    else:
        animal_eids = ['not_added_to_this_file']
    return obs_mat, trans_mat, y, session, left_probs, animal_eids


def get_mouse_info_all(mouse_data):
    container = np.load(mouse_data, allow_pickle=True)
    data = [container[key] for key in container]
    obs_mat = data[0]
    y = data[1]
    y = y.astype('int')
    session = data[2]
    if len(data) > 3:
        animal_eids = data[3]
    else:
        animal_eids = ['not_added_to_this_file']
    return obs_mat, y, session, animal_eids


def get_old_ibl_data(mouse_data):
    container = np.load(mouse_data, allow_pickle=True)
    data = [container[key] for key in container]
    obs_mat = data[0]
    y = data[1]
    y = y.astype('int')
    session = data[3]
    return obs_mat, y, session


def get_params_global_fit_old_ibl(global_params_file):
    container = np.load(global_params_file, allow_pickle=True)
    data = [container[key] for key in container]
    global_params = data
    global_params = [global_params[0], [global_params[1]], global_params[2]]
    return global_params



def mice_names_info(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    mice_names = data[0]
    return mice_names


def load_fold_session_map(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    fold_mapping_session = data[0]
    return fold_mapping_session


def get_glm_info(glm_vectors_file):
    container = np.load(glm_vectors_file)
    data = [container[key] for key in container]
    train_calculate_LL = data[0]
    recovered_weights = data[1]
    standard_deviation = data[2]
    return train_calculate_LL, recovered_weights, standard_deviation


def model_data_glmhmm(data_file):
    container = np.load(data_file, allow_pickle=True)
    data = [container[key] for key in container]
    this_Params_model = data[0]
    lls = data[1]
    return [this_Params_model, lls]


def cross_validation_vector(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    diff_folds_fit = data[0]
    return diff_folds_fit


def colors_func(K):
    if K < 4:
        cols = ["#15b01a", "#0165fc", "#e74c3c", "#8e82fe", "#c20078", "#520c3c"]
    elif K == 4:
        cols = ["#0165fc", "#c20078", "#8e82fe", "#e74c3c", "#520c3c",
                "#15b01a"]
    elif K == 5:
        cols = ["#0165fc", "#c20078", "#e74c3c", "#8e82fe", "#15b01a", "#520c3c"]
    elif K == 6:
        cols = ["#15b01a", "#e74c3c", "#0165fc", "#c20078", "#8e82fe", "#520c3c"]
    return cols


def colors_func_indiv(K):
    if K < 4:
        cols = ["#15b01a", "#0165fc", "#e74c3c", "#8e82fe", "#c20078", "#520c3c"]
    elif K == 4:
        cols = ["#8e82fe", "#c20078", "#0165fc",
                "#e74c3c"]
    elif K == 5:
        cols = ["#c20078", "#15b01a", "#e74c3c", "#0165fc", "#8e82fe",
                "#520c3c"]
    # No 6-states for individual
    # elif K == 6:  # this is for k = 6-state model
    # This needs modification based on colors cols = ["#15b01a", "#e74c3c", "#0165fc", "#c20078", "#8e82fe", "#520c3c"]
    return cols


def state_glob_label(params):
    wights = -params[2]
    M = wights.shape[2] - 1
    weights_seq = np.sort(wights[:, 0, 0])
    loc1 = np.where((wights[:, 0, 0] == weights_seq[-1]))[0][0]
    weights_remove = np.copy(wights)
    weights_remove[loc1, 0, M] = max(wights[:, 0, M]) - 0.001
    loc2 = np.where((weights_remove[:, 0, 0] == weights_seq[-2]))[0][0]
    loc_r = np.where((weights_remove[:, 0, M] == max(weights_remove[:, 0, M])))[0][0]
    loc_f = np.where((weights_remove[:, 0, M] == min(weights_remove[:, 0, M])))[0][0]
    # state_order = [loc1, loc2, loc_f, loc_r]
    Weights_label = np.array([loc_f, loc2, loc1, loc_r])
    return Weights_label


def make_cross_valid_for_figure(cv_file, idx):
    diff_folds_fit = cross_validation_vector(cv_file)
    glm_lapse_model = diff_folds_fit[:3, ]
    diff_folds_fit = diff_folds_fit[idx, :]
    # Identify best cvbt:
    mean_cvbt = np.mean(diff_folds_fit, axis=1)
    loc_best = np.where(mean_cvbt == max(mean_cvbt))[0]
    best_val = max(mean_cvbt)
    # Create dataframe for plotting
    fits_num = diff_folds_fit.shape[0]
    cross_valid_num_fold = diff_folds_fit.shape[1]
    # Create pandas dataframe:
    data_for_plotting_df = pd.DataFrame(
        {'model': np.repeat(np.arange(fits_num), cross_valid_num_fold), 'cv_bit_trial': diff_folds_fit.flatten()})
    return data_for_plotting_df, loc_best, best_val, glm_lapse_model

def addBiasBlocks(fig, pL):
    BIAS_COLORS = {50: 'None', 20: "#ffdbe0", 80: "#b7d2e8"}  # light blue: 20% right and light pink: 80% right
    plt.sca(fig.gca())
    i = 0
    while i < len(pL):
        start = i
        while i+1 < len(pL) and np.linalg.norm(pL[i] - pL[i+1]) < 0.0001:
            i += 1
        fc = BIAS_COLORS[int(100 * pL[start])]
        plt.axvspan(start, i+1, facecolor=fc, alpha=0.2, edgecolor=None)
        i += 1
    return fig

def log_likelihood_base_for_test(train_y, test_y, C):
    """
    Calculate baseline loglikelihood for CV bit/trial calculation.  This is log(p(y|p0)) = n_right(log(p0)) + (n_total-n_right)log(1-p0), where p0 is the proportion of trials
    in which the animal went right in the training set and n_right is the number of trials in which the animal went right in the test set
    :param train_y
    :param test_y
    :return: baseline loglikelihood for CV bit/trial calculation
    """
    _, train_class_totals = np.unique(train_y, return_counts=True)
    train_class_probs = train_class_totals / train_y.shape[0]
    _, test_class_totals = np.unique(test_y, return_counts=True)
    ll0 = 0
    for c in range(C):
        ll0 += test_class_totals[c] * np.log(train_class_probs[c])
    return ll0


def log_likelihood_for_test_glm(params_glm, test_y, test_obs_mat, M, C):
    train_calculate_LL, glm_vectors, glm_standard_deviation = get_glm_info(params_glm)
    # Calculate test loglikelihood
    new_glm = glm(M, C)
    # Set parameters to fit parameters:
    new_glm.params = glm_vectors
    # Get loglikelihood of training data:
    loglikelihood_test = new_glm.log_marginal([test_y], [test_obs_mat], None, None)
    return loglikelihood_test


def get_params_global_fit(global_params_file):
    container = np.load(global_params_file, allow_pickle=True)
    data = [container[key] for key in container]
    global_params = data[0]
    return global_params


def log_likelihood_for_test_glmhmm(globe, prior_sigma, glm_hmm_dir, test_datas, test_inputs, test_inputs_trans,
                                         test_nonviolation_masks, K, D, M, M_trans, C):
    """
    calculate test loglikelihood for GLM-HMM model.  Loop through all initializations for fold of interest, and check that final train LL is same for top initializations
    :return:
    """
    this_file_name = glm_hmm_dir + 'iter_*/glm_hmm_raw_parameters_*.npz'
    params_and_LLs = glob.glob(this_file_name, recursive=True)
    train_ll_vals_across_iters = []
    test_ll_vals_across_iters = []

    for file in params_and_LLs:
        # Loop through initializations and calculate BIC:
        this_Params_model, lls = model_data_glmhmm(file)
        train_ll_vals_across_iters.append(lls[-1])

        # Instantiate a new HMM and calculate test loglikelihood:
        if globe is True:
            this_hmm = ssm.HMM_TO(K, D, M_trans=M_trans, M_obs=M, observations="input_driven_obs_diff_inputs",
                                  observation_kwargs=dict(C=C, prior_sigma=prior_sigma),
                                  transitions="inputdrivenalt", transition_kwargs=dict(alpha=1, kappa=0))
        else:
            num_inputs = 4
            path_data = "../../glm-hmm_package/data/ibl/Della_cluster_data/"
            needed_info_for_init = path_data + 'optimum_model/optimum_model_K_' + str(K) + '_num_inputs_' + str(num_inputs) + '.npz'
            params_for_initialization = get_params_global_fit(needed_info_for_init)
            Wk_glob = copy.deepcopy(params_for_initialization[2])
            this_hmm = ssm.HMM_TO(K, D, M_trans=M_trans, M_obs=M, observations="input_driven_obs_diff_inputs",
                                  observation_kwargs=dict(C=C, prior_sigma=prior_sigma),
                                  transitions="inputdrivenalt",
                                  transition_kwargs=dict(prior_sigma=prior_sigma, alpha=1, kappa=0))

        this_hmm.params = this_Params_model
        test_ll = this_hmm.log_likelihood(test_datas, transition_input=test_inputs_trans, observation_input=test_inputs)
        test_ll_vals_across_iters.append(test_ll)
    # Order initializations by train LL (don't train on test data!):
    train_ll_vals_across_iters = np.array(train_ll_vals_across_iters)
    test_ll_vals_across_iters = np.array(test_ll_vals_across_iters)
    # Order raw files by train LL
    file_ordering_by_train = np.argsort(-train_ll_vals_across_iters)
    params_and_LL_ordering_by_train = np.array(params_and_LLs)[file_ordering_by_train]
    # Get initialization number from params_and_LL ordering
    train_for_arranging_initials = [int(re.findall(r'\d+', file)[-1]) for file in params_and_LL_ordering_by_train]
    return test_ll_vals_across_iters, train_for_arranging_initials, file_ordering_by_train


def glmhmm_normalized_loglikelihood(globe, prior_sigma, obs_mat, trans_mat, y, session, fold_mapping_session, fold, K, D, C,
                      path_analysis_glm_hmm):
    """
    For a given fold, return NLL for both train and test datasets for GLM-HMM model with K, D, C.  Requires reading in best
    parameters over all initializations for GLM-HMM (hence why path_analysis_glm_hmm is required as an input)
    :param obs_mat:
    :param y:
    :param session:
    :param fold_mapping_session:
    :param fold:
    :param K:
    :param D:
    :param C:
    :param path_analysis_glm_hmm:
    :return:
    """
    test_obs_mat, test_obs_mat_tran, test_y, test_nonviolation_mask, this_test_session, train_obs_mat, train_trans_mat, train_y, train_nonviolation_mask, this_train_session, M, M_trans, n_test, n_train = data_for_cross_validation(
        obs_mat, trans_mat, y, session, fold_mapping_session, fold)

    ll0 = log_likelihood_base_for_test(train_y[train_nonviolation_mask == 1, :], test_y[test_nonviolation_mask == 1, :],
                                     C)
    ll0_train = log_likelihood_base_for_test(train_y[train_nonviolation_mask == 1, :],
                                           train_y[train_nonviolation_mask == 1, :], C)
    test_y[test_nonviolation_mask == 0, :] = 1
    # For GLM-HMM, need to partition data by session
    test_inputs, test_inputs_trans, test_datas, test_nonviolation_masks = data_segmentation_session(test_obs_mat,
                                                                                                    test_obs_mat_tran,
                                                                                                    test_y,
                                                                                                    test_nonviolation_mask,
                                                                                                    this_test_session)

    train_y[train_nonviolation_mask == 0, :] = 1
    train_inputs, train_inputs_trans, train_datas, train_nonviolation_masks = data_segmentation_session(train_obs_mat,
                                                                                                        train_trans_mat,
                                                                                                        train_y,
                                                                                                        train_nonviolation_mask,
                                                                                                        this_train_session)
    dir_to_check = path_analysis_glm_hmm + '/Model/glmhmm_#state=' + str(K) + '/fld_num=' + str(fold) + '/'
    test_ll_vals_across_iters, train_for_arranging_initials, file_ordering_by_train = log_likelihood_for_test_glmhmm(
        globe, prior_sigma, dir_to_check, test_datas, test_inputs, test_inputs_trans, test_nonviolation_masks, K, D, M,
        M_trans, C)
    train_ll_vals_across_iters, _, _ = log_likelihood_for_test_glmhmm(
        globe, prior_sigma, dir_to_check, train_datas, train_inputs, train_inputs_trans, train_nonviolation_masks, K, D,
        M, M_trans, C)
    test_ll_vals_across_iters = test_ll_vals_across_iters[file_ordering_by_train]
    train_ll_vals_across_iters = train_ll_vals_across_iters[file_ordering_by_train]
    glm_log_likelihood_hmm_this_K = test_ll_vals_across_iters[0]
    cvbt_thismodel_thisfold = cross_valid_bpt_compute(glm_log_likelihood_hmm_this_K, ll0, n_test)
    train_cvbt_thismodel_thisfold = cross_valid_bpt_compute(train_ll_vals_across_iters[0],
                                                           ll0_train, n_train)
    return cvbt_thismodel_thisfold, train_cvbt_thismodel_thisfold, glm_log_likelihood_hmm_this_K, train_ll_vals_across_iters[
        0], train_for_arranging_initials


def return_glm_log_likelihoodhmm_individual_animal(animal_path_data, path_analysis_glm_hmm, animal, num_inputs, fold, K, D, C,
                                       ibl_data=False):
    mouse_data = animal_path_data + animal + '_processed.npz'
    animal_fold_mapping_session = load_fold_session_map(
        animal_path_data + animal + '_fold_session_map.npz')
    if ibl_data is False:
        animal_obs_mat, animal_trans_mat, animal_y, animal_session = get_mouse_info(mouse_data)
    else:
        animal_obs_mat, animal_y, animal_session = get_old_ibl_data(mouse_data)
    if num_inputs == 0:
        animal_obs_mat = animal_obs_mat[:, [0]]
    elif num_inputs == 1:
        animal_obs_mat = animal_obs_mat[:, [0, 1]]
    elif num_inputs == 2:
        animal_obs_mat = animal_obs_mat[:, [0, 1, 2]]

    animal_test_obs_mat, animal_test_obs_mat_tran, animal_test_y, animal_test_nonviolation_mask, animal_this_test_session, animal_train_obs_mat, animal_train_obs_mat_tran, animal_train_y, animal_train_nonviolation_mask, animal_this_train_session, M, M_trans, _, _ = data_for_cross_validation(
        animal_obs_mat, animal_trans_mat, animal_y, animal_session, animal_fold_mapping_session, fold)
    animal_test_y[animal_test_nonviolation_mask == 0, :] = 1
    test_inputs, test_inputs_trans, test_datas, test_nonviolation_masks = data_segmentation_session(animal_test_obs_mat,
                                                                                                    animal_test_obs_mat_tran,
                                                                                                    animal_test_y,
                                                                                                    animal_test_nonviolation_mask,
                                                                                                    animal_this_test_session)
    animal_train_y[animal_train_nonviolation_mask == 0, :] = 1
    train_inputs, train_inputs_trans, train_datas, train_nonviolation_masks = data_segmentation_session(
        animal_train_obs_mat, animal_train_obs_mat_tran, animal_train_y,
        animal_train_nonviolation_mask,
        animal_this_train_session)
    dir_to_check = path_analysis_glm_hmm + '/Model/glmhmm_#state=' + str(K) + '/fld_num=' + str(fold) + '/'
    test_ll_vals_across_iters, train_for_arranging_initials, file_ordering_by_train = log_likelihood_for_test_glmhmm(
        dir_to_check, test_datas, test_inputs, test_inputs_trans, test_nonviolation_masks, K, D, M, M_trans, C)
    train_ll_vals_across_iters, _, _ = log_likelihood_for_test_glmhmm(
        dir_to_check, train_datas, train_inputs_trans, train_inputs, train_nonviolation_masks, K, D, M, M_trans, C)
    test_ll_vals_across_iters = test_ll_vals_across_iters[file_ordering_by_train]
    train_ll_vals_across_iters = train_ll_vals_across_iters[file_ordering_by_train]
    glm_log_likelihood_hmm_this_K = test_ll_vals_across_iters[0]
    train_glm_log_likelihood_hmm_this_K = train_ll_vals_across_iters[0]
    return glm_log_likelihood_hmm_this_K, train_glm_log_likelihood_hmm_this_K


def cross_valid_bpt_compute(ll_model, ll_0, n_trials):
    cv_bit_trial = ((ll_model - ll_0) / n_trials) / np.log(2)
    return cv_bit_trial

def get_min_train_test_fold_size(path_data, animal):
    # Get minimum train and test set size across folds:
    obs_mat, trans_mat, y, session, left_probs, animal_eids = get_mouse_info(path_data + animal + '_processed.npz')
    fold_mapping_session = load_fold_session_map(path_data + animal + '_fold_session_map.npz')
    min_test_size = 100000
    min_train_size = 100000
    for fold in range(5):
        train_appropriate_sessions = fold_mapping_session[np.where(fold_mapping_session[:, 1] != fold), 0]
        train_indexes_sess_fold = [str(sess) in train_appropriate_sessions and y[id, 0] != -1 for id, sess in
                               enumerate(session)]
        test_appropriate_sessions = fold_mapping_session[np.where(fold_mapping_session[:, 1] == fold), 0]
        test_indexes_sess_fold = [str(sess) in test_appropriate_sessions and y[id, 0] != -1 for id, sess in
                              enumerate(session)]
        num_train = np.sum(train_indexes_sess_fold)
        num_test = np.sum(test_indexes_sess_fold)
        if num_train < min_train_size:
            min_train_size = num_train
        if num_test < min_test_size:
            min_test_size = num_test
    return min_train_size, min_test_size

def make_cross_valid_for_figure_indiv(cv_file):
    diff_folds_fit = cross_validation_vector(cv_file)
    glm_lapse_model = diff_folds_fit[:3,]
    idx = np.array([0, 3,4,5,6])
    diff_folds_fit = diff_folds_fit[idx,:]
    # Identify best cvbt:
    mean_cvbt = np.mean(diff_folds_fit, axis=1)
    loc_best = np.where(mean_cvbt == max(mean_cvbt))[0]
    best_val = max(mean_cvbt)

    # Create dataframe for plotting
    fits_num = diff_folds_fit.shape[0]
    cross_valid_num_fold = diff_folds_fit.shape[1]
    # Create pandas dataframe:
    data_for_plotting_df = pd.DataFrame(
        {'model': np.repeat(np.arange(fits_num), cross_valid_num_fold), 'cv_bit_trial': diff_folds_fit.flatten()})

    return data_for_plotting_df, loc_best, best_val, glm_lapse_model
