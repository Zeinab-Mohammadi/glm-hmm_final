import numpy as np
import ssm
import copy
import json
import numpy.random as npr
import math
from io_utils import cross_validation_vector, model_data_glmhmm


def makeRaisedCosBasis(bias_num):
    """
    Create a Raised Cosine Basis function.
    """
    num_trials_consider_basis = 100
    nB = bias_num  # number of basis functions
    peakRange = [0, num_trials_consider_basis]  # number of trials in a session
    timeRange = [0, num_trials_consider_basis]

    def raisedCosFun(x, ctr, dCtr):  # Define function for single raised cosine basis function
        return (np.cos(np.maximum(-math.pi, np.minimum(math.pi, (x - ctr) * math.pi / dCtr / 2))) + 1) / 2

    # Compute location for cosine basis centers
    dCtr = np.diff(peakRange) / (nB - 1)  # spacing between raised cosine peaks
    Bctrs = np.arange(peakRange[0], peakRange[1] + .1, dCtr)  # peaks for cosine basis vectors
    basisPeaks = Bctrs  # vector of raised cosine centers
    # if isempty(timeRange)
    #     minT = peakRange(1)-2*dCtr  # min time bin (where 1st basis vector starts)
    #     maxT = peakRange(2)+2*dCtr  # max time bin (where last basis vector stops)
    # end
    dt = 1
    minT = timeRange[0]
    maxT = timeRange[1]
    tgrid = np.arange(minT, maxT + .1, dt)  # time grid
    nT = tgrid.shape[0]  # number of time points in basis
    # Make the basis
    cosBasis = raisedCosFun(np.tile(tgrid, (nB, 1)).T, np.tile(Bctrs, (nT, 1)), dCtr)
    return cosBasis, tgrid, basisPeaks


def get_params_global_fit(global_params_file):
    container = np.load(global_params_file, allow_pickle = True)
    data = [container[key] for key in container]
    global_params = data[0]
    return global_params


def calculate_posterior_given_data_GLM_O(globe, inputs, datas, Params_model, K, perm, transition_alpha, prior_sigma):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M = inputs[0].shape[1]
    D = datas[0].shape[1]
    if globe == True:
        prior_sigma = 100
    this_hmm = ssm.HMM(K, D, M, observations="input_driven_obs", observation_kwargs=dict(C=2, prior_sigma=prior_sigma),
                       transitions="sticky", transition_kwargs=dict(alpha=transition_alpha, kappa=0))
    this_hmm.params = Params_model
    # Get expected states:
    expectations = [this_hmm.expected_states(data=data, input=input)[0]  
                    for data, input
                    in zip(datas, inputs)]
    # Convert this now to one array:
    posterior_probs = np.concatenate(expectations, axis=0)
    posterior_probs = posterior_probs[:, perm]
    return posterior_probs



def state_weights_calcu_bias(Params_model):
    wights = -Params_model[2]
    M = wights.shape[2] - 1
    weights_seq = np.sort(wights[:, 0, 0])
    loc1 = np.where((wights[:, 0, 0] == weights_seq[-1]))[0][0]
    weights_remove = np.copy(wights)
    weights_remove[loc1, 0, M] = max(wights[:, 0, M]) - 0.001
    loc2 = np.where((weights_remove[:, 0, 0] == weights_seq[-2]))[0][0]
    loc_r = np.where((weights_remove[:, 0, M] == max(weights_remove[:, 0, M])))[0][0]
    loc_f = np.where((weights_remove[:, 0, M] == min(weights_remove[:, 0, M])))[0][0]
    la_R = np.array([loc_r, loc1, loc_f, loc2])
    la_L = np.array([loc1, loc_r, loc2, loc_f])
    return la_L, la_R


def state_occupancies_bias(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma, left_probs):
    label = range(K)
    posterior_probs, P = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma, label)
    bias = 'Left'
    state_occupancies_L = state_max_bias_all(posterior_probs,Params_model, left_probs, bias, K)
    bias = 'Right'
    state_occupancies_R = state_max_bias_all(posterior_probs,Params_model, left_probs, bias, K)
    state_occupancies_R_lab = copy.deepcopy(state_occupancies_R)
    state_occupancies_L_lab = copy.deepcopy(state_occupancies_L)
    return state_occupancies_L_lab, state_occupancies_R_lab


def state_max_bias_all(posterior_probs, Params_model, left_probs, bias, K):
    states_max_posterior = np.argmax(posterior_probs, axis=1)
    la_L, la_R = state_weights_calcu_bias(Params_model)
    state_occupancies = []
    T = 0
    if bias == 'Left':
        left_bias_index = np.where(left_probs == 0.2)
        states_max_posterior_bias = states_max_posterior[left_bias_index]
        for k in range(K):
            # Get state occupancy:
            occ = len(np.where(states_max_posterior_bias == k)[0]) / len(states_max_posterior_bias)
            state_occupancies.append(occ)
        state_occupancies_l = copy.deepcopy(state_occupancies)
        for k in range(K):
            state_occupancies_l[k] = state_occupancies[la_L[k]]
    if bias == 'Right':
        left_bias_index = np.where(left_probs == 0.8)
        states_max_posterior_bias = states_max_posterior[left_bias_index]
        for k in range(K):
            # Get state occupancy:
            occ = len(np.where(states_max_posterior_bias == k)[0]) / len(states_max_posterior_bias)
            state_occupancies.append(occ)
        state_occupancies_l = copy.deepcopy(state_occupancies)
        for k in range(K):
            state_occupancies_l[k] = state_occupancies[la_R[k]]
    return state_occupancies_l

# def state_occupancies_final(state_occupancies_R_all, state_occupancies_L_all, Params_model):
#     state_occupancies_R_m = np.mean(state_occupancies_R_all, axis=0)
#     print('state_occupancies_R_all.shape=', np.array(state_occupancies_R_all).shape)
#     print('state_occupancies_R_m.shape=', np.array(state_occupancies_R_m).shape)
#     state_occupancies_L_m = np.mean(state_occupancies_L_all, axis=0)
    # state_occupancies_R_final = copy.deepcopy(state_occupancies_R_m)
    # wights = -Params_model[2]
    # K = 4
    # M = wights.shape[2] - 1
    # weights_seq = np.sort(wights[:, 0, 0])
    # loc1 = np.where((wights[:, 0, 0] == weights_seq[-1]))[0][0]
    # weights_remove = np.copy(wights)
    # weights_remove[loc1, 0, M] = max(wights[:, 0, M]) - 0.001
    # loc2 = np.where((weights_remove[:, 0, 0] == weights_seq[-2]))[0][0]
    # loc_r = np.where((weights_remove[:, 0, M] == max(weights_remove[:, 0, M])))[0][0]
    # loc_f = np.where((weights_remove[:, 0, M] == min(weights_remove[:, 0, M])))[0][0]
    # label_L = np.array([loc_r, loc1, loc_f, loc2])
    # for k in range(K):
    #     state_occupancies_R_final[k] = state_occupancies_R_m[label_L[k]]
    # label_R = np.array([loc2, loc_r, loc1, loc_f])
    # state_occupancies_L_final = copy.deepcopy(state_occupancies_L_m)
    # for k in range(K):
    #     state_occupancies_L_final[k] = state_occupancies_L_m[label_R[k]]
    # return state_occupancies_L_final, state_occupancies_R_final

def calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, transition_alpha, prior_sigma, label):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M_trans = np.array(inputs_trans[0]).shape[1]
    M = inputs[0].shape[1]
    D = datas[0].shape[1]
    if globe == True:
        prior_sigma = 100
    this_hmm = ssm.HMM_TO(K, D, M_trans=M_trans, M_obs=M, observations="input_driven_obs_diff_inputs",
                          observation_kwargs=dict(C=2, prior_sigma=prior_sigma),
                          transitions="inputdrivenalt", transition_kwargs=dict(prior_sigma=prior_sigma, alpha=transition_alpha, kappa=0))
    this_hmm.params = Params_model

    # Get expected states:
    expectations = [this_hmm.expected_states(data=data, transition_input=input_trans, observation_input=input)[0]
                    for data, input_trans, input
                    in zip(datas, inputs_trans, inputs)]
    Ps = [this_hmm.Ps_matrix(data=data, transition_input=input_trans, observation_input=input)
                    for data, input_trans, input
                    in zip(datas, inputs_trans, inputs)]
    # Convert this now to one array:
    posterior_probs= np.concatenate(expectations, axis=0)
    Ps_all = np.concatenate(Ps, axis=0)
    posterior_probs = posterior_probs[:, label]
    return posterior_probs, Ps_all



def calculate_posterior_given_data_post(globe, inputs, inputs_trans, datas, Params_model, K, transition_alpha, prior_sigma, perm):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M_trans = np.array(inputs_trans[0]).shape[1]
    M = inputs[0].shape[1]
    D = datas[0].shape[1]

    if globe == True:
        prior_sigma = 100

    this_hmm = ssm.HMM_TO(K, D, M_trans = M_trans, M_obs=M, observations="input_driven_obs_diff_inputs",
                          observation_kwargs=dict(C=2, prior_sigma=prior_sigma),
                          transitions = "inputdrivenalt", transition_kwargs=dict(prior_sigma=prior_sigma,alpha=transition_alpha, kappa=0))
    this_hmm.params = Params_model
    # Get expected states:
    expectations = [this_hmm.expected_states(data = data, transition_input = input_trans, observation_input=input)[0]
                    for data, input_trans, input
                    in zip(datas, inputs_trans, inputs)]
    # Convert this now to one array:
    posterior_probs = np.concatenate(expectations, axis=0)
    posterior_probs = posterior_probs[:, perm]
    return posterior_probs


def get_global_weights(global_directory, K):
    cv_file = global_directory + "/diff_folds_fit.npz"
    diff_folds_fit = cross_validation_vector(cv_file)
    with open(global_directory + "/optimal_initialize_dict.json", 'r') as f:
        optimal_initialize_dict = json.load(f)
    params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, global_directory, optimal_initialize_dict)
    Params_model, lls = model_data_glmhmm(params_and_LL)
    # perm = find_corresponding_states(Params_model)
    return Params_model


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
    test_obs_mat, test_trans_mat, test_y, test_mask, this_test_session = obs_mat[idx_test, :], trans_mat[idx_test, :], y[idx_test, :], mask[idx_test], session[idx_test]
    train_obs_mat, train_trans_mat, train_y, train_mask, this_train_session = obs_mat[idx_train, :], trans_mat[idx_train, :], y[idx_train, :], mask[idx_train], session[idx_train]
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
    unique_sessions = [session[index] for index in sorted(indexes)]  # ensure that unique sessions are ordered as they are in session
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


def state_max_bias_all_simulated_data(animal_latents, left_probs, bias, K):
    states_max_posterior = animal_latents
    state_occupancies = []
    T = 0
    if bias == 'Left':
        left_bias_index = np.where(left_probs == 0.2)
        states_max_posterior_bias = states_max_posterior[left_bias_index]
        for k in range(K):
            # Get state occupancy:
            occ = len(np.where(states_max_posterior_bias == k)[0]) / len(states_max_posterior_bias)
            state_occupancies.append(occ)
        state_occupancies_l = copy.deepcopy(state_occupancies)
    if bias == 'Right':
        left_bias_index = np.where(left_probs == 0.8)
        states_max_posterior_bias = states_max_posterior[left_bias_index]
        for k in range(K):
            # Get state occupancy:
            occ = len(np.where(states_max_posterior_bias == k)[0]) / len(states_max_posterior_bias)
            state_occupancies.append(occ)
        state_occupancies_l = copy.deepcopy(state_occupancies)
    return state_occupancies_l


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

def get_file_name_for_best_model_fold_indiv(diff_folds_fit, K, path_main_folder, optimal_initialize_dict):
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
    base_path = path_main_folder + '/GLM_HMM_K_' + str(K) + '/fold_' + str(best_fold)
    keys = '/GLM_HMM_K_' + str(K) + '/fold_' + str(best_fold)
    best_iter = optimal_initialize_dict[keys]
    params_and_LL = base_path + '/iter_' + str(best_iter) + '/glm_hmm_raw_parameters_itr_' + str(best_iter) + '.npz'
    return params_and_LL

def permute_transition_matrix(transition_matrix, perm):
    transition_matrix = transition_matrix[np.ix_(perm, perm)]
    return transition_matrix

def find_corresponding_states(Params_model):
    glm_weights = -Params_model[2]
    K = glm_weights.shape[0]
    if K == 3:
        M = glm_weights.shape[2] - 1
        engaged_loc = np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        reduced_weights = np.copy(glm_weights)
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        state_order = [engaged_loc, bias_left_loc]
        bias_right_loc = np.arange(3)[np.where([range(3)[i] not in state_order for i in range(3)])][0]
        perm = np.array([engaged_loc, bias_left_loc, bias_right_loc])
    elif K == 4:
        M = glm_weights.shape[2] - 1
        engaged_loc = np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        reduced_weights = np.copy(glm_weights)
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        bias_right_loc = np.where((reduced_weights[:, 0, M] == max(reduced_weights[:, 0, M])))[0][0]
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        state_order = [engaged_loc, bias_left_loc, bias_right_loc]
        other_loc = np.arange(4)[np.where([range(4)[i] not in state_order for i in range(4)])][0]
        perm = np.array([engaged_loc, bias_left_loc, bias_right_loc, other_loc])
    else:
        perm = np.argsort(-glm_weights[:, 0, 0])
    return perm


def find_corresponding_states_new(Params_model):
    glm_weights = -Params_model[2]
    K = glm_weights.shape[0]
    if K == 3:
        M = glm_weights.shape[2] - 1
        engaged_loc = np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        reduced_weights = np.copy(glm_weights)
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        state_order = [engaged_loc, bias_left_loc]
        bias_right_loc = np.arange(3)[np.where([range(3)[i] not in state_order for i in range(3)])][0]
        perm = np.array([engaged_loc, bias_left_loc, bias_right_loc])
    elif K == 4:
        M = glm_weights.shape[2] - 1
        weights_sort = np.sort(glm_weights[:, 0, 0])
        engaged_loc1 = np.where((glm_weights[:, 0, 0] == weights_sort[-1]))[0][0]
        reduced_weights = np.copy(glm_weights)
        reduced_weights[engaged_loc1, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        engaged_loc2 = np.where((reduced_weights[:, 0, 0] == weights_sort[-2]))[0][0]
        bias_right_loc = np.where((reduced_weights[:, 0, M] == max(reduced_weights[:, 0, M])))[0][0]
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        perm = np.array([engaged_loc1, engaged_loc2, bias_left_loc, bias_right_loc])
    elif K == 5:
        M = glm_weights.shape[2] - 1
        weights_sort = np.sort(glm_weights[:, 0, 0])
        engaged_loc1 = np.where((glm_weights[:, 0, 0] == weights_sort[-1]))[0][0]
        reduced_weights = np.copy(glm_weights)
        reduced_weights[engaged_loc1, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        engaged_loc2 = np.where((reduced_weights[:, 0, 0] == weights_sort[-2]))[0][0]
        bias_right_loc = np.where((reduced_weights[:, 0, M] == max(reduced_weights[:, 0, M])))[0][0]
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        state_order = [engaged_loc1, engaged_loc2, bias_left_loc, bias_right_loc]
        other_loc = np.arange(K)[np.where([range(K)[i] not in state_order for i in range(K)])][0]
        perm = np.array([engaged_loc1, engaged_loc2, bias_left_loc, bias_right_loc, other_loc])
    return perm


def calculate_predictive_accuracy(inputs, inputs_trans, datas, Params_model, K,
                                  perm, transition_alpha, prior_sigma,
                                  y, idx_to_exclude):
    if K == 4:
        perm = find_corresponding_states_new(Params_model)
    D, C = 1, 2
    M = inputs[0].shape[1]
    M_trans = inputs_trans[0].shape[1]
    this_hmm = ssm.HMM_TO(K, D, M_trans=M_trans, M_obs=M, observations="input_driven_obs_diff_inputs",
                      observation_kwargs=dict(C=C, prior_sigma=prior_sigma),
                      transitions="inputdrivenalt",
                      transition_kwargs=dict(prior_sigma=prior_sigma, alpha=transition_alpha, kappa=0))

    this_hmm.params = Params_model
    # Get expected states:
    expectations = [this_hmm.expected_states(data=data, transition_input=input_trans, observation_input=input)[0]
                    for data, input_trans, input
                    in zip(datas, inputs_trans, inputs)]

    # Convert this now to one array:
    posterior_probs = np.concatenate(expectations, axis=0)
    posterior_probs = posterior_probs[:, perm]
    prob_right = [np.exp(this_hmm.observations.calculate_logits(observation_input=input))
                  for data, input in zip(datas, inputs)]
    prob_right = np.concatenate(prob_right, axis=0)
    prob_right = prob_right[:, :, 1]
    final_prob_right = np.sum(np.multiply(posterior_probs, prob_right), axis=1)
    # Get the predicted label for each time step:
    predicted_label = np.around(final_prob_right, decimals=0).astype('int')
    # Examine at appropriate idx
    predictive_acc = np.sum(y[idx_to_exclude,0] == predicted_label[idx_to_exclude]) / len(idx_to_exclude)
    return predictive_acc


def create_train_test_trials_for_pred_acc(y, cross_valid_num_fold=5):
    # only select trials that are not violation trials for prediction:
    num_trials = len(np.where(y[:, 0] != -1)[0])
    # Map sessions to folds:
    folds_without_shuffle = np.repeat(np.arange(cross_valid_num_fold), np.ceil(num_trials / cross_valid_num_fold))
    folds_with_shuffle = np.random.permutation(folds_without_shuffle)[:num_trials]
    assert len(np.unique(folds_with_shuffle)) == 5, "require at least one session per fold for each animal!"
    # Look up table of shuffle-folds:
    folds_with_shuffle = np.array(folds_with_shuffle, dtype='O')
    trial_fold_lookup_table = np.transpose(
        np.vstack([np.where(y[:, 0] != -1), folds_with_shuffle]))
    return trial_fold_lookup_table


def calculate_predictive_acc_glm(glm_weights, obs_mat, y, idx_to_exclude):
    M = obs_mat.shape[1]
    C = 2
    # Calculate test loglikelihood
    from GLM_class import glm
    new_glm = glm(M, C)
    # Set parameters to fit parameters:
    new_glm.params = glm_weights
    # time dependent logits:
    prob_right = np.exp(new_glm.calculate_logits(obs_mat))
    prob_right = prob_right[:, 0, 1]
    # Get the predicted label for each time step:
    predicted_label = np.around(prob_right, decimals=0).astype('int')
    # Examine at appropriate idx
    predictive_acc = np.sum(y[idx_to_exclude, 0] == predicted_label[idx_to_exclude]) / len(idx_to_exclude)
    return predictive_acc

def load_correct_incorrect_mat(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    correct_mat = data[0]
    num_trials = data[1]
    return correct_mat, num_trials


