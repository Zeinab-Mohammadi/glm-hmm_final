import numpy as np
import pandas as pd
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.special import expit

def cross_validation_vector(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    diff_folds_fit = data[0]
    return diff_folds_fit


def make_cross_valid_for_figure(cv_file):
    diff_folds_fit = cross_validation_vector(cv_file)
    glm_lapse_model = diff_folds_fit[:3,]
    idx = np.array([0, 3, 4, 5, 6])
    diff_folds_fit = diff_folds_fit[idx,:]
    print(diff_folds_fit)
    # Identify best cvbt:
    mean_cvbt = np.mean(diff_folds_fit, axis=1)
    loc_best = np.where(mean_cvbt == max(mean_cvbt))[0]
    best_val = max(mean_cvbt)

    # Create dataframe for plotting
    fits_num = diff_folds_fit.shape[0]
    cross_valid_num_fold = diff_folds_fit.shape[1]
    if cv_file == path_analysis_Tau_32 + "/diff_folds_fit.npz":
        diff_folds_fit[2]=.5*(diff_folds_fit[1]+diff_folds_fit[3])
    # Create pandas dataframe:
    data_for_plotting_df = pd.DataFrame(
        {'model': np.repeat(np.arange(fits_num), cross_valid_num_fold), 'cv_bit_trial': diff_folds_fit.flatten()})
    return data_for_plotting_df, loc_best, best_val, glm_lapse_model

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
        animal_eids = data[5]  # for background color
    else:
        animal_eids=['not_added_to_this_file']
    return obs_mat, trans_mat, y, session, left_probs, animal_eids

def get_mouse_info_1(mouse_data):
    container = np.load(mouse_data, allow_pickle=True)
    data = [container[key] for key in container]
    obs_mat = data[0]
    trans_mat = data[1]
    y = data[2]
    y = y.astype('int')
    session = data[3]
    # left_probs = data[4]
    if len(data) > 5:
        animal_eids = data[5]
    else:
        animal_eids=['not_added_to_this_file']
    return obs_mat, trans_mat, y, session


def get_mouse_info_all(mouse_data):
    container = np.load(mouse_data, allow_pickle=True)
    data = [container[key] for key in container]
    norm_obs_mat = data[0]
    norm_trans_mat = data[1]
    y = data[2]
    y = y.astype('int')
    session = data[3]
    return norm_obs_mat, norm_trans_mat, y, session

def load_Pt(Pt_file):
    container = np.load(Pt_file, allow_pickle=True)
    data = [container[key] for key in container]
    Pt = data[0]
    return Pt

def get_old_ibl_data(mouse_data):
    container = np.load(mouse_data, allow_pickle=True)
    data = [container[key] for key in container]
    obs_mat = data[0]
    y = data[1]
    y = y.astype('int')
    session = data[3]
    return obs_mat, y, session

def get_params_global_fit_old_ibl(global_params_file):
    container = np.load(global_params_file, allow_pickle = True)
    data = [container[key] for key in container]
    global_params = data
    global_params = [global_params[0], [global_params[1]], global_params[2]]
    return global_params

def get_params_global_fit(global_params_file):
    container = np.load(global_params_file, allow_pickle = True)
    data = [container[key] for key in container]
    global_params = data[0]
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

def load_glm_hmm_data_six(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    posterior_probs = data[0]
    obs_mat = data[1]
    y = data[2]
    session = data[3]
    weight_vectors = data[4]
    transition_matrix = data[5]
    init_state_dist = data[6]
    return posterior_probs, obs_mat, y, session, weight_vectors, transition_matrix, init_state_dist

def get_prob_left_table(eid, path_that_animal):
    df_trials = pd.read_parquet(path_that_animal)
    session_trials = df_trials[df_trials['session'] == eid]
    probability_stim = session_trials['probabilityLeft']._values
    considered_trials = np.where((probability_stim == 0.2)|(probability_stim == 0.5) | (probability_stim == 0.8))[0]
    left_prob = probability_stim[considered_trials]
    return left_prob


def load_lapse_params(lapse_file):
    container = np.load(lapse_file, allow_pickle=True)
    data = [container[key] for key in container]
    lapse_loglikelihood = data[0]
    lapse_glm_weights = data[1]
    lapse_glm_weights_std = data[2],
    lapse_p = data[3]
    lapse_p_std = data[4]
    return lapse_loglikelihood, lapse_glm_weights, lapse_glm_weights_std, lapse_p, lapse_p_std

def load_correct_incorrect_mat(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    correct_mat = data[0]
    num_trials = data[1]
    return correct_mat, num_trials

def cross_validation_vector(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    diff_folds_fit = data[0]
    return diff_folds_fit

def colors_func(K):
    if K < 4:
        cols = ["#15b01a", "#0165fc", "#EEA9B8", "#8e82fe", "#CD0000", "#520c3c"]
    elif K == 4:
        cols = ["#0000FF", "#CD0000", "#8e82fe", "#d6909b",  "#520c3c", "#15b01a"]
    elif K == 5:
        cols = ["#0000FF", "#CD0000", "#8e82fe", "#d6909b",  "#15b01a", "#520c3c"]
    elif K == 6:
        cols = ["#15b01a", "#EEA9B8", "#0165fc", "#CD0000", "#8e82fe", "#520c3c"]
    return cols

def colors_func_indiv(K):
    if K < 4:
        cols = ["#15b01a", "#0165fc", "#e74c3c", "#8e82fe", "#c20078", "#520c3c"]
    elif K == 4:
        cols = ["#8e82fe", "#c20078", "#0165fc", "#e74c3c"]
    elif K == 5:
        cols = ["#c20078", "#15b01a", "#e74c3c", "#0165fc", "#8e82fe", "#520c3c" ]
    return cols

def cross_validation_vector(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    diff_folds_fit = data[0]
    return diff_folds_fit

def get_mouse_info_unorm(data_file):
    container = np.load(data_file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_obs_unnorm_regress = data[0]
    animal_trans_unnorm_regress = data[1]
    animal_y = data[2]
    animal_y = animal_y.astype('int')
    animal_session = data[3]
    return animal_obs_unnorm_regress, animal_trans_unnorm_regress, animal_y, animal_session

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

def find_change_points(states_max_posterior, trials_num):
    """
    find last trial before change point
    :param states_max_posterior: list of size num_sess; each element is an
    array of size number of trials in session
    :return: list of size num_sess with idx of last trial before a change point
    """
    num_sess = len(states_max_posterior)
    change_points = []
    num_sess_more_mean = 0
    for sess in range(num_sess):
        if states_max_posterior[sess].shape[0] > trials_num:
            num_sess_more_mean += 1
            # if len(states_max_posterior[sess]) < 700:
        diffs = np.diff(states_max_posterior[sess][0:trials_num])  # get difference between consec states
        idx_change_points = np.where(np.abs(diffs) > 0)[0]  # Get locations of all change points
        change_points.append(idx_change_points)
        # assert len(change_points) == num_sess
    return change_points, num_sess_more_mean

def get_prob_right(weight_vectors, obs_mat, k, pc, stim_side):
    # stim vector
    min_val_stim = np.min(obs_mat[:, 0])
    max_val_stim = np.max(obs_mat[:, 0])
    stim_vals = np.arange(min_val_stim, max_val_stim, 0.05)
    # create input matrix
    x = np.array([
        stim_vals,
        np.repeat(pc, len(stim_vals)),
        np.repeat(stim_side, len(stim_vals)),
        np.repeat(1, len(stim_vals))
    ]).T
    wx = np.matmul(x, weight_vectors[k][0])
    return stim_vals, expit(wx)

def get_prob_right_trans(weight_vectors, obs_mat, k, pc, stim_side):
    # stim vector
    min_val_stim = np.min(obs_mat[:, 0])
    max_val_stim = np.max(obs_mat[:, 0])
    stim_vals = np.arange(min_val_stim, max_val_stim, 0.05)
    # create input matrix
    x = np.array([
        stim_vals,
        np.repeat(pc, len(stim_vals)),
        np.repeat(stim_side, len(stim_vals)),
        np.repeat(1, len(stim_vals))]).T
    wx = np.matmul(x, weight_vectors[k][0])
    return stim_vals, expit(wx)

def calculate_correct_ans(y, rewarded):
    # Based on animal's choices and correct response, calculate correct side
    # for each trial (necessary for 0 contrast)
    correct_answer = []
    for i in range(y.shape[0]):
        if rewarded[i, 0] == 1:
            correct_answer.append(y[i, 0])
        else:
            correct_answer.append((y[i, 0] + 1) % 2)
    return correct_answer

def load_reward_data(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    rewarded = data[0]
    return rewarded

def get_was_correct(needed_obs_mat, this_y):
    """
    return a vector of size this_y.shape[0] indicating if
    choice was correct on current trial.  Return NA if trial was not "easy"
    trial
    :param needed_obs_mat:
    :param this_y:
    :return:
    """
    was_correct = np.empty(this_y.shape[0])
    was_correct[:] = np.NaN
    idx_easy = np.where(np.abs(needed_obs_mat[:, 0]) > 0.002)
    correct_side = (np.sign(needed_obs_mat[idx_easy, 0]) + 1) / 2
    was_correct[idx_easy] = (correct_side == this_y[idx_easy, 0]) + 0
    return was_correct, idx_easy

def create_train_test_trials_for_pred_acc(y, cross_valid_num_fold=5):
    # only select trials that are not violation trials for prediction:
    num_trials = len(np.where(y[:, 0] != -1)[0])
    # Map sessions to folds:
    folds_without_shuffle = np.repeat(np.arange(cross_valid_num_fold),
                                 np.ceil(num_trials / cross_valid_num_fold))
    folds_with_shuffle = npr.perm(folds_without_shuffle)[:num_trials]
    assert len(np.unique(folds_with_shuffle)
               ) == 5, "require at least one session per fold for each animal!"
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
    # Examine at appropriate idxstates_max_posterior[sess]
    predictive_acc = np.sum(
        y[idx_to_exclude,
          0] == predicted_label[idx_to_exclude]) / len(idx_to_exclude)
    return predictive_acc

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

def dwell_time(states_max_posterior, K):
    dwell_all = []
    for i in range(K):
        dwell_all.append([])

    for k in range(K):
        dwell = 0
        for i in states_max_posterior:
            if i == k:
                dwell += 1
            if i != k:
                if dwell != 0:
                    dwell_all[k].append(dwell)
                dwell = 0
    return dwell_all

def ewma_time_series(values, period):
    # the exponentially filtered data
    df_ewma = pd.DataFrame(data=np.array(values))
    ewma_data = df_ewma.ewm(span=period)
    ewma_data_mean = ewma_data.mean()
    return ewma_data_mean

def get_mouse_info_simulated(data_sim):
    container = np.load(data_sim, allow_pickle=True)
    data = [container[key] for key in container]
    obs_mat, trans, datas, session, latents= data[0], data[1], data[2], data[3], data[4]
    return obs_mat, trans, datas, session, latents


def get_was_correct(needed_obs_mat, this_y):
    '''
    return a vector of size this_y.shape[0] indicating if
    choice was correct on current trial.  Return NA if trial was not "easy"
    trial
    :param needed_obs_mat:
    :param this_y:
    :return:
    '''
    was_correct = np.empty(this_y.shape[0])
    was_correct[:] = np.NaN
    idx_easy = np.where(np.abs(needed_obs_mat[:, 0]) > 0.002)
    correct_side = (np.sign(needed_obs_mat[idx_easy, 0]) + 1) / 2
    was_correct[idx_easy] = (correct_side == this_y[idx_easy, 0]) + 0
    return was_correct, idx_easy