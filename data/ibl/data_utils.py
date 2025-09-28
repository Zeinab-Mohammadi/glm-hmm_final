"""

Assorted auxiliary functions for data processing and making designs matrices

"""

import math
import numpy as np
import numpy.random as npr
import json
import pandas as pd
from scipy.stats import bernoulli
from numpy import linalg as LA
from pathlib import Path
from one.alf.files import add_uuid_string
from one.remote import aws

def calculate_condition_number(obs_mat):
    full_obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))
    condition_number = LA.cond(full_obs_mat)
    # print("Condition number of input matrix = " + str(condition_number))
    return condition_number

def download_subjectTrials(one, target_path=None, tag='2023_Q1_Mohammadi_et_al', overwrite=False, check_updates=True):
    """
    Function to download the aggregated clusters information associated with the given data release tag from AWS.

    Parameters
    ----------
    one: one.api.ONE
        Instance to be used to connect to database.
    target_path: str or pathlib.Path
        Directory to which files should be downloaded. If None, downloads to one.cache_dir/aggregates
    tag: str
        Data release tag to download _ibl_subjectTrials.table datasets from. Default is '2023_Q1_Mohammadi_et_al'.
    overwrite : bool
        If True, will re-download files even if file exists locally and file sizes match.
    check_updates : bool
        If True, will check if file sizes match and skip download if they do. If False, will just return the paths
        and not check if the data was updated on AWS.

    Returns
    -------
    list of pathlib.Path
        Paths to the downloaded files
    """

    if target_path is None:
        target_path = Path(one.cache_dir).joinpath('aggregates')
        target_path.mkdir(exist_ok=True)
    else:
        assert target_path.exists(), 'The target_path you passed does not exist.'

    # Get the datasets
    datasets = one.alyx.rest('datasets', 'list', name='_ibl_subjectTrials.table.pqt', tag=tag)

    # Set up the bucket
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)

    out_paths = []
    for ds in datasets:
        relative_path = add_uuid_string(ds['file_records'][0]['relative_path'], ds['url'][-36:])
        src_path = 'aggregates/' + str(relative_path)
        dst_path = target_path.joinpath(relative_path)
        if check_updates:
            out = aws.s3_download_file(src_path, dst_path, s3=s3, bucket_name=bucket_name, overwrite=overwrite)
        else:
            out = dst_path

        if out and out.exists():
            out_paths.append(out)
        else:
            print(f'Downloading of {src_path} table failed.')
    return out_paths

def load_animal_eid_dict(file):
    with open(file, 'r') as f:
        animal_eid_dict = json.load(f)
    return animal_eid_dict

def data_segmentation_session(obs_mat, trans_mat, output, mask, session):
    """
    Partition obs_mat, trans_mat, output, mask by session
    :param obs_mat: arr of size TxM_obs
    :param trans_mat: arr of size TxM_trans
    :param output:  arr of size T x D
    :param mask: Boolean arr of size T indicating if element is violation or not
    :param session: list of size T containing session ids
    :return: list of obs_mat arrays, data arrays and mask arrays, where the number of elements in list = number of sessions and each array size is number of trials in session
    """
    inputs = []
    inputs_trans = []
    datas = []
    masks = []
    ii = 0
    indx = np.unique(session, return_index=True)[1]
    unique_sessions = [session[index] for index in sorted(
        indx)]  # ensure that unique sessions are ordered as they are in session (so we can map inputs back to obs_mat)
    for sess in unique_sessions:
        idx = np.where(session == sess)[0]
        ii += len(idx)
        inputs.append(obs_mat[idx, :])
        inputs_trans.append(trans_mat[idx, :])
        datas.append(output[idx, :])
        masks.append(mask[idx])
    assert ii == obs_mat.shape[0], "not all trials assigned to session!"
    return inputs, inputs_trans, datas, masks


def decisions_vector_new(decision):
    # raw choice vector has CW = 1 (correct response for stim on left), CCW = -1 (correct response for stim on right) and viol = 0.  Let's remap so that CW = 0, CCw = 1, and viol = -1
    decisions_new = {1: 0, -1: 1, 0: -1}
    decisions_vector_update = [decisions_new[old_decision] for old_decision in decision]
    return decisions_vector_update

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
    animal_inpt = data[0]
    trans_mat = data[1]
    animal_y = data[2]
    animal_session = data[3]
    left_probs = data[4]
    mice_session_eids = data[5]
    return animal_inpt, trans_mat, animal_y, animal_session, left_probs, mice_session_eids
def ewma_time_series(values, period):
    # to get the exponentially filtered data
    df_ewma = pd.DataFrame(data=np.array(values))
    ewma_data = df_ewma.ewm(span=period)
    ewma_data_mean = ewma_data.mean()
    return ewma_data_mean

def divide_sessions_for_test_train(session, cross_valid_num_fold=5):
    # create a session-fold lookup table
    sessions_count = len(np.unique(session))
    # Map sessions to folds:
    folds_without_shuffle = np.repeat(np.arange(cross_valid_num_fold), np.ceil(sessions_count / cross_valid_num_fold))
    folds_with_shuffle = npr.permutation(folds_without_shuffle)[:sessions_count]
    assert len(np.unique(folds_with_shuffle)) == 5, "require at least one session per fold for each mouse!"
    # Look up table of shuffle-folds:
    sess_id = np.array(np.unique(session), dtype='str')
    folds_with_shuffle = np.array(folds_with_shuffle, dtype='O')
    fold_mapping_session = np.transpose(np.vstack([sess_id, folds_with_shuffle]))
    return fold_mapping_session



def mask_for_violations(index_viols, T):
    """
    Return indices of nonviolations and also a Boolean mask for inclusion (1 = nonviolation; 0 = violation)
    """
    mask = np.array([i not in index_viols for i in range(T)])
    nonindex_viols = np.arange(T)[mask]
    mask = mask + 0
    assert len(nonindex_viols) + len(index_viols) == T, "violation and non-violation idx do not include all data!"
    return nonindex_viols, mask


def makeRaisedCosBasis(bias_num):
    # the basis function for transition matrix
    num_trials_consider_basis = 100  # as some sessions have length less than 100
    nB = bias_num  # number of basis functions
    peakRange = [0, num_trials_consider_basis]  # number of trials in a session
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

def mouse_identifier(eid_path):
    mouse = str(eid_path).split('tables_new/')[1]
    return mouse




def mice_names_info(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    mice_names = data[0]
    return mice_names


def mouse_eids_sessions_info(file):
    with open(file, 'r') as f:
        mice_session_eids = json.load(f)
    return mice_session_eids

def obs_inputs_matrix_values(decision, left_contrast, right_contrast, rewarded):
    # There is no Tau here for Observation
    stim = values_for_stimulus(left_contrast, right_contrast)
    T = len(stim)  # number of stimulus
    obs_inputs = np.zeros((T, 4))
    obs_inputs[:, 0] = stim
    # make choice vector so that correct response for stim>0 is choice =1 and is 0 for stim <0 (viol is mapped to -1)
    decision = decisions_vector_new(decision)
    # create past choice vector:
    past_decision, adapt_indexes = past_decision_values(decision)
    # create stim_side vector:
    stim_side, past_reward_value = stimulus_side_regressor(past_decision, rewarded, adapt_indexes)
    # map previous choice to {-1,1}
    obs_inputs[:, 1] = 2 * past_decision - 1
    obs_inputs[:, 2] = stim_side  # for (size of Tau) = 2
    return obs_inputs

def past_decision_values(decision):
    """
    decision: decision vector of size T
    past_decision : vector of size T with previous choice made by mouse - output is in {0, 1}, where 0 corresponds to a previous left choice; 1 corresponds to right.
    If the previous choice was a violation, replace this with the choice on the previous trial that was not a violation.
    adapt_indexes: array of size (~num_viols)x2, where the entry in column 1 is the location in the previous choice vector that was a remapping due to a violation and the
    entry in column 2 is the location in the previous choice vector that this location was remapped to
    """
    past_decision = np.hstack([np.array(decision[0]), decision])[:-1]  # all choice except the last/current one
    viols_indexes = np.where(past_decision == -1)[0]  # locations with violations
    nonviols_indexes = np.where(past_decision != -1)[0]  # locations with true choices
    first_decision_idx = nonviols_indexes[0]
    adapt_indexes = np.zeros((len(viols_indexes) - first_decision_idx, 2), dtype='int')

    for i, trial_indx in enumerate(viols_indexes):
        if trial_indx < first_decision_idx:  # if correct, then this is the first element of locations with true choices
            past_decision[trial_indx] = bernoulli.rvs(0.5, 1) - 1  # final output is in {0,1}
        else:
            # find nearest loc that has a previous choice value that is not -1, and that is earlier than current trial
            potential_matches = nonviols_indexes[np.where(nonviols_indexes < trial_indx)]
            absolute_val_diffs = np.abs(trial_indx - potential_matches)
            absolute_val_diffs_ind = absolute_val_diffs.argmin()
            nearest_loc = potential_matches[absolute_val_diffs_ind]
            adapt_indexes[i - first_decision_idx, 0] = int(trial_indx)
            adapt_indexes[i - first_decision_idx, 1] = int(nearest_loc)
            past_decision[trial_indx] = past_decision[nearest_loc]
    assert len(np.unique(past_decision)) <= 2, "previous choice should be in {0, 1}; " + str(
        np.unique(past_decision))
    return past_decision, adapt_indexes



def stimulus_side_regressor(past_decision, success, adapt_indexes):
    """
    inputs:
    success: vector of size T, entries are in {-1, 1} and 0 corresponds to failure, 1 corresponds to success
    past_decision: vector of size T, entries are in {0, 1} and 0 corresponds to left choice, 1 corresponds to right choice
    adapt_indexes: location remapping dictionary due to violations

    output:
    stim_side: vector of size T, entries are in {-1, 1}.  1 corresponds to previous choice = right and success OR previous choice = left and failure; -1 corresponds to
    previous choice = left and success OR previous choice = right and failure
    """
    # remap previous choice vals to {-1, 1}
    remapped_past_decision = 2 * past_decision - 1
    past_reward_value = np.hstack([np.array(success[0]), success])[:-1]  # all rewards except the last one

    # Now need to go through and update previous reward to correspond to same trial as previous choice:
    for i, trial_indx in enumerate(adapt_indexes[:, 0]):
        nearest_loc = adapt_indexes[i, 1]
        past_reward_value[trial_indx] = past_reward_value[nearest_loc]
    stim_side = past_reward_value * remapped_past_decision
    assert len(np.unique(stim_side)) == 2, "stim_side should be in {-1, 1}"
    return stim_side, past_reward_value

def tables_ibl_data(eid, path_that_mouse):
    eid_sess = eid
    mouse = path_that_mouse.split('tables_new/')[1]
    df_trials = pd.read_parquet(path_that_mouse)
    session_trials = df_trials[df_trials['session'] == eid]
    decision = session_trials['choice']._values
    left_contrast = session_trials['contrastLeft']._values
    right_contrast = session_trials['contrastRight']._values
    rewarded = session_trials['feedbackType']._values
    probability_stim = session_trials['probabilityLeft']._values
    return mouse, eid_sess, left_contrast, right_contrast, rewarded, decision, probability_stim


def values_for_stimulus(left_contrast, right_contrast):
    """
    Return right_contrast - left_contrast
    """
    # Replace NaNs with 0:
    left_contrast = np.nan_to_num(left_contrast, nan=0)
    right_contrast = np.nan_to_num(right_contrast, nan=0)
    # now get 1D stim
    signed_contrast = right_contrast - left_contrast
    return signed_contrast











def trans_inputs_matrix_values(choice, left_contrast, right_contrast, rewarded, Taus_tran):
    # This is the transition design mat
    Taus_size = np.array(Taus_tran).shape[0]
    num_basis = 3
    num_inputs_for_trans = num_basis + (3 * Taus_size)
    # Create obs_unnorm_regress: with first column = right_contrast - left_contrast, second column as past choice, third column as stim_side
    stim = values_for_stimulus(left_contrast, right_contrast)
    T = len(stim)  # number of stimulus
    trans_inputs = np.zeros((T, num_inputs_for_trans))
    # make choice vector so that correct response for stim>0 is choice =1 and is 0 for stim <0 (viol is mapped to -1)
    choice = decisions_vector_new(choice)
    # create past choice vector:
    past_decision, adapt_indexes = past_decision_values(choice)
    past_decision_in_range = 2 * past_decision - 1
    # making pc_exp_filtered
    for i, tau in enumerate(Taus_tran):
        trans_inputs[:, i] = ewma_time_series(past_decision_in_range, tau)[
            0]  # we don't need stimulus here so no i+1
        aa = np.sum(trans_inputs[:, i])
        if np.isnan(aa) == True:
            print('np.isnan(aa)=', np.isnan(aa).shape)

    # create stim_side vector:
    stim_side, past_reward_value = stimulus_side_regressor(past_decision, rewarded, adapt_indexes)
    for i, tau in enumerate(Taus_tran):
        trans_inputs[:, i + Taus_size] = ewma_time_series(stim_side, tau)[0]
    for i, tau in enumerate(Taus_tran):
        trans_inputs[:, i + 2 * Taus_size] = ewma_time_series(past_reward_value, tau)[0]
    cosBasis, tgrid, basisPeaks = makeRaisedCosBasis(num_basis)
    # below as the other elements with no basis are zero
    trans_inputs[0:cosBasis.shape[0], (3 * Taus_size): ((3 * Taus_size) + num_basis)] = cosBasis
    return trans_inputs


def session_unnorm_values(eid, path_that_mouse):
    # Load raw data
    mouse, eid_sess, left_contrast, right_contrast, rewarded, choice, probability_stim = tables_ibl_data(eid, path_that_mouse)
    considered_trials = np.where((probability_stim == 0.2) | (probability_stim == 0.5) | (probability_stim == 0.8))[0]
    violations_count = len(np.where(choice[considered_trials] == 0)[0])  # number of violations
    if violations_count < 50:
        # Create design mat = matrix of size T x 3, with entries for stim/past choice/stim_side
        # Taus = [1, 4, 16, 32 ]
        # Taus = [2, 8, 32]
        # Tau_obs= [8] # No tau for GLM OBS
        Taus_Tran = [4]  # [2], [8], [32]
        obs_unnorm_regress = obs_inputs_matrix_values(choice[considered_trials], left_contrast[considered_trials],
                                                  right_contrast[considered_trials], rewarded[considered_trials])
        trans_unnorm_regress = trans_inputs_matrix_values(choice[considered_trials], left_contrast[considered_trials],
                                                          right_contrast[considered_trials], rewarded[considered_trials],
                                                          Taus_Tran)
        left_probs = probability_stim[considered_trials]
        output = np.expand_dims(decisions_vector_new(choice[considered_trials]),
                           axis=1)  # np.expand_dims: Insert a new axis that will appear at the axis position in the expanded array shape
        session = [eid_sess for i in range(output.shape[0])]
        rewarded = np.expand_dims(rewarded[considered_trials], axis=1)
    else:
        Len = 90
        obs_unnorm_regress = np.zeros((Len, 3))
        trans_unnorm_regress = np.zeros((Len, 7))
        output = np.zeros((Len, 1))
        session = []
        left_probs = []
        rewarded = np.zeros((Len, 1))
    return mouse, obs_unnorm_regress, trans_unnorm_regress, output, session, violations_count, rewarded, left_probs






