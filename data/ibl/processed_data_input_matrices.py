"""

Processing the data and making the transition and observation matrices 

"""


import copy
import numpy as np
import numpy.random as npr
import json
import os
from collections import defaultdict
from sklearn import preprocessing
from data_utils import mice_names_info, mask_for_violations, data_segmentation_session, tables_ibl_data, session_unnorm_values, \
    calculate_condition_number, mouse_eids_sessions_info, divide_sessions_for_test_train 
npr.seed(65)

if __name__ == '__main__':
    num_sess_considered = 0
    min_sess_count = 40  # Require that each mouse has at least 40 sessions

    path_data = os.getcwd() + '/Della_cluster_data/'
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'

    # Load mouse list:
    mice_names = mice_names_info(os.getcwd() + '/mice_names.npz')
    # Load mouse-eid dict (keys are mice and vals are list of eids for biased block sessions)
    mice_session_eids = mouse_eids_sessions_info(os.getcwd() + '/mice_session_eids.json')

    for mouse in mice_names:
        sess_count = len(mice_session_eids[mouse])
        if sess_count < min_sess_count:
            mice_names = np.delete(mice_names, np.where(mice_names == mouse))

    # Identify idx in all_data array where each mouse's data starts and ends:
    mouse_init = {}
    mouse_final = {}
    eid_size_issue = []

    final_mice_session_eids = defaultdict(list)
    # WORKHORSE: iterate through each mouse and each mouse's set of eids; obtain unnorm data.  Write out each mouse's data and then also write to all_data array
    for z, mouse in enumerate(mice_names):
        print('z=', z)
        mouse_path = path_of_the_directory + mouse
        sess_counter = 0
        for eid in mice_session_eids[mouse]:  # for each eid of the mouse
            mouse, eid_sess, left_contrast, right_contrast, reward_succ, choice, probability_stim = tables_ibl_data(eid,
                                                                                                   mouse_path)
            if left_contrast.shape == right_contrast.shape == reward_succ.shape == choice.shape:
                mouse, obs_unnorm_regress, trans_unnorm_regress, output, session, violations_count, reward_succ, left_probs = session_unnorm_values(
                    eid, mouse_path)

                if violations_count < 50:  # Increasing this means more data. Only include session if number of viols is less than 50
                    num_sess_considered = num_sess_considered + 1
                    if sess_counter == 0:
                        mouse_obs_unnorm_regress = np.copy(obs_unnorm_regress)
                        mouse_trans_unnorm_regress = np.copy(trans_unnorm_regress)
                        mouse_output = np.copy(output)
                        mouse_session = session
                        mouse_left_probs = np.copy(left_probs)
                        mouse_rewarded = np.copy(reward_succ)
                    else:
                        mouse_obs_unnorm_regress = np.vstack((mouse_obs_unnorm_regress, obs_unnorm_regress))
                        mouse_trans_unnorm_regress = np.vstack(
                            (mouse_trans_unnorm_regress, trans_unnorm_regress))
                        mouse_output = np.vstack((mouse_output, output))
                        mouse_session = np.concatenate((mouse_session, session))
                        mouse_left_probs = np.concatenate((mouse_left_probs, left_probs))
                        mouse_rewarded = np.vstack((mouse_rewarded, reward_succ))
                    sess_counter += 1
                    final_mice_session_eids[mouse].append(eid)
            else:
                eid_size_issue.append(eid)

        # Write out mouse's unnorm data matrix:
        np.savez(path_data + 'separate_mouse_data/' + mouse + '_unnorm.npz', mouse_obs_unnorm_regress,
                 mouse_trans_unnorm_regress,
                 mouse_output, mouse_session)

        mouse_fold_session_map = divide_sessions_for_test_train (mouse_session, 5)  # why 5?
        np.savez(path_data + 'separate_mouse_data/' + mouse + "_fold_session_map" + ".npz",
                 mouse_fold_session_map)

        # mouse rewarded: reward for each mouse (one.get_mouse_infoset(eid, '_ibl_trials.feedbackType'))
        np.savez(path_data + 'separate_mouse_data/' + mouse + '_rewarded.npz', mouse_rewarded)
        assert mouse_rewarded.shape[0] == mouse_output.shape[0]

        # Now create or append data to all_data array across all mice:
        if z == 0:
            all_data_obs_mat = np.copy(mouse_obs_unnorm_regress)
            all_data_trans_mat = np.copy(mouse_trans_unnorm_regress)
            mouse_init[mouse] = 0
            mouse_final[mouse] = all_data_obs_mat.shape[0] - 1
            all_data_output = np.copy(mouse_output)
            all_data_session = mouse_session
            all_data_left_probs = np.copy(mouse_left_probs)
            all_data_fold_mapping_session = mouse_fold_session_map
            all_data_rewarded = np.copy(mouse_rewarded)

        else:
            mouse_init[mouse] = all_data_obs_mat.shape[0]
            all_data_obs_mat = np.vstack(
                (all_data_obs_mat, mouse_obs_unnorm_regress))
            all_data_trans_mat = np.vstack((all_data_trans_mat,
                                           mouse_trans_unnorm_regress))

            mouse_final[mouse] = all_data_obs_mat.shape[0] - 1  # number of all mice eids
            # size of all_data_output and all_data_session are the same
            all_data_output = np.vstack((all_data_output, mouse_output))
            all_data_session = np.concatenate((all_data_session,
                                             mouse_session))
            all_data_left_probs = np.concatenate((all_data_left_probs, mouse_left_probs))
            all_data_fold_mapping_session = np.vstack(
                (all_data_fold_mapping_session, mouse_fold_session_map))
            all_data_rewarded = np.vstack((all_data_rewarded, mouse_rewarded))

    Taus_trans_size = 1  # trans_obs_size
    i = 0
    for i in range(3):
        all_data_trans_mat[:, i + 1 + 2 * Taus_trans_size] = all_data_trans_mat[:, i + 1 + 2 * Taus_trans_size] - np.mean(all_data_trans_mat[:, i + 1 + 2 * Taus_trans_size])

    # Write out data from across mice
    assert np.shape(all_data_obs_mat)[0] == np.shape(all_data_output)[0], "obs_regressors and output not same length"
    assert np.shape(all_data_trans_mat)[0] == np.shape(all_data_output)[0], "trans_regressors and output not same length"
    assert np.shape(all_data_rewarded)[0] == np.shape(all_data_output)[0], "reward_succ and output not same length"
    assert len(np.unique(all_data_session)) == np.shape(all_data_fold_mapping_session)[0], "number of unique sessions and session fold lookup don't match"
    norm_obs_mat = np.copy(all_data_obs_mat)
    norm_trans_mat = np.copy(all_data_trans_mat)
    norm_obs_mat[:, 0] = preprocessing.scale(norm_obs_mat[:, 0])
    norm_trans_mat[:, 0] = preprocessing.scale(norm_trans_mat[:, 0])
    calculate_condition_number(preprocessing.scale(norm_obs_mat))
    calculate_condition_number(preprocessing.scale(norm_trans_mat))

    np.savez(path_data + 'combined_all_mice' + '.npz', norm_obs_mat, norm_trans_mat, all_data_output, all_data_session)
    np.savez(path_data + 'unnorm_all_mice' + '.npz', all_data_obs_mat, all_data_trans_mat, all_data_output, all_data_session)
    np.savez(path_data + 'fold_mapping_session_all_mice' + '.npz', all_data_fold_mapping_session)
    np.savez(path_data + 'rewarded_all_mice' + '.npz', all_data_rewarded)
    np.savez(path_data + 'separate_mouse_data/' + 'mice_names.npz', mice_names)

    json = json.dumps(final_mice_session_eids)
    f = open(path_data + "final_mice_session_eids.json", "w")
    f.write(json)
    f.close()

    # Now write out norm data (when norm across all mice) for each mouse:
    counter = 0
    all_sess_size = []
    for mouse in mouse_init.keys():
        init = mouse_init[mouse]
        final = mouse_final[mouse]
        obs_mat = norm_obs_mat[range(init, final + 1)]
        trans_mat = norm_trans_mat[range(init, final + 1)]
        output = all_data_output[range(init, final + 1)]
        session = all_data_session[range(init, final + 1)]  # this is all session for one mouse
        left_probs = all_data_left_probs[range(init, final + 1)]
        counter += obs_mat.shape[0]
        np.savez(path_data + 'separate_mouse_data/' + mouse + '_processed.npz', obs_mat, trans_mat, output,
                 session, left_probs, mice_session_eids[
                     mouse])

        index_viols = []
        nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])
        output_sess = copy.deepcopy(output)
        output_sess[np.where(output_sess == -1), :] = 1
        inputs_sess, inputs_trans_sess, datas_sess, train_masks_sess = data_segmentation_session(obs_mat, trans_mat,
                                                                                                 output_sess, mask, session)
        for i in range(np.array(inputs_sess).shape[0]):
            all_sess_size.append(np.array(inputs_sess[i]).shape)
    assert counter == all_data_obs_mat.shape[0]

# plot covariates and save figures
Len = 700
path_analysis = '../../glm-hmm_package/results/' + 'covariates/'
if not os.path.exists(path_analysis):
    os.makedirs(path_analysis)

print('all_data_trans_mat.shape=', all_data_trans_mat.shape)

#========== plot inputs ===========
# for i in range (5):
#     fig = plt.figure()
#     # fig = plt.figure(figsize=(6 * 8, 10), dpi=80, facecolor='w', edgecolor='k')
#     plt.plot(all_data_obs_mat[0:Len,i])
#     # plt.yticks(fontsize=30)
#     # plt.legend(fontsize=30)
#     # plt.axhline(y=0, color="k", alpha=0.5, ls="--")
#     # plt.ylim((-3, 14))
#     plt.ylabel("Observation Weights")
#     plt.xlabel("Covariate")
#
#     if i == 0:
#         plt.title("Obs_Stimulus")
#     elif i == 4:
#         plt.title("Obs_stim_side")
#     else:
#         plt.title("Obs_Previous choice")
#
#     fig.savefig(path_analysis + 'Obs_inputs' + str(i) + '.png')


# plot trans inputs  one for stimulus, 3 for filtered pc, 3 for exp filtered stim_side, 3 for previous rewards, 10 for basis function

# for i in range (20):
#     plt.plot(all_data_trans_mat[:,i],  lw=2)
#     plt.yticks(fontsize=30)
#     plt.legend(fontsize=30)
#     plt.axhline(y=0, color="k", alpha=0.5, ls="--")
#     # plt.ylim((-3, 14))
#     plt.ylabel("Observation Weights", fontsize=30)
#     plt.xlabel("Covariate", fontsize=30, labelpad=20)


# fig = plt.figure(figsize=(6 * 8, 10), dpi=80, facecolor='w', edgecolor='k')
# fig = plt.figure()
# plt.plot(all_data_trans_mat[0:Len, 0])
# plt.title("trans_Stimulus")
# fig.savefig(path_analysis + 'trans_Stimulus'  + '.png')

# fig = plt.figure(figsize=(6 * 8, 10), dpi=80, facecolor='w', edgecolor='k')
# fig = plt.figure()
#
# for i in [1, 2, 3]:
#     # plt.subplots_adjust(left=0.1, bottom=0.24, right=0.95, top=0.7, wspace=0.8, hspace=0.5)
#     # plt.subplots_adjust(left=0.15, bottom=0.25, right=0.95, top=0.83, wspace=0.3, hspace=1)
#     plt.subplot(3, 1, i)
#     plt.plot(all_data_trans_mat[0:Len, i])
#     if i==1:
#         plt.title("trans_filtered_Previous choice")
#
# # plt.title("trans_filtered_Previous choice")
# fig.savefig(path_analysis + 'trans_filtered_Previous choice'  + '.png')
#
#
# # fig = plt.figure(figsize=(6 * 8, 10), dpi=80, facecolor='w', edgecolor='k')
# fig = plt.figure()
#
# for i in [4, 5, 6]:
#     # plt.subplots_adjust(left=0.1, bottom=0.24, right=0.95, top=0.7, wspace=0.8, hspace=0.5)
#     # plt.subplots_adjust(left=0.15, bottom=0.25, right=0.95, top=0.83, wspace=0.3, hspace=1)
#     plt.subplot(3, 1, i-3)
#     plt.plot(all_data_trans_mat[0:Len, i])
#     if i == 4:
#         plt.title("trans_filtered_stim_side")
#
# # plt.title("trans_filtered_stim_side")
# fig.savefig(path_analysis + 'trans_filtered_stim_side' + '.png')
#
# fig = plt.figure()
# for i in [7, 8, 9]:
#     # plt.subplots_adjust(left=0.1, bottom=0.24, right=0.95, top=0.7, wspace=0.8, hspace=0.5)
#     # plt.subplots_adjust(left=0.15, bottom=0.25, right=0.95, top=0.83, wspace=0.3, hspace=1)
#     plt.subplot(3, 1, i-6)
#     plt.plot(all_data_trans_mat[0:Len, i])
#     if i==7:
#         plt.title("trans_filtered_reward")
# # plt.title("trans_filtered_reward")
# fig.savefig(path_analysis + 'trans_filtered_reward' + '.png')
#
# fig = plt.figure()
# for i in range(10,20):
#     # plt.subplots_adjust(left=0.1, bottom=0.24, right=0.95, top=0.7, wspace=0.8, hspace=0.5)
#     # plt.subplots_adjust(left=0.15, bottom=0.25, right=0.95, top=0.83, wspace=0.3, hspace=1)
#     # plt.subplot(10, 1, i-9)
#     plt.plot(all_data_trans_mat[0:Len, i])
# plt.title("trans_basis")
# fig.savefig(path_analysis + 'trans_basis' + '.png')

