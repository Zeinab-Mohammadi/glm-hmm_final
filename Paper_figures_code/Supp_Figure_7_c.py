import numpy as np
import matplotlib.pyplot as plt
import json
from io_utils import get_was_correct, cross_validation_vector, model_data_glmhmm, get_mouse_info
from analyze_results_utils import get_file_name_for_best_model_fold, get_global_weights, calculate_posterior_given_data, data_segmentation_session, mask_for_violations, find_corresponding_states_new
from Review_utils import mice_names_info, get_file_name_for_best_model_fold_GLM_O, get_marginal_posterior

# Parameters
num_inputs = 4
alpha_val = 2.0
prior_sigma = 4.0
K = 4
cols = ["green", "orange", "blue", "purple"]  # Assign colors for states

# Paths
path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'
figure_dir = '../../glm-hmm_package/results/figures_for_paper/fig_reviews/Rev1_states_vs_rewarded/'
global_directory = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(num_inputs) + '/'

# Get all animal names
mice_names = mice_names_info(path_data + 'mice_names.npz')

# Storage for reward rates across animals
reward_rates_all = []
all_trials_state =  np.zeros((1, K))
trials_in_state_animal = []
trials_in_state_all = []
trials_in_state_animal_no_GLM_T = []
trials_in_state_all_no_GLM_T = []

# Loop over all animals
for z, animal in enumerate(mice_names):
    print(f"Processing animal {z+1}/{len(mice_names)}: {animal}")

    path_analysis = f'../../glm-hmm_package/results/model_indiv_ibl/num_regress_obs_{num_inputs}/prior_sigma_{prior_sigma}_transition_alpha_{alpha_val}/{animal}/'
    cv_file = path_analysis + "/diff_folds_fit.npz"
    diff_folds_fit = cross_validation_vector(cv_file)

    with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
        optimal_initialize_dict = json.load(f)

    # Load the best-fit model
    params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
    Params_model, lls = model_data_glmhmm(params_and_LL)

    # Load behavioral data for this animal
    obs_mat, trans_mat, y, session, left_probs, animal_eids = get_mouse_info(path_data + animal + '_processed.npz')
    Params_model_glob = get_global_weights(global_directory, K)
    global_weights = -Params_model_glob[2]
    perm = find_corresponding_states_new(Params_model_glob)

    # Preprocess
    obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))
    not_viols = np.where(y != -1)
    y = y[not_viols[0], :]
    obs_mat = obs_mat[not_viols[0], :]
    session = session[not_viols[0]]
    left_probs = left_probs[not_viols[0]]

    obs_mat = obs_mat[:, [0, 1, 2, 3]]  # only keep needed covariates

    # Mask violations
    index_viols = np.where(y == -1)[0]
    _, mask = mask_for_violations(index_viols, obs_mat.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, inputs_trans, datas, train_masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)

    # Compute posterior state probabilities
    globe = False
    posterior_probs, _ = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma, perm)
    all_sessions = np.unique(session)

    # Assign each trial to the most likely state
    states_max_posterior = np.argmax(posterior_probs, axis=1)

    # Making the rewarded array for all sessions of each animal
    rewarded_array = []
    for i, sess in enumerate(all_sessions):
        idx_session = np.where(session == sess)
        needed_obs_mat, this_y = obs_mat[idx_session[0], :], y[idx_session[0], :]
        was_correct, idx_easy = get_was_correct(needed_obs_mat, this_y)
        rewarded_array.append(was_correct)

    rewarded_array = np.concatenate(rewarded_array)

    # Calculate reward rate per state
    reward_rates = []
    for state in range(K):
        trials_in_state = np.where(states_max_posterior == state)[0]
        was_correct_state = rewarded_array[trials_in_state]
        locs_correct = np.where(was_correct_state == 1)[0]
        rewards_in_state = len(locs_correct)  # Assuming y=1 for rewarded trials
        reward_rate = rewards_in_state
        reward_rates.append(reward_rate)
    reward_rates_all.append(np.array(reward_rates)/len(states_max_posterior))

    # Plot for individual animal
    plt.figure(figsize=(6, 4))
    # Define engaged and disengaged reward rates
    engaged_reward = (reward_rates[0] + reward_rates[1]) / 2  # Average of Engaged Left & Right
    disengaged_reward = (reward_rates[2] + reward_rates[3]) / 2  # Average of Bias Left & Right
    x_labels = ["Engaged", "Disengaged"]
    x_positions = [0, 1]
    plt.bar(x_positions, [engaged_reward, disengaged_reward], tick_label=x_labels, color=[cols[0], cols[1]])
    plt.ylabel("Average Reward")
    plt.title(f"Reward Rate - {animal}")
    plt.savefig(f"{figure_dir}reward_rate_{str(z)}.png")
    plt.close()
print(1)
#------------------------------- Plotting No-GLM-T

reward_rates_all_no_GLM_T = []

for z, animal in enumerate(mice_names):
    print(f"Processing animal {z+1}/{len(mice_names)}: {animal}")

    path_analysis = f'../../glm-hmm_package/glm-hmm_all_no_GLM-T_to_compare/results/model_indiv_ibl/num_regress_obs_{num_inputs}/prior_sigma_{prior_sigma}_transition_alpha_{alpha_val}/{animal}/'
    cv_file = path_analysis + "/diff_folds_fit.npz"
    diff_folds_fit = cross_validation_vector(cv_file)

    with open(path_analysis + "/best_init_cvbt_dict.json", 'r') as f:
        optimal_initialize_dict = json.load(f)

    # Load the best-fit model
    params_and_LL = get_file_name_for_best_model_fold_GLM_O(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
    Params_model, lls = model_data_glmhmm(params_and_LL)

    # Load behavioral data for this animal
    obs_mat, trans_mat, y, session, left_probs, animal_eids = get_mouse_info(path_data + animal + '_processed.npz')
    perm = range(K)

    # Preprocess
    obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))
    not_viols = np.where(y != -1)
    y = y[not_viols[0], :]
    obs_mat = obs_mat[not_viols[0], :]
    session = session[not_viols[0]]
    left_probs = left_probs[not_viols[0]]

    obs_mat = obs_mat[:, [0, 1, 2, 3]]  # only keep needed covariates

    # Mask violations
    index_viols = np.where(y == -1)[0]
    _, mask = mask_for_violations(index_viols, obs_mat.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, inputs_trans, datas, train_masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)

    # Compute posterior state probabilities
    globe = False
    posterior_probs = get_marginal_posterior(globe, inputs, datas, Params_model, K, perm, alpha_val, prior_sigma)
    all_sessions = np.unique(session)

    # Assign each trial to the most likely state
    states_max_posterior = np.argmax(posterior_probs, axis=1)

    # Making the rewarded array for all sessions of each animal
    rewarded_array = []
    for i, sess in enumerate(all_sessions):
        idx_session = np.where(session == sess)
        needed_obs_mat, this_y = obs_mat[idx_session[0], :], y[idx_session[0], :]
        was_correct, idx_easy = get_was_correct(needed_obs_mat, this_y)
        rewarded_array.append(was_correct)

    rewarded_array = np.concatenate(rewarded_array)
    # Calculate reward rate per state
    reward_rates = []
    for state in range(K):
        trials_in_state = np.where(states_max_posterior == state)[0]
        was_correct_state = rewarded_array[trials_in_state]
        locs_correct = np.where(was_correct_state == 1)[0]
        rewards_in_state = len(locs_correct)  # Assuming y=1 for rewarded trials
        # all_trials_state_no_GLM_T[0, state] = all_trials_state_no_GLM_T[0, state] + len(trials_in_state)
        reward_rate = rewards_in_state
        reward_rates.append(reward_rate)

    reward_rates_all_no_GLM_T.append(np.array(reward_rates) / len(states_max_posterior))

    engaged_reward = (reward_rates[0] + reward_rates[1]) / 2  # Average of Engaged Left & Right
    disengaged_reward = (reward_rates[2] + reward_rates[3]) / 2  # Average of Bias Left & Right
    x_labels = ["Engaged", "Disengaged"]
    x_positions = [0, 1]


#--------------------- plot both all animals together for 4 columns -----------------------#
reward_rates_all_no_GLM_T = np.array(reward_rates_all_no_GLM_T)  # (num_animals, 4)
reward_rates_all = np.array(reward_rates_all)  # (num_animals, 4)

cols = ["green", "green", "orange", "orange"]  # Colors for states
bar_width = 0.4  # Make bars thinner
x_positions = np.array([0, .5, 1, 1.5])  # Base positions
fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.5), sharey=True)

# Plot "Without GLM-T"
axes[0].bar(x_positions, np.nanmean(reward_rates_all_no_GLM_T, axis=0), color=cols, width=bar_width)
axes[0].set_xticks(x_positions)
axes[0].set_xticklabels(["Eng L", "Eng R", "Bias L", "Bias R"], fontsize=9)
axes[0].set_ylabel("Reward rate", fontsize=9)
axes[0].set_title("Model without GLM-T", fontsize=9)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Plot "With GLM-T"
axes[1].bar(x_positions, np.nanmean(reward_rates_all, axis=0), color=cols, width=bar_width)
axes[1].set_xticks(x_positions)
axes[1].set_xticklabels(["Eng L", "Eng R", "Bias L", "Bias R"], fontsize=9)
axes[1].set_title("Model with GLM-T", fontsize=9)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
plt.yticks(np.arange(0, 0.35, 0.05), fontsize=9)
plt.tight_layout()
plt.show()
plt.savefig(figure_dir+"reward_rate_comparison.pdf")
