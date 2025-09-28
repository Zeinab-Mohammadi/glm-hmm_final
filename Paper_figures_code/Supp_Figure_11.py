import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
from io_utils import cross_validation_vector, model_data_glmhmm, get_mouse_info, mice_names_info
from analyze_results_utils import get_file_name_for_best_model_fold, get_global_weights, find_corresponding_states_new, \
    calculate_posterior_given_data, data_segmentation_session, mask_for_violations

def collect_transitions_by_new_state(states_max_posterior, K=4, T=5, T2=5):
    """
    For each possible new state (0 to K-1), collect the previous stable state before switching into it.

    Args:
        states_max_posterior (array-like): Sequence of inferred states (int values from 0 to K-1).
        K (int): Number of latent states.
        T (int): Length of consecutive trials in initial stable state.
        T2 (int): Length of consecutive trials in the new state after switching.

    Returns:
        dict: A dictionary where keys are target new states (0 to K-1),
              and values are lists of unique previous states that transitioned into them.
    """
    states = np.array(states_max_posterior)
    N = len(states)
    results = {k: [] for k in range(K)}

    i = 0
    while i <= N - (T + T2):
        current_state = states[i]
        # Check T-long stability
        if np.all(states[i:i+T] == current_state):
            next_start = i + T
            new_state = states[next_start]
            # Make sure new state is different and stable for T2
            if new_state != current_state and np.all(states[next_start:next_start+T2] == new_state):
                if current_state not in results[new_state]:
                    results[new_state].append(current_state)
                i = next_start + T2
                continue
        i += 1
    return results


num_inputs = 4
alpha_val = 2.0
prior_sigma = 4.0
K = 4
trial_diff_all = []
trial_diff_until_end = []

path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
global_directory = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(
    num_inputs) + '/'
figure_dir = '../../glm-hmm_package/results/figures_for_paper/fig_reviews/Rev2_previous_state_role_in_switching/'

Params_model_glob = get_global_weights(global_directory, K)
global_weights = -Params_model_glob[2]
label = find_corresponding_states_new(Params_model_glob)
mice_names = mice_names_info(path_data + 'mice_names.npz')
prev_state_counts = np.zeros(K)
all_transition_matrices = []
labels = ['Engaged-L', 'Engaged-R', 'Bias-L', 'Bias-R']

for z, animal in enumerate(mice_names):
    path_analysis = '../../glm-hmm_package/results/model_indiv_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/' + 'prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(
        alpha_val) + '/' + animal + '/'
    cv_file = path_analysis + "/diff_folds_fit.npz"
    diff_folds_fit = cross_validation_vector(cv_file)

    with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
        optimal_initialize_dict = json.load(f)

    # Get the file name corresponding to the best initialization for given K value
    params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
    Params_model, lls = model_data_glmhmm(params_and_LL)

    # Save parameters for initializing individual fits
    weight_vectors = Params_model[2]
    log_transition_matrix = Params_model[1][0]
    init_state_dist = Params_model[0][0]

    # Also get data for animal:
    obs_mat, trans_mat, y, session, left_probs, animal_eids = get_mouse_info(path_data + animal + '_processed.npz')
    obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))
    all_sessions = np.unique(session)
    y_init_size = y.shape[0]
    not_viols = np.where(y != -1)
    not_viols_size = y[not_viols].shape[0]
    y = y[not_viols[0], :]
    obs_mat = obs_mat[not_viols[0], :]
    session = session[not_viols[0]]
    left_probs = left_probs[not_viols[0]]

    left_probs_bias_trial = np.where((left_probs == .2) | (left_probs == .8))[0]
    obs_mat = obs_mat[:, [0, 1, 2, 3]]
    # Create mask:
    # Identify violations for exclusion:
    index_viols = np.where(y == -1)[0]
    nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, inputs_trans, datas, train_masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)
    perm = range(K)
    globe = False
    posterior_probs, P = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val,
                                                        prior_sigma, perm)
    states_max_posterior = np.argmax(posterior_probs, axis=1)

    results = collect_transitions_by_new_state(states_max_posterior, K=4, T=5, T2=5)
    # Count transitions from previous states to each current state
    K = 4
    transition_matrix = np.zeros((K, K))
    for new_state in range(K):
        for prev_state in results[new_state]:
            transition_matrix[new_state, prev_state] += 1

    # Normalize each row (new_state) to get fractions
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = transition_matrix / row_sums
    all_transition_matrices.append(transition_matrix)

# -------------------- Aggregate and plot average matrix across all animals --------------------
mean_matrix = np.mean(all_transition_matrices, axis=0)
row_sums = mean_matrix.sum(axis=1, keepdims=True)
normalized_mean_matrix = mean_matrix / row_sums

fig = plt.figure(figsize=(7, 5))
ax = sns.heatmap(normalized_mean_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                 xticklabels=labels,
                 yticklabels=labels)
plt.title("Average Fraction of Transitions from Each Previous State")
plt.xlabel("Previous state")
plt.ylabel("Current state")
plt.tight_layout()
fig.savefig(figure_dir + 'average_transition_heatmap_all_animals.pdf', bbox_inches='tight', dpi=300)
plt.show()


