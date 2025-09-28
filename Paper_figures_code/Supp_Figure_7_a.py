import os
from io_utils import model_data_glmhmm, colors_func, cross_validation_vector, make_cross_valid_for_figure
from analyze_results_utils import get_file_name_for_best_model_fold, permute_transition_matrix, find_corresponding_states_new
from Review_utils import calculate_state_permutation_all_data, load_glmhmm_data, load_cv_arr, get_file_name_for_best_model_fold_GLM_O
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
    num_inputs = 4
    K = 4
    cols = colors_func(K)

    path_analysis = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper//fig_reviews/Rev1_plotting_GLM_O_weights_with_without_GLM_T/'

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    cv_file = path_analysis + "/diff_folds_fit.npz"
    diff_folds_fit = cross_validation_vector(cv_file)

    with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
        optimal_initialize_dict = json.load(f)

    params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
    Params_model, lls = model_data_glmhmm(params_and_LL)
    perm = find_corresponding_states_new(Params_model)

    # Save parameters for initializing individual fits
    weight_vectors = Params_model[2][perm]
    log_transition_matrix = permute_transition_matrix(Params_model[1][0], perm)
    init_state_dist = Params_model[0][0][perm]

#--------------------- GLM-O weights without GLM-T
    covar_set = 2
    data_dir = '/Users/zm6112/Dropbox/Python_code/Pycharm_Z_code_github/glm-hmm_all_no_GLM-T_to_compare/data/ibl/data_for_cluster/'  # '/Users/zashwood/Documents/glm-hmm/data/ibl/data_for_cluster/'
    results_dir = '/Users/zm6112/Dropbox/Python_code/Pycharm_Z_code_github/glm-hmm_all_no_GLM-T_to_compare/results/ibl_global_fit/' + 'covar_set_' + str(
        covar_set) + '/'

    cv_file = results_dir + "/cvbt_folds_model.npz"
    cvbt_folds_model = load_cv_arr(cv_file)

    with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)

    # Get the file name corresponding to the best initialization for given K value
    raw_file = get_file_name_for_best_model_fold_GLM_O(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)
    permutation = calculate_state_permutation_all_data(hmm_params)
    weight_vectors_no_GLM_T = hmm_params[2][permutation]

    # plot parameters
    fig, axes = plt.subplots(1, K, figsize=(9.5, 2.8))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.8, wspace=0.4)  # Adjust the figure layout
    M = weight_vectors.shape[2] - 1
    figure_covariates_names = ['$\Delta$ contrast', 'past choice', 'stim. side', 'bias']
    figure_covariates_names_trans = ['filtered choice', 'filtered stim. side', 'filtered reward', 'basis_1', 'basis_2',
                                     'basis_3']

    for k in range(K):
        ax = axes[k]  # Select the subplot for this iteration
        if k == 0:
            label = 'Engaged-L'
        elif k == 1:
            label = 'Engaged-R'
        elif k == 2:
            label = 'Biased-L'
        elif k == 3:
            label = 'Biased-R'

        # Plot data
        line1, = ax.plot(range(M + 1), -weight_vectors[k][0], marker='o', markersize=5, color=cols[k], lw=1,
                         alpha=0.9, label='with GLM-T' if k == 0 else None)
        line2, = ax.plot(range(M + 1), -weight_vectors_no_GLM_T[k][0], marker='*', color=cols[k], lw=1,
                         alpha=0.9, linestyle='--', label='without GLM-T' if k == 0 else None)

        # Set title and axis labels
        ax.set_title(label, fontsize=10)
        ax.set_xticks(list(range(0, len(figure_covariates_names))))
        ax.set_xticklabels(figure_covariates_names, rotation=60, fontsize=8)

        # Add legend only for the first subplot
        if k == 0:
            ax.legend(handles=[line1, line2], fontsize=8, loc='upper right')
            ax.set_ylabel("Observation GLM Weights", fontsize=10)

        # Add grid and horizontal line
        ax.axhline(y=0, color="k", alpha=0.5, ls="--", lw=0.5)

        # Customize spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.show()
    fig.savefig(figure_dir + 'Obs_weights_K=' + str(K) + '.pdf')


