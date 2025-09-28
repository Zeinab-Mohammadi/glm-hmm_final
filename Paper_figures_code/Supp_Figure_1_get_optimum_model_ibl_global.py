import numpy as np
import json
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io_utils import model_data_glmhmm, colors_func, cross_validation_vector, make_cross_valid_for_figure
from analyze_results_utils import get_file_name_for_best_model_fold, permute_transition_matrix, find_corresponding_states_new

if __name__ == '__main__':
    num_inputs = 4
    K = 5
    cols = colors_func(K)

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/'
    path_analysis = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/Supp_figures/figure_1/'

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

    # standardize the plotted GLM transition weights
    trans_weight_append_zero = np.vstack((Params_model[1][1], np.zeros((1, Params_model[1][1].shape[1]))))
    trans_weight_append_zero_standard = copy.deepcopy(trans_weight_append_zero)
    v1 = - np.mean(trans_weight_append_zero, axis=0)  # this is v1 instead of w1=0
    trans_weight_append_zero_standard[-1, :] = v1

    for i in range(K - 1):
        trans_weight_append_zero_standard[i, :] = v1 + trans_weight_append_zero[i, :]  # Here we have vi = v1 + wi
    weight_vectors_trans = trans_weight_append_zero_standard[perm]

    params_for_individual_initialization = [[Params_model[0][0]], [Params_model[1][0], Params_model[1][1]], Params_model[2]]
    if not os.path.exists(path_analysis + 'perms/perm_K_' + str(K)):
        os.makedirs(path_analysis + 'perms/perm_K_' + str(K))
    np.savez(path_analysis + 'perms/perm_K_' + str(K) + '.npz', perm)

    # plot parameters
    fig = plt.figure(figsize=(3.3, 3))
    plt.subplots_adjust(left=0.3, bottom=0.27) #, right=0.8, top=0.9)
    M = weight_vectors.shape[2] - 1
    figure_covariates_names = ['$\Delta$ contrast', 'past choice', 'stimulus side', 'bias']
    figure_covariates_names_trans = ['filtered choice', 'filtered stim. side', 'filtered reward', 'basis_1', 'basis_2', 'basis_3']

    for k in range(K):
        if k == 0:
            label = 'Engaged-L'  # "Engaged " + str(k + 1)
        if k == 1:
            label = 'Engaged-R'  # "State " + str(k + 1)
        if k == 2:
            label = 'Biased-L'  # "State " + str(k + 1)
        if k == 3:
            label = 'Biased-R'  # "State " + str(k + 1)
        if k == 4:
            label = 'Past-choice'  # "State " + str(k + 1)
        plt.plot(range(M + 1), -weight_vectors[k][0], marker='o', markersize=5, label=label, color=cols[k], lw=1, alpha=0.9)

    plt.xticks(list(range(0, len(figure_covariates_names))), figure_covariates_names, rotation=60, fontsize=8)
    plt.yticks([-2.5, 0, 2.5, 5, 7.5, 10], fontsize=10)
    plt.ylabel("Observation GLM Weights", fontsize=10)
    plt.legend(fontsize=8, loc='upper right')
    plt.axhline(y=0, color="k", alpha=0.5, ls="--", lw=0.5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    fig.savefig(figure_dir + 'Obs_weights_K=' + str(K) + '.pdf')

    # Comparison plot
    cols_compare = ["#7e1e9c", "#0343df"]
    fig = plt.figure(figsize=(3.3, 3))
    plt.subplots_adjust(left=0.3, bottom=0.27)  # , right=0.8, top=0.9)
    cv_file = path_analysis + "/diff_folds_fit.npz"
    idx = np.array([0, 3, 4, 5, 6])
    data_for_plotting_df, loc_best, best_val, glm_lapse_model = make_cross_valid_for_figure(
        cv_file, idx)
    cv_file_train = path_analysis + "/train_folds_fit.npz"
    train_data_for_plotting_df, train_loc_best, train_best_val, train_glm_lapse_model = make_cross_valid_for_figure(
        cv_file_train, idx)

    # plot the data
    g = sns.lineplot(data=data_for_plotting_df, x="model", y="cv_bit_trial", err_style="bars",
                     err_kws={"linewidth": 0.75}, mew=0, color=cols_compare[0], errorbar=('ci', 68), label="With GLM-T", alpha=1,
                     lw=0.75)

    # Model without GLM-T
    path_analysis_NO_GLM_T = '../../glm-hmm_package/glm-hmm_all_no_GLM-T_to_compare/results/model_global_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/'
    cv_file_NO_GLM_T = path_analysis_NO_GLM_T + "/diff_folds_fit.npz"
    data_for_plotting_df_NO_GLM_T, loc_best_NO_GLM_T, best_val_NO_GLM_T, glm_lapse_model_NO_GLM_T = make_cross_valid_for_figure( cv_file_NO_GLM_T, idx)
    sns.lineplot(data=data_for_plotting_df_NO_GLM_T, x="model", y="cv_bit_trial",
                 err_style="bars", err_kws={"linewidth": 0.75}, mew=0, color=cols[2], errorbar=('ci', 68), label="Without GLM-T", alpha=1, lw=0.75)

    plt.xlabel("# states", fontsize=10)
    plt.ylabel("Test LL", fontsize=10)
    plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'], rotation=45, fontsize=10)
    plt.yticks([.35, .4, .45, .5], fontsize=10)
    plt.legend(loc='center right', fontsize=7)

    plt.yticks(fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    fig.savefig(figure_dir + 'LL_Model Comparison' + '.pdf')







