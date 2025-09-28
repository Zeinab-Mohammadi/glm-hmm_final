import numpy as np
import os
import matplotlib.pyplot as plt
from io_utils import colors_func, make_cross_valid_for_figure

if __name__ == '__main__':
    num_inputs = 4
    K = 4
    cols = colors_func(K)

    # to compare models for With GLM-T
    path_analysis = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/synthetic_data_with_GLM-T/'
    path_analysis_NO_GLM_T = '../../glm-hmm_package/glm-hmm_all_no_GLM-T_to_compare/results/model_global_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/' + '/GLM-O_fit_synthetic_data_with_GLM-T/'
    title = "Simulated data from model with GLM-T"

    cv_file = path_analysis + "/diff_folds_fit.npz"

    # Comparison plot
    cols_compare = ["#7e1e9c", "#0343df"]
    fig, ax = plt.subplots(figsize=(3.3, 3.2))
    plt.subplots_adjust(left=0.2, bottom=0.27)  # , right=0.8, top=0.9)
    cv_file = path_analysis + "/diff_folds_fit.npz"
    idx = np.array([0, 3, 4, 5, 6])
    data_for_plotting_df, loc_best, best_val, glm_lapse_model = make_cross_valid_for_figure(
        cv_file, idx)
    cv_file_train = path_analysis + "/train_folds_fit.npz"
    train_data_for_plotting_df, train_loc_best, train_best_val, train_glm_lapse_model = make_cross_valid_for_figure(
        cv_file_train, idx)

    # Model without GLM-T
    cv_file_NO_GLM_T = path_analysis_NO_GLM_T + "/diff_folds_fit.npz"
    data_for_plotting_df_NO_GLM_T, loc_best_NO_GLM_T, best_val_NO_GLM_T, glm_lapse_model_NO_GLM_T = make_cross_valid_for_figure(cv_file_NO_GLM_T, idx)

    # plot both
    # Filter data to only include model = 4
    # Filter data to only include model = 4
    data_model_4 = data_for_plotting_df[data_for_plotting_df["model"] == 4]
    data_model_4_NO_GLM_T = data_for_plotting_df_NO_GLM_T[data_for_plotting_df_NO_GLM_T["model"] == 4]

    # Extract means and standard errors
    mean_with_GLM_T = data_model_4["cv_bit_trial"].mean()
    std_with_GLM_T = data_model_4["cv_bit_trial"].std()

    mean_without_GLM_T = data_model_4_NO_GLM_T["cv_bit_trial"].mean()
    std_without_GLM_T = data_model_4_NO_GLM_T["cv_bit_trial"].std()

    # Compute percentage increase
    percentage_increase = (mean_with_GLM_T - mean_without_GLM_T)

    # X-axis positions (No GLM-T, With GLM-T)
    x_vals = [4.9, 5.1]


    # Plot points with error bars
    plt.errorbar(x_vals[0], mean_without_GLM_T, yerr=std_without_GLM_T, fmt='o', color=cols[2],
                 capsize=5, markersize=7)
    plt.errorbar(x_vals[1], mean_with_GLM_T, yerr=std_with_GLM_T, fmt='o', color=cols_compare[0],
                 capsize=5, markersize=7)

    # Connect points with dashed line
    plt.plot(x_vals, [mean_without_GLM_T, mean_with_GLM_T], linestyle="dashed", color="gray", alpha=0.7)

    # Add text beside the points
    plt.text(4, 0.486,
             f"{'Improvement'} = {abs(percentage_increase):.3f} bits/trial",
             fontsize=8, color="black", ha="left", va="center")

    # plot to compare models for NO GLM-T
    path_analysis = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/synthetic_data/'
    path_analysis_NO_GLM_T = '../../glm-hmm_package/glm-hmm_all_no_GLM-T_to_compare/results/model_global_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/' + '/GLM-O_fit_synthetic_data/'

    figure_dir = '../../glm-hmm_package/results/figures_for_paper//fig_reviews/Rev1_compare_LL_models_for_synthetic_data_only_4-state/'
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    cv_file = path_analysis + "/diff_folds_fit.npz"

    # Comparison plot
    cols_compare = ["#7e1e9c", "#0343df"]
    cv_file = path_analysis + "/diff_folds_fit.npz"
    idx = np.array([0, 3, 4, 5, 6])
    data_for_plotting_df, loc_best, best_val, glm_lapse_model = make_cross_valid_for_figure(
        cv_file, idx)
    cv_file_train = path_analysis + "/train_folds_fit.npz"
    train_data_for_plotting_df, train_loc_best, train_best_val, train_glm_lapse_model = make_cross_valid_for_figure(
        cv_file_train, idx)

    # Model without GLM-T
    cv_file_NO_GLM_T = path_analysis_NO_GLM_T + "/diff_folds_fit.npz"
    data_for_plotting_df_NO_GLM_T, loc_best_NO_GLM_T, best_val_NO_GLM_T, glm_lapse_model_NO_GLM_T = make_cross_valid_for_figure(
        cv_file_NO_GLM_T, idx)

    # plot both
    # Filter data to only include model = 4
    data_model_4 = data_for_plotting_df[data_for_plotting_df["model"] == 4]
    data_model_4_NO_GLM_T = data_for_plotting_df_NO_GLM_T[data_for_plotting_df_NO_GLM_T["model"] == 4]

    # Extract means and standard errors
    mean_with_GLM_T = data_model_4["cv_bit_trial"].mean()
    std_with_GLM_T = data_model_4["cv_bit_trial"].std()

    mean_without_GLM_T = data_model_4_NO_GLM_T["cv_bit_trial"].mean()
    std_without_GLM_T = data_model_4_NO_GLM_T["cv_bit_trial"].std()

    # Compute percentage increase
    percentage_increase = -(mean_with_GLM_T - mean_without_GLM_T)

    # X-axis positions (No GLM-T, With GLM-T)
    x_vals = [4, 4.2]

    # Plot points with error bars
    plt.errorbar(x_vals[0], mean_without_GLM_T, yerr=std_without_GLM_T, fmt='o', color=cols[2],
                 capsize=5, markersize=7, label="Without GLM-T")
    plt.errorbar(x_vals[1], mean_with_GLM_T, yerr=std_with_GLM_T, fmt='o', color=cols_compare[0],
                 capsize=5, markersize=7, label="With GLM-T")

    # Connect points with dashed line
    plt.plot(x_vals, [mean_without_GLM_T, mean_with_GLM_T], linestyle="dashed", color="gray", alpha=0.7)
    plt.xticks([5, 4.1], ['Data with GLM-T', 'Data without GLM-T'])
    plt.ylabel("Test LL (bits/trial)", fontsize=9)

    plt.legend(fontsize=9, loc='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    fig.savefig(figure_dir + 'LL_Model_Comparison.pdf')


