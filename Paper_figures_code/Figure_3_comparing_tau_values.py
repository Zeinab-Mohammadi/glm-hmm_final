import numpy as np
from io_utils import make_cross_valid_for_figure
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    for num_inputs in [4]:
        Figure_path = '../../glm-hmm_package/results/figures_for_paper/figure_3/'
        path_analysis = '../../glm-hmm_package/results/model_global_ibl'
        path_analysis_Tau_2 = path_analysis + '_tau=2' + '/' + 'num_regress_obs_' + str(num_inputs) + '/'
        path_analysis_Tau_4 = path_analysis + '/' + 'num_regress_obs_' + str(
            num_inputs) + '/'  # this is for tau=4 which is the best fit and have chosen as the final model
        path_analysis_Tau_8 = path_analysis + '_tau=8' + '/' + 'num_regress_obs_' + str(num_inputs) + '/'
        path_analysis_Tau_16 = path_analysis + '_tau=16' + '/' + 'num_regress_obs_' + str(num_inputs) + '/'

        for K in range(2, 6):
            cols = ["#7e1e9c", "#0343df", "#15b01a", "#bf77f6", "#95d0fc", "#96f97b"]

        # plot the comparison of LL_test for different Taus
        idx = np.array([0, 3, 4, 5, 6])
        cv_file_Tau_2 = path_analysis_Tau_2 + "/diff_folds_fit.npz"
        data_for_plotting_df_Tau_2, loc_best_2, best_val_2, glm_lapse_model_2 = make_cross_valid_for_figure(
            cv_file_Tau_2, idx)
        cv_file_Tau_4 = path_analysis_Tau_4 + "/diff_folds_fit.npz"
        data_for_plotting_df_Tau_4, loc_best_4, best_val_4, glm_lapse_model_4 = make_cross_valid_for_figure(
            cv_file_Tau_4, idx)
        cv_file_Tau_8 = path_analysis_Tau_8 + "/diff_folds_fit.npz"
        data_for_plotting_df_Tau_8, loc_best_8, best_val_8, glm_lapse_model_8 = make_cross_valid_for_figure(
            cv_file_Tau_8, idx)
        cv_file_Tau_16 = path_analysis_Tau_16 + "/diff_folds_fit.npz"
        data_for_plotting_df_Tau_16, loc_best_16, best_val_16, glm_lapse_model_16 = make_cross_valid_for_figure(
            cv_file_Tau_16, idx)
        fig = plt.figure(figsize=(2.7, 2.7), dpi=80, facecolor='w', edgecolor='k')
        # This is when Tau = 2
        g = sns.lineplot(data=data_for_plotting_df_Tau_2, x="model", y="cv_bit_trial", err_style="bars",
                         mew=0, color=cols[0], marker='o', label="\u03C4 = 2", alpha=1, lw=1)
        # This is when Tau =4
        sns.lineplot(data=data_for_plotting_df_Tau_4, x="model", y="cv_bit_trial", err_style="bars",
                     mew=0, color=cols[1], marker='o', label="\u03C4 = 4", alpha=1, lw=1)
        # This is when Tau =8
        sns.lineplot(data=data_for_plotting_df_Tau_8, x="model", y="cv_bit_trial", err_style="bars",
                     mew=0, color=cols[2], marker='o', label="\u03C4 = 8", alpha=1, lw=1)
        # This is when Tau =16
        sns.lineplot(data=data_for_plotting_df_Tau_16, x="model", y="cv_bit_trial", err_style="bars",
                     mew=0, color=cols[3], marker='o', label="\u03C4 = 16", alpha=1, lw=1)

        plt.xlabel("state #", fontsize=10)
        plt.ylabel("Normalized test LL", fontsize=10)
        plt.xticks([0, 1, 2, 3, 4],
                   ['1', '2', '3', '4', '5'], rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='lower right', fontsize=8)
        plt.tick_params(axis='y')
        plt.yticks([0.35, 0.40, 0.45, 0.50], ['0.35', '0.40', '0.45', '0.50'], fontsize=10)
        plt.title("Model comparison", fontsize=10)
        fig.tight_layout()
        fig.savefig(Figure_path + 'LL comparison plot for different Taus' + '.pdf')

        # =================== Plotting Tau comparison ===================
        fig = plt.figure(figsize=(2.7, 2.7), dpi=80, facecolor='w', edgecolor='k')

        cols_states = ["#808000", "#FFA500", "#008000", "#A52A2A", "#96f97b"]
        i = 0
        for K in range(1, 6):
            a = (K - 1) * 5
            data_y_plot_each_state = [np.mean(data_for_plotting_df_Tau_2['cv_bit_trial'][a:a + 5]),
                                      np.mean(data_for_plotting_df_Tau_4['cv_bit_trial'][a:a + 5]),
                                      np.mean(data_for_plotting_df_Tau_8['cv_bit_trial'][a:a + 5]),
                                      np.mean(data_for_plotting_df_Tau_16['cv_bit_trial'][a:a + 5])]
            data_x_plot_each_state = [2, 4, 8, 16]
            label = str(K) + "-state"
            plt.plot(data_x_plot_each_state, data_y_plot_each_state, mew=0, color=cols_states[i], marker='o', alpha=1,
                     lw=2, label=label)
            i = i + 1
        plt.xlabel("\u03C4 values", fontsize=11)
        plt.ylabel("Normalized test LL", fontsize=10)
        plt.xticks([2, 4, 8, 16], ['2', '4', '8', '16'], rotation=0, fontsize=10)
        plt.yticks([0.35, 0.40, 0.45, 0.50], ['0.35', '0.40', '0.45', '0.50'], fontsize=10)
        plt.legend(loc='best', fontsize=8)
        plt.tick_params(axis='y')
        plt.yticks(fontsize=10)
        # plt.ylim((0.3, 0.52))
        plt.title("\u03C4 comparison", fontsize=10)
        fig.tight_layout()
        fig.savefig(Figure_path + 'LL comparison plot for different states' + '.pdf')



