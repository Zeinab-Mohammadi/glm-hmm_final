# plot differences in 90th percentile response times for engaged and
# disengaged states for IBL animals

import json
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')

from Review_utils import load_glmhmm_data, load_cv_arr, \
    mice_names_info, load_rts, get_file_name_for_best_model_fold, \
    data_segmentation_session, mask_for_violations, get_mouse_info_all, \
    read_bootstrapped_median, perform_bootstrap_individual_animal, get_marginal_posterior, \
    get_file_name_for_best_model_fold_no_GLM_T

from analyze_results_utils import calculate_posterior_given_data, find_corresponding_states_new, get_global_weights

if __name__ == '__main__':
    K = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    Thr = 0.55
    data_to_plot = []

    data_dir = '/Users/zm6112/Dropbox/Python_code/glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    response_time_dir = '/Users/zm6112/Dropbox/Python_code/glm-hmm_package/data/ibl/response_times/separate_mouse_data/'
    overall_dir = '/Users/zm6112/Dropbox/Python_code/glm-hmm_package/results/model_indiv_ibl/num_regress_obs_4/prior_sigma_4.0_transition_alpha_2.0/'
    figure_dir = '/Users/zm6112/Dropbox/Python_code/glm-hmm_package/results/figures_for_paper/fig_reviews/Rev1_reaction_time_3_plot_response_times_90th_percentile_both/'
    animal_list = mice_names_info(data_dir + 'mice_names.npz')

    fig, ax = plt.subplots(figsize=(4.5, 3))
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.9, top=0.9)

    for z, animal in enumerate(animal_list):
        results_dir = overall_dir + animal
        cv_file = results_dir + "/diff_folds_fit.npz"
        cvbt_folds_model = load_cv_arr(cv_file)

        with open(results_dir + "/optimal_initialize_dict.json", 'r') as f:
            best_init_cvbt_dict = json.load(f)

        raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                                     results_dir,
                                                     best_init_cvbt_dict)
        hmm_params, lls = load_glmhmm_data(raw_file)
        obs_mat, trans_mat, y, session, left_probs, animal_eid_dict = get_mouse_info_all(data_dir + animal + '_processed.npz')
        all_sessions = np.unique(session)
        violation_idx = np.where(y == -1)[0]
        # Create mask:
        # Identify violations for exclusion:
        index_viols = np.where(y == -1)[0]
        nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])

        y[np.where(y == -1), :] = 1
        inputs, inputs_trans, datas, train_masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)

        global_directory = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_4' + '/'

        animal_list = mice_names_info(data_dir + 'mice_names.npz')
        Params_model_glob = get_global_weights(global_directory, K)
        global_weights = -Params_model_glob[2]
        label = find_corresponding_states_new(Params_model_glob)
        permutation = range(K)  # ??????
        globe = False
        posterior_probs, P = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, hmm_params, K,
                                                            alpha_val, prior_sigma, permutation)
        # Read in RTs
        rts, rts_sess = load_rts(response_time_dir + animal + '.npz')

        rts_engaged = rts[np.where((posterior_probs[:, label[0]] >= Thr) | (posterior_probs[:, label[1]] >= Thr))[0]]
        rts_engaged = rts_engaged[np.where(~np.isnan(rts_engaged))]

        rts_disengaged = rts[np.where((posterior_probs[:, label[2]] >= Thr) | (posterior_probs[:, label[3]] >= Thr))[0]]
        rts_disengaged = rts_disengaged[np.where(~np.isnan(rts_disengaged))]

        # Get 90th percentile for each
        quant = 0.90
        eng_quantile = np.quantile(rts_engaged, quant)
        dis_quantile = np.quantile(rts_disengaged, quant)
        diff_quantile = dis_quantile - eng_quantile

        # Perform bootstrap to get error bars:
        lower_eng, upper_eng, min_val_eng, max_val_eng, frac_above_true \
            = perform_bootstrap_individual_animal(rts_engaged, rts_disengaged,
                                                  diff_quantile, quant)

        # Store values in a list
        data_to_plot.append((animal, diff_quantile, lower_eng, upper_eng))

    # Sort by diff_quantile (ascending order)
    data_to_plot.sort(key=lambda x: x[1])
    data_to_plot_glmT_sorted = list(data_to_plot)

    # Plot after sorting
    for sorted_z, (animal, diff_quantile, lower_eng, upper_eng) in enumerate(data_to_plot):
        if sorted_z == 0:
            plt.scatter(sorted_z, diff_quantile, label="With GLM-T", color='purple', marker='*', s=7)  # Mean point
        else:
            plt.scatter(sorted_z, diff_quantile, color='purple', marker='*', s=7)  # Mean point
        plt.plot([sorted_z, sorted_z], [lower_eng, upper_eng], color='purple', lw=1)  # Vertical line (error bar)
        plt.hlines([lower_eng, upper_eng], sorted_z - 0.2, sorted_z + 0.2, color='purple', lw=1)  # Horizontal caps

    median, lower, upper, mean_viol_rate_dist = read_bootstrapped_median(
        overall_dir + 'median_response_bootstrap.npz')
    # plt.plot([lower, upper], [z + 1, z + 1], color='#0343df', lw=0.75)
    # plt.scatter(median, z + 1, color='b', s=1)
    plt.xlabel("Mouse #", fontsize=9)
    plt.ylabel("T₉₀,diseng. - T₉₀,engaged", fontsize=9)
    # plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15],
    #            ['0', '', '5', '', '10', '', '15'],
    #            fontsize=10)
    # plt.ayvline(y=0, linestyle='--', color='k', alpha=0.5, lw=0.75)
    plt.xticks(np.arange(0, 36, 5))
    plt.yticks(np.arange(0, 41, 5))
    plt.axhline(y=0, linestyle='--', color='k', alpha=0.5, lw=0.75)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    #---------------------
    # Compute mean across all animals
    mean_diff_quantile = np.mean([dis - eng for dis, eng in zip(rts_disengaged, rts_engaged)])
    middle_x = len(animal_list) / 2


    #------------------------------- no GLM-T ------------------------------------#
    K = 4
    alpha_val = 2.0
    prior_sigma = 4.0

    data_dir = '/Users/zm6112/Dropbox/Python_code/glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    response_time_dir = '/Users/zm6112/Dropbox/Python_code/glm-hmm_package/data/ibl/response_times/separate_mouse_data/'
    overall_dir = '/Users/zm6112/Dropbox/Python_code/Pycharm_Z_code_github/glm-hmm_all_no_GLM-T_to_compare/results/ibl_individual_fit/covar_set_2/prior_sigma_4.0_transition_alpha_2.0/'

    animal_list = mice_names_info(data_dir + 'mice_names.npz')
    data_to_plot = []
    for z, animal in enumerate(animal_list):
        print('z=', z)
        results_dir = overall_dir + animal

        cv_file = results_dir + "/cvbt_folds_model.npz"
        cvbt_folds_model = load_cv_arr(cv_file)

        with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
            best_init_cvbt_dict = json.load(f)

        raw_file = get_file_name_for_best_model_fold_no_GLM_T(cvbt_folds_model, K,
                                                              results_dir,
                                                              best_init_cvbt_dict)
        hmm_params, lls = load_glmhmm_data(raw_file)

        # Also get data for animal:
        obs_mat, trans_mat, y, session, left_probs, animal_eid_dict = get_mouse_info_all(
            data_dir + animal + '_processed.npz')
        all_sessions = np.unique(session)
        violation_idx = np.where(y == -1)[0]
        # Create mask:
        # Identify violations for exclusion:
        index_viols = np.where(y == -1)[0]
        nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])

        y[np.where(y == -1), :] = 1
        inputs, inputs_trans, datas, train_masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)

        permutation = range(K)
        label = [2, 1, 0, 3]
        globe = False
        posterior_probs = get_marginal_posterior(globe, inputs, datas, hmm_params, K, permutation, alpha_val,
                                                 prior_sigma)

        # Read in RTs
        rts, rts_sess = load_rts(response_time_dir + animal + '.npz')

        rts_engaged = rts[
            np.where((posterior_probs[:, label[0]] >= Thr) | (posterior_probs[:, label[1]] >= Thr))[0]]
        rts_engaged = rts_engaged[np.where(~np.isnan(rts_engaged))]

        rts_disengaged = rts[
            np.where((posterior_probs[:, label[2]] >= Thr) | (posterior_probs[:, label[3]] >= Thr))[0]]
        rts_disengaged = rts_disengaged[np.where(~np.isnan(rts_disengaged))]

        # Get 90th percentile for each
        quant = 0.90
        # Ensure arrays are not empty before computing quantiles
        if len(rts_engaged) > 0:
            eng_quantile = np.quantile(rts_engaged, quant)
        else:
            eng_quantile = np.nan  # Assign NaN if no engaged trials

        if len(rts_disengaged) > 0:
            dis_quantile = np.quantile(rts_disengaged, quant)
        else:
            dis_quantile = np.nan  # Assign NaN if no disengaged trials

        # Compute difference only if both values are valid
        if not np.isnan(eng_quantile) and not np.isnan(dis_quantile):
            diff_quantile = dis_quantile - eng_quantile
        else:
            diff_quantile = np.nan  # If any is NaN, assign NaN

        diff_quantile = dis_quantile - eng_quantile

        # Perform bootstrap to get error bars:
        lower_eng, upper_eng, min_val_eng, max_val_eng, frac_above_true \
            = perform_bootstrap_individual_animal(rts_engaged, rts_disengaged,
                                                  diff_quantile, quant)

        # Store values in a list
        data_to_plot.append((animal, diff_quantile, lower_eng, upper_eng))

    # Sort by diff_quantile (ascending order)
    data_to_plot.sort(key=lambda x: x[1])
    data_to_plot_noGLMT_sorted = list(data_to_plot)  # keep for Source Data export

    # Plot after sorting
    for sorted_z, (animal, diff_quantile, lower_eng, upper_eng) in enumerate(data_to_plot):
        if sorted_z == 0:
            plt.scatter(sorted_z, diff_quantile, label="Without GLM-T", color='blue', marker='o', s=7)  # Mean point
        else:
            plt.scatter(sorted_z, diff_quantile, color='blue', marker='o', s=7)  # Mean point
        plt.plot([sorted_z, sorted_z], [lower_eng, upper_eng], color='blue', lw=1)  # Vertical line (error bar)
        plt.hlines([lower_eng, upper_eng], sorted_z - 0.2, sorted_z + 0.2, color='blue', lw=1)  # Horizontal caps

    median, lower, upper, mean_viol_rate_dist = read_bootstrapped_median(
        overall_dir + 'median_response_bootstrap_no_GLM_T.npz')
    # plt.plot([lower, upper], [z + 1, z + 1], color='#0343df', lw=0.75)
    # plt.scatter(median, z + 1, color='b', s=1)
    plt.xlabel("Mouse #", fontsize=9)
    plt.ylabel("T₉₀,diseng. - T₉₀,engaged", fontsize=9)
    # plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15],
    #            ['0', '', '5', '', '10', '', '15'],
    #            fontsize=10)
    # plt.ayvline(y=0, linestyle='--', color='k', alpha=0.5, lw=0.75)
    plt.xticks(np.arange(0, 36, 5))
    plt.yticks(np.arange(-20, 39, 5))
    plt.axhline(y=0, linestyle='--', color='k', alpha=0.5, lw=0.75)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.legend(fontsize=9, scatterpoints=1, markerscale=2)  # Increase marker size in legend

    #------------------------
    # Compute mean across all animals
    mean_diff_quantile = np.mean([dis - eng for dis, eng in zip(rts_disengaged, rts_engaged)])

    # Get the middle x position
    middle_x = len(animal_list) / 2
    #------------------------
    # plt.title('No GLM-T')
    plt.show()
    fig.savefig(figure_dir + 'Reaction_time.pdf')

