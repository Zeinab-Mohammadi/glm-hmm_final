import numpy as np
import matplotlib.pyplot as plt
import json
from io_utils import cross_validation_vector, get_mouse_info, mice_names_info, colors_func, get_mouse_info_simulated, model_data_glmhmm
from analyze_results_utils import state_occupancies_bias, get_file_name_for_best_model_fold, data_segmentation_session, mask_for_violations


if __name__ == '__main__':
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    K = 4
    cols = colors_func(K)

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/Supp_figures/figure_5/'
    mice_names = mice_names_info(path_data + 'mice_names.npz')


    not_viols_ratio = []
    state_occupancies_R_all = []
    state_occupancies_L_all = []

    for animal in mice_names:
        sim_data_path = '../../glm-hmm_package/results/figures_for_paper/Supp_figures/figure_5/with GLM-T/sim_data/'
        path_analysis = '../../glm-hmm_package/results/model_indiv_ibl/' + 'num_regress_obs_' + str(
            num_inputs) + '/' + 'prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(
            alpha_val) + '/' + animal + '/'

        path_that_animal = path_of_the_directory + animal

        # Get the file name corresponding to the best initialization for given K value
        with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
            optimal_initialize_dict = json.load(f)

        # Get the file name corresponding to the best initialization for given K value
        cv_file = path_analysis + "/diff_folds_fit.npz"
        diff_folds_fit = cross_validation_vector(cv_file)
        params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
        Params_model, lls = model_data_glmhmm(params_and_LL)

        _, _, _, _, left_probs, _ = get_mouse_info(path_data + animal + '_processed.npz')
        j = 0
        animal_obs_mat, animal_trans_mat, animal_datas, animal_session, animal_latents = get_mouse_info_simulated(sim_data_path + animal + '/simulation_' + str("%03d" % j) + '.npz')
        obs_mat = animal_obs_mat
        trans_mat = animal_trans_mat
        y = animal_datas
        session = animal_session
        all_sessions = np.unique(session)
        y_init_size = y.shape[0]
        not_viols = np.where(y != -1)
        not_viols_size = y[not_viols].shape[0]
        y = y[not_viols[0], :]
        obs_mat = obs_mat[not_viols[0], :]
        session = session[not_viols[0]]
        left_probs = left_probs[not_viols[0]]
        not_viols_ratio.append((not_viols_size / y_init_size))
        obs_mat = obs_mat[:, [0, 1, 2, 3]]
        # Create mask:
        # Identify violations for exclusion:
        index_viols = np.where(y == -1)[0]
        nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])
        y[np.where(y == -1), :] = 1

        inputs, inputs_trans, datas, train_masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)
        globe = False
        state_occupancies_L, state_occupancies_R = state_occupancies_bias(globe, inputs, inputs_trans, datas,
                                                                          Params_model, K, alpha_val, prior_sigma,
                                                                          left_probs)
        state_occupancies_L_all.append(state_occupancies_L)
        state_occupancies_R_all.append(state_occupancies_R)

    state_occupancies_R_mean = np.mean(state_occupancies_R_all, axis=0)
    state_occupancies_L_mean = np.mean(state_occupancies_L_all, axis=0)

    # plotting
    fig = plt.figure(figsize=(2.7, 2.7))
    # plt.subplots_adjust(left=0.3, bottom=0.1, top=0.2)
    plt.subplots_adjust(left=0.4, bottom=0.3, right=0.95, top=0.9)
    for z, occ in enumerate(state_occupancies_R_mean):
        plt.bar(z, occ, width=0.8, color=cols[z])
    # plt.ylim((0, 1))
    plt.xticks([0, 1, 2, 3], ['1', '2', '3', '4'], fontsize=10)
    plt.yticks([0, 0.25, 0.5], ['0', '0.25', '0.5'], fontsize=10)
    plt.xlabel('state', fontsize=10)
    plt.ylabel('Occurrence (%)', fontsize=10)
    plt.title('Bias_R blocks', color= 'hotpink', fontsize=10)
    fig.savefig(figure_dir + 'GLM_T-simul_state_occupancies_Bias_R_all_animals.pdf')

    # plotting
    fig = plt.figure(figsize=(2.7, 2.7))
    # plt.subplots_adjust(left=0.3, bottom=0.1, top=0.2)
    plt.subplots_adjust(left=0.4, bottom=0.3, right=0.95, top=0.9)
    for z, occ in enumerate(state_occupancies_L_mean):
        plt.bar(z, occ, width=0.8, color=cols[z])
    # plt.ylim((0, 1))
    plt.xticks([0, 1, 2, 3], ['1', '2', '3', '4'], fontsize=10)
    plt.yticks([0, 0.25, 0.5], ['0', '0.25', '0.5'], fontsize=10)
    plt.xlabel('state', fontsize=10)
    plt.ylabel('Occurrence (%)', fontsize=10)
    plt.title('Bias_L blocks', color= 'blue', fontsize=10)
    fig.savefig(figure_dir + 'GLM_T-simul_state_occupancies_Bias_L_all_animals.pdf')


