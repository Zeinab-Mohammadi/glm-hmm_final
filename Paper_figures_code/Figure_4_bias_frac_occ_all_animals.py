import numpy as np
import json
from io_utils import cross_validation_vector, model_data_glmhmm, get_mouse_info, mice_names_info, colors_func
from analyze_results_utils import state_occupancies_bias, get_file_name_for_best_model_fold, data_segmentation_session, mask_for_violations
import matplotlib.pyplot as plt

if __name__ == '__main__':
    K = 4  # number of states
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    cols = colors_func(K)

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/figure_4/'
    mice_names = mice_names_info(path_data + 'mice_names.npz')
    print('mice_names.shape', np.array(mice_names).shape)

    not_viols_ratio = []
    state_occupancies_R_all = []
    state_occupancies_L_all = []

    for animal in mice_names:
        path_that_animal = path_of_the_directory + animal
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
        not_viols_ratio.append((not_viols_size / y_init_size))
        obs_mat = obs_mat[:, [0, 1, 2, 3]]

        # Create mask:
        # Identify violations for exclusion:
        index_viols = np.where(y == -1)[0]
        nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])
        y[np.where(y == -1), :] = 1

        inputs, inputs_trans, datas, train_masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)
        globe = False
        state_occupancies_L, state_occupancies_R = state_occupancies_bias(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma, left_probs)
        state_occupancies_L_all.append(state_occupancies_L)
        state_occupancies_R_all.append(state_occupancies_R)

    state_occupancies_R_mean = np.mean(state_occupancies_R_all, axis=0)
    state_occupancies_L_mean = np.mean(state_occupancies_L_all, axis=0)

    # plotting
    fig = plt.figure(figsize=(2.2, 2.2))
    plt.subplots_adjust(left=0.2, top=0.85, bottom=0.2)  # , right=0.95,
    for z, occ in enumerate(state_occupancies_R_mean):
        plt.bar(z, occ, width=0.8, color=cols[z])
    # plt.ylim((0, 1))
    plt.xticks([0, 1, 2, 3], ['1', '2', '3', '4'], fontsize=9)
    plt.yticks([.1, 0.2, 0.3, 0.4, 0.5], ['10', '20', '30', '40', '50'], fontsize=9)
    plt.xlabel('state #', fontsize=9)
    plt.title('Right-Bias blocks', color='hotpink', fontsize=9)
    fig.savefig(figure_dir + 'state_occupancies_Bias_R_all_animals.pdf')

    # plotting
    fig = plt.figure(figsize=(2.2, 2.2))
    plt.subplots_adjust(left=0.2, top=0.85, bottom=0.2)  # , right=0.95, top=0.9)
    # plt.subplots_adjust(left=0.4, bottom=0.3, right=0.95, top=0.9)
    for z, occ in enumerate(state_occupancies_L_mean):
        plt.bar(z, occ, width=0.8, color=cols[z])
    # plt.ylim((0, 1))
    plt.xticks([0, 1, 2, 3], ['1', '2', '3', '4'], fontsize=9)
    plt.yticks([.1, 0.2, 0.3, 0.4, 0.5], ['10', '20', '30', '40', '50'], fontsize=9)
    plt.xlabel('state #', fontsize=9)
    plt.ylabel('Occurrence (%)', fontsize=9)  # , labelpad=0.5)
    plt.title('Left-Bias blocks', color='blue', fontsize=9)
    fig.savefig(figure_dir + 'state_occupancies_Bias_L_all_animals.pdf')



