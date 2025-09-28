from analyze_results_utils import calculate_posterior_given_data
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from analyze_results_utils import get_file_name_for_best_model_fold, data_segmentation_session, mask_for_violations, \
    cross_validation_vector, model_data_glmhmm
from Review_utils import load_glmhmm_data, get_file_name_for_best_model_fold_GLM_O
from io_utils import get_mouse_info, mice_names_info, colors_func


def addBiasBlocks_modif(ax, pL):
    BIAS_COLORS = {50: 'None', 20: "#ff99a2", 80: "#4f91d4"}  # light blue: 20% right, light pink: 80% right
    i = 0
    while i < len(pL):
        start = i
        while i + 1 < len(pL) and np.linalg.norm(pL[i] - pL[i + 1]) < 0.0001:
            i += 1
        fc = BIAS_COLORS.get(int(100 * pL[start]), 'None')  # Avoid key error
        if fc != 'None':
            ax.axvspan(start, i + 1, facecolor=fc, alpha=0.2, edgecolor=None)  # Use ax instead of plt
        i += 1


if __name__ == '__main__':
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    K = 4
    cols = colors_func(K)
    length = 400

    sess_to_plot = ["dab3d729-8983-4431-9d88-0640b8ba4fdd"]

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/fig_reviews/Rev1_transition(not Fig3e)_over_all_trials/'

    animal = "churchlandlab/CSHL_014/_ibl_subjectTrials.table.61f6982a-40fb-44a6-8f2f-170235951e26.pqt"
    path_that_animal = path_of_the_directory + animal
    path_analysis = '../../glm-hmm_package/results/model_indiv_ibl/' + 'num_regress_obs_' + str(
        num_inputs) + '/' + 'prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(alpha_val) + '/' + animal + '/'

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

    not_viols_ratio = []
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
    perm = range(K)
    globe = False
    posterior_probs, Ps = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma, perm)

    T = 400
    for i, sess in enumerate(sess_to_plot):
        plt.subplot(6, 1, (2 * i + 2))
        idx_session = np.where(session == sess)
        needed_obs_mat = obs_mat[idx_session[0], :]
        Ps_sess = Ps[idx_session[0], :, :]
        posterior_probs_needed_this_session = posterior_probs[idx_session[0], :]
        this_left_probs = left_probs[idx_session[0]]
        Prob_left = this_left_probs[0:T]
        cols = colors_func(K)

        titles = ['Engaged-L', 'Engaged-R', 'Biased-L', 'Biased-R']
        fig, axes = plt.subplots(K, K, figsize=(10, 10))
        plt.subplots_adjust(hspace=0.5)
        axes[0, 0].plot([], [], color='purple', label='With GLM-T')  # Dummy plot for legend
        axes[0, 0].legend(loc='best')
        for i in range(K):
            for j in range(K):
                if i == j:
                    axes[i, j].plot(range(T), Ps_sess[0:T, i, j], color='purple')  # color=cols[i])
                    axes[i, j].set_ylim(0.8, 1)
                else:
                    axes[i, j].plot(range(T), Ps_sess[0:T, i, j], color='purple')
                axes[i, j].set_title(f"{titles[i]} → {titles[j]}")
                addBiasBlocks_modif(axes[i, j], Prob_left)
                axes[i, j].spines['top'].set_visible(False)
                axes[i, j].spines['right'].set_visible(False)
                if i == 0 and j == 0:
                    axes[i, j].set_xlabel("Trial", fontsize=9)
                    axes[i, j].set_ylabel("Transition Probability", fontsize=9)
        # plt.legend('Model with GLM-T')
    #---------------------------- Model without GLM-T
    num_inputs = 2
    alpha_val = 2.0
    not_viols_ratio = []
    trans_p_posterior = []

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    mice_names = mice_names_info(path_data + 'mice_names.npz')
    path_of_the_directory = '../../glm-glm-hmm_all_no_GLM-T_to_compare/data/ibl/tables_new/'
    trans_log_p_posterior_all_animals = np.zeros((mice_names.shape[0], K, K))
    animal = "churchlandlab/CSHL_014/_ibl_subjectTrials.table.61f6982a-40fb-44a6-8f2f-170235951e26.pqt"

    path_that_animal = path_of_the_directory + animal
    path_analysis = '/Users/zm6112/Dropbox/Python_code/Pycharm_Z_code_github/glm-hmm_all_no_GLM-T_to_compare/results/ibl_individual_fit/' + 'covar_set_' + str(
            num_inputs) + '/' + 'prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(alpha_val) + '/' + animal + '/'

    cv_file = path_analysis + "/cvbt_folds_model.npz"
    cvbt_folds_model = cross_validation_vector(cv_file)
    with open(path_analysis + "/best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)

    # Get the file name corresponding to the best initialization for given K value
    params_and_LL = get_file_name_for_best_model_fold_GLM_O(cvbt_folds_model, K, path_analysis, best_init_cvbt_dict)
    Params_model, lls = load_glmhmm_data(params_and_LL)

    # Save parameters for initializing individual fits
    log_transition_matrix = np.exp(Params_model[1][0])
    init_state_dist = Params_model[0][0]

    weight_vectors_trans = log_transition_matrix
    # Also get data for animal:
    obs_mat, trans_mat, y, session, left_probs, animal_eids = get_mouse_info(path_data + animal + '_processed.npz')
    obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))

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
    globe = True
    unique_sessions = np.unique(session, return_index=True)[1]
    T = 400

    for i, sess in enumerate(sess_to_plot):
        idx_session = np.where(session == sess)
        needed_obs_mat = obs_mat[idx_session[0], :]
        needed_trans_mat = np.ones((len(idx_session[0]), 4))
        this_left_probs = left_probs[idx_session[0]]
        Weights_multi_inputs = np.tile(weight_vectors_trans, (1, 400))
        weight_vectors_trans = weight_vectors_trans[0:T]
        titles = ['Engaged-L', 'Engaged-R', 'Biased-L', 'Biased-R']
        cols = colors_func(K)
        axes[0, 0].plot([], [], color='red', label='Without GLM-T')
        axes[0, 0].legend(loc='best')

        for i in range(K):
            for j in range(K):
                if i == j:
                    axes[i, j].plot(range(T), np.tile(weight_vectors_trans[i, j], (400)), color='red') #color=cols[i])
                    axes[i, j].set_ylim(0.5, 1)
                else:
                    axes[i, j].plot(range(T), np.tile(weight_vectors_trans[i, j], (400)), color='red')
                    axes[i, j].set_ylim(-0.1, 1)
                axes[i, j].set_title(f"{titles[i]} → {titles[j]}", fontsize=9)
                addBiasBlocks_modif(axes[i, j], Prob_left)
                axes[i, j].spines['top'].set_visible(False)
                axes[i, j].spines['right'].set_visible(False)
                axes[i, j].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

                if i == 0 and j == 0:
                    axes[i, j].set_xlabel("Trial")
                    axes[i, j].set_ylabel("Transition Probability")
        plt.show()
        fig.savefig(figure_dir + str(sess) + 'session_my.pdf')




