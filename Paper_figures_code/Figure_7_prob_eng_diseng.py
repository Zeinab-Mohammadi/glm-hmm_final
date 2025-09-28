import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from io_utils import (
    cross_validation_vector, model_data_glmhmm, get_mouse_info, colors_func
)
from analyze_results_utils import (
    get_file_name_for_best_model_fold, calculate_posterior_given_data,
    data_segmentation_session, mask_for_violations, get_global_weights,
    find_corresponding_states_new
)

def mice_names_info(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    mice_names = data[0]
    return mice_names

if __name__ == '__main__':
    # ----- config -----
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    K = 4
    num_points = 100  # interpolation points

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    path_of_the_directory = '../../glm-hmm_package/data/ibl/tables_new/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/fig_reviews/Rev1_avg_session_plot_merge_states_all_animals/'
    global_directory = '../../glm-hmm_package/results/model_global_ibl/' + f'num_regress_obs_{num_inputs}/'

    os.makedirs(figure_dir, exist_ok=True)

    mice_names = mice_names_info(path_data + 'mice_names.npz')

    # will hold per-animal curves (each length = num_points)
    all_animals_ave1 = []  # avg of states 0–1 (engaged)
    all_animals_ave2 = []  # avg of states 2–3 (disengaged)

    for z, animal in enumerate(mice_names):
        print('z=', z, 'animal=', animal)

        path_analysis = (
            '../../glm-hmm_package/results/model_indiv_ibl/'
            f'num_regress_obs_{num_inputs}/'
            f'prior_sigma_{prior_sigma}_transition_alpha_{alpha_val}/'
            f'{animal}/'
        )

        cv_file = path_analysis + "diff_folds_fit.npz"
        diff_folds_fit = cross_validation_vector(cv_file)

        with open(path_analysis + "optimal_initialize_dict.json", 'r') as f:
            optimal_initialize_dict = json.load(f)

        params_and_LL = get_file_name_for_best_model_fold(
            diff_folds_fit, K, path_analysis, optimal_initialize_dict
        )
        Params_model, _ = model_data_glmhmm(params_and_LL)

        # individual model params
        weight_vectors = Params_model[2]
        log_transition_matrix = Params_model[1][0]
        init_state_dist = Params_model[0][0]

        # data for this animal
        obs_mat, trans_mat, y, session, left_probs, animal_eids = get_mouse_info(
            path_data + animal + '_processed.npz'
        )
        # add bias term and keep the first 4 regressors only
        obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))
        obs_mat = obs_mat[:, [0, 1, 2, 3]]
        all_sessions = np.unique(session)

        # drop violations
        not_viols = np.where(y != -1)
        y = y[not_viols[0], :]
        obs_mat = obs_mat[not_viols[0], :]
        session = session[not_viols[0]]
        left_probs = left_probs[not_viols[0]]

        # build masks / segment by session
        index_viols = np.where(y == -1)[0]
        nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])
        y[np.where(y == -1), :] = 1
        inputs, inputs_trans, datas, train_masks = data_segmentation_session(
            obs_mat, trans_mat, y, mask, session
        )

        # align to global model (state permutation)
        Params_model_glob = get_global_weights(global_directory, K)
        global_weights = -Params_model_glob[2]
        label = find_corresponding_states_new(Params_model_glob)
        perm = label
        globe = False

        posterior_probs, _ = calculate_posterior_given_data(
            globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma, perm
        )

        # ----- interpolate within each session to num_points, then average over sessions -----
        session_curves = []
        for sess in all_sessions:
            idx = np.where(session == sess)[0]
            if idx.size == 0:
                continue

            pp_sess = posterior_probs[idx, :]
            n = pp_sess.shape[0]
            x_old = np.linspace(0, 1, n)
            x_new = np.linspace(0, 1, num_points)
            interp_states = np.vstack([
                interp1d(x_old, pp_sess[:, k], kind='linear', bounds_error=False, fill_value="extrapolate")(x_new)
                for k in range(pp_sess.shape[1])
            ]).T
            session_curves.append(interp_states)

        if len(session_curves) == 0:
            continue

        session_curves = np.array(session_curves)          # (n_sessions, num_points, K)
        avg_over_sessions = session_curves.mean(axis=0)    # (num_points, K)

        # merge states: first 2 = engaged, last 2 = disengaged
        avg_posterior_first = avg_over_sessions[:, :2].mean(axis=1)
        avg_posterior_sec   = avg_over_sessions[:, 2:].mean(axis=1)

        all_animals_ave1.append(avg_posterior_first)
        all_animals_ave2.append(avg_posterior_sec)

    # ----- across-animal averages -----
    if len(all_animals_ave1) == 0:
        raise RuntimeError("No animals produced valid curves; nothing to plot.")

    all_animals_ave1 = np.array(all_animals_ave1)  # (n_animals, num_points)
    all_animals_ave2 = np.array(all_animals_ave2)

    avg_first_two_states = all_animals_ave1.mean(axis=0)
    avg_last_two_states = all_animals_ave2.mean(axis=0)

    # SEM across animals
    n_animals = all_animals_ave1.shape[0]
    sem_first = all_animals_ave1.std(axis=0, ddof=1) / np.sqrt(n_animals)
    sem_last = all_animals_ave2.std(axis=0, ddof=1) / np.sqrt(n_animals)

    # ----- single final plot (across all mice) -----
    fig = plt.figure(figsize=(4.8, 3.2))
    ax = plt.gca()

    ax.plot(2 * avg_first_two_states, label="Engaged states", lw=2, color="Green")
    ax.plot(2 * avg_last_two_states, label="Disengaged states", lw=2, color="orange")

    ax.set_xticks(range(0, num_points + 1, 10))
    ax.set_xticklabels([str(i) for i in range(0, num_points + 1, 10)], fontsize=9)
    ax.set_yticks([0.2, 0.3, 0.4, .5, .6, .7, .8])
    ax.set_yticklabels(["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"], fontsize=9)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title("Average session across all mice", fontsize=9)
    ax.set_xlabel("Trial percentage", fontsize=9)
    ax.set_ylabel("p(state)", fontsize=9)
    ax.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, 'all_interpol_merge_average_session_interpolated.pdf'))

