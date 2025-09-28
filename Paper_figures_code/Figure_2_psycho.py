import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import pandas as pd
import numpy.random as npr
from scipy.stats import bernoulli
from ssm import HMM_TO
from io_utils import calculate_correct_ans, cross_validation_vector, model_data_glmhmm, get_mouse_info_all, colors_func, get_prob_right, load_reward_data
from analyze_results_utils import get_file_name_for_best_model_fold, find_corresponding_states_new, calculate_posterior_given_data, data_segmentation_session, mask_for_violations

sys.path.append('../')

if __name__ == '__main__':
    K = 4
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/'
    figure_dir = '../../glm-hmm_package/results/figures_for_paper/figure_2/'
    path_analysis = '../../glm-hmm_package/results/model_global_ibl/' + 'num_regress_obs_' + str(num_inputs)

    obs_mat, trans_mat, y, session = get_mouse_info_all(path_data + 'combined_all_mice.npz')
    rewarded = load_reward_data(path_data + 'rewarded_all_mice.npz')
    correct_answer = calculate_correct_ans(y, rewarded)

    # Create masks for violation trials
    index_viols = np.where(y == -1)[0]
    nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])
    y[np.where(y == -1), :] = 1
    obs_mat = np.hstack((obs_mat, np.ones((obs_mat.shape[0], 1))))
    obs_mat = obs_mat[:, [0, 1, 2, 3]]
    inputs, inputs_trans, datas, masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)
    cv_file = path_analysis + "/diff_folds_fit.npz"
    diff_folds_fit = cross_validation_vector(cv_file)
    with open(path_analysis + "/optimal_initialize_dict.json", 'r') as f:
        optimal_initialize_dict = json.load(f)

    # Get the file name corresponding to the best initialization for given K value
    params_and_LL = get_file_name_for_best_model_fold(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
    Params_model, lls = model_data_glmhmm(params_and_LL)
    weight_vectors = Params_model[2]

    globe = True
    posterior_probs, Ps_all = calculate_posterior_given_data(globe, inputs, inputs_trans, datas, Params_model, K, alpha_val, prior_sigma, range(K))
    states_max_posterior = np.argmax(posterior_probs, axis=1)
    cols = colors_func(K)
    perm = find_corresponding_states_new(Params_model)

    fig = plt.figure(figsize=(10, 3.2), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.13, bottom=0.3, right=0.9, top=0.89, wspace=.4, hspace=None)

    for k in range(K):
        if k == 0:
            plt.subplot(1, 3, 1)
        # if k == 1:
        #     plt.subplot(1, 4, 2)
        if k == 2:
            plt.subplot(1, 3, 2)
        # if k == 3:
        #     plt.subplot(1, 4, 4)

        # use GLM weights to get prob right
        stim_vals, prob_right_max = get_prob_right(-weight_vectors, obs_mat, perm[k], 1, 1)
        _, prob_right_min = get_prob_right(-weight_vectors, obs_mat, perm[k], -1, -1)
        plt.plot(stim_vals, prob_right_max, '-', color=cols[k], alpha=1, lw=1.4, zorder=5)

        if k == 0:
            plt.subplot(1, 3, 1)
            plt.xticks([min(stim_vals), min(stim_vals)/2, 0, max(stim_vals)/2, max(stim_vals)], labels=['-100', '-50', '0', '50', '100'], fontsize=10)
            plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0', '0.25', '0.5', '0.75', '1'], fontsize=10)
            plt.ylabel('P(R)', fontsize=10)
            plt.xlabel('stimulus', fontsize=10)
            plt.text(.3, .3, "Engaged-L", fontsize=9, color=cols[k])

        if k == 1:
            plt.subplot(1, 3, 1)
            plt.text(-2.2, .8, "Engaged-R", fontsize=9, color=cols[k])

        if k == 2:
            plt.subplot(1, 3, 2)
            plt.xticks([min(stim_vals), min(stim_vals)/2, 0, max(stim_vals)/2, max(stim_vals)], labels=['', '', '', '', ''], fontsize=8)
            plt.yticks([0, 0.25, 0.5, 0.75, 1], ['', '', '', '', ''], fontsize=8)
            plt.gca().tick_params(axis='x')
            plt.gca().tick_params(axis='y')
            plt.text(.3, .2, "Biased-L", fontsize=9, color=cols[k])

        if k == 3:
            plt.subplot(1, 3, 2)
            plt.xticks([min(stim_vals), min(stim_vals)/2, 0, max(stim_vals)/2, max(stim_vals)], labels=['', '', '', '', ''], fontsize=8)
            plt.yticks([0, 0.25, 0.5, 0.75, 1], ['', '', '', '', ''], fontsize=8)
            plt.gca().tick_params(axis='x')
            plt.gca().tick_params(axis='y')
            plt.text(-2.1, .85, "Biased-R", fontsize=9, color=cols[k])

        plt.axhline(y=0.5, color="k", alpha=0.45, ls=":", linewidth=0.5)
        plt.axvline(x=0, color="k", alpha=0.45, ls=":", linewidth=0.5)
        plt.ylim((-0.01, 1.01))
    # fig.savefig(figure_dir + 'Psycho.pdf')

    # Plot psycho all
    plt.subplot(1, 3, 3)
    D, C = 1, 2
    M = obs_mat.shape[1]
    M_trans = trans_mat.shape[1]
    this_hmm = HMM_TO(K, D, M_trans=M_trans, M_obs=M, observations="input_driven_obs_diff_inputs", observation_kwargs=dict(C=C, prior_sigma=prior_sigma),
                          transitions="inputdrivenalt", transition_kwargs=dict(prior_sigma=prior_sigma, alpha=alpha_val, kappa=0))

    this_hmm.params = Params_model

    # Sample y and z from this GLM-HMM:
    trial_num = 0
    stim_vals, _ = get_prob_right(-weight_vectors, obs_mat, 0, 1, 1)

    psychometric_grid = []
    latents = []
    datas = []
    for i, input in enumerate(inputs):
        T = input.shape[0]
        pc = (2 * bernoulli.rvs(0.5, size=1)) - 1  # Sample a value for t = 0 value of past choice covariate
        input[0, 1] = pc
        stim_side = (2 * bernoulli.rvs(0.5, size=1)) - 1  # Sample a value for t = 0 value of stim_side covariate
        input[0, 2] = stim_side
        latent_z = np.zeros(input.shape[0], dtype=int)
        data = np.zeros(input.shape[0], dtype=int)

        # Now loop through each time and get the state and the observation
        # for each time step:
        pi0 = np.exp(Params_model[0][0])
        latent_z[0] = int(npr.choice(K, p=pi0))

        # Get psychometric for each state:
        psychometric_this_t = np.zeros((K, len(stim_vals)))
        for k in range(K):
            _, psychometric_k_t = get_prob_right(-weight_vectors, obs_mat, k, pc, stim_side)
            psychometric_this_t[k] = pi0[k] * np.array(psychometric_k_t)
        psychometric_grid.append(np.sum(psychometric_this_t, axis=0))
        # Pt = load_Pt(path_analysis + 'transition_Posterior_K=' + str(K) + '.npz')
        Pt = np.exp(Params_model[1][0])

        for k in range(K):
            Pt[k, :] = Pt[k, :] / np.sum(Pt [k, :])

        for t in range(0, T):
            this_input = np.expand_dims(input[t], axis=0)
            # Get observation at current trial (based on state)
            data[t] = this_hmm.observations.sample_x(z=latent_z[t], xhist=None, observation_input=np.expand_dims(input[t], axis=0))

            # Get state at next trial
            if t < T - 1:
                latent_z[t + 1] = int(npr.choice(K, p=Pt[latent_z[t]]))
                # update past choice and stim_side based on sampled y and correct answer
                input[t + 1, 1] = 2 * data[t] - 1
                obs_mat[trial_num + 1, 1] = 2 * data[t] - 1
                rewarded = 2 * (data[t] == correct_answer[trial_num]) - 1
                input[t + 1, 2] = input[t + 1, 1] * rewarded
                obs_mat[trial_num + 1, 2] = input[t + 1, 1] * rewarded
                # Get psychometric for each state:
                psychometric_this_t = np.zeros((K, len(stim_vals)))
                for k in range(K):
                    _, psychometric_k_t = get_prob_right(-weight_vectors, obs_mat, k, 2 * data[t] - 1, (2 * data[t] - 1) * rewarded)
                    psychometric_this_t[k] = Pt[latent_z[t]][k] * np.array(psychometric_k_t)
                psychometric_grid.append(np.sum(psychometric_this_t, axis=0))
            trial_num += 1
        latents.append(latent_z)
        datas.append(data)
    latents_flattened = np.concatenate(latents)
    datas_flattened = np.concatenate(datas)
    assert trial_num == obs_mat.shape[0]

    # Plot psychometric curve
    obs_mat_df = pd.DataFrame({'signed_contrast': obs_mat[:, 0], 'choice': y[:, 0]})

    plt.plot(stim_vals, np.mean(psychometric_grid, axis=0), '-', color='black', linewidth=0.9)
    sns.lineplot(data=obs_mat_df, x="signed_contrast", y="choice", err_style="bars", linewidth=0, linestyle='None', mew=0,
                 marker='^', markersize=4.5, errorbar=('ci', 95), err_kws={"linewidth": 0.75}, zorder=3, color='green')
    plt.xlabel('', fontsize=10)
    plt.ylabel('', fontsize=8)
    plt.axhline(y=0.5, color="k", alpha=0.45, ls=":", linewidth=0.5)
    plt.axvline(x=0, color="k", alpha=0.45, ls=":", linewidth=0.5)
    plt.xticks([min(stim_vals), min(stim_vals) / 2, 0, max(stim_vals) / 2, max(stim_vals)], labels=['', '', '', '', ''],
               fontsize=8)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], ['', '', '', '', ''], fontsize=8)
    plt.gca().tick_params(axis='x')
    plt.gca().tick_params(axis='y')
    fig.savefig(figure_dir + 'psycho_all.pdf')




