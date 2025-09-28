# Simulate data from global fit GLM-HMM
"""this is for only GLM simulated data"""

import numpy as np
import numpy.random as npr
import json
import ssm
from io_utils import get_mouse_info_all, cross_validation_vector, model_data_glmhmm, get_mouse_info, mice_names_info, load_reward_data, calculate_correct_ans
from analyze_results_utils import get_file_name_for_best_model_fold_indiv, data_segmentation_session, mask_for_violations
from scipy.stats import bernoulli

npr.seed(67)

if __name__ == '__main__':
    D = 1  # data dimension
    C = 2  # number of output types/categories
    num_inputs = 4
    alpha_val = 2.0
    prior_sigma = 4.0
    K = 4
    num_simulations = 1  # Simulation number

    path_data = '../../glm-hmm_package/data/ibl/Della_cluster_data/separate_mouse_data/'
    mice_names = mice_names_info(path_data + 'mice_names.npz')

    for j in range(num_simulations):
        all_animals_latents = []
        all_animals_latents = []
        all_animals_datas = []

        for z, animal in enumerate(mice_names):
            path_analysis = '../../glm-hmm_package/glm-hmm_all_no_GLM-T_to_compare/results/model_indiv_ibl/' + 'num_regress_obs_' + str(
                num_inputs) + '/' + 'prior_sigma_' + str(prior_sigma) + '_transition_alpha_' + str(alpha_val) + '/' + animal + '/'
            dir_for_save = '../../glm-hmm_package/results/figures_for_paper/Supp_figures/figure_5/without GLM-T/sim_data/'
            figure_dir = '../../glm-hmm_package/results/figures_for_paper/Supp_figures/figure_5/without GLM-T'
            cv_file = path_analysis + "/diff_folds_fit.npz"
            diff_folds_fit = cross_validation_vector(cv_file)

            # Also get data for animal:
            obs_mat, trans_mat, y, session, left_probs, animal_eids = get_mouse_info(path_data + animal + '_processed.npz')
            rewarded = load_reward_data(path_data + animal + '_rewarded.npz')
            correct_answer = calculate_correct_ans(y, rewarded)
            T = obs_mat.shape[0]

            index_viols = np.where(y == -1)[0]
            nonindex_viols, mask = mask_for_violations(index_viols, obs_mat.shape[0])
            y[np.where(y == -1), :] = 1
            inputs, inputs_trans, datas, masks = data_segmentation_session(obs_mat, trans_mat, y, mask, session)

            M_trans = np.array(inputs_trans[0]).shape[1]
            with open(path_analysis + "/best_init_cvbt_dict.json", 'r') as f:
                optimal_initialize_dict = json.load(f)

            # Get the file name corresponding to the best initialization for given K value
            params_and_LL = get_file_name_for_best_model_fold_indiv(diff_folds_fit, K, path_analysis, optimal_initialize_dict)
            Params_model, lls = model_data_glmhmm(params_and_LL)

            D, M, C = 1, 3, 2
            this_hmm = ssm.HMM(K, D, M, observations="input_driven_obs",
                               observation_kwargs=dict(C=C, prior_sigma=prior_sigma),
                               transitions="sticky", transition_kwargs=dict(alpha=alpha_val, kappa=0))

            this_hmm.params = Params_model
            latents = []
            datas = []
            trial_num = 0
            for i, input in enumerate(inputs):  # for each session
                T = input.shape[0]
                # Sample a value for t = 0 value of past choice covariate
                pc = (2*bernoulli.rvs(0.5, size=1)) - 1
                input[0, 1] = pc
                # Sample a value for t = 0 value of stim_side covariate
                stim_side = (2*bernoulli.rvs(0.5, size=1)) - 1
                input[0, 2] = stim_side
                latent_z = np.zeros(input.shape[0], dtype=int)
                data = np.zeros(input.shape[0], dtype=int)

                # Now loop through each time and get the state and the observation for each time step:
                pi0 = np.exp(Params_model[0][0])
                latent_z[0] = int(npr.choice(K, p=pi0))

                for t in range(0, T):
                    Pt = np.exp(Params_model[1][0])
                    data[t] = this_hmm.observations.sample_x(latent_z[t], xhist=None, input=np.expand_dims(input[t], axis=0), tag=None)

                    # Get state at next trial
                    if t < T-1:
                        latent_z[t+1] = int(npr.choice(K, p=Pt[latent_z[t]]))
                        # update past choice and stim_side based on sampled y and correct answer
                        pc = 2 * data[t] - 1
                        input[t+1, 1] = pc
                        rewarded = 2*(data[t] == correct_answer[trial_num]) - 1
                        stim_side = pc * rewarded
                        input[t+1, 2] = stim_side

                    trial_num += 1
                latents.append(latent_z)
                datas.append(data)
            latents_flattened = np.concatenate(latents)
            datas_flattened = np.concatenate(datas)
            animal_dir_for_save = dir_for_save + animal
            import os
            os.makedirs(animal_dir_for_save, exist_ok=True)
            np.savez(animal_dir_for_save + '/simulation_' + str("%03d" % j) + '.npz', obs_mat, trans_mat,  # trans_mat here, is only for the structure of functions and it will not be used in the data analysis
                     np.expand_dims(datas_flattened.astype('int'), axis=1), session, latents_flattened)

            # Save data for all animals
            all_animals_latents.append(latents_flattened)
            all_animals_datas.append(datas_flattened)

        all_animals_latents = np.concatenate(all_animals_latents)
        all_animals_datas = np.concatenate(all_animals_datas)

        all_animals_obs_mat, all_animals_trans_mat, all_data_y, all_animals_session = norm_obs_mat = get_mouse_info_all('../../glm-hmm_package/data/ibl/Della_cluster_data/combined_all_mice.npz')
        assert all_animals_obs_mat.shape[0] == all_animals_datas.shape[0]
        assert all_animals_obs_mat.shape[0] == all_animals_latents.shape[0]

        os.makedirs(dir_for_save, exist_ok=True)
        np.savez(dir_for_save + '/all_animals_simulation_' + str("%03d" % j) + '.npz', all_animals_obs_mat, all_animals_trans_mat,
                 np.expand_dims(all_animals_datas.astype('int'), axis=1), all_animals_session, all_animals_latents)