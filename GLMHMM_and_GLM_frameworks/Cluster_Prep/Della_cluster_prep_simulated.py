import numpy as np

# ======== for Global fit ========
K_vals = [3]  # number of states, for simulated data only 3 is needed
transition_alphas = [2.0]
prior_sigmas = [2.0]
num_inputss = [1]
# cross_valid_num_fold = 5
model_init_num = 50

# ======== for Individual fit ========
# K_vals = [2, 3, 4, 5]  # number of states
# transition_alphas = [1.0, 2.0]
# prior_sigmas = [0.25, 0.5, 1.0, 2.0, 4.0]
# num_inputss = [2]
# cross_valid_num_fold = 5
# model_init_num = 1


if __name__ == '__main__':
    cluster_job_arr = []
    # Get every combination of range(cross_valid_num_fold), range(model_init_num) and K_vals.  Index by z (cluster job id)
    # fold_idx = range(cross_valid_num_fold)
    # iter_idx = range(model_init_num)
    # for animal in animals:
    # for c in c_vals:

    # ======== for Global fit ========
    for num_inputs in num_inputss:
        for K in K_vals:
            # for i in range(cross_valid_num_fold):
            for j in range(model_init_num):
                cluster_job_arr.append([num_inputs, K, j])  # [prior_sigma, transition_alpha, K, i, j])

    # ======== for Individual fit ========
    # for num_inputs in num_inputss:
    #     for prior_sigma in prior_sigmas:
    #         for transition_alpha in transition_alphas:
    #             for K in K_vals:
    #                 for i in range(cross_valid_num_fold):
    #                     for j in range(model_init_num):
    #                         cluster_job_arr.append([prior_sigma, transition_alpha, K, num_inputs, i, j])#[prior_sigma, transition_alpha, K, i, j])
    print('len(cluster_job_arr) = ', len(cluster_job_arr))

    np.savez(
        '/data/simulated_data/cluster_job_arr.npz',
        cluster_job_arr)
