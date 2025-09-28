
import numpy.random as npr
import numpy as np
import ssm


def get_file_name_for_best_model_fold(diff_folds_fit, K, path_main_folder, optimal_initialize_dict):
    """
    Get the file name for the best initialization for the K value specified
    :param diff_folds_fit:
    :param K:
    :param models:
    :param path_main_folder:
    :param optimal_initialize_dict:
    :return:
    """
    # Identify best fold for best model:
    loc_best = 0
    best_fold = np.where(diff_folds_fit[loc_best, :] == max(diff_folds_fit[loc_best, :]))[0][0]
    base_path = path_main_folder + '/Model/glmhmm_#state=' + str(K) + '/fld_num=' + str(best_fold)
    keys = '/glmhmm_#state=' + str(K) + '/fld_num=' + str(best_fold)
    best_iter = optimal_initialize_dict[keys]
    params_and_LL = base_path + '/iter_' + str(best_iter) + '/glm_hmm_raw_parameters_itr_' + str(best_iter) + '.npz'
    return params_and_LL

def get_file_name_for_best_model_fold_no_GLM_T(cvbt_folds_model, K, overall_dir,
                                      best_init_cvbt_dict):
    '''
    Get the file name for the best initialization for the K value specified
    :param cvbt_folds_model:
    :param K:
    :param models:
    :param overall_dir:
    :param best_init_cvbt_dict:
    :return:
    '''
    # Identify best fold for best model:
    # loc_best = K - 1
    loc_best = 0
    best_fold = np.where(cvbt_folds_model[loc_best, :] == max(cvbt_folds_model[
        loc_best, :]))[0][0]
    base_path = overall_dir + '/GLM_HMM_K_' + str(K) + '/fold_' + str(
        best_fold)
    key_for_dict = '/GLM_HMM_K_' + str(K) + '/fold_' + str(best_fold)
    best_iter = best_init_cvbt_dict[key_for_dict]
    raw_file = base_path + '/iter_' + str(
        best_iter) + '/glm_hmm_raw_parameters_itr_' + str(best_iter) + '.npz'
    return raw_file

def get_marginal_posterior(globe, inputs, datas, hmm_params, K, permutation, transition_alpha, prior_sigma):
    M = inputs[0].shape[1]
    D = datas[0].shape[1]
    if globe== True:
        prior_sigma =100
    this_hmm = ssm.HMM(K, D, M,
                           observations="input_driven_obs", observation_kwargs=dict(C=2, prior_sigma=prior_sigma),
                           transitions="sticky", transition_kwargs=dict(alpha=transition_alpha, kappa=0))

    this_hmm.params = hmm_params
    # Get expected states:
    expectations = [this_hmm.expected_states(data=data, input=input)[0] #, train_mask=train_mask)[0]
                    for data, input
                    in zip(datas, inputs)]
    # Convert this now to one array:
    posterior_probs= np.concatenate(expectations, axis=0)
    posterior_probs = posterior_probs[:,permutation]
    return posterior_probs
def load_rts(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    rt_dta = data[0]
    rt_session = data[1]
    return rt_dta, rt_session

def mice_names_info(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    mice_names = data[0]
    return mice_names

def load_cv_arr(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    cvbt_folds_model = data[0]
    return cvbt_folds_model

def load_glmhmm_data(data_file):
    container = np.load(data_file, allow_pickle=True)
    data = [container[key] for key in container]
    this_hmm_params = data[0]
    lls = data[1]
    return [this_hmm_params, lls]

def calculate_state_permutation_all_data(hmm_params):
    '''
    If K = 3, calculate the permutation that results in states being ordered as engaged/bias left/bias right
    Else: order states so that they are ordered by engagement
    :param hmm_params:
    :return: permutation
    '''

    glm_weights = -hmm_params[2]
    K = glm_weights.shape[0]
    if K ==3:
        # want states ordered as engaged/bias left/bias right
        M = glm_weights.shape[2] - 1
        # bias coefficient is last entry in dimension 2
        engaged_loc = np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        reduced_weights = np.copy(glm_weights)
        # set row in reduced weights corresponding to engaged to have a bias that will not
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        state_order = [engaged_loc, bias_left_loc]
        bias_right_loc = np.arange(3)[np.where([range(3)[i] not in state_order for i in range(3)])][0]
        permutation = np.array([engaged_loc, bias_left_loc, bias_right_loc])

    elif K ==4:
        # want states ordered as engaged/bias left/bias right
        M = glm_weights.shape[2] - 1
        # bias coefficient is last entry in dimension 2

        first_max = np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        # Get index for the second highest value.
        second_max= glm_weights[:, 0, 0].argsort()[-2]
        engaged_loc = np.where((glm_weights[:, 0, M] == min((glm_weights[first_max, 0, M], glm_weights[second_max, 0, M]))))[0][0]
        #finding second state
        glm_weights_2nd=np.copy(glm_weights)
        glm_weights_2nd[engaged_loc,0,0]=0 # to remove the effect of first max
        engaged_loc_2nd= np.where((glm_weights_2nd[:, 0, 0] == max(glm_weights_2nd[:, 0, 0])))[0][0] # this is the second max and so the second state

        reduced_weights = np.copy(glm_weights)
        # set row in reduced weights corresponding to engaged to have a bias that will not
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        reduced_weights[engaged_loc_2nd, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        # bias_right_loc = np.where((reduced_weights[:, 0, M] == max(reduced_weights[:, 0, M])))[0][0]
        # print('bias_right_loc=', bias_right_loc)
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        # print('bias_left_loc=', bias_left_loc)
        state_order = [engaged_loc, engaged_loc_2nd, bias_left_loc]
        other_loc = np.arange(4)[np.where([range(4)[i] not in state_order for i in range(4)])][0]
        # permutation = np.array([engaged_loc, engaged_loc_2nd, bias_left_loc, bias_right_loc])
        permutation = np.array([engaged_loc, engaged_loc_2nd, bias_left_loc, other_loc])

        print('permutation=', permutation)

    elif K == 5:
        # want states ordered as engaged/bias left/bias right
        M = glm_weights.shape[2] - 1      # bias coefficient is last entry in dimension 2
        M2=glm_weights.shape[2] - 3       # Past choice is the third entry from last

        first_max = np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        # Get index for the second highest value.
        second_max = glm_weights[:, 0, 0].argsort()[-2]
        engaged_loc = np.where((glm_weights[:, 0, M] == min((glm_weights[first_max, 0, M], glm_weights[second_max, 0, M]))))[0][0]

        # finding second state
        glm_weights_2nd = np.copy(glm_weights)
        glm_weights_2nd[engaged_loc, 0, 0] = 0  # to remove the effect of first max
        engaged_loc_2nd = np.where((glm_weights_2nd[:, 0, 0] == max(glm_weights_2nd[:, 0, 0])))[0][0]  # this is the second max and so the second state
        # finding fifth state (past choice)
        past_choice_loc = np.where((glm_weights_2nd[:, 0, M2] == max(glm_weights_2nd[:, 0, M2])))[0][0]
        reduced_weights = np.copy(glm_weights)
        # set row in reduced weights corresponding to engaged to have a bias that will not
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        reduced_weights[engaged_loc_2nd, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        reduced_weights[past_choice_loc, 0, M] = 10000 #this is just to have a huge number so not to go for biase selection in bias_left_loc
        # bias_right_loc = np.where((reduced_weights[:, 0, M] == max(reduced_weights[:, 0, M])))[0][0]
        # print('bias_right_loc=', bias_right_loc)
        bias_left_loc = np.where((reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[0][0]
        # print('bias_left_loc=', bias_left_loc)
        state_order = [engaged_loc, engaged_loc_2nd, bias_left_loc, past_choice_loc]
        # permutation = np.array([engaged_loc, engaged_loc_2nd, bias_left_loc, past_choice_loc])

        # order states by engagement: with the most engaged being first.  Note: argsort sorts inputs from smallest to largest (hence why we convert to -ve glm_weights)
        other_loc = np.arange(5)[np.where([range(5)[i] not in state_order for i in range(5)])][0]
        permutation = np.array([engaged_loc, engaged_loc_2nd, bias_left_loc, other_loc, past_choice_loc])
        # other_loc = np.arange(5)[np.where([range(5)[i] not in state_order for i in range(5)])][0]
        # permutation = np.array([engaged_loc, engaged_loc_2nd, bias_left_loc, bias_right_loc, other_loc])
    else:
        permutation = np.argsort(-glm_weights[:, 0, 0])
    return permutation

def get_mouse_info_all(mouse_data):
    container = np.load(mouse_data, allow_pickle=True)
    data = [container[key] for key in container]
    animal_inpt = data[0]
    trans_mat = data[1]
    animal_y = data[2]
    animal_session = data[3]
    left_probs = data[4]
    mice_session_eids = data[5]
    return animal_inpt, trans_mat, animal_y, animal_session, left_probs, mice_session_eids

def read_bootstrapped_median(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    median, lower, upper, mean_viol_rate_dist = data[0], data[1], data[2], \
                                                data[3]
    return median, lower, upper, mean_viol_rate_dist


def perform_bootstrap_individual_animal(rt_eng_vec,
                                        rt_dis_vec,
                                        data_quantile,
                                        quantile=0.9):
    distribution = []
    for b in range(5000):
        # Resample points with replacement
        sample_eng = np.random.choice(rt_eng_vec, len(rt_eng_vec))
        # Get sample quantile
        sample_eng_quantile = np.quantile(sample_eng, quantile)
        sample_dis = np.random.choice(rt_dis_vec, len(rt_dis_vec))
        sample_dis_quantile = np.quantile(sample_dis, quantile)
        distribution.append(sample_dis_quantile - sample_eng_quantile)
    # Now return 2.5 and 97.5
    max_val = np.max(distribution)
    min_val = np.min(distribution)
    lower = np.quantile(distribution, 0.025)
    upper = np.quantile(distribution, 0.975)
    frac_above_true = np.sum(distribution >= data_quantile) / len(distribution)
    return lower, upper, min_val, max_val, frac_above_true

def mask_for_violations(index_viols, T):
    """
    Return indices of nonviolations and also a Boolean mask for inclusion (1 = nonviolation; 0 = violation)
    :param test_idx:
    :param T:
    :return:
    """
    mask = np.array([i not in index_viols for i in range(T)])
    nonindex_viols = np.arange(T)[mask]
    mask = mask + 0
    assert len(nonindex_viols) + len(index_viols) == T, "violation and non-violation idx do not include all data!"
    return nonindex_viols, mask



def data_segmentation_session(obs_mat, trans_mat, y, mask, session):
    """
    Partition obs_mat, y, mask by session
    :param obs_mat: arr of size TxM
    :param y:  arr of size T x D
    :param mask: Boolean arr of size T indicating if element is violation or not
    :param session: list of size T containing session ids
    :return: list of obs_mat arrays, data arrays and mask arrays, where the number of elements in list = number of sessions and each array size is number of trials in session
    """
    inputs = []
    inputs_trans = []
    datas = []
    indexes = np.unique(session, return_index=True)[1]
    unique_sessions = [session[index] for index in sorted(indexes)]  # ensure that unique sessions are ordered as they are in session
    counter = 0
    masks = []
    for sess in unique_sessions:
        idx = np.where(session == sess)[0]
        counter += len(idx)
        inputs.append(obs_mat[idx, :])
        inputs_trans.append(trans_mat[idx, :])
        datas.append(y[idx, :])
        masks.append(mask[idx])
    assert counter == obs_mat.shape[0], "not all trials assigned to session!"
    return inputs, inputs_trans, datas, masks
def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    inpt_trans = data[1]
    y = data[2]
    y = y.astype('int')
    session = data[3]
    left_probs = data[4]
    # print('len(data)=',len(data))
    if len(data) > 3:
        animal_eids = data[5]
    else:
        animal_eids=['not_added_to_this_file']
    return inpt, inpt_trans, y, session, left_probs, animal_eids


def get_file_name_for_best_model_fold_GLM_O(cvbt_folds_model, K, overall_dir, best_init_cvbt_dict):
    '''
    Get the file name for the best initialization for the K value specified
    :param cvbt_folds_model:
    :param K:
    :param models:
    :param overall_dir:
    :param best_init_cvbt_dict:
    :return:
    '''
    # Identify best fold for best model:
    #loc_best = K - 1
    loc_best = 0
    best_fold = np.where(cvbt_folds_model[loc_best, :] == max(cvbt_folds_model[loc_best, :]))[0][0]
    print('overall_dir =',overall_dir )
    base_path = overall_dir + '/GLM_HMM_K_' + str(K) + '/fold_' + str(best_fold)
    key_for_dict =  '/GLM_HMM_K_' + str(K) + '/fold_' + str(best_fold)
    best_iter = best_init_cvbt_dict[key_for_dict]
    raw_file = base_path + '/iter_' + str(best_iter) + '/glm_hmm_raw_parameters_itr_' + str(best_iter) + '.npz'
    return raw_file