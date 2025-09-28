"""

Assorted auxiliary functions for GLM fit

"""
import autograd.numpy as np
import pandas as pd
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
import matplotlib.pyplot as plt
from GLM_class import glm

npr.seed(1)

C = 2
model_init_num = 10  # number of initializations


def get_mouse_info(mouse_data):
    container = np.load(mouse_data, allow_pickle=True)
    info = [container[key] for key in container]
    obs_mat = info[0]
    trans_mat = info[1]
    output = info[2]
    session = info[3]
    return obs_mat, trans_mat, output, session


# Separate data by session
def data_segmentation_session(obs_mat, y, session):
    inputs = []
    datas = []
    unique_sessions = np.unique(session)
    for sess in unique_sessions:
        idx = np.where(session == sess)[0]
        inputs.append(obs_mat[idx, :])
        datas.append(y[idx, :])
    return inputs, datas


def fit_glm(inputs, datas, M, **kwargs):
    new_glm = glm(M, C)
    new_glm.fit_glm(datas, inputs, masks=None, tags=None)

    # Obtain the log-likelihood of the training data:
    train_calculate_LL = new_glm.log_marginal(datas, inputs, None, None)
    train_calculate_LL = train_calculate_LL._value._value
    hessian = new_glm.hessian
    recovered_weights = new_glm.Wk._value._value
    # Update to include the first entry:
    std_dev = standard_deviation(hessian, M)

    # compare to CompHess
    def _obj(params):
        _obj_glm = glm(M, C)
        _obj_glm.params = params
        obj = _obj_glm.log_marginal(datas, inputs, masks=None, tags=None)
        return -obj

    pred_acc_by_session, class_log_probs_by_session = prediction_precision_compute(datas, inputs, new_glm)
    class_probs_by_session = np.exp(class_log_probs_by_session)
    return train_calculate_LL, recovered_weights, std_dev, class_probs_by_session


def prediction_precision_compute(datas, inputs, this_glm):
    pred_acc_by_session = []
    # Determine the most likely observation class at each time step
    for i, (data, obs_mat) in enumerate(zip(datas, inputs)):
        time_dependent_logits = this_glm.calculate_logits(obs_mat)
        # Perform marginalization over the latent dimension, which is of size 1 in this instance.
        # Therefore, no alterations should occur other than within the array structure.
        time_dependent_class_log_probs = logsumexp(time_dependent_logits, axis=1)
        assert time_dependent_class_log_probs.shape == (
            obs_mat.shape[0], time_dependent_logits.shape[2]), "wrong shape for time_dependent_class_log_probs"
        # find the location of the max along the C dimension
        predicted_class_labels = np.argmax(time_dependent_class_log_probs, axis=1)
        # Determine the alignment of y and predicted class labels
        predictive_acc = np.sum(data[:, 0] == predicted_class_labels) / data.shape[0]
        if i == 0:
            pred_acc_by_session.append(predictive_acc)
            class_log_probs_by_session = time_dependent_class_log_probs._value._value
        else:
            pred_acc_by_session.append(predictive_acc)
            class_log_probs_by_session = np.append(class_log_probs_by_session,
                                                   time_dependent_class_log_probs._value._value, axis=0)
    return pred_acc_by_session, class_log_probs_by_session


# Reconfigure the Hessian matrix and compute its inverse
def standard_deviation(hessian, M):
    hessian = np.reshape(hessian, (((C - 1) * (M + 1)), ((C - 1) * (M + 1))))
    inv_hessian = np.linalg.inv(hessian)
    # Extract the diagonal elements and compute their square roots.
    standard_deviation = np.sqrt(np.diag(inv_hessian))
    return standard_deviation


# Add a column of zeros to the weights matrix at the suitable position
def add_zero_column(weights):
    weights_tranpose = np.transpose(weights, (1, 0, 2))
    weights = np.transpose(
        np.vstack([weights_tranpose, np.zeros((1, weights_tranpose.shape[1], weights_tranpose.shape[2]))]), (1, 0, 2))
    return weights


def add_no_zero_column(weights):
    weights_tranpose = np.transpose(weights, (1, 0, 2))
    weights = np.transpose(
        np.vstack([weights_tranpose, np.zeros((1, weights_tranpose.shape[1], weights_tranpose.shape[2]))]), (1, 0, 2))
    return weights


# Determine the empirical probability of selecting the correct option based on the stimulus.
def empirical_prob_compute(y, stimulus):
    df = pd.DataFrame({'choice': y[:, 0], 'stim': stimulus})
    total_right = df.groupby('stim').choice.agg('sum')
    total_choice = df.groupby('stim').choice.agg('count')
    empirical_choice_distribution = total_right / total_choice
    # Calculate 95% confidence interval:
    standard_err = 2 * np.sqrt((empirical_choice_distribution * (1 - empirical_choice_distribution)) / total_choice)
    return empirical_choice_distribution, standard_err


# Retrieve the predictive distribution, including the mean and standard deviation for each stimulus value
def predictive_mean_std_compute(temporal_probabilities, stimulus):
    df = pd.DataFrame({'predictive_prob': temporal_probabilities[:, 1], 'stim': stimulus})
    mean_by_stim = df.groupby('stim').predictive_prob.agg('mean')
    std_by_stim = df.groupby('stim').predictive_prob.agg('std')
    return mean_by_stim, std_by_stim


# Transform class probabilities to conform to the shape TxC, as opposed to sessions_count x num_trials per session x C
def reshape_class_probabilities(temporal_probabilities):
    temporal_probabilities = np.array(temporal_probabilities[0])
    x_dim = temporal_probabilities.shape[0]
    y_dim = temporal_probabilities.shape[1]
    z_dim = temporal_probabilities.shape[2]
    temporal_probabilities = temporal_probabilities.reshape(x_dim * y_dim, z_dim)
    return temporal_probabilities


def load_fold_session_map(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    fold_mapping_session = data[0]
    return fold_mapping_session


def load_synthetic_data(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    synthetic_data = data[0]
    return synthetic_data


def read_test_idx_file(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    test_idx = data[0]
    return test_idx


def create_train_idx(test_idx, T):
    """
    Get array of train indices and 1/0 mask indicating inclusion in training data
    :param test_idx:
    :param T:
    :return:
    """
    mask = np.array([i not in test_idx for i in range(T)])
    train_idx = np.arange(T)[mask]
    mask = mask + 0
    assert len(train_idx) + len(test_idx) == T, "train and test idx do not include all dta!"
    return train_idx, mask


def mice_names_info(list_file):
    container = np.load(list_file, allow_pickle=True)
    data = [container[key] for key in container]
    mice_names = data[0]
    return mice_names


def regressors_weights_Figure(Ws, standard_deviation, figure_directory, title='true', save_title="true",
                       figure_covariates_names=[]):
    K = Ws.shape[0]
    K_prime = Ws.shape[1]
    M = Ws.shape[2] - 1
    # Iterate over the combinations of m and n for a given model, then visualize the weight vector corresponding to each combination
    fig = plt.figure(figsize=(7, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.15, bottom=0.27, right=0.95, top=0.95, wspace=0.3, hspace=0.3)

    for j in range(K):
        for k in range(K_prime - 1):
            # plt.subplot(K, K_prime, 1+j*K_prime+k)
            plt.plot(range(M + 1), -Ws[j][k], marker='o')
            if k != K_prime - 1:  # only have std dev information for non-zero weights
                # if real_weights.any() != None:
                # plt.plot(range(M+1), real_weights[j][k], marker='x', color = 'g', label = "generative weights")
                # plt.legend(loc = "upper right")
                plt.plot(range(M + 1), -Ws[j][k] + standard_deviation[range(k * (M + 1), (k + 1) * (M + 1))],
                         marker='o',
                         color='r')
                plt.plot(range(M + 1), -Ws[j][k] - standard_deviation[range(k * (M + 1), (k + 1) * (M + 1))],
                         marker='o',
                         color='r')
                plt.fill_between(range(M + 1),
                                 -Ws[j][k] - standard_deviation[range(k * (M + 1), (k + 1) * (M + 1))],
                                 -Ws[j][k] + standard_deviation[range(k * (M + 1), (k + 1) * (M + 1))], color='grey',
                                 alpha=0.2)
            plt.plot(range(-1, M + 2), np.repeat(0, M + 3), 'k', alpha=0.2)
            plt.axhline(y=0, color="k", alpha=0.5, ls="--")
            if len(figure_covariates_names) > 0:
                plt.xticks(list(range(0, len(figure_covariates_names))), figure_covariates_names, rotation='90',
                           fontsize=12)
            else:
                plt.xticks(list(range(0, 3)), ['Stimulus', 'Past Choice', 'Bias'], rotation='90', fontsize=12)
            plt.ylim((-3, 6))

    fig.text(0.04, 0.5, "Weight", ha="center", va="center", rotation=90, fontsize=15)
    fig.suptitle("GLM Weights: " + title,
                 y=0.99, fontsize=14)
    fig.savefig(figure_directory + 'glm_weights_' + save_title + '.png')
