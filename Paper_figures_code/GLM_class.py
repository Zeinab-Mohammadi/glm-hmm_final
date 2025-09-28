import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import hessian
from autograd.scipy.special import logsumexp

# Import generic useful functions from ssm package
from ssm.util import ensure_args_are_lists
from ssm.optimizers import adam, bfgs, rmsprop, sgd
import ssm.stats as stats

# GLM class

class glm(object):
    
    def __init__(self, M, C):
        """
        @param C:  number of classes in the categorical observations 
        """
        #super(glm, self).__init__(M)
        self.M = M
        self.C = C
        # Parameters linking input to state distribution
        self.Wk = npr.randn(1, C-1, M+1)
        
    @property
    def params(self):
        return self.Wk
    
    @params.setter
    def params(self, value):
        self.Wk = value

    def log_prior(self):
        return 0
    
    # Calculate time dependent logits - output is matrix of size Tx1xC
    # Input is size TxM
    def calculate_logits(self, input):
        # Update input to include offset term:
        input = np.append(input, np.ones((input.shape[0],1)), axis = 1)
        # Add additional row (of zeros) to second dimension of self.Wk
        Wk_tranpose = np.transpose(self.Wk, (1,0,2))
        Wk = np.transpose(np.vstack([Wk_tranpose, np.zeros((1,Wk_tranpose.shape[1], Wk_tranpose.shape[2]))]), (1,0,2))
        # Input effect; transpose so that output has dims TxKxC
        time_dependent_logits = np.transpose(np.dot(Wk,input.T), (2,0,1))
        time_dependent_logits = time_dependent_logits - logsumexp(time_dependent_logits, axis=2, keepdims=True)
        return time_dependent_logits

    # Calculate log-likelihood of observed data
    def log_likelihoods(self, data, input, mask, tag):
        time_dependent_logits = self.calculate_logits(input)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.categorical_logpdf(data[:, None, :], time_dependent_logits[:,:,None,:], mask=mask[:, None, :])

    def sample_x(self, input, tag=None, with_noise=True):
        time_dependent_logits = self.calculate_logits(input)
        ps = np.exp(time_dependent_logits)
        T = time_dependent_logits.shape[0]
        if T>1:
            sample = np.array([npr.choice(self.C, p=ps[t, 0]) for t in range(T)])
        else:
            sample = np.array([npr.choice(self.C, p=ps[0, 0])])
        return sample

    # log marginal likelihood of data
    @ensure_args_are_lists
    def log_marginal(self, datas, inputs, masks, tags):
        elbo = self.log_prior()
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            lls =self.log_likelihoods(data, input, mask, tag)
            elbo += np.sum(lls)
        return elbo

    @ensure_args_are_lists
    def fit_glm(self, datas, inputs, masks, tags,
               optimizer="bfgs", num_iters=1000, **kwargs):
        optimizer_label = optimizer
        optimizer = dict(adam=adam, bfgs=bfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]
        
        # define optimization target
        def _objective(params, itr):
            self.params = params
            obj = self.log_marginal(datas, inputs, masks, tags)
            return -obj

        if optimizer_label == "bfgs":
            self.params = optimizer(_objective, self.params, num_iters=num_iters, **kwargs)
        else:
            self.params = optimizer(_objective, self.params, num_iters=num_iters, **kwargs)
        # Obtain Autograd Hessian and evaluate it at MAP parameters (to obtain posterior credible intervals)
        autograd_hessian = hessian(_objective)
        # Evaluate the Hessian at the MAP weights
        self.hessian = autograd_hessian(self.params, -1)
