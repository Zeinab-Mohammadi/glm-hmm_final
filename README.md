# glm-hmm package

This package introduces an accessible framework for code designed to analyze and model various strategies and their transitions in decision-making task. The study focuses on exploring uncharacterized switching behavior in non-stationary environments, employing an internal state model with input-driven transitions. The methodology integrates a hidden Markov model (HMM) enriched with two sets of per-state generalized linear models (GLMs): a Bernoulli GLM for mice choices on each trial and a multinomial GLM for state transitions. This sophisticated framework adeptly captures the dynamic interplay between covariates, mouse choices, and state transitions, offering a more refined description of behavioral activity compared to classical models. For a detailed understanding of the modeling approach, please refer to our paper: [Identifying the factors governing internal state switches during nonstationary sensory decision-making](https://www.biorxiv.org/content/10.1101/2024.02.02.578482v2.abstract).

To understand how our model works, we recommend studying the Bayesian State Space Modeling (SSM) framework ([Linderman et al., 2020](https://github.com/lindermanlab/ssm)), which we used and extended. We added new functionality to the GLM-HMM and provided a code script, which was subsequently used for model inference in this manuscript.
Our modified version of the SSM package is available at [this repository](https://github.com/Zeinab-Mohammadi/ssm). For a practical demonstration of the package's application in our analysis, please refer to [this notebook](https://github.com/Zeinab-Mohammadi/ssm/blob/master/notebooks/2c-Input-Driven-Transitions-and-Observations-GLM-HMM.ipynb) which shows how a GLM-HMM can be implemented with **independent** covariates for transition GLM and observation GLM. For additional details, see [this code script](https://github.com/Zeinab-Mohammadi/ssm/blob/master/ssm/hmm_TO.py), a modified version of the hmm.py from the SSM package.

**Generalization of the code:** This package is designed to be applicable to any dataset. To use it with your data, you only need to adjust the data format accordingly. You can achieve this by editing the files in the data/ibl folder. The code supports both global and individual fits for modeling the data.

## Code Framework

This code is designed to execute the aforementioned GLM-HMM for the IBL dataset, incorporating HMM inference performed through the Expectation Maximization (EM) algorithm. The code is organized into the following sections:

- Data: Download and prepare the data, creating the design matrix for observation and transition covariates.
- GLMHMM_and_GLM_frameworks: Apply both GLM and GLM-HMM frameworks to the prepared data.
- After_fitting_evaluation_scripts_ibl: Prepared scripts to analyze the results of the models.
- Paper_figures_code: Generate figures for the paper.

### Running Instructions:

1) For data download, preprocessing, and preparing input design matrices for transition and observation models, run the scripts in the data/ibl folder.
   
2) For a global fit on pooled data from all mice, follow these steps:
     - Execute the `GLM_for_global_ibl.py` script from the `GLMHMM_and_GLM_frameworks` folder.
     - Run the `GLMHMM_global_ibl` script from the `GLMHMM_and_GLM_frameworks` folder.
     - Analyze model results by sequentially running `cross_validation_global_fit.py`, `determine_optimal_model_global.py`, `plot_conditional_probability_block_shade_global.py`, and `plot_weights_multiply_inputs_traces_global.py` in the `after_fitting_evaluation_scripts_ibl` folder.
       
3) For individual fits (separate fit for each mouse), run the following scripts:
     - Execute `GLM_for_individual_ibl` from the `GLMHMM_and_GLM_frameworks` folder.
     - Run `GLMHMM_individual_ibl` from the `GLMHMM_and_GLM_frameworks` folder.
     - Analyze each mouse model fit by sequentially running `cross_validation_indiv_fit.py`, `determine_optimal_model_indiv.py`, `plot_conditional_probability_block_shade_indiv.py`, and `plot_weights_multiply_inputs_traces_indiv.py` in the `after_fitting_evaluation_scripts_ibl` folder.

Note: If running the code on a cluster, execute `Della_cluster_prep.py` before each glm-hmm fit.

4) To plot corresponding figures mentioned in our paper [here](https://www.biorxiv.org/content/10.1101/2024.02.02.578482v2.abstract), run the scripts in the figures folder (Paper_figures_code).


## Installation

#### Prerequisites:

Ensure that Python 3.9 is installed for the proper execution of the code. Additionally, install our extended version of the SSM package from the Scott Linderman group by using the forked version available at [this repository](https://github.com/Zeinab-Mohammadi/ssm). This version supports the use of both observation and transition GLMs with different and independent covariates for a single model. Alternatively, you can install the original SSM package from their website and add [this code script](https://github.com/Zeinab-Mohammadi/ssm/blob/master/ssm/hmm_TO.py) to the same folder as the hmm.py code (and [this example notebook](https://github.com/Zeinab-Mohammadi/ssm/blob/master/notebooks/2c-Input-Driven-Transitions-and-Observations-GLM-HMM.ipynb) to the "notebooks" folder). Refer to the instructions on their website or use the command provided below to install this package:

```
cd ssm
pip install numpy cython
pip install -e .
```

#### Setup Instructions:
For a seamless experience, create a new virtual environment before running the code. This step helps prevent potential conflicts with other projects and version control issues.

#### Code Installation:
Download the code using the provided link. Alternatively, open a terminal window and enter the following command:

```
git clone https://github.com/Zeinab-Mohammadi/glm-hmm_final.git
```
This structure improves the flow and readability while providing clear instructions for running the code. 
