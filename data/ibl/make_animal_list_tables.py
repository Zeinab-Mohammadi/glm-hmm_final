"""

Identify the distinctive animals within the IBL dataset

"""

# Identify unique animals in IBL dataset that enter biased blocks, and save a dictionary with each animal and
# a list of their eids in the biased blocks
import numpy as np
import numpy.random as npr
import json
import os
import pandas as pd
from collections import defaultdict
from one.api import ONE
from data_utils import mouse_identifier

npr.seed(65)


if __name__ == '__main__':
    ii = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    mice_names = []  # to include animals names

    one = ONE(base_url='https://alyx.internationalbrainlab.org')
    mice_session_eids = defaultdict(list)  # eids based on the subjects (mice IDs)
    path = '../../glm-hmm_package/data/ibl/tables_new'

    for dirname, dirs, files in os.walk(path):
        for filename in files:
            filename_without_extension, extension = os.path.splitext(filename)
            if extension == '.pqt':  # each filename is an animal
                pqt_file = os.path.join(dirname + '/' + filename)
                df_trials = pd.read_parquet(pqt_file)
                eids = list(df_trials['session'])
                unique_eids = list(df_trials['session'].unique())
                num_1 += 1

                for i, eid_ONE in enumerate(unique_eids):
                    session_trials = df_trials[df_trials['session'] == unique_eids[i]]
                    # below code is because sometimes there is not any probabilityLeft
                    try:
                        probability_stim = session_trials['probabilityLeft']._values
                    except Exception:
                        probability_stim = []
                        ii += 1
                        continue
                    assess_values = np.unique(probability_stim) == np.array([0.2, 0.5, 0.8])

                    if isinstance(assess_values, np.ndarray):  # Check if the comparison is an array
                        # update def of comparison to single True/False
                        assess_values = assess_values.all()
                    if assess_values == True:
                        num_3 += 1
                        mouse_name = mouse_identifier(pqt_file)
                        if mouse_name not in mice_names:
                            num_2 += 1
                            mice_names.append(mouse_name)
                        mice_session_eids[mouse_name].append(eid_ONE)

    json = json.dumps(mice_session_eids)
    f = open(os.getcwd() + "/mice_session_eids.json", "w")
    f.write(json)
    f.close()

    np.savez(os.getcwd() + '/mice_names.npz', mice_names)
    print('number of animals=', np.array(mice_names).shape)
