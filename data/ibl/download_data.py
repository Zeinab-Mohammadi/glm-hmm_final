"""

Downloading IBL data in a correct format

"""

from one.api import ONE

# Please use the Alyx password or other information if needed from 'one_params' at this link: https://int-brain-lab.github.io/iblenv/_modules/oneibl/params.html
# For example, the Alyx password is international. For any other issues or problems, please feel free to contact IBL.
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international')
datasets = one.alyx.rest('datasets', 'list', tag='2023_Q1_Mohammadi_et_al')
subjects = [d['file_records'][0]['relative_path'].split('/')[2] for d in datasets]

for subject in subjects:
    trials = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')




# ========= Below you can find another way to download the data. =========#

# import pandas as pd
# from one.api import ONE
# from data_utils import download_subjectTrials
#
# # use the function to download the data
# one = ONE(base_url='https://openalyx.internationalbrainlab.org')
# pqt_paths = download_subjectTrials(one)
#
# # load the data
# df = pd.read_parquet(pqt_paths[0])
# print(df)

