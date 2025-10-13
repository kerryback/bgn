import pandas as pd, numpy as np

###########
## Read data
###########

R = pd.read_parquet(
    '/Volumes/sjpextp/DataWork/JKP_GlobalFactor/make20240726_25.parquet',
    columns = ["('return', 'ret_exc_lead1m')"] # flattened by pyarrow
    )

R = R.unstack() # months are rows, permnos are columns
R = R.droplevel(['type','name'], axis=1) # drop non-permno column names

