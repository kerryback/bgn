import pandas as pd
import numpy as np

def load_and_process_data(folder='new_results'):
    """
    Load CSV files and process them into the required dataframes for analysis.
    
    Parameters:
    folder (str): Directory containing the CSV files
    
    Returns:
    tuple: (dkkm, fama, ipca, dkkm_hjd_tables, dkkm_sharpe_tables)
    """
    
    # Replicate the exact construction from analysis3.ipynb
    df = None
    for theory in ['bgn', 'kp', 'gs']:
        sdf = pd.read_csv(f"{folder}/results_tseries_{theory}.csv")[['iter', 'month', 'sdf_ret']]
        for method in ['dkkm', 'fama', 'ipca', 'model']:
            data = pd.read_csv(f"{folder}/results_{method}_{theory}.csv")
            if method == 'dkkm':
                data = data.rename(columns={
                    'include_mkt': 'market',
                    'alpha': 'kappa'})
                data['market'] = data.market.astype(bool)
                data['standardized'] = data['method'].map({'rs': True, 'nors': False})
            elif method=='ipca': 
                data = data.rename(columns={'ipca factors': 'ipca_factors'})
            data['factor_method'] = method 
            data['theory'] = theory
            data = data.merge(sdf, on = ['iter', 'month'])
            df = pd.concat((df, data)) 
    df['sq_diff'] = (df.xret-df.sdf_ret)**2
    df['sharpe'] = df.mn / df.stdev

    # Create aggregations
    aggs = None 
    for method in ['dkkm', 'fama', 'ipca']:
        data = df[df.factor_method==method]
        if method=='dkkm':
            agg = data.groupby([
                'theory', 'standardized', 'market', 'kappa', 'nfeatures', 
            ])[['sq_diff', 'sharpe']].mean()
        if method=='fama':
                agg = data.groupby([
                'theory', 'method'
            ])[['sq_diff', 'sharpe']].mean()
        if method=='ipca':
                agg = data.groupby([
                'theory', 'ipca_factors'
            ])[['sq_diff', 'sharpe']].mean()
        agg.reset_index(inplace=True)
        agg['factor_method'] = method
        aggs = pd.concat((aggs, agg))

    # Create separate dataframes
    dkkm = aggs[aggs.factor_method=='dkkm'][[
        'theory', 'kappa', 'nfeatures', 'standardized', 'market', 'sq_diff', 'sharpe'
    ]]
    fama = aggs[aggs.factor_method=='fama'][['theory', 'method', 'sq_diff', 'sharpe']]
    ipca = aggs[aggs.factor_method=='ipca'][['theory', 'ipca_factors', 'sq_diff', 'sharpe']]

    dkkm['nfeatures'] = dkkm.nfeatures.astype(int)
    ipca['ipca_factors'] = ipca.ipca_factors.astype(int)

    # Create dictionaries for both HJD and Sharpe
    dkkm_hjd_tables = {}
    dkkm_sharpe_tables = {}
    for theory in ['bgn', 'kp', 'gs']:
        # HJD tables
        table = dkkm[dkkm.theory==theory][['standardized', 'market', 'kappa', 'nfeatures', 'sq_diff']]
        table = table.set_index(['standardized', 'market', 'kappa', 'nfeatures']).unstack()
        dkkm_hjd_tables[theory] = table
        
        # Sharpe tables
        table = dkkm[dkkm.theory==theory][['standardized', 'market', 'kappa', 'nfeatures', 'sharpe']]
        table = table.set_index(['standardized', 'market', 'kappa', 'nfeatures']).unstack()
        dkkm_sharpe_tables[theory] = table
    
    return dkkm, fama, ipca, dkkm_hjd_tables, dkkm_sharpe_tables