import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def compute_sorted_portfolios(panel, start, end, n_jobs=10):
    """
    Compute 5x5 sorted portfolio returns for each non-size characteristic.
    
    Args:
        panel: DataFrame with columns including size, bm, agr, roe, mom, mve, xret
        start: Starting month
        end: Ending month
        n_jobs: Number of parallel jobs
    
    Returns:
        DataFrame with columns for each portfolio return series
    """
    
    non_size_chars = ['bm', 'agr', 'roe', 'mom']
    
    def compute_month_portfolios(month):
        """Compute portfolio returns for a single month"""
        
        data = panel.loc[month].copy()
        
        # Drop any rows with missing values
        data = data.dropna(subset=['size', 'mve', 'xret'] + non_size_chars)
        
        if len(data) == 0:
            return None
        
        # Create size quintiles
        data['size_quintile'] = pd.qcut(data['size'], q=5, labels=['S1', 'S2', 'S3', 'S4', 'S5'])
        
        month_returns = {'month': month}
        
        for char in non_size_chars:
            # Create characteristic quintiles
            try:
                data[f'{char}_quintile'] = pd.qcut(data[char], q=5, labels=['C1', 'C2', 'C3', 'C4', 'C5'])
            except:
                # Handle case where we can't create 5 unique quintiles
                data[f'{char}_quintile'] = pd.cut(data[char], bins=5, labels=['C1', 'C2', 'C3', 'C4', 'C5'])
            
            # Create 25 groups
            for size_q in ['S1', 'S2', 'S3', 'S4', 'S5']:
                for char_q in ['C1', 'C2', 'C3', 'C4', 'C5']:
                    # Filter to this group
                    group = data[(data['size_quintile'] == size_q) & 
                                 (data[f'{char}_quintile'] == char_q)]
                    
                    if len(group) > 0:
                        # Equal-weighted return
                        ew_ret = group['xret'].mean()
                        
                        # Value-weighted return
                        weights = group['mve'] / group['mve'].sum()
                        vw_ret = (group['xret'] * weights).sum()
                    else:
                        ew_ret = np.nan
                        vw_ret = np.nan
                    
                    # Store returns with descriptive column names
                    month_returns[f'{char}_{size_q}_{char_q}_ew'] = ew_ret
                    month_returns[f'{char}_{size_q}_{char_q}_vw'] = vw_ret
        
        return month_returns
    
    # Compute portfolios for all months in parallel
    results_lst = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(compute_month_portfolios)(month) for month in range(start, end+1)
    )
    
    # Filter out None results
    results_lst = [r for r in results_lst if r is not None]
    
    if len(results_lst) == 0:
        return pd.DataFrame()
    
    # Convert to DataFrame
    portfolio_returns = pd.DataFrame(results_lst)
    portfolio_returns.set_index('month', inplace=True)
    portfolio_returns.sort_index(inplace=True)
    
    return portfolio_returns



def save_portfolio_results(portfolio_returns, model, iter_num):
    """
    Save portfolio returns to CSV files.
    
    Args:
        portfolio_returns: DataFrame with 5x5 sorted portfolio returns
        model: Model name ('bgn', 'kp', or 'gs')
        iter_num: Iteration number
    """
    
    # Add iteration column
    portfolio_returns['iter'] = iter_num
    
    # Save to CSV (append mode)
    portfolio_returns.to_csv(f'sorted_portfolios_{model}.csv', 
                            mode='a', 
                            header=not pd.io.common.file_exists(f'sorted_portfolios_{model}.csv'))
    
    return portfolio_returns