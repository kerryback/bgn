# specify how many iterations to run 
startiter, numiters = 0, 10
model = 'gs' 

## other parameters 
# N= number of firms, T= number of time periods (not including burnin = 200)
# max_features= max DKKM features
# n_ipca_rff= number of RFF features in IPCA-DKKM hybrid
N, T, n_ipca_rff = 100, 400, 36

include_mkt = False # dummy to include market in DKKM
nmat = 1 # number of weights matrices for DKKM
nfeatures_lst = [6, 36, 360]#, 3600]  # different feature numbers used for DKKM
max_features = max(nfeatures_lst) 
alpha_lst_fama =  [0] # B-J shrinkage for Fama methods
alpha_lst =  [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1] # B-J shrinkage for DKKM
if model =='gs':
    alpha_lst = [0, 0.0000001, 0.000001,0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
ipca_nfactors_lst = [1, 2]#, 3] #, 4]#, 5, 6] # number of IPCA factors considered
n_jobs = 10 # number of jobs in parallelized tasks

import numpy as np 
import pandas as pd 
import os
import csv
import scipy.linalg as linalg
from joblib import Parallel, delayed
from datetime import datetime

from strategies import fama_functions as fama 
from strategies import dkkm_functions as dkkm
from models import panel_functions as bgn
from models import panel_functions_kp14 as kp
from models import panel_functions_gs21 as gs
from strategies import ipca_functions as ipca
from models import sdf_compute as sdf_bgn
from models import sdf_compute_kp14 as sdf_kp
from models import sdf_compute_gs21 as sdf_gs
from models.parameters import *
from strategies import sorted_portfolios as sorts

# dictionaries to map "model"
panels = {'bgn': bgn,'kp': kp, 'gs':gs}
sdf = {'bgn': sdf_bgn,'kp': sdf_kp, 'gs': sdf_gs}
loading = {'bgn': ['A_1_', 'A_2_'],
           'kp': ['A_1_', 'A_2_'],
           'gs': ['A_1_']}
factor = {'bgn': ['f_1_', 'f_2_'],
          'kp': ['f_1_', 'f_2_'],
          'gs': ['f_1_']}


# Remove old portfolio weight files
files_to_clear = [f'port_{model}.csv'] + [f'port_{m}_{model}.csv' for m in ['dkkm', 'fama', 'ipca', 'model']] + [f'sorted_portfolios_{model}.csv']
for file_path in files_to_clear:
    if os.path.exists(file_path):
        os.remove(file_path)

# Define fieldnames for each output file
file_specs = {
    f'port_{model}.csv': ['iter', 'month'] + [f'firm_{i+1}' for i in range(N)],
    f'port_model_{model}.csv': ['iter', 'month', 'method'] + [f'firm_{i+1}' for i in range(N)],
    f'port_dkkm_{model}.csv': ['iter', 'month', 'include_mkt', 'method', 'mat', 'nfeatures', 'alpha'] 
                               + [f'firm_{i+1}' for i in range(N)],
    f'port_fama_{model}.csv': ['iter', 'month', 'method', 'alpha'] + [f'firm_{i+1}' for i in range(N)],
    f'port_ipca_{model}.csv': ['iter', 'month', 'nfactors'] + [f'firm_{i+1}' for i in range(N)],
}

# Write headers
for file_path, fieldnames in file_specs.items():
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

def run_panel(iter):

    # this is BGN_panels up to the point of passing to SDF_compute
    arr_tuple = panels[model].create_arrays(N, T+burnin)
    panel = panels[model].create_panel(N, T+burnin, arr_tuple) 


    # sdf_loop is the function applied in a loop in SDF_compute
    # we will apply it to each month separately
    sdf_loop = sdf[model].sdf_compute(N, T+burnin, arr_tuple)

    # this was at the beginning of the fama and dkkm scripts
    panel["size"] = np.log(panel.mve)
    panel = panel[panel.month>=2]
    panel.replace([np.inf, -np.inf], np.nan, inplace=True)
    panel.set_index(["month", "firmid"], inplace=True)
    nans = panel[chars+["mve", "xret"]].isnull().any(axis=1)
    keep = nans[~nans].index
    panel = panel.loc[keep]
    start = panel.index.unique("month").min()
    end = panel.index.unique("month").max()
    
    # compute factors for latent factor model
    model_premia = {}
    for method in ["taylor", "proj"]:
        model_premia[method] = panel.groupby(['month']).apply(lambda df: pd.Series(    
            np.linalg.lstsq(df[[x + method for x in loading[model]]].values, df['xret'].values, rcond=None)[0],
            index=[x + method for x in factor[model]])) 
    
    print(f"finished model factors at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    

    # compute factor returns and weights for IPCA

    # list of weight matrices
    #W_lst = {}
    #for i in range(20):
    #    W = np.random.normal(size=(int(n_ipca_rff/2), nchars))
    #    gamma = np.random.choice(gamma_grid, size=(int(n_ipca_rff/2), 1))
    #    W_lst[i] = gamma*W

    # IPCA w/ RFF characteristics
    #ipca_rff_lst = Parallel(n_jobs=n_jobs, verbose=0)(
    #    delayed(ipca.fit_ipca_360)(panel, i, N, start, end, 1, W_lst[j]) for i in ipca_nfactors_lst for j in range(20) 
    #)

    # IPCA w/ base characteristics
    ipca_lst = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(ipca.fit_ipca_360)(panel, i, N, start, end, 0, 0) for i in ipca_nfactors_lst
    )

    # retrieve weights from output
    #ipca_rff_weights_on_stocks, ipca_rff_factor_weights = {}, {}
    #for (i, m) in enumerate(ipca_nfactors_lst):
    #    ipca_rff_weights_on_stocks[m] = {}
    #    ipca_rff_factor_weights[m] = {}
    #    for j in range(20):
    #        ipca_rff_weights_on_stocks[m][j]= ipca_rff_lst[20*i + j][0]
    #        ipca_rff_factor_weights[m][j]= ipca_rff_lst[20*i+j][1]


    ipca_weights_on_stocks, ipca_factor_weights = {}, {}
    for (i, m) in enumerate(ipca_nfactors_lst):
        ipca_weights_on_stocks[m]= ipca_lst[i][0]
        ipca_factor_weights[m]= ipca_lst[i][1]

    print(f"finished ipca factors at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")

    # compute sorted portfolios
    sorted_ports = sorts.compute_sorted_portfolios(panel, start, end, n_jobs=n_jobs)
    print(f"finished sorted portfolios at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")

    # compute factors as in fama script for classical methods
    ff_rets = fama.factors(fama.fama_french, panel, n_jobs=n_jobs, start=start, end=end)
    fm_rets = fama.factors(fama.fama_macbeth, panel, n_jobs=n_jobs, start=start, end=end)

    #fm_rets = panel.groupby('month').apply(lambda x: dkkm.rank_standardize(x[chars]).T @ x.xret)
    #fm_rets['mkt_rf'] = panel.groupby('month')['xret'].mean()

    all_rets = panel.xret.unstack()

    print(f"finished fama factors at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")

    # compute max_features DKKM features 
    def generate_rff_panel(i):
        W = np.random.normal(size=(int(max_features/2), nchars + (model == 'bgn'))) #*  np.repeat(np.eye(nchars), int(max_features/(2*nchars)), axis=0)# append entry for rf rate
        gamma = np.random.choice(gamma_grid, size=(int(max_features/2), 1))
        W = gamma*W
        print(f'started weight matrix {i}')
        del gamma
        res_rs, res_nors = dkkm.factors(panel=panel, W=W, n_jobs=n_jobs, start=start, end=end, model=model) 
        return W, res_rs, res_nors
    
    dkkm_lst = [generate_rff_panel(i) for i in range(nmat)]
    dkkm_lst_rs = [(dkkm_lst[i][0], dkkm_lst[i][1]) for i in range(nmat)]
    dkkm_lst_nors = [(dkkm_lst[i][0], dkkm_lst[i][2]) for i in range(nmat)]
    print(f"finished DKKM factors at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")

    # output for a single month
    def run_month(month, iter):

        # returns from date 1 to date 2 (month 1) are the first elements 
        # of sdf_loop array, so at month 0
        sdf_ret, max_sr, rp, cond_var = sdf_loop(month-1, iter)
  
        # stock parameters
        data = panel.loc[month]
        keep_this_month = [tpl[1] for tpl in keep if tpl[0] == month]
        stock_cov = cond_var[keep_this_month, :][:, keep_this_month]
        rp = rp[keep_this_month]
        cov_inv = linalg.pinv(stock_cov)
        second_moment = stock_cov + np.outer(rp, rp)
        second_moment_inv = linalg.pinv(second_moment)
        
        # projected mu and xi
        projs = data[[x + 'proj' for x in loading[model]]].loc[keep_this_month].to_numpy()
        rets = data.xret.loc[keep_this_month].to_numpy().reshape(-1, 1)
        proj = (cov_inv @ projs).T @ rets
        #proj = (cov_inv@data[[x + 'proj' for x in loading[model]]].loc[keep_this_month]).T @ data.xret.loc[keep_this_month]
        #proj_2 = (cov_inv@data['A_2_proj'].loc[keep_this_month]).T @ data.xret.loc[keep_this_month]

        # model results
        model_results = {}
        for method in ["taylor", "proj"]:
            X = data[[x + method for x in loading[model]]]
            factor_weights = ((linalg.pinv(X.T@X))@X.T).T
            port_of_factors = fama.mve_data(model_premia[method], month, 0)

            weights_on_stocks = factor_weights.values @ port_of_factors.values

            stdev = np.sqrt(weights_on_stocks @ stock_cov @ weights_on_stocks)
            mean = weights_on_stocks @ rp
            xret = weights_on_stocks @ data.xret

            errs = rp - second_moment @ weights_on_stocks
            hjd = errs @ second_moment_inv @ errs
            model_results[method] = [stdev, mean, xret, hjd]
            weights = np.zeros((N,))
            weights[keep_this_month] = weights_on_stocks
            dat = {
                'iter': iter,
                'month':month,
                'method': method,
                **{f'firm_{i+1}': val for i, val in enumerate(weights)}
            }
            new_row = pd.DataFrame([dat])
            new_row.to_csv('port_model_' + model + '.csv', mode='a', header=False, index=False)

        model_results = pd.DataFrame(model_results).T
        model_results.columns = ["stdev", "mn", "xret", "hjd"]
        model_results.index.names = ["method"]
        model_results = model_results.reset_index()
        model_results["month"] = month

        def process_dkkm_block(method, include_mkt, dkkm_list, get_weights_fn):
            results = {}
            half = max_features // 2
            
            for i, tpl in enumerate(dkkm_list):
                W, frets = tpl

                for nfeatures in nfeatures_lst:
                    num = int(nfeatures / 2)
                    nf_indx = np.concatenate([np.arange(num), np.arange(half, half + num)])
                    
                    rf = data.rf_stand if model == 'bgn' else None
                    factor_weights = get_weights_fn(data[chars], rf, W[:num, :], model)
                    factor_weights.columns = [str(ind) for ind in nf_indx]

                    if include_mkt:
                        factor_weights['mkt_rf'] = fama.fama_french(data[chars], mve=data.mve)[:, -1]

                    for alpha in alpha_lst:
                        port_of_factors = dkkm.mve_data(
                            frets.iloc[:, nf_indx], month, nfeatures * alpha,
                            ff_rets.iloc[:, -1] if include_mkt else None
                        )

                        weights_on_stocks = factor_weights @ port_of_factors
                        dkkm_ports[(alpha, nfeatures, i, int(include_mkt), method)] = weights_on_stocks

                        weights = np.zeros((N,))
                        weights[keep_this_month] = weights_on_stocks

                        dat = {
                            'iter': iter,
                            'month': month,
                            'include_mkt': int(include_mkt),
                            'method': method,
                            'mat': i,
                            'nfeatures': nfeatures,
                            'alpha': alpha,
                            **{f'firm_{i+1}': val for i, val in enumerate(weights)}
                        }

                        new_row = pd.DataFrame([dat])
                        new_row.to_csv(
                            f'port_dkkm_{model}.csv', mode='a',
                            header=False, index=False
                        )

            # Compute results
            for alpha in alpha_lst:
                for nfeatures in nfeatures_lst:
                    for i in range(nmat):
                        weights_on_stocks = dkkm_ports[(alpha, nfeatures, i, int(include_mkt), method)]

                        stdev = np.sqrt(weights_on_stocks @ stock_cov @ weights_on_stocks)
                        mean = weights_on_stocks @ rp
                        xret = weights_on_stocks @ data.xret

                        errs = rp - second_moment @ weights_on_stocks
                        hjd = errs @ second_moment_inv @ errs

                        results[(alpha, nfeatures, i, int(include_mkt), method)] = [stdev, mean, xret, hjd]

            results_df = pd.DataFrame(results).T
            results_df.columns = ["stdev", "mn", "xret", "hjd"]
            results_df.index.names = ["alpha", "nfeatures", "mat", "include_mkt", "method"]
            results_df = results_df.reset_index()
            results_df["month"] = month
            return results_df

        # Init output
        dkkm_results = None
        dkkm_ports = {
            (a, n, i, k, l): np.empty((len(data),)) 
            for a in alpha_lst 
            for n in nfeatures_lst 
            for i in range(nmat) 
            for k in range(2) 
            for l in ['rs', 'nors']
        }

        # Define accessors
        get_rs_weights = lambda x, rf, W, model: dkkm.rff(x, rf, W=W, model=model)[0]
        get_nors_weights = lambda x, rf, W, model: dkkm.rff(x, rf, W=W, model=model)[1]

        # Run the four scenarios
        dkkm_results = pd.concat([
            process_dkkm_block('rs', True, dkkm_lst_rs, get_rs_weights),
            process_dkkm_block('rs', False, dkkm_lst_rs, get_rs_weights),
            process_dkkm_block('nors', True, dkkm_lst_nors, get_nors_weights),
            process_dkkm_block('nors', False, dkkm_lst_nors, get_nors_weights)
        ])

        # fama results
        fama_results = {}
        for m in ["ff", "fm"]:
            frets = ff_rets if m=="ff" else fm_rets
            method = fama.fama_french if m=="ff" else fama.fama_macbeth
            factor_weights = method(data[chars], mve=data.mve)

            for alpha in alpha_lst_fama:
                port_of_factors = fama.mve_data(frets, month, alpha)
                weights_on_stocks = factor_weights @ port_of_factors
                stdev = np.sqrt(weights_on_stocks @ stock_cov @ weights_on_stocks)
                mean = weights_on_stocks @ rp
                xret = weights_on_stocks @ data.xret

                errs = rp - second_moment @ weights_on_stocks
                hjd = errs @ second_moment_inv @ errs
                fama_results[(alpha, m)] = [stdev, mean, xret, hjd]
                weights = np.zeros((N,))
                weights[keep_this_month] = weights_on_stocks
                    
                dat = {
                    'iter': iter,
                    'month':month,
                    'method': m,
                    'alpha': alpha,
                    **{f'firm_{i+1}': val for i, val in enumerate(weights)}
                }
                new_row = pd.DataFrame([dat])
                new_row.to_csv('port_fama_' + model + '.csv', mode='a', header=False, index=False)

        fama_results = pd.DataFrame(fama_results).T
        fama_results.columns = ["stdev", "mn", "xret", "hjd"]
        fama_results.index.names = ["alpha", "method"]
        fama_results = fama_results.reset_index()
        fama_results["month"] = month
        
        # ipca results
        
        ipca_results = {}
        for m in ipca_nfactors_lst:
            firms = data.xret.index
            factor_weights = ipca_weights_on_stocks[m][:, firms, month - start - 360].T
            port_of_factors = ipca_factor_weights[m][:, month - start - 360]
            weights_on_stocks = factor_weights @ port_of_factors
            stdev = np.sqrt(weights_on_stocks @ stock_cov @ weights_on_stocks)
            mean = weights_on_stocks @ rp
            xret = weights_on_stocks @ data.xret

            errs = rp - second_moment @ weights_on_stocks
            hjd = errs @ second_moment_inv @ errs
            ipca_results[m] = [stdev, mean, xret, hjd]
            weights = np.zeros((N,))
            weights[keep_this_month] = weights_on_stocks
                    
            dat = {
                'iter': iter,
                'month':month,
                'nfactors': m,
                **{f'firm_{i+1}': val for i, val in enumerate(weights)}
            }
            new_row = pd.DataFrame([dat])
            new_row.to_csv('port_ipca_' + model + '.csv', mode='a', header=False, index=False)
            

        ipca_results = pd.DataFrame(ipca_results).T
        ipca_results.columns = ["stdev", "mn", "xret", "hjd"]
        ipca_results.index.names = ["ipca factors"]
        ipca_results = ipca_results.reset_index()
        ipca_results["month"] = month
        ipca_rets = (ipca_weights_on_stocks[2][:, firms, month - start - 360] @ data.xret).reshape((1, 2))
        
        '''
        # ipca rff results
        ipca_rff_results = {}
        ipca_rff_ports = {
                        n: np.empty((len(data), 20)) for n in ipca_nfactors_lst
                    }
        
        for i in range(20):
            for m in ipca_nfactors_lst:
                firms = data.xret.index
                factor_weights = ipca_rff_weights_on_stocks[m][i][:, firms, month - start - 360].T
                port_of_factors = ipca_rff_factor_weights[m][i][:, month - start - 360]
                weights_on_stocks = factor_weights @ port_of_factors
                ipca_rff_ports[m][:, i] = weights_on_stocks

        
        for m in ipca_nfactors_lst:
            weights_on_stocks = ipca_rff_ports[m].mean(axis=1)
            stdev = np.sqrt(weights_on_stocks @ stock_cov @ weights_on_stocks)
            mean = weights_on_stocks @ rp
            xret = weights_on_stocks @ data.xret

            errs = rp - second_moment @ weights_on_stocks
            hjd = errs @ second_moment_inv @ errs
            ipca_rff_results[m] = [stdev, mean, xret, hjd]

        ipca_rff_results = pd.DataFrame(ipca_rff_results).T
        ipca_rff_results.columns = ["stdev", "mn", "xret", "hjd"]
        ipca_rff_results.index.names = ["ipca factors"]
        ipca_rff_results = ipca_rff_results.reset_index()
        ipca_rff_results["month"] = month
        '''

        

        if month % 50 == 0:
            print(f"finished month {month} at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
            
        return (model_results, dkkm_results, fama_results, ipca_results, 
                month, sdf_ret, max_sr, proj, #, #data.f_mu_true.mean(), data.f_xi_true.mean(), 
                ipca_rets)
            
    results_lst = Parallel(n_jobs=n_jobs, verbose=0)(
      delayed(run_month)(month, iter) for month in range(start+360, end+1)
    )        
    #results_lst = [run_month(month, iter) for month in range(start+360, end + 1)]
    
    model_results = pd.concat([x[0] for x in results_lst])
    dkkm_results = pd.concat([x[1] for x in results_lst])
    fama_results = pd.concat([x[2] for x in results_lst])
    ipca_results = pd.concat([x[3] for x in results_lst])
    #ipca_rff_results = pd.concat([x[3] for x in results_lst])
    tseries = pd.DataFrame(
        {
            "month": [x[4] for x in results_lst],
            "sdf_ret": [x[5] for x in results_lst],
            "max_sr": [x[6] for x in results_lst] #"mu_proj": [x[7] for x in results_lst],# "xi_proj": [x[8] for x in results_lst] #"f_mu_true": [x[7] for x in results_lst],"f_xi_true": [x[8] for x in results_lst]
        }
    )
    ipca_rets = pd.concat([pd.DataFrame(x[-1]) for x in results_lst])

    return panel, model_results, dkkm_results, fama_results, tseries, model_premia, fm_rets, ff_rets, ipca_results , ipca_rets, sorted_ports

# now run it for multiple iterations/panels
panel_out = None
model_out = None
dkkm_out = None
fama_out = None
ipca_out = None
#ipca_rff_out = None
tseries_out = None
for iter in range(startiter, startiter + numiters):

    print(f"started iter {iter} at {datetime.now().strftime('%a %d %b %Y, %I:%M%p')}")
    
    panel, model_results, dkkm_results, fama_results, tseries, model_premia, fm_rets, ff_rets, ipca_results, ipca_rets, sorted_ports = run_panel(iter) #

    panel["iter"] = iter 
    panel_out = pd.concat((panel_out, panel.reset_index()))
    panel_out.to_csv(f"panel_{model}.csv", index=False)

    model_results["iter"] = iter 
    model_out = pd.concat((model_out, model_results))
    model_out.to_csv(f"results_model_{model}.csv", index=False)

    dkkm_results["iter"] = iter 
    dkkm_out = pd.concat((dkkm_out, dkkm_results))
    dkkm_out.to_csv(f"results_dkkm_{model}.csv", index=False)

    fama_results["iter"] = iter
    fama_out = pd.concat((fama_out, fama_results))
    fama_out.to_csv(f"results_fama_{model}.csv", index=False)

    ipca_results["iter"] = iter
    ipca_out = pd.concat((ipca_out, ipca_results))
    ipca_out.to_csv(f"results_ipca_{model}.csv", index=False)
 
    #ipca_rff_results["iter"] = iter
    #ipca_rff_out = pd.concat((ipca_rff_out, ipca_rff_results))
    #ipca_rff_out.to_csv("results_ipca_rff.csv", index=False)
    
    tseries["iter"] = iter
    fm_rets.columns = ['fm_' + st for st in ['size', 'bm', 'agr', 'roe', 'mom', 'mkt_rf']] 
    ff_rets.columns = ['ff_' + st for st in ['hml', 'smb', 'cma', 'rmw', 'umd', 'mkt_rf']]
    ipca_rets.columns = ['ipca' + str(ipca_rets.columns[i]) for i in range(ipca_rets.shape[1])] 

    fm_rets = fm_rets.iloc[360:,:]
    ff_rets = ff_rets.iloc[360:,:]

    fm_rets.index = tseries.index
    ff_rets.index = tseries.index
    ipca_rets.index = tseries.index

    for m in ["taylor", "proj"]:
        model_premia[m] = model_premia[m][model_premia[m].index >= 360 + burnin]
        model_premia[m].index = tseries.index

    tseries = pd.concat((tseries, fm_rets, ff_rets, ipca_rets, model_premia["taylor"], model_premia["proj"]), axis = 1)
    tseries_out = pd.concat((tseries_out, tseries))
    tseries_out.to_csv(f"results_tseries_{model}.csv", index=False)

    # Save sorted portfolio results
    sorts.save_portfolio_results(sorted_ports, model, iter)