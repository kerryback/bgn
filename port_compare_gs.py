import numpy as np
import pandas as pd
import re
import datetime
import statsmodels.api as sm

### This file modifies port_compare.py to include gs, 
# and to include specifications with/wo market (include_mkt) and with/wo rank standardizations (method)

## currently does not output PCA results (across DKKM mats)
print('port_compare_gs.py started at %s' % datetime.datetime.now())

####
# basic specifications

# econ models considered
econ_models = ['bgn','kp', 'gs'] # HACK HACK HACK lop off the 1 for the results files below

# where the simulation data files live
results_dir = 'new_results/' # if dir, end with "/"

# iter_month
iter_month = True 
# if True, have columns "iter" and "month" and redefine "month" to be the iter_month str
# if False, only "month" is there

## excel writing
write_summary_excel = True # if True, write Excel with multiple sheets

excel_name = 'port_compare_plus_gs.xlsx' # name of the Excel file (perhaps correspond with results_dir)

excel_append = False 
# if False, create new Excel
# if True, append to existing
#   if you choose True, make sure excel_name already exists

#### Data loading
#
## CHECK THE HARDCODING OF SPECIFICATIONS
# import port weights for all econ models and all estimation methods
# import results for all econ models and all estimation methods
# specify the specification columns for each estimation method in a dict
# simulated firms go across columns
#  those columns are named feature_x, firm_x, or firms_x in various -- so change to uniform naming
#

port = {} # preallocate
results = {} # preallocate
tseries = {} # preallocate
spec_cols = {}
for em in econ_models:
    # port weight files
    port[em] = {}
    port[em]['true'] = pd.read_csv(results_dir + 'port_' + em + '.csv') # true from model simulation
    port[em]['model'] = pd.read_csv(results_dir + 'port_model_' + em + '.csv') # taylor and proj based
    port[em]['fama'] = pd.read_csv(results_dir + 'port_fama_' + em + '.csv') # ff and fm
    port[em]['ipca'] = pd.read_csv(results_dir + 'port_ipca_' + em + '.csv') # ipca with K=1, 2
    port[em]['dkkm'] = pd.read_csv(results_dir + 'port_dkkm_' + em + '.csv') # dkkm with various features, alpha, 20 RFF panels
    print('Read in port data for econ model %s at %s' % (em,datetime.datetime.now()))
    # dkkm ports column name change
    port[em]['dkkm'].rename({'include_market':'include_mkt'}, axis=1, inplace=True)
    port[em]['dkkm'].rename({'nfeature':'nfeatures'}, axis=1, inplace=True)
    
    # specifications for each estimator
    spec_cols[em] = {}
    spec_cols[em]['true'] = [] # empty list
    spec_cols[em]['model'] = ['method']
    spec_cols[em]['fama'] = ['method','alpha']
    spec_cols[em]['ipca'] = ['nfactors']
    spec_cols[em]['dkkm'] = ['mat','nfeatures','alpha', 'method', 'include_mkt']
    
    # results files -- HACK HACK HACK lop off the number in em (last string char)
    results[em] = {}
    results[em]['model'] = pd.read_csv(results_dir + 'results_model_' + em + '.csv') # taylor and proj based
    results[em]['fama'] = pd.read_csv(results_dir + 'results_fama_' + em + '.csv') # ff and fm
    results[em]['ipca'] = pd.read_csv(results_dir + 'results_ipca_' + em + '.csv') # ipca with K=1,2
    results[em]['dkkm'] = pd.read_csv(results_dir + 'results_dkkm_' + em + '.csv') # dkkm with various features, alpha, 20 RFF panels
    # ipca results column name change
    results[em]['ipca'].rename({'ipca factors':'nfactors'}, axis=1, inplace=True)
    
    # tseries file -- HACK HACK HACK lop off the number in em (last string char)
    tseries[em] = pd.read_csv(results_dir + 'results_tseries_' + em + '.csv') # tseries combines
    
    print('Read in results data for econ model %s at %s' % (em,datetime.datetime.now()))


# change columns in port with firm weights to uniform naming: firm_x.
# also check iter_month and adjust to string
# start with iter_month_fun: a helper function 
def iter_month_fun(df, number_to_string=True):
    # takes two numeric columns named 'iter' and 'month' 
    # TO a string column named 'month' of iter_month strings (if number_to_string=True; default)
    # OR does the reverse (if number_to_string=False)

    if number_to_string:
        df['itermonth'] = df['iter'].astype(str) + '_' + df['month'].astype(str)
        df.drop(['iter','month'], axis=1, inplace=True)
        df.rename(columns={'itermonth': 'month'}, inplace=True)
    else:
        df[['iter', 'monthtmp']] = df['month'].str.split("_", expand=True).astype(int)
        df = df.drop(['month'], axis=1)
        df.rename(columns={'monthtmp':'month'}, inplace=True)
    return df

for em in econ_models:
    for k in port[em].keys():
        # make firm names all firm_x
        col = list(port[em][k].columns)
        for j,c in enumerate(col):
            # only do this if one of the following
            if any(k for k in ['feature_', 'features_', 'firm_', 'firms_'] if k in c):
                csplit = re.split('(\d+)',c)
                # csplti is single element list if c has no number (should not be the case if we got here)
                # if c has a number, csplit has several elements: 2nd should be the number
                if len(csplit)>1:
                    col[j] = 'firm_' + csplit[1]
        # complete the replacement of firm names
        port[em][k].columns = col
        
        # check iter_month: if True, make month a string of iter_month
        if iter_month: # true means replace "month" with iter_month and drop iter
            port[em][k] = iter_month_fun(port[em][k])
            
        # generate the columns to keep: iter, month, spec_cols, and firm_X (for number X)
        keepcols = ['month'] + spec_cols[em][k] + [c for c in port[em][k].columns if ('firm_' in c)]
        # drop other columns
        port[em][k].drop([c for c in port[em][k].columns if not (c in keepcols)], axis=1, inplace=True)

# check iter_month for results and replace month with string iter_month
if iter_month:
    for em in econ_models:
        for k in results[em].keys():
            results[em][k] = results[em][k] = iter_month_fun(results[em][k])

# check iter_month for tseries and replace month with string iter_month
if iter_month:
    for em in econ_models:
        tseries[em] = iter_month_fun(tseries[em])

#print(' done with data manipulations at %s' % (em,datetime.datetime.now()))





#####
# calculate correlation of weights, with true

corr_weights = {} # preallocate dict for results
nfc_dict = {} # preallocate dict for not_firm_cols lists

for em in econ_models:
    corr_weights[em] = {} # preallocate sub dict for results
    nfc_dict[em] = {} # preallocate sub dict for not_firm_cols list
    
    # set up a true weight df here, which is indexed by the unique months
    # these are the SDF weights, so negative of the efficient portfolio
    # for that reason, we multiply them by -1
    true_weights = (-1)*port[em]['true'].set_index('month').copy()
    
    # do for every estimate of weights
    for k in [k for k in port[em].keys() if not (k=='true')]: # the estimated weights; not the true weights
        # find column names that are not firm weights (e.g. not 'firm_1')
        not_firm_cols = [
            c for c in port[em][k].columns if (len(re.split('(\d+)',c))==1)
            ]
        firm_cols = [c for c in port[em][k].columns if (c not in not_firm_cols)] # should be always the same
        # put not_firm_cols list into dict -- this will be used by corr_weights calculation too
        nfc_dict[em][k] = [nfc for nfc in not_firm_cols if not (nfc=='month')]
        
        # concat the true weights at the end: the non_firm_cols for its rows
        # should be nan or empty (depending on dtype of non_firm_cols for this (em,k)
        tmp = pd.concat( (port[em][k].set_index(['month']), true_weights) )
        # groupby preserves the order in each group; so when we groupby below,
        # the correlation with truth should be the last row of the df produced
        # by corr()
        
        # this helper file does the corr, and can be groupby().apply() -ed
        def groupbycorr(gdf):
            nfc = [c for c in not_firm_cols if not (c=='month')]
            return pd.Series(index=pd.MultiIndex.from_frame(gdf[nfc].iloc[:-1,:]), 
                             data=gdf[firm_cols].T.corr().iloc[-1,:-1].values,
                             name='corr')
        corr_weights[em][k] = tmp.groupby('month').apply(groupbycorr)  
        
        print('corr_weights done with econ model %s and estimate %s at %s' % (em,k,datetime.datetime.now()))

# each df in corr_weights[em][k] now has months as the index,
# specification as column multiindex,
# and the CS corr as the value




####
# construct performance measures for every specification

perform_measures = {} # preallocate dict

for em in econ_models:
    perform_measures[em] = {} # preallocate sub dict
    
    for k in [k for k in results[em].keys()]:
    
        # merge in the tseries sdf_ret
        tmp = results[em][k].merge(tseries[em][['sdf_ret', 'max_sr','month']], 
                                   left_on=['month'], right_on=['month'], 
                                   how='left').copy()
        
        # index the df: use the nfc_dict just created to identify the columns 
        # where estimation specifications are given
        tmp = tmp.set_index(nfc_dict[em][k]).sort_index()
        
        # create performance measure df
        perform_measures[em][k] = pd.DataFrame(
            index = tmp.index.drop_duplicates(),
            columns = ['Realized HJD', 'Mean Conditional HJD', 
                       'Realized SR', 'Mean Conditional SR',
                       'Model Mean Conditional SR', 'Model Realized SR'],
            dtype = float
            )
        # Mean Conditional HJD
        perform_measures[em][k].loc[:, 'Mean Conditional HJD'] = \
            tmp['hjd'].groupby(tmp.index.names).mean()
        # Realized HJD
        perform_measures[em][k].loc[:, 'Realized HJD'] = \
            ((tmp['xret']-tmp['sdf_ret'])**2)\
            .groupby(tmp.index.names).mean()
        # Realized SR
        def srcalc(srs): # SR calculation function to apply in the groupby
            return srs.mean()/srs.std(ddof=0)
        perform_measures[em][k].loc[:, 'Realized SR'] = \
            tmp['xret'].groupby(tmp.index.names).apply(srcalc)
        # Mean Conditional SR
        perform_measures[em][k].loc[:, 'Mean Conditional SR'] = \
            (tmp['mn']/tmp['stdev']).groupby(
                tmp.index.names).mean()
        # Model Mean Conditional SR -- identical for every estimation spec, so just do once
        perform_measures[em][k].loc[:, 'Model Mean Conditional SR'] = \
            tmp.loc[perform_measures[em][k].index[0],'max_sr'].mean()
        # Model Realized SR -- identical again
        perform_measures[em][k].loc[:, 'Model Realized SR'] = \
            tmp.loc[perform_measures[em][k].index[0],'sdf_ret'].mean()\
                /tmp.loc[perform_measures[em][k].index[0],'sdf_ret'].std()
        # SDF regression
        def sdfreg(dfin): # regression dict calculation function to apply in the groupby
            r = sm.OLS(dfin['sdf_ret'], exog=dfin['xret'], missing='drop').fit()
            # could have returned dict with more info, but are not for now
            # return {'params': r.params.copy(), 'tvalues': r.tvalues.copy(),
            #         'rsquared': r.rsquared}
            return r.rsquared
        perform_measures[em][k] = perform_measures[em][k].join(
            (tmp[['sdf_ret','xret']].groupby(
                tmp.index.names).apply(sdfreg)).rename('SDF R2'),
            how='left')
        print('For the performance measures, done with econ model %s and est method %s at %s'
              % (em,k,datetime.datetime.now()))
    
    # average across mat for dkkm
    # call it 'dkkm avgmat'
    # there may only be one mat 
    perform_measures[em]['dkkm avgmat'] =\
        perform_measures[em]['dkkm'].groupby(['nfeatures','alpha', 'include_mkt', 'method']).mean()




####
# calculate summary stats for these cross-sectional correlations

dflist = [] # allocate empty list for dfs -- will pd.concat the list

## every specification
for em in econ_models:
    for k in corr_weights[em].keys():
        tmp = corr_weights[em][k].describe().T
        tmp.index = pd.MultiIndex.from_product(
            [ 
                [em],
                [k],
                [tuple(tmp.index.names)],
                tmp.index.to_flat_index()
                ],
            names = ['econ model', 'estimation method', 'spec names', 'specification']
            )
        dflist = dflist + [tmp]
    print("done with cs corr for econ model %s and estimation %s at %s" %
          (em,k,datetime.datetime.now()))

# sum_stats_cscorr puts all the specifications together        
sum_stats_cscorr = pd.concat(dflist)


## For dkkm, we also want to consider summary stats of TS of CS corrs across mat
# for the same (nfeatures,alpha,include_mkt,method) tuples.
# We select the (nfeatures,alpha,include_mkt,method) tuples as those that do "best" by different measures
# To select "best", we average over the mat in perform_measure[em]['dkkm']

# dkkm best
# we measure best various ways, but the process is the same, so we loop through those measures
dflist = []
topchoice = 5 # how many we will take as top
for em in econ_models:
    
    # df of port wts indexed by nfeatures,alpha,month
    # modified from port_compare.py to include ['include_mkt', 'method'] columns
    tmp = port[em]['dkkm'].drop('mat',axis=1).\
        set_index(['nfeatures','alpha', 'include_mkt', 'method']).\
            reset_index().set_index(['nfeatures','alpha', 'include_mkt', 'method','month']).sort_index().copy()
            
    # This looks for PC explained variance of portfolio wts, for each month, 
    # across different mats for the same (nfeatures,alpha)
    # These are not correlations with the true weights: they are correlations
    # across random RFF seeds for the same (nfeatures,alpha) specification
    # We do this for each month, then describe() across months
    
    # Because the number of unique (nfeatures,alpha) pairs is small, we do it for all
    # Then in the "for bestway in..." loop we select just the best rows (by various measures)
        
    # PC explained variance function
    # we use this to summarize multiple correlations with a single number
    def pcexpl(dfin, pcnum=1):
        _,s,_ = np.linalg.svd(dfin/(np.sqrt((dfin**2).sum(axis=1)).to_frame().values))
        return ((s[:pcnum]**2)/np.sum(s**2))[0]
    # groupby for each month

    ### COMMENTED OUT PCA - not meaningful with only one mat

    #tmp = tmp.groupby(tmp.index.names).apply(pcexpl)
    print("groupby for pc done for %s at %s" % (em,datetime.datetime.now()))
    
    # now describe these over the months
    # a describe function
    def descfcn(dfin):
        return dfin.describe() 
    
    # COMMENTED OUT DESCRIBE
    
    #tmp = (tmp.groupby(['nfeatures','alpha', 'include_mkt', 'method']).apply(descfcn)).unstack()
    # the unstack makes the describe() index go across the columns

    ## the following produces tmp2 for each bestway
    # we select the best rows of tmp2 using best_here
    
    # estimation method will be 'dkkm' + bestway
    for bestway in ['Realized HJD', 'Realized SR', 'SDF R2', 'Mean Conditional SR']:
        if bestway=='Realized HJD': # these are the top in terms of lowest Realized HJD
            sortchoice = True # ascending is True, descending is False
        elif bestway=='Realized SR': # these are the top in terms of highest Realized SR
            sortchoice = False
        elif bestway=='SDF R2': # these are the top in terms of highest R2
            sortchoice = False
        elif bestway=='Mean Conditional SR': # these are the top in terms of highest mean conditional SR
            sortchoice = False
            
        best_here = ((perform_measures[em]['dkkm'][bestway].\
            groupby(['nfeatures','alpha', 'include_mkt', 'method']).mean()).\
                 sort_values(ascending=sortchoice)).head(topchoice)
        # best_here should have non-duplicate index values
        
        

        # round the best_here measure to 6 decimals so the index is nicer to read
        # COMMENTED OUT JOIN TO TMP
        #tmp2 = (best_here.round(6).to_frame().join(tmp, how='left')).\
        #        reset_index().set_index(['nfeatures','alpha',bestway]).copy()
        
        tmp2 = (best_here.round(6).to_frame()).\
                reset_index().set_index(['nfeatures','alpha','include_mkt', 'method', bestway]).copy()
                # how='left' restricts it to the best_here
        tmp2.index = pd.MultiIndex.from_product(
            [
                [em],
                ['dkkm ' + bestway + ' across mat'],
                [tuple(tmp2.index.names)],
                tmp2.index.to_flat_index()
                ],
            names = ['econ model', 'estimation method', 'spec names', 'specification']
            )
        dflist = dflist + [tmp2]  
        
        print(" cross est cs corr done for %s and bestway %s at %s" %
              (em,bestway,datetime.datetime.now()))
        
        
    ## the following produces tmp3
    # We find the correlation of each mat's port wts with the truth, for
    # each (nfeatures,alpha,include_mkt,method) tuple. Then we average them for the month
    # Then we describe() across the months
    
    # There are few enough specifications to do it for all
    # The estimation method will be 'dkkm avgmat'
    
    # This can be done by averaging the correct rows of corr_weights[em]['dkkm']
    tmp3 = (
        corr_weights[em]['dkkm'].T.groupby(['nfeatures','alpha', 'include_mkt', 'method']).mean())\
        .T.describe().T
    tmp3.index = pd.MultiIndex.from_product(
        [
            [em],
            ['dkkm avgmat'],
            [tuple(tmp3.index.names)],
            tmp3.index.to_flat_index()
            ],
        names = ['econ model', 'estimation method', 'spec names', 'specification']
        )
    dflist = dflist + [tmp3]
        
# concat with sum_stats_cscorr
sum_stats_cscorr = pd.concat([sum_stats_cscorr, pd.concat(dflist)])




#### 
# Put together spreadsheets of perform measures
dflist = []
for em in econ_models:
    # Find the (nfeatures,alpha,include_mkt,method) combos that appear in any best set
    y = list(set(
        [x for x in sum_stats_cscorr.loc[em].index.get_level_values('estimation method') 
         if ('across mat' in x)]
        )) # those 'estimation method' level index values with 'across mat' (found as best)
    any_best = list(set(
        [x[:4] for x in sum_stats_cscorr.sort_index().loc[
            pd.IndexSlice[em,y,:,:]].index.get_level_values('specification')]
        )) # just the (nfeatures,alpha,include_mkt,method) tuples appearing in any best
    
    for k in [x for x in perform_measures[em].keys() if not (x=='dkkm')]: # all but dkkm (this should INCLUDE dkkm avgmat)
        tmp = perform_measures[em][k].copy()
        if k=='dkkm avgmat':
            tmp = tmp.loc[sorted(any_best)]
        tmp.index = pd.MultiIndex.from_product(
            [
                [em],
                [k],
                [tuple(tmp.index.names)],
                tmp.index.to_flat_index()
                ],
            names = ['econ model', 'estimation method', 'spec names', 'specification']
            )
        dflist = dflist + [tmp.copy()]
        print(" done putting together perform measures for econ model %s and method %s at %s" %
              (em,k,datetime.datetime.now()))
sum_perform_measures = pd.concat(dflist)

## open ExcelWriter and fill sheets
if write_summary_excel:
    print("Choose to write to excel: starting at %s" % datetime.datetime.now())
    exmode = 'w' # default is write
    if excel_append:
        exmode = 'a'
    
    with pd.ExcelWriter(excel_name, mode=exmode) as writer:
        for em in econ_models:
            sum_perform_measures.loc[em].to_excel(writer, sheet_name=(em + ' perform'))
            sum_stats_cscorr.loc[em].drop('dkkm',level='estimation method').\
                to_excel(writer, sheet_name=(em + ' port corr'))
        

print("THE SPREADSHEET %s HAS BEEN COMPLETED at %s" % (excel_name, datetime.datetime.now()))
print(" ")
print("port_compare_gs.py is still running to calculate persistence of the weights")
            


#### 
# calculate portfolio weight persistence
#
# not written into Excel (yet?)
# calculate AR1 parameters and innovation variances, on average across firms
# (for dkkm, across mat and firm)

# helper function finds describe stats of AR1 across firms/simulations
def ar1ls(srs):
    r = sm.OLS(srs.shift(0), exog=sm.add_constant(srs.shift(1)), missing='drop').fit()
    return pd.Series(index=['rho', 'sigmae'], 
                     data=(r.params.iloc[1], # ar1 param
                           r.resid.std()) # resid var
                     )
def ardesc(df, level0name, addl_unstack=None):
    tmp = df.copy()
    if iter_month: # use iter_month from outside
        tmp = iter_month_fun(tmp.reset_index(), number_to_string=False)
        if addl_unstack is None:
            tmp.set_index(['iter','month'], inplace=True)
            tmp = tmp.unstack(level='iter')
        else:
            tmp.set_index(['iter', addl_unstack, 'month'], inplace=True)
            tmp = tmp.unstack(level=['iter', addl_unstack])
    tmp = (tmp.apply(ar1ls, axis=0)).T.describe()
    tmp.columns = pd.MultiIndex.from_product((
        [level0name], tmp.columns
        ), names=['empirical model','param'])
    return tmp.T

# concatenate
print('Starting AR1 analysis at %s' % (datetime.datetime.now()))
dflist = []
for em in econ_models:
    # true
    tmp = ardesc(port[em]['true'], level0name=('true')).reset_index()
    tmp['economic model'] = em
    tmp['specification'] = ('')
    tmp.set_index(['economic model', 'empirical model', 'specification', 'param'])
    dflist = dflist + [tmp]
    
    # fama, ipca, model
    # we groupby the specifications here
    # then the dataframe entering ardesc() is just of the firm wts for the spec
    def dflisthelp(k):
        tmp = port[em][k].groupby(spec_cols[em][k], group_keys=True).apply(
            lambda d: ardesc(d, k), include_groups=False
            ).reset_index()
        tmp['economic model'] = em
        tmp['specification'] = list(zip(*(tmp[c] for c in spec_cols[em][k])))
        tmp.drop(spec_cols[em][k], axis=1, inplace=True)
        tmp = tmp.set_index(['economic model', 'empirical model', 'specification', 'param'])
        return tmp
    
    for k in ['fama', 'ipca', 'model']:
        dflist = dflist + [dflisthelp(k)]
    print('Done with econ model %s and emp models fama, ipca, model at %s' % (em,datetime.datetime.now()))
    print(' starting dkkm which will take longer')
    
    # dkkm
    # we groupby the specifications EXCEPT 'mat'
    # for other purposes, mat is a useful specification indicator
    # but here, we want to add it as a column -- as though we have #firms x #mat AR1 processes to estimate
    # and describe.
    # So we drop 'mat' from the groupby, but then add it to the unstack (via ardesc() input addl_unstack )
    sc = [s for s in spec_cols[em]['dkkm'] if not (s == 'mat')]
    tmp = port[em]['dkkm'].groupby(sc, group_keys=True).apply(
        lambda d: ardesc(d, 'dkkm', addl_unstack='mat'), include_groups=False
        ).reset_index()
    tmp['economic model'] = em
    tmp['specification'] = list(zip(*(tmp[c] for c in sc)))
    tmp.drop(sc, axis=1, inplace=True)
    tmp = tmp.set_index(['economic model', 'empirical model', 'specification', 'param'])
    dflist = dflist + [tmp]
    
    print(' finished dkkm at %s' % datetime.datetime.now())
    
AR1_descriptions = pd.concat(dflist)

# HACK HACK HACK problem with indexing (dunno why)
for j in range(AR1_descriptions.shape[0]):
    if AR1_descriptions[['economic model', 'empirical model', 'specification', 'param']].iloc[j,:].notna().any():
        AR1_descriptions.index.values[j] = list(zip(
            AR1_descriptions[['economic model', 'empirical model', 'specification', 'param']].iloc[j,:]
            ))
AR1_descriptions.drop(['economic model', 'empirical model', 'specification', 'param'], axis=1, inplace=True)


print('Done with AR1 analysis at %s' % (datetime.datetime.now()))

if write_summary_excel:
    print("Choose to write to excel: starting at %s" % datetime.datetime.now())
    exmode = 'w' # default is write
    if excel_append:
        exmode = 'a'
    
    with pd.ExcelWriter('persistence_plus_gs.xlsx', mode=exmode) as writer:
        AR1_descriptions.to_excel(writer)












# # full print helper function -- don't know where to put it, so put it here
# def print_full(x):
#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', 2000)
#     pd.set_option('display.float_format', '{:20,.3f}'.format)
#     pd.set_option('display.max_colwidth', None)
#     print(x)
#     pd.reset_option('display.max_rows')
#     pd.reset_option('display.max_columns')
#     pd.reset_option('display.width')
#     pd.reset_option('display.float_format')
#     pd.reset_option('display.max_colwidth')
    





# # # scratch print code (can be commented out)
# # idxs = pd.IndexSlice # shorthand
# # for em in ['bgn','kp']:
# #     print('******* %s ******' % em)
# #     print(' ** Top 10')
# #     print_full(
# #         sum_stats_cscorr.loc[em,:].sort_values('mean',ascending=False)[['mean','50%','std']].head(10),
# #         )
# #     print(' ** others of interest')
# #     print_full(
# #         sum_stats_cscorr.loc[idxs[em,'model',:,:],:][['mean','50%','std']]
# #         )
# #     print_full(
# #         sum_stats_cscorr.loc[idxs[em,'ipca',:,:],:][['mean','50%','std']]
# #         )
# #     print_full(
# #         sum_stats_cscorr.loc[idxs[em,'fama',:,:],:][['mean','50%','std']]
# #         )



# ## ChatGPT written autosize and write function
# def autosize_and_export_to_excel(df, path, sheet_name="Sheet1", exmode='w',
#                                  wrap_threshold=45, max_width=65, extra_padding=2):
#     # Compute approximate max text length per column (including header)
#     df_for_len = df.copy()
#     lens = {}
#     for col in df_for_len.columns:
#         s = df_for_len[col].astype(str).replace({"NaT":"","NaN":"","nan":""})
#         max_len = max(len(str(col)), (s.map(len).max() if not s.empty else 0))
#         lens[col] = max_len

#     with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
#         df.to_excel(writer, sheet_name=sheet_name, index=False)
#         wb = writer.book
#         ws = writer.sheets[sheet_name]

#         fmt_wrap = wb.add_format({"text_wrap": True})
#         fmt_num = wb.add_format({"num_format": "0.00"})
#         fmt_int = wb.add_format({"num_format": "0"})
#         fmt_date = wb.add_format({"num_format": "yyyy-mm-dd"})
#         fmt_shrink = wb.add_format({"shrink": True})

#         for idx, col in enumerate(df.columns):
#             width = lens[col] + extra_padding
#             ser = df[col]
#             col_fmt = None
#             if pd.api.types.is_datetime64_any_dtype(ser):
#                 col_fmt = fmt_date
#                 width = max(width, 12)
#             elif pd.api.types.is_float_dtype(ser):
#                 col_fmt = fmt_num
#                 width = max(width, 10)
#             elif pd.api.types.is_integer_dtype(ser):
#                 col_fmt = fmt_int
#                 width = max(width, 8)

#             if width > wrap_threshold:
#                 ws.set_column(idx, idx, min(max_width, wrap_threshold), fmt_wrap)
#             else:
#                 ws.set_column(idx, idx, min(max_width, width), col_fmt or fmt_shrink)

#         ws.autofilter(0, 0, len(df), len(df.columns) - 1)
#         ws.freeze_panes(1, 0)
#         ws.fit_to_pages(1, 0)  # print: fit to one page wide


# what went in "if write_summary_excel"
    # for em in econ_models:
    #     # performance sheet
    #     autosize_and_export_to_excel(
    #         sum_perform_measures.loc[em],
    #         excel_name,
    #         sheet_name=(em + ' perform'),
    #         exmode=exmode
    #         )
    #     # port corr sheet
    #     autosize_and_export_to_excel(
    #         sum_stats_cscorr.loc[em].drop('dkkm',level='estimation method'), 
    #         excel_name,
    #         sheet_name=(em + ' port corr'),
    #         exmode=exmode
    #         )