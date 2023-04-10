import pandas as pd
import numpy as np

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def analyze_results(results_df, variable):
    variables = results_df[variable].unique()
    n_variables = len(results_df[variable].unique())
    # Calculate mean and confidence interval for each model's distribution
    mean_conf = results_df.groupby(variable)['Score'].agg(['mean', 'sem'])
    mean_conf['ci_low'] = mean_conf['mean'] - 1.96 * mean_conf['sem']
    mean_conf['ci_high'] = mean_conf['mean'] + 1.96 * mean_conf['sem']
    mean_conf.reset_index(inplace=True)

    # Contrast all models using nonparametric t-tests
    pvals = np.zeros((n_variables, n_variables))
    tvals = np.zeros((n_variables, n_variables))
    for i, model1 in enumerate(variables):
        for j, model2 in enumerate(variables):
            if i >= j:
                continue
            res = ttest_ind(results_df[results_df[variable] == model1]['Score'].dropna(), 
                            results_df[results_df[variable] == model2]['Score'].dropna(), 
                            alternative='two-sided')
            pvals[i, j] = res.pvalue
            pvals[j, i] = res.pvalue
            tvals[i, j] = res.statistic
            tvals[j, i] = res.statistic

    # Correct for multiple comparisons
    reject, pvals_corrected, _,_ = multipletests(pvals.flatten(), 
                                                 alpha=0.05, 
                                                 method='fdr_bh', 
                                                 is_sorted=False, 
                                                 returnsorted=False)

    pvals_table = pd.DataFrame(index=variables, columns=variables, dtype=float)
    tvals_table = pd.DataFrame(index=variables, columns=variables, dtype=float)
    for i, model1 in enumerate(variables):
        for j, model2 in enumerate(variables):
            if i >= j:
                continue
            pvals_table.loc[model1, model2] = pvals_corrected[i*n_variables + j]
            pvals_table.loc[model2, model1] = pvals_corrected[i*n_variables + j]
            tvals_table.loc[model1, model2] = tvals[i, j]
            tvals_table.loc[model2, model1] = tvals[j, i]

    return mean_conf, pvals_table, tvals_table