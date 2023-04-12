import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def create_heatmap(mean_conf, variable, tvals_table, pvals_table, pal, order):
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(tvals_table, dtype=bool))

    # Set up the matplotlib figure
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25, len(mean_conf[variable].unique())*1.2))
    sns.barplot(y=variable, x='mean', data=mean_conf, ax=axs[0], palette=pal, order=order)
    axs[0].set_ylabel('Score')
    axs[0].set_title('Mean Scores with Confidence Intervals')

    # Generate the heat map for t-values
    sns.heatmap(tvals_table, mask=mask, annot=True, fmt=".2f", cmap="magma",
                cbar_kws={'label': 't-value'}, ax=axs[1])
    axs[1].set_title('Contrasts - t-values')

    # Generate the heat map for p-values
    pval_stars = pvals_table.applymap(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')
    norm = plt.Normalize(vmin=np.nanmin(pvals_table.values), vmax=np.nanmax(pvals_table.values))
    pal_p = sns.color_palette("magma_r", as_cmap=True)
    sns.heatmap(pd.DataFrame(norm(pvals_table.values)), mask=mask, annot=pval_stars, fmt="", cmap=pal_p,
                cbar_kws={'label': 'p-value'}, ax=axs[2],norm=norm)
    axs[2].set_title('Contrasts - p-values')

    plt.tight_layout()
    plt.show()


def analyze_results(results_df, variable, order):
    variables = order
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