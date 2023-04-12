import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def create_heatmap(mean_conf, variable, tvals_table, pvals_table, pal, order, color_order=None, xlim=(60, 90), save=None, large=10):
    if color_order is not None:
        pal = color_order
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(tvals_table, dtype=bool))

    # Set up the matplotlib figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(large, len(mean_conf[variable].unique())*1.2))
    
    if color_order is not None:
        pal = dict(zip(order, color_order))

    bar_plot = sns.barplot(x=variable, y='mean', data=mean_conf, ax=axs[0], palette=pal, order=order, ci=None)
    
    # Add error bars to the barplot
    for i, model in enumerate(order):
        model_data = mean_conf[mean_conf[variable] == model]
        error = model_data['std'].values[0]
        axs[0].errorbar(x=i, y=model_data['mean'].values[0], yerr=error, fmt='none', color='black', capsize=5)
        
    axs[0].set_xlabel('Score')
    #axs[0].set_xticklabels(order, rotation=20)
    axs[0].set_title('Mean Scores with Standard Deviations')

    # Set y-axis ticks manually
    yticks = np.linspace(xlim[0], xlim[1], num=6)  # Change num to the desired number of ticks
    bar_plot.set_yticks(yticks)
    bar_plot.set_ylim(xlim)

    # Generate the combined heat map for t-values and p-values
    tvals_annot = tvals_table.applymap(lambda x: f"\n{x:.2f}")
    pval_stars = pvals_table.applymap(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')
    combined_annot = pval_stars + tvals_annot
    max_tval = np.abs(tvals_table.values[~mask]).max()
    sns.heatmap(tvals_table, mask=mask, annot=combined_annot, fmt="", cmap="coolwarm",
            cbar_kws={'label': 't-value'}, ax=axs[1], vmin=-max_tval, vmax=max_tval)

    axs[1].set_title('Contrasts - t-values and p-values')
    #axs[1].set_xticklabels(order, rotation=20)

    plt.tight_layout()
    if save is not None:
        plt.savefig('{}.png'.format(save), dpi=300)
    plt.show()




def analyze_results(results_df, variable, order):
    variables = order
    n_variables = len(results_df[variable].unique())
    # Calculate mean and confidence interval for each model's distribution
    mean_conf = results_df.groupby(variable)['Score'].agg(['mean', 'sem', 'std'])
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