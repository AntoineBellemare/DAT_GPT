import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from pingouin import compute_effsize_from_t


def analyze_results(results_df, variable, order):
    variables = order
    n_variables = len(results_df[variable].unique())
    # Calculate mean and confidence interval for each model's distribution
    mean_conf = results_df.groupby(variable)["Score"].agg(["mean", "median", "sem", "std"])
    mean_conf["ci_low"] = mean_conf["mean"] - 1.96 * mean_conf["sem"]
    mean_conf["ci_high"] = mean_conf["mean"] + 1.96 * mean_conf["sem"]
    mean_conf["1std"] = mean_conf["mean"] + mean_conf["std"]
    mean_conf["-1std"] = mean_conf["mean"] - mean_conf["std"]
    mean_conf.reset_index(inplace=True)

    # Contrast all models using nonparametric t-tests
    pvals = np.zeros((n_variables, n_variables))
    tvals = np.zeros((n_variables, n_variables))
    cohen_ds = np.zeros((n_variables, n_variables))
    for i, model1 in enumerate(variables):
        for j, model2 in enumerate(variables):
            if i >= j:
                continue
            score1 = results_df[results_df[variable] == model1]["Score"].dropna()
            score2 = results_df[results_df[variable] == model2]["Score"].dropna()
            res = ttest_ind(
                score1,
                score2,
                alternative="two-sided",
                equal_var=False,
            )
            pvals[i, j] = res.pvalue
            pvals[j, i] = res.pvalue
            tvals[i, j] = res.statistic
            tvals[j, i] = res.statistic
            
            # Compute Cohen's d
            tval = res.statistic
            nx, ny = len(score1), len(score2)
            cohen_ds[i, j] = compute_effsize_from_t(tval, nx=nx, ny=ny, eftype='cohen')
            cohen_ds[j, i] = cohen_ds[i, j]

    # Correct for multiple comparisons
    reject, pvals_corrected, _, _ = multipletests(
        pvals.flatten(),
        alpha=0.05,
        method="fdr_bh",
        is_sorted=False,
        returnsorted=False,
    )

    # Prepare tables for p-values, t-values and Cohen's d
    pvals_table = pd.DataFrame(index=variables, columns=variables, dtype=float)
    tvals_table = pd.DataFrame(index=variables, columns=variables, dtype=float)
    cohen_d_table = pd.DataFrame(index=variables, columns=variables, dtype=float)
    for i, model1 in enumerate(variables):
        for j, model2 in enumerate(variables):
            if i >= j:
                continue
            pvals_table.loc[model1, model2] = pvals_corrected[i * n_variables + j]
            pvals_table.loc[model2, model1] = pvals_corrected[i * n_variables + j]
            tvals_table.loc[model1, model2] = tvals[i, j]
            tvals_table.loc[model2, model1] = tvals[j, i]
            cohen_d_table.loc[model1, model2] = cohen_ds[i, j]
            cohen_d_table.loc[model2, model1] = cohen_ds[j, i]

    return mean_conf, pvals_table, tvals_table, cohen_d_table


def create_heatmap(
    mean_conf,
    variable,
    tvals_table,
    pvals_table,
    cohen_d_table=None,
    heatmap_type='t-values',  # Options: 't-values' or 'cohen-d'
    pal=None,
    order=None,
    color_order=None,
    only_stars=False,
    xlim=(60, 90),
    save=None,
    large=(10, 5),
    rotation=90,
    axis_name='Mean',
    title_name='Mean DAT score',
):
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(tvals_table, dtype=bool))

    # Set up the matplotlib figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=large)

    if color_order is not None:
        pal = dict(zip(order, color_order))

    # Bar plot
    bar_plot = sns.barplot(
        x=variable,
        y="mean",
        data=mean_conf,
        ax=axs[0],
        palette=pal,
        order=order,
        errorbar=None,
    )
    bar_plot.set_xticklabels(order, rotation=rotation)
    for i, model in enumerate(order):
        model_data = mean_conf[mean_conf[variable] == model]
        error = model_data["sem"].values[0]
        axs[0].errorbar(
            x=i,
            y=model_data["mean"].values[0],
            yerr=error,
            fmt="none",
            color="black",
            capsize=5,
        )
    axs[0].set_xlabel("")
    axs[0].set_ylabel(axis_name)
    axs[0].set_title(title_name)
    yticks = np.linspace(xlim[0], xlim[1], num=6)
    bar_plot.set_yticks(yticks)
    bar_plot.set_ylim(xlim)
    bar_plot.tick_params(labelsize=16)
    bar_plot.set_title(bar_plot.get_title(), fontsize=18, y=1.05)
    sns.despine(ax=axs[0], top=True, right=True, left=False, bottom=False)

    # Heatmap configuration
    if heatmap_type == 'cohen-d' and cohen_d_table is not None:
        heatmap_data = cohen_d_table
        cmap = "Purples"
        cbar_label = "Cohen's d"
        vmax = heatmap_data.abs().max().max()
    else:
        heatmap_data = tvals_table
        cmap = "coolwarm"
        cbar_label = "t-values"
        vmax = heatmap_data.abs().max().max()

    # Prepare heatmap annotations
    mask = mask[1:, :-1]
    heatmap_data = heatmap_data.iloc[1:, :-1]
    pval_stars = pvals_table.iloc[1:, :-1].applymap(
        lambda x: "***" if x < 0.001 else "**" if x < 0.01 else "*" if x < 0.05 else ""
    )
    annotations = (
        heatmap_data.applymap(lambda x: f"\n{x:.2f}")
        if heatmap_type == 'cohen-d'
        else heatmap_data.applymap(lambda x: f"\n{x:.2f}")
    )
    if only_stars is True:
        combined_annot = pval_stars
    else:
        combined_annot = pval_stars + annotations

    sns.heatmap(
        heatmap_data,
        mask=mask,
        annot=combined_annot,
        fmt="",
        cmap=cmap,
        cbar_kws={"label": cbar_label},
        ax=axs[1],
        vmin=-vmax if heatmap_type != 'cohen-d' else 0,
        vmax=vmax,
    )

    axs[1].tick_params(labelsize=16)
    axs[1].set_title(f"Pairwise contrasts ({heatmap_type})", fontsize=18, y=1.05)
    axs[1].set_yticklabels(axs[1].get_yticklabels(), rotation=0)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=rotation)
    colorbar = axs[1].collections[0].colorbar
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label(cbar_label, fontsize=14)
    axs[1].set_xlabel("")
    axs[1].set_ylabel("")

    plt.tight_layout()
    if save is not None:
        plt.savefig(f"{save}.png", dpi=300)
    plt.show()


def most_common_words(words_list, n):
    # Convert each word to lowercase
    counter = collections.Counter(words_list)
    return counter.most_common(n)

def create_bar_plot(word_counts, n_lists=None, ylim=(0, 90),
                    palette_name='Set2', save=False, modelname=' ',
                    temp=' ', strategy=' ', alpha=0.8, title='Top 10 words'):
    #title = '{} ({} temperature)'.format(modelname, temp)
    filename = '{}_word-counts_{}_{}.png'.format(modelname, temp, strategy)
    if n_lists is None:
        n_lists = len(word_counts)/10
    words, counts = zip(*word_counts)
    num_words = len(words)
    
    # Create an array of colors
    #colors = plt.get_cmap(palette_name)(np.linspace(0.6, 0.8, num_words))
    
    # Set the bar width to have more space between bars
    bar_width = 0.5
    
    # Create the bar plot
    fig, ax = plt.subplots()
    bars = ax.bar(words, [(x/n_lists)*100 for x in counts], width=bar_width, color=palette_name, alpha=alpha)
    
    # change x label size
    ax.tick_params(axis='x', labelsize=12)
    
    #change y label size
    ax.tick_params(axis='y', labelsize=12)
    # Add angle to the bar labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    
    # Set plot labels and title
    ax.set_xlabel('Words', fontsize=14)
    ax.set_ylabel('Percentage (%)', fontsize=14)
    #ax.set_title('Top 10 Most Common Words - {}'.format(title))
    ax.set_title(title, fontsize=18)
    ax.set_ylim(ylim)
    # Adjust the bottom margin to prevent cropping of labels when saving
    plt.subplots_adjust(bottom=0.25)
    
    if save:
        plt.savefig(filename, dpi=300)
    else :
        plt.show()