import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


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
    for i, model1 in enumerate(variables):
        for j, model2 in enumerate(variables):
            if i >= j:
                continue
            res = ttest_ind(
                results_df[results_df[variable] == model1]["Score"].dropna(),
                results_df[results_df[variable] == model2]["Score"].dropna(),
                alternative="two-sided",
            )
            pvals[i, j] = res.pvalue
            pvals[j, i] = res.pvalue
            tvals[i, j] = res.statistic
            tvals[j, i] = res.statistic

    # Correct for multiple comparisons
    reject, pvals_corrected, _, _ = multipletests(
        pvals.flatten(),
        alpha=0.05,
        method="fdr_bh",
        is_sorted=False,
        returnsorted=False,
    )

    pvals_table = pd.DataFrame(index=variables, columns=variables, dtype=float)
    tvals_table = pd.DataFrame(index=variables, columns=variables, dtype=float)
    for i, model1 in enumerate(variables):
        for j, model2 in enumerate(variables):
            if i >= j:
                continue
            pvals_table.loc[model1, model2] = pvals_corrected[i * n_variables + j]
            pvals_table.loc[model2, model1] = pvals_corrected[i * n_variables + j]
            tvals_table.loc[model1, model2] = tvals[i, j]
            tvals_table.loc[model2, model1] = tvals[j, i]

    return mean_conf, pvals_table, tvals_table


def create_heatmap(
    mean_conf,
    variable,
    tvals_table,
    pvals_table,
    pal,
    order,
    color_order=None,
    xlim=(60, 90),
    save=None,
    large=(10, 5),
    rotation=90,
    axis_name = 'Mean',
    title_name = 'Mean DAT score',
):
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(tvals_table, dtype=bool))

    # Set up the matplotlib figure
    fig, axs = plt.subplots(
        nrows=1, ncols=2, figsize=large
    )

    if color_order is not None:
        pal = dict(zip(order, color_order))

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
    # Add error bars to the barplot
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
    # remove x-axis title
    axs[0].set_xlabel("")
    axs[0].set_ylabel(axis_name)
    # axs[0].set_xticklabels(order, rotation=20)
    axs[0].set_title(title_name)

    # Set y-axis ticks manually
    yticks = np.linspace(
        xlim[0], xlim[1], num=6
    )  # Change num to the desired number of ticks
    bar_plot.set_yticks(yticks)
    bar_plot.set_ylim(xlim)
    # set size of all fonts
    bar_plot.tick_params(labelsize=16)
    # set size of title
    bar_plot.set_title(bar_plot.get_title(), fontsize=18, y=1.05)
    # set size of x-axis label
    bar_plot.set_xlabel(bar_plot.get_xlabel(), fontsize=18)
    # set size of y-axis label
    bar_plot.set_ylabel(bar_plot.get_ylabel(), fontsize=18)
    # remove box around plot
    sns.despine(ax=axs[0], top=True, right=True, left=False, bottom=False)
    # Generate the combined heat map for t-values and p-values
    tvals_annot = tvals_table.applymap(lambda x: f"\n{x:.2f}")
    pval_stars = pvals_table.applymap(
        lambda x: "***" if x < 0.001 else "**" if x < 0.01 else "*" if x < 0.05 else ""
    )
    combined_annot = pval_stars + tvals_annot
    max_tval = np.abs(tvals_table.values[~mask]).max()
    # remove the first row and last column from the mask and the table
    mask = mask[1:, :-1]
    tvals_table = tvals_table.iloc[1:, :-1]
    combined_annot = combined_annot.iloc[1:, :-1]
    sns.heatmap(
        tvals_table,
        mask=mask,
        annot=combined_annot,
        fmt="",
        cmap="coolwarm",
        cbar_kws={"label": "t-value"},
        ax=axs[1],
        vmin=-max_tval,
        vmax=max_tval,
    )
    # change label size
    axs[1].tick_params(labelsize=16)
    # change title size
    axs[1].set_title("Pairwise contrasts")
    # set y label rotation
    axs[1].set_yticklabels(axs[1].get_yticklabels(), rotation=0)
    # set axis title size
    axs[1].set_xlabel(axs[1].get_xlabel(), fontsize=14)
    axs[1].set_ylabel(axs[1].get_ylabel(), fontsize=14)
    # set title size
    axs[1].set_title(axs[1].get_title(), fontsize=18, y=1.05)
    # change rotation of x-axis labels
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=rotation)
    # change heatmap colorbar label size
    colorbar = axs[1].collections[0].colorbar
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label("t-values", fontsize=14)
    

    
    
    # axs[1].set_xticklabels(order, rotation=20)
    # remove axis titles
    axs[1].set_xlabel("")
    axs[1].set_ylabel("")

    plt.tight_layout()
    if save is not None:
        plt.savefig(f"{save}.png", dpi=300)
    plt.show()


def most_common_words(words_list, n):
    # Convert each word to lowercase
    lowercase_words_list = [word.lower() for word in words_list]
    counter = collections.Counter(lowercase_words_list)
    return counter.most_common(n)

def create_bar_plot(word_counts, n_lists=None, ylim=(0, 90),
                    palette_name='Set2', save=False, modelname=' ',
                    temp=' ', strategy=' ', alpha=0.8):
    title = '{} ({} temperature)'.format(modelname, temp)
    filename = '{}_word-counts_{}_{}.png'.format(modelname, temp, strategy)
    if n_lists is None:
        n_lists = len(word_counts)/10
    words, counts = zip(*word_counts)
    num_words = len(words)
    
    # Create an array of colors
    colors = plt.get_cmap(palette_name)(np.linspace(0.6, 0.8, num_words))
    
    # Set the bar width to have more space between bars
    bar_width = 0.5
    
    # Create the bar plot
    fig, ax = plt.subplots()
    bars = ax.bar(words, [(x/n_lists)*100 for x in counts], width=bar_width, color=colors, alpha=alpha)
    
    # Add angle to the bar labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    
    # Set plot labels and title
    ax.set_xlabel('Words')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Top 10 Most Common Words - {}'.format(title))
    ax.set_ylim(ylim)
    # Adjust the bottom margin to prevent cropping of labels when saving
    plt.subplots_adjust(bottom=0.25)
    
    if save:
        plt.savefig(filename, dpi=300)
    else :
        plt.show()