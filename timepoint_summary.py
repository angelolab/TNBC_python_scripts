import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# create dataset
timepoint_df = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_timepoint.csv'))
timepoint_metadata = pd.read_csv(os.path.join(data_dir, 'TONIC_data_per_timepoint.csv'))
timepoint_metadata = timepoint_metadata.loc[:, ['Tissue_ID', 'TONIC_ID', 'Timepoint', 'Localization']]
timepoint_df = timepoint_df.merge(timepoint_metadata, on='Tissue_ID')


# create stacked barplot
def create_barplot(plot_df, x_var, data_var, values_var, xlabel, ylabel, title, colors_dict=None,
                   colormap='husl', savepath=None):
    plot_cross = pd.pivot(plot_df, index=x_var, columns=data_var, values=values_var)

    # order columns by count
    means = plot_cross.mean(axis=0).sort_values(ascending=False)
    plot_cross[x_var] = plot_cross.index
    plot_cross.columns = pd.CategoricalIndex(plot_cross.columns.values, ordered=True,
                                             categories=means.index.tolist() + [x_var])
    plot_cross = plot_cross.sort_index(axis=1)

    # order rows by count of most common x_var
    row_counts = plot_cross[means.index[0]].sort_values(ascending=False)
    plot_cross.index = pd.CategoricalIndex(plot_cross.index.values, ordered=True,
                                           categories=row_counts.index.tolist())
    plot_cross = plot_cross.sort_index(axis=0)

    # set consistent plotting if colors not supplied
    if colors_dict is None:
        color_labels = plot_df[data_var].unique()
        color_labels.sort()

        # List of colors in the color palettes
        rgb_values = sns.color_palette(colormap, len(color_labels))

        # Map continents to the colors
        colors_dict = dict(zip(color_labels, rgb_values))

    # plot barplot
    plot_cross.plot(x=x_var, kind='bar', stacked=True, figsize=(12, 5), color=colors_dict)

    # reverse legend ordering
    handles, labels = plt.gca().get_legend_handles_labels()
    order = list(np.arange(len(plot_cross.columns) - 1))
    order.reverse()
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # annotate plot
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=15)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


# broad clusters across primary tumors
plot_df = timepoint_df.loc[timepoint_df.Timepoint == 'primary_untreated', :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_broad_freq', :]

create_barplot(plot_df=plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
               xlabel='Patient ID', ylabel='Proportion of total cells', colormap='bright',
               title='Frequency of broad clusters across primary tumors',
               savepath=os.path.join(plot_dir, 'Primary_tumor_barplot_freq_broad_cluster.png'))


# tcell clusters across primary tumors
plot_df = timepoint_df.loc[timepoint_df.Timepoint == 'primary_untreated', :]
plot_df = plot_df.loc[plot_df.metric == 'tcell_freq', :]

create_barplot(plot_df=plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
               xlabel='Patient ID', ylabel='Proportion of T cells', colormap='bright',
               title='Frequency of T cell clusters across primary tumors',
               savepath=os.path.join(plot_dir, 'Primary_tumor_barplot_freq_tcell.png'))


# broad clusters across baseline tumors
plot_df = timepoint_df.loc[timepoint_df.Timepoint == 'baseline', :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_broad_freq', :]

create_barplot(plot_df=plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
               xlabel='Patient ID', ylabel='Proportion of total cells', colormap='bright',
               title='Frequency of broad clusters across baseline metastatic tumors',
               savepath=os.path.join(plot_dir, 'Baseline_tumor_barplot_freq_broad_cluster.png'))


# tcell clusters across baseline tumors
plot_df = timepoint_df.loc[timepoint_df.Timepoint == 'baseline', :]
plot_df = plot_df.loc[plot_df.metric == 'tcell_freq', :]

create_barplot(plot_df=plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
               xlabel='Patient ID', ylabel='Proportion of T cells', colormap='bright',
               title='Frequency of T cell clusters across baseline metastatic tumors',
               savepath=os.path.join(plot_dir, 'Baseline_tumor_barplot_freq_tcell.png'))


# cell proportions across timepoints
for cluster_name, plot_name in zip(['cluster_broad_freq', 'cluster_freq'], ['broad_cluster', 'cluster']):

    plot_df = timepoint_df.loc[timepoint_df.metric == cluster_name, :]
    plot_df = plot_df.loc[plot_df.Timepoint.isin(['primary_untreated', 'baseline',
                                                  'post_induction', 'on_nivo'])]

    g = sns.FacetGrid(plot_df, col='cell_type', col_wrap=4, hue='cell_type',
                      palette=['Black'], sharey=False, aspect=1.7)
    g.map(sns.stripplot, 'Timepoint', 'mean')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Cell_prevelance_timepoints_by_{}.png'.format(plot_name)))
    plt.close()


# cell proportions across tissues
plot_df = timepoint_df.loc[timepoint_df.metric == 'cluster_broad_freq', :]
plot_df = plot_df.loc[plot_df.Localization.isin(['Lymphnode', 'Breast', 'Bone', 'Unknown',
                                                 'Muscle', 'Skin', 'Liver',   'Lung']), :]

g = sns.FacetGrid(plot_df, col='cell_type', col_wrap=4, hue='cell_type',
                  palette=['Black'], sharey=False, aspect=2)
g.map(sns.stripplot, 'Localization', 'mean')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Cell_prevelance_tissue_by_broad_cluster.png'))
plt.close()


