import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# create dataset
timepoint_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_timepoint.csv'))
timepoint_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_timepoint.csv'))


# create stacked barplot
def create_stacked_barplot(plot_df, x_var, data_var, values_var, xlabel, ylabel, title, colors_dict=None,
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


def create_sorted_barplot(plot_df, x_var, xlabel, ylabel, title, colors_dict=None,
                          colormap='husl', savepath=None):
    # create categorical index ordered by count
    counts = plot_df.groupby(x_var).size()
    counts = counts.sort_values(ascending=False)

    # set consistent plotting if colors not supplied
    if colors_dict is None:
        color_labels = plot_df[x_var].unique()
        color_labels.sort()

        # List of colors in the color palettes
        rgb_values = sns.color_palette(colormap, len(color_labels))

        # Map continents to the colors
        colors_dict = dict(zip(color_labels, rgb_values))

    # plot barplot
    sns.catplot(plot_df, x=x_var,  kind='count', order=counts.index, color=colors_dict)

    # # reverse legend ordering
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = list(np.arange(len(plot_cross.columns) - 1))
    # order.reverse()
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # annotate plot
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=15)
    plt.tight_layout()
    # if savepath is not None:
    #     plt.savefig(savepath)
    #     plt.close()


# broad clusters across primary tumors
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.Timepoint == 'primary_untreated', :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_broad_freq', :]

create_stacked_barplot(plot_df=plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
               xlabel='Patient ID', ylabel='Proportion of total cells', colormap='bright',
               title='Frequency of broad clusters across primary tumors',
               savepath=os.path.join(plot_dir, 'Primary_tumor_barplot_freq_broad_cluster.png'))


# tcell clusters across primary tumors
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.Timepoint == 'primary_untreated', :]
plot_df = plot_df.loc[plot_df.metric == 'tcell_freq', :]

create_stacked_barplot(plot_df=plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
               xlabel='Patient ID', ylabel='Proportion of T cells', colormap='bright',
               title='Frequency of T cell clusters across primary tumors',
               savepath=os.path.join(plot_dir, 'Primary_tumor_barplot_freq_tcell.png'))


# broad clusters across metastatic tumors
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.Timepoint == 'baseline', :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_broad_freq', :]

create_stacked_barplot(plot_df=plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
               xlabel='Patient ID', ylabel='Proportion of total cells', colormap='bright',
               title='Frequency of broad clusters across metastatic tumors',
               savepath=os.path.join(plot_dir, 'Metastatic_tumor_barplot_freq_broad_cluster.png'))


# tcell clusters across metastatic tumors
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.Timepoint == 'baseline', :]
plot_df = plot_df.loc[plot_df.metric == 'tcell_freq', :]

create_stacked_barplot(plot_df=plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
               xlabel='Patient ID', ylabel='Proportion of T cells', colormap='bright',
               title='Frequency of T cell clusters across metastatic tumors',
               savepath=os.path.join(plot_dir, 'Metastatic_tumor_barplot_freq_tcell.png'))

# kmeans freqs across primary tumors
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.Timepoint == 'primary_untreated', :]
plot_df = plot_df.loc[plot_df.metric == 'kmeans_freq', :]

create_stacked_barplot(plot_df=plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
                       xlabel='Patient ID', ylabel='Proportion of cells', colormap='hls',
                       title='Frequency of neighborhoods across primary tumors')
                       #savepath=os.path.join(plot_dir, 'Metastatic_tumor_barplot_freq_tcell.png'))

# generate paired plots for broad cluster and medium cluster resolutions
#for cluster_name, plot_name in zip(['cluster_broad_freq', 'cluster_freq', 'immune_freq'], ['broad_cluster', 'cluster', 'immune_cluster']):
for cluster_name, plot_name in zip(['kmeans_freq'],  ['kmeans_clusters']):

    # cell proportions across timepoints
    plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.metric == cluster_name, :]
    plot_df = plot_df.loc[plot_df.Timepoint.isin(['primary_untreated', 'baseline',
                                                  'post_induction', 'on_nivo'])]

    g = sns.FacetGrid(plot_df, col='cell_type', col_wrap=4, hue='cell_type',
                      palette=['Black'], sharey=False, aspect=1.7)
    g.map(sns.stripplot, 'Timepoint', 'mean', order=['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Cell_prevelance_timepoints_by_{}.png'.format(plot_name)))
    plt.close()

    # cell proportions across tissues
    plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.metric == cluster_name, :]
    plot_df = plot_df.loc[plot_df.Localization.isin(['Lymphnode', 'Breast', 'Bone', 'Unknown',
                                                     'Muscle', 'Skin', 'Liver',   'Lung']), :]

    g = sns.FacetGrid(plot_df, col='cell_type', col_wrap=4, hue='cell_type',
                      palette=['Black'], sharey=False, aspect=2)
    g.map(sns.stripplot, 'Localization', 'mean', order=['Lymphnode', 'Breast', 'Bone', 'Unknown',
                                                        'Muscle', 'Skin', 'Liver', 'Lung'])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Cell_prevelance_tissue_by_{}.png'.format(plot_name)))
    plt.close()


#
# functional marker plotting
#

# functional markers across cell types and timepoints
#for cluster_name, plot_name in zip(['avg_per_cluster_broad', 'avg_per_cluster'], ['broad_cluster', 'cluster']):
for cluster_name, plot_name in zip(['cluster_freq'], ['cluster']):
#for cluster_name, plot_name in zip(['kmeans_freq'], ['kmeans_cluster']):
    #for timepoint in ['primary_untreated', 'baseline', 'post_induction', 'on_nivo', 'all']:
    for timepoint in ['all']:
        plot_df = timepoint_df_func.loc[np.logical_and(timepoint_df_func.metric == cluster_name, ~timepoint_df_func.functional_marker.isin(['PDL1_cancer_dim'])), :]

        if timepoint == 'all':
            plot_df = plot_df.loc[plot_df.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
        else:
            plot_df = plot_df.loc[plot_df.Timepoint == timepoint]

        g = sns.catplot(data=plot_df, x='cell_type', y='mean', col='functional_marker', col_wrap=5, kind='bar', sharey=False)
        for ax in g.axes_dict.values():
            ax.tick_params(labelrotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Functional_marker_barplot_by_{}_in_{}.png'.format(plot_name, timepoint)))
        plt.close()


# heatmap of functional marker expression per cell type
plot_df = timepoint_df_func.loc[timepoint_df_func.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_freq', :]
plot_df = plot_df.loc[~plot_df.functional_marker.isin(['PDL1_cancer_dim']), :]
plot_df = plot_df.loc[~plot_df.functional_marker.isin(['H3K9ac_H3K27me3_ratio']), :]
plot_df = plot_df.loc[~plot_df.functional_marker.isin(['CD45RO_CD45RB_ratio']), :]

# # compute z-score within each functional marker
# plot_df['zscore'] = plot_df.groupby('functional_marker')['mean'].transform(lambda x: (x - x.mean()) / x.std())

# average the z-score across cell types
plot_df = plot_df.groupby(['cell_type', 'functional_marker']).mean().reset_index()
plot_df = pd.pivot(plot_df, index='cell_type', columns='functional_marker', values='mean')
plot_df = plot_df.apply(lambda x: (x - x.min()), axis=0)
plot_df = plot_df.apply(lambda x: (x / x.max()), axis=0)

# plot heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(plot_df, cmap=sns.color_palette("Greys", as_cmap=True), vmin=0, vmax=1)
plt.savefig(os.path.join(plot_dir, 'Functional_marker_heatmap_min_max_normalized.png'))
plt.close()

# functional markers across cell types and timepoints
# TODO: make top for loop work with correct names
for cluster_name, plot_name in zip(['cluster_broad_freq', 'cluster_freq'], ['broad_cluster', 'cluster']):
    for marker in timepoint_df_func.functional_marker.unique():
        if marker == 'PDL1_cancer_dim':
            continue
        plot_df = timepoint_df_func.loc[timepoint_df_func.metric == 'avg_per_cluster', :]
        plot_df = plot_df.loc[plot_df.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
        plot_df = plot_df.loc[plot_df.functional_marker == marker, ]

        g = sns.catplot(data=plot_df, x='Timepoint', y='mean', col='cell_type', col_wrap=5, kind='strip')
        for ax in g.axes_dict.values():
            ax.tick_params(labelrotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Functional_marker_{}_swarm_per_timepoint_by_cluster.png'.format(marker, timepoint)))
        plt.close()
