import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# create dataset
timepoint_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_timepoint.csv'))
timepoint_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_timepoint.csv'))


# create stacked barplot

def create_summary_stacked_barplot(plot_df, x_var, data_var, values_var, category_var, xlabel, ylabel, title, colors_dict=None,
                   colormap='husl', savepath=None):
    plot_cross = pd.pivot(plot_df, index=[x_var, category_var], columns=data_var, values=values_var)
    plot_cross = plot_cross.reset_index()

    # order columns by count
    means = plot_cross.mean(axis=0).sort_values(ascending=False)
    #plot_cross[x_var] = plot_cross.index
    plot_cross.columns = pd.CategoricalIndex(plot_cross.columns.values, ordered=True,
                                             categories=means.index.tolist() + [x_var])
    plot_cross = plot_cross.sort_index(axis=1)

    # order rows by count of most common x_var
    row_counts1 = plot_cross.loc[plot_cross[category_var], means.index[0]].sort_values(ascending=False)
    row_counts2 = plot_cross.loc[~plot_cross[category_var], means.index[0]].sort_values(ascending=False)

    plot_cross.index = pd.CategoricalIndex(plot_cross.index.values, ordered=True,
                                           categories=row_counts1.index.tolist() + row_counts2.index.tolist())
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
    order = list(np.arange(len(plot_cross.columns) - 2))
    order.reverse()
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # remove x tick labels
    plt.xticks([])

    # despine
    sns.despine(left=True, bottom=True)

    # annotate plot
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=15)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()

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


# stacked bar across all timepoints
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.metric == 'cluster_broad_freq', :]
plot_df = plot_df.loc[plot_df.Timepoint.isin(['primary_untreated', 'primary', 'baseline','local_recurrence',
                                              'metastasis', 'post_induction', 'on_nivo', 'biopsy'])]
plot_df = plot_df.loc[plot_df.subset == 'all', :]
plot_df = plot_df.loc[plot_df.MIBI_data_generated, :]
plot_df['primary'] = plot_df.Timepoint.isin(['primary_untreated', 'primary', 'biopsy', 'local_recurrence'])

create_summary_stacked_barplot(plot_df=plot_df, x_var='Tissue_ID', data_var='cell_type', values_var='mean',
                       category_var='primary',
                     xlabel='Timepoint', ylabel='Proportion of total cells', colormap='husl',
                     title='Frequency of broad clusters across all timepoints',
                     savepath=os.path.join(plot_dir, 'Figure2_primary_vs_met_broad_cluster.png'))


# broad clusters across primary tumors
tumor_regions = timepoint_df_cluster.subset.unique()
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.Timepoint == 'primary_untreated', :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_broad_freq', :]

for tumor_region in tumor_regions:
    region_plot_df = plot_df.loc[plot_df.subset == tumor_region, :]
    create_stacked_barplot(plot_df=region_plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
                   xlabel='Patient ID', ylabel='Proportion of total cells', colormap='bright',
                   title='Frequency of broad clusters across primary tumors in {}'.format(tumor_region),
                   savepath=os.path.join(plot_dir, 'Primary_tumor_barplot_freq_broad_cluster_{}.png'.format(tumor_region)))


# tcell clusters across primary tumors
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.Timepoint == 'primary_untreated', :]
plot_df = plot_df.loc[plot_df.metric == 'tcell_freq', :]

for tumor_region in tumor_regions:
    region_plot_df = plot_df.loc[plot_df.subset == tumor_region, :]
    create_stacked_barplot(plot_df=region_plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
                   xlabel='Patient ID', ylabel='Proportion of T cells', colormap='bright',
                   title='Frequency of T cell clusters across primary tumors in {}'.format(tumor_region),
                   savepath=os.path.join(plot_dir, 'Primary_tumor_barplot_freq_tcell_cluster_{}.png'.format(tumor_region)))


# broad clusters across metastatic tumors
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.Timepoint == 'baseline', :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_broad_freq', :]

for tumor_region in tumor_regions:
    region_plot_df = plot_df.loc[plot_df.subset == tumor_region, :]
    create_stacked_barplot(plot_df=region_plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
                   xlabel='Patient ID', ylabel='Proportion of total cells', colormap='bright',
                   title='Frequency of broad clusters across metastatic tumors in {}'.format(tumor_region),
                   savepath=os.path.join(plot_dir, 'Metastatic_tumor_barplot_freq_broad_cluster_{}.png'.format(tumor_region)))


# tcell clusters across metastatic tumors
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.Timepoint == 'baseline', :]
plot_df = plot_df.loc[plot_df.metric == 'tcell_freq', :]

for tumor_region in tumor_regions:
    region_plot_df = plot_df.loc[plot_df.subset == tumor_region, :]
    create_stacked_barplot(plot_df=region_plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
                   xlabel='Patient ID', ylabel='Proportion of T cells', colormap='bright',
                   title='Frequency of T cell clusters across metastatic tumors in {}'.format(tumor_region),
                   savepath=os.path.join(plot_dir, 'Metastatic_tumor_barplot_freq_tcell_cluster_{}.png'.format(tumor_region)))

# kmeans freqs across primary tumors
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.Timepoint == 'primary_untreated', :]
plot_df = plot_df.loc[plot_df.metric == 'kmeans_freq', :]

for tumor_region in tumor_regions:
    region_plot_df = plot_df.loc[plot_df.subset == tumor_region, :]
    create_stacked_barplot(plot_df=region_plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
                   xlabel='Patient ID', ylabel='Proportion of total cells', colormap='hls',
                   title='Frequency of kmeans clusters across primary tumors in {}'.format(tumor_region),
                   savepath=os.path.join(plot_dir, 'Primary_tumor_barplot_freq_kmeans_cluster_{}.png'.format(tumor_region)))


# kmeans freqs across metastatic tumors
plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.Timepoint == 'baseline', :]
plot_df = plot_df.loc[plot_df.metric == 'kmeans_freq', :]

for tumor_region in tumor_regions:
    region_plot_df = plot_df.loc[plot_df.subset == tumor_region, :]
    create_stacked_barplot(plot_df=region_plot_df, x_var='TONIC_ID', data_var='cell_type', values_var='mean',
                   xlabel='Patient ID', ylabel='Proportion of total cells', colormap='hls',
                   title='Frequency of kmeans clusters across metastatic tumors in {}'.format(tumor_region),
                   savepath=os.path.join(plot_dir, 'Metastatic_tumor_barplot_freq_kmeans_cluster_{}.png'.format(tumor_region)))


# generate paired plots for broad cluster and medium cluster resolutions
# for cluster_name, plot_name in zip(['cluster_broad_freq', 'cluster_freq', 'immune_freq'], ['broad_cluster', 'cluster', 'immune_cluster']):
for cluster_name, plot_name in zip(['kmeans_freq'],  ['kmeans_clusters']):

    # cell proportions across timepoints
    plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.metric == cluster_name, :]
    plot_df = plot_df.loc[plot_df.Timepoint.isin(['primary_untreated', 'baseline',
                                                  'post_induction', 'on_nivo'])]

    g = sns.FacetGrid(plot_df, col='cell_type', col_wrap=3 , hue='cell_type',
                      palette=['Black'], sharey=False, aspect=2.5)
    g.map(sns.stripplot, 'subset', 'mean', order=['cancer_core', 'cancer_border', 'stroma_border', 'stroma_core', 'all'])

    # add a title
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Cell frequencies in {} by tumor region'.format(plot_name), fontsize=20)

    g.savefig(os.path.join(plot_dir, 'Cell_freq_tumor_region_by_{}.png'.format(plot_name)))
    plt.close()

    # cell proportions across tissues
    # plot_df = timepoint_df_cluster.loc[timepoint_df_cluster.metric == cluster_name, :]
    # plot_df = plot_df.loc[plot_df.Localization.isin(['Lymphnode', 'Breast', 'Bone', 'Unknown',
    #                                                  'Muscle', 'Skin', 'Liver',   'Lung']), :]
    #
    # g = sns.FacetGrid(plot_df, col='cell_type', col_wrap=4, hue='cell_type',
    #                   palette=['Black'], sharey=False, aspect=2)
    # g.map(sns.stripplot, 'Localization', 'mean', order=['Lymphnode', 'Breast', 'Bone', 'Unknown',
    #                                                     'Muscle', 'Skin', 'Liver', 'Lung'])
    # plt.tight_layout()
    # plt.savefig(os.path.join(plot_dir, 'Cell_prevelance_tissue_by_{}.png'.format(plot_name)))
    # plt.close()


#
# functional marker plotting
#

# one plot per cell type, across cell types by tumor region
#for cluster_name, plot_name in zip(['cluster_freq', 'cluster_broad_freq'], ['cluster', 'cluster_broad']):
for cluster_name, plot_name in zip(['cluster_freq'], ['cluster']):
    plot_df = timepoint_df_func.loc[timepoint_df_func.metric == cluster_name, :]
    plot_df = plot_df.loc[plot_df.Timepoint.isin(['primary_untreated', 'baseline']), :]

    cell_types = plot_df.cell_type.unique()
    for cell_type in cell_types:
        cell_df = plot_df.loc[plot_df.cell_type == cell_type, :]

        g = sns.catplot(data=cell_df, x='subset', y='mean', col='functional_marker', col_wrap=5, kind='box', sharey=False,
                        order=['cancer_core', 'cancer_border', 'stroma_border', 'stroma_core', 'all'])
        for ax in g.axes_dict.values():
            ax.tick_params(labelrotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Functional_marker_boxplot_by_{}_for_{}.png'.format(plot_name, cell_type)))
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
