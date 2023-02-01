import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
post_dir = os.path.join(data_dir, 'post_processing')
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# create dataset
core_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
core_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core.csv'))


# filter functional markers to only include FOVs with at least the specified number of cells
min_cells = 5

filtered_dfs = []
metrics = [['cluster_broad_count', 'cluster_broad_freq'],
           ['cluster_count', 'cluster_freq'],
           ['meta_cluster_count', 'meta_cluster_freq']]

for metric in metrics:
    # subset count df to include cells at the relevant clustering resolution
    count_df = core_df_cluster[core_df_cluster.metric == metric[0]]
    count_df = count_df[count_df.subset == 'all']

    # subset functional df to only include functional markers at this resolution
    func_df = core_df_func[core_df_func.metric.isin(metric)]

    # for each cell type, determine which FOVs have high enough counts to be included
    for cell_type in count_df.cell_type.unique():
        keep_df = count_df[count_df.cell_type == cell_type]
        keep_df = keep_df[keep_df.value >= min_cells]
        keep_fovs = keep_df.fov.unique()

        # subset functional df to only include FOVs with high enough counts
        keep_markers = func_df[func_df.cell_type == cell_type]
        keep_markers = keep_markers[keep_markers.fov.isin(keep_fovs)]

        # append to list of filtered dfs
        filtered_dfs.append(keep_markers)

filtered_func_df = pd.concat(filtered_dfs)

# identify combinations of markers and cell types to include in analysis based on threshold
mean_percent_positive = 0.05
broad_df = filtered_func_df[filtered_func_df.metric == 'cluster_broad_freq']
broad_df = broad_df[broad_df.subset == 'all']
broad_df = broad_df[broad_df.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
broad_df_agg = broad_df[['fov', 'functional_marker', 'cell_type', 'value']].groupby(['cell_type', 'functional_marker']).agg(np.mean)

broad_df_agg.reset_index(inplace=True)
broad_df = broad_df_agg.pivot(index='cell_type', columns='functional_marker', values='value')
broad_df_include = broad_df > mean_percent_positive

# include for all cells
general_markers = ['Ki67', 'HLA1', 'H3K9ac_H3K27me3_ratio']
broad_df_include[[general_markers]] = True

# CD45 isoform ratios
double_pos = np.logical_and(broad_df_include['CD45RO'], broad_df_include['CD45RB'])
broad_df_include['CD45RO_CD45RB_ratio'] = double_pos

# Cancer expression
broad_df_include.loc['Cancer', ['HLADR', 'CD57']] = True

broad_df_include.to_csv(os.path.join(post_dir, 'broad_inclusion_matrix.csv'))

# apply thresholds to medium level clustering
assignment_dict_med = {'Cancer': ['Cancer', 'Cancer_EMT', 'Cancer_Other'],
                     'Mono_Mac': ['M1_Mac', 'M2_Mac', 'Mac_Other', 'Monocyte', 'APC'],
                     'B': ['B'],
                     'T': ['CD4T', 'CD8T', 'Treg', 'T_Other', 'Immune_Other'],
                     'Granulocyte': ['Neutrophil', 'Mast'],
                     'Stroma': ['Endothelium', 'Fibroblast', 'Stroma'],
                     'NK': ['NK'],
                     'Other': ['Other']}

# get a list of all cell types
cell_types = np.concatenate([assignment_dict_med[key] for key in assignment_dict_med.keys()])
cell_types.sort()

med_df_include = pd.DataFrame(index=cell_types, columns=broad_df.columns)

for key in assignment_dict_med.keys():
    values = assignment_dict_med[key]
    med_df_include.loc[values] = broad_df_include.loc[key].values

# check if assignment makes sense
med_df = filtered_func_df[filtered_func_df.metric == 'cluster_freq']
med_df = med_df[med_df.subset == 'all']
med_df = med_df[med_df.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
med_df_agg = med_df[['fov', 'functional_marker', 'cell_type', 'value']].groupby(['cell_type', 'functional_marker']).agg(np.mean)

med_df_agg.reset_index(inplace=True)
med_df = med_df_agg.pivot(index='cell_type', columns='functional_marker', values='value')
med_df_bin = med_df > mean_percent_positive

# add CD38 signal
med_df_include.loc[['Endothelium', 'Immune_Other'], 'CD38'] = True

# add IDO signal
med_df_include.loc[['APC'], 'IDO'] = True

# compare to see where assignments disagree, to see if any others need to be added
new_includes = (med_df_bin == True) & (med_df_include == False)

med_df_include.to_csv(os.path.join(post_dir, 'med_inclusion_matrix.csv'))

# do the same for the fine-grained clustering
assignment_dict_meta = {'Cancer': ['Cancer_CD56', 'Cancer_CK17', 'Cancer_Ecad'],
                   'Cancer_EMT': ['Cancer_SMA', 'Cancer_Vim'],
                   'Cancer_Other': ['Cancer_Other', 'Cancer_Mono'],
                   'M1_Mac': ['CD68'],
                   'M2_Mac': ['CD163'],
                   'Mac_Other': ['CD68_CD163_DP'],
                   'Monocyte': ['CD4_Mono', 'CD14'],
                   'APC': ['CD11c_HLADR'],
                   'B':  ['CD20'],
                   'Endothelium': ['CD31', 'CD31_VIM'],
                   'Fibroblast': ['FAP', 'FAP_SMA', 'SMA'],
                   'Stroma': ['Stroma_Collagen', 'Stroma_Fibronectin', 'VIM'],
                   'NK': ['CD56'],
                   'Neutrophil': ['Neutrophil'],
                   'Mast': ['Mast'],
                   'CD4T': ['CD4T','CD4T_HLADR'],
                   'CD8T': ['CD8T'],
                   'Treg': ['Treg'],
                   'T_Other': ['CD3_DN','CD4T_CD8T_DP'],
                   'Immune_Other': ['Immune_Other'],
                   'Other': ['Other']}

# get a list of all cell types
cell_types = np.concatenate([assignment_dict_meta[key] for key in assignment_dict_meta.keys()])
cell_types.sort()

meta_df_include = pd.DataFrame(index=cell_types, columns=broad_df.columns)

for key in assignment_dict_meta.keys():
    values = assignment_dict_meta[key]
    meta_df_include.loc[values] = med_df_include.loc[key].values

# check if assignment makes sense
meta_df = filtered_func_df[filtered_func_df.metric == 'meta_cluster_freq']
meta_df = meta_df[meta_df.subset == 'all']
meta_df = meta_df[meta_df.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo'])]
meta_df_agg = meta_df[['fov', 'functional_marker', 'cell_type', 'value']].groupby(['cell_type', 'functional_marker']).agg(np.mean)

meta_df_agg.reset_index(inplace=True)
meta_df = meta_df_agg.pivot(index='cell_type', columns='functional_marker', values='value')
meta_df_bin = meta_df > mean_percent_positive

# compare to see where assignments disagree
new_includes = (meta_df_bin == True) & (meta_df_include == False)

meta_df_include.to_csv(os.path.join(post_dir, 'meta_inclusion_matrix.csv'))

# process functional data so that only the specified cell type/marker combos are included


# old code
# identify combinations of markers that are expressed more often than expected by chance
functional_markers = ['Ki67','CD38','CD45RB','CD45RO','CD57','CD69','GLUT1','IDO',
 'PD1','PDL1','TBET','TCF1','TIM3', 'LAG3'] # 'HLA1',

# create a list of all possible combinations of markers
from itertools import combinations

ratios = []
for cell_type in cell_table_func.cell_cluster_broad.unique():
    observed_ratio = np.zeros((len(functional_markers), len(functional_markers)))
    observed_ratio = pd.DataFrame(observed_ratio, index=functional_markers, columns=functional_markers)

    expected_ratio = observed_ratio.copy()
    cell_table_subset = cell_table_func[cell_table_func['cell_cluster_broad'] == cell_type]

    for marker1, marker2 in combinations(functional_markers, 2):
        # calculate the observed ratio of double positive cells
        marker1_pos = cell_table_subset[marker1].values
        marker2_pos = cell_table_subset[marker2].values
        double_pos_observed = np.logical_and(marker1_pos, marker2_pos)
        observed_ratio.loc[marker1, marker2] = np.sum(double_pos_observed) / len(cell_table_subset)

        # calculated the expected ratio of double positive cells
        double_pos_expected = (np.sum(marker1_pos) / len(cell_table_subset)) * (np.sum(marker2_pos) / len(cell_table_subset))
        expected_ratio.loc[marker1, marker2] = double_pos_expected

    obs_exp = np.log2(observed_ratio / expected_ratio)
    ratios.append(obs_exp)
    # create heatmap of observed and observed vs expected ratios
    import seaborn as sns
    # fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    # sns.heatmap(observed_ratio, ax=ax[0], cmap='viridis')
    # sns.heatmap(expected_ratio, ax=ax[1],cmap='viridis',)
    # sns.heatmap(obs_exp, ax=ax[2], cmap='vlag', vmin=-3, vmax=3)
    # ax[0].set_title('Observed Ratio')
    # ax[1].set_title('Expected Ratio')
    # ax[2].set_title('Observed / Expected Ratio')
    # plt.suptitle(cell_type)
    # plt.tight_layout()
    g = sns.heatmap(obs_exp, cmap='vlag', vmin=-3, vmax=3)
    plt.title(cell_type)
    plt.savefig('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Plots/Functional_Markers_Broad/' + cell_type + '_functional_marker_heatmap.png', dpi=300)
    plt.close()


# collapse ratios into single array
ratios = np.stack(ratios, axis=0)
avg = np.nanmean(ratios, axis=0)
avg = pd.DataFrame(avg, index=functional_markers, columns=functional_markers)
sns.heatmap(avg, cmap='vlag', vmin=-3, vmax=3)
plt.tight_layout()
plt.savefig('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Plots/Functional_Markers/avg_functional_marker_heatmap.png', dpi=300)
plt.close()





# heatmap of functional marker expression per cell type
plot_df = core_df_func.loc[core_df_func.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_freq', :]
plot_df = plot_df.loc[~plot_df.functional_marker.isin(['PDL1_cancer_dim']), :]
plot_df = plot_df.loc[~plot_df.functional_marker.isin(['PDL1']), :]
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
plt.savefig(os.path.join(plot_dir, 'Functional_marker_heatmap_min_max_normalized_lag3.png'))
plt.close()