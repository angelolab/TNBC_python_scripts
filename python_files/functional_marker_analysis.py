import os
import pandas as pd
import matplotlib.pyplot as plt

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# create dataset
core_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_timepoint.csv'))
core_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core.csv'))


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