# identify combinations of markers that are expressed more often than expected by chance
functional_markers = ['Ki67','CD38','CD45RB','CD45RO','CD57','CD69','GLUT1','IDO',
 'PD1','PDL1','HLADR','TBET','TCF1','TIM3'] # 'HLA1',

# create a list of all possible combinations of markers
from itertools import combinations

ratios = []
for cell_type in cell_table.cell_cluster.unique():
    observed_ratio = np.zeros((len(functional_markers), len(functional_markers)))
    observed_ratio = pd.DataFrame(observed_ratio, index=functional_markers, columns=functional_markers)

    expected_ratio = observed_ratio.copy()
    cell_table_subset = cell_table[cell_table['cell_cluster'] == cell_type]

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
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    sns.heatmap(observed_ratio, ax=ax[0], cmap='viridis')
    sns.heatmap(expected_ratio, ax=ax[1],cmap='viridis',)
    sns.heatmap(obs_exp, ax=ax[2], cmap='vlag', vmin=-3, vmax=3)
    ax[0].set_title('Observed Ratio')
    ax[1].set_title('Expected Ratio')
    ax[2].set_title('Observed / Expected Ratio')
    plt.suptitle(cell_type)
    plt.tight_layout()
    fig.savefig('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Plots/Functional_Markers/' + cell_type + '_functional_marker_heatmap.png', dpi=300)
    plt.close()