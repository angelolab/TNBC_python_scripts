import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr



data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

core_df_cluster = pd.read_csv(os.path.join(data_dir, 'cluster_df_per_core.csv'))
core_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core_filtered.csv'))
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'harmonized_metadata.csv'))

fov_metadata = harmonized_metadata[harmonized_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'on_nivo', 'post_induction'])]
fov_metadata = fov_metadata[fov_metadata.fov.isin(core_df_cluster.fov.unique())]
keep_fov_1 = []
keep_fov_2 = []
for name, items in fov_metadata[['fov', 'Tissue_ID']].groupby('Tissue_ID'):
    if len(items) > 1:
        keep_fov_1.append(items.iloc[0].fov)
        keep_fov_2.append(items.iloc[1].fov)


conserved_df = core_df_cluster[core_df_cluster.fov.isin(keep_fov_1 + keep_fov_2)]
conserved_df['placeholder'] = conserved_df.fov.isin(keep_fov_1).astype(int)
conserved_df['placeholder'] = conserved_df['placeholder'].replace({0: 'fov1', 1: 'fov2'})

conserved_df = conserved_df[conserved_df.metric.isin(['cluster_density'])]
conserved_df = conserved_df.pivot(index=['Tissue_ID', 'metric', 'subset', 'cell_type'], columns='placeholder', values='value')
conserved_df = conserved_df.reset_index()

p_vals = []
cors = []
names = []
for metric, cell_type, subset in conserved_df[['metric', 'cell_type', 'subset']].drop_duplicates().values:
    values = conserved_df[(conserved_df.metric == metric) & (conserved_df.cell_type == cell_type) & (conserved_df.subset == subset)]
    values.dropna(inplace=True)
    cor, p_val = spearmanr(values.fov1, values.fov2)
    p_vals.append(p_val)
    cors.append(cor)
    names.append(f'{metric}__{cell_type}__{subset}')

p_df = pd.DataFrame({'name': names, 'p_val': p_vals, 'cor': cors})
p_df['log_pval'] = -np.log10(p_df.p_val)

sns.scatterplot(data=p_df, x='cor', y='log_pval')
plt.savefig(os.path.join(plot_dir, 'conserved_features_volcano.png'))
plt.close()

sns.histplot(data=p_df, x='cor')
plt.savefig(os.path.join(plot_dir, 'conserved_features_cor_random.png'))
plt.close()


plot_subset = p_df[(p_df.log_pval > 3) & (p_df.cor > 0.6)]

# get a random row
row = plot_subset.sample(1).iloc[0]
metric, cell_type, subset = row['name'].split('__')
correlation = row['cor']
values = conserved_df[(conserved_df.metric == metric) & (conserved_df.cell_type == cell_type) & (conserved_df.subset == subset)]
values.dropna(inplace=True)
sns.scatterplot(data=values, x='fov1', y='fov2')
plt.title(f'{metric}__{cell_type}__{subset}__{correlation}')
plt.savefig(os.path.join(plot_dir, 'conserved_features_5.png'))
plt.close()


### same thing for functional markers
conserved_df = core_df_func[core_df_func.fov.isin(keep_fov_1 + keep_fov_2)]
conserved_df['placeholder'] = conserved_df.fov.isin(keep_fov_1).astype(int)
conserved_df['placeholder'] = conserved_df['placeholder'].replace({0: 'fov1', 1: 'fov2'})

conserved_df = conserved_df[conserved_df.metric.isin(['cluster_freq'])]
conserved_df = conserved_df.pivot(index=['Tissue_ID', 'metric', 'subset', 'cell_type', 'functional_marker'], columns='placeholder', values='value')
conserved_df = conserved_df.reset_index()

p_vals = []
cors = []
names = []
for metric, cell_type, subset, func in conserved_df[['metric', 'cell_type', 'subset', 'functional_marker']].drop_duplicates().values:
    values = conserved_df[(conserved_df.metric == metric) & (conserved_df.cell_type == cell_type) &
                          (conserved_df.subset == subset) & (conserved_df.functional_marker == func)]
    values.dropna(inplace=True)

    # remove infs
    values = values[~values.isin([np.inf, -np.inf]).any(1)]

    if len(values) > 20:
        cor, p_val = spearmanr(values.fov1, values.fov2)
        p_vals.append(p_val)
        cors.append(cor)
        names.append(f'{metric}__{cell_type}__{subset}__{func}')

p_df = pd.DataFrame({'name': names, 'p_val': p_vals, 'cor': cors})
p_df['log_pval'] = -np.log10(p_df.p_val)

sns.scatterplot(data=p_df, x='cor', y='log_pval')
plt.savefig(os.path.join(plot_dir, 'conserved_features_volcano_functional.png'))
plt.close()

sns.histplot(data=p_df, x='cor')
plt.savefig(os.path.join(plot_dir, 'conserved_features_cor_functional.png'))
plt.close()


plot_subset = p_df[(p_df.log_pval > 5) & (p_df.cor > 0.7)]

# get a random row
row = plot_subset.sample(1).iloc[0]
metric, cell_type, subset, func = row['name'].split('__')
correlation = row['cor']
values = conserved_df[(conserved_df.metric == metric) & (conserved_df.cell_type == cell_type) & (conserved_df.subset == subset)
                      & (conserved_df.functional_marker == func)]
values.dropna(inplace=True)
sns.scatterplot(data=values, x='fov1', y='fov2')
plt.title(f'{metric}__{cell_type}__{subset}__{func}__{correlation}')
plt.savefig(os.path.join(plot_dir, 'example_plot_func_6.png'))
plt.close()

# create cascading plot across p values and correlation values
fig, ax = plt.subplots(4, 4, figsize=(20, 20))
for p_idx, p_val in enumerate([1, 5, 10, 20]):
    for c_idx, cor_val in enumerate([0.5, 0.6, 0.7, 0.8]):
        plot_subset = p_df[(p_df.log_pval > p_val - 1) & (p_df.log_pval < p_val + 1)]
        plot_subset = plot_subset[(plot_subset.cor > cor_val - 0.05) & (plot_subset.cor < cor_val + 0.05)]
        if len(plot_subset) > 0:
            # get a random row
            row = plot_subset.sample(1).iloc[0]
            metric, cell_type, subset, func = row['name'].split('__')
            correlation = row['cor']
            values = conserved_df[(conserved_df.metric == metric) & (conserved_df.cell_type == cell_type) & (conserved_df.subset == subset)
                                  & (conserved_df.functional_marker == func)]
            values.dropna(inplace=True)
            sns.scatterplot(data=values, x='fov1', y='fov2', ax=ax[p_idx, c_idx])
            ax[p_idx, c_idx].set_title(f'{p_val}__{cor_val}')

plt.savefig(os.path.join(plot_dir, 'conserved_features_pval_cor_cascade.png'))
plt.close()
## compare correlation across compartments
# plot correlation for all cell types
cell_types = plot_df.cell_type.unique()
regions = plot_df.subset.unique()
regions = [x for x in regions if x not in ['empty_slide', 'tls']]

from scipy.stats import pearsonr, spearmanr
fig, ax = plt.subplots(len(cell_types), len(regions), figsize=(len(regions)*5, len(cell_types)*5))
for i, cell_type in enumerate(cell_types):
    for j, region in enumerate(regions):
        temp_df = plot_df[(plot_df.subset == region) & (plot_df.cell_type == cell_type)]
        temp_df = temp_df.dropna()

        sns.regplot(x='fov1', y='fov2', data=temp_df, ax=ax[i, j])
        ax[i, j].set_title('Correlation of {} in {}'.format(cell_type, region))
        correlation = spearmanr(temp_df.fov1, temp_df.fov2)[0]
        #r2_val = r2_score(temp_df.fov1, temp_df.fov2)

        ax[i, j].text(0.05, 0.95, 'Spearman R: {:.2f}'.format(correlation), transform=ax[i, j].transAxes, fontsize=12, verticalalignment='top')
        #ax[i, j].text(0.05, 0.85, 'R2: {:.2f}'.format(r2_val), transform=ax[i, j].transAxes, fontsize=12, verticalalignment='top')


plt.tight_layout()
plt.savefig('/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/correlation_freq_plots.png', dpi=150)
plt.close()
