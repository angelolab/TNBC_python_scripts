import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import spearmanr

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'


# combine overlays together into a single image for easier viewing of what changes are happening over time
cluster_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/cell_cluster_overlay'
plot_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/overlay_dir/baseline_nivo_overlay'
harmonized_metadata = pd.read_csv(os.path.join(data_dir, 'metadata/harmonized_metadata.csv'))

#patients = harmonized_metadata.loc[harmonized_metadata.primary_baseline == True, 'TONIC_ID'].unique()
patients = harmonized_metadata.loc[harmonized_metadata.baseline_on_nivo == True, 'TONIC_ID'].unique()
fov_df = harmonized_metadata.loc[harmonized_metadata.TONIC_ID.isin(patients), ['TONIC_ID', 'fov', 'Timepoint']]
fov_df = fov_df.loc[fov_df.fov.isin(cell_table_clusters.fov.unique())]
for patient in patients:

    # get all primary samples
    timepoint_1 = fov_df.loc[(fov_df.TONIC_ID == patient) & (fov_df.Timepoint == 'baseline'), 'fov'].unique()
    timepoint_2 = fov_df.loc[(fov_df.TONIC_ID == patient) & (fov_df.Timepoint == 'on_nivo'), 'fov'].unique()

    max_len = max(len(timepoint_1), len(timepoint_2))

    fig, axes = plt.subplots(2, max_len, figsize=(max_len*5, 10))
    for i in range(len(timepoint_1)):
        try:
            axes[0, i].imshow(plt.imread(os.path.join(cluster_dir, timepoint_1[i] + '.png')))
            axes[0, i].axis('off')
            axes[0, i].set_title('Baseline')
        except:
            print('No primary image for {}'.format(patient))

    for i in range(len(timepoint_2)):
        try:
            axes[1, i].imshow(plt.imread(os.path.join(cluster_dir, timepoint_2[i] + '.png')))
            axes[1, i].axis('off')
            axes[1, i].set_title('On Nivo')
        except:
            print('No baseline image for {}'.format(patient))

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'TONIC_{}.png'.format(patient)), dpi=300)
    plt.close()

# identify features that are conserved over time
timepoint_features = pd.read_csv(os.path.join(data_dir, 'conserved_features/timepoint_features_conserved.csv'))
keep_ids = harmonized_metadata.Tissue_ID[harmonized_metadata.baseline_on_nivo == True].unique()
timepoint_features = timepoint_features.loc[timepoint_features.Tissue_ID.isin(keep_ids)]
timepoint_features = timepoint_features.merge(harmonized_metadata[['TONIC_ID', 'Tissue_ID', 'Timepoint']].drop_duplicates(), on='Tissue_ID')
timepoint_features = timepoint_features.loc[timepoint_features.Timepoint.isin(['baseline', 'on_nivo'])]

# generate paired df by patient
paired_df = timepoint_features.pivot(index=['feature_name', 'TONIC_ID'], columns='Timepoint', values='mean')
paired_df = paired_df.reset_index()

p_vals = []
cors = []
names = []
for feature_name in paired_df.feature_name.unique():
    values = paired_df[(paired_df.feature_name == feature_name)].copy()
    values.dropna(inplace=True)
    #values = values[~values.isin([np.inf, -np.inf]).any(1)]

    if len(values) > 20:
        cor, p_val = spearmanr(values.on_nivo, values.baseline)
        p_vals.append(p_val)
        cors.append(cor)
        names.append(feature_name)

ranked_features = pd.DataFrame({'feature_name': names, 'p_val': p_vals, 'cor': cors})
ranked_features['log_pval'] = -np.log10(ranked_features.p_val)

# combine with feature metadata
ranked_features = ranked_features.merge(paired_df[['feature_name', 'compartment', 'cell_pop', 'feature_type']].drop_duplicates(), on='feature_name', how='left')

sns.scatterplot(data=ranked_features, x='cor', y='log_pval')
plt.savefig(os.path.join(plot_dir, 'conserved_features_volcano.png'))
plt.close()

ranked_features['conserved'] = (ranked_features.log_pval >= 6) & (ranked_features.cor >= 0.5)

ranked_features['highly_conserved'] = (ranked_features.log_pval >= 10) & (ranked_features.cor >= 0.7)

row = 49
name = ranked_features.loc[row, 'feature_name']
correlation = ranked_features.loc[row, 'cor']
p_val = ranked_features.loc[row, 'p_val']
values = paired_df[(paired_df.feature_name == name)]
values.dropna(inplace=True)
fig, ax = plt.subplots()
sns.scatterplot(data=values, x='on_nivo', y='baseline', ax=ax)
ax.text(0.05, 0.95, f'cor: {correlation:.2f}', transform=ax.transAxes, fontsize=10,
           verticalalignment='top')
ax.text(0.65, 0.95, f'p: {p_val:.2e}', transform=ax.transAxes, fontsize=10,
           verticalalignment='top')
plt.title(f'{name}')
plt.savefig(os.path.join(plot_dir, 'evolution_baseline_nivo_features_{}.png'.format(name)))
plt.close()

# generate distance metric between each pair of patients to cluster based on similarity

wide_timepoint_features = timepoint_features.pivot(index=['Tissue_ID', 'TONIC_ID', 'Timepoint'], columns='feature_name', values='mean')

# z score features in each column
wide_timepoint_features_norm = wide_timepoint_features.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
wide_timepoint_features_norm = wide_timepoint_features_norm.reset_index()

from scipy.spatial.distance import pdist, squareform

distances = []
ids = []

# calculate distances
for tonic_id, data in wide_timepoint_features_norm.groupby('TONIC_ID'):
    data = data.drop(['TONIC_ID', 'Tissue_ID'], axis=1)
    if len(data) == 1:
        print('Only one timepoint for {}'.format(tonic_id))
        continue

    # drop columns with any Nans
    data = data.dropna(axis=1, how='any')

    data = data.drop(['Timepoint'], axis=1)

    # calculate distance between each pair of patients
    dist = pdist(data.values, metric='euclidean')[0]

    # add to list
    distances.append(dist)
    ids.append(tonic_id)

# convert to df
distances = pd.DataFrame({'distances': distances, 'TONIC_ID': ids})

g = sns.catplot(data=distances, y='distances', kind='swarm')
g.set(ylim=(0, 30))
g.savefig(os.path.join(plot_dir, 'paired_distances_primary_met.png'))
plt.close()

# compute randomized distances
randomized_distances = []
randomized_ids = []

indices = np.arange(len(wide_timepoint_features_norm) - 1)
np.random.shuffle(indices)

for i in np.arange(0, len(indices), 2):
    # get two random patients
    patient1 = wide_timepoint_features_norm.iloc[indices[i]]
    patient2 = wide_timepoint_features_norm.iloc[indices[i+1]]

    # calculate distance between each pair of patients
    combined = pd.concat([patient1, patient2], axis=1).T
    combined = combined.drop(['TONIC_ID', 'Tissue_ID', 'Timepoint'], axis=1)

    # drop columns with any Nans
    combined = combined.dropna(axis=1, how='any')

    dist = pdist(combined.values.astype('float'), metric='euclidean')[0]

    # add to list
    randomized_distances.append(dist)
    randomized_ids.append(i)

# convert to df
randomized_distances = pd.DataFrame({'distances': randomized_distances, 'TONIC_ID': randomized_ids})

g = sns.catplot(data=randomized_distances, y='distances', kind='swarm', )
g.set(ylim=(0, 30))
g.savefig(os.path.join(plot_dir, 'randomized_distances_primary_met.png'))
plt.close()

