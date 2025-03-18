import os
import anndata
import pandas as pd

from SpaceCat.features import SpaceCat


BASE_DIR = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis_files')
INTERMEDIATE_DIR = os.path.join(BASE_DIR, 'intermediate_files')
FORMATTED_DIR = os.path.join(INTERMEDIATE_DIR, 'formatted_files')

# read in data
cell_table = pd.read_csv(os.path.join(ANALYSIS_DIR, 'combined_cell_table_normalized_cell_labels_updated.csv'))
func_table = pd.read_csv(os.path.join(ANALYSIS_DIR, 'cell_table_func_all.csv'))
func_table = func_table.drop(columns=['H3K9ac_H3K27me3_ratio', 'CD45RO_CD45RB_ratio'])

for col in func_table.columns:
    if col not in ['fov', 'label', 'cell_cluster_broad', 'cell_cluster', 'cell_meta_cluster']:
        func_table[col] = func_table[col].astype(int)
        func_table = func_table.rename(columns={col: f'{col}+'})

cell_table = cell_table.merge(func_table, on=['fov', 'label', 'cell_cluster_broad', 'cell_cluster', 'cell_meta_cluster'])

linear_dists = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'spatial_analysis/cell_neighbor_analysis/cell_distances_filtered.csv'))
cell_table = cell_table.merge(linear_dists, on=['fov', 'label', 'cell_cluster_broad'], how='left')

# add compartment annotations
compartment_annotations = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'mask_dir/cell_annotation_mask.csv'))
compartment_areas = pd.read_csv(os.path.join(INTERMEDIATE_DIR, 'mask_dir/fov_annotation_mask_area.csv'))
compartment_annotations = compartment_annotations.rename(columns={'mask_name': 'compartment'})
compartment_areas = compartment_areas.rename(columns={'area': 'compartment_area'})

cell_table = cell_table.merge(compartment_annotations, on=['fov', 'label'])
cell_table = cell_table.merge(compartment_areas, on=['fov', 'compartment'])


# define column groupings
markers = ['CD11c', 'CD14', 'CD163', 'CD20', 'CD3', 'CD31', 'CD38', 'CD4', 'CD45', 'CD45RB', 'CD45RO', 'CD56', 'CD57',
           'CD68', 'CD69', 'CD8', 'CK17', 'Calprotectin', 'ChyTr', 'Collagen1', 'ECAD', 'FAP', 'FOXP3', 'Fe', 'Fibronectin',
           'GLUT1', 'H3K27me3', 'H3K9ac', 'HLA1', 'HLADR', 'IDO', 'Ki67', 'LAG3', 'PD1', 'PDL1', 'SMA', 'TBET',
           'TCF1', 'TIM3', 'Vim'] + [col for col in cell_table.columns if '+' in col]
centroid_cols = ['centroid-0', 'centroid-1']
cell_data_cols = ['fov', 'label', 'cell_meta_cluster', 'cell_cluster', 'cell_cluster_broad',
                  'compartment', 'compartment_area',
                  'area', 'area_nuclear', 'cell_size', 'cell_size_nuclear', 'centroid-0', 'centroid-1',
                  'centroid-0_nuclear', 'centroid-1_nuclear','centroid_dif', 'centroid_dif_nuclear', 'major_axis_length',
                  'distance_to__B', 'distance_to__Cancer', 'distance_to__Granulocyte', 'distance_to__Mono_Mac',
                  'distance_to__NK', 'distance_to__Other', 'distance_to__Structural', 'distance_to__T']

# create anndata from table subsetted for marker info, which will be stored in adata.X
adata = anndata.AnnData(cell_table.loc[:, markers])

# store all other cell data in adata.obs
adata.obs = cell_table.loc[:, cell_data_cols, ]
adata.obs_names = [str(i) for i in adata.obs_names]

# store cell centroid data in adata.obsm
adata.obsm['spatial'] = cell_table.loc[:, centroid_cols].values

# save the anndata object
adata.write_h5ad(os.path.join(ANALYSIS_DIR, 'adata.h5ad'))

# run SpaceCat
adata = anndata.read_h5ad(os.path.join(ANALYSIS_DIR, 'adata.h5ad'))

# read in image level dataframes
fiber_df = pd.read_csv(os.path.join(FORMATTED_DIR, 'fiber_stats_table.csv'))
fiber_tile_df = pd.read_csv(os.path.join(FORMATTED_DIR, 'fiber_stats_per_tile.csv'))
mixing_df = pd.read_csv(os.path.join(FORMATTED_DIR, 'formatted_mixing_scores.csv'))
kmeans_img_proportions = pd.read_csv(os.path.join(FORMATTED_DIR, 'neighborhood_image_proportions.csv'))
kmeans_compartment_proportions = pd.read_csv(os.path.join(FORMATTED_DIR, 'neighborhood_compartment_proportions.csv'))
pixie_ecm = pd.read_csv(os.path.join(FORMATTED_DIR, 'pixie_ecm_stats.csv'))
ecm_frac = pd.read_csv(os.path.join(FORMATTED_DIR, 'ecm_fraction_stats.csv'))
ecm_clusters = pd.read_csv(os.path.join(FORMATTED_DIR, 'ecm_cluster_stats.csv'))

# specify cell type pairs to compute a ratio for
ratio_pairings = [('CD8T', 'CD4T'), ('CD4T', 'Treg'), ('CD8T', 'Treg'), ('CD68_Mac', 'CD163_Mac')]

# specify addtional per cell and per image stats
per_cell_stats = [
    ['morphology', 'cell_cluster', ['area', 'major_axis_length']],
    ['linear_distance', 'cell_cluster_broad', ['distance_to__B', 'distance_to__Cancer', 'distance_to__Granulocyte', 'distance_to__Mono_Mac',
                                               'distance_to__NK', 'distance_to__Other', 'distance_to__Structural', 'distance_to__T']]

]

per_img_stats = [
    ['fiber', fiber_df],
    ['fiber', fiber_tile_df],
    ['mixing_score', mixing_df],
    ['kmeans_cluster', kmeans_img_proportions],
    ['kmeans_cluster', kmeans_compartment_proportions],
    ['pixie_ecm', pixie_ecm],
    ['ecm_fraction', ecm_frac],
    ['ecm_cluster', ecm_clusters]
]

features = SpaceCat(adata, image_key='fov', seg_label_key='label', cell_area_key='area',
                    cluster_key=['cell_cluster', 'cell_cluster_broad'],
                    compartment_key='compartment', compartment_area_key='compartment_area')

# Generate features and save anndata
adata_processed = features.run_spacecat(functional_feature_level='cell_cluster', diversity_feature_level='cell_cluster', pixel_radius=50,
                                        specified_ratios_cluster_key='cell_cluster', specified_ratios=ratio_pairings,
                                        per_cell_stats=per_cell_stats, per_img_stats=per_img_stats)

adata_processed.write_h5ad(os.path.join(ANALYSIS_DIR, 'adata', 'adata_processed.h5ad'))

# Save finalized tables to csv
adata_processed.uns['combined_feature_data'].to_csv(os.path.join(ANALYSIS_DIR, 'combined_feature_data.csv'), index=False)
adata_processed.uns['combined_feature_data_filtered'].to_csv(os.path.join(ANALYSIS_DIR, 'combined_feature_data_filtered.csv'), index=False)
adata_processed.uns['feature_metadata'].to_csv(os.path.join(ANALYSIS_DIR, 'feature_metadata.csv'), index=False)
adata_processed.uns['excluded_features'].to_csv(os.path.join(ANALYSIS_DIR, 'excluded_features.csv'), index=False)

# filter out immune_agg features
combined_feature_data = pd.read_csv(os.path.join(ANALYSIS_DIR, 'combined_feature_data.csv'))
combined_feature_data = combined_feature_data[~combined_feature_data.feature_name_unique.str.contains('immune_agg')]
combined_feature_data.to_csv(os.path.join(ANALYSIS_DIR, 'combined_feature_data.csv'), index=False)

combined_feature_data_filtered = pd.read_csv(os.path.join(ANALYSIS_DIR, 'combined_feature_data_filtered.csv'))
combined_feature_data_filtered = combined_feature_data_filtered[~combined_feature_data_filtered.feature_name_unique.str.contains('immune_agg')]
combined_feature_data_filtered.to_csv(os.path.join(ANALYSIS_DIR, 'combined_feature_data_filtered.csv'), index=False)

feature_metadata = pd.read_csv(os.path.join(ANALYSIS_DIR, 'feature_metadata.csv'))
feature_metadata = feature_metadata[~feature_metadata.feature_name_unique.str.contains('immune_agg')]
feature_metadata.to_csv(os.path.join(ANALYSIS_DIR, 'feature_metadata.csv'), index=False)

excluded_features = pd.read_csv(os.path.join(ANALYSIS_DIR, 'excluded_features.csv'))
excluded_features = excluded_features[~excluded_features.feature_name_unique.str.contains('immune_agg')]
excluded_features.to_csv(os.path.join(ANALYSIS_DIR, 'excluded_features.csv'), index=False)

# group by timepoint
adata_processed = anndata.read_h5ad(os.path.join(ANALYSIS_DIR, 'adata_processed.h5ad'))
harmonized_metadata = pd.read_csv(os.path.join(ANALYSIS_DIR, 'harmonized_metadata.csv'))

fov_data_df = adata_processed.uns['combined_feature_data']
fov_data_df = pd.merge(fov_data_df, harmonized_metadata[['Tissue_ID', 'fov']], on='fov', how='left')
grouped = fov_data_df.groupby(['Tissue_ID', 'feature_name', 'feature_name_unique', 'compartment',
                               'cell_pop_level', 'feature_type']).agg({'raw_value': ['mean', 'std'],
                                                                       'normalized_value': ['mean', 'std']})
grouped.columns = ['raw_mean', 'raw_std', 'normalized_mean', 'normalized_std']
grouped = grouped.reset_index()
grouped.to_csv(os.path.join(ANALYSIS_DIR, 'timepoint_features.csv'), index=False)

fov_data_df_filtered = adata_processed.uns['combined_feature_data_filtered']
fov_data_df_filtered = pd.merge(fov_data_df_filtered, harmonized_metadata[['Tissue_ID', 'fov']], on='fov', how='left')
grouped = fov_data_df_filtered.groupby(['Tissue_ID', 'feature_name', 'feature_name_unique', 'compartment',
                                 'cell_pop_level', 'feature_type']).agg({'raw_value': ['mean', 'std'],
                                                                            'normalized_value': ['mean', 'std']})

grouped.columns = ['raw_mean', 'raw_std', 'normalized_mean', 'normalized_std']
grouped = grouped.reset_index()
grouped.to_csv(os.path.join(ANALYSIS_DIR, 'timepoint_features_filtered.csv'), index=False)

## run 7_create_evolution_df.py and nivo_outcomes.py
