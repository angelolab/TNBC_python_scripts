import os
import anndata
import pandas as pd

BASE_DIR = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis_files')

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