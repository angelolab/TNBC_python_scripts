import os
import shutil

import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io as io


base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
metadata_dir = os.path.join(base_dir, 'intermediate_files/metadata')
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/figures/'
harmonized_metadata = pd.read_csv(os.path.join(metadata_dir, 'harmonized_metadata.csv'))
image_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/image_data/samples/'

study_fovs = harmonized_metadata.loc[harmonized_metadata.Timepoint.isin(['primary_untreated', 'baseline', 'post_induction', 'on_nivo']), 'fov'].values

# representative images
fov_list = ['TONIC_TMA21_R6C6', 'TONIC_TMA2_R6C6', 'TONIC_TMA18_R4C5', 'TONIC_TMA20_R12C4', 'TONIC_TMA10_R3C1', 'TONIC_TMA10_R6C2']

# copy images to new folder
for fov in fov_list:
    fov_folder = os.path.join(image_dir, fov)
    save_folder = os.path.join(plot_dir, 'Figure2_representative_images/{}'.format(fov))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    fov_files = os.listdir(fov_folder)
    fov_files = [x for x in fov_files if x.endswith('.tiff')]

    for file in fov_files:
        shutil.copy(os.path.join(fov_folder, file), os.path.join(save_folder, file))


# generate crops
crop_dict = {'TONIC_TMA2_R6C6': [[300, 400], [1300, 800], 'CD45_ECAD_Overlay.tif'],
            'TONIC_TMA10_R3C1': [[750, 750], [1300, 1100], 'CD8_CD45RO_HLADR.tif']}

for fov, crop_info in crop_dict.items():
    save_folder = os.path.join(plot_dir, 'Figure2_representative_images/{}'.format(fov))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    crop_coords_1, crop_coords_2, file = crop_info

    # crop image
    for crop_coords in [crop_coords_1, crop_coords_2]:
        crop_img = io.imread(os.path.join(save_folder, file))
        crop_img = crop_img[crop_coords[0]:crop_coords[0] + 500, crop_coords[1]:crop_coords[1] + 500, :]
        io.imsave(os.path.join(save_folder, 'crop_{}.tiff'.format(crop_coords[0])), crop_img)


# cell cluster heatmap

# Markers to include in the heatmap
markers = ["ECAD", "CK17", "CD45", "CD3", "CD4", "CD8", "FOXP3", "CD20", "CD56", "CD14", "CD68",
           "CD163", "CD11c", "HLADR", "ChyTr", "Calprotectin", "FAP", "SMA", "Vim", "Fibronectin",
           "Collagen1", "CD31"]

cell_ordering = ['Cancer', 'Cancer_EMT', 'Cancer_Other', 'CD4T', 'CD8T', 'Treg', 'T_Other', 'B',
                 'NK', 'M1_Mac', 'M2_Mac', 'Mac_Other', 'Monocyte', 'APC','Mast', 'Neutrophil',
                 'Fibroblast', 'Stroma','Endothelium']

# Get average across each cell phenotype

# cell_counts = pd.read_csv(os.path.join(data_dir, "post_processing/cell_table_counts.csv"))
# phenotype_col_name = "cell_cluster"
# cell_counts = cell_counts.loc[cell_counts.fov.isin(study_fovs), :]
# mean_counts = cell_counts.groupby(phenotype_col_name)[markers].mean()
# mean_counts.to_csv(os.path.join(plot_dir, "figure2/cell_cluster_marker_means.csv"))

# read previously generated
mean_counts = pd.read_csv(os.path.join(plot_dir, "figure2_cell_cluster_marker_means.csv"))
mean_counts = mean_counts.set_index('cell_cluster')
mean_counts = mean_counts.reindex(cell_ordering)

# set column order
mean_counts = mean_counts[markers]


# functional marker expression per cell type
core_df_func = pd.read_csv(os.path.join(data_dir, 'functional_df_per_core_filtered_all_combos.csv'))
plot_df = core_df_func.loc[core_df_func.Timepoint.isin(study_fovs), :]
plot_df = plot_df.loc[plot_df.metric == 'cluster_freq', :]
plot_df = plot_df.loc[plot_df.subset == 'all', :]
plot_df = plot_df.loc[~plot_df.functional_marker.isin(['Vim', 'CD45RO_CD45RB_ratio', 'H3K9ac_H3K27me3_ratio', 'HLA1'])]

# average across cell types
plot_df = plot_df[['cell_type', 'functional_marker', 'value']].groupby(['cell_type', 'functional_marker']).mean().reset_index()
plot_df = pd.pivot(plot_df, index='cell_type', columns='functional_marker', values='value')

# set index based on cell_ordering
plot_df = plot_df.reindex(cell_ordering)

cols = ['PDL1','Ki67','GLUT1','CD45RO','CD69', 'PD1','CD57','TBET', 'TCF1',
        'CD45RB', 'TIM3', 'Fe','IDO','CD38']
plot_df = plot_df[cols]

# combine together
combined_df = pd.concat([mean_counts, plot_df], axis=1)

# plot heatmap
sns.clustermap(combined_df, z_score=1, cmap="vlag", center=0, vmin=-3, vmax=3, xticklabels=True, yticklabels=True,
                    row_cluster=False,col_cluster=False)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'Figure2_combined_heatmap.pdf'))
plt.close()