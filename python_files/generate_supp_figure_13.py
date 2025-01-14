import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import os
import pandas as pd

import python_files.supplementary_plot_helpers as supplementary_plot_helpers


BASE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/"
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "intermediate_files")
SUPPLEMENTARY_FIG_DIR = os.path.join(BASE_DIR, "supplementary_figs")
dist_mat_dir = os.path.join(INTERMEDIATE_DIR, "spatial_analysis", "dist_mats")

# Neighborhood/diversity tuning
neighbors_mat_dir = os.path.join(SUPPLEMENTARY_FIG_DIR, "supp_figure_13_data")
diversity_mixing_pipeline_tuning_dir = os.path.join(
    SUPPLEMENTARY_FIG_DIR, "supp_figure_13_robustness"
)
if not os.path.exists(neighbors_mat_dir):
    os.makedirs(neighbors_mat_dir)
if not os.path.exists(diversity_mixing_pipeline_tuning_dir):
    os.makedirs(diversity_mixing_pipeline_tuning_dir)

# vary the parameters for each marker threshold
cell_table_full = pd.read_csv(os.path.join(BASE_DIR, "analysis_files/cell_table_clusters.csv"))
supplementary_plot_helpers.run_diversity_mixing_tuning_tests(
    cell_table_full, dist_mat_dir=dist_mat_dir,
    neighbors_mat_dir=neighbors_mat_dir,
    save_dir=diversity_mixing_pipeline_tuning_dir,
    threshold_mults=[1/4, 1/2, 3/4, 7/8, 1, 8/7, 4/3, 2, 4],
    mixing_info=supplementary_plot_helpers.MIXING_INFO,
    base_pixel_radius=50, cell_type_col="cell_cluster_broad"
)
