import os
import pandas as pd
import seaborn as sns

plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/figures/'
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/data/post_processing/'


cell_colors = pd.read_csv(os.path.join(plot_dir, "cell_cluster_colors.csv"))
dat = pd.read_csv(os.path.join(data_dir, "combined_cell_table_normalized_cell_labels_updated.csv"))

phenotype_col_name = "cell_cluster"

# Markers to include in the heatmap
markers = ["CD45", "SMA", "Vim", "FAP", "Fibronectin", "Collagen1",
           "CK17", "ECAD", "ChyTr", "Calprotectin", "CD3",
           "CD4", "CD8", "CD11c", "CD14", "CD20", "CD31",
           "CD56", "CD68", "CD163", "HLADR", "FOXP3"]

# Get average across each cell phenotype
mean_dat = dat.groupby(phenotype_col_name)[markers].mean()
mean_dat['phenotype'] = mean_dat.index


mean_dat = mean_dat.merge(cell_colors)

# Keep channels only
dat_chan = mean_dat.drop(['phenotype','color'], axis=1)
# Set index to be cluster names
dat_chan.index = mean_dat.phenotype.values

# Make heatmap
f = sns.clustermap(data=dat_chan,
                   z_score=1,
                   cmap="vlag",
                   center=0,
                   vmin=-3,
                   vmax=3,
                   xticklabels=True,
                   yticklabels=True,
                   row_colors=mean_dat.color.values)
f.fig.subplots_adjust(wspace =0.01)
f.ax_cbar.set_position((0.1, 0.82, 0.03, 0.15))
f.ax_heatmap.set_xlabel("Marker")
f.ax_heatmap.set_ylabel("Cell cluster")

f.savefig(os.path.join(plot_dir, "figure1/cell_cluster_marker.pdf"))

