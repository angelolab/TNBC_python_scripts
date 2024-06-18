import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
import python_files.supplementary_plot_helpers as supplementary_plot_helpers


BASE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/"
SUPPLEMENTARY_FIG_DIR = os.path.join(BASE_DIR, "supplementary_figs")
seg_dir = os.path.join(BASE_DIR, "segmentation_data/deepcell_output")

# HnE Core, FOV and Segmentation Overlays
hne_fovs = [
    "TONIC_TMA2_R7C4",
    "TONIC_TMA4_R12C4"]

hne_path = Path(SUPPLEMENTARY_FIG_DIR) / "supp_figure_1b_masks"

save_dir = Path(SUPPLEMENTARY_FIG_DIR) / "supp_figure_1b"
save_dir.mkdir(exist_ok=True, parents=True)
for fov in hne_fovs:
    supplementary_plot_helpers.CorePlot(
        fov=fov, hne_path=hne_path, seg_dir=seg_dir
    ).make_plot(save_dir=save_dir)

# ROI selection
metadata = pd.read_csv(os.path.join(BASE_DIR, 'analysis_files/harmonized_metadata.csv'))
metadata = metadata.loc[metadata.MIBI_data_generated, :]
metadata = metadata.loc[metadata.Timepoint.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo']), :]

fov_counts = metadata.groupby('Tissue_ID').size().values
fov_counts = pd.DataFrame(fov_counts, columns=['FOV Count'])
sns.histplot(data=fov_counts, x='FOV Count')
sns.despine()
plt.title("Number of FOVs per Timepoint")
plt.xlabel("Number of FOVs")
plt.tight_layout()

plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, "supp_figure_1c.pdf"), dpi=300)
plt.close()