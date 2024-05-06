import os
import pandas as pd
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
from venny4py.venny4py import venny4py

base_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort'
metadata_dir = os.path.join(base_dir, 'intermediate_files/metadata')
sequence_dir = os.path.join(base_dir, 'sequencing_data')

plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/figures/'
timepoint_metadata = pd.read_csv(os.path.join(metadata_dir, 'TONIC_data_per_timepoint.csv'))
harmonized_metadata = pd.read_csv(os.path.join(metadata_dir, 'harmonized_metadata.csv'))

# create venn diagrams
timepoint_metadata = timepoint_metadata.loc[timepoint_metadata.MIBI_data_generated, :]
baseline_ids = timepoint_metadata.loc[timepoint_metadata.Timepoint == 'baseline', 'Patient_ID'].values
induction_ids = timepoint_metadata.loc[timepoint_metadata.Timepoint == 'post_induction', 'Patient_ID'].values
nivo_ids = timepoint_metadata.loc[timepoint_metadata.Timepoint == 'on_nivo', 'Patient_ID'].values
primary_ids = timepoint_metadata.loc[timepoint_metadata.Timepoint == 'primary_untreated', 'Patient_ID'].values

# dict of sets
sets = {
    'primary': set(primary_ids),
    'baseline': set(baseline_ids),
    'induction': set(induction_ids),
    'nivo': set(nivo_ids)}

venny4py(sets=sets)

plt.savefig(os.path.join(plot_dir, 'figure1_venn_diagram.pdf'), dpi=300, bbox_inches='tight')

# create venn diagram across modalities
wes_metadata = pd.read_csv(os.path.join(sequence_dir, 'preprocessing/TONIC_WES_meta_table.tsv'), sep='\t')
rna_metadata = pd.read_csv(os.path.join(sequence_dir, 'preprocessing/TONIC_tissue_rna_id.tsv'), sep='\t')
rna_metadata = rna_metadata.merge(harmonized_metadata[['Patient_ID', 'Tissue_ID']].drop_duplicates(), on='Tissue_ID', how='left')

MIBI_ids = set(timepoint_metadata.Patient_ID.values)
WES_ids = set(wes_metadata['Individual.ID'].values)
RNA_ids = set(rna_metadata.Patient_ID.values)

sets = {
    'MIBI': MIBI_ids,
    'WES': WES_ids,
    'RNA': RNA_ids}

venny4py(sets=sets)
plt.savefig(os.path.join(plot_dir, 'figure1_venn_diagram_modalities.pdf'), dpi=300, bbox_inches='tight')
plt.close()