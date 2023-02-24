import pandas as pd
import os
import numpy as np

TMA_dir = '/Users/noahgreenwald/Downloads/TMA_QC'

TMA = 'h120_adj'

tma_map = pd.read_csv(os.path.join(TMA_dir, TMA + '_map.csv'), header=None)
tma_cores = pd.read_csv(os.path.join(TMA_dir, TMA + '_cores.csv'))
tma_blocks = pd.read_csv(os.path.join(TMA_dir, TMA + '_blocks.csv'))
tma_metadata = pd.read_csv(os.path.join(TMA_dir, TMA + '_metadata.csv'))
tma_metadata['random_id'] = np.arange(len(tma_metadata))
tma_metadata_long = pd.melt(tma_metadata, id_vars=['random_id'], value_vars=['Primary block', 'Relapse block'])

# check that all blocks present in the map are found in the cores sheet
tma_map_long = [tma_map.iloc[:, i].astype('str') for i in range(tma_map.shape[1])]
tma_map_long = pd.concat(tma_map_long, axis=0)
tma_map_long_unique = tma_map_long.unique()

map_blocks_not_in_core_sheet = [i for i in tma_map_long_unique if i not in tma_cores.TMA_map_Tissue_ID.unique()]

core_blocks_not_in_block_sheet = [i for i in tma_cores.Tissue_ID.unique() if i not in tma_blocks.Tissue_ID.unique()]
block_sheet_blocks_not_in_core_sheet = [i for i in tma_blocks.Tissue_ID.unique() if i not in tma_cores.Tissue_ID.unique()]


mismatched_number_of_cores = []
for i in tma_map_long_unique:
      map_num = np.sum(tma_map_long == i)
      core_num = np.sum(tma_cores.TMA_map_Tissue_ID == i)
      if map_num == 0 or core_num == 0:
          continue
      if map_num != core_num:
            mismatched_number_of_cores.append(i)

# check that blocks are assigned to the same patient in the metadata sheet as in the block sheet
# mismatched_block_patient = []
# for patient in tma_blocks.Patient_ID.unique():
#       block_ids = tma_blocks[tma_blocks.Patient_ID == patient].Tissue_ID.values
#       metadata_pat_id = tma_metadata_long[tma_metadata_long.value == block_ids[0]].random_id.values
#       if len(metadata_pat_id) == 0:
#             mismatched_block_patient.append(patient)
#             continue
#
#       metadata_block_ids = tma_metadata_long[tma_metadata_long.random_id == metadata_pat_id[0]].value.values
#       if not np.array_equal(block_ids, metadata_block_ids):
#             mismatched_block_patient.append(patient)


# summarize the results
if len(map_blocks_not_in_core_sheet) > 0:
      print("the following block IDs were present in the original TMA map, but are not found in the "
            "cores sheet: {}: ".format(map_blocks_not_in_core_sheet))

if len(core_blocks_not_in_block_sheet) > 0:
    print("the following block IDs were present in the cores sheet, but are not found in the "
      "blocks sheet: {}: ".format(core_blocks_not_in_block_sheet))

if len(block_sheet_blocks_not_in_core_sheet) > 0:
    print("the following block IDs were present in the blocks sheet, but are not found in the "
      "cores sheet: {}: ".format(block_sheet_blocks_not_in_core_sheet))

if len(mismatched_number_of_cores) > 0:
    print("the following block IDs have a different number of cores in the TMA map than "
          "in the cores sheet: {}: ".format(mismatched_number_of_cores))

# if len(mismatched_block_patient) > 0:
#     print("the following patients have different blocks in the metadata sheet "
#           "and the blocks sheet: {}: ".format(mismatched_block_patient))
#
#

