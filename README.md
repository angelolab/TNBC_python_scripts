# TNBC_python_scripts

## Files 
This repo contains working scripts for analyzing the TNBC MIBI data. It is organized as follows:

`data_dir`: This directory contains a cell table (generated with ark and annotated by Pixie). The scripts expect a column named 
`cell_meta_cluster` containing the cell clusters, as well `fov` with the specifc image name. In addition, this directory should also contain metadata
about each fov, each timepoint, and each patient, as appropriate for your study.

Files:
`preprocessing_cell_table_updates.py`: This file takes the cell table generated by Pixie, and transforms it for plotting. Some of this functionality is 
has now been incorporated into notebook 4 in `ark`. Other parts, however, have not yet been put into `ark`, such as aggregating cell populations. 

`preprocessing_metadata.py`: This file transforms the cell table and metadata files for analysis. It creates a simplified cell table
with only the necessary columns for specific plotting tasks, and creates annotations in the metadata files that need to be computed from the
data, such as which patients have data from multiple timepoints

`create_dfs_per_core.py`: This file creates the dfs which will be used for plotting core-level information. It transforms the cell table into
a series of long-format dfs which can be easily used for data visualization. It creates separate dfs for cell population evaluations and functional marker
evaluation

`create_image_masks.py`: This file creates masks for each image based on supplied critieria. It identifies background based on the gold channel, tumor compartments based on ECAD staining patterns, and TLS structures. It then takes these masks, and assigns each cell each image to the mask that it overlaps most with


## Data Structures
In order to facilitate different analyses, there are a small number of distinct formats for storing data. 

*Cell table*: This is the lowest level representation of the data, from which almost all other data formats are derived. Each row represents a single cell from a single image. Columns represent the different features for each cell. For example, the unique ID for each cell is located in the `label` column. The image that the cell came from is noted in the `fov` column, and the intensity of staining for CD68 protein is indicated by the `CD68` column. 
In addition, there are often multiple levels of granularity in the clustering scheme, which are represented here as different columns. For example, `cell_cluster_detail` has more fine-grained assignments, with more distinct cell types, than `cell_cluster_broad`, which has a simpler schema. 

| label | fov | Ecadherin | CD68 | CD3 | Cell_cluster_detail |  Cell_cluster_broad |
| :---:  | :---:  |  :---:  |  :---:  |  :---:  | :---:  |  :---:  | 
| 1 | TMA1_FOV1| 0.4  | 0.01 | 0.01 |  Cancer | Cancer |
| 2 | TMA1_FOV1| 0.01  | 0.0 | 0.8 |  T cell |  Immune | 
| 19 | TMA2_FOV4| 0.01  | 0.8 | 0.01 |  Macrophage |  Immune | 

*segmentation mask*: This is the lowest level spatial representation of the data, from which most other spatial data formats are derived. Each image has a single segmentation mask, which has the locations of each cell. Cells are represented on a per-pixel basis, based on their `label` in the `cell_table`. For example, all of the pixels belonging to cell 1 would have a value of 1, all of the pixels belonging to cell 2 would have a value of 2, etc etc. Shown below is a simplified example, with cell 1 on the left and cell 2 on the right. 
```
0 0 0 0 0 0 0 0 0 0 
0 1 1 0 0 0 0 2 2 0 
1 1 1 1 0 0 2 2 2 2 
1 1 1 1 0 0 2 2 2 0 
1 1 0 0 0 0 0 2 2 0 
1 0 0 0 0 0 0 2 0 0 
0 0 0 0 0 0 0 0 0 0 
```

*distance_matrix.xr*: this data structure represents the distances between all cells in an image. The rows and columns are labeled according to the cell ID of each cell in an image, with the value at `ij`th cell representing the euclidian distance, in pixels, between cell `i` and cell `j.

|  | 1 | 3 | 6 | 8 | 
| :---:  |  :---:  |  :---:  | :---:  | :---: | 
| 1| 0 | 200 | 30 | 21 | 
| 3| 200  | 0 | 22 | 25 | 
| 6| 30  | 22 | 0 | 300 | 
| 8| 21  | 25 | 300 | 0 | 



*neighborhood_matrix*: This data structures summarizes information about the composition of a cell's neighbors. Each row represents an individual cell, with the columns representing the neighboring cells. For example, the first row would represent the number of cells of each cell type present within some pre-determined distance around the first cell in the image. 


| fov | label | T cell | B cell | Macrophage | Treg | 
| :---:  |  :---:  |  :---:  |  :---:  | :---:  | :---: | 
| TMA1_FOV1| 1 | 10 | 400| 30 | 1 | 
| TMA1_FOV1| 2 | 5 | 0 | 30 |  5 | 
| TMA2_FOV4| 5 | 0 | 0 | 30 |  6 | 

## Output Files

### Image Level Features
*fov_features.csv*: This file is a combination of all feature metrics calculated on a per image basis. The file *fov_features_filtered.csv* is also produced, which is the entire features file with any highly correlated features removed.

| Tissue_ID | fov | raw_value | normalized_value |    feature_name_unique     | cell_pop |
|:---------:|:---:|:---------:|:----------------:|:--------------------------:|:--------:|
|    T1     |  1  |    0.1    |       2.6        |        area_Cancer         | Cancer |
|    T2     |  2  |   -0.01   |       -0.6       |  cluster_broad_diversity   | Immune |
|    T3     |  5  |   -1.8    |       -0.7       |     max_fiber_density      | all |

The individual feature data can be found in the corresponding files detailed below.
Each of the data frames in this section can be further stratified based on the feature relevancy and redundancy. The files below can have any of the following suffixes:
* *_filtered*: features removed if there are less than 5 cells of the specified type
* *_deduped*: redundant features removed
* *_filtered_deduped*: both of the above filtering used


1. *cluster_df*: This data structure summarizes key informaton about cell clusters on a per-image basis, rather than a per-cell basis. Each row represents a specific summary observation for a specific image of a specific cell type. For example, the number of B cells in a given image. The key columns are `fov`, which specifies the image the observation is from; `cell_type`, which specifies the cell type the observation is from; `metric`, which describes the specific summary statistic that was calculated; and `value`, which is the actual value of the summary statistic. For example, one statistic might be `cell_count_broad`, which would represent the number of cells per image, enumerated according the cell types in the `broad` clustering scheme. Another might be `cell_freq_detail`, which would be the frequency of the specified cell type out of all cells in the image, enumerated based on the detailed clustering scheme.

| fov | cell_type | value | metric | disease_stage | 
| :---:  |  :---:  |  :---:  |  :---:  | :---:  | 
| TMA1_FOV1| Immune | 100 | cell_count_broad|  Stage I | 
| TMA1_FOV1| Treg | 0.1 | cell_freq_detail |  Stage II  | 
| TMA2_FOV4| Macrophage  | 20 | cell_count_detail |  Stage II |  

In addition to these core columns, metadata can be added to faciliate easy analysis, such as disease stage, prognosis, anaotomical location, or other information that is useful for plotting purposes. 


2. *functional_df*: This data structure summarizes information about the functional marker status of cells on a per-image basis. Each row represents the functional marker status of a single functional marker, in a single cell type, in a single image. The columns are the same as above, but with an additional `functional_marker` column which indicates which functional marker is being summarized. For example, one row might show the number of Tregs in a given image which are positive for Ki67, while another shows the proportion of cancer cells in an image that are PDL1+. 


| fov | cell_type | value | metric | functional marker | disease_stage | 
| :---:  |  :---:  |  :---:  |  :---:  | :---:  | :---: | 
| TMA1_FOV1| Immune | 100 | cell_count_broad| Ki67 | Stage I | 
| TMA1_FOV1| Treg | 0.4 | cell_freq_detail | PDL1 |  Stage II  | 
| TMA2_FOV4| Macrophage  | 20 | cell_count_detail | TIM3 |  Stage II | 

3. *morph_df*: This data structure summarizes information about the morphology of cells on a per-image basis.  Each row represents the morphological statistic, in a single cell type, in a single image.

| fov | cell_type |  morphology  | value | metric | disease_stage | 
| :---:  |  :---:  |:------------:|:-----:|  :---:  | :---: | 
| TMA1_FOV1| Immune |     area     |  100  | cell_count_broad| Stage I | 
| TMA1_FOV1| Treg | area_nuclear |  0.4  | cell_freq_detail |  Stage II  | 
| TMA2_FOV4| Macrophage  |   nc_ratio   |  20   | cell_count_detail |  Stage II | 

4. *distance_df*: This data structure summarizes information about the closest linear distance between cell types on a per-image basis.

| fov | cell_type | linear_distance | value | metric | disease_stage | 
| :---:  |  :---:  |:---------------:|:-----:|  :---:  | :---: | 
| TMA1_FOV1| Immune |      Immune       |  100  | cluster_broad_freq| Stage I | 
| TMA1_FOV1| Immune |  Treg   |  0.4  | cluster_broad_freq |  Stage II  | 
| TMA2_FOV4| MacImmunerophage  |    Macrophage     |  20   | cluster_broad_freq |  Stage II | 


5. *diversity_df*: This data structure summarizes information about the diversity of cell types on a per-image basis.

| fov | cell_type |      diversity_feature       | value | metric | disease_stage | 
| :---:  |  :---:  |:----------------------------:|:-----:|  :---:  | :---: | 
| TMA1_FOV1| Immune | diversity_cell_cluster_broad |  1.1  | cluster_broad_freq| Stage I | 
| TMA1_FOV1| Immune |    diversity_cell_cluster    |  0.4  | cluster_broad_freq |  Stage II  | 
| TMA2_FOV4| MacImmunerophage  | diversity_cell_cluster_broad |   2   | cluster_broad_freq |  Stage II | 

6. *fiber_df / fiber_df_per_tile*:


### Timepoint Level Features
*timepoint.csv*: While the data tables above were aggregated *per_core*, this data is a combination of all feature metrics calculated on a per sample timepoint basis.  The file *timepoint_features_filtered.csv* is also produced, which is the entire features file with any highly correlated features removed.

| Tissue_ID | feature_name_unique | cell_pop | raw_mean | raw_std | normalized_mean | normalized_std |
|:---------:|:---:|:--------:|:-------:|:-------:|:---------------:|:--------------:|
|    T1     |  area_Cancer  |  Cancer  |   0.1    |   1.3   |       2.6       |      0.3       |
|    T2     |  cluster_broad_diversity  |  Immune  |  -0.01   |   0.3   |      -0.6       |      1.1       |
|    T3     |  max_fiber_density  |   all    |   -1.8   |   -16   |      -0.7       |      0.2       |


### Nivo Outcomes
*combined_df.csv*: 

1. *evolution_df.csv*:

|      feature_name       | Patient_ID |       comparison        | normalized_value | raw_value |
|:-----------------------:|:----------:|:-----------------------:|:----------------:|:---------:|
|       area_Cancer       |     1      |    primary__baseline    |       2.6        |    0.1    |
| cluster_broad_diversity |     2      | post_induction__on_nivo |      -0.06       |   -0.01   |
|    max_fiber_density    |     5      |    baseline__on_nivo    |       -0.7       |   -1.8    |

### Metadata
*harmonized_metadata.csv*:

*feature_metadata.csv*: This file gives more detailed information about the specifications that make up each of the features in the fov and timepoint feature tables. The columns include, geenral feature name, unique feature name, compartment, cell population, cell population level, and feature type details.

