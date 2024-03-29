# TNBC_python_scripts

This repo contains working scripts for analyzing the TNBC MIBI data. Below is a description of how to navigate the TNBC datasets, with specific information regarding the data file formats, as well as the scripts used to generate the data.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Data Structures](#data-structures)
- [Analysis Files](#analysis-files)
- [Output Files](#output-files)
- [Scripts](#scripts)


## Directory Structure
### Top Level Folders
`image_data`: Contains the channel images for each FOV. 

`segmentation_data`: Contains the whole cell and nuclear segmentation masks for each FOV.

`analysis_files`: This directory should initially contain a cell table (generated with ark and annotated by Pixie). The scripts expect a column named 
"cell_meta_cluster" containing the cell clusters, as well "fov" with the specific image name. 
This folder will also contain the final data tables generated by the TNBC scripts.

`output_files`: This directory will be created in *5_create_dfs_per_core.py* and store the per core and per timepoint data files for each feature. These will be aggregated to form the final data tables stored in *analysis_files*.

`intermediate_files`: This directory should contain subfolders storing any fov and cell level feature analysis done on the data. In addition, there should be a subdirectory containing the metadata
about each fov, each timepoint, and each patient, as appropriate for your study.

### Directory Tree
* TONIC_Cohort (base directory)
  * image_data 
  * segmentation_data
    * deepcell_output
  * analysis_files
  * output_files
  * intermediate_files
    * metadata
    * post_processing - contains specifications for the filtering of the data tables in *output_files*  
    * mask_dir - contains the compartment masks generated in *3_create_image_masks.py*
    * fiber_segmentation_processed_data - image level fiber analysis
      * tile_stats_512 - 512x512 tile analysis
    * spatial_analysis
      * dist_mats
      * neighborhood_mats - neighboring cell count/frequency at specified pixel radius and cell cluster level
      * mixing_score - image level mixing score of various cell population combinations
      * cell_neighbor_analysis - data detailing cell diversity and linear distance between cell populations in an image
      * neighborhood_analysis - kmeans neighborhood analysis 
    * ecm
    * ecm_pixel_clustering



## Data Structures
In order to facilitate different analyses, there are a small number of distinct formats for storing data. 

*cell table*: This is the lowest level representation of the data, from which almost all other data formats are derived. Each row represents a single cell from a single image. Columns represent the different features for each cell. For example, the unique ID for each cell is located in the `label` column. The image that the cell came from is noted in the `fov` column, and the intensity of staining for CD68 protein is indicated by the `CD68` column. 
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

## Analysis Files

*harmonized_metadata.csv*: This data frame details the various FOVs and their associated tissue and patient IDs, localization, timepoint, etc.

*feature_metadata.csv*: This file gives more detailed information about the specifications that make up each of the features in the fov and timepoint feature tables. The columns include, general feature name, unique feature name, compartment, cell population, cell population level, and feature type details.

*timepoint_combined_features.csv*: This dataframe details feature data for patients at various timepoints and includes the relevant metadata.

|    feature_name_unique    | raw_mean | normalized_mean | Patient_ID | disease_stage |
|:-------------------------:|:--------:|:---------------:|:----------:|:-------------:|
|        area_Cancer        |   0.1    |       2.6       |     1      |    Stage I    |
|  cluster_broad_diversity  |  -0.01   |      -0.6       |     2      |   Stage II    |
|     max_fiber_density     |   -1.8   |      -0.7       |     3      |   Stage III   |


*combined_cell_table_normalized_cell_labels_updated*: The original cell table with all cell level data included. See the cell table description in [Data Structures](#Data-Structures) for more information. 

*cell_table_clusters*: Subset of the cell table containing just the FOV name, cell label, and different cluster labels.

*cell_table_counts*: Consolidated cell table with only marker count data.

*cell_table_morph*: Subset of the cell table containing only the morphological data for each cell (area, perimeter, major_axis_length, etc.). 

*cell_table_func_single_positive*: A cell table containing only the functional marker positivity data.

*cell_table_func_all*: A cell table containing all possible pairwise marker positivity data.

*fov_features.csv*: This file is a combination of all feature metrics calculated on a per image basis. The file *fov_features_filtered.csv* is also produced, which is the entire feature file with any highly correlated features removed.

The fov_features table aggregates features of many different types together, all of which are detailed in [Ouput Files](#Output-Files).

| Tissue_ID | fov | raw_value | normalized_value |   feature_name    |      feature_name_unique       |  compartment  | cell_pop |   feature_type   |
|:---------:|:---:|:---------:|:----------------:|:-----------------:|:------------------------------:|:-------------:|:--------:|:----------------:|
|    T1     |  1  |    0.1    |       2.6        | B__Cancer__ratio  |  B__Cancer__ratio_cancer_core  |  cancer_core  | multiple |  density_ratio   |
|    T2     |  2  |   -0.01   |       -0.6       | cancer_diversity  | cancer_diversity_cancer_border | cancer_border |  Cancer  | region_diversity |
|    T3     |  5  |   -1.8    |       -0.7       | max_fiber_density |       max_fiber_density        |  stroma_core  |   all    |      fiber       |

In the example table above, we see there are multiple columns that contain descriptive information about the statistics contained in each row. While `feature_name_unique` obviously gives the most granular description of the value, we can also use the other columns to quickly subset the data for specific analysis. 
For example, to look at all features within one region type across every image, we simply filter the compartment for only "cancer_core". 
Alternatively, we could compare the granular cell type diversity of all immune classified cells across regions by filtering both the feature_type as "cell_diversity" and cell_pop as "immune".


*timepoint.csv*: While the data table above is aggregated *per_core*, this data is a combination of all feature metrics calculated on a per sample timepoint basis.  The file *timepoint_features_filtered.csv* is also produced, which is the entire feature file with any highly correlated features removed.

| Tissue_ID |   feature_name    |      feature_name_unique       |  compartment  | cell_pop | raw_mean | raw_std | normalized_mean | normalized_std |
|:---------:|:-----------------:|:------------------------------:|:-------------:|:--------:|:-------:|:-------:|:---------------:|:--------------:|
|    T1     | B__Cancer__ratio  |  B__Cancer__ratio_cancer_core  |  cancer_core  | multiple |   0.1    |   1.3   |       2.6       |      0.3       |
|    T2     | cancer_diversity  | cancer_diversity_cancer_border | cancer_border |  Cancer  |  -0.01   |   0.3   |      -0.6       |      1.1       |
|    T3     | max_fiber_density |       max_fiber_density        |  stroma_core  |   all    |   -1.8   |   -16   |      -0.7       |      0.2       |



## Output Files

The individual feature data that combines into *fov_features.csv* and *timepoint_features.csv* can be found in the corresponding files detailed below.
Each of the data frames in this section can be further stratified based on the feature relevancy and redundancy. The files below can have any of the following suffixes:
* *_filtered*: features removed if there are less than 5 cells of the specified type
* *_deduped*: redundant features removed
* *_filtered_deduped*: both of the above filtering applied


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

6. *fiber_df / fiber_df_per_tile*: This data structure summarizes statistics about the collagen fibers at an image-level and also within 512x512 sized pixel crops of the image.

| Tissue_ID |     fiber_metric      | mean | std | disease_stage | 
|:---------:|:---------------------:|:----:|:---:| :---: | 
| TMA1_FOV1 | fiber_alignment_score | 2.2  | 0.5 | Stage I | 
| TMA1_FOV1 |       fiber_are       | 270  | 30  |  Stage II  | 
| TMA2_FOV4 |   fiber_major_axis_length    |  35  |1.9  |  Stage II | 


## Scripts 

`1_postprocessing_cell_table_updates.py`: This file takes the cell table generated by Pixie, and transforms it for plotting. Some of this functionality is 
has now been incorporated into notebook 4 in `ark`. Other parts, however, have not yet been put into `ark`, such as aggregating cell populations. It also creates simplified cell tables
with only the necessary columns for specific plotting tasks.

`2_postprocessing_metadata.py`: This file transforms the metadata files for analysis. It creates annotations in the metadata files that need to be computed from the
data, such as which patients have data from multiple timepoints.

`3_create_image_masks.py`: This file creates masks for each image based on supplied criteria. It identifies background based on the gold channel and tumor compartments based on ECAD staining patterns. It then takes these masks, and assigns each cell each image to the mask that it overlaps most with.

`5_create_dfs_per_core.py`: This file creates the dfs which will be used for plotting core-level information. It transforms the cell table into
a series of long-format dfs which can be easily used for data visualization. It creates separate dfs for cell population evaluations, functional marker
evaluation, etc.

`6_create_fov_stats.py`: This file aggregates the various fov features and timepoint features into separate files, and  additionally filters out any unnecessary features based on their correlation within compartments.

`7_create_evolution_df.py`: This file compares features across various timepoints and treatments. 

