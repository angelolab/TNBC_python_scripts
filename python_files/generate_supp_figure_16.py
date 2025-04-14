import os
import pandas as pd
import matplotlib
import matplotlib.lines as mlines

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import seaborn as sns
from venny4py.venny4py import venny4py

BASE_DIR = "/Volumes/Shared/Noah Greenwald/TONIC_Cohort/"
SUPPLEMENTARY_FIG_DIR = os.path.join(BASE_DIR, "supplementary_figs")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis_files")


# plot top baseline features
NT_DIR = '/Volumes/Shared/Noah Greenwald/NTPublic'
combined_df = pd.read_csv(
    os.path.join(NT_DIR, 'SpaceCat/analysis_files/timepoint_combined_features_immunotherapy+chemotherapy.csv'))

for feature in combined_df.feature_name_unique.unique():
    if 'Epithelial' in feature:
        feature_new = feature.replace('Epithelial', 'Cancer')
        combined_df = combined_df.replace({feature: feature_new})

for timepoint in ['Baseline', 'On-treatment']:
    for feature, lims in zip(['Ki67+__Cancer_1', 'Ki67+__CD8T', 'Cancer_1__cell_cluster_density'],
                             [[0, 1], [0, 0.6], [0, 1]]):
        plot_df = combined_df.loc[(combined_df.feature_name_unique == feature) &
                                  (combined_df.Timepoint == timepoint), :]

        fig, ax = plt.subplots(1, 1, figsize=(2, 4))
        sns.stripplot(data=plot_df, x='pCR', y='raw_mean', order=['pCR', 'RD'], color='black', ax=ax)
        sns.boxplot(data=plot_df, x='pCR', y='raw_mean', order=['pCR', 'RD'], color='grey', ax=ax, showfliers=False,
                    width=0.3)
        ax.set_title(timepoint)
        ax.set_ylim(lims)
        sns.despine()
        plt.tight_layout()
        os.makedirs(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_16'), exist_ok=True)
        plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_16', '{}_{}.pdf'.format(feature, timepoint)))
        plt.close()


# WANG DATA COMPARISON
NT_DIR = '/Volumes/Shared/Noah Greenwald/NTPublic'
pred_chemo = pd.read_csv(os.path.join(NT_DIR, 'SpaceCat/prediction_model/chemotherapy/patient_outcomes/all_timepoints_results.csv'))
pred_chemo_immuno = pd.read_csv(os.path.join(NT_DIR, 'SpaceCat/prediction_model/immunotherapy+chemotherapy/patient_outcomes/all_timepoints_results.csv'))
pred_chemo = pred_chemo.rename(columns={'auc_baseline_list': 'baseline_C', 'auc_on_treatment_list': 'on_treatment_C',
                                        'auc_baseline__on_treatment_list': 'both_C'})
pred_chemo_immuno = pred_chemo_immuno.rename(columns={'auc_baseline_list': 'baseline_C&I', 'auc_on_treatment_list': 'on_treatment_C&I',
                                                      'auc_baseline__on_treatment_list': 'both_C&I'})
NT_preds = pd.concat([pred_chemo, pred_chemo_immuno], axis=1)
NT_preds = NT_preds[['baseline_C', 'baseline_C&I', 'on_treatment_C', 'on_treatment_C&I', 'both_C', 'both_C&I']]
NT_preds = NT_preds.rename(columns={'baseline_C': 'Baseline (C)', 'baseline_C&I': 'Baseline (C&I)', 'on_treatment_C': 'On-treatment (C)',
                                    'on_treatment_C&I': 'On-treatment (C&I)', 'both_C': 'Both (C)', 'both_C&I': 'Both (C&I)'})
NT_preds = NT_preds.melt()
NT_preds['cancer_revised'] = 1

pred_chemo = pd.read_csv(os.path.join(NT_DIR, 'SpaceCat_NT_combined/prediction_model/chemotherapy/patient_outcomes/all_timepoints_results.csv'))
pred_chemo_immuno = pd.read_csv(os.path.join(NT_DIR, 'SpaceCat_NT_combined/prediction_model/immunotherapy+chemotherapy/patient_outcomes/all_timepoints_results.csv'))
pred_chemo = pred_chemo.rename(columns={'auc_baseline_list': 'baseline_C', 'auc_on_treatment_list': 'on_treatment_C',
                                        'auc_baseline__on_treatment_list': 'both_C'})
pred_chemo_immuno = pred_chemo_immuno.rename(columns={'auc_baseline_list': 'baseline_C&I', 'auc_on_treatment_list': 'on_treatment_C&I',
                                                      'auc_baseline__on_treatment_list': 'both_C&I'})
combo_preds = pd.concat([pred_chemo, pred_chemo_immuno], axis=1)
combo_preds = combo_preds[['baseline_C', 'baseline_C&I', 'on_treatment_C', 'on_treatment_C&I', 'both_C', 'both_C&I']]
combo_preds = combo_preds.rename(columns={'baseline_C': 'Baseline (C)', 'baseline_C&I': 'Baseline (C&I)', 'on_treatment_C': 'On-treatment (C)',
                                          'on_treatment_C&I': 'On-treatment (C&I)', 'both_C': 'Both (C)', 'both_C&I': 'Both (C&I)'})
combo_preds = combo_preds.melt()
combo_preds['cancer_revised'] = 2

og_preds = pd.read_csv(os.path.join(NT_DIR, 'NT_preds.csv'))
og_preds = og_preds.replace('Base&On', 'Both')
og_preds['variable'] = og_preds['Timepoint'] + ' (' + og_preds['Arm'] + ')'
og_preds = og_preds[['Fold', 'LassoAUC', 'variable']]
og_preds = og_preds.pivot(index='Fold', columns='variable')
og_preds.columns = og_preds.columns.droplevel(0)
og_preds = og_preds.melt()
og_preds['cancer_revised'] = 0
all_preds = pd.concat([NT_preds, og_preds, combo_preds])

fig, ax = plt.subplots()
sns.boxplot(data=all_preds, x='variable', y='value', ax=ax, hue='cancer_revised',
            palette=sns.color_palette(["gold", "#1f77b4", "darkseagreen"]), showfliers=False)
sns.stripplot(data=all_preds, x='variable', y='value', ax=ax, hue='cancer_revised',
              palette=sns.color_palette(["gold", "#1f77b4", "darkseagreen"]), dodge=True)
fig.set_figheight(4)
fig.set_figwidth(8)
plt.xticks(rotation=45)
plt.title('Wang et al. dataset predictions')
plt.ylabel('AUC')
plt.xlabel('')
plt.ylim((0, 1))

# Add the custom legend
yellow_line = mlines.Line2D([], [], color="gold", marker="o", label="Wang et al. predictions", linestyle='None')
blue_line = mlines.Line2D([], [], color="#1f77b4", marker="o", label="SpaceCat predictions", linestyle='None')
green_line = mlines.Line2D([], [], color="darkseagreen", marker="o", label="SpaceCat & Wang et al. features", linestyle='None')
plt.legend(handles=[yellow_line, blue_line, green_line], loc='lower right')
sns.despine()
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_16f.pdf'), bbox_inches='tight', dpi=300)

# TONIC DATA COMPARISON
SpaceCat_dir = os.path.join(BASE_DIR, 'TONIC_SpaceCat')
data_dir = '/Volumes/Shared/Noah Greenwald/TONIC_Cohort/TONIC_SpaceCat/NT_features_only'
preds = pd.read_csv(os.path.join(SpaceCat_dir, 'SpaceCat/prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))
preds = preds[['auc_primary_list', 'auc_baseline_list', 'auc_post_induction_list', 'auc_on_nivo_list']]
preds = preds.rename(columns={'auc_primary_list': 'Primary', 'auc_baseline_list': 'Baseline',
                              'auc_post_induction_list': 'Pre nivo', 'auc_on_nivo_list': 'On nivo'})
preds = preds.melt()
preds['Analysis'] = 0

adj_preds = pd.read_csv(os.path.join(SpaceCat_dir, 'SpaceCat_NT_combined/prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))
adj_preds = adj_preds[['auc_primary_list', 'auc_baseline_list', 'auc_post_induction_list', 'auc_on_nivo_list']]
adj_preds = adj_preds.rename(columns={'auc_primary_list': 'Primary', 'auc_baseline_list': 'Baseline',
                                      'auc_post_induction_list': 'Pre nivo', 'auc_on_nivo_list': 'On nivo'})
adj_preds = adj_preds.melt()
adj_preds['Analysis'] = 2

nt_feats_preds = pd.read_csv(os.path.join(SpaceCat_dir, 'NT_features_only/prediction_model/patient_outcomes/all_timepoints_results_MIBI.csv'))
nt_feats_preds = nt_feats_preds[['auc_primary_list', 'auc_baseline_list', 'auc_post_induction_list', 'auc_on_nivo_list']]
nt_feats_preds = nt_feats_preds.rename(columns={'auc_primary_list': 'Primary', 'auc_baseline_list': 'Baseline',
                                                'auc_post_induction_list': 'Pre nivo', 'auc_on_nivo_list': 'On nivo'})
nt_feats_preds = nt_feats_preds.melt()
nt_feats_preds['Analysis'] = 1
all_preds = pd.concat([preds, adj_preds, nt_feats_preds])

fig, ax = plt.subplots()
sns.boxplot(data=all_preds, x='variable', y='value', ax=ax, hue='Analysis',
            palette=sns.color_palette(["#1f77b4", 'gold', "darkseagreen"]), showfliers=False)
sns.stripplot(data=all_preds, x='variable', y='value', ax=ax, hue='Analysis',
              palette=sns.color_palette(["#1f77b4", 'gold', "darkseagreen"]), dodge=True)

fig.set_figheight(4)
fig.set_figwidth(8)
plt.xticks(rotation=45)
plt.title('TONIC dataset prediction')
plt.ylabel('AUC')
plt.xlabel('')
plt.ylim((0, 1))
plt.legend(loc='lower right').set_title('')
sns.despine()

blue_line = mlines.Line2D([], [], color="#1f77b4", marker="o", label="SpaceCat features", linestyle='None')
yellow_line = mlines.Line2D([], [], color="gold", marker="o", label="Wang et al. features", linestyle='None')
green_line = mlines.Line2D([], [], color="darkseagreen", marker="o", label="SpaceCat & Wang et al. features", linestyle='None')
plt.legend(handles=[blue_line, yellow_line, green_line], loc='lower right')
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_16g.pdf'), bbox_inches='tight', dpi=300)


# NT feature enrichment on TONIC data
harmonized_metadata = pd.read_csv(os.path.join(ANALYSIS_DIR, 'harmonized_metadata.csv'))
ranked_features_all = pd.read_csv(os.path.join(BASE_DIR, 'TONIC_SpaceCat/NT_features_only/analysis_files/feature_ranking.csv'))
ranked_features = ranked_features_all.loc[ranked_features_all.comparison.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo'])]
top_features = ranked_features.loc[ranked_features.comparison.isin(['primary', 'baseline', 'pre_nivo', 'on_nivo']), :]
top_features = top_features.iloc[:100, :]

# summarize distribution of top features
top_features_by_comparison = top_features[['feature_name_unique', 'comparison']].groupby('comparison').count().reset_index()
top_features_by_comparison.columns = ['comparison', 'num_features']
top_features_by_comparison = top_features_by_comparison.sort_values('num_features', ascending=False)

fig, ax = plt.subplots(figsize=(4, 4))
sns.barplot(data=top_features_by_comparison, x='comparison', y='num_features', color='grey', ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_16j.pdf'))
plt.close()

## 2.8 / 4.8  Pre-treatment and On-treatment NT vs TONIC comparisons ##
# Original NT features
file_path = os.path.join(NT_DIR, 'data/41586_2023_6498_MOESM3_ESM.xlsx')
NT_features = pd.read_excel(file_path, sheet_name=None)
cell_table = pd.read_csv(os.path.join(NT_DIR, 'analysis_files/cell_table.csv'))
cell_table = cell_table.replace({'{': '', '}': ''}, regex=True)
cell_table['cell_meta_cluster'] = [cell.replace('^', '')for cell in cell_table['cell_meta_cluster']]
cell_dict = dict(zip(cell_table.cell_meta_cluster, cell_table.cell_cluster))

density_pvals = NT_features['Table 8 Densities p values'][['Time point', 'Cell phenotype', 'p.value', 'Arm']]
density_pvals = density_pvals.rename(columns={'Cell phenotype': 'feature_name_unique'})
density_pvals = density_pvals.replace(cell_dict)
density_pvals['feature_name_unique'] = density_pvals['feature_name_unique'] + '__cell_cluster_density'

ki67_pvals = NT_features['Table 11 Ki67 p values'][['Time point', 'Cell phenotype', 'p.value', 'Unnamed: 10']]
ki67_pvals = ki67_pvals.rename(columns={'Unnamed: 10': 'Arm'})
ki67_pvals = ki67_pvals.rename(columns={'Cell phenotype': 'feature_name_unique'})
ki67_pvals = ki67_pvals.replace(cell_dict)
ki67_pvals['feature_name_unique'] = 'Ki67+__' + ki67_pvals['feature_name_unique']

interaction_pvals = NT_features['Table 10 Interaction p values'][['Time point', 'from', 'to cell phenotype', 'p.value', 'Arm']]
ep_cells = cell_table[cell_table.isEpithelial == 1].cell_meta_cluster.unique()
tme_cells = cell_table[cell_table.isEpithelial == 0].cell_meta_cluster.unique()
interaction_pvals['from'] = interaction_pvals['from'].replace('Epithelial', 'Epi')
interaction_pvals['isEpi'] = ['Epi' if cell in ep_cells else 'TME' for cell in interaction_pvals['to cell phenotype']]
interaction_pvals['interaction'] = ['Hom' if f == Epi else 'Het' for f, Epi in zip(interaction_pvals['from'], interaction_pvals['isEpi'])]
interaction_pvals = interaction_pvals.replace(cell_dict)
interaction_pvals['feature_name_unique'] = interaction_pvals['to cell phenotype'] + '__' + interaction_pvals['from'] + interaction_pvals['interaction']
interaction_pvals = interaction_pvals[['Time point', 'feature_name_unique', 'p.value', 'Arm']]

features = pd.concat([density_pvals, ki67_pvals, interaction_pvals])
features = features[features['p.value']<=0.05]
sig_features = features.replace({'Cancer_1__TMEHet': 'Cancer_Immune_mixing_score',
                                 'Cancer_2__TMEHet': 'Cancer_Immune_mixing_score',
                                 'Cancer_3__TMEHet': 'Cancer_Immune_mixing_score',
                                 'CD4T__TMEHom': 'Structural_T_mixing_score',
                                 'CD8T__TMEHom': 'Structural_T_mixing_score',
                                 'Treg__TMEHom': 'Structural_T_mixing_score'})
pre_treatment_features = sig_features[sig_features['Time point'] == 'Baseline']
on_treatment_features = sig_features[sig_features['Time point'] == 'On-treatment']

tonic_features = pd.read_csv(os.path.join(ANALYSIS_DIR, 'feature_ranking.csv'))
tonic_features = tonic_features[tonic_features['fdr_pval'] <= 0.05]
tonic__sig_features = tonic_features[tonic_features.compartment == 'all']
tonic__sig_features = tonic__sig_features[~tonic__sig_features.feature_name_unique.str.contains('core')]
tonic__sig_features = tonic__sig_features[~tonic__sig_features.feature_name_unique.str.contains('border')]
tonic_pre_treatment_features = tonic__sig_features[tonic__sig_features.comparison.isin(['baseline', 'pre_nivo'])]
tonic_pre_treatment_features = tonic_pre_treatment_features[['feature_name_unique', 'pval', 'comparison']]
tonic_on_treatment_features = tonic__sig_features[tonic__sig_features.comparison == 'on_nivo']
tonic_on_treatment_features = tonic_on_treatment_features[['feature_name_unique', 'pval', 'comparison']]

NT_feats = set(pre_treatment_features.feature_name_unique.unique())
TONIC_feats = set(tonic_pre_treatment_features.feature_name_unique.unique())
sets = {'Wang et al.': NT_feats, 'TONIC': TONIC_feats}
venny4py(sets=sets, colors="yb")
plt.title("Pre-treatment Features")
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_16a.pdf'), bbox_inches='tight', dpi=300)

NT_feats = set(on_treatment_features.feature_name_unique.unique())
TONIC_feats = set(tonic_on_treatment_features.feature_name_unique.unique())
sets = {'Wang et al.': NT_feats, 'TONIC': TONIC_feats}
venny4py(sets=sets, colors="yb")
plt.title("On-treatment Features")
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_16h.pdf'), bbox_inches='tight', dpi=300)

# compare SpaceCat features
NT_features = pd.read_csv(os.path.join(NT_DIR, 'SpaceCat/analysis_files/feature_ranking_immunotherapy+chemotherapy.csv'))
NT_features = NT_features[NT_features['fdr_pval'] <= 0.05]
NT__sig_features = NT_features[NT_features.compartment == 'all']
for feature in NT__sig_features.feature_name_unique.unique():
    if 'Epithelial' in feature:
        feature_new = feature.replace('Epithelial', 'Cancer')
        NT__sig_features = NT__sig_features.replace({feature: feature_new})
NT__sig_features = NT__sig_features[~NT__sig_features.feature_name_unique.str.contains('core')]
NT__sig_features = NT__sig_features[~NT__sig_features.feature_name_unique.str.contains('border')]
NT_pre_treatment_features = NT__sig_features[NT__sig_features.comparison == 'Baseline']
NT_pre_treatment_features = NT_pre_treatment_features[['feature_name_unique', 'pval', 'comparison']]
NT_on_treatment_features = NT__sig_features[NT__sig_features.comparison == 'On-treatment']
NT_on_treatment_features = NT_on_treatment_features[['feature_name_unique', 'pval', 'comparison']]

NT_feats = set(NT_pre_treatment_features.feature_name_unique.unique())
TONIC_feats = set(tonic_pre_treatment_features.feature_name_unique.unique())
sets = {'Wang et al.': NT_feats, 'TONIC': TONIC_feats}
venny4py(sets=sets, colors="yb")
plt.title("Pre-treatment Features")
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_16b.pdf'), bbox_inches='tight', dpi=300)

NT_feats = set(NT_on_treatment_features.feature_name_unique.unique())
TONIC_feats = set(tonic_on_treatment_features.feature_name_unique.unique())
sets = {'Wang et al.': NT_feats, 'TONIC': TONIC_feats}
venny4py(sets=sets, colors="yb")
plt.title("On-treatment Features")
plt.savefig(os.path.join(SUPPLEMENTARY_FIG_DIR, 'supp_figure_16i.pdf'), bbox_inches='tight', dpi=300)
